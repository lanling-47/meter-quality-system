"""MGQ-Former: multi-granularity quality fusion model.

The model is intentionally lightweight for local deployment:
- meter-level tabular features are encoded as one token;
- batch-level sampled-aging profile features are encoded as another token;
- a small Transformer encoder fuses the two granularities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BATCH_FEATURE_NAMES = [
    "sample_count",
    "high_risk_rate",
    "avg_final_error_max",
    "max_final_error_max",
    "avg_error_degradation_rate",
    "max_clock_drift_rate",
    "max_rs485_total_errors",
    "confidence",
]


@dataclass
class MGQFormerConfig:
    single_dim: int
    batch_dim: int = len(BATCH_FEATURE_NAMES)
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.15
    epochs: int = 160
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


class MGQFormerNet(nn.Module):
    """Two-token Transformer for meter-level and batch-level quality evidence."""

    def __init__(self, config: MGQFormerConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.token_type = nn.Parameter(torch.zeros(1, 3, config.d_model))
        self.single_encoder = nn.Sequential(
            nn.LayerNorm(config.single_dim),
            nn.Linear(config.single_dim, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.batch_encoder = nn.Sequential(
            nn.LayerNorm(config.batch_dim),
            nn.Linear(config.batch_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x_single: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        n = x_single.shape[0]
        cls = self.cls_token.expand(n, -1, -1)
        single_token = self.single_encoder(x_single).unsqueeze(1)
        batch_token = self.batch_encoder(x_batch).unsqueeze(1)
        tokens = torch.cat([cls, single_token, batch_token], dim=1) + self.token_type
        fused = self.fusion(tokens)
        return self.classifier(fused[:, 0, :]).squeeze(-1)


class MGQFormerPackage:
    """Pickle-safe prediction wrapper used by predict_service.load_active_model."""

    def __init__(
        self,
        model: MGQFormerNet,
        single_scaler: StandardScaler,
        batch_scaler: StandardScaler,
        feature_names: list[str],
        batch_feature_names: list[str],
        config: dict[str, Any],
    ):
        self.model = model.cpu().eval()
        self.single_scaler = single_scaler
        self.batch_scaler = batch_scaler
        self.feature_names = feature_names
        self.batch_feature_names = batch_feature_names
        self.config = config
        self.model_type = "mgqformer"

    def predict_proba(self, X, batch_features=None):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if batch_features is None:
            batch_arr = np.zeros((X_arr.shape[0], len(self.batch_feature_names)), dtype=np.float32)
        else:
            batch_arr = np.asarray(batch_features, dtype=np.float32)
            if batch_arr.ndim == 1:
                batch_arr = batch_arr.reshape(1, -1)
            if batch_arr.shape[0] == 1 and X_arr.shape[0] > 1:
                batch_arr = np.repeat(batch_arr, X_arr.shape[0], axis=0)

        X_scaled = self.single_scaler.transform(X_arr).astype(np.float32)
        batch_scaled = self.batch_scaler.transform(batch_arr).astype(np.float32)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(X_scaled),
                torch.from_numpy(batch_scaled),
            )
            probs = torch.sigmoid(logits).numpy()
        return np.vstack([1.0 - probs, probs]).T


def _profile_to_batch_features(profile) -> list[float]:
    if profile is None:
        return [0.0 for _ in BATCH_FEATURE_NAMES]
    return [
        float(profile.sample_count or 0),
        float(profile.high_risk_rate or 0.0),
        float(profile.avg_final_error_max or 0.0),
        float(profile.max_final_error_max or 0.0),
        float(profile.avg_error_degradation_rate or 0.0),
        float(profile.max_clock_drift_rate or 0.0),
        float(profile.max_rs485_total_errors or 0),
        float(profile.confidence or 0.0),
    ]


def batch_features_from_key(batch_key: str | None, db) -> list[float]:
    from app.models.db_models import BatchQualityProfile

    profile = None
    if batch_key:
        profile = db.query(BatchQualityProfile).filter_by(batch_key=batch_key).first()
        if profile is None:
            profile = db.query(BatchQualityProfile).filter_by(order_no=batch_key).first()
    return _profile_to_batch_features(profile)


def build_deep_training_matrix(db):
    """Return meter features, batch profile features, labels and metadata."""
    from app.models.db_models import BatchQualityProfile, QualityFeatureSnapshot
    from app.services.data_linkage_service import (
        build_quality_feature_snapshots,
        get_snapshot_feature_matrix,
    )

    build_quality_feature_snapshots(db, include_synthetic_demo=False)
    snapshots = db.query(QualityFeatureSnapshot).order_by(QualityFeatureSnapshot.meter_no).all()
    X_single_matrix, y_arr, feature_meta = get_snapshot_feature_matrix(db)
    profiles = db.query(BatchQualityProfile).all()
    profiles_by_key = {p.batch_key: p for p in profiles if p.batch_key}
    fallback_profile = profiles[0] if len(profiles) == 1 else None

    X_single, X_batch, y, meta = [], [], [], []
    for idx, snapshot in enumerate(snapshots):
        X_single.append(X_single_matrix[idx].tolist())
        profile = profiles_by_key.get(snapshot.batch_key) if snapshot.batch_key else fallback_profile
        X_batch.append(_profile_to_batch_features(profile))
        y.append(int(y_arr[idx] if idx < len(y_arr) else snapshot.quality_label or 0))
        snapshot_meta = feature_meta[idx] if idx < len(feature_meta) else {}
        meta.append({
            "meter_no": snapshot.meter_no,
            "batch_key": snapshot.batch_key,
            "process_profile_key": snapshot_meta.get("process_profile_key"),
            "source": snapshot.source,
            "confidence": snapshot.confidence,
        })

    return (
        np.asarray(X_single, dtype=np.float32),
        np.asarray(X_batch, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        meta,
    )


def _emit(emit_fn, event: str, data: dict):
    if emit_fn:
        emit_fn(event, data)


def train_mgqformer(db, feature_names: list[str], emit_fn=None) -> dict:
    """Train MGQ-Former and return a package compatible with predict_service."""
    torch.manual_seed(42)
    np.random.seed(42)

    X_single, X_batch, y, meta = build_deep_training_matrix(db)
    n_samples = len(y)
    n_positive = int(y.sum())
    _emit(emit_fn, "progress", {
        "step": 1,
        "total": 6,
        "message": f"MGQ-Former 数据准备完成：{n_samples} 条样本，{n_positive} 条高风险",
    })
    if n_samples == 0:
        raise ValueError("无可训练的质量快照样本")

    single_scaler = StandardScaler()
    batch_scaler = StandardScaler()
    X_single_scaled = single_scaler.fit_transform(X_single).astype(np.float32)
    X_batch_scaled = batch_scaler.fit_transform(X_batch).astype(np.float32)

    can_split = n_samples >= 8 and len(set(y.tolist())) > 1 and min(np.bincount(y)) >= 2
    if can_split:
        train_idx, val_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    else:
        train_idx = val_idx = np.arange(n_samples)

    config = MGQFormerConfig(single_dim=len(feature_names))
    model = MGQFormerNet(config)
    pos_weight = 1.0
    if n_positive > 0:
        pos_weight = max(1.0, float(n_samples - n_positive) / max(1.0, float(n_positive)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_dataset = TensorDataset(
        torch.from_numpy(X_single_scaled[train_idx]),
        torch.from_numpy(X_batch_scaled[train_idx]),
        torch.from_numpy(y[train_idx].astype(np.float32)),
    )
    loader = DataLoader(train_dataset, batch_size=min(config.batch_size, max(1, len(train_dataset))), shuffle=True)

    _emit(emit_fn, "progress", {"step": 2, "total": 6, "message": "开始训练跨粒度深度融合网络..."})
    train_loss_curve = []
    best_state = None
    best_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        for xb_single, xb_batch, yb in loader:
            optimizer.zero_grad()
            logits = model(xb_single, xb_batch)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        loss_val = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_loss_curve.append({"step": epoch + 1, "loss": round(loss_val, 5)})
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(
                    torch.from_numpy(X_single_scaled[val_idx]),
                    torch.from_numpy(X_batch_scaled[val_idx]),
                )
                val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_y = y[val_idx]
            live_auc = roc_auc_score(val_y, val_prob) if len(set(val_y.tolist())) > 1 else 0.0
            _emit(emit_fn, "curve_update", {
                "step": epoch + 1,
                "n_estimators": config.epochs,
                "loss": round(loss_val, 5),
                "auc": round(float(live_auc), 5),
                "progress_pct": int((epoch + 1) / config.epochs * 70),
            })

    if best_state:
        model.load_state_dict(best_state)

    _emit(emit_fn, "progress", {"step": 3, "total": 6, "message": "评估 MGQ-Former 训练效果..."})
    package = MGQFormerPackage(
        model=model,
        single_scaler=single_scaler,
        batch_scaler=batch_scaler,
        feature_names=feature_names,
        batch_feature_names=BATCH_FEATURE_NAMES,
        config=config.__dict__,
    )
    y_prob = package.predict_proba(X_single, X_batch)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y, y_prob) if len(set(y.tolist())) > 1 else 0.0
    f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    if len(set(y.tolist())) > 1:
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
    else:
        roc_data = []

    feature_importance = _gradient_importance(package, X_single, X_batch, feature_names)

    return {
        "model_package": package,
        "X_single": X_single,
        "X_batch": X_batch,
        "y": y,
        "meta": meta,
        "auc_score": float(auc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "confusion_matrix": cm,
        "roc_data": roc_data,
        "train_loss_curve": train_loss_curve,
        "feature_importance": feature_importance,
        "architecture": get_mgqformer_architecture(),
    }


def _gradient_importance(package: MGQFormerPackage, X_single, X_batch, feature_names):
    x_single = torch.tensor(package.single_scaler.transform(X_single).astype(np.float32), requires_grad=True)
    x_batch = torch.tensor(package.batch_scaler.transform(X_batch).astype(np.float32), requires_grad=True)
    logits = package.model(x_single, x_batch).sum()
    logits.backward()
    single_imp = x_single.grad.detach().abs().mean(dim=0).numpy()
    batch_imp = x_batch.grad.detach().abs().mean(dim=0).numpy()
    single_items = [
        {"name": name, "label": name, "importance": round(float(val), 6), "group": "meter"}
        for name, val in zip(feature_names, single_imp)
    ]
    batch_items = [
        {"name": name, "label": name, "importance": round(float(val), 6), "group": "batch"}
        for name, val in zip(BATCH_FEATURE_NAMES, batch_imp)
    ]
    items = single_items + batch_items
    items.sort(key=lambda item: item["importance"], reverse=True)
    return items


def get_mgqformer_architecture() -> dict:
    return {
        "name": "MGQ-Former",
        "full_name": "Multi-Granularity Quality Fusion Transformer",
        "inputs": [
            {"name": "meter_token", "description": "46维单表质量、生产追溯、电路板工艺与可靠性样机画像特征"},
            {"name": "batch_token", "description": "8维批次抽检老化质量画像特征"},
        ],
        "modules": [
            "Meter Feature Encoder: LayerNorm + MLP",
            "Batch Aging Profile Encoder: LayerNorm + MLP",
            "Fusion Transformer: CLS + meter token + batch token",
            "Risk Head: GELU MLP + sigmoid probability",
            "Gradient Saliency: meter/batch feature contribution",
        ],
        "mermaid": """flowchart LR
    A[单表生产/质量/电路板工艺/可靠性样机特征 46维] --> B[Meter Encoder]
    C[批次抽检老化画像 8维] --> D[Batch Encoder]
    B --> E[[CLS + 两粒度Token]]
    D --> E
    E --> F[Transformer Fusion]
    F --> G[风险概率/风险等级]
    F --> H[梯度重要性解释]""",
    }
