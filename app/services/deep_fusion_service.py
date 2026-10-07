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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


BATCH_FEATURE_NAMES = [
    "sample_count",
    "high_risk_rate",
    "avg_final_error_max",
    "max_final_error_max",
    "avg_error_degradation_rate",
    "max_clock_drift_rate",
    "max_rs485_total_errors",
    "profile_confidence",
    "posterior_high_risk_mean",
    "posterior_high_risk_variance",
    "mapping_confidence",
    "batch_available",
]

# These fields directly participate in the current rule-generated label.  A
# research evaluation that feeds them back to the model measures rule recovery,
# not generalization to future failures.  They are masked by default and kept in
# metadata so the operational rule engine can still display them separately.
RULE_LABEL_FEATURE_NAMES = {
    "final_error_max",
    "clock_drift_rate",
    "clock_daily_error_max",
    "error_consistency_max",
    "rs485_total_errors",
}


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
    reliability_dim: int = 5
    consistency_loss_weight: float = 0.10
    consistency_sample_kappa: float = 8.0
    consistency_variance_reference: float = 0.02
    mask_rule_label_features: bool = True


class MGQFormerNet(nn.Module):
    """Reliability-aware multi-granularity quality fusion network (RA-MGQ)."""

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
        self.reliability_gate = nn.Sequential(
            nn.Linear(config.d_model * 2 + config.reliability_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
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

    def forward(
        self,
        x_single: torch.Tensor,
        x_batch: torch.Tensor,
        reliability: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        n = x_single.shape[0]
        cls = self.cls_token.expand(n, -1, -1)
        single_token = self.single_encoder(x_single).unsqueeze(1)
        batch_token = self.batch_encoder(x_batch).unsqueeze(1)
        if not hasattr(self, "reliability_gate"):
            # Backward-compatible path for locally serialized legacy models.
            tokens = torch.cat([cls, single_token, batch_token], dim=1) + self.token_type
            fused = self.fusion(tokens)
            return self.classifier(fused[:, 0, :]).squeeze(-1)
        if reliability is None:
            reliability = x_single.new_ones((n, self.config.reliability_dim))
        mapping_confidence = reliability[:, 3:4].clamp(0.0, 1.0)
        batch_available = reliability[:, 4:5].clamp(0.0, 1.0)
        learned_gate = torch.sigmoid(self.reliability_gate(torch.cat([
            single_token.squeeze(1), batch_token.squeeze(1), reliability,
        ], dim=-1)))
        effective_gate = learned_gate * mapping_confidence * batch_available
        batch_token = batch_token * effective_gate.unsqueeze(1)
        tokens = torch.cat([cls, single_token, batch_token], dim=1) + self.token_type
        padding_mask = torch.zeros((n, 3), dtype=torch.bool, device=x_single.device)
        padding_mask[:, 2] = (mapping_confidence.squeeze(-1) <= 0.0) | (batch_available.squeeze(-1) <= 0.0)
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        logits = self.classifier(fused[:, 0, :]).squeeze(-1)
        if return_aux:
            return {
                "logits": logits,
                "batch_gate": effective_gate.squeeze(-1),
                "learned_gate": learned_gate.squeeze(-1),
            }
        return logits


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
        masked_feature_indices: list[int] | None = None,
    ):
        self.model = model.cpu().eval()
        self.single_scaler = single_scaler
        self.batch_scaler = batch_scaler
        self.feature_names = feature_names
        self.batch_feature_names = batch_feature_names
        self.config = config
        self.masked_feature_indices = list(masked_feature_indices or [])
        self.model_type = "mgqformer"

    def predict_proba(self, X, batch_features=None):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        masked_indices = getattr(self, "masked_feature_indices", [])
        if masked_indices:
            X_arr = X_arr.copy()
            X_arr[:, masked_indices] = 0.0
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
        reliability = _reliability_from_batch_matrix(batch_arr, self.batch_feature_names)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(X_scaled),
                torch.from_numpy(batch_scaled),
                torch.from_numpy(reliability),
            )
            probs = torch.sigmoid(logits).numpy()
        return np.vstack([1.0 - probs, probs]).T


def _profile_feature_map(profile, mapping_confidence: float = 0.0) -> dict[str, float]:
    if profile is None:
        return {name: 0.0 for name in BATCH_FEATURE_NAMES}
    sample_count = max(int(profile.sample_count or 0), 0)
    high_risk_count = int(profile.high_risk_count or 0)
    if high_risk_count == 0 and sample_count > 0:
        high_risk_count = int(round(float(profile.high_risk_rate or 0.0) * sample_count))
    high_risk_count = min(max(high_risk_count, 0), sample_count)
    alpha = 1.0 + high_risk_count
    beta = 1.0 + sample_count - high_risk_count
    posterior_mean = alpha / (alpha + beta)
    posterior_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    mapping_confidence = float(np.clip(mapping_confidence, 0.0, 1.0))
    return {
        "sample_count": float(sample_count),
        "high_risk_rate": float(profile.high_risk_rate or 0.0),
        "avg_final_error_max": float(profile.avg_final_error_max or 0.0),
        "max_final_error_max": float(profile.max_final_error_max or 0.0),
        "avg_error_degradation_rate": float(profile.avg_error_degradation_rate or 0.0),
        "max_clock_drift_rate": float(profile.max_clock_drift_rate or 0.0),
        "max_rs485_total_errors": float(profile.max_rs485_total_errors or 0),
        "profile_confidence": float(np.clip(profile.confidence or 0.0, 0.0, 1.0)),
        "posterior_high_risk_mean": float(posterior_mean),
        "posterior_high_risk_variance": float(posterior_variance),
        "mapping_confidence": mapping_confidence,
        "batch_available": float(sample_count > 0 and mapping_confidence > 0.0),
    }


def _profile_to_batch_features(
    profile,
    mapping_confidence: float = 0.0,
    feature_names: list[str] | None = None,
) -> list[float]:
    names = feature_names or BATCH_FEATURE_NAMES
    feature_map = _profile_feature_map(profile, mapping_confidence)
    # Legacy packages called the final field ``confidence``.
    feature_map["confidence"] = feature_map["profile_confidence"] * mapping_confidence
    return [float(feature_map.get(name, 0.0)) for name in names]


def _reliability_from_batch_matrix(
    matrix: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Extract raw gate inputs without passing them through StandardScaler."""
    index = {name: column for column, name in enumerate(feature_names)}

    def column(name: str, default: float = 0.0) -> np.ndarray:
        if name in index and index[name] < matrix.shape[1]:
            return matrix[:, index[name]].astype(np.float32)
        return np.full(matrix.shape[0], default, dtype=np.float32)

    sample_count = np.log1p(np.clip(column("sample_count"), 0.0, None))
    posterior_variance = np.clip(column("posterior_high_risk_variance"), 0.0, None)
    posterior_mean = np.clip(column("posterior_high_risk_mean", default=0.5), 0.0, 1.0)
    mapping_confidence = np.clip(
        column("mapping_confidence", default=1.0), 0.0, 1.0
    )
    batch_available = np.clip(
        column("batch_available", default=1.0), 0.0, 1.0
    )
    return np.column_stack([
        sample_count,
        posterior_mean,
        posterior_variance,
        mapping_confidence,
        batch_available,
    ]).astype(np.float32)


def batch_features_from_key(
    batch_key: str | None,
    db,
    feature_names: list[str] | None = None,
    mapping_confidence: float | None = None,
) -> list[float]:
    from app.models.db_models import BatchQualityProfile

    profile = None
    if batch_key:
        profile = db.query(BatchQualityProfile).filter_by(batch_key=batch_key).first()
        if profile is None:
            profile = db.query(BatchQualityProfile).filter_by(order_no=batch_key).first()
    resolved_confidence = (
        float(mapping_confidence)
        if mapping_confidence is not None
        else (1.0 if batch_key and profile is not None else 0.0)
    )
    return _profile_to_batch_features(profile, resolved_confidence, feature_names)


def build_deep_training_matrix(db):
    """Return meter features, batch profile features, labels and metadata."""
    from app.models.db_models import BatchQualityProfile, QualityChain, QualityFeatureSnapshot
    from app.services.data_linkage_service import (
        build_quality_feature_snapshots,
        get_snapshot_feature_matrix,
    )

    build_quality_feature_snapshots(db, include_synthetic_demo=False)
    snapshots = db.query(QualityFeatureSnapshot).order_by(QualityFeatureSnapshot.meter_no).all()
    X_single_matrix, y_arr, feature_meta = get_snapshot_feature_matrix(db)
    profiles = db.query(BatchQualityProfile).all()
    profiles_by_key = {p.batch_key: p for p in profiles if p.batch_key}
    profiles_by_key.update({p.order_no: p for p in profiles if p.order_no})
    augmented_meters = {
        row[0]
        for row in db.query(QualityChain.meter_no).filter(QualityChain.is_augmented.is_(True)).all()
    }

    X_single, X_batch, y, meta = [], [], [], []
    for idx, snapshot in enumerate(snapshots):
        if snapshot.meter_no in augmented_meters or snapshot.source in {"synthetic_demo", "sampled_batch_demo"}:
            continue
        X_single.append(X_single_matrix[idx].tolist())
        mapping_confidence = float(np.clip(snapshot.confidence or 0.0, 0.0, 1.0))
        profile = (
            profiles_by_key.get(snapshot.batch_key)
            if snapshot.batch_key and mapping_confidence > 0.0
            else None
        )
        X_batch.append(_profile_to_batch_features(profile, mapping_confidence))
        y.append(int(y_arr[idx] if idx < len(y_arr) else snapshot.quality_label or 0))
        snapshot_meta = feature_meta[idx] if idx < len(feature_meta) else {}
        meta.append({
            "meter_no": snapshot.meter_no,
            "batch_key": snapshot.batch_key,
            "process_profile_key": snapshot_meta.get("process_profile_key"),
            "source": snapshot.source,
            "confidence": snapshot.confidence,
            "batch_available": profile is not None,
            "is_augmented": False,
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


def _group_holdout_indices(y: np.ndarray, meta: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Split by trusted batch; unlinked meters remain independent groups."""
    indices = np.arange(len(y))
    groups = np.asarray([
        (
            f"batch:{item.get('batch_key')}"
            if item.get("batch_key") and float(item.get("confidence") or 0.0) > 0.0
            else f"meter:{item.get('meter_no')}"
        )
        for item in meta
    ])
    can_split = len(y) >= 8 and len(np.unique(groups)) >= 2 and len(np.unique(y)) > 1
    if not can_split:
        return indices, indices, "resubstitution_insufficient_groups"
    first_candidate = None
    splitter = GroupShuffleSplit(n_splits=24, test_size=0.25, random_state=seed)
    for train_idx, val_idx in splitter.split(indices, y, groups):
        if first_candidate is None:
            first_candidate = (train_idx, val_idx)
        if len(np.unique(y[train_idx])) > 1 and len(np.unique(y[val_idx])) > 1:
            return train_idx, val_idx, "group_holdout"
    if first_candidate is not None:
        return first_candidate[0], first_candidate[1], "group_holdout_single_class_fold"
    return indices, indices, "resubstitution_insufficient_groups"


def _batch_consistency_loss(
    probability: torch.Tensor,
    reliability: torch.Tensor,
    batch_group_ids: torch.Tensor,
    sample_kappa: float = 8.0,
    variance_reference: float = 0.02,
) -> torch.Tensor:
    """Match each trusted batch's aggregated meter risk to its posterior mean.

    Unlinked samples carry group id ``-1`` and make no contribution.  The
    aggregation is deliberately performed before the posterior comparison so a
    sampled batch statistic is never copied into every meter as a pseudo-label.
    Mapping confidence has two distinct roles: it defines the relative weights
    inside a batch aggregate, and its batch mean scales the absolute loss
    contribution.  The normalization intentionally excludes that confidence
    factor, so uniformly reducing a batch's confidence cannot cancel out.
    """
    posterior_mean = reliability[:, 1]
    posterior_variance = reliability[:, 2]
    sample_count = torch.expm1(reliability[:, 0]).clamp_min(0.0)
    mapping_confidence = reliability[:, 3].clamp(0.0, 1.0)
    available = reliability[:, 4].clamp(0.0, 1.0)
    sample_weight = mapping_confidence * available
    valid = (batch_group_ids >= 0) & (sample_weight > 0)
    if not torch.any(valid):
        return probability.sum() * 0.0

    weighted_errors = []
    normalization_weights = []
    for group_id in torch.unique(batch_group_ids[valid]):
        group_mask = valid & (batch_group_ids == group_id)
        group_sample_weight = sample_weight[group_mask]
        # ``valid`` guarantees every selected weight is positive.  Dividing by
        # the actual sum preserves the relative aggregate even for a uniformly
        # tiny positive mapping confidence; a fixed floor would distort it.
        normalizer = group_sample_weight.sum()
        aggregate_probability = (
            probability[group_mask] * group_sample_weight
        ).sum() / normalizer
        aggregate_posterior = (
            posterior_mean[group_mask] * group_sample_weight
        ).sum() / normalizer
        aggregate_variance = (
            posterior_variance[group_mask] * group_sample_weight
        ).sum() / normalizer
        aggregate_sample_count = (
            sample_count[group_mask] * group_sample_weight
        ).sum() / normalizer
        aggregate_available = available[group_mask].max()
        aggregate_mapping_confidence = mapping_confidence[group_mask].mean()
        sample_term = aggregate_sample_count / (
            aggregate_sample_count + max(float(sample_kappa), 1e-6)
        )
        variance_term = float(variance_reference) / (
            aggregate_variance + max(float(variance_reference), 1e-6)
        )
        base_batch_weight = aggregate_available * sample_term * variance_term
        effective_batch_weight = base_batch_weight * aggregate_mapping_confidence
        aggregate_probability = aggregate_probability.clamp(1e-6, 1.0 - 1e-6)
        cross_entropy = -(
            aggregate_posterior * torch.log(aggregate_probability)
            + (1.0 - aggregate_posterior) * torch.log(1.0 - aggregate_probability)
        )
        weighted_errors.append(cross_entropy * effective_batch_weight)
        normalization_weights.append(base_batch_weight)

    total_weight = torch.stack(normalization_weights).sum()
    if not torch.any(total_weight > 0):
        return probability.sum() * 0.0
    return torch.stack(weighted_errors).sum() / total_weight.clamp_min(1e-6)


def _trusted_batch_group_ids(meta: list[dict]) -> np.ndarray:
    """Encode only provenance-linked batches; ``-1`` denotes no trusted link."""
    group_lookup: dict[str, int] = {}
    group_ids = np.full(len(meta), -1, dtype=np.int64)
    for index, item in enumerate(meta):
        batch_key = item.get("batch_key")
        confidence = float(item.get("confidence") or 0.0)
        if not batch_key or confidence <= 0.0 or not item.get("batch_available"):
            continue
        key = str(batch_key)
        if key not in group_lookup:
            group_lookup[key] = len(group_lookup)
        group_ids[index] = group_lookup[key]
    return group_ids


def train_mgqformer(db, feature_names: list[str], emit_fn=None) -> dict:
    """Train RA-MGQ with group holdout and provenance-aware batch gating."""
    torch.manual_seed(42)
    np.random.seed(42)

    X_single, X_batch, y, meta = build_deep_training_matrix(db)
    n_samples = len(y)
    n_positive = int(y.sum())
    _emit(emit_fn, "progress", {
        "step": 1,
        "total": 6,
        "message": f"RA-MGQ 数据准备完成：{n_samples} 条真实样本，{n_positive} 条高风险",
    })
    if n_samples == 0:
        raise ValueError("无可训练的质量快照样本")

    config = MGQFormerConfig(single_dim=len(feature_names))
    masked_feature_indices = (
        [index for index, name in enumerate(feature_names) if name in RULE_LABEL_FEATURE_NAMES]
        if config.mask_rule_label_features
        else []
    )
    X_model = X_single.copy()
    if masked_feature_indices:
        X_model[:, masked_feature_indices] = 0.0
    train_idx, val_idx, evaluation_scope = _group_holdout_indices(y, meta, config.seed)
    single_scaler = StandardScaler().fit(X_model[train_idx])
    batch_scaler = StandardScaler().fit(X_batch[train_idx])
    X_single_scaled = single_scaler.transform(X_model).astype(np.float32)
    X_batch_scaled = batch_scaler.transform(X_batch).astype(np.float32)
    reliability = _reliability_from_batch_matrix(X_batch, BATCH_FEATURE_NAMES)
    batch_group_ids = _trusted_batch_group_ids(meta)

    model = MGQFormerNet(config)
    pos_weight = 1.0
    train_positive = int(y[train_idx].sum())
    if train_positive > 0:
        pos_weight = max(
            1.0,
            float(len(train_idx) - train_positive) / max(1.0, float(train_positive)),
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_dataset = TensorDataset(
        torch.from_numpy(X_single_scaled[train_idx]),
        torch.from_numpy(X_batch_scaled[train_idx]),
        torch.from_numpy(reliability[train_idx]),
        torch.from_numpy(y[train_idx].astype(np.float32)),
    )
    loader = DataLoader(train_dataset, batch_size=min(config.batch_size, max(1, len(train_dataset))), shuffle=True)

    _emit(emit_fn, "progress", {"step": 2, "total": 6, "message": "开始训练可靠度感知跨粒度融合网络..."})
    train_loss_curve = []
    best_state = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        for xb_single, xb_batch, xb_reliability, yb in loader:
            optimizer.zero_grad()
            output = model(xb_single, xb_batch, xb_reliability, return_aux=True)
            logits = output["logits"]
            supervised_loss = criterion(logits, yb)
            supervised_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(supervised_loss.detach().cpu()))

        # Evaluate the batch term over the complete training partition so every
        # trusted batch is aggregated before comparison with its posterior.
        optimizer.zero_grad()
        consistency_output = model(
            torch.from_numpy(X_single_scaled[train_idx]),
            torch.from_numpy(X_batch_scaled[train_idx]),
            torch.from_numpy(reliability[train_idx]),
        )
        consistency_loss = _batch_consistency_loss(
            torch.sigmoid(consistency_output),
            torch.from_numpy(reliability[train_idx]),
            torch.from_numpy(batch_group_ids[train_idx]),
            sample_kappa=config.consistency_sample_kappa,
            variance_reference=config.consistency_variance_reference,
        )
        consistency_penalty_value = 0.0
        if consistency_loss.requires_grad and float(consistency_loss.detach()) > 0.0:
            weighted_consistency = config.consistency_loss_weight * consistency_loss
            weighted_consistency.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            consistency_penalty_value = float(weighted_consistency.detach().cpu())

        loss_val = (float(np.mean(epoch_losses)) if epoch_losses else 0.0) + consistency_penalty_value
        model.eval()
        with torch.no_grad():
            val_output = model(
                torch.from_numpy(X_single_scaled[val_idx]),
                torch.from_numpy(X_batch_scaled[val_idx]),
                torch.from_numpy(reliability[val_idx]),
                return_aux=True,
            )
            val_logits = val_output["logits"]
            validation_loss = criterion(
                val_logits,
                torch.from_numpy(y[val_idx].astype(np.float32)),
            )
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        validation_loss_value = float(validation_loss.detach().cpu())
        train_loss_curve.append({
            "step": epoch + 1,
            "loss": round(loss_val, 5),
            "validation_loss": round(validation_loss_value, 5),
        })
        if validation_loss_value < best_loss - 1e-5:
            best_loss = validation_loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 20 == 0:
            val_y = y[val_idx]
            live_auc = roc_auc_score(val_y, val_prob) if len(set(val_y.tolist())) > 1 else 0.0
            _emit(emit_fn, "curve_update", {
                "step": epoch + 1,
                "n_estimators": config.epochs,
                "loss": round(loss_val, 5),
                "auc": round(float(live_auc), 5),
                "progress_pct": int((epoch + 1) / config.epochs * 70),
            })
        if stale >= 30:
            break

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
        masked_feature_indices=masked_feature_indices,
    )
    all_prob = package.predict_proba(X_single, X_batch)[:, 1]
    y_eval = y[val_idx]
    y_prob = all_prob[val_idx]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_eval, y_prob) if len(set(y_eval.tolist())) > 1 else 0.0
    f1 = f1_score(y_eval, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_eval, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_eval, y_pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)
    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1]).tolist()
    if len(set(y_eval.tolist())) > 1:
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
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
        "all_probabilities": all_prob,
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
        "evaluation_scope": evaluation_scope,
        "validation_sample_count": int(len(val_idx)),
        "masked_rule_features": [feature_names[index] for index in masked_feature_indices],
        "leakage_audit": {
            "scaler_fit_scope": "training_groups_only",
            "model_selection_metric": "validation_bce",
            "synthetic_rows_excluded": True,
            "rule_label_features_masked": bool(masked_feature_indices),
            "unlinked_batch_profiles_blocked": True,
        },
    }


def _gradient_importance(package: MGQFormerPackage, X_single, X_batch, feature_names):
    X_model = np.asarray(X_single, dtype=np.float32).copy()
    masked_indices = getattr(package, "masked_feature_indices", [])
    if masked_indices:
        X_model[:, masked_indices] = 0.0
    x_single = torch.tensor(package.single_scaler.transform(X_model).astype(np.float32), requires_grad=True)
    x_batch = torch.tensor(package.batch_scaler.transform(X_batch).astype(np.float32), requires_grad=True)
    reliability = torch.from_numpy(
        _reliability_from_batch_matrix(np.asarray(X_batch), package.batch_feature_names)
    )
    logits = package.model(x_single, x_batch, reliability).sum()
    logits.backward()
    single_imp = x_single.grad.detach().abs().mean(dim=0).numpy()
    batch_imp = x_batch.grad.detach().abs().mean(dim=0).numpy()
    if masked_indices:
        single_imp[np.asarray(masked_indices, dtype=np.int64)] = 0.0
    single_items = [
        {"name": name, "label": name, "importance": round(float(val), 6), "group": "meter"}
        for name, val in zip(feature_names, single_imp)
    ]
    batch_items = [
        {"name": name, "label": name, "importance": round(float(val), 6), "group": "batch"}
        for name, val in zip(package.batch_feature_names, batch_imp)
    ]
    items = single_items + batch_items
    items.sort(key=lambda item: item["importance"], reverse=True)
    return items


def get_mgqformer_architecture() -> dict:
    return {
        "name": "RA-MGQ",
        "full_name": "Reliability-Aware Multi-Granularity Quality Fusion",
        "inputs": [
            {"name": "meter_token", "description": "46维单表质量、生产追溯、电路板工艺与可靠性样机画像特征"},
            {"name": "batch_token", "description": "批次抽检统计量及Beta-Binomial风险后验"},
            {"name": "reliability", "description": "样本量、后验方差、映射置信度和批次可用掩码"},
        ],
        "modules": [
            "Meter Feature Encoder: LayerNorm + MLP",
            "Batch Posterior Encoder: sampled-aging profile + Beta-Binomial posterior",
            "Reliability Gate: exact zero contribution for an unlinked batch",
            "Fusion Transformer: CLS + meter token + gated batch token",
            "Batch-meter posterior consistency regularization",
            "Risk Head: GELU MLP + sigmoid probability",
            "Leakage audit: group split, train-only scaling, rule-feature masking",
        ],
        "mermaid": """flowchart LR
    A[单表生产/质量/电路板工艺/可靠性样机特征 46维] --> B[Meter Encoder]
    C[批次抽检计数与退化画像] --> D[Beta-Binomial后验/Batch Encoder]
    R[映射置信度/缺失掩码] --> H[可靠度门控]
    D --> H
    B --> E[[CLS + Meter + 门控Batch Token]]
    H --> E
    E --> F[Transformer Fusion]
    F --> G[风险概率/风险等级]
    F --> I[来源审计/特征贡献]""",
    }
