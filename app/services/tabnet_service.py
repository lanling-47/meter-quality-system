"""
TabNet 质量风险预测服务
基于 "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

架构亮点:
  - Sequential Attention: 每步自适应选择最相关特征（内建可解释性）
  - Sparsemax 注意力: 产生稀疏特征权重，直观显示关键特征
  - Instance-wise 特征重要度: 每个样本有独立的特征注意力分布
"""
import os
import sys
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg


# ─── 模型参数（针对小样本优化）────────────────────────────────────────
TABNET_PARAMS = {
    "n_d": 16,          # 决策步特征宽度
    "n_a": 16,          # Attention 嵌入宽度（与 n_d 相同为最佳实践）
    "n_steps": 3,       # Sequential Attention 步数（3-5 合适）
    "gamma": 1.3,       # 特征重用惩罚系数（>1 鼓励不同步使用不同特征）
    "n_independent": 2, # 每步独立层数
    "n_shared": 2,      # 共享层数
    "momentum": 0.02,   # BN动量（小样本用小值）
    "mask_type": "sparsemax",  # 稀疏注意力（vs softmax 更具可解释性）
    "device_name": "cpu",
    "verbose": 0,
    "seed": 42,
}

TABNET_FIT_PARAMS = {
    "max_epochs": 200,
    "patience": 30,     # 早停耐心值
    "batch_size": 64,
    "virtual_batch_size": 32,
    "num_workers": 0,
    "drop_last": False,
}


def _build_tabnet():
    """创建 TabNet 分类器实例"""
    from pytorch_tabnet.tab_model import TabNetClassifier
    return TabNetClassifier(**TABNET_PARAMS)


def train_tabnet(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    emit_fn=None,
) -> dict:
    """
    训练 TabNet 模型

    Parameters
    ----------
    X : 特征矩阵 (n_samples, n_features)
    y : 标签向量
    feature_names : 特征名列表
    emit_fn : SSE 进度推送函数（可选）

    Returns
    -------
    dict 包含:
        model       : 训练好的 TabNetClassifier
        scaler      : 归一化器
        auc_score   : 全量训练集 AUC
        f1_score    : F1（高风险类）
        confusion_matrix : 混淆矩阵
        feature_importance : 特征重要度列表
        attention_masks    : 3步注意力权重（用于可视化）
        roc_data    : ROC 曲线数据点
        cv_metrics  : 交叉验证指标（mean ± std）
        train_loss_curve : 训练损失曲线
    """
    from imblearn.over_sampling import SMOTE

    def emit(event, data):
        if emit_fn:
            emit_fn(event, data)

    n_samples = len(X)
    n_positive = int(y.sum())

    emit("progress", {"step": 1, "total": 6, "message": f"TabNet 启动: {n_samples} 样本, {n_positive} 高风险"})

    # ── Step 1: MinMaxScaler 归一化（TabNet 对量纲敏感）─────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 2: SMOTE 过采样 ─────────────────────────────────────────────
    emit("progress", {"step": 2, "total": 6, "message": "SMOTE 过采样处理类别不平衡..."})
    if n_positive > 1 and n_positive < n_samples - 1:
        try:
            k_neighbors = max(1, min(n_positive - 1, 5))
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = sm.fit_resample(X_scaled, y)
            emit("smote_done", {
                "original": n_samples,
                "resampled": len(X_res),
                "message": f"SMOTE: {n_samples} → {len(X_res)} 样本"
            })
        except Exception as e:
            X_res, y_res = X_scaled, y
            emit("progress", {"step": 2, "total": 6, "message": f"SMOTE 跳过: {e}"})
    else:
        X_res, y_res = X_scaled, y

    # ── Step 3: 5折交叉验证 ────────────────────────────────────────────
    emit("progress", {"step": 3, "total": 6, "message": "5折交叉验证评估..."})
    cv_metrics = _cross_validate_tabnet(X_res, y_res, emit)

    # ── Step 4: 全量数据训练最终模型 ────────────────────────────────────
    emit("progress", {"step": 4, "total": 6, "message": "全量数据训练最终 TabNet 模型..."})
    model = _build_tabnet()

    # 记录训练损失曲线（通过 epoch_logs 回调）
    train_loss_curve = []

    class LossCurveCallback:
        def on_epoch_end(self, epoch, logs=None):
            if logs and "train_logloss" in logs:
                train_loss_curve.append({
                    "step": epoch + 1,
                    "loss": round(float(logs["train_logloss"]), 4),
                })
                if (epoch + 1) % 20 == 0:
                    emit("curve_update", {
                        "step": epoch + 1,
                        "n_estimators": TABNET_FIT_PARAMS["max_epochs"],
                        "loss": round(float(logs["train_logloss"]), 4),
                        "auc": round(float(logs.get("train_auc", 0.5)), 4),
                        "progress_pct": int((epoch + 1) / TABNET_FIT_PARAMS["max_epochs"] * 60),
                    })

    try:
        callback = LossCurveCallback()
        model.fit(
            X_res, y_res.astype(int),
            eval_metric=["auc", "logloss"],
            callbacks=[callback],
            **TABNET_FIT_PARAMS,
        )
    except Exception as e:
        emit("progress", {"step": 4, "total": 6, "message": f"训练回调异常(忽略): {e}"})
        model = _build_tabnet()
        model.fit(X_res, y_res.astype(int), **TABNET_FIT_PARAMS)

    # ── Step 5: 评估指标 ────────────────────────────────────────────────
    emit("progress", {"step": 5, "total": 6, "message": "计算评估指标..."})
    y_prob_train = model.predict_proba(X_scaled)[:, 1]
    y_pred_train = (y_prob_train >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y, y_prob_train) if len(set(y)) > 1 else 0.0
        f1 = f1_score(y, y_pred_train, pos_label=1, zero_division=0)
        cm = confusion_matrix(y, y_pred_train).tolist()
        fpr, tpr, _ = roc_curve(y, y_prob_train) if len(set(y)) > 1 else ([0, 1], [0, 1], [])
        roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
    except Exception:
        auc, f1, cm, roc_data = 0.0, 0.0, [[0, 0], [0, 0]], []

    # ── Step 6: 特征重要度 & 注意力 ────────────────────────────────────
    emit("progress", {"step": 6, "total": 6, "message": "提取 TabNet 注意力权重..."})
    feature_importance, attention_masks = _extract_tabnet_importance(
        model, X_scaled, feature_names
    )

    return {
        "model": model,
        "scaler": scaler,
        "auc_score": float(auc),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "roc_data": roc_data,
        "feature_importance": feature_importance,
        "attention_masks": attention_masks,
        "cv_metrics": cv_metrics,
        "train_loss_curve": train_loss_curve,
    }


def _cross_validate_tabnet(X: np.ndarray, y: np.ndarray, emit_fn=None) -> dict:
    """5折交叉验证，返回 mean ± std 指标"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"auc": [], "f1": [], "precision": [], "recall": [], "accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if len(set(y_train)) < 2 or len(set(y_val)) < 2:
            continue

        fold_model = _build_tabnet()
        try:
            fold_model.fit(X_train, y_train.astype(int), **{
                **TABNET_FIT_PARAMS,
                "max_epochs": 100,  # CV 用较少轮次
            })
            prob = fold_model.predict_proba(X_val)[:, 1]
            pred = (prob >= 0.5).astype(int)

            metrics["auc"].append(roc_auc_score(y_val, prob))
            metrics["f1"].append(f1_score(y_val, pred, pos_label=1, zero_division=0))
            metrics["precision"].append(precision_score(y_val, pred, pos_label=1, zero_division=0))
            metrics["recall"].append(recall_score(y_val, pred, pos_label=1, zero_division=0))
            metrics["accuracy"].append(accuracy_score(y_val, pred))

            if emit_fn:
                emit_fn("cv_fold", {
                    "fold": fold + 1,
                    "auc": round(float(metrics["auc"][-1]), 4),
                    "f1": round(float(metrics["f1"][-1]), 4),
                })
        except Exception as e:
            if emit_fn:
                emit_fn("progress", {"step": 3, "total": 6, "message": f"Fold {fold+1} 失败: {e}"})

    result = {}
    for k, vals in metrics.items():
        if vals:
            result[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "values": [round(float(v), 4) for v in vals],
            }
        else:
            result[k] = {"mean": 0.0, "std": 0.0, "values": []}
    return result


def _extract_tabnet_importance(model, X_scaled: np.ndarray, feature_names: list):
    """
    提取 TabNet 特征重要度与注意力权重

    TabNet 的特征重要度通过累加每步 attention mask 得到:
      importance_j = sum_t M_t[i,j] for all steps t and samples i

    Returns
    -------
    feature_importance : [{name, label, importance}]
    attention_masks    : [{step, weights: {feature: weight}}]
    """
    from app.services.predict_service import FEATURE_LABELS

    try:
        # TabNet explain 返回 (attention_masks, mask_weights)
        explain_matrix, masks = model.explain(X_scaled)

        # explain_matrix: (n_samples, n_features) 每个样本的全局重要度
        global_importance = np.abs(explain_matrix).mean(axis=0)

        feature_importance = []
        for i, fname in enumerate(feature_names):
            feature_importance.append({
                "name": fname,
                "label": FEATURE_LABELS.get(fname, fname),
                "importance": round(float(global_importance[i]), 6),
            })
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # 每步注意力权重（取样本均值）
        attention_masks_out = []
        if isinstance(masks, (list, tuple)):
            for step_idx, mask in enumerate(masks):
                step_weights = {}
                mean_mask = np.abs(mask).mean(axis=0) if mask.ndim > 1 else mask
                for i, fname in enumerate(feature_names[:len(mean_mask)]):
                    step_weights[fname] = round(float(mean_mask[i]), 6)
                attention_masks_out.append({
                    "step": step_idx + 1,
                    "weights": step_weights,
                })

        return feature_importance, attention_masks_out

    except Exception as e:
        # 降级：使用内置 feature_importances_
        try:
            fi = model.feature_importances_
            fi_list = []
            for i, fname in enumerate(feature_names[:len(fi)]):
                fi_list.append({
                    "name": fname,
                    "label": FEATURE_LABELS.get(fname, fname),
                    "importance": round(float(fi[i]), 6),
                })
            fi_list.sort(key=lambda x: x["importance"], reverse=True)
            return fi_list, []
        except Exception:
            return [], []


def predict_with_tabnet(model, scaler, feature_vector: np.ndarray) -> tuple:
    """
    用 TabNet 进行预测

    Returns
    -------
    (prob, attention_weights_for_sample)
    """
    x_scaled = scaler.transform(feature_vector)
    prob = float(model.predict_proba(x_scaled)[0][1])

    # 提取本样本的注意力权重
    try:
        explain_matrix, masks = model.explain(x_scaled)
        sample_importance = explain_matrix[0].tolist()
    except Exception:
        sample_importance = None

    return prob, sample_importance


def save_tabnet_model(model, scaler, feature_names: list, version_tag: str) -> Path:
    """保存 TabNet 模型（pkl 格式，与 LightGBM 兼容）"""
    cfg.ML_MODELS_DIR.mkdir(exist_ok=True)
    model_path = cfg.ML_MODELS_DIR / f"{version_tag}.pkl"
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_type": "tabnet",
    }, model_path)
    return model_path


def load_tabnet_model(model_path: str):
    """加载 TabNet 模型包"""
    pkg = joblib.load(model_path)
    return pkg.get("model"), pkg.get("scaler"), pkg.get("feature_names")
