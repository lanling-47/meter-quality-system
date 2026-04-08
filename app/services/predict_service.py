"""
AI 质量风险预测服务（含 SSE 实时训练进度推送）
"""
import json
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg

FEATURE_NAMES = [
    # 来料特征（3个）
    "crystal_ppm",
    "ct_ratio_err",
    "clock_accuracy_sd",
    # 出厂特征（1个）
    "factory_fault",
    # 老化试验基础特征（4个）
    "initial_error_max",
    "final_error_max",
    "error_degradation_rate",
    "clock_drift_rate",
    # 老化试验新增特征（来自未使用Sheet，6个）
    "clock_daily_error_max",    # 日计时最大日误差 s/d
    "error_variability_max",    # 误差变差最大值 %
    "error_consistency_max",    # 误差一致性偏差最大值 %
    "energy_balance_ratio",     # 正反向电量比
    "energy_peak_ratio",        # 尖电量占比
    # 通信特征（1个）
    "rs485_total_errors",
]

FEATURE_LABELS = {
    "crystal_ppm": "晶振频率误差(ppm)",
    "ct_ratio_err": "互感器比差(%)",
    "clock_accuracy_sd": "出厂时钟精度(s/d)",
    "factory_fault": "出厂故障标记",
    "initial_error_max": "初始最大基本误差(%)",
    "final_error_max": "末期最大基本误差(%)",
    "error_degradation_rate": "误差退化速率(%/100h)",
    "clock_drift_rate": "时钟漂移速率(s/d)",
    "clock_daily_error_max": "日计时最大日误差(s/d)",
    "error_variability_max": "误差变差最大值(%)",
    "error_consistency_max": "误差一致性偏差(%)",
    "energy_balance_ratio": "正反向电量比",
    "energy_peak_ratio": "尖电量占比",
    "rs485_total_errors": "RS485通信异常次数",
}


def _get_feature_matrix(db: Session):
    from app.models.db_models import QualityChain
    chains = db.query(QualityChain).all()
    X_rows, y, meta = [], [], []
    for c in chains:
        row = [
            c.crystal_ppm or 9.0,
            c.ct_ratio_err or 0.5,
            c.clock_accuracy_sd or 0.018,
            1.0 if c.factory_fault else 0.0,
            c.initial_error_max or 0.0,
            c.final_error_max or 0.0,
            c.error_degradation_rate or 0.0,
            c.clock_drift_rate or 0.0,
            # 新增特征（有缺失时用合理均值填充）
            c.clock_daily_error_max if c.clock_daily_error_max is not None else 0.22,
            c.error_variability_max if c.error_variability_max is not None else 0.006,
            c.error_consistency_max if c.error_consistency_max is not None else 0.086,
            c.energy_balance_ratio if c.energy_balance_ratio is not None else 1.0,
            c.energy_peak_ratio if c.energy_peak_ratio is not None else 0.246,
            float(c.rs485_total_errors or 0),
        ]
        X_rows.append(row)
        y.append(c.quality_label)
        meta.append({"meter_no": c.meter_no, "is_augmented": c.is_augmented})
    return np.array(X_rows), np.array(y), meta


# ─── 全局训练进度队列（支持 SSE 流式推送） ─────────────────────────────
_train_progress_queues: list[asyncio.Queue] = []


def _emit(event: str, data: dict):
    """向所有订阅者推送训练进度事件"""
    msg = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    for q in list(_train_progress_queues):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


async def training_event_stream() -> AsyncGenerator[str, None]:
    """SSE 事件流生成器"""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _train_progress_queues.append(q)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30)
                yield f"data: {msg}\n\n"
                if json.loads(msg).get("event") == "done":
                    break
            except asyncio.TimeoutError:
                yield "data: {\"event\":\"heartbeat\"}\n\n"
    finally:
        if q in _train_progress_queues:
            _train_progress_queues.remove(q)


def train_model(db: Session) -> dict:
    """训练质量风险预测模型（带详细进度事件）"""
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE

    X, y, meta = _get_feature_matrix(db)
    n_samples = len(X)
    n_positive = int(y.sum())
    version_tag = f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result = {
        "n_samples": n_samples,
        "n_positive": n_positive,
        "feature_names": FEATURE_NAMES,
        "model_type": "lightgbm",
        "version_tag": version_tag,
        "log": [],
    }

    _emit("start", {
        "n_samples": n_samples,
        "n_positive": n_positive,
        "version_tag": version_tag,
        "message": f"开始训练：共 {n_samples} 个样本，{n_positive} 个高风险样本"
    })

    if n_samples < 5:
        result["error"] = "样本数不足（需至少5条）"
        _emit("error", {"message": result["error"]})
        return result

    # ── SMOTE 过采样 ──────────────────────────────────────────────────
    _emit("progress", {"step": 1, "total": 5, "message": "执行 SMOTE 过采样处理类别不平衡..."})
    if n_positive > 0 and n_positive < n_samples and n_samples >= 6:
        try:
            k_neighbors = max(1, min(n_positive - 1, 3))
            sm = SMOTE(random_state=cfg.RANDOM_STATE, k_neighbors=k_neighbors)
            X_res, y_res = sm.fit_resample(X, y)
            msg = f"SMOTE完成：{n_samples} → {len(X_res)} 个样本"
            result["log"].append(msg)
            _emit("smote_done", {"original": n_samples, "resampled": len(X_res), "message": msg})
        except Exception as e:
            X_res, y_res = X, y
            result["log"].append(f"SMOTE跳过：{e}")
            _emit("progress", {"step": 1, "total": 5, "message": f"SMOTE跳过：{e}"})
    else:
        X_res, y_res = X, y

    # ── 超参数设置 ─────────────────────────────────────────────────────
    scale_pos_weight = max(1, (len(y_res) - int(y_res.sum())) / max(1, int(y_res.sum())))
    n_estimators = 100
    _emit("progress", {"step": 2, "total": 5,
          "message": f"配置 LightGBM（n_estimators={n_estimators}, scale_pos_weight={scale_pos_weight:.2f}）..."})

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.RANDOM_STATE,
        verbose=-1,
    )

    # ── 逐轮训练，记录 loss/auc 曲线 ───────────────────────────────────
    _emit("progress", {"step": 3, "total": 5, "message": "逐步训练中，收集损失和 AUC 曲线..."})

    train_loss_curve = []
    cv_auc_curve = []
    cv_folds = min(3, max(2, n_samples))

    # 模拟逐步训练进度（每10轮一次）
    step_sizes = list(range(10, n_estimators + 1, 10))
    for step in step_sizes:
        partial_model = lgb.LGBMClassifier(
            n_estimators=step, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=cfg.RANDOM_STATE, verbose=-1,
        )
        try:
            partial_model.fit(X_res, y_res)
            y_prob_train = partial_model.predict_proba(X_res)[:, 1]
            # 训练集 AUC 作为代理指标
            auc_train = roc_auc_score(y_res, y_prob_train) if len(set(y_res)) > 1 else 0.5
            # 粗略损失（log loss）
            eps = 1e-7
            log_loss = -float(np.mean(
                y_res * np.log(y_prob_train + eps) + (1 - y_res) * np.log(1 - y_prob_train + eps)
            ))
            train_loss_curve.append({"step": step, "loss": round(log_loss, 4)})
            cv_auc_curve.append({"step": step, "auc": round(auc_train, 4)})
            _emit("curve_update", {
                "step": step,
                "n_estimators": n_estimators,
                "loss": round(log_loss, 4),
                "auc": round(auc_train, 4),
                "progress_pct": int(step / n_estimators * 60),
            })
        except Exception:
            pass

    # ── 最终全量训练 ───────────────────────────────────────────────────
    _emit("progress", {"step": 4, "total": 5, "message": "最终训练并评估..."})
    model.fit(X_res, y_res)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, y_prob) if len(set(y)) > 1 else 0.0
        f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y, y_pred).tolist()
        # ROC 曲线
        fpr, tpr, _ = roc_curve(y, y_prob) if len(set(y)) > 1 else ([0, 1], [0, 1], [])
        roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
    except Exception:
        auc, f1, cm, roc_data = 0.0, 0.0, [[0, 0], [0, 0]], []

    result.update({
        "auc_score": float(auc),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "roc_data": roc_data,
        "train_loss_curve": train_loss_curve,
        "cv_auc_curve": cv_auc_curve,
    })
    result["log"].append(f"Train AUC: {auc:.3f}, F1: {f1:.3f}")

    # 特征重要度
    fi = model.feature_importances_.tolist()
    feature_importance = [
        {"name": FEATURE_NAMES[i], "label": FEATURE_LABELS[FEATURE_NAMES[i]], "importance": round(float(fi[i]), 4)}
        for i in range(len(FEATURE_NAMES))
    ]
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    result["feature_importance"] = feature_importance

    # ── 保存模型 ───────────────────────────────────────────────────────
    _emit("progress", {"step": 5, "total": 5, "message": "保存模型文件..."})
    cfg.ML_MODELS_DIR.mkdir(exist_ok=True)
    model_path = cfg.ML_MODELS_DIR / f"{version_tag}.pkl"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)
    result["model_path"] = str(model_path)

    # 更新数据库预测结果
    from app.models.db_models import QualityChain, ModelVersion
    chains = db.query(QualityChain).all()
    for i, c in enumerate(chains):
        if i < len(X):
            prob = float(model.predict_proba([X[i]])[0][1])
            c.predicted_risk = prob
            c.predicted_label = 1 if prob >= 0.5 else 0
    db.commit()

    # 保存模型版本
    db.query(ModelVersion).update({"is_active": False})
    mv = ModelVersion(
        version_tag=version_tag,
        model_type="lightgbm",
        model_path=str(model_path),
        n_samples=n_samples,
        auc_score=float(auc),
        f1_score=float(f1),
        feature_names_json=json.dumps(FEATURE_NAMES, ensure_ascii=False),
        is_active=True,
        train_log="\n".join(result["log"]),
    )
    db.add(mv)
    db.commit()

    _emit("done", {
        "version_tag": version_tag,
        "auc": float(auc),
        "f1": float(f1),
        "confusion_matrix": cm,
        "roc_data": roc_data,
        "feature_importance": feature_importance,
        "train_loss_curve": train_loss_curve,
        "cv_auc_curve": cv_auc_curve,
        "message": f"训练完成！AUC={auc:.3f}, F1={f1:.3f}",
    })

    return result


def load_active_model(db: Session):
    from app.models.db_models import ModelVersion
    mv = db.query(ModelVersion).filter_by(is_active=True).order_by(ModelVersion.created_at.desc()).first()
    if mv and mv.model_path and Path(mv.model_path).exists():
        pkg = joblib.load(mv.model_path)
        return pkg["model"], pkg["feature_names"], mv.version_tag
    return None, None, None


def activate_model_version(version_tag: str, db: Session) -> dict:
    """激活指定版本的模型"""
    from app.models.db_models import ModelVersion
    mv = db.query(ModelVersion).filter_by(version_tag=version_tag).first()
    if not mv:
        return {"success": False, "message": f"模型版本 {version_tag} 不存在"}
    if not mv.model_path or not Path(mv.model_path).exists():
        return {"success": False, "message": f"模型文件不存在：{mv.model_path}"}
    db.query(ModelVersion).update({"is_active": False})
    mv.is_active = True
    db.commit()
    return {"success": True, "message": f"已激活模型版本 {version_tag}"}


def upload_and_activate_model(model_bytes: bytes, model_type: str, db: Session) -> dict:
    """
    上传外部 pkl 模型文件并激活
    model_bytes: pkl 文件内容
    model_type: 模型类型描述
    """
    from app.models.db_models import ModelVersion
    version_tag = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg.ML_MODELS_DIR.mkdir(exist_ok=True)
    model_path = cfg.ML_MODELS_DIR / f"{version_tag}.pkl"

    # 保存文件
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    # 验证文件可加载
    try:
        pkg = joblib.load(model_path)
        if "model" not in pkg:
            raise ValueError("pkl 文件格式错误，缺少 'model' 键")
        feat_names = pkg.get("feature_names", FEATURE_NAMES)
    except Exception as e:
        model_path.unlink(missing_ok=True)
        return {"success": False, "message": f"模型文件无效：{e}"}

    # 快速测试
    test_X = np.zeros((1, len(feat_names)))
    try:
        pkg["model"].predict_proba(test_X)
    except Exception as e:
        model_path.unlink(missing_ok=True)
        return {"success": False, "message": f"模型预测测试失败：{e}"}

    # 写入数据库
    db.query(ModelVersion).update({"is_active": False})
    mv = ModelVersion(
        version_tag=version_tag,
        model_type=model_type or "外部上传",
        model_path=str(model_path),
        n_samples=0,
        auc_score=None,
        f1_score=None,
        feature_names_json=json.dumps(feat_names, ensure_ascii=False),
        is_active=True,
        train_log=f"外部上传模型，版本：{version_tag}",
    )
    db.add(mv)
    db.commit()
    return {"success": True, "version_tag": version_tag, "message": f"模型上传并激活成功：{version_tag}"}


def predict_single(features: dict, db: Session) -> dict:
    model, feat_names, version = load_active_model(db)
    x = np.array([[
        features.get("crystal_ppm", 9.0),
        features.get("ct_ratio_err", 0.5),
        features.get("clock_accuracy_sd", 0.018),
        float(features.get("factory_fault", False)),
        features.get("initial_error_max", 0.0),
        features.get("final_error_max", 0.0),
        features.get("error_degradation_rate", 0.0),
        features.get("clock_drift_rate", 0.0),
        features.get("clock_daily_error_max", 0.22),
        features.get("error_variability_max", 0.006),
        features.get("error_consistency_max", 0.086),
        features.get("energy_balance_ratio", 1.0),
        features.get("energy_peak_ratio", 0.246),
        float(features.get("rs485_total_errors", 0)),
    ]])
    if model is None:
        risk_prob = _rule_based_risk(features)
        risk_label = 1 if risk_prob >= 0.5 else 0
        risk_level = "高" if risk_prob >= 0.7 else ("中" if risk_prob >= 0.4 else "低")
        return {"risk_level": risk_level, "risk_prob": risk_prob, "risk_label": risk_label,
                "model_version": "规则引擎(无训练模型)", "shap_values": None}
    # 特征数对齐（允许外部上传的旧模型使用旧特征集）
    if feat_names and len(feat_names) != x.shape[1]:
        # 截断或填充到匹配的维度
        n = len(feat_names)
        if x.shape[1] > n:
            x = x[:, :n]
        else:
            pad = np.zeros((1, n - x.shape[1]))
            x = np.hstack([x, pad])
    prob = float(model.predict_proba(x)[0][1])
    label = 1 if prob >= 0.5 else 0
    level = "高" if prob >= 0.7 else ("中" if prob >= 0.4 else "低")
    return {"risk_level": level, "risk_prob": prob, "risk_label": label,
            "model_version": version, "shap_values": None,
            "feature_values": dict(zip(FEATURE_NAMES, x[0].tolist()))}


def _rule_based_risk(features: dict) -> float:
    score = 0.0
    if features.get("crystal_ppm", 0) > 15: score += 0.3
    if features.get("ct_ratio_err", 0) > 0.8: score += 0.3
    if features.get("factory_fault", False): score += 0.4
    if features.get("rs485_total_errors", 0) > 0: score += 0.3
    if features.get("clock_drift_rate", 0) > 0.3: score += 0.2
    return min(score, 1.0)
