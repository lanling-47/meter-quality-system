"""?????????????????????????????"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sqlalchemy.orm import Session

import config as cfg

FACTORY_FEATURE_NAMES = [
    "crystal_ppm", "ct_ratio_err", "clock_accuracy_sd", "meter_constant_err",
    "ccd_retry_count", "func_retry_count", "reverify_retry_count", "total_retry_count",
    "smt_aoi_defect", "wave_aoi_defect", "repaired", "repair_count",
    "leak_current_max", "prod_quality_risk_score",
]

FACTORY_FEATURE_LABELS = {
    "crystal_ppm": "\u6676\u632f\u9891\u7387\u8bef\u5dee(ppm)", "ct_ratio_err": "\u4e92\u611f\u5668\u6bd4\u5dee(%)",
    "clock_accuracy_sd": "\u51fa\u5382\u65f6\u949f\u51c6\u786e\u5ea6(s/d)", "meter_constant_err": "\u4eea\u8868\u5e38\u6570\u8bef\u5dee(%)",
    "ccd_retry_count": "CCD\u91cd\u6d4b\u6b21\u6570", "func_retry_count": "\u529f\u80fd\u6d4b\u8bd5\u91cd\u6d4b\u6b21\u6570",
    "reverify_retry_count": "\u590d\u6821\u91cd\u6d4b\u6b21\u6570", "total_retry_count": "\u603b\u91cd\u6d4b\u6b21\u6570",
    "smt_aoi_defect": "\u8d34\u7247AOI\u7f3a\u9677", "wave_aoi_defect": "\u6ce2\u5cf0\u710aAOI\u7f3a\u9677",
    "repaired": "\u662f\u5426\u7ef4\u4fee", "repair_count": "\u7ef4\u4fee\u6b21\u6570",
    "leak_current_max": "\u6700\u5927\u6f0f\u7535\u6d41", "prod_quality_risk_score": "\u751f\u4ea7\u8d28\u91cf\u98ce\u9669\u5206",
}

NORMAL = "\u6b63\u5e38"
FAULT_TYPES = ["\u9891\u5dee\u504f\u5927", "\u6bd4\u5dee\u504f\u5927", "\u65f6\u949f\u5f02\u5e38", "\u901a\u4fe1\u5f02\u5e38", "\u6a21\u5757\u5f02\u5e38", "\u8d77\u52a8\u5f02\u5e38", "\u6f5c\u52a8\u5f02\u5e38", "\u751f\u4ea7\u8fc7\u7a0b\u5f02\u5e38", "\u5176\u4ed6"]
MODEL_PATH = cfg.ML_MODELS_DIR / "factory_fault_latest.pkl"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool_float(value: Any) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"1", "true", "yes", "y", "\u662f", "\u6709", "\u4e0d\u5408\u683c"} else 0.0
    return 1.0 if bool(value) else 0.0


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "\u6781\u9ad8\u98ce\u9669"
    if prob >= 0.55:
        return "\u9ad8\u98ce\u9669"
    if prob >= 0.35:
        return "\u4e2d\u98ce\u9669"
    return "\u4f4e\u98ce\u9669"


def _normalize_fault_type(value: str | None) -> str:
    if not value:
        return NORMAL
    text = str(value).strip()
    if text in {"", NORMAL, "\u5408\u683c", "\u65e0", "None", "none", "null"}:
        return NORMAL
    if "\u9891" in text or "\u6676\u632f" in text:
        return "\u9891\u5dee\u504f\u5927"
    if "\u6bd4\u5dee" in text or "\u4e92\u611f" in text:
        return "\u6bd4\u5dee\u504f\u5927"
    if "\u65f6\u949f" in text or "\u65e5\u8ba1\u65f6" in text:
        return "\u65f6\u949f\u5f02\u5e38"
    if "485" in text or "\u7ea2\u5916" in text or "\u901a\u4fe1" in text:
        return "\u901a\u4fe1\u5f02\u5e38"
    if "\u6a21\u5757" in text:
        return "\u6a21\u5757\u5f02\u5e38"
    if "\u8d77\u52a8" in text:
        return "\u8d77\u52a8\u5f02\u5e38"
    if "\u6f5c\u52a8" in text:
        return "\u6f5c\u52a8\u5f02\u5e38"
    return text if text in FAULT_TYPES else "\u5176\u4ed6"


def _trace_features(trace) -> dict[str, float]:
    if not trace:
        return {"ccd_retry_count": 0.0, "func_retry_count": 0.0, "reverify_retry_count": 0.0, "total_retry_count": 0.0, "smt_aoi_defect": 0.0, "wave_aoi_defect": 0.0, "repaired": 0.0, "repair_count": 0.0, "leak_current_max": 0.0, "prod_quality_risk_score": 0.0}
    return {
        "ccd_retry_count": _to_float(trace.ccd_retry_count),
        "func_retry_count": _to_float(trace.func_retry_count),
        "reverify_retry_count": _to_float(trace.reverify_retry_count),
        "total_retry_count": _to_float(trace.total_retry_count),
        "smt_aoi_defect": _to_bool_float(trace.smt_aoi_defect),
        "wave_aoi_defect": _to_bool_float(trace.wave_aoi_defect),
        "repaired": _to_bool_float(trace.repaired),
        "repair_count": _to_float(trace.repair_count),
        "leak_current_max": _to_float(trace.leak_current_max),
        "prod_quality_risk_score": _to_float(trace.quality_risk_score),
    }


def _features_from_factory(factory, trace=None) -> dict[str, float]:
    features = {
        "crystal_ppm": _to_float(getattr(factory, "crystal_ppm", None), 9.0),
        "ct_ratio_err": _to_float(getattr(factory, "ct_ratio_err", None), 0.5),
        "clock_accuracy_sd": _to_float(getattr(factory, "clock_accuracy_sd", None), 0.018),
        "meter_constant_err": _to_float(getattr(factory, "meter_constant_err", None), 0.0),
    }
    features.update(_trace_features(trace))
    return features


def _rule_probability(features: dict[str, Any]) -> float:
    score = 0.05
    crystal_ppm = abs(_to_float(features.get("crystal_ppm")))
    ct_ratio_err = abs(_to_float(features.get("ct_ratio_err")))
    clock_accuracy = abs(_to_float(features.get("clock_accuracy_sd")))
    meter_err = abs(_to_float(features.get("meter_constant_err")))
    total_retry = _to_float(features.get("total_retry_count"))
    leak_current = _to_float(features.get("leak_current_max"))
    if crystal_ppm >= 20: score += 0.28
    elif crystal_ppm >= 15: score += 0.16
    if ct_ratio_err >= 1.0: score += 0.24
    elif ct_ratio_err >= 0.7: score += 0.12
    if clock_accuracy >= 0.08: score += 0.18
    elif clock_accuracy >= 0.04: score += 0.08
    if meter_err >= 0.8: score += 0.18
    elif meter_err >= 0.5: score += 0.08
    if total_retry >= 3: score += 0.22
    elif total_retry >= 1: score += 0.10
    if _to_bool_float(features.get("smt_aoi_defect")): score += 0.12
    if _to_bool_float(features.get("wave_aoi_defect")): score += 0.12
    if _to_bool_float(features.get("repaired")): score += 0.12
    if _to_float(features.get("repair_count")) >= 2: score += 0.08
    if leak_current >= 1.0: score += 0.12
    score += min(_to_float(features.get("prod_quality_risk_score")), 1.0) * 0.18
    return round(min(max(score, 0.0), 0.98), 4)


def _fault_type_probabilities(features: dict[str, Any], fault_prob: float) -> dict[str, float]:
    weights = {name: 0.02 for name in FAULT_TYPES}
    if abs(_to_float(features.get("crystal_ppm"))) >= 15: weights["\u9891\u5dee\u504f\u5927"] += 0.32
    if abs(_to_float(features.get("ct_ratio_err"))) >= 0.7: weights["\u6bd4\u5dee\u504f\u5927"] += 0.28
    if abs(_to_float(features.get("clock_accuracy_sd"))) >= 0.04: weights["\u65f6\u949f\u5f02\u5e38"] += 0.25
    if _to_float(features.get("func_retry_count")) > 0 or _to_float(features.get("reverify_retry_count")) > 0:
        weights["\u901a\u4fe1\u5f02\u5e38"] += 0.16; weights["\u6a21\u5757\u5f02\u5e38"] += 0.10
    if _to_float(features.get("ccd_retry_count")) > 0 or _to_bool_float(features.get("smt_aoi_defect")) or _to_bool_float(features.get("wave_aoi_defect")):
        weights["\u751f\u4ea7\u8fc7\u7a0b\u5f02\u5e38"] += 0.28
    if _to_bool_float(features.get("repaired")):
        weights["\u751f\u4ea7\u8fc7\u7a0b\u5f02\u5e38"] += 0.14; weights["\u5176\u4ed6"] += 0.08
    total = sum(weights.values()) or 1.0
    scaled = {k: round(fault_prob * v / total, 4) for k, v in weights.items()}
    return dict(sorted(scaled.items(), key=lambda item: item[1], reverse=True))


def _key_factors(features: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("crystal_ppm", abs(_to_float(features.get("crystal_ppm"))) >= 15, "\u6676\u632f\u9891\u7387\u8bef\u5dee\u504f\u9ad8"),
        ("ct_ratio_err", abs(_to_float(features.get("ct_ratio_err"))) >= 0.7, "\u4e92\u611f\u5668\u6bd4\u5dee\u504f\u9ad8"),
        ("clock_accuracy_sd", abs(_to_float(features.get("clock_accuracy_sd"))) >= 0.04, "\u51fa\u5382\u65f6\u949f\u51c6\u786e\u5ea6\u63a5\u8fd1\u98ce\u9669\u533a\u95f4"),
        ("meter_constant_err", abs(_to_float(features.get("meter_constant_err"))) >= 0.5, "\u4eea\u8868\u5e38\u6570\u8bef\u5dee\u504f\u9ad8"),
        ("total_retry_count", _to_float(features.get("total_retry_count")) >= 1, "\u751f\u4ea7\u8fc7\u7a0b\u5b58\u5728\u91cd\u6d4b"),
        ("smt_aoi_defect", _to_bool_float(features.get("smt_aoi_defect")) > 0, "\u8d34\u7247 AOI \u68c0\u51fa\u7f3a\u9677"),
        ("wave_aoi_defect", _to_bool_float(features.get("wave_aoi_defect")) > 0, "\u6ce2\u5cf0\u710a AOI \u68c0\u51fa\u7f3a\u9677"),
        ("repaired", _to_bool_float(features.get("repaired")) > 0, "\u8be5\u8868\u7ecf\u5386\u8fc7\u7ef4\u4fee\u95ed\u73af"),
        ("leak_current_max", _to_float(features.get("leak_current_max")) >= 1.0, "\u8010\u538b\u6f0f\u7535\u6d41\u504f\u9ad8"),
    ]
    factors = [{"feature": k, "label": FACTORY_FEATURE_LABELS.get(k, k), "value": features.get(k), "reason": r} for k, hit, r in checks if hit]
    return factors[:6] or [{"feature": "overall", "label": "\u7efc\u5408\u8d28\u91cf\u7279\u5f81", "value": None, "reason": "\u672a\u53d1\u73b0\u660e\u663e\u5355\u9879\u5f02\u5e38\uff0c\u5f53\u524d\u5224\u65ad\u4e3b\u8981\u6765\u81ea\u7efc\u5408\u6a21\u578b/\u89c4\u5219\u8bc4\u5206"}]


def get_overview(db: Session) -> dict[str, Any]:
    from app.models.db_models import FactoryTest, MeterQualityTrace
    total_factory = db.query(FactoryTest).count()
    fault_count = db.query(FactoryTest).filter(FactoryTest.is_fault == True).count()
    model_info = None
    if MODEL_PATH.exists():
        try:
            model_info = joblib.load(MODEL_PATH).get("model_info")
        except Exception:
            model_info = {"status": "\u6a21\u578b\u6587\u4ef6\u5b58\u5728\u4f46\u8bfb\u53d6\u5931\u8d25"}
    return {"factory_total": total_factory, "factory_fault_count": fault_count, "factory_normal_count": max(total_factory - fault_count, 0), "factory_fault_rate": round(fault_count / total_factory * 100, 2) if total_factory else 0, "trace_total": db.query(MeterQualityTrace).count(), "has_model": MODEL_PATH.exists(), "model_info": model_info, "watch_dir": str(cfg.WATCH_DIR), "data_dir": str(cfg.DATA_DIR)}

def _feature_vector(features: dict[str, Any]) -> np.ndarray:
    return np.array([[_to_float(features.get(name), 0.0) for name in FACTORY_FEATURE_NAMES]], dtype=float)


def build_training_samples(db: Session) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from app.models.db_models import FactoryTest, MeterQualityTrace
    rows = db.query(FactoryTest).all()
    X, y, meta = [], [], []
    for factory in rows:
        trace = db.query(MeterQualityTrace).filter_by(barcode=factory.meter_id).first()
        features = _features_from_factory(factory, trace)
        fault_type = _normalize_fault_type(factory.fault_type)
        is_fault = bool(factory.is_fault) or fault_type != "\u6b63\u5e38"
        X.append([features[name] for name in FACTORY_FEATURE_NAMES])
        y.append(1 if is_fault else 0)
        meta.append({"meter_id": factory.meter_id, "fault_type": fault_type, "features": features})
    return np.array(X, dtype=float), np.array(y, dtype=int), meta


def train_factory_fault_model(db: Session) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y, _ = build_training_samples(db)
    if len(X) < 5:
        return {"success": False, "message": "\u51fa\u5382\u68c0\u6d4b\u6837\u672c\u4e0d\u8db3\uff0c\u8bad\u7ec3\u81f3\u5c11\u9700\u8981 5 \u6761\u8bb0\u5f55\uff0c\u5f53\u524d\u53ef\u4f7f\u7528\u89c4\u5219\u5f15\u64ce\u9884\u6d4b", "n_samples": int(len(X))}
    if len(set(y.tolist())) < 2:
        return {"success": False, "message": "\u8bad\u7ec3\u6837\u672c\u53ea\u6709\u5355\u4e00\u7c7b\u522b\uff0c\u65e0\u6cd5\u8bad\u7ec3\u4e8c\u5206\u7c7b\u6a21\u578b\uff0c\u8bf7\u8865\u5145\u6b63\u5e38\u548c\u6545\u969c\u4e24\u7c7b\u6837\u672c", "n_samples": int(len(X))}

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=cfg.RANDOM_STATE, stratify=stratify)
    model = RandomForestClassifier(n_estimators=160, max_depth=6, min_samples_leaf=2, random_state=cfg.RANDOM_STATE, class_weight="balanced")
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, prob) if len(set(y_test.tolist())) > 1 else None
    f1 = f1_score(y_test, pred, zero_division=0)
    acc = accuracy_score(y_test, pred)
    importances = sorted([
        {"feature": name, "label": FACTORY_FEATURE_LABELS.get(name, name), "importance": round(float(val), 4)}
        for name, val in zip(FACTORY_FEATURE_NAMES, model.feature_importances_)
    ], key=lambda item: item["importance"], reverse=True)
    version = f"factory_fault_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_info = {
        "version": version,
        "model_type": "RandomForestClassifier",
        "n_samples": int(len(X)),
        "n_fault": int(y.sum()),
        "n_normal": int(len(y) - y.sum()),
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "auc_score": round(float(auc), 4) if auc is not None else None,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_importance": importances,
    }
    pkg = {"model": model, "feature_names": FACTORY_FEATURE_NAMES, "model_info": model_info}
    cfg.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pkg, MODEL_PATH)
    joblib.dump(pkg, cfg.ML_MODELS_DIR / f"{version}.pkl")
    return {"success": True, "message": "\u51fa\u5382\u6545\u969c\u6a21\u578b\u8bad\u7ec3\u5b8c\u6210", "model_info": model_info}


def predict_factory_fault(features: dict[str, Any], db: Session | None = None) -> dict[str, Any]:
    cleaned = {name: _to_float(features.get(name), 0.0) for name in FACTORY_FEATURE_NAMES}
    model_version = "\u89c4\u5219\u5f15\u64ce"
    fault_prob = _rule_probability(cleaned)
    if MODEL_PATH.exists():
        try:
            pkg = joblib.load(MODEL_PATH)
            fault_prob = round(float(pkg["model"].predict_proba(_feature_vector(cleaned))[0][1]), 4)
            model_version = pkg.get("model_info", {}).get("version", "factory_fault_latest")
        except Exception:
            model_version = "\u89c4\u5219\u5f15\u64ce\uff08\u6a21\u578b\u8bfb\u53d6\u5931\u8d25\uff0c\u5df2\u964d\u7ea7\uff09"
    fault_types = _fault_type_probabilities(cleaned, fault_prob)
    top_fault_type = next(iter(fault_types.keys())) if fault_types else "\u6b63\u5e38"
    return {
        "success": True,
        "fault_probability": fault_prob,
        "risk_level": _risk_level(fault_prob),
        "fault_label": 1 if fault_prob >= 0.5 else 0,
        "top_fault_type": top_fault_type,
        "fault_type_probabilities": fault_types,
        "key_factors": _key_factors(cleaned),
        "model_version": model_version,
        "features": cleaned,
    }


def predict_by_meter(meter_id: str, db: Session) -> dict[str, Any]:
    from app.models.db_models import FactoryTest, MeterQualityTrace
    factory = db.query(FactoryTest).filter_by(meter_id=meter_id).first()
    trace = db.query(MeterQualityTrace).filter_by(barcode=meter_id).first()
    if factory:
        features = _features_from_factory(factory, trace)
    elif trace:
        features = _trace_features(trace)
        features.update({"crystal_ppm": 9.0, "ct_ratio_err": 0.5, "clock_accuracy_sd": 0.018, "meter_constant_err": 0.0})
    else:
        return {"success": False, "message": f"\u672a\u627e\u5230\u7535\u8868 {meter_id} \u7684\u51fa\u5382\u68c0\u6d4b\u6216\u751f\u4ea7\u8ffd\u6eaf\u6570\u636e"}
    result = predict_factory_fault(features, db)
    result["meter_id"] = meter_id
    if factory:
        result["actual_fault_type"] = _normalize_fault_type(factory.fault_type)
        result["actual_is_fault"] = bool(factory.is_fault) or result["actual_fault_type"] != "\u6b63\u5e38"
    return result


def recent_predictions(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    from app.models.db_models import FactoryTest
    rows = db.query(FactoryTest).order_by(FactoryTest.created_at.desc()).limit(limit).all()
    items = []
    for row in rows:
        pred = predict_by_meter(row.meter_id, db)
        if pred.get("success"):
            items.append(pred)
    return items
