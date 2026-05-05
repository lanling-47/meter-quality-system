"""
消融实验服务
方向C 核心：系统评估生产过程特征对质量预测的贡献度

实验设计:
  特征组 A - 来料特征（4维）
  特征组 B - 来料 + 老化试验（14维）
  特征组 C - 来料 + 老化 + 生产过程（全特征，24维）
  特征组 D - 仅生产过程特征（10维）

对比算法（5种）:
  1. Rule-based  - 规则引擎基线
  2. RandomForest - 传统集成学习
  3. XGBoost     - 梯度提升（工业标准）
  4. LightGBM    - 原方法
  5. TabNet      - 本文方法（前沿深度学习）

评估指标（5个，5折交叉验证 mean ± std）:
  AUC-ROC / F1(高风险) / Precision / Recall / Accuracy
"""
import os
import sys
import json
import numpy as np
import time
from typing import Optional, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg

from app.services.predict_service import FEATURE_NAMES, FEATURE_LABELS

# ─── 特征组定义 ───────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "A": {
        "name": "Group-A: 来料特征",
        "desc": "晶振/互感器/时钟/出厂故障（4维）",
        "features": [
            "crystal_ppm", "ct_ratio_err", "clock_accuracy_sd", "factory_fault",
        ],
        "color": "#3498db",
    },
    "B": {
        "name": "Group-B: 来料 + 老化试验",
        "desc": "来料4维 + 老化10维 = 14维",
        "features": [
            "crystal_ppm", "ct_ratio_err", "clock_accuracy_sd", "factory_fault",
            "initial_error_max", "final_error_max", "error_degradation_rate",
            "clock_drift_rate", "clock_daily_error_max", "error_variability_max",
            "error_consistency_max", "energy_balance_ratio", "energy_peak_ratio",
            "rs485_total_errors",
        ],
        "color": "#27ae60",
    },
    "C": {
        "name": "Group-C: 全特征（本文方法）",
        "desc": "来料4维 + 老化10维 + 生产过程10维 = 24维",
        "features": FEATURE_NAMES,  # 全部24维
        "color": "#e74c3c",
    },
    "D": {
        "name": "Group-D: 仅生产过程特征",
        "desc": "生产过程10维（验证生产特征的独立贡献）",
        "features": [
            "prod_ccd_retry", "prod_func_retry", "prod_reverify_retry", "prod_total_retry",
            "prod_smt_aoi_defect", "prod_wave_aoi_defect", "prod_repaired", "prod_repair_count",
            "prod_leak_current_max", "prod_quality_risk_score",
        ],
        "color": "#f0a500",
    },
}

# ─── 算法注册表 ────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "rule_based": {
        "name": "规则引擎",
        "name_en": "Rule-based",
        "color": "#95a5a6",
        "is_baseline": True,
    },
    "random_forest": {
        "name": "随机森林",
        "name_en": "Random Forest",
        "color": "#3498db",
        "is_baseline": False,
    },
    "xgboost": {
        "name": "XGBoost",
        "name_en": "XGBoost",
        "color": "#27ae60",
        "is_baseline": False,
    },
    "lightgbm": {
        "name": "LightGBM",
        "name_en": "LightGBM",
        "color": "#f0a500",
        "is_baseline": False,
    },
    "tabnet": {
        "name": "TabNet（本文）",
        "name_en": "TabNet (Ours)",
        "color": "#e74c3c",
        "is_baseline": False,
    },
}

# 结果缓存路径
ABLATION_RESULTS_PATH = cfg.BASE_DIR / "data" / "ablation_results.json"


def _get_full_feature_matrix(db):
    """从数据库获取全量特征矩阵（24维），与 predict_service 一致"""
    from app.models.db_models import QualityChain
    chains = db.query(QualityChain).all()
    if not chains:
        return None, None

    X_rows, y = [], []
    for c in chains:
        row = [
            c.crystal_ppm or 9.0, c.ct_ratio_err or 0.5,
            c.clock_accuracy_sd or 0.018, 1.0 if c.factory_fault else 0.0,
            c.initial_error_max or 0.0, c.final_error_max or 0.0,
            c.error_degradation_rate or 0.0, c.clock_drift_rate or 0.0,
            c.clock_daily_error_max if c.clock_daily_error_max is not None else 0.22,
            c.error_variability_max if c.error_variability_max is not None else 0.006,
            c.error_consistency_max if c.error_consistency_max is not None else 0.086,
            c.energy_balance_ratio if c.energy_balance_ratio is not None else 1.0,
            c.energy_peak_ratio if c.energy_peak_ratio is not None else 0.246,
            float(c.rs485_total_errors or 0),
            # 生产特征（当前默认0）
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        X_rows.append(row)
        y.append(c.quality_label)

    return np.array(X_rows), np.array(y)


def _select_features(X_full: np.ndarray, group_key: str) -> np.ndarray:
    """根据特征组key，选取对应列的子集"""
    selected_names = FEATURE_GROUPS[group_key]["features"]
    indices = [FEATURE_NAMES.index(f) for f in selected_names if f in FEATURE_NAMES]
    return X_full[:, indices]


# ─── 算法训练函数 ──────────────────────────────────────────────────────

def _train_rule_based(X, y):
    """规则引擎：基于晶振误差+互感器+出厂故障的简单阈值"""
    class RuleModel:
        def predict_proba(self, X):
            probs = []
            for row in X:
                score = 0.0
                if len(row) > 0 and row[0] > 15: score += 0.3  # crystal_ppm
                if len(row) > 1 and row[1] > 0.8: score += 0.3  # ct_ratio_err
                if len(row) > 3 and row[3] > 0: score += 0.4   # factory_fault
                prob = min(score, 1.0)
                probs.append([1 - prob, prob])
            return np.array(probs)
    return RuleModel(), None


def _train_random_forest(X, y, scaler=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42,
                                class_weight="balanced")
    m.fit(X_s, y)
    return m, sc


def _train_xgboost(X, y, scaler=None):
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    pos_weight = max(1, (len(y) - int(y.sum())) / max(1, int(y.sum())))
    m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=pos_weight, random_state=42,
                            eval_metric="logloss", verbosity=0, use_label_encoder=False)
    m.fit(X_s, y)
    return m, sc


def _train_lightgbm(X, y, scaler=None):
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    pos_weight = max(1, (len(y) - int(y.sum())) / max(1, int(y.sum())))
    m = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                             scale_pos_weight=pos_weight, random_state=42, verbose=-1)
    m.fit(X_s, y)
    return m, sc


def _train_tabnet_for_ablation(X, y):
    """TabNet（消融版，无SSE推送，适合批量实验）"""
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.preprocessing import MinMaxScaler
    from app.services.tabnet_service import TABNET_PARAMS, TABNET_FIT_PARAMS

    sc = MinMaxScaler()
    X_s = sc.fit_transform(X)

    m = TabNetClassifier(**TABNET_PARAMS)
    m.fit(X_s, y.astype(int), **{
        **TABNET_FIT_PARAMS,
        "max_epochs": 150,  # 消融实验用适中轮次
        "patience": 20,
    })
    return m, sc


TRAIN_FNS = {
    "rule_based": _train_rule_based,
    "random_forest": _train_random_forest,
    "xgboost": _train_xgboost,
    "lightgbm": _train_lightgbm,
    "tabnet": _train_tabnet_for_ablation,
}


def _evaluate_model(model, scaler, X, y) -> dict:
    """在测试集上评估单个模型"""
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    )
    X_eval = scaler.transform(X) if scaler is not None else X
    prob = model.predict_proba(X_eval)[:, 1]
    pred = (prob >= 0.5).astype(int)

    if len(set(y)) < 2:
        return {k: 0.0 for k in ["auc", "f1", "precision", "recall", "accuracy"]}

    return {
        "auc": float(roc_auc_score(y, prob)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def _cv_evaluate(X, y, model_key: str, n_splits=5) -> dict:
    """
    K折交叉验证评估

    Returns
    -------
    {metric: {mean, std, values}}
    """
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {k: [] for k in ["auc", "f1", "precision", "recall", "accuracy"]}

    train_fn = TRAIN_FNS[model_key]

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if len(set(y_tr)) < 2 or len(set(y_val)) < 2:
            continue

        # SMOTE（针对 ML 模型）
        if model_key not in ("rule_based",):
            n_pos = int(y_tr.sum())
            if n_pos > 1 and n_pos < len(y_tr) - 1:
                try:
                    k = max(1, min(n_pos - 1, 3))
                    sm = SMOTE(random_state=42, k_neighbors=k)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except Exception:
                    pass

        try:
            model, scaler = train_fn(X_tr, y_tr)
            m = _evaluate_model(model, scaler, X_val, y_val)
            for k, v in m.items():
                fold_metrics[k].append(v)
        except Exception as e:
            continue

    result = {}
    for k, vals in fold_metrics.items():
        if vals:
            result[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "values": [round(float(v), 4) for v in vals],
            }
        else:
            result[k] = {"mean": 0.0, "std": 0.0, "values": []}
    return result


def run_ablation_experiment(db, emit_fn: Optional[Callable] = None) -> dict:
    """
    运行全部消融实验（4特征组 × 5算法 = 20次实验）

    Parameters
    ----------
    db       : SQLAlchemy Session
    emit_fn  : SSE 进度推送函数（可选）

    Returns
    -------
    完整实验结果字典，含 20 组 CV 指标
    """
    def emit(event, data):
        if emit_fn:
            emit_fn(event, data)

    emit("ablation_start", {"total": 20, "message": "开始消融实验（4特征组 × 5算法 = 20组）..."})

    # 加载数据
    X_full, y = _get_full_feature_matrix(db)
    if X_full is None or len(X_full) < 10:
        return {"error": "训练数据不足"}

    emit("ablation_progress", {"message": f"数据加载完成: {len(X_full)} 样本, {int(y.sum())} 高风险"})

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(X_full)),
        "n_positive": int(y.sum()),
        "n_splits": 5,
        "feature_groups": {k: {
            "name": v["name"],
            "desc": v["desc"],
            "n_features": len(v["features"]),
            "color": v["color"],
        } for k, v in FEATURE_GROUPS.items()},
        "models": {k: {
            "name": v["name"],
            "name_en": v["name_en"],
            "color": v["color"],
        } for k, v in MODEL_REGISTRY.items()},
        "experiments": {},  # {group_key: {model_key: {metrics}}}
        "errors": [],
    }

    exp_count = 0
    total_exp = len(FEATURE_GROUPS) * len(MODEL_REGISTRY)

    for g_key, g_info in FEATURE_GROUPS.items():
        results["experiments"][g_key] = {}
        X_group = _select_features(X_full, g_key)

        for m_key in MODEL_REGISTRY:
            exp_count += 1
            exp_name = f"{g_info['name']} × {MODEL_REGISTRY[m_key]['name']}"
            emit("ablation_progress", {
                "count": exp_count,
                "total": total_exp,
                "message": f"[{exp_count}/{total_exp}] {exp_name}...",
                "group": g_key,
                "model": m_key,
                "progress_pct": int(exp_count / total_exp * 100),
            })

            try:
                t0 = time.time()
                cv_result = _cv_evaluate(X_group, y, m_key, n_splits=5)
                elapsed = round(time.time() - t0, 2)

                results["experiments"][g_key][m_key] = {
                    "cv_metrics": cv_result,
                    "elapsed_sec": elapsed,
                }

                emit("ablation_exp_done", {
                    "group": g_key,
                    "model": m_key,
                    "auc_mean": cv_result.get("auc", {}).get("mean", 0),
                    "f1_mean": cv_result.get("f1", {}).get("mean", 0),
                    "elapsed": elapsed,
                    "message": f"完成: AUC={cv_result.get('auc',{}).get('mean',0):.4f} ± {cv_result.get('auc',{}).get('std',0):.4f}",
                })

            except Exception as e:
                err_msg = f"{exp_name} 失败: {str(e)}"
                results["errors"].append(err_msg)
                results["experiments"][g_key][m_key] = {"error": str(e)}
                emit("ablation_error", {"message": err_msg})

    # 计算关键洞察
    results["insights"] = _compute_insights(results)

    # 保存缓存
    try:
        ABLATION_RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(ABLATION_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    emit("ablation_done", {
        "message": f"消融实验完成！共 {exp_count} 组实验",
        "n_errors": len(results["errors"]),
    })

    return results


def _compute_insights(results: dict) -> dict:
    """计算关键研究洞察"""
    insights = {}
    exps = results.get("experiments", {})

    # 1. 生产特征的增量贡献（Group-C vs Group-B）
    b_tabnet_auc = _get_metric(exps, "B", "tabnet", "auc")
    c_tabnet_auc = _get_metric(exps, "C", "tabnet", "auc")
    if b_tabnet_auc and c_tabnet_auc:
        insights["production_feature_gain"] = {
            "group_b_auc": b_tabnet_auc,
            "group_c_auc": c_tabnet_auc,
            "gain": round(c_tabnet_auc - b_tabnet_auc, 4),
            "description": "生产过程特征（Group-B→C）对 TabNet AUC 的提升量",
        }

    # 2. TabNet vs LightGBM 对比（Group-C，全特征）
    lgbm_c_auc = _get_metric(exps, "C", "lightgbm", "auc")
    tabnet_c_auc = _get_metric(exps, "C", "tabnet", "auc")
    if lgbm_c_auc and tabnet_c_auc:
        insights["tabnet_vs_lgbm"] = {
            "lightgbm_auc": lgbm_c_auc,
            "tabnet_auc": tabnet_c_auc,
            "gain": round(tabnet_c_auc - lgbm_c_auc, 4),
            "description": "全特征集下 TabNet vs LightGBM AUC 差值",
        }

    # 3. 各特征组下 TabNet 最优表现
    group_best = {}
    for g_key in ["A", "B", "C", "D"]:
        auc = _get_metric(exps, g_key, "tabnet", "auc")
        f1 = _get_metric(exps, g_key, "tabnet", "f1")
        if auc is not None:
            group_best[g_key] = {"auc": auc, "f1": f1}
    insights["tabnet_by_group"] = group_best

    # 4. 最优模型（Group-C 下 AUC 最高）
    best_model = None
    best_auc = 0.0
    for m_key in MODEL_REGISTRY:
        auc = _get_metric(exps, "C", m_key, "auc")
        if auc and auc > best_auc:
            best_auc = auc
            best_model = m_key
    if best_model:
        insights["best_model_on_full_features"] = {
            "model": best_model,
            "model_name": MODEL_REGISTRY[best_model]["name"],
            "auc": best_auc,
        }

    return insights


def _get_metric(exps: dict, group: str, model: str, metric: str) -> Optional[float]:
    try:
        return exps[group][model]["cv_metrics"][metric]["mean"]
    except (KeyError, TypeError):
        return None


def load_ablation_results() -> Optional[dict]:
    """加载缓存的消融实验结果"""
    if ABLATION_RESULTS_PATH.exists():
        try:
            with open(ABLATION_RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
