"""
SHAP 根因追溯服务
基于 SHAP TreeExplainer 实现可解释性分析，输出追溯报告
"""
import json
import sys
import os
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.predict_service import FEATURE_NAMES, FEATURE_LABELS, load_active_model


def _align_feature_columns(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == len(FEATURE_NAMES):
        return x
    if x.shape[1] > len(FEATURE_NAMES):
        return x[:, :len(FEATURE_NAMES)]
    pad = np.zeros((x.shape[0], len(FEATURE_NAMES) - x.shape[1]))
    return np.hstack([x, pad])


def _snapshot_feature_row(db, meter_no: int) -> Optional[np.ndarray]:
    try:
        from app.services.data_linkage_service import (
            build_quality_feature_snapshots,
            get_snapshot_feature_matrix,
        )
        build_quality_feature_snapshots(db, include_synthetic_demo=False)
        X, _, meta = get_snapshot_feature_matrix(db)
        for idx, item in enumerate(meta):
            if item.get("meter_no") == meter_no:
                return X[idx:idx + 1]
    except Exception:
        return None
    return None


def explain_meter(meter_no: int, db) -> dict:
    """
    对指定电表编号生成 SHAP 解释
    返回：
      - feature_contributions: [{name, value, shap, label}]
      - waterfall_data: ECharts 瀑布图数据
      - root_cause_text: 文字追溯报告
      - base_value: 基准预测值
    """
    from app.models.db_models import QualityChain
    chain = db.query(QualityChain).filter_by(meter_no=meter_no).first()
    if not chain:
        return {"error": f"未找到电表 {meter_no} 的质量链路数据"}

    model, feat_names, version, scaler, model_type = load_active_model(db)

    # 构建兜底特征向量；优先使用快照矩阵中的46维特征。
    x = np.array([[
        chain.crystal_ppm or 9.0,
        chain.ct_ratio_err or 0.5,
        chain.clock_accuracy_sd or 0.03,
        float(chain.factory_fault or False),
        chain.initial_error_max or 0.0,
        chain.final_error_max or 0.0,
        chain.error_degradation_rate or 0.0,
        chain.clock_drift_rate or 0.0,
        chain.clock_daily_error_max if chain.clock_daily_error_max is not None else 0.22,
        chain.error_variability_max if chain.error_variability_max is not None else 0.006,
        chain.error_consistency_max if chain.error_consistency_max is not None else 0.086,
        chain.energy_balance_ratio if chain.energy_balance_ratio is not None else 1.0,
        chain.energy_peak_ratio if chain.energy_peak_ratio is not None else 0.246,
        float(chain.rs485_total_errors or 0),
        # 生产过程特征（默认0）
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ]])
    snapshot_x = _snapshot_feature_row(db, meter_no)
    x = snapshot_x if snapshot_x is not None else _align_feature_columns(x)

    # 当前特征值字典
    feat_vals = dict(zip(FEATURE_NAMES, x[0].tolist()))

    # 尝试用 SHAP（若无模型，用规则归因）
    if model is None:
        return _rule_based_explain(feat_vals, chain)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        # LightGBM 二分类：shap_values 可能是列表
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # 正类的 SHAP 值
        else:
            sv = shap_values[0]

        base_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

        contributions = []
        for i, fname in enumerate(FEATURE_NAMES):
            contributions.append({
                "name": fname,
                "label": FEATURE_LABELS.get(fname, fname),
                "value": float(feat_vals[fname]),
                "shap": float(sv[i]),
            })
        # 按 SHAP 绝对值降序
        contributions.sort(key=lambda d: abs(d["shap"]), reverse=True)

        waterfall = _build_waterfall_data(contributions, base_val)
        root_cause_text = _generate_root_cause_text(contributions, chain, model)

        return {
            "meter_no": meter_no,
            "risk_prob": float(model.predict_proba(x)[0][1]),
            "quality_label": chain.quality_label,
            "base_value": base_val,
            "contributions": contributions,
            "waterfall_data": waterfall,
            "root_cause_text": root_cause_text,
            "model_version": version,
        }
    except Exception as e:
        return _rule_based_explain(feat_vals, chain, error=str(e))


def _build_waterfall_data(contributions: list, base_val: float) -> dict:
    """构建 ECharts 瀑布图数据"""
    cumulative = base_val
    waterfall_items = []
    for c in contributions[:6]:  # 取前6个最重要特征
        shap_v = c["shap"]
        start = cumulative
        cumulative += shap_v
        waterfall_items.append({
            "name": c["label"],
            "shap": round(shap_v, 4),
            "start": round(start, 4),
            "end": round(cumulative, 4),
            "direction": "up" if shap_v > 0 else "down",
        })
    return {
        "base_value": round(base_val, 4),
        "final_value": round(cumulative, 4),
        "items": waterfall_items,
    }


def _generate_root_cause_text(contributions: list, chain, model=None) -> str:
    """自动生成文字追溯报告"""
    lines = []
    top3 = [c for c in contributions[:3] if abs(c["shap"]) > 0.01]

    risk_str = "高风险" if chain.quality_label == 1 else "低风险"
    lines.append(f"📊 **电表 {chain.meter_no} 质量风险分析报告**")
    lines.append(f"风险等级：{risk_str} | 标注原因：{chain.quality_reason}")
    lines.append("")
    lines.append("**主要风险因子（按贡献度降序）：**")

    for i, c in enumerate(top3, 1):
        direction = "↑ 增大风险" if c["shap"] > 0 else "↓ 降低风险"
        lines.append(f"{i}. {c['label']}（当前值：{c['value']:.2f}）— SHAP贡献 {c['shap']:+.4f} {direction}")

    # 具体的追溯建议
    lines.append("")
    lines.append("**追溯建议：**")
    for c in contributions[:3]:
        if abs(c["shap"]) < 0.01:
            continue
        fname = c["name"]
        val = c["value"]
        if fname == "crystal_ppm" and val > 15:
            lines.append(f"• 晶振频率误差 {val:.1f}ppm 接近限值(20ppm)，建议核查批次来料检测报告")
        elif fname == "ct_ratio_err" and val > 0.8:
            lines.append(f"• 互感器比差 {val:.2f}% 偏大，建议复检互感器批次")
        elif fname == "rs485_total_errors" and val > 0:
            lines.append(f"• RS485通信累计 {int(val)} 次异常，建议检查通信模块焊接质量或电路板设计")
        elif fname == "clock_drift_rate" and val > 0.3:
            lines.append(f"• 时钟漂移速率 {val:.3f}s/d 偏高，与晶振频偏存在强相关，建议从来料把控")
        elif fname == "final_error_max" and val > 0.3:
            lines.append(f"• 末期最大基本误差 {val:.3f}% 异常，建议分析退化曲线判断失效模式")
        elif fname == "prod_repaired" and val > 0:
            lines.append(f"• 该电表经过维修，建议结合维修记录分析故障根因")
        elif fname == "prod_func_retry" and val > 0:
            lines.append(f"• 功能测试重测 {int(val)} 次，说明生产过程中存在不稳定因素")
        elif fname == "prod_reverify_retry" and val > 0:
            lines.append(f"• 复校重测 {int(val)} 次，计量性能存在波动，建议核查校准设备状态")
        elif fname == "prod_smt_aoi_defect" and val > 0:
            lines.append(f"• 贴片AOI发现焊接缺陷，可能影响长期可靠性，建议追查缺陷类型")
        elif fname == "prod_wave_aoi_defect" and val > 0:
            lines.append(f"• 波峰焊AOI发现缺陷（如气孔/缺焊），建议检查波峰焊工艺参数")

    return "\n".join(lines)


def _rule_based_explain(feat_vals: dict, chain, error: str = None) -> dict:
    """无模型时的规则解释"""
    contributions = []
    rule_weights = {
        "crystal_ppm": lambda v: (v - 9.0) / 11.0 * 0.3 if v > 9 else 0,
        "ct_ratio_err": lambda v: (v - 0.5) / 0.5 * 0.25 if v > 0.5 else 0,
        "factory_fault": lambda v: 0.4 if v else 0,
        "rs485_total_errors": lambda v: min(v / 10, 0.3),
        "clock_drift_rate": lambda v: (v - 0.2) / 0.3 * 0.2 if v > 0.2 else 0,
        "final_error_max": lambda v: (v - 0.3) / 0.2 * 0.25 if v > 0.3 else 0,
        "prod_repaired": lambda v: 0.2 if v > 0 else 0,
        "prod_total_retry": lambda v: min(v * 0.05, 0.2),
        "prod_smt_aoi_defect": lambda v: 0.15 if v > 0 else 0,
        "prod_wave_aoi_defect": lambda v: 0.1 if v > 0 else 0,
    }
    for fname in FEATURE_NAMES:
        val = feat_vals.get(fname, 0)
        weight_fn = rule_weights.get(fname, lambda v: 0)
        try:
            shap = weight_fn(val)
        except Exception:
            shap = 0.0
        contributions.append({
            "name": fname,
            "label": FEATURE_LABELS.get(fname, fname),
            "value": float(val),
            "shap": float(shap),
        })
    contributions.sort(key=lambda d: abs(d["shap"]), reverse=True)
    root_text = _generate_root_cause_text(contributions, chain)
    if error:
        root_text = f"[使用规则引擎，原因：{error}]\n\n" + root_text

    return {
        "meter_no": chain.meter_no,
        "quality_label": chain.quality_label,
        "base_value": 0.2,
        "contributions": contributions,
        "waterfall_data": _build_waterfall_data(contributions, 0.2),
        "root_cause_text": root_text,
        "model_version": "规则引擎",
    }


def get_all_shap_summary(db) -> dict:
    """获取全体电表的全局特征重要度"""
    from app.models.db_models import QualityChain, ModelVersion
    model, feat_names, version, scaler, model_type = load_active_model(db)
    if model is None:
        return {"error": "无激活模型"}

    from app.models.db_models import QualityChain
    chains = db.query(QualityChain).all()
    if not chains:
        return {"error": "无数据"}

    import numpy as np
    X = np.array([[
        c.crystal_ppm or 9.0,
        c.ct_ratio_err or 0.5,
        c.clock_accuracy_sd or 0.03,
        float(c.factory_fault or False),
        c.initial_error_max or 0.0,
        c.final_error_max or 0.0,
        c.error_degradation_rate or 0.0,
        c.clock_drift_rate or 0.0,
        c.clock_daily_error_max if c.clock_daily_error_max is not None else 0.22,
        c.error_variability_max if c.error_variability_max is not None else 0.006,
        c.error_consistency_max if c.error_consistency_max is not None else 0.086,
        c.energy_balance_ratio if c.energy_balance_ratio is not None else 1.0,
        c.energy_peak_ratio if c.energy_peak_ratio is not None else 0.246,
        float(c.rs485_total_errors or 0),
        # 生产过程特征（默认0）
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ] for c in chains])
    try:
        from app.services.data_linkage_service import (
            build_quality_feature_snapshots,
            get_snapshot_feature_matrix,
        )
        build_quality_feature_snapshots(db, include_synthetic_demo=False)
        snapshot_X, _, _ = get_snapshot_feature_matrix(db)
        if len(snapshot_X) > 0:
            X = snapshot_X
    except Exception:
        pass
    X = _align_feature_columns(X)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals

        importance = np.abs(sv).mean(axis=0).tolist()
        feature_importance = [
            {"name": FEATURE_NAMES[i], "label": FEATURE_LABELS[FEATURE_NAMES[i]], "importance": round(importance[i], 4)}
            for i in range(len(FEATURE_NAMES))
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        return {"feature_importance": feature_importance, "model_version": version}
    except Exception as e:
        # 使用模型自带特征重要度
        try:
            fi = model.feature_importances_
            feature_importance = [
                {"name": FEATURE_NAMES[i], "label": FEATURE_LABELS[FEATURE_NAMES[i]], "importance": round(float(fi[i]), 4)}
                for i in range(len(FEATURE_NAMES))
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            return {"feature_importance": feature_importance, "model_version": version}
        except Exception:
            return {"error": str(e)}
