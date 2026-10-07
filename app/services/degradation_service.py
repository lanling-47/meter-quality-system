"""
退化曲线分析服务（移除 RUL，保留退化趋势分析）
"""
import sys
import os
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg
from app.services.unified_degradation_features import (
    ResponseSpec,
    aggregate_degradation_evidence,
    extract_degradation_features,
)


def get_meter_degradation(meter_no: int, db) -> dict:
    """获取电表退化曲线数据（不含 RUL 预测）"""
    from app.models.db_models import AgingTest
    records = db.query(AgingTest).filter_by(meter_no=meter_no).order_by(AgingTest.test_hours).all()

    if not records:
        return {"error": f"未找到电表 {meter_no} 的老化试验数据"}

    time_points = [r.test_date_label for r in records]
    hours = [r.test_hours for r in records]
    error_series = [r.error_max if r.error_max is not None else 0.0 for r in records]
    clock_series = [r.clock_offset_s if r.clock_offset_s is not None else 0 for r in records]

    # 线性趋势拟合（仅用于可视化辅助线）
    trend = _fit_trend(hours, error_series)
    clock_trend = _fit_trend(hours, [abs(c) for c in clock_series])

    # 退化模式判断
    trend_type = _detect_trend_type(error_series)

    error_features = extract_degradation_features(
        hours,
        error_series,
        ResponseSpec(
            name="basic_error",
            degradation_direction=1,
            warning_threshold=cfg.ERROR_THRESHOLD,
        ),
    )
    clock_abs_series = [abs(value) for value in clock_series]
    clock_features = extract_degradation_features(
        hours,
        clock_abs_series,
        ResponseSpec(
            name="clock_offset_abs",
            degradation_direction=1,
            warning_threshold=cfg.CLOCK_THRESHOLD,
        ),
    )
    unified_features = [error_features, clock_features]
    unified_evidence = aggregate_degradation_evidence(unified_features)

    return {
        "meter_no": meter_no,
        "time_points": time_points,
        "hours": hours,
        "error_series": error_series,
        "clock_series": clock_series,
        "trend": trend,
        "clock_trend": clock_trend,
        "trend_type": trend_type,
        "unified_degradation": {
            "responses": [item.to_dict() for item in unified_features],
            "evidence": unified_evidence,
            "note": "health_risk_index 为统一退化描述指标，不等同于校准后的失效概率",
        },
        "threshold_error": cfg.ERROR_THRESHOLD,
        "threshold_clock": cfg.CLOCK_THRESHOLD,
        "stats": {
            "initial_error": round(error_series[0], 4) if error_series else None,
            "final_error": round(error_series[-1], 4) if error_series else None,
            "max_error": round(max(error_series), 4) if error_series else None,
            "error_change": round(error_series[-1] - error_series[0], 4) if len(error_series) >= 2 else None,
            "slope": trend.get("slope"),
            "r2": trend.get("r2"),
            "final_clock": clock_series[-1] if clock_series else None,
            "max_clock": max([abs(c) for c in clock_series], default=0),
        },
        "chart_data": {
            "xAxis": time_points,
            "error_line": error_series,
            "clock_line": clock_series,
            "threshold_error": [cfg.ERROR_THRESHOLD] * len(time_points),
            "regression_line": trend.get("fitted_points", []),
        }
    }


def _fit_trend(hours: list, values: list) -> dict:
    """线性趋势拟合（仅用于辅助线）"""
    if len(hours) < 2:
        return {"slope": None, "r2": None, "fitted_points": []}
    x = np.array(hours, dtype=float)
    y = np.array(values, dtype=float)
    try:
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        y_pred = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        fitted_points = [round(float(np.polyval(coeffs, h)), 4) for h in x]
        return {
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r2": round(r2, 4),
            "fitted_points": fitted_points,
        }
    except Exception:
        return {"slope": None, "r2": None, "fitted_points": []}


def _detect_trend_type(values: list) -> str:
    """判断退化模式"""
    if len(values) < 3:
        return "数据不足"
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    if std_diff > abs(mean_diff) * 2:
        return "波动型（随机漂移）"
    if mean_diff > 0.01:
        return "渐进劣化型（线性增大）"
    if mean_diff < -0.01:
        return "改善型（误差减小）"
    return "稳定型（低速退化）"


def get_all_meters_degradation_summary(db) -> list:
    """获取所有电表的退化摘要"""
    from app.models.db_models import AgingTest
    import collections

    records = db.query(AgingTest).all()
    meter_groups = collections.defaultdict(list)
    for r in records:
        meter_groups[r.meter_no].append(r)

    summaries = []
    for meter_no, recs in sorted(meter_groups.items()):
        recs.sort(key=lambda x: x.test_hours)
        initial_err = recs[0].error_max if recs else None
        final_err = recs[-1].error_max if recs else None
        final_clock = recs[-1].clock_offset_s if recs else None
        errors = [r.error_max or 0.0 for r in recs]
        hours = [r.test_hours or 0.0 for r in recs]
        clocks = [abs(r.clock_offset_s or 0.0) for r in recs]

        trend = _detect_trend_type(errors)
        evidence = aggregate_degradation_evidence([
            extract_degradation_features(
                hours,
                errors,
                ResponseSpec("basic_error", 1, cfg.ERROR_THRESHOLD),
            ),
            extract_degradation_features(
                hours,
                clocks,
                ResponseSpec("clock_offset_abs", 1, cfg.CLOCK_THRESHOLD),
            ),
        ])

        from app.models.db_models import QualityChain
        chain = db.query(QualityChain).filter_by(meter_no=meter_no).first()
        risk_label = chain.quality_label if chain else 0

        summaries.append({
            "meter_no": meter_no,
            "initial_error_max": round(initial_err, 4) if initial_err else None,
            "final_error_max": round(final_err, 4) if final_err else None,
            "final_clock_offset": final_clock,
            "trend_type": trend,
            "health_risk_index": round(evidence["health_risk_index"], 4),
            "risk_label": risk_label,
            "n_records": len(recs),
        })
    return summaries
