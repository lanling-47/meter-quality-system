"""
用电异常检测服务（升级版）
整合正反向电量累计、日计时、误差变差数据进行多维异常检测
检测维度：
1. 基本误差异常（3σ + 阈值双重判断）
2. 时钟漂移异常（日误差时序趋势）
3. 电量比例异常（正反向比 / 尖峰平谷结构）
4. 通信异常（RS485）
5. 计量稳定性异常（误差变差超限）
"""
import sys, os, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg


# ─── 异常检测阈值 ────────────────────────────────────────────────────
THRESHOLDS = {
    "error_max": cfg.ERROR_THRESHOLD,        # 基本误差阈值 %
    "clock_s": cfg.CLOCK_THRESHOLD,          # 时钟偏差阈值 s
    "clock_daily_sd": getattr(cfg, "CLOCK_DAILY_THRESHOLD", 0.3),  # 日误差 s/d
    "consistency": getattr(cfg, "CONSISTENCY_THRESHOLD", 0.12),    # 一致性偏差 %
    "variability": 0.012,                    # 误差变差阈值 %
    "energy_balance_min": 0.995,             # 正反向电量比下限（接近1为纯正向）
    "energy_peak_ratio_min": 0.235,          # 尖电量占比合理范围
    "energy_peak_ratio_max": 0.260,
}


def _zscore_anomaly(values: list, meter_nos: list, threshold_sigma: float = 2.5) -> set:
    """用 Z-score 找统计异常的电表编号"""
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) < 3:
        return set()
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return set()
    anomaly_set = set()
    for mn, v in zip(meter_nos, values):
        if v is not None and abs(v - mean) > threshold_sigma * std:
            anomaly_set.add(mn)
    return anomaly_set


def get_power_data(db) -> dict:
    """
    多维用电异常检测
    整合基本误差、日计时、正反向电量、误差变差、RS485 数据
    """
    from app.models.db_models import AgingTest, QualityChain

    records = db.query(AgingTest).order_by(AgingTest.meter_no, AgingTest.test_hours).all()
    if not records:
        return {"meters": [], "anomaly_meters": [], "stats": {}, "chart_data": {}}

    # ── 按电表分组，取末期记录 ──────────────────────────────────────
    meter_data = defaultdict(list)
    for r in records:
        meter_data[r.meter_no].append(r)

    meter_nos = sorted(meter_data.keys())
    meter_summaries = []

    # 提取各维度数据（用于 Z-score 计算）
    final_errors, daily_errs, variabilities, balance_ratios = [], [], [], []

    for mn in meter_nos:
        recs = sorted(meter_data[mn], key=lambda x: x.test_hours)
        final = recs[-1]
        chain = db.query(QualityChain).filter_by(meter_no=mn, is_augmented=False).first()

        final_errors.append(final.error_max)
        daily_errs.append(abs(final.clock_daily_error_sd) if final.clock_daily_error_sd is not None else None)
        variabilities.append(final.error_variability)
        balance_ratios.append(final.energy_balance_ratio)

    # Z-score 异常集合
    zscore_error = _zscore_anomaly(final_errors, meter_nos, 2.0)
    zscore_daily = _zscore_anomaly(daily_errs, meter_nos, 2.0)
    zscore_var = _zscore_anomaly(variabilities, meter_nos, 2.5)

    # 统计基准
    valid_errors = [v for v in final_errors if v is not None]
    mean_err = float(np.mean(valid_errors)) if valid_errors else 0.0
    std_err = float(np.std(valid_errors)) if valid_errors else 0.0
    threshold_3sigma = mean_err + 2.5 * std_err

    valid_daily = [v for v in daily_errs if v is not None]
    mean_daily = float(np.mean(valid_daily)) if valid_daily else 0.0

    # ── 逐表判断异常 ───────────────────────────────────────────────
    anomaly_meters = []

    for i, mn in enumerate(meter_nos):
        recs = sorted(meter_data[mn], key=lambda x: x.test_hours)
        final = recs[-1]
        chain = db.query(QualityChain).filter_by(meter_no=mn, is_augmented=False).first()

        anomaly_reasons = []
        anomaly_types = []   # 用于分类展示

        # ① 基本误差异常
        err = final.error_max or 0.0
        if err > THRESHOLDS["error_max"]:
            anomaly_reasons.append(f"末期误差{err:.3f}%超差(>{THRESHOLDS['error_max']}%)")
            anomaly_types.append("误差超差")
        elif mn in zscore_error:
            anomaly_reasons.append(f"误差统计异常(Z-score偏高，{err:.3f}%)")
            anomaly_types.append("误差统计异常")

        # ② 时钟异常（静态偏差 + 日漂移）
        clock = final.clock_offset_s or 0
        if abs(clock) > THRESHOLDS["clock_s"]:
            anomaly_reasons.append(f"时钟偏差{clock}s超限(>{THRESHOLDS['clock_s']}s)")
            anomaly_types.append("时钟超差")
        daily_err_val = abs(final.clock_daily_error_sd) if final.clock_daily_error_sd is not None else None
        if daily_err_val is not None and daily_err_val > THRESHOLDS["clock_daily_sd"]:
            anomaly_reasons.append(f"日误差{daily_err_val:.3f}s/d偏高(>{THRESHOLDS['clock_daily_sd']}s/d)")
            anomaly_types.append("日误差偏高")
        elif mn in zscore_daily:
            anomaly_reasons.append(f"日误差统计异常({daily_err_val:.3f}s/d)")
            anomaly_types.append("日误差统计异常")

        # ③ 正反向电量比异常（应接近1.0，纯正向用电）
        balance = final.energy_balance_ratio
        if balance is not None and balance < THRESHOLDS["energy_balance_min"]:
            anomaly_reasons.append(f"正反向电量比{balance:.4f}<{THRESHOLDS['energy_balance_min']}（有反向电量流入）")
            anomaly_types.append("电量比例异常")

        # ④ 尖峰平谷比例异常
        peak_r = final.energy_peak_ratio
        if peak_r is not None:
            if peak_r < THRESHOLDS["energy_peak_ratio_min"]:
                anomaly_reasons.append(f"尖电量占比{peak_r:.3f}<{THRESHOLDS['energy_peak_ratio_min']}（异常低）")
                anomaly_types.append("电量结构异常")
            elif peak_r > THRESHOLDS["energy_peak_ratio_max"]:
                anomaly_reasons.append(f"尖电量占比{peak_r:.3f}>{THRESHOLDS['energy_peak_ratio_max']}（异常高）")
                anomaly_types.append("电量结构异常")

        # ⑤ 计量稳定性（误差变差）
        var = final.error_variability
        if var is not None and var > THRESHOLDS["variability"]:
            anomaly_reasons.append(f"误差变差{var:.4f}%偏大(>{THRESHOLDS['variability']}%)")
            anomaly_types.append("稳定性异常")
        elif mn in zscore_var:
            anomaly_reasons.append(f"误差变差统计异常({var:.4f}%)")
            anomaly_types.append("稳定性统计异常")

        # ⑥ 通信异常
        rs485 = final.rs485_errors or 0
        if rs485 > 0:
            anomaly_reasons.append(f"RS485通信异常{rs485}次")
            anomaly_types.append("通信异常")

        # 综合异常评分（0-100）
        anomaly_score = min(100, len(anomaly_reasons) * 20 + (
            20 if err > THRESHOLDS["error_max"] else 0
        ))

        entry = {
            "meter_no": mn,
            "final_error": round(err, 4),
            "final_clock": clock,
            "daily_error": round(daily_err_val, 4) if daily_err_val is not None else None,
            "energy_balance": round(balance, 4) if balance is not None else None,
            "energy_peak_ratio": round(peak_r, 4) if peak_r is not None else None,
            "error_variability": round(var, 5) if var is not None else None,
            "rs485_errors": rs485,
            "risk_label": chain.quality_label if chain else 0,
            "anomaly_type": list(set(anomaly_types)),
            "anomaly_reasons": anomaly_reasons,
            "is_anomaly": len(anomaly_reasons) > 0,
            "anomaly_score": anomaly_score,
        }
        meter_summaries.append(entry)
        if len(anomaly_reasons) > 0:
            anomaly_meters.append(entry)

    # ── 图表数据 ─────────────────────────────────────────────────────
    chart_data = {
        "meter_nos": [ms["meter_no"] for ms in meter_summaries],
        "errors": [ms["final_error"] for ms in meter_summaries],
        "daily_errors": [ms["daily_error"] for ms in meter_summaries],
        "balance_ratios": [ms["energy_balance"] for ms in meter_summaries],
        "peak_ratios": [ms["energy_peak_ratio"] for ms in meter_summaries],
        "variabilities": [ms["error_variability"] for ms in meter_summaries],
        "anomaly_scores": [ms["anomaly_score"] for ms in meter_summaries],
        "threshold_line": [THRESHOLDS["error_max"]] * len(meter_summaries),
        "threshold_3sigma": threshold_3sigma,
        "colors": ["#e74c3c" if ms["is_anomaly"] else "#2ecc71" for ms in meter_summaries],
    }

    # 异常类型分布统计
    type_counter = defaultdict(int)
    for ms in anomaly_meters:
        for t in ms["anomaly_type"]:
            type_counter[t] += 1

    return {
        "meters": meter_summaries,
        "anomaly_meters": sorted(anomaly_meters, key=lambda x: -x["anomaly_score"]),
        "stats": {
            "total": len(meter_summaries),
            "anomaly_count": len(anomaly_meters),
            "mean_error": round(mean_err, 4),
            "std_error": round(std_err, 4),
            "threshold_3sigma": round(threshold_3sigma, 4),
            "mean_daily_error": round(mean_daily, 4),
            "anomaly_rate": round(len(anomaly_meters) / max(len(meter_summaries), 1) * 100, 1),
            "type_distribution": dict(type_counter),
        },
        "chart_data": chart_data,
        "thresholds": THRESHOLDS,
    }


def get_power_trend(meter_no: int, db) -> dict:
    """
    获取单表多维用电异常趋势
    整合：基本误差时序 / 日计时时序 / 电量累计时序
    """
    from app.models.db_models import AgingTest

    records = db.query(AgingTest).filter_by(meter_no=meter_no).order_by(AgingTest.test_hours).all()
    if not records:
        return {"error": f"未找到电表 {meter_no}"}

    time_labels = [r.test_date_label for r in records]
    hours = [r.test_hours or 0 for r in records]
    errors = [r.error_max or 0.0 for r in records]
    clocks = [r.clock_offset_s or 0 for r in records]
    daily_errs = [abs(r.clock_daily_error_sd) if r.clock_daily_error_sd is not None else None for r in records]
    variabilities = [r.error_variability for r in records]
    energy_totals = [r.energy_total for r in records]
    balance_ratios = [r.energy_balance_ratio for r in records]
    peak_ratios = [r.energy_peak_ratio for r in records]

    # 计算电量增量（每时间段新增电量）
    energy_increments = [None]
    for i in range(1, len(energy_totals)):
        prev = energy_totals[i - 1]
        curr = energy_totals[i]
        if prev is not None and curr is not None and curr >= prev:
            energy_increments.append(round(curr - prev, 3))
        else:
            energy_increments.append(None)

    # 误差趋势分析（线性拟合）
    valid_pts = [(h, e) for h, e in zip(hours, errors) if e is not None]
    error_slope = None
    if len(valid_pts) >= 2:
        x = np.array([p[0] for p in valid_pts])
        y = np.array([p[1] for p in valid_pts])
        try:
            coeffs = np.polyfit(x, y, 1)
            error_slope = round(float(coeffs[0]) * 100, 6)  # %/100h
        except Exception:
            pass

    # 异常时间点识别
    anomaly_points = []
    for i, r in enumerate(records):
        reasons = []
        if errors[i] > THRESHOLDS["error_max"]:
            reasons.append(f"误差{errors[i]:.3f}%超差")
        if abs(clocks[i]) > THRESHOLDS["clock_s"]:
            reasons.append(f"时钟偏差{clocks[i]}s超限")
        if daily_errs[i] is not None and daily_errs[i] > THRESHOLDS["clock_daily_sd"]:
            reasons.append(f"日误差{daily_errs[i]:.3f}s/d偏高")
        if balance_ratios[i] is not None and balance_ratios[i] < THRESHOLDS["energy_balance_min"]:
            reasons.append(f"正反向电量比{balance_ratios[i]:.4f}异常")
        if reasons:
            anomaly_points.append({"time": time_labels[i], "hours": hours[i], "reasons": reasons})

    return {
        "meter_no": meter_no,
        "time_labels": time_labels,
        "hours": hours,
        # 误差时序
        "error_series": errors,
        "error_threshold": THRESHOLDS["error_max"],
        "error_slope": error_slope,
        # 时钟时序
        "clock_series": clocks,
        "daily_error_series": daily_errs,
        "clock_threshold": THRESHOLDS["clock_s"],
        # 电量时序（正反向电量累计）
        "energy_total_series": energy_totals,
        "energy_increment_series": energy_increments,
        "balance_ratio_series": balance_ratios,
        "peak_ratio_series": peak_ratios,
        # 误差稳定性
        "variability_series": variabilities,
        # 汇总
        "anomaly_points": anomaly_points,
        "summary": f"共 {len(anomaly_points)} 个时间点异常，误差退化斜率 {error_slope}%/100h" if anomaly_points else f"未发现异常，误差退化斜率 {error_slope}%/100h",
    }


def get_energy_comparison(db) -> dict:
    """
    获取全部电表正反向电量数据对比
    用于展示各时间点的电量结构（尖/峰/平/谷占比雷达图）
    """
    from app.models.db_models import AgingTest

    # 取最后一个时间点
    records = db.query(AgingTest).order_by(AgingTest.meter_no, AgingTest.test_hours).all()
    meter_final = {}
    for r in records:
        if r.energy_total is not None:
            meter_final[r.meter_no] = r

    if not meter_final:
        return {"data": [], "has_data": False}

    data = []
    for mn, r in sorted(meter_final.items()):
        if r.energy_total and r.energy_total > 0:
            data.append({
                "meter_no": mn,
                "energy_total": round(r.energy_total, 3),
                "energy_forward": round(r.energy_forward, 3) if r.energy_forward else None,
                "balance_ratio": round(r.energy_balance_ratio, 5) if r.energy_balance_ratio else None,
                "peak_ratio": round(r.energy_peak_ratio, 4) if r.energy_peak_ratio else None,
                "daily_error": round(abs(r.clock_daily_error_sd), 4) if r.clock_daily_error_sd else None,
            })

    return {"data": data, "has_data": True, "count": len(data)}
