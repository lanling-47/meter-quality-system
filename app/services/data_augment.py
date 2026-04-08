"""
数据增强服务
基于真实31块电表的分布参数，生成合成样本扩充训练集
- 特征分布从真实数据中拟合（均值/标准差）
- 时序特征用物理退化模型（线性 + 高斯噪声）
- 质量标签用与真实完全一致的规则生成
- 每条增强数据带 is_augmented=True 标记
"""
import json
import random
import math
from typing import Optional
from sqlalchemy.orm import Session

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import config as cfg


# ─── 真实电表观测到的特征分布参数（从31块电表统计）─────────────────────
# 这些参数通过对真实数据统计分析确定，保证增强数据物理可信
REAL_DISTRIBUTIONS = {
    "crystal_ppm": {"mean": 9.2, "std": 3.5, "min": 0.5, "max": 22.0},
    "ct_ratio_err": {"mean": 0.82, "std": 0.45, "min": 0.05, "max": 2.5},
    "clock_accuracy_sd": {"mean": 0.018, "std": 0.008, "min": 0.005, "max": 0.05},
    "initial_error_max": {"mean": 0.08, "std": 0.06, "min": 0.001, "max": 0.25},
    # 退化速率：误差增加速率 %/100h
    "error_degradation_rate": {"mean": 0.045, "std": 0.025, "min": 0.0, "max": 0.12},
    # 时钟漂移：s/d
    "clock_drift_rate": {"mean": 0.18, "std": 0.08, "min": 0.01, "max": 0.5},
    # 日误差最大值 s/d
    "clock_daily_error_max": {"mean": 0.22, "std": 0.07, "min": 0.01, "max": 0.40},
    # 误差变差最大值 %
    "error_variability_max": {"mean": 0.006, "std": 0.003, "min": 0.0001, "max": 0.015},
    # 误差一致性偏差最大值 %
    "error_consistency_max": {"mean": 0.086, "std": 0.035, "min": 0.01, "max": 0.18},
    # 正反向电量比（老化试验中基本为1.0，轻微波动）
    "energy_balance_ratio": {"mean": 0.9998, "std": 0.0003, "min": 0.999, "max": 1.001},
    # 尖电量占比（试验中约24.6%）
    "energy_peak_ratio": {"mean": 0.246, "std": 0.003, "min": 0.240, "max": 0.255},
}

# 高风险分布（用于生成约30%的高风险样本）
HIGH_RISK_DISTRIBUTIONS = {
    "crystal_ppm": {"mean": 15.0, "std": 4.0, "min": 10.0, "max": 22.0},
    "ct_ratio_err": {"mean": 1.4, "std": 0.4, "min": 0.8, "max": 2.5},
    "clock_accuracy_sd": {"mean": 0.035, "std": 0.010, "min": 0.02, "max": 0.06},
    "initial_error_max": {"mean": 0.12, "std": 0.08, "min": 0.05, "max": 0.35},
    "error_degradation_rate": {"mean": 0.075, "std": 0.030, "min": 0.04, "max": 0.15},
    "clock_drift_rate": {"mean": 0.32, "std": 0.10, "min": 0.2, "max": 0.6},
    "clock_daily_error_max": {"mean": 0.36, "std": 0.05, "min": 0.28, "max": 0.50},
    "error_variability_max": {"mean": 0.010, "std": 0.004, "min": 0.004, "max": 0.020},
    "error_consistency_max": {"mean": 0.130, "std": 0.030, "min": 0.10, "max": 0.20},
    "energy_balance_ratio": {"mean": 0.9995, "std": 0.0006, "min": 0.998, "max": 1.001},
    "energy_peak_ratio": {"mean": 0.246, "std": 0.004, "min": 0.238, "max": 0.256},
}

# 试验时长点（小时）
AGING_HOURS = [0.0, 264.0, 480.0, 672.0, 864.0, 1128.0]


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _normal_sample(dist: dict, rng: random.Random) -> float:
    """从正态分布采样，截断到[min, max]"""
    # Box-Muller方法
    u1 = rng.random()
    u2 = rng.random()
    z = math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)
    val = dist["mean"] + dist["std"] * z
    return _clamp(val, dist["min"], dist["max"])


def _generate_single_sample(sample_id: int, rng: random.Random, is_high_risk: bool) -> dict:
    """
    生成一条完整的增强电表记录（QualityChain 格式）
    包含所有14个特征 + 质量标签
    """
    dist = HIGH_RISK_DISTRIBUTIONS if is_high_risk else REAL_DISTRIBUTIONS

    # 基础来料特征
    crystal_ppm = _normal_sample(dist["crystal_ppm"], rng)
    ct_ratio_err = _normal_sample(dist["ct_ratio_err"], rng)
    clock_accuracy_sd = _normal_sample(dist["clock_accuracy_sd"], rng)

    # 老化初始误差
    initial_error_max = _normal_sample(dist["initial_error_max"], rng)

    # 退化速率
    error_deg_rate = _normal_sample(dist["error_degradation_rate"], rng)
    clock_drift = _normal_sample(dist["clock_drift_rate"], rng)

    # 末期误差（物理退化模型：线性增长 + 少量噪声）
    final_hours = 1128.0
    noise = rng.gauss(0, 0.02)
    final_error_max = _clamp(initial_error_max + error_deg_rate * final_hours / 100.0 + noise, 0.01, 2.0)

    # 时钟特征
    clock_drift_rate = clock_drift
    initial_clock = 0
    final_clock = int(_clamp(clock_drift * (final_hours / 24.0) + rng.gauss(0, 0.5), -30, 30))

    # RS485 异常（高风险时概率更高）
    rs485_prob = 0.15 if is_high_risk else 0.03
    rs485_total = rng.randint(1, 15) if rng.random() < rs485_prob else 0

    # 新增特征
    clock_daily_max = _normal_sample(dist["clock_daily_error_max"], rng)
    variability_max = _normal_sample(dist["error_variability_max"], rng)
    consistency_max = _normal_sample(dist["error_consistency_max"], rng)
    energy_balance = _normal_sample(dist["energy_balance_ratio"], rng)
    energy_peak = _normal_sample(dist["energy_peak_ratio"], rng)

    # 是否出厂故障（高风险略高概率）
    factory_fault = rng.random() < (0.12 if is_high_risk else 0.04)

    # 质量标签（与真实规则完全一致）
    label = 0
    reasons = []
    if final_error_max > cfg.ERROR_THRESHOLD:
        label = 1
        reasons.append(f"末期误差{final_error_max:.3f}%超差")
    if abs(final_clock) > cfg.CLOCK_THRESHOLD:
        label = 1
        reasons.append(f"时钟偏差{final_clock}s超限")
    if rs485_total > 0:
        label = 1
        reasons.append(f"RS485异常{rs485_total}次")
    if clock_drift_rate > 0.35:
        label = 1
        reasons.append(f"时钟漂移{clock_drift_rate:.3f}s/d偏高")
    if abs(clock_daily_max) > getattr(cfg, 'CLOCK_DAILY_THRESHOLD', 0.3):
        label = 1
        reasons.append(f"日误差{clock_daily_max:.3f}s/d超限")
    if consistency_max > getattr(cfg, 'CONSISTENCY_THRESHOLD', 0.12):
        label = 1
        reasons.append(f"一致性偏差{consistency_max:.3f}%偏大")

    return {
        "meter_no": 10000 + sample_id,  # 虚拟电表编号（10000+）
        "crystal_batch": f"AUG_BATCH_{(sample_id // 10) + 1:03d}",
        "ct_batch": f"AUG_CT_{(sample_id // 10) + 1:03d}",
        "crystal_ppm": round(crystal_ppm, 4),
        "ct_ratio_err": round(ct_ratio_err, 4),
        "clock_accuracy_sd": round(clock_accuracy_sd, 4),
        "factory_fault": factory_fault,
        "factory_fault_type": "频差偏大" if factory_fault and crystal_ppm > 15 else ("比差偏大" if factory_fault else None),
        "initial_error_max": round(initial_error_max, 4),
        "final_error_max": round(final_error_max, 4),
        "error_degradation_rate": round(error_deg_rate, 6),
        "initial_clock_offset": initial_clock,
        "final_clock_offset": final_clock,
        "clock_drift_rate": round(clock_drift_rate, 4),
        "rs485_total_errors": rs485_total,
        "clock_daily_error_max": round(clock_daily_max, 4),
        "error_variability_max": round(variability_max, 6),
        "error_consistency_max": round(consistency_max, 4),
        "cover_open_count": 0,
        "energy_balance_ratio": round(energy_balance, 6),
        "energy_peak_ratio": round(energy_peak, 4),
        "is_augmented": True,
        "quality_label": label,
        "quality_reason": "；".join(reasons) if reasons else "正常（增强）",
    }


def generate_augmented_data(n: int = 200, seed: int = 42) -> list[dict]:
    """
    生成 n 条增强样本，目标正负比约 3:7（30%高风险）
    返回 QualityChain 格式的 dict list
    """
    rng = random.Random(seed)
    samples = []

    # 前30%强制为高风险方向，后70%为正常方向（允许规则自然生成标签）
    n_high_risk_seed = int(n * 0.30)

    for i in range(n):
        # 高风险种子用于保证足够的正类样本
        force_high_risk = i < n_high_risk_seed
        sample = _generate_single_sample(i + 1, rng, is_high_risk=force_high_risk)
        samples.append(sample)

    return samples


def apply_augmented_data(db: Session, n: int = 200, seed: int = 42) -> dict:
    """
    将增强数据写入数据库 quality_chain 表（幂等操作）
    清除旧增强数据后重新生成
    """
    from app.models.db_models import QualityChain

    # 幂等：先删旧增强数据
    deleted = db.query(QualityChain).filter(QualityChain.is_augmented == True).delete()
    db.commit()

    samples = generate_augmented_data(n=n, seed=seed)
    added = 0
    for s in samples:
        db.add(QualityChain(**s))
        added += 1
    db.commit()

    # 统计
    total = db.query(QualityChain).count()
    real_count = db.query(QualityChain).filter(QualityChain.is_augmented == False).count()
    aug_count = db.query(QualityChain).filter(QualityChain.is_augmented == True).count()
    high_risk = db.query(QualityChain).filter(QualityChain.quality_label == 1).count()
    high_risk_real = db.query(QualityChain).filter(
        QualityChain.quality_label == 1, QualityChain.is_augmented == False).count()
    high_risk_aug = db.query(QualityChain).filter(
        QualityChain.quality_label == 1, QualityChain.is_augmented == True).count()

    return {
        "deleted_old": deleted,
        "added": added,
        "total": total,
        "real_count": real_count,
        "aug_count": aug_count,
        "high_risk_total": high_risk,
        "high_risk_real": high_risk_real,
        "high_risk_aug": high_risk_aug,
        "normal_total": total - high_risk,
    }
