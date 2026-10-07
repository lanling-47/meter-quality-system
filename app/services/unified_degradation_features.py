"""Device-agnostic degradation feature extraction.

The feature layer is deliberately independent from PyTorch so it can be used by
the existing FastAPI service, offline dataset converters, and the deep model.
Device physics is retained through ``degradation_direction`` and optional
thresholds; only the representation of the resulting degradation evidence is
shared across devices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class ResponseSpec:
    """Metadata that maps one physical response to common degradation space.

    ``degradation_direction`` is ``1`` when a larger value means worse health
    and ``-1`` when a smaller value means worse health.
    """

    name: str
    degradation_direction: int = 1
    warning_threshold: float | None = None

    def __post_init__(self):
        if self.degradation_direction not in (-1, 1):
            raise ValueError("degradation_direction must be 1 or -1")


@dataclass(frozen=True)
class DegradationFeatureSet:
    response_name: str
    baseline: float
    final_relative_drift: float
    max_relative_drift: float
    normalized_slope: float
    volatility: float
    max_step_change: float
    degradation_auc: float
    monotonicity: float
    threshold_exceedance_ratio: float
    observation_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validated_series(times: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("times and values must have the same length")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        raise ValueError("series contains no finite observations")
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def normalize_degradation(
    values: Sequence[float],
    degradation_direction: int = 1,
    baseline: float | None = None,
) -> np.ndarray:
    """Convert a physical response into positive-is-worse relative drift."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if degradation_direction not in (-1, 1):
        raise ValueError("degradation_direction must be 1 or -1")
    base = float(y[0] if baseline is None else baseline)
    scale = max(abs(base), float(np.nanmedian(np.abs(y))), EPS)
    return degradation_direction * (y - base) / scale


def extract_degradation_features(
    times: Sequence[float],
    values: Sequence[float],
    spec: ResponseSpec,
) -> DegradationFeatureSet:
    """Extract comparable evidence from one device response trajectory."""
    x, y = _validated_series(times, values)
    baseline = float(y[0])
    degradation = normalize_degradation(y, spec.degradation_direction, baseline)

    if x.size >= 2 and float(np.ptp(x)) > EPS:
        normalized_time = (x - x[0]) / float(np.ptp(x))
        slope = float(np.polyfit(normalized_time, degradation, 1)[0])
        diffs = np.diff(degradation)
        volatility = float(np.std(diffs))
        max_step = float(np.max(np.abs(diffs)))
        informative = np.abs(diffs) > EPS
        if np.any(informative):
            monotonicity = float(np.mean(diffs[informative] >= 0.0))
        else:
            monotonicity = 1.0
        auc = float(np.trapezoid(np.maximum(degradation, 0.0), normalized_time))
    else:
        slope = volatility = max_step = auc = 0.0
        monotonicity = 1.0

    if spec.warning_threshold is None:
        exceedance = 0.0
    elif spec.degradation_direction == 1:
        exceedance = float(np.mean(y >= spec.warning_threshold))
    else:
        exceedance = float(np.mean(y <= spec.warning_threshold))

    return DegradationFeatureSet(
        response_name=spec.name,
        baseline=baseline,
        final_relative_drift=float(degradation[-1]),
        max_relative_drift=float(np.max(degradation)),
        normalized_slope=slope,
        volatility=volatility,
        max_step_change=max_step,
        degradation_auc=auc,
        monotonicity=monotonicity,
        threshold_exceedance_ratio=exceedance,
        observation_count=int(x.size),
    )


def aggregate_degradation_evidence(features: Iterable[DegradationFeatureSet]) -> dict:
    """Aggregate multiple physical responses into one common evidence vector."""
    items = list(features)
    if not items:
        raise ValueError("at least one feature set is required")
    drift = np.asarray([item.final_relative_drift for item in items], dtype=float)
    slopes = np.asarray([item.normalized_slope for item in items], dtype=float)
    volatility = np.asarray([item.volatility for item in items], dtype=float)
    steps = np.asarray([item.max_step_change for item in items], dtype=float)
    auc = np.asarray([item.degradation_auc for item in items], dtype=float)
    monotonicity = np.asarray([item.monotonicity for item in items], dtype=float)
    exceedance = np.asarray([item.threshold_exceedance_ratio for item in items], dtype=float)

    # The score is descriptive evidence, not a calibrated failure probability.
    positive_drift = np.maximum(drift, 0.0)
    health_risk_index = float(np.clip(
        0.35 * np.mean(positive_drift)
        + 0.20 * np.mean(np.maximum(slopes, 0.0))
        + 0.15 * np.mean(volatility)
        + 0.10 * np.mean(steps)
        + 0.10 * np.mean(auc)
        + 0.10 * np.mean(exceedance),
        0.0,
        1.0,
    ))
    return {
        "health_risk_index": health_risk_index,
        "mean_final_relative_drift": float(np.mean(drift)),
        "max_final_relative_drift": float(np.max(drift)),
        "mean_positive_slope": float(np.mean(np.maximum(slopes, 0.0))),
        "mean_volatility": float(np.mean(volatility)),
        "max_step_change": float(np.max(steps)),
        "mean_degradation_auc": float(np.mean(auc)),
        "mean_monotonicity": float(np.mean(monotonicity)),
        "mean_threshold_exceedance_ratio": float(np.mean(exceedance)),
        "response_count": len(items),
    }


def build_common_feature_vector(features: Iterable[DegradationFeatureSet]) -> np.ndarray:
    """Return the fixed-width vector used by classical baselines."""
    evidence = aggregate_degradation_evidence(features)
    names = [
        "mean_final_relative_drift",
        "max_final_relative_drift",
        "mean_positive_slope",
        "mean_volatility",
        "max_step_change",
        "mean_degradation_auc",
        "mean_monotonicity",
        "mean_threshold_exceedance_ratio",
    ]
    return np.asarray([evidence[name] for name in names], dtype=np.float32)
