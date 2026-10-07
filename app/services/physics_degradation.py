"""Compact physics priors and consistency checks for accelerated aging data.

The physical layer deliberately predicts a device health proxy rather than a
universal physical parameter.  Capacitor, IGBT and MOSFET responses have
different units, so the shared quantity is normalized damage stage ``D`` in
[0, 1].  Device-specific equations remain interpretable and can be replaced
when measured material parameters become available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr


BOLTZMANN_EV_PER_K = 8.617333262e-5


@dataclass(frozen=True)
class PhysicsParameters:
    activation_energy_ev: float = 0.35
    stress_exponent: float = 2.0
    rate_scale: float = 1.0
    thermal_exponent: float = 1.5
    response_scale: float = 1.0


DEFAULT_PHYSICS_PARAMETERS: Mapping[str, PhysicsParameters] = {
    "capacitor": PhysicsParameters(activation_energy_ev=0.30, stress_exponent=2.0, thermal_exponent=1.0),
    "igbt": PhysicsParameters(activation_energy_ev=0.45, stress_exponent=1.5, thermal_exponent=1.6),
    "mosfet": PhysicsParameters(activation_energy_ev=0.40, stress_exponent=1.5, thermal_exponent=1.8),
}


def arrhenius_rate(temperature_kelvin, activation_energy_ev: float) -> np.ndarray:
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    temperature = np.clip(temperature, 200.0, 700.0)
    return np.exp(-activation_energy_ev / (BOLTZMANN_EV_PER_K * temperature))


def capacitor_damage_rate(temperature_kelvin, voltage_ratio, parameters=None) -> np.ndarray:
    """Electrical overstress damage rate: Arrhenius x voltage acceleration."""
    p = parameters or DEFAULT_PHYSICS_PARAMETERS["capacitor"]
    return p.rate_scale * arrhenius_rate(temperature_kelvin, p.activation_energy_ev) * np.power(
        np.clip(np.asarray(voltage_ratio, dtype=np.float64), 0.0, 4.0), p.stress_exponent
    )


def semiconductor_damage_rate(temperature_kelvin, delta_temperature, stress_ratio, device_type, parameters=None) -> np.ndarray:
    """Thermal-cycle damage prior for IGBT/MOSFET power semiconductors."""
    if device_type not in {"igbt", "mosfet"}:
        raise ValueError("device_type must be igbt or mosfet")
    p = parameters or DEFAULT_PHYSICS_PARAMETERS[device_type]
    return p.rate_scale * arrhenius_rate(temperature_kelvin, p.activation_energy_ev) * np.power(
        np.clip(np.abs(np.asarray(delta_temperature, dtype=np.float64)), 1e-4, 500.0),
        p.thermal_exponent,
    ) * np.power(np.clip(np.asarray(stress_ratio, dtype=np.float64), 0.0, 4.0), p.stress_exponent)


def integrate_damage(rate: Iterable[float], elapsed: Iterable[float]) -> np.ndarray:
    rate = np.nan_to_num(np.asarray(rate, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    elapsed = np.nan_to_num(np.asarray(elapsed, dtype=np.float64), nan=0.0)
    if rate.size == 0:
        return rate
    increments = np.maximum(rate, 0.0) * np.maximum(np.diff(np.r_[0.0, elapsed]), 0.0)
    cumulative = np.cumsum(increments)
    scale = max(float(cumulative[-1]), 1e-12)
    return np.clip(cumulative / scale, 0.0, 1.0)


def physical_damage_stage(device_type: str, stresses: np.ndarray, times: np.ndarray, parameters=None) -> np.ndarray:
    """Return normalized damage stage from stress history only."""
    stresses = np.asarray(stresses, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if stresses.ndim != 2 or stresses.shape[0] != len(times):
        raise ValueError("stresses must have shape [T, S] and match times")
    if device_type == "capacitor":
        voltage_ratio = np.clip(stresses[:, 0], 0.0, 4.0)
        # The public capacitor transient records expose voltage level but not
        # junction temperature; use ambient temperature plus a stress proxy.
        temperature = 298.15 + 40.0 * voltage_ratio
        rate = capacitor_damage_rate(temperature, voltage_ratio, parameters)
    elif device_type in {"igbt", "mosfet"}:
        temperature = 273.15 + np.clip(stresses[:, 0] * 300.0, -50.0, 500.0)
        delta_temperature = np.maximum(stresses[:, 0] * 300.0 - 25.0, 1.0)
        rate = semiconductor_damage_rate(temperature, delta_temperature, stresses[:, 1], device_type, parameters)
    else:
        raise ValueError(f"unsupported device type: {device_type}")
    elapsed = np.maximum(times - times[0], 0.0)
    return integrate_damage(rate, elapsed)


def response_damage_proxy(device_type: str, responses: np.ndarray) -> np.ndarray:
    """Convert heterogeneous responses into a monotone health proxy candidate."""
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError("responses must have shape [T, R]")
    if device_type == "capacitor":
        indices = (2, 4) if responses.shape[1] >= 5 else (0,)
    elif device_type == "igbt":
        indices = (2,) if responses.shape[1] >= 3 else (0,)
    elif device_type == "mosfet":
        indices = (0,)
    else:
        raise ValueError(f"unsupported device type: {device_type}")
    selected = responses[:, indices]
    baseline = np.nanmedian(selected[: max(1, len(selected) // 10)], axis=0)
    scale = np.nanmedian(np.abs(selected - baseline), axis=0)
    scale = np.maximum(scale, 1e-6)
    change = np.nanmean((selected - baseline) / scale, axis=1)
    # A clipped positive response change is an observable health proxy, not a label.
    return np.clip((change - np.nanmin(change)) / max(np.nanmax(change) - np.nanmin(change), 1e-6), 0.0, 1.0)


def physics_consistency_metrics(dataset) -> dict:
    """Evaluate stress-only physics prior against observed response proxies."""
    rows = []
    for item, group in zip(dataset.series, dataset.groups):
        observed = response_damage_proxy(dataset.device_type, item.responses)
        predicted = physical_damage_stage(dataset.device_type, item.stresses, item.times)
        if len(observed) < 2:
            continue
        rho = spearmanr(observed, predicted).statistic
        rows.append({
            "group": group,
            "sample_id": item.sample_id,
            "proxy_final": float(observed[-1]),
            "physics_final": float(predicted[-1]),
            "proxy_physics_mae": float(np.mean(np.abs(observed - predicted))),
            "spearman": float(rho) if np.isfinite(rho) else float("nan"),
            "physics_monotonicity": float(np.mean(np.diff(predicted) < 0.0)),
        })
    if not rows:
        raise ValueError(f"no valid physics rows for {dataset.device_type}")
    valid_rho = [row["spearman"] for row in rows if np.isfinite(row["spearman"])]
    return {
        "device_type": dataset.device_type,
        "groups": len(rows),
        "mean_proxy_physics_mae": float(np.mean([row["proxy_physics_mae"] for row in rows])),
        "mean_spearman": float(np.mean(valid_rho)) if valid_rho else float("nan"),
        "mean_monotonicity_violation": float(np.mean([row["physics_monotonicity"] for row in rows])),
        "rows": rows,
    }
