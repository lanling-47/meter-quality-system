"""Load NASA accelerated-aging data into the common degradation contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import h5py
from scipy.io import loadmat

from app.services.unified_degradation_data import DegradationSeries


STRESS_DIM = 8


@dataclass(frozen=True)
class LoadedDataset:
    device_type: str
    response_names: tuple[str, ...]
    response_directions: tuple[int, ...]
    series: list[DegradationSeries]
    groups: list[str]
    source_files: list[str]


def _mat_items(value) -> list:
    if value is None:
        return []
    return list(np.atleast_1d(value).flat)


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _bin_median(rows: np.ndarray, max_points: int) -> np.ndarray:
    if len(rows) <= max_points:
        return rows
    edges = np.linspace(0, len(rows), max_points + 1, dtype=int)
    aggregated = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end > start:
            aggregated.append(np.nanmedian(rows[start:end], axis=0))
    return np.asarray(aggregated, dtype=np.float64)


def _prefix_series(
    device_type: str,
    group: str,
    times: np.ndarray,
    stresses: np.ndarray,
    responses: np.ndarray,
    fractions: Iterable[float],
    min_points: int = 8,
    output_points: int = 32,
) -> list[DegradationSeries]:
    output = []
    length = len(times)
    for fraction in fractions:
        end = max(min_points, int(round(length * fraction)))
        end = min(end, length)
        if end < min_points:
            continue
        prefix_times = times[:end]
        total_duration = float(times[-1] - times[0])
        target = (
            float((prefix_times[-1] - times[0]) / total_duration)
            if total_duration > 1e-8
            else float((end - 1) / max(1, length - 1))
        )
        prefix_stresses = stresses[:end]
        prefix_responses = responses[:end]
        if end != output_points:
            query = np.linspace(prefix_times[0], prefix_times[-1], output_points)
            prefix_stresses = np.column_stack([
                np.interp(query, prefix_times, prefix_stresses[:, column])
                for column in range(prefix_stresses.shape[1])
            ]).astype(np.float32)
            prefix_responses = np.column_stack([
                np.interp(query, prefix_times, prefix_responses[:, column])
                for column in range(prefix_responses.shape[1])
            ]).astype(np.float32)
            prefix_times = query.astype(np.float32)
        output.append(DegradationSeries(
            device_type=device_type,
            sample_id=f"{group}::prefix-{end:04d}",
            times=prefix_times,
            stresses=prefix_stresses,
            responses=prefix_responses,
            target=target,
        ))
    return output


def _igbt_steady_state_rows(path: Path) -> np.ndarray:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    measurement = data.get("measurement")
    if measurement is None or not hasattr(measurement, "steadyState"):
        return np.empty((0, 6), dtype=np.float64)
    rows = []
    for item in _mat_items(measurement.steadyState):
        time_epoch = _finite_float(getattr(item, "timeEpoch", None))
        domain = getattr(item, "timeDomain", None)
        if not time_epoch or domain is None:
            continue
        node1 = _finite_float(getattr(domain, "node1Voltage", None))
        node2 = _finite_float(getattr(domain, "node2Voltage", None))
        current = _finite_float(getattr(domain, "collectorEmitterCurrent", None))
        package_temp = _finite_float(getattr(domain, "packageTemperature", None))
        supply = _finite_float(getattr(domain, "supplyVoltage", None))
        if None in (node1, node2, current, package_temp, supply):
            continue
        vce = abs(node1 - node2)
        current_abs = abs(current)
        if current_abs < 0.02 or not (-40.0 <= package_temp <= 350.0):
            continue
        if not (0.0 <= vce <= 10.0 and 0.0 < supply <= 30.0):
            continue
        on_indicator = np.clip(vce / current_abs, 0.0, 100.0)
        rows.append((time_epoch, vce, current_abs, on_indicator, package_temp, supply))
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    array = np.asarray(rows, dtype=np.float64)
    return array[np.argsort(array[:, 0], kind="stable")]


def load_igbt_dataset(
    root: str | Path,
    max_points: int = 96,
    prefix_fractions: tuple[float, ...] = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
) -> LoadedDataset:
    """Load valid thermal-overstress IGBT runs from NASA MATLAB files."""
    root = Path(root)
    series: list[DegradationSeries] = []
    groups: list[str] = []
    source_files: list[str] = []
    for path in sorted(root.rglob("*.mat")):
        path_text = str(path).lower()
        if "smu parameter characterization" in path_text or "new devices" in path_text:
            continue
        if "check" in path.name.lower():
            continue
        rows = _igbt_steady_state_rows(path)
        if len(rows) < 40:
            continue
        rows = _bin_median(rows, max_points=max_points)
        times = (rows[:, 0] - rows[0, 0]) / 3600.0
        responses = rows[:, 1:4].astype(np.float32)
        temperature = np.clip(rows[:, 4] / 300.0, -0.2, 1.5)
        voltage_ratio = np.clip(rows[:, 5] / 20.0, 0.0, 2.0)
        current_ratio = np.clip(rows[:, 2] / 50.0, 0.0, 2.0)
        stresses = np.column_stack([
            temperature,
            voltage_ratio,
            current_ratio,
            np.zeros(len(rows)),
            np.ones(len(rows)),
            np.ones(len(rows)),
            np.ones(len(rows)),
            np.zeros(len(rows)),
        ]).astype(np.float32)
        group = path.relative_to(root).as_posix()
        prefixes = _prefix_series(
            "igbt", group, times, stresses, responses, prefix_fractions
        )
        series.extend(prefixes)
        groups.extend([group] * len(prefixes))
        source_files.append(str(path))
    if not series:
        raise ValueError(f"no valid IGBT aging sequences found under {root}")
    return LoadedDataset(
        device_type="igbt",
        response_names=("collector_emitter_voltage", "collector_current", "on_state_indicator"),
        response_directions=(1, 1, 1),
        series=series,
        groups=groups,
        source_files=source_files,
    )


def _matlab_char(value) -> str:
    """Decode a MATLAB v7.3 char dataset or attribute."""
    array = np.asarray(value)
    if array.dtype.kind == "S":
        return "".join(bytes(item).decode("utf-8", errors="ignore") for item in array.ravel())
    return "".join(chr(int(item)) for item in array.ravel())


def _capacitor_transient_rows(path: Path, root_name: str, max_points: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract compact waveform health features from MATLAB v7.3 transient records."""
    rows = []
    with h5py.File(path, "r") as handle:
        transient = handle[root_name]["Transient_Data"]
        for group_name, group in transient.items():
            if not isinstance(group, h5py.Group) or "VO" not in group or "VL" not in group:
                continue
            vo_dataset = group["VO"]
            vl_dataset = group["VL"]
            count = vo_dataset.shape[0]
            if count < 20:
                continue
            indices = np.unique(np.linspace(0, count - 1, max_points, dtype=np.int64))
            vo = np.asarray(vo_dataset[indices, :], dtype=np.float64)
            vl = np.asarray(vl_dataset[indices, :], dtype=np.float64)
            valid = (np.isfinite(vo).sum(axis=1) >= 20) & (np.isfinite(vl).sum(axis=1) >= 20)
            if valid.sum() < 12:
                continue
            vo = vo[valid]
            vl = vl[valid]
            # Each row is one charge/discharge transient; trailing columns are padded NaN.
            features = np.column_stack([
                np.nanpercentile(vo, 95, axis=1) - np.nanpercentile(vo, 5, axis=1),
                np.nanpercentile(vl, 95, axis=1) - np.nanpercentile(vl, 5, axis=1),
                np.sqrt(np.nanmean(np.square(vo - vl), axis=1)),
                np.nanstd(vl, axis=1),
                np.nanmean(np.abs(np.diff(vl, axis=1)), axis=1),
            ])
            finite = np.isfinite(features).all(axis=1)
            if finite.sum() >= 12:
                rows.append((group_name, features[finite]))
    return rows


def load_capacitor_dataset(
    root: str | Path,
    max_points: int = 128,
    prefix_fractions: tuple[float, ...] = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
) -> LoadedDataset:
    """Load NASA electrical-stress capacitor transient records from v7.3 MAT files."""
    root = Path(root)
    series: list[DegradationSeries] = []
    groups: list[str] = []
    source_files: list[str] = []
    for path in sorted(root.rglob("ES*.mat")):
        root_name = path.stem
        try:
            voltage = float(root_name[2:])
        except ValueError:
            continue
        try:
            transient_rows = _capacitor_transient_rows(path, root_name, max_points)
        except (OSError, KeyError):
            # Ignore stale partial MAT files left by an interrupted download.
            continue
        for device_name, features in transient_rows:
            times = np.arange(len(features), dtype=np.float64)
            stresses = np.column_stack([
                np.full(len(features), np.clip(voltage / 14.0, 0.0, 2.0)),
                np.clip(features[:, 1] / 4.0, 0.0, 2.0),
                np.clip(features[:, 2] / 8.0, 0.0, 2.0),
                np.clip(features[:, 3] / 2.0, 0.0, 2.0),
                np.ones(len(features)),
                np.ones(len(features)),
                np.zeros(len(features)),
                np.zeros(len(features)),
            ]).astype(np.float32)
            response = features.astype(np.float32)
            group = f"{path.name}::{device_name}"
            prefixes = _prefix_series(
                "capacitor", group, times, stresses, response, prefix_fractions,
            )
            series.extend(prefixes)
            groups.extend([group] * len(prefixes))
            source_files.append(str(path))
    if not series:
        raise ValueError(f"no valid capacitor aging sequences found under {root}")
    return LoadedDataset(
        device_type="capacitor",
        response_names=("voltage_span", "load_span", "waveform_error", "load_std", "rise_slope"),
        response_directions=(1, 1, 1, 1, 1),
        series=series,
        groups=groups,
        source_files=sorted(set(source_files)),
    )


def _mosfet_steady_state_rows(paths: list[Path]) -> np.ndarray:
    rows = []
    for path in paths:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        measurement = data.get("measurement")
        if measurement is None or not hasattr(measurement, "steadyState"):
            continue
        for item in _mat_items(measurement.steadyState):
            epoch = _finite_float(getattr(item, "timeEpoch", None))
            domain = getattr(item, "timeDomain", None)
            if epoch is None or domain is None:
                continue
            voltage = _finite_float(getattr(domain, "drainSourceVoltage", None))
            current = _finite_float(getattr(domain, "drainCurrent", None))
            package_temp = _finite_float(getattr(domain, "packageTemperature", None))
            flange_temp = _finite_float(getattr(domain, "flangeTemperature", None))
            supply = _finite_float(getattr(domain, "supplyVoltage", None))
            if None in (voltage, current, package_temp, flange_temp, supply):
                continue
            if current <= 0.05 or voltage < 0.0 or voltage > 30.0:
                continue
            if not (-40.0 <= package_temp <= 350.0 and -40.0 <= flange_temp <= 350.0):
                continue
            rds = voltage / current
            if not np.isfinite(rds) or rds <= 0.0 or rds > 100.0:
                continue
            rows.append((epoch, rds, current, package_temp, flange_temp, supply, voltage))
    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    result = np.asarray(rows, dtype=np.float64)
    return result[np.argsort(result[:, 0], kind="stable")]


def load_mosfet_dataset(
    root: str | Path,
    max_points: int = 96,
    prefix_fractions: tuple[float, ...] = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
) -> LoadedDataset:
    """Load NASA MOSFET thermal-overstress runs and normalized RDS(ON)."""
    root = Path(root)
    grouped: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("Test_*_run*.mat")):
        match = re.search(r"(Test_\d+)_run", path.name, flags=re.IGNORECASE)
        if match:
            grouped.setdefault(match.group(1), []).append(path)
    series: list[DegradationSeries] = []
    groups: list[str] = []
    source_files: list[str] = []
    for group, paths in sorted(grouped.items()):
        rows = _mosfet_steady_state_rows(paths)
        if len(rows) < 40:
            continue
        rows = _bin_median(rows, max_points=max_points)
        times = (rows[:, 0] - rows[0, 0]) * 24.0  # MATLAB serial date to hours.
        baseline = float(np.nanmedian(rows[: max(3, len(rows) // 10), 1]))
        baseline = max(baseline, 1e-8)
        responses = np.column_stack([
            rows[:, 1] / baseline,
            rows[:, 2],
            rows[:, 3],
            rows[:, 6],
        ]).astype(np.float32)
        stresses = np.column_stack([
            np.clip(rows[:, 3] / 300.0, -0.2, 1.5),
            np.clip(rows[:, 5] / 30.0, 0.0, 2.0),
            np.clip(rows[:, 2] / 10.0, 0.0, 3.0),
            np.clip(rows[:, 4] / 300.0, -0.2, 1.5),
            np.ones(len(rows)),
            np.ones(len(rows)),
            np.zeros(len(rows)),
            np.zeros(len(rows)),
        ]).astype(np.float32)
        prefixes = _prefix_series("mosfet", group, times, stresses, responses, prefix_fractions)
        series.extend(prefixes)
        groups.extend([group] * len(prefixes))
        source_files.extend(str(path) for path in paths)
    if not series:
        raise ValueError(f"no valid MOSFET aging sequences found under {root}")
    return LoadedDataset(
        device_type="mosfet",
        response_names=("normalized_rds_on", "drain_current", "package_temperature", "drain_source_voltage"),
        response_directions=(1, 1, 1, 1),
        series=series,
        groups=groups,
        source_files=sorted(set(source_files)),
    )
