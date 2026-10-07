"""Leakage-safe training and evaluation utilities for the NASA experiment."""
from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from app.services.unified_degradation_data import (
    DegradationSeries,
    SameDeviceSeriesDataset,
    collate_degradation_series,
)
from app.services.unified_degradation_features import (
    ResponseSpec,
    build_common_feature_vector,
    extract_degradation_features,
)
from app.services.unified_degradation_model import (
    CausalPhysicsUDMNet,
    UnifiedDegradationConfig,
    UnifiedDegradationNet,
    build_ablation_configs,
    train_multidevice_epoch,
)
from experiments.unified_degradation.nasa_loaders import LoadedDataset, STRESS_DIM


@dataclass(frozen=True)
class DatasetSplit:
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class StressLevelProtocol:
    """A pre-declared, physically meaningful group-level stress protocol."""

    device_type: str
    stress_index: int
    stress_name: str
    level_centers: tuple[float, ...]
    assignment_tolerance: float
    max_within_group_range: float
    physical_scale: float
    physical_unit: str
    minimum_groups_per_level: int = 3


@dataclass(frozen=True)
class GroupStressSummary:
    group: str
    representative_value: float
    within_group_range: float
    assigned_level: float | None
    exclusion_reason: str | None


@dataclass(frozen=True)
class StressLevelAudit:
    device_type: str
    supported: bool
    reason: str
    protocol: StressLevelProtocol | None
    groups_by_level: tuple[tuple[float, tuple[str, ...]], ...]
    excluded_groups: tuple[str, ...]
    group_summaries: tuple[GroupStressSummary, ...]


@dataclass(frozen=True)
class StressLevelSplit:
    """One group-disjoint leave-one-stress-level-out evaluation fold."""

    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    device_type: str
    stress_index: int
    stress_name: str
    held_out_level: float
    held_out_physical_level: float
    physical_unit: str
    excluded_groups: tuple[str, ...]


# These levels come from externally applied quantities retained by the NASA
# loaders. They are deliberately declared here instead of inferred by generic
# clustering: response-derived stress columns must not become artificial test
# conditions. Values are in the normalized representation stored in the cache.
NASA_STRESS_LEVEL_PROTOCOLS: dict[str, StressLevelProtocol] = {
    "capacitor": StressLevelProtocol(
        device_type="capacitor",
        stress_index=0,
        stress_name="applied_voltage",
        level_centers=(10.0 / 14.0, 12.0 / 14.0, 14.0 / 14.0),
        assignment_tolerance=0.01,
        max_within_group_range=0.01,
        physical_scale=14.0,
        physical_unit="V",
    ),
    "mosfet": StressLevelProtocol(
        device_type="mosfet",
        stress_index=1,
        stress_name="supply_voltage",
        level_centers=(0.187, 0.299),
        assignment_tolerance=0.01,
        max_within_group_range=0.01,
        physical_scale=30.0,
        physical_unit="V",
    ),
}


NASA_STRESS_LEVEL_LIMITATIONS: dict[str, str] = {
    "igbt": (
        "IGBT does not support a discrete group-level stress holdout: its measured "
        "temperature and supply-voltage columns vary substantially within each run, "
        "while the nearly constant current column is response-derived and has only "
        "one effective level."
    ),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def split_groups(dataset: LoadedDataset, seed: int, test_fraction: float = 0.25) -> DatasetSplit:
    groups = sorted(set(dataset.groups))
    if len(groups) < 3:
        raise ValueError(f"{dataset.device_type} needs at least three independent groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    test_count = max(1, int(round(len(groups) * test_fraction)))
    validation_count = max(1, int(round(len(groups) * 0.15)))
    if test_count + validation_count >= len(groups):
        test_count = validation_count = 1
    return DatasetSplit(
        train_groups=tuple(groups[test_count + validation_count:]),
        validation_groups=tuple(groups[test_count:test_count + validation_count]),
        test_groups=tuple(groups[:test_count]),
    )


def _complete_group_series(dataset: LoadedDataset) -> dict[str, DegradationSeries]:
    """Select the longest observed prefix once for every original group."""
    if len(dataset.series) != len(dataset.groups):
        raise ValueError("dataset.series and dataset.groups must have equal length")
    grouped: dict[str, list[DegradationSeries]] = {}
    for item, group in zip(dataset.series, dataset.groups):
        grouped.setdefault(group, []).append(item)

    def completeness(item: DegradationSeries) -> tuple[float, int, float, str]:
        duration = float(item.times[-1] - item.times[0]) if len(item.times) > 1 else 0.0
        return duration, len(item.times), float(item.target), str(item.sample_id)

    return {
        group: max(items, key=completeness)
        for group, items in sorted(grouped.items())
    }


def audit_nasa_stress_levels(
    dataset: LoadedDataset,
    protocol: StressLevelProtocol | None = None,
) -> StressLevelAudit:
    """Audit whether cached groups form repeatable, externally defined stress levels.

    Each original run/device is counted once using its most complete prefix. A group
    is usable only when its median stress is near a declared level and its stress
    range stays within the protocol limit. This prevents repeated prefixes, noisy
    response-derived columns, and mixed-condition runs from creating fake levels.
    """
    if protocol is None:
        protocol = NASA_STRESS_LEVEL_PROTOCOLS.get(dataset.device_type)
    if protocol is None:
        reason = NASA_STRESS_LEVEL_LIMITATIONS.get(
            dataset.device_type,
            f"{dataset.device_type} has no declared group-level stress protocol",
        )
        return StressLevelAudit(
            device_type=dataset.device_type,
            supported=False,
            reason=reason,
            protocol=None,
            groups_by_level=(),
            excluded_groups=tuple(sorted(set(dataset.groups))),
            group_summaries=(),
        )
    if protocol.device_type != dataset.device_type:
        raise ValueError(
            f"stress protocol is for {protocol.device_type}, not {dataset.device_type}"
        )
    if not protocol.level_centers:
        raise ValueError("stress protocol must declare at least one level")
    if protocol.stress_index < 0:
        raise ValueError("stress_index must be non-negative")
    if protocol.assignment_tolerance < 0 or protocol.max_within_group_range < 0:
        raise ValueError("stress tolerances must be non-negative")
    if protocol.minimum_groups_per_level < 3:
        raise ValueError("minimum_groups_per_level must be at least three")

    groups_by_level: dict[float, list[str]] = {
        float(level): [] for level in protocol.level_centers
    }
    summaries = []
    for group, item in _complete_group_series(dataset).items():
        if item.stresses.shape[1] <= protocol.stress_index:
            raise ValueError(
                f"{dataset.device_type} stress index {protocol.stress_index} is out of range"
            )
        values = np.asarray(item.stresses[:, protocol.stress_index], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summaries.append(GroupStressSummary(
                group=group,
                representative_value=math.nan,
                within_group_range=math.nan,
                assigned_level=None,
                exclusion_reason="non_finite_stress",
            ))
            continue
        representative = float(np.median(values))
        within_group_range = float(np.max(values) - np.min(values))
        nearest_level = min(
            groups_by_level,
            key=lambda level: (abs(level - representative), level),
        )
        if abs(nearest_level - representative) > protocol.assignment_tolerance:
            assigned_level = None
            exclusion_reason = "outside_declared_levels"
        elif within_group_range > protocol.max_within_group_range:
            assigned_level = None
            exclusion_reason = "mixed_or_unstable_stress"
        else:
            assigned_level = nearest_level
            exclusion_reason = None
            groups_by_level[nearest_level].append(group)
        summaries.append(GroupStressSummary(
            group=group,
            representative_value=representative,
            within_group_range=within_group_range,
            assigned_level=assigned_level,
            exclusion_reason=exclusion_reason,
        ))

    groups_by_level_tuple = tuple(
        (level, tuple(sorted(groups)))
        for level, groups in sorted(groups_by_level.items())
    )
    usable_levels = {
        level
        for level, groups in groups_by_level_tuple
        if len(groups) >= protocol.minimum_groups_per_level
    }
    excluded_groups = tuple(sorted(
        summary.group
        for summary in summaries
        if summary.assigned_level not in usable_levels
    ))
    supported = len(usable_levels) >= 2
    counts = ", ".join(
        f"{level * protocol.physical_scale:.3g}{protocol.physical_unit}: {len(groups)} groups"
        for level, groups in groups_by_level_tuple
    )
    if supported:
        reason = f"supported group-level {protocol.stress_name} levels ({counts})"
    else:
        reason = (
            f"{dataset.device_type} needs at least two stress levels with "
            f"{protocol.minimum_groups_per_level} groups each; observed {counts}"
        )
    return StressLevelAudit(
        device_type=dataset.device_type,
        supported=supported,
        reason=reason,
        protocol=protocol,
        groups_by_level=groups_by_level_tuple,
        excluded_groups=excluded_groups,
        group_summaries=tuple(summaries),
    )


def leave_one_stress_level_out_splits(
    dataset: LoadedDataset,
    seed: int,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
    protocol: StressLevelProtocol | None = None,
) -> tuple[StressLevelSplit, ...]:
    """Build deterministic folds with every original group in exactly one part.

    The test set is one complete stress level. Validation and conformal-calibration
    groups are sampled separately from every remaining level, and at least one
    group from each level remains for fitting. Rejected groups appear only in
    ``excluded_groups``.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    audit = audit_nasa_stress_levels(dataset, protocol)
    if not audit.supported or audit.protocol is None:
        raise ValueError(audit.reason)
    protocol = audit.protocol
    level_groups = {
        level: groups
        for level, groups in audit.groups_by_level
        if len(groups) >= protocol.minimum_groups_per_level
    }
    rng = np.random.default_rng(seed)
    folds = []
    for held_out_level, test_groups in level_groups.items():
        validation_groups = []
        calibration_groups = []
        train_groups = []
        for level, groups in level_groups.items():
            if level == held_out_level:
                continue
            shuffled = np.asarray(groups, dtype=object)
            rng.shuffle(shuffled)
            validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
            calibration_count = max(1, int(round(len(shuffled) * calibration_fraction)))
            while validation_count + calibration_count >= len(shuffled):
                if validation_count >= calibration_count and validation_count > 1:
                    validation_count -= 1
                elif calibration_count > 1:
                    calibration_count -= 1
                else:
                    break
            validation_groups.extend(str(group) for group in shuffled[:validation_count])
            calibration_end = validation_count + calibration_count
            calibration_groups.extend(
                str(group) for group in shuffled[validation_count:calibration_end]
            )
            train_groups.extend(str(group) for group in shuffled[calibration_end:])
        folds.append(StressLevelSplit(
            train_groups=tuple(sorted(train_groups)),
            validation_groups=tuple(sorted(validation_groups)),
            calibration_groups=tuple(sorted(calibration_groups)),
            test_groups=tuple(sorted(test_groups)),
            device_type=dataset.device_type,
            stress_index=protocol.stress_index,
            stress_name=protocol.stress_name,
            held_out_level=float(held_out_level),
            held_out_physical_level=float(held_out_level * protocol.physical_scale),
            physical_unit=protocol.physical_unit,
            excluded_groups=audit.excluded_groups,
        ))
    return tuple(folds)


def _select_series(dataset: LoadedDataset, selected_groups: tuple[str, ...]):
    allowed = set(selected_groups)
    return [item for item, group in zip(dataset.series, dataset.groups) if group in allowed]


def build_loaders(
    datasets: Mapping[str, LoadedDataset],
    splits: Mapping[str, DatasetSplit | StressLevelSplit],
    part: str,
    batch_size: int,
    shuffle: bool,
) -> dict[str, DataLoader]:
    loaders = {}
    for device_type, dataset in datasets.items():
        group_names = getattr(splits[device_type], f"{part}_groups")
        selected = _select_series(dataset, group_names)
        loaders[device_type] = DataLoader(
            SameDeviceSeriesDataset(selected),
            batch_size=min(batch_size, len(selected)),
            shuffle=shuffle,
            collate_fn=collate_degradation_series,
        )
    return loaders


@torch.no_grad()
def evaluate_model(
    model: UnifiedDegradationNet,
    loaders: Mapping[str, DataLoader],
) -> tuple[dict, list[dict]]:
    model.eval()
    per_device = {}
    predictions = []
    for device_type, loader in loaders.items():
        y_true, y_pred, y_var = [], [], []
        monotonic_violations = 0
        monotonic_pairs = 0
        physics_values, learned_values = [], []
        for batch in loader:
            output = model(
                device_type,
                batch["responses"],
                batch["stresses"],
                batch["times"],
                batch["mask"],
            )
            target = batch["target"].numpy()
            predicted = output["prediction"].cpu().numpy()
            variance = output["variance"].cpu().numpy()
            endpoint_tensor = output.get("dynamic_endpoint")
            endpoints = (
                endpoint_tensor.detach().cpu().numpy()
                if endpoint_tensor is not None
                else predicted
            )
            trajectory = output["trajectory"].cpu().numpy()
            trajectory_mask = output["mask"].cpu().numpy().astype(bool)
            if "physics_rate" in output:
                physics_rate = output["physics_rate"].cpu().numpy()
                damage_rate = output["damage_rate"].cpu().numpy()
                valid_rate = trajectory_mask & (physics_rate > 0)
                physics_values.extend(physics_rate[valid_rate].tolist())
                learned_values.extend(damage_rate[valid_rate].tolist())
            for row, valid in zip(trajectory, trajectory_mask):
                values = row[valid]
                if len(values) > 1:
                    differences = np.diff(values)
                    monotonic_violations += int(np.sum(differences < -1e-6))
                    monotonic_pairs += len(differences)
            for sample_id, actual, estimate, var, endpoint in zip(
                batch["sample_ids"], target, predicted, variance, endpoints
            ):
                predictions.append({
                    "device_type": device_type,
                    "sample_id": sample_id,
                    "target": float(actual),
                    "prediction": float(estimate),
                    "variance": float(var),
                    "dynamic_endpoint": float(endpoint),
                })
            y_true.extend(target.tolist())
            y_pred.extend(predicted.tolist())
            y_var.extend(variance.tolist())
        device_metrics = regression_metrics(y_true, y_pred, y_var)
        device_metrics["trajectory_monotonic_violation_rate"] = (
            float(monotonic_violations / monotonic_pairs) if monotonic_pairs else math.nan
        )
        if len(physics_values) > 1 and np.std(physics_values) > 1e-12 and np.std(learned_values) > 1e-12:
            device_metrics["physics_rate_spearman"] = float(
                spearmanr(physics_values, learned_values).statistic
            )
        else:
            device_metrics["physics_rate_spearman"] = math.nan
        per_device[device_type] = device_metrics
    all_true = [row["target"] for row in predictions]
    all_pred = [row["prediction"] for row in predictions]
    all_var = [row["variance"] for row in predictions]
    overall = regression_metrics(all_true, all_pred, all_var)
    overall["trajectory_monotonic_violation_rate"] = _finite_mean(
        [
            metrics.get("trajectory_monotonic_violation_rate", math.nan)
            for metrics in per_device.values()
        ]
    )
    overall["physics_rate_spearman"] = _finite_mean(
        [
            metrics.get("physics_rate_spearman", math.nan)
            for metrics in per_device.values()
        ]
    )
    return {"overall": overall, "devices": per_device}, predictions


def evaluate_stress_level_holdout(
    model: UnifiedDegradationNet,
    dataset: LoadedDataset,
    split: StressLevelSplit,
    batch_size: int = 64,
) -> tuple[dict, list[dict]]:
    """Evaluate a fold-specific fitted model only on its unseen stress level.

    The caller must fit ``model`` on ``split.train_groups`` and use only
    ``split.validation_groups`` for model selection. This helper validates the
    group boundary, constructs the held-out loader, and records the condition in
    both metrics and prediction rows.
    """
    if dataset.device_type != split.device_type:
        raise ValueError(
            f"stress split is for {split.device_type}, not {dataset.device_type}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    train = set(split.train_groups)
    validation = set(split.validation_groups)
    calibration = set(split.calibration_groups)
    test = set(split.test_groups)
    excluded = set(split.excluded_groups)
    if not train or not validation or not calibration or not test:
        raise ValueError(
            "stress holdout needs non-empty train, validation, calibration, and test groups"
        )
    assigned_parts = (train, validation, calibration, test)
    if any(
        left & right
        for index, left in enumerate(assigned_parts)
        for right in assigned_parts[index + 1:]
    ):
        raise ValueError("stress holdout groups must be disjoint")
    if excluded & (train | validation | calibration | test):
        raise ValueError("excluded stress groups cannot appear in a data split")
    known_groups = set(dataset.groups)
    unknown_groups = (train | validation | calibration | test | excluded) - known_groups
    if unknown_groups:
        raise ValueError(f"stress holdout contains unknown groups: {sorted(unknown_groups)}")
    unassigned_groups = known_groups - (
        train | validation | calibration | test | excluded
    )
    if unassigned_groups:
        raise ValueError(
            f"stress holdout does not account for groups: {sorted(unassigned_groups)}"
        )

    metrics, predictions = evaluate_model(
        model,
        build_loaders(
            {dataset.device_type: dataset},
            {dataset.device_type: split},
            "test",
            batch_size,
            False,
        ),
    )
    holdout_metadata = {
        "device_type": split.device_type,
        "stress_index": split.stress_index,
        "stress_name": split.stress_name,
        "held_out_level": split.held_out_level,
        "held_out_physical_level": split.held_out_physical_level,
        "physical_unit": split.physical_unit,
        "train_group_count": len(train),
        "validation_group_count": len(validation),
        "calibration_group_count": len(calibration),
        "test_group_count": len(test),
        "excluded_group_count": len(excluded),
    }
    metrics["stress_holdout"] = holdout_metadata
    for row in predictions:
        row.update({
            "stress_name": split.stress_name,
            "held_out_level": split.held_out_level,
            "held_out_physical_level": split.held_out_physical_level,
            "physical_unit": split.physical_unit,
        })
    return metrics, predictions


def regression_metrics(y_true, y_pred, variance=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "spearman": math.nan}
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else math.nan
    result = {
        "count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
        "spearman": float(rho) if np.isfinite(rho) else math.nan,
    }
    if variance is not None:
        variance = np.clip(np.asarray(variance, dtype=float), 1e-6, None)
        result["gaussian_nll"] = float(np.mean(
            0.5 * (np.log(variance) + np.square(y_true - y_pred) / variance)
        ))
        result["mean_variance"] = float(np.mean(variance))
        interval_radius = 1.96 * np.sqrt(variance)
        lower = y_pred - interval_radius
        upper = y_pred + interval_radius
        result["normal_95_coverage"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        result["normal_95_width"] = float(np.mean(upper - lower))
    return result


def model_config_for(
    datasets: Mapping[str, LoadedDataset],
    physics_loss_weight: float = 0.05,
    variant: str = "legacy",
) -> UnifiedDegradationConfig:
    variant = variant.lower().replace("-", "_")
    use_latent_dynamics = variant in {"cp_udm", "causal_physics", "causal_physics_udm"}
    return UnifiedDegradationConfig(
        response_dims={name: len(dataset.response_names) for name, dataset in datasets.items()},
        response_directions={name: dataset.response_directions for name, dataset in datasets.items()},
        stress_dim=STRESS_DIM,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.15,
        physics_loss_weight=physics_loss_weight,
        use_latent_dynamics=use_latent_dynamics,
        # The legacy path can be run with the same duration-blind protocol;
        # setting this false is the default and is required for CP-UDM.
        include_absolute_duration=False,
        monotonicity_loss_weight=0.10 if use_latent_dynamics else 0.25,
        forecast_loss_weight=0.10 if use_latent_dynamics else 0.0,
        ranking_loss_weight=0.10 if use_latent_dynamics else 0.0,
        endpoint_consistency_weight=0.10 if use_latent_dynamics else 0.0,
        alignment_loss_weight=0.05 if use_latent_dynamics else 0.0,
    )


def train_model(
    datasets: Mapping[str, LoadedDataset],
    splits: Mapping[str, DatasetSplit | StressLevelSplit],
    config: UnifiedDegradationConfig,
    seed: int,
    epochs: int = 80,
    patience: int = 15,
) -> tuple[UnifiedDegradationNet, list[dict]]:
    set_seed(seed)
    model = CausalPhysicsUDMNet(config) if config.use_latent_dynamics else UnifiedDegradationNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_loaders = build_loaders(datasets, splits, "train", 16, True)
    validation_loaders = build_loaders(datasets, splits, "validation", 32, False)
    history = []
    best_state = None
    best_mae = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        train_losses = train_multidevice_epoch(model, train_loaders, optimizer)
        validation_metrics, _ = evaluate_model(model, validation_loaders)
        validation_mae = validation_metrics["overall"]["mae"]
        history.append({
            "epoch": epoch,
            "validation_mae": validation_mae,
            **{f"train_{key}": value for key, value in train_losses.items()},
        })
        if validation_mae < best_mae - 1e-5:
            best_mae = validation_mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _classical_vector(series, directions: tuple[int, ...]) -> np.ndarray:
    feature_sets = [
        extract_degradation_features(
            series.times,
            series.responses[:, column],
            ResponseSpec(f"response_{column}", directions[column]),
        )
        for column in range(series.responses.shape[1])
    ]
    common = build_common_feature_vector(feature_sets)
    stress_summary = np.concatenate([
        np.nanmean(series.stresses, axis=0),
        np.nanmax(series.stresses, axis=0),
    ]).astype(np.float32)
    return np.concatenate([common, stress_summary])


def run_random_forest_baseline(
    datasets: Mapping[str, LoadedDataset],
    splits: Mapping[str, DatasetSplit | StressLevelSplit],
    seed: int,
) -> tuple[dict, list[dict]]:
    device_names = sorted(datasets)
    X_train, y_train = [], []
    test_rows = []
    for device_type, dataset in datasets.items():
        split = splits[device_type]
        for item, group in zip(dataset.series, dataset.groups):
            vector = _classical_vector(item, dataset.response_directions)
            one_hot = np.asarray([float(device_type == name) for name in device_names], dtype=np.float32)
            vector = np.concatenate([vector, one_hot])
            # Keep the baseline protocol identical to neural training: the
            # validation groups are reserved for model selection and are not
            # used to fit the final regressor.
            if group in split.train_groups:
                X_train.append(vector); y_train.append(item.target)
            elif group in split.test_groups:
                test_rows.append((device_type, item.sample_id, vector, item.target))
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(np.asarray(X_train), np.asarray(y_train))
    X_test = np.asarray([row[2] for row in test_rows])
    predicted = model.predict(X_test)
    predictions = [{
        "device_type": row[0],
        "sample_id": row[1],
        "target": float(row[3]),
        "prediction": float(estimate),
        "variance": None,
        "dynamic_endpoint": None,
    } for row, estimate in zip(test_rows, predicted)]
    per_device = {}
    for device_type in device_names:
        rows = [row for row in predictions if row["device_type"] == device_type]
        per_device[device_type] = regression_metrics(
            [row["target"] for row in rows], [row["prediction"] for row in rows]
        )
    return {
        "overall": regression_metrics(
            [row["target"] for row in predictions],
            [row["prediction"] for row in predictions],
        ),
        "devices": per_device,
    }, predictions


def _response_only_vector(series, directions: tuple[int, ...]) -> np.ndarray:
    """Build a duration/stress-blind vector for the response-only control."""
    feature_sets = [
        extract_degradation_features(
            series.times,
            series.responses[:, column],
            ResponseSpec(f"response_{column}", directions[column]),
        )
        for column in range(series.responses.shape[1])
    ]
    return build_common_feature_vector(feature_sets)


def _device_one_hot(device_type: str, device_names: list[str]) -> np.ndarray:
    return np.asarray(
        [float(device_type == name) for name in device_names], dtype=np.float32
    )


def _prediction_metrics(predictions: list[dict], device_names: list[str]) -> dict:
    per_device = {}
    for device_type in device_names:
        rows = [row for row in predictions if row["device_type"] == device_type]
        per_device[device_type] = regression_metrics(
            [row["target"] for row in rows],
            [row["prediction"] for row in rows],
        )
    return {
        "overall": regression_metrics(
            [row["target"] for row in predictions],
            [row["prediction"] for row in predictions],
        ),
        "devices": per_device,
    }


def run_response_only_baseline(
    datasets: Mapping[str, LoadedDataset],
    splits: Mapping[str, DatasetSplit | StressLevelSplit],
    seed: int,
) -> tuple[dict, list[dict]]:
    """Fit a classical response-only model with no stress or absolute time."""
    device_names = sorted(datasets)
    X_train, y_train, test_rows = [], [], []
    for device_type, dataset in datasets.items():
        split = splits[device_type]
        for item, group in zip(dataset.series, dataset.groups):
            vector = np.concatenate([
                _response_only_vector(item, dataset.response_directions),
                _device_one_hot(device_type, device_names),
            ])
            if group in split.train_groups:
                X_train.append(vector)
                y_train.append(item.target)
            elif group in split.test_groups:
                test_rows.append((device_type, item.sample_id, vector, item.target))
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(np.asarray(X_train), np.asarray(y_train))
    predicted = model.predict(np.asarray([row[2] for row in test_rows]))
    predictions = [
        {
            "device_type": row[0],
            "sample_id": row[1],
            "target": float(row[3]),
            "prediction": float(estimate),
            "variance": None,
            "dynamic_endpoint": None,
        }
        for row, estimate in zip(test_rows, predicted)
    ]
    return _prediction_metrics(predictions, device_names), predictions


def run_duration_only_baseline(
    datasets: Mapping[str, LoadedDataset],
    splits: Mapping[str, DatasetSplit | StressLevelSplit],
    seed: int,
) -> tuple[dict, list[dict]]:
    """Expose the absolute-duration label shortcut as an explicit control."""
    device_names = sorted(datasets)
    X_train, y_train, test_rows = [], [], []
    for device_type, dataset in datasets.items():
        split = splits[device_type]
        for item, group in zip(dataset.series, dataset.groups):
            duration = max(float(item.times[-1] - item.times[0]), 0.0)
            vector = np.concatenate([
                np.asarray([np.log1p(duration)], dtype=np.float32),
                _device_one_hot(device_type, device_names),
            ])
            if group in split.train_groups:
                X_train.append(vector)
                y_train.append(item.target)
            elif group in split.test_groups:
                test_rows.append((device_type, item.sample_id, vector, item.target))
    model = Ridge(alpha=1e-3, random_state=seed)
    model.fit(np.asarray(X_train), np.asarray(y_train))
    predicted = np.clip(model.predict(np.asarray([row[2] for row in test_rows])), 0.0, 1.0)
    predictions = [
        {
            "device_type": row[0],
            "sample_id": row[1],
            "target": float(row[3]),
            "prediction": float(estimate),
            "variance": None,
            "dynamic_endpoint": None,
        }
        for row, estimate in zip(test_rows, predicted)
    ]
    return _prediction_metrics(predictions, device_names), predictions


def split_conformal_interval(
    calibration_targets: np.ndarray | list[float],
    calibration_predictions: np.ndarray | list[float],
    test_predictions: np.ndarray | list[float],
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a finite-sample split-conformal interval for stage predictions."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    calibration_targets = np.asarray(calibration_targets, dtype=float).reshape(-1)
    calibration_predictions = np.asarray(calibration_predictions, dtype=float).reshape(-1)
    test_predictions = np.asarray(test_predictions, dtype=float).reshape(-1)
    if calibration_targets.size == 0 or calibration_targets.size != calibration_predictions.size:
        raise ValueError("calibration arrays must be non-empty and have equal length")
    residuals = np.abs(calibration_targets - calibration_predictions)
    rank = int(np.ceil((residuals.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), residuals.size)
    quantile = float(np.sort(residuals)[rank - 1])
    return (
        np.clip(test_predictions - quantile, 0.0, 1.0),
        np.clip(test_predictions + quantile, 0.0, 1.0),
        quantile,
    )


def conformal_metrics(
    calibration_predictions: list[dict],
    test_predictions: list[dict],
    alpha: float = 0.1,
) -> dict:
    """Evaluate group-held-out conformal coverage and interval width."""
    if not calibration_predictions or not test_predictions:
        return {"coverage": math.nan, "mean_width": math.nan, "radius": math.nan}
    lower, upper, radius = split_conformal_interval(
        [row["target"] for row in calibration_predictions],
        [row["prediction"] for row in calibration_predictions],
        [row["prediction"] for row in test_predictions],
        alpha=alpha,
    )
    targets = np.asarray([row["target"] for row in test_predictions], dtype=float)
    return {
        "coverage": float(np.mean((targets >= lower) & (targets <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "radius": float(radius),
        "alpha": float(alpha),
    }


def group_conformal_metrics(
    calibration_predictions: list[dict],
    test_predictions: list[dict],
    calibration_groups: tuple[str, ...],
    test_groups: tuple[str, ...],
    alpha: float = 0.1,
) -> dict:
    """Calibrate intervals with one worst-prefix residual per independent group.

    Prefixes from the same aging run are correlated. Taking the maximum residual
    within each calibration group makes the conformal sample unit match the split
    unit and also permits reporting simultaneous whole-trajectory group coverage.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    def rows_by_group(rows: list[dict], groups: tuple[str, ...]) -> dict[str, list[dict]]:
        grouped = {}
        for group in groups:
            prefix = f"{group}::"
            matches = [
                row for row in rows
                if str(row["sample_id"]) == group
                or str(row["sample_id"]).startswith(prefix)
            ]
            if not matches:
                raise ValueError(f"predictions contain no rows for group {group}")
            grouped[group] = matches
        return grouped

    calibration_by_group = rows_by_group(
        calibration_predictions, calibration_groups
    )
    test_by_group = rows_by_group(test_predictions, test_groups)
    group_scores = np.asarray([
        max(abs(float(row["target"]) - float(row["prediction"])) for row in rows)
        for rows in calibration_by_group.values()
    ], dtype=float)
    test_estimates = np.asarray(
        [float(row["prediction"]) for row in test_predictions], dtype=float
    )
    test_targets = np.asarray(
        [float(row["target"]) for row in test_predictions], dtype=float
    )
    _, _, radius = split_conformal_interval(
        group_scores,
        np.zeros_like(group_scores),
        test_estimates,
        alpha=alpha,
    )
    lower = np.clip(test_estimates - radius, 0.0, 1.0)
    upper = np.clip(test_estimates + radius, 0.0, 1.0)
    covered = (test_targets >= lower) & (test_targets <= upper)

    coverage_by_sample_id = {
        str(row["sample_id"]): bool(is_covered)
        for row, is_covered in zip(test_predictions, covered)
    }
    simultaneous_group_coverage = np.mean([
        all(coverage_by_sample_id[str(row["sample_id"])] for row in rows)
        for rows in test_by_group.values()
    ])
    picp = float(np.mean(covered))
    mpiw = float(np.mean(upper - lower))
    return {
        "coverage": picp,
        "mean_width": mpiw,
        "picp": picp,
        "mpiw": mpiw,
        "simultaneous_group_coverage": float(simultaneous_group_coverage),
        "radius": float(radius),
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "calibration_group_count": len(calibration_by_group),
        "test_group_count": len(test_by_group),
    }


def save_experiment_outputs(
    output_dir: str | Path,
    metrics: dict,
    predictions: list[dict],
    splits: Mapping[str, DatasetSplit | StressLevelSplit],
    histories: dict,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "splits": {name: asdict(split) for name, split in splits.items()},
        "histories": histories,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    if predictions:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
