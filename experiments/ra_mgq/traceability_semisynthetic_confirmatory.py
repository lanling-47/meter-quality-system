"""Confirm event-time counterfactual traceability on transparent semi-synthetic data.

SEMI-SYNTHETIC EVIDENCE ONLY.  This module never reads the application database.
It is an implementation-level mechanism experiment, not evidence of an identified
causal effect, enterprise deployment performance, or field generalization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT_CAUSES = (
    "component_lot_drift",
    "reflow_peak_excursion",
    "calibration_offset",
    "firmware_timing_mismatch",
    "humidity_exposure",
)
NO_ROOT_CONTROL = "no_injected_fault"
SPLIT_STRATA = ROOT_CAUSES + (NO_ROOT_CONTROL,)
NUISANCE_CANDIDATE = "station_vibration"
LOW_SUPPORT_CANDIDATE = "rare_coating_marker"
POST_OUTCOME_CANDIDATE = "post_outcome_rework_code"
ALL_CANDIDATES = ROOT_CAUSES + (
    NUISANCE_CANDIDATE,
    LOW_SUPPORT_CANDIDATE,
    POST_OUTCOME_CANDIDATE,
)
ABSOLUTE_DEVIATION_CANDIDATES = (
    "reflow_peak_excursion",
    "calibration_offset",
)
METHODS = (
    "proposed_temporal_cf",
    "marginal_correlation",
    "static_unfiltered_cf",
    "time_shuffled_cf",
    "random_eligible",
)

CANDIDATE_METADATA = {
    "component_lot_drift": ("component_lot", "component"),
    "reflow_peak_excursion": ("reflow_profile", "process"),
    "calibration_offset": ("calibration_record", "process"),
    "firmware_timing_mismatch": ("firmware_configuration", "configuration"),
    "humidity_exposure": ("storage_environment", "process"),
    "station_vibration": ("station_sensor", "nuisance_control"),
    "rare_coating_marker": ("coating_inspection", "low_support_control"),
    "post_outcome_rework_code": ("rework_disposition", "post_outcome_leakage_control"),
}


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20260903
    split_seed: int = 4701
    production_group_count: int = 120
    meters_per_group: int = 24
    train_fraction: float = 2.0 / 3.0
    validation_fraction: float = 1.0 / 6.0
    missing_rate: float = 0.04
    rare_observations_per_group: int = 2


@dataclass(frozen=True)
class RunConfig:
    seeds: tuple[int, ...] = (17, 42, 73)
    trees: int = 128
    min_samples_leaf: int = 8
    reference_draws: int = 16
    kappa: float = 8.0
    min_effective_n: float = 6.0
    min_overlap: float = 0.15
    min_reference_count: int = 12
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 92317


@dataclass
class TraceabilityDataset:
    base_features: np.ndarray
    base_feature_names: tuple[str, ...]
    event_values: np.ndarray
    event_observed: np.ndarray
    event_confidence: np.ndarray
    event_timestamps: np.ndarray
    outcome_timestamps: np.ndarray
    labels: np.ndarray
    true_probabilities: np.ndarray
    sample_ids: np.ndarray
    group_ids: np.ndarray
    product_models: np.ndarray
    shifts: np.ndarray
    true_roots: np.ndarray
    split_groups: dict[str, tuple[str, ...]]
    group_manifest: list[dict[str, Any]]

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(np.isin(self.group_ids, self.split_groups[split]))


@dataclass
class RiskModel:
    classifier: RandomForestClassifier
    medians: np.ndarray
    calibration_coef: float
    calibration_intercept: float
    temporal_filter: bool
    timestamps: np.ndarray
    seed: int

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(matrix), matrix, self.medians)
        raw = np.clip(self.classifier.predict_proba(clean)[:, 1], 1e-6, 1.0 - 1e-6)
        raw_logit = np.log(raw / (1.0 - raw))
        calibrated_logit = self.calibration_intercept + self.calibration_coef * raw_logit
        return _sigmoid(calibrated_logit)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def _split_groups(
    group_ids: list[str], roots: list[str], config: GeneratorConfig
) -> dict[str, tuple[str, ...]]:
    """Create a root-stratified, strictly group-disjoint split."""
    rng = np.random.default_rng(config.split_seed)
    result: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for root in SPLIT_STRATA:
        members = np.asarray(
            [group_id for group_id, assigned in zip(group_ids, roots) if assigned == root],
            dtype=object,
        )
        members = members[rng.permutation(len(members))]
        train_end = int(round(config.train_fraction * len(members)))
        validation_end = train_end + int(round(config.validation_fraction * len(members)))
        if train_end <= 0 or validation_end <= train_end or validation_end >= len(members):
            raise ValueError("each root cause needs train, validation, and test groups")
        result["train"].extend(str(value) for value in members[:train_end])
        result["validation"].extend(
            str(value) for value in members[train_end:validation_end]
        )
        result["test"].extend(str(value) for value in members[validation_end:])
    return {name: tuple(sorted(values)) for name, values in result.items()}


def generate_traceability_dataset(
    config: GeneratorConfig = GeneratorConfig(),
) -> TraceabilityDataset:
    """Generate auditable meter events with one known injected upstream cause.

    Incident groups have one balanced root cause from ``ROOT_CAUSES`` and a
    separate balanced stratum supplies no-fault reference groups.  The five
    upstream values enter the outcome logit.  ``station_vibration`` is a
    measured nuisance, ``rare_coating_marker`` is an intentionally unsupported
    candidate, and ``post_outcome_rework_code`` is generated from the outcome
    and timestamped after it.  The latter must never enter a valid prediction.
    """
    if config.production_group_count % len(SPLIT_STRATA):
        raise ValueError("production_group_count must be divisible by all split strata")
    if config.production_group_count < 36:
        raise ValueError("production_group_count must be at least 36")
    if config.meters_per_group < 8:
        raise ValueError("meters_per_group must be at least 8")
    if not 0 <= config.rare_observations_per_group < config.meters_per_group:
        raise ValueError("rare_observations_per_group must be smaller than group size")

    rng = np.random.default_rng(config.seed)
    group_ids = [f"production_group_{index:03d}" for index in range(config.production_group_count)]
    root_sequence = [
        SPLIT_STRATA[index % len(SPLIT_STRATA)]
        for index in range(config.production_group_count)
    ]
    split_groups = _split_groups(group_ids, root_sequence, config)
    group_to_split = {
        group_id: split for split, members in split_groups.items() for group_id in members
    }

    candidate_count = len(ALL_CANDIDATES)
    root_coefficients = np.asarray((1.65, 1.65, 1.65, 1.65, 1.65), dtype=float)
    stage_offsets = np.asarray((12, 24, 38, 50, 63, 72, 81, 108), dtype=float)
    product_effects = {"model_A": -0.18, "model_B": 0.00, "model_C": 0.16}
    shift_effects = {"day": -0.08, "night": 0.10}

    base_rows: list[np.ndarray] = []
    event_value_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    timestamp_rows: list[np.ndarray] = []
    outcomes: list[int] = []
    probabilities: list[float] = []
    sample_ids: list[str] = []
    repeated_groups: list[str] = []
    products: list[str] = []
    shifts: list[str] = []
    roots: list[str] = []
    outcome_times: list[float] = []
    group_manifest: list[dict[str, Any]] = []

    for group_index, (group_id, root) in enumerate(zip(group_ids, root_sequence)):
        root_index = ROOT_CAUSES.index(root) if root in ROOT_CAUSES else None
        product = ("model_A", "model_B", "model_C")[group_index % 3]
        shift = ("day", "night")[(group_index // 3) % 2]
        group_start = float(1_700_000_000 + group_index * 1_000)
        outcome_time = group_start + 100.0
        group_mapping_confidence = float(np.clip(rng.normal(0.93, 0.035), 0.78, 1.0))

        group_levels = rng.normal(0.0, 0.24, size=len(ROOT_CAUSES))
        injected_strength = (
            float(np.clip(rng.normal(1.90, 0.14), 1.55, 2.20))
            if root_index is not None else 0.0
        )
        # Thermal-profile and calibration faults are two-sided deviations.  Their
        # signs alternate across groups, making a global marginal correlation an
        # intentionally meaningful but incomplete comparator rather than a copy of
        # the nonlinear conditional risk model.
        root_direction = (
            float(rng.choice((-1.0, 1.0))) if root_index in (1, 2)
            else (1.0 if root_index is not None else 0.0)
        )
        if root_index is not None:
            group_levels[root_index] += root_direction * injected_strength
        sensitivity = rng.normal(0.0, 1.0, size=config.meters_per_group)
        ambient = rng.normal(0.0, 1.0, size=config.meters_per_group)
        load = rng.normal(0.0, 1.0, size=config.meters_per_group)
        causal_values = (
            group_levels[None, :]
            + rng.normal(0.0, 0.24, size=(config.meters_per_group, len(ROOT_CAUSES)))
        )
        nuisance = (
            0.35 * ambient
            + rng.normal(0.0, 0.85, size=config.meters_per_group)
        )
        transformed_causal = causal_values.copy()
        transformed_causal[:, 1] = np.abs(transformed_causal[:, 1])
        transformed_causal[:, 2] = np.abs(transformed_causal[:, 2])
        logits = (
            -3.55
            + transformed_causal @ root_coefficients
            + 0.40 * sensitivity
            + 0.12 * ambient
            + 0.08 * load
            + product_effects[product]
            + shift_effects[shift]
        )
        true_probability = _sigmoid(logits)
        labels = rng.binomial(1, true_probability).astype(np.int64)

        event_values = np.zeros((config.meters_per_group, candidate_count), dtype=float)
        event_values[:, : len(ROOT_CAUSES)] = causal_values
        event_values[:, ALL_CANDIDATES.index(NUISANCE_CANDIDATE)] = nuisance
        rare_positions = rng.choice(
            config.meters_per_group,
            size=config.rare_observations_per_group,
            replace=False,
        )
        rare_center = 5.8 if group_to_split[group_id] == "test" else 0.0
        event_values[rare_positions, ALL_CANDIDATES.index(LOW_SUPPORT_CANDIDATE)] = (
            rare_center + rng.normal(0.0, 0.08, size=len(rare_positions))
        )
        event_values[:, ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)] = (
            4.0 * labels + rng.normal(0.0, 0.08, size=config.meters_per_group)
        )

        observed = rng.random((config.meters_per_group, candidate_count)) >= config.missing_rate
        observed[:, ALL_CANDIDATES.index(LOW_SUPPORT_CANDIDATE)] = False
        observed[rare_positions, ALL_CANDIDATES.index(LOW_SUPPORT_CANDIDATE)] = True
        observed[:, ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)] = True
        confidence = np.clip(
            rng.normal(group_mapping_confidence, 0.055, size=(config.meters_per_group, candidate_count)),
            0.45,
            1.0,
        )
        confidence[:, ALL_CANDIDATES.index(NUISANCE_CANDIDATE)] *= 0.85
        confidence[:, ALL_CANDIDATES.index(LOW_SUPPORT_CANDIDATE)] = 0.98
        confidence[:, ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)] = 1.0
        confidence[~observed] = 0.0

        timestamps = (
            group_start
            + stage_offsets[None, :]
            + rng.uniform(-1.0, 1.0, size=(config.meters_per_group, candidate_count))
        )
        product_one_hot = np.asarray(
            [float(product == value) for value in ("model_A", "model_B", "model_C")]
        )
        shift_one_hot = np.asarray([float(shift == "night")])
        for meter_index in range(config.meters_per_group):
            base_rows.append(np.concatenate((
                np.asarray([sensitivity[meter_index], ambient[meter_index], load[meter_index]]),
                product_one_hot,
                shift_one_hot,
            )))
            event_value_rows.append(event_values[meter_index])
            observed_rows.append(observed[meter_index])
            confidence_rows.append(confidence[meter_index])
            timestamp_rows.append(timestamps[meter_index])
            outcomes.append(int(labels[meter_index]))
            probabilities.append(float(true_probability[meter_index]))
            sample_ids.append(f"{group_id}::meter_{meter_index:03d}")
            repeated_groups.append(group_id)
            products.append(product)
            shifts.append(shift)
            roots.append(root)
            outcome_times.append(outcome_time)

        group_manifest.append({
            "data_kind": "semi_synthetic",
            "group_id": group_id,
            "split": group_to_split[group_id],
            "true_root": root,
            "product_model": product,
            "shift": shift,
            "meter_count": config.meters_per_group,
            "group_start_timestamp": group_start,
            "outcome_timestamp": outcome_time,
            "injected_root_strength": injected_strength,
            "injected_root_direction": root_direction,
            "mapping_confidence": group_mapping_confidence,
            "realized_positive_rate": float(labels.mean()),
            "expected_positive_rate": float(true_probability.mean()),
        })

    return TraceabilityDataset(
        base_features=np.asarray(base_rows, dtype=np.float64),
        base_feature_names=(
            "meter_sensitivity",
            "ambient_temperature",
            "load_factor",
            "product_model_A",
            "product_model_B",
            "product_model_C",
            "night_shift",
        ),
        event_values=np.asarray(event_value_rows, dtype=np.float64),
        event_observed=np.asarray(observed_rows, dtype=bool),
        event_confidence=np.asarray(confidence_rows, dtype=np.float64),
        event_timestamps=np.asarray(timestamp_rows, dtype=np.float64),
        outcome_timestamps=np.asarray(outcome_times, dtype=np.float64),
        labels=np.asarray(outcomes, dtype=np.int64),
        true_probabilities=np.asarray(probabilities, dtype=np.float64),
        sample_ids=np.asarray(sample_ids, dtype=object),
        group_ids=np.asarray(repeated_groups, dtype=object),
        product_models=np.asarray(products, dtype=object),
        shifts=np.asarray(shifts, dtype=object),
        true_roots=np.asarray(roots, dtype=object),
        split_groups=split_groups,
        group_manifest=group_manifest,
    )


def shuffled_event_timestamps(dataset: TraceabilityDataset, seed: int) -> np.ndarray:
    """Destroy event order within each meter while preserving timestamp marginals."""
    rng = np.random.default_rng(seed)
    shuffled = dataset.event_timestamps.copy()
    for row in range(len(shuffled)):
        shuffled[row] = shuffled[row, rng.permutation(len(ALL_CANDIDATES))]
    return shuffled


def _availability(
    dataset: TraceabilityDataset,
    indices: np.ndarray,
    temporal_filter: bool,
    timestamps: np.ndarray,
) -> np.ndarray:
    available = dataset.event_observed[indices].copy()
    if temporal_filter:
        available &= timestamps[indices] < dataset.outcome_timestamps[indices, None]
    return available


def assemble_matrix(
    dataset: TraceabilityDataset,
    indices: np.ndarray,
    temporal_filter: bool,
    timestamps: np.ndarray,
) -> np.ndarray:
    available = _availability(dataset, indices, temporal_filter, timestamps)
    events = np.where(available, dataset.event_values[indices], np.nan)
    for candidate in ABSOLUTE_DEVIATION_CANDIDATES:
        events[:, ALL_CANDIDATES.index(candidate)] = np.abs(
            events[:, ALL_CANDIDATES.index(candidate)]
        )
    return np.concatenate((dataset.base_features[indices], events), axis=1)


def fit_risk_model(
    dataset: TraceabilityDataset,
    seed: int,
    run: RunConfig,
    temporal_filter: bool,
    timestamps: np.ndarray,
) -> RiskModel:
    """Fit on train groups and fit Platt calibration on validation groups only."""
    train = dataset.indices("train")
    validation = dataset.indices("validation")
    raw_train = assemble_matrix(dataset, train, temporal_filter, timestamps)
    medians = np.asarray([
        0.0 if np.all(~np.isfinite(raw_train[:, column]))
        else float(np.nanmedian(raw_train[:, column]))
        for column in range(raw_train.shape[1])
    ])
    clean_train = np.where(np.isfinite(raw_train), raw_train, medians)
    classifier = RandomForestClassifier(
        n_estimators=run.trees,
        min_samples_leaf=run.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    classifier.fit(clean_train, dataset.labels[train])

    raw_validation = assemble_matrix(dataset, validation, temporal_filter, timestamps)
    clean_validation = np.where(np.isfinite(raw_validation), raw_validation, medians)
    validation_probability = np.clip(
        classifier.predict_proba(clean_validation)[:, 1], 1e-6, 1.0 - 1e-6
    )
    validation_logit = np.log(validation_probability / (1.0 - validation_probability))
    calibrator = LogisticRegression(C=10.0, solver="lbfgs", random_state=seed)
    calibrator.fit(validation_logit[:, None], dataset.labels[validation])
    return RiskModel(
        classifier=classifier,
        medians=medians,
        calibration_coef=float(calibrator.coef_[0, 0]),
        calibration_intercept=float(calibrator.intercept_[0]),
        temporal_filter=temporal_filter,
        timestamps=timestamps,
        seed=seed,
    )


def predict_dataset(
    model: RiskModel,
    dataset: TraceabilityDataset,
    indices: np.ndarray,
) -> np.ndarray:
    return model.predict_matrix(
        assemble_matrix(dataset, indices, model.temporal_filter, model.timestamps)
    )


def calibrated_entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """Normalized Bernoulli entropy of validation-calibrated probabilities."""
    probability = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    entropy = -(
        probability * np.log(probability)
        + (1.0 - probability) * np.log(1.0 - probability)
    ) / math.log(2.0)
    return np.clip(entropy, 0.0, 0.995)


def effective_sample_size(weights: np.ndarray, epsilon: float = 1e-12) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / (np.square(weights).sum() + epsilon))


def counterfactual_score(
    deltas: np.ndarray,
    weights: np.ndarray,
    overlap: float,
    kappa: float,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Implement Eq. (counterfactual-score), including positive-part delta."""
    deltas = np.asarray(deltas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_eff = effective_sample_size(weights, epsilon)
    weighted_positive_delta = float(
        np.sum(weights * np.maximum(0.0, deltas)) / (weights.sum() + epsilon)
    )
    shrinkage = float(n_eff / (n_eff + kappa))
    score = float(weighted_positive_delta * shrinkage * overlap)
    return {
        "weighted_positive_delta": weighted_positive_delta,
        "effective_sample_size": n_eff,
        "sample_size_shrinkage": shrinkage,
        "overlap": float(overlap),
        "score": score,
    }


def _reference_pool(
    dataset: TraceabilityDataset,
    candidate_index: int,
    target_indices: np.ndarray,
    train_indices: np.ndarray,
    temporal_filter: bool,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Qualified references matched on product model and work shift."""
    product = str(dataset.product_models[target_indices[0]])
    shift = str(dataset.shifts[target_indices[0]])
    available = _availability(dataset, train_indices, temporal_filter, timestamps)[:, candidate_index]
    qualified = dataset.labels[train_indices] == 0
    all_available_values = dataset.event_values[
        train_indices[available], candidate_index
    ]
    if len(all_available_values) >= 4:
        q25, median, q75 = np.quantile(all_available_values, (0.25, 0.50, 0.75))
        robust_scale = max(float(q75 - q25), 0.10)
        nominal = (
            np.abs(dataset.event_values[train_indices, candidate_index] - median)
            <= 1.5 * robust_scale
        )
    else:
        nominal = np.ones(len(train_indices), dtype=bool)
    strata = (
        (dataset.product_models[train_indices] == product)
        & (dataset.shifts[train_indices] == shift)
    )
    selected = available & qualified & nominal & strata
    if selected.sum() < 12:
        selected = (
            available & qualified & nominal
            & (dataset.product_models[train_indices] == product)
        )
    if selected.sum() < 12:
        selected = available & qualified & nominal
    return dataset.event_values[train_indices[selected], candidate_index].astype(float)


def _common_support_overlap(
    values: np.ndarray,
    weights: np.ndarray,
    training_values: np.ndarray,
    reference_values: np.ndarray,
) -> float:
    """Soft empirical support score in [0,1] using frozen training quantiles."""
    if len(reference_values) < 2 or len(training_values) < 2 or weights.sum() <= 0:
        return 0.0
    train_lo, train_hi = np.quantile(training_values, (0.01, 0.99))
    ref_lo, ref_q25, ref_q75, ref_hi = np.quantile(
        reference_values, (0.01, 0.25, 0.75, 0.99)
    )
    ref_iqr = max(float(ref_q75 - ref_q25), 0.10)
    lower = max(float(train_lo), float(ref_lo - 1.5 * ref_iqr))
    upper = min(float(train_hi), float(ref_hi + 1.5 * ref_iqr))
    if lower > upper:
        return 0.0
    scale = max(float(train_hi - train_lo) / 4.0, ref_iqr, 0.10)
    distance = np.maximum(np.maximum(lower - values, values - upper), 0.0) / scale
    membership = np.exp(-0.5 * np.square(distance))
    return float(np.sum(weights * membership) / max(float(weights.sum()), 1e-12))


def _counterfactual_probability(
    model: RiskModel,
    dataset: TraceabilityDataset,
    group_indices: np.ndarray,
    candidate_index: int,
    reference_values: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    matrix = assemble_matrix(
        dataset, group_indices, model.temporal_filter, model.timestamps
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(reference_values, size=(len(group_indices), draws), replace=True)
    repeated = np.repeat(matrix, draws, axis=0)
    event_column = len(dataset.base_feature_names) + candidate_index
    replacement = chosen.reshape(-1)
    if ALL_CANDIDATES[candidate_index] in ABSOLUTE_DEVIATION_CANDIDATES:
        replacement = np.abs(replacement)
    repeated[:, event_column] = replacement
    return model.predict_matrix(repeated).reshape(len(group_indices), draws).mean(axis=1)


def _candidate_status(
    candidate: str,
    weights: np.ndarray,
    reference_count: int,
    n_eff: float,
    overlap: float,
    run: RunConfig,
) -> str:
    if weights.sum() <= 1e-12:
        if candidate == POST_OUTCOME_CANDIDATE:
            return "excluded_post_outcome"
        return "evidence_insufficient_missing_or_time"
    if reference_count < run.min_reference_count:
        return "evidence_insufficient_reference"
    if overlap < run.min_overlap:
        return "evidence_insufficient_overlap"
    if n_eff < run.min_effective_n:
        return "evidence_insufficient_effective_n"
    return "eligible"


def rank_counterfactual_candidates(
    dataset: TraceabilityDataset,
    model: RiskModel,
    group_id: str,
    run: RunConfig,
    method: str,
) -> list[dict[str, Any]]:
    """Rank a group's candidates with the full thesis scoring equation."""
    group_indices = np.flatnonzero(dataset.group_ids == group_id)
    train_indices = dataset.indices("train")
    current_probability = predict_dataset(model, dataset, group_indices)
    uncertainty = calibrated_entropy_uncertainty(current_probability)
    available_group = _availability(
        dataset, group_indices, model.temporal_filter, model.timestamps
    )
    available_train = _availability(
        dataset, train_indices, model.temporal_filter, model.timestamps
    )
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(ALL_CANDIDATES):
        observed = available_group[:, candidate_index].astype(float)
        confidence = dataset.event_confidence[group_indices, candidate_index]
        weights = observed * confidence * (1.0 - uncertainty)
        references = _reference_pool(
            dataset,
            candidate_index,
            group_indices,
            train_indices,
            model.temporal_filter,
            model.timestamps,
        )
        training_values = dataset.event_values[
            train_indices[available_train[:, candidate_index]], candidate_index
        ]
        overlap = _common_support_overlap(
            dataset.event_values[group_indices, candidate_index],
            weights,
            training_values,
            references,
        )
        if len(references) > 0 and weights.sum() > 0:
            counterfactual_probability = _counterfactual_probability(
                model,
                dataset,
                group_indices,
                candidate_index,
                references,
                run.reference_draws,
                _stable_seed(model.seed, method, group_id, candidate),
            )
            deltas = current_probability - counterfactual_probability
        else:
            deltas = np.zeros(len(group_indices), dtype=float)
        parts = counterfactual_score(deltas, weights, overlap, run.kappa)
        status = _candidate_status(
            candidate,
            weights,
            len(references),
            parts["effective_sample_size"],
            overlap,
            run,
        )
        rows.append({
            "data_kind": "semi_synthetic",
            "seed": model.seed,
            "method": method,
            "group_id": group_id,
            "true_root": str(dataset.true_roots[group_indices[0]]),
            "candidate": candidate,
            "candidate_event_type": CANDIDATE_METADATA[candidate][0],
            "candidate_role": CANDIDATE_METADATA[candidate][1],
            "is_true_root": int(candidate == dataset.true_roots[group_indices[0]]),
            "status": status,
            "eligible": int(status == "eligible"),
            "rank": None,
            "score": parts["score"] if status == "eligible" else None,
            "raw_score_before_eligibility": parts["score"],
            "weighted_positive_delta": parts["weighted_positive_delta"],
            "effective_sample_size": parts["effective_sample_size"],
            "sample_size_shrinkage": parts["sample_size_shrinkage"],
            "overlap": parts["overlap"],
            "reference_count": int(len(references)),
            "observed_count": int(observed.sum()),
            "pre_outcome_fraction": float(
                np.mean(model.timestamps[group_indices, candidate_index]
                        < dataset.outcome_timestamps[group_indices])
            ),
            "mean_association_confidence": float(
                confidence[observed.astype(bool)].mean() if observed.sum() else 0.0
            ),
            "mean_uncertainty": float(uncertainty.mean()),
            "mean_current_risk": float(current_probability.mean()),
        })
    _assign_ranks(rows)
    return rows


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(key=lambda row: (-float(row["score"]), str(row["candidate"])))
    for rank, row in enumerate(eligible, start=1):
        row["rank"] = rank


def _marginal_coefficients(
    dataset: TraceabilityDataset,
    timestamps: np.ndarray,
) -> np.ndarray:
    train = dataset.indices("train")
    available = _availability(dataset, train, True, timestamps)
    coefficients = np.zeros(len(ALL_CANDIDATES), dtype=float)
    for candidate_index in range(len(ALL_CANDIDATES)):
        selected = available[:, candidate_index]
        values = dataset.event_values[train[selected], candidate_index]
        labels = dataset.labels[train[selected]].astype(float)
        if len(values) >= 3 and np.std(values) > 1e-12 and np.std(labels) > 1e-12:
            coefficients[candidate_index] = float(np.corrcoef(values, labels)[0, 1])
    return coefficients


def rank_simple_candidates(
    dataset: TraceabilityDataset,
    group_id: str,
    run: RunConfig,
    seed: int,
    method: str,
    marginal_coefficients: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Rank temporally eligible candidates with marginal or random scores."""
    if method not in {"marginal_correlation", "random_eligible"}:
        raise ValueError(f"unsupported simple method: {method}")
    group_indices = np.flatnonzero(dataset.group_ids == group_id)
    train = dataset.indices("train")
    available_group = _availability(dataset, group_indices, True, dataset.event_timestamps)
    available_train = _availability(dataset, train, True, dataset.event_timestamps)
    rng = np.random.default_rng(_stable_seed(seed, method, group_id))
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(ALL_CANDIDATES):
        observed = available_group[:, candidate_index].astype(float)
        confidence = dataset.event_confidence[group_indices, candidate_index]
        weights = observed * confidence
        references = _reference_pool(
            dataset,
            candidate_index,
            group_indices,
            train,
            True,
            dataset.event_timestamps,
        )
        training_values = dataset.event_values[
            train[available_train[:, candidate_index]], candidate_index
        ]
        overlap = _common_support_overlap(
            dataset.event_values[group_indices, candidate_index],
            weights,
            training_values,
            references,
        )
        n_eff = effective_sample_size(weights)
        status = _candidate_status(
            candidate, weights, len(references), n_eff, overlap, run
        )
        if method == "marginal_correlation" and len(references):
            coefficient = max(0.0, float(marginal_coefficients[candidate_index]))
            deviation = max(
                0.0,
                float(np.average(dataset.event_values[group_indices, candidate_index], weights=weights))
                - float(np.median(references)),
            ) if weights.sum() > 0 else 0.0
            raw_score = coefficient * deviation
            score = raw_score * (n_eff / (n_eff + run.kappa)) * overlap
        else:
            coefficient = None
            deviation = None
            raw_score = float(rng.random())
            score = raw_score
        rows.append({
            "data_kind": "semi_synthetic",
            "seed": seed,
            "method": method,
            "group_id": group_id,
            "true_root": str(dataset.true_roots[group_indices[0]]),
            "candidate": candidate,
            "candidate_event_type": CANDIDATE_METADATA[candidate][0],
            "candidate_role": CANDIDATE_METADATA[candidate][1],
            "is_true_root": int(candidate == dataset.true_roots[group_indices[0]]),
            "status": status,
            "eligible": int(status == "eligible"),
            "rank": None,
            "score": float(score) if status == "eligible" else None,
            "raw_score_before_eligibility": float(raw_score),
            "weighted_positive_delta": None,
            "effective_sample_size": n_eff,
            "sample_size_shrinkage": float(n_eff / (n_eff + run.kappa)),
            "overlap": overlap,
            "reference_count": int(len(references)),
            "observed_count": int(observed.sum()),
            "pre_outcome_fraction": float(
                np.mean(dataset.event_timestamps[group_indices, candidate_index]
                        < dataset.outcome_timestamps[group_indices])
            ),
            "mean_association_confidence": float(
                confidence[observed.astype(bool)].mean() if observed.sum() else 0.0
            ),
            "mean_uncertainty": None,
            "mean_current_risk": None,
            "marginal_coefficient": coefficient,
            "marginal_deviation": deviation,
        })
    _assign_ranks(rows)
    return rows


def ranking_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(int(row["seed"]), str(row["method"]), str(row["group_id"])) for row in rows})
    metrics: list[dict[str, Any]] = []
    for seed, method, group_id in keys:
        group_rows = [
            row for row in rows
            if row["seed"] == seed and row["method"] == method and row["group_id"] == group_id
        ]
        true_row = next(row for row in group_rows if row["is_true_root"])
        rank = true_row["rank"]
        metrics.append({
            "seed": seed,
            "method": method,
            "group_id": group_id,
            "true_root": true_row["true_root"],
            "true_root_rank": rank,
            "top1": float(rank == 1),
            "top3": float(rank is not None and rank <= 3),
            "reciprocal_rank": float(1.0 / rank if rank is not None else 0.0),
            "ndcg": float(1.0 / math.log2(rank + 1.0) if rank is not None else 0.0),
        })
    return metrics


def _bootstrap_interval(
    rows: list[dict[str, Any]],
    metric: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Cluster bootstrap: sample independent test production groups."""
    groups = sorted({str(row["group_id"]) for row in rows})
    values_by_group = {
        group: np.asarray([float(row[metric]) for row in rows if row["group_id"] == group])
        for group in groups
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draws[index] = float(np.mean(np.concatenate([values_by_group[group] for group in sampled])))
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def summarize_rankings(
    metric_rows: list[dict[str, Any]], run: RunConfig
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in metric_rows if row["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "seed_count": len({row["seed"] for row in selected}),
            "test_group_count": len({row["group_id"] for row in selected}),
        }
        for metric in ("top1", "top3", "reciprocal_rank", "ndcg"):
            seed_means = [
                float(np.mean([row[metric] for row in selected if row["seed"] == seed]))
                for seed in sorted({row["seed"] for row in selected})
            ]
            low, high = _bootstrap_interval(
                selected,
                metric,
                run.bootstrap_replicates,
                _stable_seed(run.bootstrap_seed, method, metric),
            )
            summary[f"{metric}_mean"] = float(np.mean(seed_means))
            summary[f"{metric}_std"] = float(
                np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0
            )
            summary[f"{metric}_bootstrap_ci95"] = [low, high]
        summaries.append(summary)
    return summaries


def paired_bootstrap_differences(
    metric_rows: list[dict[str, Any]], run: RunConfig
) -> list[dict[str, Any]]:
    proposed = "proposed_temporal_cf"
    result: list[dict[str, Any]] = []
    lookup = {
        (row["seed"], row["method"], row["group_id"]): row for row in metric_rows
    }
    groups = sorted({str(row["group_id"]) for row in metric_rows})
    seeds = sorted({int(row["seed"]) for row in metric_rows})
    for baseline in METHODS:
        if baseline == proposed:
            continue
        for metric in ("top1", "top3", "reciprocal_rank", "ndcg"):
            by_group = {
                group: np.asarray([
                    float(lookup[(seed, proposed, group)][metric])
                    - float(lookup[(seed, baseline, group)][metric])
                    for seed in seeds
                ])
                for group in groups
            }
            point = float(np.mean(np.concatenate(list(by_group.values()))))
            rng = np.random.default_rng(
                _stable_seed(run.bootstrap_seed, baseline, metric, "paired")
            )
            draws = np.empty(run.bootstrap_replicates, dtype=float)
            for index in range(run.bootstrap_replicates):
                sampled = rng.choice(groups, size=len(groups), replace=True)
                draws[index] = float(
                    np.mean(np.concatenate([by_group[group] for group in sampled]))
                )
            low, high = np.quantile(draws, (0.025, 0.975))
            result.append({
                "comparison": f"{proposed}_minus_{baseline}",
                "metric": metric,
                "mean_difference": point,
                "batch_bootstrap_ci95": [float(low), float(high)],
            })
    return result


def _risk_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def post_outcome_invariance_audit(
    dataset: TraceabilityDataset,
    model: RiskModel,
    run: RunConfig,
) -> dict[str, Any]:
    """Inject extreme result-derived values and compare predictions and rankings."""
    leak_index = ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)
    changed_values = dataset.event_values.copy()
    changed_values[:, leak_index] = (
        1_000_000.0 + np.arange(len(changed_values), dtype=float)
    )
    injected = replace(dataset, event_values=changed_values)
    test = dataset.indices("test")
    original_probability = predict_dataset(model, dataset, test)
    injected_probability = predict_dataset(model, injected, test)
    max_probability_delta = float(
        np.max(np.abs(original_probability - injected_probability))
    )
    max_score_delta = 0.0
    max_rank_delta = 0
    comparisons = 0
    for group_id in dataset.split_groups["test"]:
        original = rank_counterfactual_candidates(
            dataset, model, group_id, run, "proposed_temporal_cf"
        )
        altered = rank_counterfactual_candidates(
            injected, model, group_id, run, "proposed_temporal_cf"
        )
        altered_lookup = {row["candidate"]: row for row in altered}
        for row in original:
            if row["candidate"] == POST_OUTCOME_CANDIDATE or not row["eligible"]:
                continue
            alternate = altered_lookup[row["candidate"]]
            max_score_delta = max(
                max_score_delta,
                abs(float(row["score"]) - float(alternate["score"])),
            )
            max_rank_delta = max(
                max_rank_delta,
                abs(int(row["rank"]) - int(alternate["rank"])),
            )
            comparisons += 1
    return {
        "audit": "inject_extreme_post_outcome_event",
        "seed": model.seed,
        "prediction_comparison_count": int(len(test)),
        "ranking_comparison_count": comparisons,
        "max_abs_prediction_delta": max_probability_delta,
        "max_abs_eligible_score_delta": max_score_delta,
        "max_abs_eligible_rank_delta": max_rank_delta,
        "passed_exact_invariance": bool(
            max_probability_delta == 0.0
            and max_score_delta == 0.0
            and max_rank_delta == 0
        ),
    }


def _summarize_invariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audit": "inject_extreme_post_outcome_event",
        "seed_count": len(rows),
        "per_seed": rows,
        "prediction_comparison_count": int(sum(
            row["prediction_comparison_count"] for row in rows
        )),
        "ranking_comparison_count": int(sum(
            row["ranking_comparison_count"] for row in rows
        )),
        "max_abs_prediction_delta": float(max(
            row["max_abs_prediction_delta"] for row in rows
        )),
        "max_abs_eligible_score_delta": float(max(
            row["max_abs_eligible_score_delta"] for row in rows
        )),
        "max_abs_eligible_rank_delta": int(max(
            row["max_abs_eligible_rank_delta"] for row in rows
        )),
        "passed_exact_invariance": bool(all(
            row["passed_exact_invariance"] for row in rows
        )),
    }


def _integrity_audit(
    dataset: TraceabilityDataset,
    rankings: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    run: RunConfig,
) -> dict[str, Any]:
    split_sets = {name: set(values) for name, values in dataset.split_groups.items()}
    overlaps = {
        "train_validation": sorted(split_sets["train"] & split_sets["validation"]),
        "train_test": sorted(split_sets["train"] & split_sets["test"]),
        "validation_test": sorted(split_sets["validation"] & split_sets["test"]),
    }
    ranking_keys = [
        (row["seed"], row["method"], row["group_id"], row["candidate"])
        for row in rankings
    ]
    prediction_keys = [
        (row["seed"], row["scenario"], row["sample_id"]) for row in predictions
    ]
    rare_rows = [
        row for row in rankings
        if row["method"] == "proposed_temporal_cf"
        and row["candidate"] == LOW_SUPPORT_CANDIDATE
    ]
    leak_rows = [
        row for row in rankings
        if row["method"] == "proposed_temporal_cf"
        and row["candidate"] == POST_OUTCOME_CANDIDATE
    ]
    shuffled_leak = [
        row for row in rankings
        if row["method"] == "time_shuffled_cf"
        and row["candidate"] == POST_OUTCOME_CANDIDATE
    ]
    static_leak = [
        row for row in rankings
        if row["method"] == "static_unfiltered_cf"
        and row["candidate"] == POST_OUTCOME_CANDIDATE
    ]
    expected_ranked_groups = sum(
        row["split"] == "test" and row["true_root"] in ROOT_CAUSES
        for row in dataset.group_manifest
    )
    expected_ranking_rows = (
        len(run.seeds) * len(METHODS) * expected_ranked_groups * len(ALL_CANDIDATES)
    )
    expected_prediction_rows = (
        len(run.seeds) * 3 * len(dataset.indices("test"))
    )
    return {
        "split_group_overlap": overlaps,
        "split_is_disjoint": not any(overlaps.values()),
        "all_groups_accounted_for": set(dataset.group_ids.tolist()) == set.union(*split_sets.values()),
        "ranking_row_count": len(rankings),
        "ranking_keys_unique": len(ranking_keys) == len(set(ranking_keys)),
        "expected_ranking_row_count": expected_ranking_rows,
        "ranking_row_count_matches_expected": len(rankings) == expected_ranking_rows,
        "prediction_row_count": len(predictions),
        "prediction_keys_unique": len(prediction_keys) == len(set(prediction_keys)),
        "expected_prediction_row_count": expected_prediction_rows,
        "prediction_row_count_matches_expected": len(predictions) == expected_prediction_rows,
        "seeds_complete": sorted({row["seed"] for row in rankings}) == sorted(run.seeds),
        "methods_complete": sorted({row["method"] for row in rankings}) == sorted(METHODS),
        "rare_control_all_rejected": bool(rare_rows and all(not row["eligible"] for row in rare_rows)),
        "rare_control_all_below_overlap_threshold": bool(
            rare_rows and all(row["overlap"] < run.min_overlap for row in rare_rows)
        ),
        "rare_control_max_overlap": float(max(row["overlap"] for row in rare_rows)),
        "rare_control_max_effective_n": float(
            max(row["effective_sample_size"] for row in rare_rows)
        ),
        "proposed_post_outcome_all_excluded": bool(
            leak_rows and all(row["status"] == "excluded_post_outcome" for row in leak_rows)
        ),
        "time_shuffled_leak_eligible_rate": float(
            np.mean([row["eligible"] for row in shuffled_leak])
        ),
        "time_shuffled_leak_top1_rate": float(
            np.mean([row["rank"] == 1 for row in shuffled_leak])
        ),
        "static_unfiltered_leak_top1_rate": float(
            np.mean([row["rank"] == 1 for row in static_leak])
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event_manifest_rows(dataset: TraceabilityDataset) -> Iterable[dict[str, Any]]:
    split_lookup = {
        group: split for split, members in dataset.split_groups.items() for group in members
    }
    for row in range(len(dataset.sample_ids)):
        for candidate_index, candidate in enumerate(ALL_CANDIDATES):
            yield {
                "data_kind": "semi_synthetic",
                "split": split_lookup[str(dataset.group_ids[row])],
                "group_id": str(dataset.group_ids[row]),
                "sample_id": str(dataset.sample_ids[row]),
                "candidate": candidate,
                "event_type": CANDIDATE_METADATA[candidate][0],
                "candidate_role": CANDIDATE_METADATA[candidate][1],
                "event_timestamp": float(dataset.event_timestamps[row, candidate_index]),
                "outcome_timestamp": float(dataset.outcome_timestamps[row]),
                "is_pre_outcome": int(
                    dataset.event_timestamps[row, candidate_index]
                    < dataset.outcome_timestamps[row]
                ),
                "observed": int(dataset.event_observed[row, candidate_index]),
                "association_confidence": float(dataset.event_confidence[row, candidate_index]),
                "event_value": float(dataset.event_values[row, candidate_index]),
                "label": int(dataset.labels[row]),
                "true_root": str(dataset.true_roots[row]),
                "is_true_root": int(candidate == dataset.true_roots[row]),
            }


def _format_metric(row: dict[str, Any], metric: str) -> str:
    low, high = row[f"{metric}_bootstrap_ci95"]
    return (
        f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f} "
        f"[{low:.3f}, {high:.3f}]"
    )


def build_report(payload: dict[str, Any]) -> str:
    summaries = payload["ranking_summaries"]
    integrity = payload["integrity"]
    invariance = payload["post_outcome_invariance"]
    lines = [
        "# Event-time counterfactual traceability confirmation",
        "",
        "> **SEMI-SYNTHETIC EVIDENCE ONLY.** No enterprise production row is used.",
        "> The results test implementation mechanisms; they do not identify a Pearl-style",
        "> causal effect and do not establish field or enterprise generalization.",
        "",
        "## Frozen protocol",
        "",
        f"- Production groups: {payload['dataset']['production_group_count']} total; "
        f"{payload['dataset']['train_groups']}/{payload['dataset']['validation_groups']}/"
        f"{payload['dataset']['test_groups']} train/validation/test.",
        f"- Meters: {payload['dataset']['meter_count']} total; groups never cross splits.",
        "- Five balanced upstream causes and a balanced no-fault reference stratum are",
        "  generated. Ranking is evaluated only on incident test groups. A station",
        "  sensor is a nuisance,",
        "  a two-meter coating marker is a low-support control, and a rework code is",
        "  generated from the result at a strictly later timestamp.",
        "- Random forests use training groups only. Platt calibration uses validation",
        "  groups only. Signed thermal/calibration offsets enter the risk model as",
        "  deviations from their frozen nominal targets. Conditional references are",
        "  qualified training meters inside a training-only robust nominal band, matched",
        "  first on product model and shift.",
        "- The proposed score is weighted positive counterfactual risk reduction times",
        "  N_eff/(N_eff+kappa) times empirical common support. Candidates below frozen",
        "  N_eff/reference/overlap thresholds are marked evidence-insufficient.",
        f"- Fixed model/randomization seeds: {payload['run_config']['seeds']}; batch",
        f"  bootstrap replicates: {payload['run_config']['bootstrap_replicates']}.",
        "",
        "## Held-out production-group root ranking",
        "",
        "Each cell is seed mean +/- seed standard deviation [production-group bootstrap 95% CI].",
        "",
        "| Method | Top-1 | Top-3 | MRR | NDCG |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {_format_metric(row, 'top1')} | "
            f"{_format_metric(row, 'top3')} | "
            f"{_format_metric(row, 'reciprocal_rank')} | "
            f"{_format_metric(row, 'ndcg')} |"
        )

    lines.extend([
        "",
        "## Paired differences from the proposed method",
        "",
        "Positive values favor `proposed_temporal_cf`; intervals resample test production groups.",
        "",
        "| Baseline | Top-1 difference [95% CI] | MRR difference [95% CI] |",
        "|---|---:|---:|",
    ])
    paired_lookup = {
        (row["comparison"], row["metric"]): row
        for row in payload["paired_batch_bootstrap_differences"]
    }
    for baseline in METHODS:
        if baseline == "proposed_temporal_cf":
            continue
        comparison = f"proposed_temporal_cf_minus_{baseline}"
        top1 = paired_lookup[(comparison, "top1")]
        mrr = paired_lookup[(comparison, "reciprocal_rank")]
        top1_low, top1_high = top1["batch_bootstrap_ci95"]
        mrr_low, mrr_high = mrr["batch_bootstrap_ci95"]
        lines.append(
            f"| {baseline} | {top1['mean_difference']:+.3f} "
            f"[{top1_low:+.3f}, {top1_high:+.3f}] | "
            f"{mrr['mean_difference']:+.3f} [{mrr_low:+.3f}, {mrr_high:+.3f}] |"
        )

    lines.extend([
        "",
        "`marginal_correlation` uses only a training-set point-biserial association",
        "times the current group deviation. `static_unfiltered_cf` admits every",
        "timestamp. `time_shuffled_cf` permutes timestamps within each meter before",
        "applying the same filter. `random_eligible` samples only candidates that pass",
        "the proposed eligibility checks.",
        "",
        "## Negative controls and integrity",
        "",
        f"- Exact post-outcome injection/deletion invariance: {invariance['passed_exact_invariance']} "
        f"(max prediction/eligible-score deltas "
        f"{invariance['max_abs_prediction_delta']:.3g}/"
        f"{invariance['max_abs_eligible_score_delta']:.3g}; "
        f"max rank delta {invariance['max_abs_eligible_rank_delta']}).",
        f"- Low/common-support coating candidates all rejected: "
        f"{integrity['rare_control_all_rejected']} (maximum overlap "
        f"{integrity['rare_control_max_overlap']:.3f}, threshold "
        f"{payload['run_config']['min_overlap']:.3f}; maximum N_eff "
        f"{integrity['rare_control_max_effective_n']:.3f}, threshold "
        f"{payload['run_config']['min_effective_n']:.3f}).",
        f"- Proposed post-outcome candidates all excluded: "
        f"{integrity['proposed_post_outcome_all_excluded']}.",
        f"- Static unfiltered leakage candidate Top-1 rate: "
        f"{integrity['static_unfiltered_leak_top1_rate']:.3f}.",
        f"- After timestamp shuffling, leakage candidate eligibility/Top-1 rates: "
        f"{integrity['time_shuffled_leak_eligible_rate']:.3f}/"
        f"{integrity['time_shuffled_leak_top1_rate']:.3f}.",
        f"- Split disjointness: {integrity['split_is_disjoint']}; ranking keys unique: "
        f"{integrity['ranking_keys_unique']}; prediction keys unique: "
        f"{integrity['prediction_keys_unique']}; expected row counts complete: "
        f"{integrity['ranking_row_count_matches_expected'] and integrity['prediction_row_count_matches_expected']}.",
        "",
        "## Interpretation boundary",
        "",
        "The experiment can support or falsify structural statements about temporal",
        "exclusion, confidence/uncertainty weighting, effective-sample shrinkage, common",
        "support, and controlled ranking under this known generator. A high ranking score",
        "is still model sensitivity, not an identified intervention effect. The static",
        "and shuffled controls may attain deceptively strong outcome prediction by using",
        "the result-derived rework code; that is leakage evidence, not a useful gain.",
        "Real production traceability remains unverified until timestamp provenance,",
        "identity links, repair/replace outcomes, and blind engineer adjudication are",
        "available on genuinely held-out enterprise batches.",
        "",
    ])
    return "\n".join(lines)


def run_experiment(
    output_dir: Path,
    generator: GeneratorConfig = GeneratorConfig(),
    run: RunConfig = RunConfig(),
) -> dict[str, Any]:
    random.seed(generator.seed)
    np.random.seed(generator.seed)
    dataset = generate_traceability_dataset(generator)
    test = dataset.indices("test")
    root_by_group = {
        row["group_id"]: row["true_root"] for row in dataset.group_manifest
    }
    test_groups = tuple(
        group for group in dataset.split_groups["test"]
        if root_by_group[group] in ROOT_CAUSES
    )
    marginal = _marginal_coefficients(dataset, dataset.event_timestamps)
    rankings: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    risk_diagnostics: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []

    for seed in run.seeds:
        shuffled = shuffled_event_timestamps(dataset, _stable_seed(seed, "shuffle"))
        model_specs = {
            "proposed_temporal_cf": (True, dataset.event_timestamps),
            "static_unfiltered_cf": (False, dataset.event_timestamps),
            "time_shuffled_cf": (True, shuffled),
        }
        models: dict[str, RiskModel] = {}
        scenario_probabilities: dict[str, np.ndarray] = {}
        for method, (temporal_filter, timestamps) in model_specs.items():
            model = fit_risk_model(
                dataset, seed, run, temporal_filter=temporal_filter, timestamps=timestamps
            )
            models[method] = model
            probability = predict_dataset(model, dataset, test)
            scenario_probabilities[method] = probability
            risk_diagnostics.append({
                "seed": int(seed),
                "scenario": method,
                **_risk_metrics(dataset.labels[test], probability),
            })
            for group_id in test_groups:
                rankings.extend(
                    rank_counterfactual_candidates(dataset, model, group_id, run, method)
                )

        for group_id in test_groups:
            rankings.extend(rank_simple_candidates(
                dataset,
                group_id,
                run,
                seed,
                "marginal_correlation",
                marginal_coefficients=marginal,
            ))
            rankings.extend(rank_simple_candidates(
                dataset, group_id, run, seed, "random_eligible"
            ))

        invariance_rows.append(post_outcome_invariance_audit(
            dataset, models["proposed_temporal_cf"], run
        ))

        proposed_uncertainty = calibrated_entropy_uncertainty(
            scenario_probabilities["proposed_temporal_cf"]
        )
        for position, dataset_index in enumerate(test):
            for scenario in model_specs:
                candidate_timestamps = models[scenario].timestamps[dataset_index]
                predictions.append({
                    "data_kind": "semi_synthetic",
                    "seed": int(seed),
                    "scenario": scenario,
                    "sample_id": str(dataset.sample_ids[dataset_index]),
                    "group_id": str(dataset.group_ids[dataset_index]),
                    "true_root": str(dataset.true_roots[dataset_index]),
                    "label": int(dataset.labels[dataset_index]),
                    "true_probability": float(dataset.true_probabilities[dataset_index]),
                    "prediction": float(scenario_probabilities[scenario][position]),
                    "proposed_calibrated_entropy_uncertainty": float(
                        proposed_uncertainty[position]
                    ),
                    "post_outcome_candidate_is_pre_outcome": int(
                        candidate_timestamps[ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)]
                        < dataset.outcome_timestamps[dataset_index]
                    ),
                })

    metric_rows = ranking_metrics(rankings)
    summaries = summarize_rankings(metric_rows, run)
    paired = paired_bootstrap_differences(metric_rows, run)
    invariance = _summarize_invariance(invariance_rows)
    integrity = _integrity_audit(dataset, rankings, predictions, run)
    split_lookup = {
        group: split for split, members in dataset.split_groups.items() for group in members
    }
    dataset_summary = {
        "production_group_count": generator.production_group_count,
        "meter_count": int(len(dataset.labels)),
        "event_count": int(len(dataset.labels) * len(ALL_CANDIDATES)),
        "positive_rate": float(dataset.labels.mean()),
        "train_groups": len(dataset.split_groups["train"]),
        "validation_groups": len(dataset.split_groups["validation"]),
        "test_groups": len(dataset.split_groups["test"]),
        "ranked_incident_test_groups": len(test_groups),
        "split_groups": dataset.split_groups,
        "root_counts_by_split": {
            split: {
                root: int(sum(
                    row["true_root"] == root and split_lookup[row["group_id"]] == split
                    for row in dataset.group_manifest
                ))
                for root in SPLIT_STRATA
            }
            for split in ("train", "validation", "test")
        },
    }
    payload = {
        "data_kind": "semi_synthetic",
        "evidence_label": "SEMI-SYNTHETIC EVIDENCE ONLY",
        "disclaimer": (
            "Mechanism evidence under a fixed synthetic generator only; not enterprise "
            "data, not field generalization, and not an identified causal effect."
        ),
        "protocol": {
            "protocol_id": "traceability-semisynthetic-v1-2026-09-03",
            "outcome_equation": (
                "Y~Bernoulli(sigmoid(-3.55 + beta_1*x_1 + beta_2*abs(x_2) "
                "+ beta_3*abs(x_3) + beta_4*x_4 + beta_5*x_5 + 0.40*sensitivity "
                "+ 0.12*ambient + 0.08*load + product_effect + shift_effect))"
            ),
            "counterfactual_score": (
                "weighted_mean(max(0, F(X)-E_ref F(X[a<-ref]))) "
                "* N_eff/(N_eff+kappa) * rho_overlap"
            ),
            "weight": "observed * association_confidence * (1-calibrated_entropy) * I(t<t*)",
            "reference_rule": (
                "qualified training meters inside the training-only median +/- 1.5 IQR "
                "nominal band, matched by product model and shift, with product-only "
                "then global qualified fallbacks"
            ),
            "split_unit": "production_group",
            "calibration_split": "validation production groups only",
            "bootstrap_unit": "test production_group",
            "candidate_order": ALL_CANDIDATES,
            "domain_feature_transforms": {
                candidate: "absolute signed deviation from frozen nominal target"
                for candidate in ABSOLUTE_DEVIATION_CANDIDATES
            },
            "root_causes": ROOT_CAUSES,
            "negative_controls": (
                "post-outcome rework code, low-support coating marker, within-meter "
                "timestamp shuffle, and static unfiltered risk model"
            ),
        },
        "generator_config": asdict(generator),
        "run_config": asdict(run),
        "dataset": dataset_summary,
        "risk_diagnostics": risk_diagnostics,
        "ranking_metrics_by_seed_group": metric_rows,
        "ranking_summaries": summaries,
        "paired_batch_bootstrap_differences": paired,
        "post_outcome_invariance": invariance,
        "integrity": integrity,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "rankings.csv", rankings)
    _write_csv(output_dir / "group_manifest.csv", dataset.group_manifest)
    _write_csv(output_dir / "event_manifest.csv", list(_event_manifest_rows(dataset)))
    (output_dir / "report.md").write_text(build_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="experiments/ra_mgq/results/traceability_semisynthetic_confirmatory",
    )
    parser.add_argument("--trees", type=int, default=RunConfig.trees)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=RunConfig.bootstrap_replicates
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_experiment(
        Path(args.output_dir),
        run=RunConfig(
            trees=args.trees,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
    )
    print(json.dumps({
        "evidence_label": payload["evidence_label"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "ranking_rows": payload["integrity"]["ranking_row_count"],
        "prediction_rows": payload["integrity"]["prediction_row_count"],
        "split_is_disjoint": payload["integrity"]["split_is_disjoint"],
        "post_outcome_invariance": payload["post_outcome_invariance"][
            "passed_exact_invariance"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
