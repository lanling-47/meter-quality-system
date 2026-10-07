"""Transparent semi-synthetic confirmation for the RA-MGQ mechanisms.

This experiment is deliberately separate from the application database.  It
tests structural claims under a fixed, inspectable generator; it is not
enterprise evidence and must not be reported as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.services.deep_fusion_service import (
    BATCH_FEATURE_NAMES,
    MGQFormerConfig,
    MGQFormerNet,
    _batch_consistency_loss,
    _reliability_from_batch_matrix,
)


MODEL_NAMES = (
    "meter_only",
    "direct_fusion",
    "ra_mgq_no_consistency",
    "ra_mgq",
)
PERTURBATION_SCENARIOS = ("confidence_aware", "overconfident")


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20260903
    split_seed: int = 4701
    batch_count: int = 72
    meters_per_batch: int = 24
    meter_feature_dim: int = 8
    inspection_sizes: tuple[int, ...] = (4, 8, 16, 32)
    train_fraction: float = 2.0 / 3.0
    validation_fraction: float = 1.0 / 6.0


@dataclass(frozen=True)
class RunConfig:
    seeds: tuple[int, ...] = (17, 42, 73)
    error_rates: tuple[float, ...] = (0.0, 0.10, 0.30, 0.50)
    epochs: int = 100
    patience: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.10
    consistency_weight: float = 0.10


@dataclass
class SemiSyntheticDataset:
    meter_features: np.ndarray
    batch_features: np.ndarray
    labels: np.ndarray
    true_probabilities: np.ndarray
    sample_ids: np.ndarray
    batch_ids: np.ndarray
    split_batches: dict[str, tuple[str, ...]]
    batch_table: list[dict[str, Any]]

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(np.isin(self.batch_ids, self.split_batches[split]))


@dataclass
class TrainedModel:
    name: str
    model: MGQFormerNet
    meter_scaler: StandardScaler
    batch_scaler: StandardScaler
    epochs_ran: int
    best_validation_loss: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def beta_posterior(sample_count: int, positive_count: int) -> tuple[float, float]:
    """Return the Beta(1 + k, 1 + n - k) posterior mean and variance."""
    if sample_count < 0 or positive_count < 0 or positive_count > sample_count:
        raise ValueError("positive_count must lie in [0, sample_count]")
    alpha = 1.0 + float(positive_count)
    beta = 1.0 + float(sample_count - positive_count)
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    return mean, variance


def posterior_contraction_audit(
    sample_sizes: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
) -> list[dict[str, float]]:
    """Audit posterior contraction at a fixed observed anomaly fraction of 0.5."""
    rows = []
    for sample_count in sample_sizes:
        if sample_count % 2:
            raise ValueError("posterior contraction audit requires even sample sizes")
        mean, variance = beta_posterior(sample_count, sample_count // 2)
        rows.append({
            "sample_count": int(sample_count),
            "positive_count": int(sample_count // 2),
            "posterior_mean": float(mean),
            "posterior_variance": float(variance),
        })
    return rows


def consistency_confidence_budget_audit(
    confidence_levels: tuple[float, ...] = (1.0, 0.5, 0.1, 1e-9, 0.0),
    sample_kappa: float = 8.0,
    variance_reference: float = 0.02,
) -> dict[str, Any]:
    """Audit the production consistency loss's absolute confidence budget.

    The fixed fixture has two three-meter batches.  Within each run every link
    receives the same mapping confidence, so relative weights inside a batch
    cancel for ``c > 0``.  The production loss deliberately excludes mapping
    confidence from its final normalization; consequently both the scalar loss
    and both its training-path logit gradient and intermediate probability
    gradient must scale linearly with ``c``.  At ``c = 0`` the production
    no-trusted-group branch must return an exactly zero loss connected to exactly
    zero gradients.
    """
    if tuple(confidence_levels) != (1.0, 0.5, 0.1, 1e-9, 0.0):
        raise ValueError(
            "the confirmatory confidence grid is frozen at 1, 0.5, 0.1, 1e-9, 0"
        )

    fixed_probabilities = (0.15, 0.45, 0.75, 0.30, 0.60, 0.90)
    fixed_group_ids = (0, 0, 0, 1, 1, 1)
    fixed_posterior_means = (0.62, 0.62, 0.62, 0.28, 0.28, 0.28)
    fixed_posterior_variances = (0.010, 0.010, 0.010, 0.015, 0.015, 0.015)
    fixed_sample_counts = (16.0, 16.0, 16.0, 24.0, 24.0, 24.0)
    group_ids = torch.tensor(fixed_group_ids, dtype=torch.int64)
    rows: list[dict[str, Any]] = []

    for confidence in confidence_levels:
        probability_fixture = torch.tensor(fixed_probabilities, dtype=torch.float64)
        logits = torch.logit(probability_fixture).detach().requires_grad_(True)
        probability = torch.sigmoid(logits)
        probability.retain_grad()
        reliability = torch.tensor([
            [
                math.log1p(sample_count),
                posterior_mean,
                posterior_variance,
                confidence,
                1.0,
            ]
            for sample_count, posterior_mean, posterior_variance in zip(
                fixed_sample_counts,
                fixed_posterior_means,
                fixed_posterior_variances,
            )
        ], dtype=torch.float64)
        loss = _batch_consistency_loss(
            probability,
            reliability,
            group_ids,
            sample_kappa=sample_kappa,
            variance_reference=variance_reference,
        )
        loss.backward()
        logit_gradient = logits.grad.detach().cpu().numpy()
        probability_gradient = probability.grad.detach().cpu().numpy()
        rows.append({
            "mapping_confidence": float(confidence),
            "consistency_loss_abs": abs(float(loss.detach().cpu())),
            "gradient_l1_abs": float(np.abs(logit_gradient).sum()),
            "gradient_l2_abs": float(np.linalg.norm(logit_gradient, ord=2)),
            "gradient_max_abs": float(np.abs(logit_gradient).max()),
            "gradient_abs_per_meter": [
                float(value) for value in np.abs(logit_gradient)
            ],
            "probability_gradient_l1_abs": float(
                np.abs(probability_gradient).sum()
            ),
            "probability_gradient_abs_per_meter": [
                float(value) for value in np.abs(probability_gradient)
            ],
        })

    full = rows[0]
    for row in rows:
        row["loss_ratio_to_c1"] = float(
            row["consistency_loss_abs"] / full["consistency_loss_abs"]
        )
        row["gradient_l1_ratio_to_c1"] = float(
            row["gradient_l1_abs"] / full["gradient_l1_abs"]
        )
        row["gradient_l2_ratio_to_c1"] = float(
            row["gradient_l2_abs"] / full["gradient_l2_abs"]
        )
        row["probability_gradient_l1_ratio_to_c1"] = float(
            row["probability_gradient_l1_abs"]
            / full["probability_gradient_l1_abs"]
        )
        row["expected_ratio"] = row["mapping_confidence"]

    positive_rows = [row for row in rows if row["mapping_confidence"] > 0.0]
    zero_row = next(row for row in rows if row["mapping_confidence"] == 0.0)
    max_loss_ratio_error = max(
        abs(row["loss_ratio_to_c1"] - row["expected_ratio"])
        for row in positive_rows
    )
    max_gradient_ratio_error = max(
        abs(row["gradient_l1_ratio_to_c1"] - row["expected_ratio"])
        for row in positive_rows
    )
    max_probability_gradient_ratio_error = max(
        abs(
            row["probability_gradient_l1_ratio_to_c1"]
            - row["expected_ratio"]
        )
        for row in positive_rows
    )
    tolerance = 1e-12
    return {
        "audit_name": "consistency_confidence_budget_audit",
        "implementation_under_test": (
            "app.services.deep_fusion_service._batch_consistency_loss"
        ),
        "gradient_target": (
            "fixed pre-sigmoid model logits; probability gradients are also retained"
        ),
        "fixture": {
            "probabilities": fixed_probabilities,
            "batch_group_ids": fixed_group_ids,
            "posterior_means": fixed_posterior_means,
            "posterior_variances": fixed_posterior_variances,
            "sample_counts": fixed_sample_counts,
            "sample_kappa": float(sample_kappa),
            "variance_reference": float(variance_reference),
        },
        "rows": rows,
        "max_abs_loss_ratio_error": float(max_loss_ratio_error),
        "max_abs_gradient_l1_ratio_error": float(max_gradient_ratio_error),
        "max_abs_probability_gradient_l1_ratio_error": float(
            max_probability_gradient_ratio_error
        ),
        "zero_confidence_loss_exactly_zero": bool(
            zero_row["consistency_loss_abs"] == 0.0
        ),
        "zero_confidence_gradient_exactly_zero": bool(
            zero_row["gradient_l1_abs"] == 0.0
            and all(value == 0.0 for value in zero_row["gradient_abs_per_meter"])
            and zero_row["probability_gradient_l1_abs"] == 0.0
            and all(
                value == 0.0
                for value in zero_row["probability_gradient_abs_per_meter"]
            )
        ),
        "passed_linear_budget_scaling": bool(
            max_loss_ratio_error <= tolerance
            and max_gradient_ratio_error <= tolerance
            and max_probability_gradient_ratio_error <= tolerance
        ),
        "tiny_positive_confidence": 1e-9,
        "tiny_positive_confidence_preserves_linear_scaling": bool(
            next(
                row for row in rows
                if row["mapping_confidence"] == 1e-9
            )["loss_ratio_to_c1"] > 0.0
            and abs(
                next(
                    row for row in rows
                    if row["mapping_confidence"] == 1e-9
                )["loss_ratio_to_c1"] - 1e-9
            ) <= tolerance
        ),
        "tolerance": tolerance,
    }


def _split_batches(batch_ids: list[str], config: GeneratorConfig) -> dict[str, tuple[str, ...]]:
    rng = np.random.default_rng(config.split_seed)
    shuffled = np.asarray(batch_ids, dtype=object)[rng.permutation(len(batch_ids))]
    train_end = int(round(config.train_fraction * len(shuffled)))
    validation_end = train_end + int(round(config.validation_fraction * len(shuffled)))
    return {
        "train": tuple(str(value) for value in shuffled[:train_end]),
        "validation": tuple(str(value) for value in shuffled[train_end:validation_end]),
        "test": tuple(str(value) for value in shuffled[validation_end:]),
    }


def generate_semisynthetic_dataset(
    config: GeneratorConfig = GeneratorConfig(),
) -> SemiSyntheticDataset:
    """Generate meters, sampled batch evidence, and outcomes from fixed equations.

    For batch b and meter i:

      B_b, U_bi ~ Normal(0, 1)
      P(Y_bi=1) = sigmoid(-0.35 + 1.25 B_b + 1.10 U_bi
                           + 0.35 tanh(B_b U_bi))

    Meter features are noisy measurements of U_bi with only weak batch signal.
    A separate binomial inspection samples each batch's expected meter risk.
    The inspection is converted to a Beta(1+k, 1+n-k) posterior.  No application
    database row or rule-generated production label is used.
    """
    if config.batch_count < 12:
        raise ValueError("batch_count must be at least 12")
    if config.meters_per_batch < 8:
        raise ValueError("meters_per_batch must be at least 8")

    rng = np.random.default_rng(config.seed)
    batch_names = [f"batch_{index:03d}" for index in range(config.batch_count)]
    split_batches = _split_batches(batch_names, config)
    batch_feature_index = {name: index for index, name in enumerate(BATCH_FEATURE_NAMES)}

    meter_rows: list[np.ndarray] = []
    batch_rows: list[np.ndarray] = []
    labels: list[int] = []
    probabilities: list[float] = []
    sample_ids: list[str] = []
    repeated_batch_ids: list[str] = []
    batch_table: list[dict[str, Any]] = []

    batch_effects = rng.normal(0.0, 1.0, size=config.batch_count)
    for batch_index, batch_id in enumerate(batch_names):
        batch_effect = float(batch_effects[batch_index])
        individual = rng.normal(0.0, 1.0, size=config.meters_per_batch)
        logits = (
            -0.35
            + 1.25 * batch_effect
            + 1.10 * individual
            + 0.35 * np.tanh(batch_effect * individual)
        )
        true_probability = _sigmoid(logits)
        batch_labels = rng.binomial(1, true_probability).astype(np.int64)

        meter = np.zeros((config.meters_per_batch, config.meter_feature_dim), dtype=np.float32)
        meter[:, 0] = individual + rng.normal(0.0, 0.85, size=config.meters_per_batch)
        meter[:, 1] = (
            0.45 * individual
            + 0.20 * batch_effect
            + rng.normal(0.0, 0.90, size=config.meters_per_batch)
        )
        meter[:, 2] = np.tanh(individual) + rng.normal(0.0, 0.65, size=config.meters_per_batch)
        meter[:, 3] = (
            0.25 * individual * batch_effect
            + rng.normal(0.0, 0.90, size=config.meters_per_batch)
        )
        if config.meter_feature_dim > 4:
            meter[:, 4:] = rng.normal(
                0.0, 1.0, size=(config.meters_per_batch, config.meter_feature_dim - 4)
            )

        inspection_count = int(config.inspection_sizes[batch_index % len(config.inspection_sizes)])
        expected_batch_risk = float(true_probability.mean())
        observed_positive = int(rng.binomial(inspection_count, expected_batch_risk))
        posterior_mean, posterior_variance = beta_posterior(
            inspection_count, observed_positive
        )
        process_noise = rng.normal(0.0, 0.20, size=4)
        batch_vector = np.zeros(len(BATCH_FEATURE_NAMES), dtype=np.float32)
        values = {
            "sample_count": float(inspection_count),
            "high_risk_rate": float(observed_positive / inspection_count),
            "avg_final_error_max": float(0.55 * batch_effect + process_noise[0]),
            "max_final_error_max": float(0.85 * batch_effect + abs(process_noise[1])),
            "avg_error_degradation_rate": float(0.45 * batch_effect + process_noise[2]),
            "max_clock_drift_rate": float(0.35 * batch_effect + process_noise[3]),
            "max_rs485_total_errors": float(max(0, round(2.0 + 1.2 * batch_effect))),
            "profile_confidence": 1.0,
            "posterior_high_risk_mean": float(posterior_mean),
            "posterior_high_risk_variance": float(posterior_variance),
            "mapping_confidence": 1.0,
            "batch_available": 1.0,
        }
        for name, value in values.items():
            batch_vector[batch_feature_index[name]] = value

        batch_table.append({
            "batch_id": batch_id,
            "split": next(name for name, members in split_batches.items() if batch_id in members),
            "batch_effect": batch_effect,
            "meter_count": int(config.meters_per_batch),
            "inspection_count": inspection_count,
            "inspection_positive": observed_positive,
            "posterior_mean": float(posterior_mean),
            "posterior_variance": float(posterior_variance),
            "expected_batch_risk": expected_batch_risk,
            "realized_batch_rate": float(batch_labels.mean()),
        })

        meter_rows.extend(meter)
        batch_rows.extend(np.repeat(batch_vector[None, :], config.meters_per_batch, axis=0))
        labels.extend(batch_labels.tolist())
        probabilities.extend(true_probability.tolist())
        sample_ids.extend(
            f"{batch_id}::meter_{index:03d}" for index in range(config.meters_per_batch)
        )
        repeated_batch_ids.extend([batch_id] * config.meters_per_batch)

    return SemiSyntheticDataset(
        meter_features=np.asarray(meter_rows, dtype=np.float32),
        batch_features=np.asarray(batch_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        true_probabilities=np.asarray(probabilities, dtype=np.float32),
        sample_ids=np.asarray(sample_ids, dtype=object),
        batch_ids=np.asarray(repeated_batch_ids, dtype=object),
        split_batches=split_batches,
        batch_table=batch_table,
    )


def perturb_batch_mappings(
    dataset: SemiSyntheticDataset,
    indices: np.ndarray,
    error_rate: float,
    scenario: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace an exact fraction of batch links with another test-batch profile.

    In ``confidence_aware`` mode, a corrupted link is assigned confidence
    ``1 - error_rate``.  In ``overconfident`` mode it incorrectly retains
    confidence 1.0.  The same corrupted rows and donor batches are used in both
    scenarios, isolating the value of informative provenance confidence.
    """
    if scenario not in PERTURBATION_SCENARIOS:
        raise ValueError(f"unknown perturbation scenario: {scenario}")
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate must lie in [0, 1]")

    raw = dataset.batch_features[indices].copy()
    true_batches = dataset.batch_ids[indices].astype(object)
    observed_batches = true_batches.copy()
    corrupted = np.zeros(len(indices), dtype=bool)
    corruption_count = int(round(error_rate * len(indices)))
    if corruption_count == 0:
        return raw, observed_batches, corrupted

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(indices), size=corruption_count, replace=False)
    available_batches = sorted(set(str(value) for value in true_batches))
    canonical = {
        batch_id: dataset.batch_features[
            indices[np.flatnonzero(true_batches == batch_id)[0]]
        ].copy()
        for batch_id in available_batches
    }
    confidence_index = BATCH_FEATURE_NAMES.index("mapping_confidence")
    available_index = BATCH_FEATURE_NAMES.index("batch_available")
    for position in selected:
        true_batch = str(true_batches[position])
        donors = [batch_id for batch_id in available_batches if batch_id != true_batch]
        donor = donors[int(rng.integers(0, len(donors)))]
        raw[position] = canonical[donor]
        raw[position, confidence_index] = (
            float(1.0 - error_rate) if scenario == "confidence_aware" else 1.0
        )
        raw[position, available_index] = 1.0
        observed_batches[position] = donor
        corrupted[position] = True
    return raw, observed_batches, corrupted


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model(name: str, meter_dim: int, run: RunConfig, seed: int) -> MGQFormerNet:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {name}")
    _set_seed(seed)
    config = MGQFormerConfig(
        single_dim=meter_dim,
        batch_dim=len(BATCH_FEATURE_NAMES),
        d_model=run.d_model,
        nhead=run.nhead,
        num_layers=run.num_layers,
        dropout=run.dropout,
        epochs=run.epochs,
        batch_size=run.batch_size,
        learning_rate=run.learning_rate,
        weight_decay=run.weight_decay,
        seed=seed,
        consistency_loss_weight=(run.consistency_weight if name == "ra_mgq" else 0.0),
    )
    model = MGQFormerNet(config)
    if name == "direct_fusion":
        delattr(model, "reliability_gate")
    return model


def _forward(
    model: MGQFormerNet,
    name: str,
    meter: torch.Tensor,
    batch: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    if name == "meter_only":
        return model(
            meter,
            torch.zeros_like(batch),
            torch.zeros_like(reliability),
        )
    if name == "direct_fusion":
        return model(meter, batch)
    return model(meter, batch, reliability)


def train_model(
    name: str,
    dataset: SemiSyntheticDataset,
    run: RunConfig,
    seed: int,
) -> TrainedModel:
    train_indices = dataset.indices("train")
    validation_indices = dataset.indices("validation")
    meter_scaler = StandardScaler().fit(dataset.meter_features[train_indices])
    batch_scaler = StandardScaler().fit(dataset.batch_features[train_indices])
    meter_scaled = meter_scaler.transform(dataset.meter_features).astype(np.float32)
    batch_scaled = batch_scaler.transform(dataset.batch_features).astype(np.float32)
    reliability = _reliability_from_batch_matrix(
        dataset.batch_features, BATCH_FEATURE_NAMES
    )
    group_lookup = {
        batch_id: index for index, batch_id in enumerate(dataset.split_batches["train"])
    }
    train_groups = np.asarray(
        [group_lookup[str(dataset.batch_ids[index])] for index in train_indices],
        dtype=np.int64,
    )

    model = _build_model(name, dataset.meter_features.shape[1], run, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay
    )
    train_positive = int(dataset.labels[train_indices].sum())
    train_negative = len(train_indices) - train_positive
    pos_weight = max(1.0, float(train_negative) / max(float(train_positive), 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(meter_scaled[train_indices]),
            torch.from_numpy(batch_scaled[train_indices]),
            torch.from_numpy(reliability[train_indices]),
            torch.from_numpy(dataset.labels[train_indices].astype(np.float32)),
        ),
        batch_size=run.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    full_train_meter = torch.from_numpy(meter_scaled[train_indices])
    full_train_batch = torch.from_numpy(batch_scaled[train_indices])
    full_train_reliability = torch.from_numpy(reliability[train_indices])
    full_train_groups = torch.from_numpy(train_groups)
    validation_meter = torch.from_numpy(meter_scaled[validation_indices])
    validation_batch = torch.from_numpy(batch_scaled[validation_indices])
    validation_reliability = torch.from_numpy(reliability[validation_indices])
    validation_labels = torch.from_numpy(
        dataset.labels[validation_indices].astype(np.float32)
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    stale = 0
    epochs_ran = 0
    for epoch in range(run.epochs):
        model.train()
        for meter_batch, batch_batch, reliability_batch, labels_batch in loader:
            optimizer.zero_grad()
            logits = _forward(
                model, name, meter_batch, batch_batch, reliability_batch
            )
            supervised_loss = criterion(logits, labels_batch)
            supervised_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if name == "ra_mgq" and run.consistency_weight > 0.0:
            optimizer.zero_grad()
            probability = torch.sigmoid(_forward(
                model,
                name,
                full_train_meter,
                full_train_batch,
                full_train_reliability,
            ))
            consistency_loss = _batch_consistency_loss(
                probability,
                full_train_reliability,
                full_train_groups,
                sample_kappa=model.config.consistency_sample_kappa,
                variance_reference=model.config.consistency_variance_reference,
            )
            (run.consistency_weight * consistency_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = _forward(
                model,
                name,
                validation_meter,
                validation_batch,
                validation_reliability,
            )
            validation_loss = float(
                criterion(validation_logits, validation_labels).cpu()
            )
        epochs_ran = epoch + 1
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= run.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return TrainedModel(
        name=name,
        model=model,
        meter_scaler=meter_scaler,
        batch_scaler=batch_scaler,
        epochs_ran=epochs_ran,
        best_validation_loss=float(best_loss),
    )


def predict_probabilities(
    trained: TrainedModel,
    meter_features: np.ndarray,
    batch_features: np.ndarray,
) -> np.ndarray:
    meter_scaled = trained.meter_scaler.transform(meter_features).astype(np.float32)
    batch_scaled = trained.batch_scaler.transform(batch_features).astype(np.float32)
    reliability = _reliability_from_batch_matrix(batch_features, BATCH_FEATURE_NAMES)
    with torch.no_grad():
        logits = _forward(
            trained.model,
            trained.name,
            torch.from_numpy(meter_scaled),
            torch.from_numpy(batch_scaled),
            torch.from_numpy(reliability),
        )
    return torch.sigmoid(logits).cpu().numpy()


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            selected = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        if not np.any(selected):
            continue
        observed = float(labels[selected].mean())
        predicted = float(probabilities[selected].mean())
        result += float(selected.mean()) * abs(observed - predicted)
    return float(result)


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    true_probabilities: np.ndarray,
    batch_ids: np.ndarray,
) -> dict[str, float]:
    unique_batches = sorted(set(str(value) for value in batch_ids))
    batch_errors = []
    within_variances = []
    true_within_variances = []
    for batch_id in unique_batches:
        selected = batch_ids == batch_id
        batch_errors.append(
            (float(probabilities[selected].mean()) - float(true_probabilities[selected].mean())) ** 2
        )
        within_variances.append(float(np.var(probabilities[selected])))
        true_within_variances.append(float(np.var(true_probabilities[selected])))
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "batch_brier": float(np.mean(batch_errors)),
        "within_batch_variance": float(np.mean(within_variances)),
        "true_within_batch_variance": float(np.mean(true_within_variances)),
    }


def zero_confidence_invariance_audit(
    trained: TrainedModel,
    meter_features: np.ndarray,
    first_batch_features: np.ndarray,
    second_batch_features: np.ndarray,
) -> float:
    if trained.name not in {"ra_mgq", "ra_mgq_no_consistency", "meter_only"}:
        raise ValueError("zero-confidence audit applies only to gated models")
    meter_scaled = trained.meter_scaler.transform(meter_features).astype(np.float32)
    first_scaled = trained.batch_scaler.transform(first_batch_features).astype(np.float32)
    second_scaled = trained.batch_scaler.transform(second_batch_features).astype(np.float32)
    zero_reliability = torch.zeros((len(meter_features), 5), dtype=torch.float32)
    with torch.no_grad():
        first = _forward(
            trained.model,
            trained.name,
            torch.from_numpy(meter_scaled),
            torch.from_numpy(first_scaled),
            zero_reliability,
        )
        second = _forward(
            trained.model,
            trained.name,
            torch.from_numpy(meter_scaled),
            torch.from_numpy(second_scaled),
            zero_reliability,
        )
    return float(torch.max(torch.abs(first - second)).cpu())


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1) if len(array) > 1 else 0.0)


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({
        (row["model"], row["scenario"], float(row["error_rate"]))
        for row in results
    })
    summaries = []
    metric_names = (
        "auroc",
        "auprc",
        "brier",
        "ece",
        "batch_brier",
        "within_batch_variance",
        "true_within_batch_variance",
    )
    for model, scenario, error_rate in keys:
        selected = [
            row for row in results
            if row["model"] == model
            and row["scenario"] == scenario
            and float(row["error_rate"]) == error_rate
        ]
        summary: dict[str, Any] = {
            "model": model,
            "scenario": scenario,
            "error_rate": error_rate,
            "seed_count": len(selected),
        }
        for metric in metric_names:
            mean, std = _mean_and_std([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)
    return summaries


def _format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def build_report(payload: dict[str, Any]) -> str:
    summaries = payload["summaries"]
    lines = [
        "# RA-MGQ semi-synthetic confirmatory experiment",
        "",
        "> SEMI-SYNTHETIC EVIDENCE ONLY. This report does not contain enterprise",
        "> production data and must not be presented as field generalization evidence.",
        "",
        "## Fixed generator and protocol",
        "",
        "For batch `b` and meter `i`, the generator draws independent `B_b, U_bi ~ N(0,1)`",
        "and samples `Y_bi` from `sigmoid(-0.35 + 1.25 B_b + 1.10 U_bi +",
        "0.35 tanh(B_b U_bi))`. Meter features are noisy measurements of `U_bi`;",
        "batch evidence comes from a separate binomial inspection and a Beta(1,1)",
        "posterior. The dataset and batch split are generated once; seeds 17/42/73",
        "change model initialization and minibatch order only.",
        "",
        f"- Batches: {payload['dataset']['batch_count']} total, "
        f"{payload['dataset']['train_batches']}/{payload['dataset']['validation_batches']}/"
        f"{payload['dataset']['test_batches']} train/validation/test",
        f"- Meters: {payload['dataset']['meter_count']} total",
        "- All preprocessing statistics are fitted on training batches only.",
        "- Error mappings are injected only after training and only into test inputs.",
        "",
        "## Beta posterior contraction",
        "",
        "| Inspection n | Positive k | Posterior mean | Posterior variance |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["posterior_contraction"]:
        lines.append(
            f"| {row['sample_count']} | {row['positive_count']} | "
            f"{row['posterior_mean']:.4f} | {row['posterior_variance']:.6f} |"
        )

    budget_audit = payload["consistency_confidence_budget_audit"]
    lines.extend([
        "",
        "## Consistency confidence budget audit",
        "",
        "The fixed two-batch fixture calls the production `_batch_consistency_loss`",
        "directly. Uniform confidence leaves each within-batch aggregate unchanged;",
        "because confidence is excluded from the final normalization, the absolute",
        "loss and its training-path logit gradient should scale linearly with confidence.",
        "",
        "| Mapping confidence | Absolute loss | Loss / c=1 | Gradient L1 | Gradient / c=1 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in budget_audit["rows"]:
        lines.append(
            f"| {row['mapping_confidence']:.10g} | "
            f"{row['consistency_loss_abs']:.10f} | "
            f"{row['loss_ratio_to_c1']:.10f} | "
            f"{row['gradient_l1_abs']:.10f} | "
            f"{row['gradient_l1_ratio_to_c1']:.10f} |"
        )
    lines.extend([
        "",
        f"Maximum loss/gradient ratio errors are "
        f"{budget_audit['max_abs_loss_ratio_error']:.3g}/"
        f"{budget_audit['max_abs_gradient_l1_ratio_error']:.3g}. "
        f"At zero confidence, exact zero loss/gradient are "
        f"{budget_audit['zero_confidence_loss_exactly_zero']}/"
        f"{budget_audit['zero_confidence_gradient_exactly_zero']}. This audit checks",
        "the fixed-logit gradient budget of the auxiliary batch-supervision path.",
        "It does not claim that the end-to-end model output is linear in confidence:",
        "the learned fusion gate and the remaining training objectives are nonlinear.",
    ])

    lines.extend([
        "",
        "## Clean held-out-batch performance",
        "",
        "| Model | AUROC | AUPRC | Brier | ECE | Batch Brier | Within-batch variance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    clean = [
        row for row in summaries
        if row["scenario"] == "confidence_aware" and row["error_rate"] == 0.0
    ]
    for row in clean:
        lines.append(
            f"| {row['model']} | {_format_pm(row['auroc_mean'], row['auroc_std'])} | "
            f"{_format_pm(row['auprc_mean'], row['auprc_std'])} | "
            f"{_format_pm(row['brier_mean'], row['brier_std'])} | "
            f"{_format_pm(row['ece_mean'], row['ece_std'])} | "
            f"{_format_pm(row['batch_brier_mean'], row['batch_brier_std'])} | "
            f"{_format_pm(row['within_batch_variance_mean'], row['within_batch_variance_std'])} |"
        )

    lines.extend([
        "",
        "## Mapping-error perturbation",
        "",
        "`confidence_aware` assigns each corrupted link confidence `1-error_rate`;",
        "`overconfident` leaves corrupted links at confidence 1.0. The latter is a",
        "negative control: structural gating cannot repair a falsely trusted mapping.",
        "",
        "| Scenario | Error rate | Model | AUPRC | Brier | Batch Brier | ECE |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['error_rate']:.0%} | {row['model']} | "
            f"{_format_pm(row['auprc_mean'], row['auprc_std'])} | "
            f"{_format_pm(row['brier_mean'], row['brier_std'])} | "
            f"{_format_pm(row['batch_brier_mean'], row['batch_brier_std'])} | "
            f"{_format_pm(row['ece_mean'], row['ece_std'])} |"
        )

    lines.extend([
        "",
        "## Structural and trade-off audits",
        "",
        "| Model | Seed | Maximum logit change at zero confidence |",
        "|---|---:|---:|",
    ])
    for row in payload["zero_confidence_invariance"]:
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['max_abs_logit_delta']:.10g} |"
        )

    lookup = {
        (row["model"], row["scenario"], row["error_rate"]): row
        for row in summaries
    }
    direct_clean = lookup[("direct_fusion", "confidence_aware", 0.0)]
    direct_corrupt = lookup[("direct_fusion", "confidence_aware", 0.5)]
    ra_clean = lookup[("ra_mgq", "confidence_aware", 0.0)]
    ra_corrupt = lookup[("ra_mgq", "confidence_aware", 0.5)]
    no_consistency = lookup[("ra_mgq_no_consistency", "confidence_aware", 0.0)]
    lines.extend([
        "",
        f"At 50% confidence-aware mapping error, direct-fusion Brier changes by "
        f"{direct_corrupt['brier_mean'] - direct_clean['brier_mean']:+.4f}, while "
        f"RA-MGQ changes by {ra_corrupt['brier_mean'] - ra_clean['brier_mean']:+.4f}.",
        f"Adding aggregate consistency changes clean batch Brier by "
        f"{ra_clean['batch_brier_mean'] - no_consistency['batch_brier_mean']:+.4f}, "
        f"AUPRC by {ra_clean['auprc_mean'] - no_consistency['auprc_mean']:+.4f}, and "
        f"within-batch prediction variance by "
        f"{ra_clean['within_batch_variance_mean'] - no_consistency['within_batch_variance_mean']:+.4f}.",
        "These are descriptive semi-synthetic effects, not pre-declared universal",
        "performance guarantees.",
        "",
        "## Interpretation boundary",
        "",
        "This experiment can falsify implementation-level claims about posterior",
        "contraction, zero-confidence invariance, sensitivity to mapping corruption,",
        "and the aggregate-consistency trade-off. It cannot establish performance on",
        "real meters, real production batches, or real causal root causes. In",
        "particular, the overconfident negative control must be retained when reporting",
        "robustness: RA-MGQ depends on provenance confidence being informative.",
        "",
    ])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    output_dir: Path,
    generator: GeneratorConfig = GeneratorConfig(),
    run: RunConfig = RunConfig(),
) -> dict[str, Any]:
    torch.set_num_threads(1)
    dataset = generate_semisynthetic_dataset(generator)
    test_indices = dataset.indices("test")
    perturbations: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scenario in PERTURBATION_SCENARIOS:
        for error_rate in run.error_rates:
            perturbations[(scenario, error_rate)] = perturb_batch_mappings(
                dataset,
                test_indices,
                error_rate,
                scenario,
                seed=generator.seed + int(round(error_rate * 1000)),
            )

    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for seed in run.seeds:
        for model_name in MODEL_NAMES:
            trained = train_model(model_name, dataset, run, seed)
            training_rows.append({
                "model": model_name,
                "seed": int(seed),
                "epochs_ran": int(trained.epochs_ran),
                "best_validation_loss": float(trained.best_validation_loss),
                "parameter_count": int(sum(p.numel() for p in trained.model.parameters())),
            })
            if model_name in {"ra_mgq", "ra_mgq_no_consistency", "meter_only"}:
                audit_count = min(64, len(test_indices))
                first = dataset.batch_features[test_indices[:audit_count]]
                second = first[::-1].copy()
                invariance_rows.append({
                    "model": model_name,
                    "seed": int(seed),
                    "max_abs_logit_delta": zero_confidence_invariance_audit(
                        trained,
                        dataset.meter_features[test_indices[:audit_count]],
                        first,
                        second,
                    ),
                })

            for scenario in PERTURBATION_SCENARIOS:
                for error_rate in run.error_rates:
                    perturbed_batch, observed_batches, corrupted = perturbations[
                        (scenario, error_rate)
                    ]
                    probability = predict_probabilities(
                        trained,
                        dataset.meter_features[test_indices],
                        perturbed_batch,
                    )
                    metrics = evaluate_probabilities(
                        dataset.labels[test_indices],
                        probability,
                        dataset.true_probabilities[test_indices],
                        dataset.batch_ids[test_indices],
                    )
                    results.append({
                        "model": model_name,
                        "seed": int(seed),
                        "scenario": scenario,
                        "error_rate": float(error_rate),
                        "actual_error_rate": float(corrupted.mean()),
                        **metrics,
                    })
                    for position, dataset_index in enumerate(test_indices):
                        predictions.append({
                            "data_kind": "semi_synthetic",
                            "model": model_name,
                            "seed": int(seed),
                            "scenario": scenario,
                            "error_rate": float(error_rate),
                            "sample_id": str(dataset.sample_ids[dataset_index]),
                            "true_batch_id": str(dataset.batch_ids[dataset_index]),
                            "observed_batch_id": str(observed_batches[position]),
                            "mapping_corrupted": int(corrupted[position]),
                            "mapping_confidence": float(
                                perturbed_batch[
                                    position,
                                    BATCH_FEATURE_NAMES.index("mapping_confidence"),
                                ]
                            ),
                            "label": int(dataset.labels[dataset_index]),
                            "true_probability": float(dataset.true_probabilities[dataset_index]),
                            "prediction": float(probability[position]),
                        })

    contraction = posterior_contraction_audit()
    confidence_budget = consistency_confidence_budget_audit()
    summaries = summarize_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_kind": "semi_synthetic",
        "disclaimer": (
            "Fixed-generator semi-synthetic mechanism evidence only; not enterprise "
            "data, field generalization evidence, or causal root-cause validation."
        ),
        "generator": asdict(generator),
        "run_config": asdict(run),
        "dataset": {
            "batch_count": generator.batch_count,
            "meter_count": int(len(dataset.labels)),
            "positive_rate": float(dataset.labels.mean()),
            "train_batches": len(dataset.split_batches["train"]),
            "validation_batches": len(dataset.split_batches["validation"]),
            "test_batches": len(dataset.split_batches["test"]),
            "split_batches": dataset.split_batches,
        },
        "posterior_contraction": contraction,
        "consistency_confidence_budget_audit": confidence_budget,
        "zero_confidence_invariance": invariance_rows,
        "training": training_rows,
        "results": results,
        "summaries": summaries,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(output_dir / "posterior_contraction.csv", contraction)
    _write_csv(output_dir / "batch_manifest.csv", dataset.batch_table)
    (output_dir / "report.md").write_text(build_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="experiments/ra_mgq/results/semisynthetic_confirmatory",
    )
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--patience", type=int, default=RunConfig.patience)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_experiment(
        Path(args.output_dir),
        run=RunConfig(epochs=args.epochs, patience=args.patience),
    )
    print(json.dumps({
        "data_kind": payload["data_kind"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "result_rows": len(payload["results"]),
        "prediction_rows": len(payload["results"]) * (
            GeneratorConfig.batch_count
            - len(payload["dataset"]["split_batches"]["train"])
            - len(payload["dataset"]["split_batches"]["validation"])
        ) * GeneratorConfig.meters_per_batch,
    }, indent=2))


if __name__ == "__main__":
    main()
