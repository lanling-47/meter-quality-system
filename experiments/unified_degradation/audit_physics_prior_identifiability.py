"""Audit physical-rate shape identifiability without retraining any model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from app.services.unified_degradation_data import collate_degradation_series
from app.services.unified_degradation_model import (
    CausalPhysicsUDMNet,
    degradation_model_loss,
)
from experiments.unified_degradation.audit_prefix_endpoint_consistency import (
    EXPECTED_MAINMATCH_CONFIGURATIONS,
    EXPECTED_SEEDS,
    MetadataOnlyUnpickler,
    load_cached_sample_metadata,
    parse_prefix_sample_id,
    read_selected_predictions,
)


LOG_RATE_CLAMP = 1e-6
SHAPE_VARIANCE_TOLERANCE = 1e-12
PRIOR_PREFIX_FIELDS = (
    "device_type",
    "sample_id",
    "group_id",
    "prefix_end_index",
    "prefix_ordinal",
    "target",
    "point_count",
    "positive_rate_point_count",
    "positive_rate_fraction",
    "above_log_clamp_point_count",
    "above_log_clamp_fraction",
    "raw_rate_min",
    "raw_rate_max",
    "raw_rate_mean",
    "normalized_rate_mean",
    "centered_log_rate_variance",
    "centered_log_rate_standard_deviation",
    "centered_log_rate_range",
    "nondegenerate_centered_log_shape",
    "stress_0_mean",
    "stress_0_range",
    "stress_1_mean",
    "stress_1_range",
    "capacitor_applied_voltage_proxy_v",
    "in_cp_udm_strict_test",
    "in_cp_udm_ablation_mainmatch_test",
)
PRIOR_SUMMARY_FIELDS = (
    "scope",
    "device_type",
    "prefix_ordinal",
    "prefix_count",
    "group_count",
    "nondegenerate_prefix_count",
    "nondegenerate_prefix_rate",
    "mean_centered_log_rate_variance",
    "median_centered_log_rate_variance",
    "maximum_centered_log_rate_variance",
    "mean_centered_log_rate_range",
    "mean_positive_rate_fraction",
    "mean_above_log_clamp_fraction",
)
SHAPE_PREFIX_FIELDS = (
    "experiment",
    "variant",
    "seed",
    "physics_loss_weight",
    "device_type",
    "sample_id",
    "group_id",
    "prefix_end_index",
    "prefix_ordinal",
    "positive_rate_point_count",
    "prior_centered_log_rate_variance",
    "learned_centered_log_rate_variance",
    "learned_vs_prior_centered_log_shape_mse",
    "saved_prediction",
    "recomputed_prediction",
    "prediction_absolute_error",
)
SHAPE_SUMMARY_FIELDS = (
    "experiment",
    "variant",
    "aggregation_scope",
    "seed",
    "device_type",
    "physics_loss_weight",
    "prefix_count",
    "positive_rate_point_count",
    "point_weighted_centered_log_shape_mse",
    "mean_prefix_centered_log_shape_mse",
    "maximum_prefix_centered_log_shape_mse",
    "mean_prior_centered_log_rate_variance",
    "mean_learned_centered_log_rate_variance",
    "maximum_prediction_absolute_error",
    "composite_loss_physics_component",
    "mse_vs_composite_loss_absolute_difference",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_processed_datasets(processed_dir: Path) -> tuple[dict[str, object], list[dict]]:
    datasets = {}
    provenance = []
    for device_type in ("capacitor", "igbt", "mosfet"):
        path = processed_dir / f"{device_type}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"required processed dataset cache is missing: {path}")
        with path.open("rb") as handle:
            dataset = MetadataOnlyUnpickler(handle).load()
        if str(dataset.device_type) != device_type:
            raise ValueError(f"cached dataset device mismatch in {path}")
        if len(dataset.series) != len(dataset.groups):
            raise ValueError(f"series/group length mismatch in {path}")
        datasets[device_type] = dataset
        provenance.append({
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "prefix_count": len(dataset.series),
            "group_count": len(set(map(str, dataset.groups))),
        })
    return datasets, provenance


def centered_log_rate_components(
    learned_rate: torch.Tensor,
    physics_rate: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reproduce the per-sequence centered-log comparison used in training."""
    if learned_rate.shape != physics_rate.shape or learned_rate.shape != mask.shape:
        raise ValueError("learned_rate, physics_rate, and mask must share shape")
    rate_mask = mask.bool() & (physics_rate > 0.0)
    count = rate_mask.sum(dim=1)
    safe_count = count.clamp_min(1).to(learned_rate.dtype)
    learned_log = torch.log(learned_rate.clamp_min(LOG_RATE_CLAMP))
    prior_log = torch.log(physics_rate.clamp_min(LOG_RATE_CLAMP))
    valid = rate_mask.to(learned_rate.dtype)
    learned_mean = (learned_log * valid).sum(dim=1) / safe_count
    prior_mean = (prior_log * valid).sum(dim=1) / safe_count
    learned_centered = (learned_log - learned_mean.unsqueeze(1)) * valid
    prior_centered = (prior_log - prior_mean.unsqueeze(1)) * valid
    residual = (learned_centered - prior_centered) * valid
    learned_variance = learned_centered.pow(2).sum(dim=1) / safe_count
    prior_variance = prior_centered.pow(2).sum(dim=1) / safe_count
    shape_mse = residual.pow(2).sum(dim=1) / safe_count
    nan = torch.full_like(shape_mse, float("nan"))
    has_rate = count > 0
    return {
        "rate_mask": rate_mask,
        "count": count,
        "learned_centered": learned_centered,
        "prior_centered": prior_centered,
        "learned_variance": torch.where(has_rate, learned_variance, nan),
        "prior_variance": torch.where(has_rate, prior_variance, nan),
        "shape_mse": torch.where(has_rate, shape_mse, nan),
    }


def _prefix_ordinals(dataset: object) -> dict[str, int]:
    by_group = defaultdict(list)
    for item, cached_group in zip(dataset.series, dataset.groups):
        group_id, prefix_end_index = parse_prefix_sample_id(str(item.sample_id))
        if group_id != str(cached_group):
            raise ValueError(
                f"sample ID group {group_id!r} disagrees with cache group {cached_group!r}"
            )
        by_group[group_id].append((prefix_end_index, str(item.sample_id)))
    ordinals = {}
    for group_id, values in by_group.items():
        values.sort()
        endpoints = [value[0] for value in values]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError(f"duplicate prefix endpoint in cached group {group_id!r}")
        for ordinal, (_, sample_id) in enumerate(values, start=1):
            ordinals[sample_id] = ordinal
    return ordinals


@torch.no_grad()
def build_prior_prefix_rows(
    reference_model: CausalPhysicsUDMNet,
    datasets: Mapping[str, object],
    strict_test_ids: set[tuple[str, str]],
    mainmatch_test_ids: set[tuple[str, str]],
) -> list[dict]:
    rows = []
    for device_type, dataset in datasets.items():
        ordinals = _prefix_ordinals(dataset)
        batch = collate_degradation_series(dataset.series)
        raw_rate = reference_model._raw_physics_rate(
            device_type, batch["stresses"]
        ).masked_fill(~batch["mask"], 0.0)
        physics_rate = reference_model._physics_rate_from_prior(
            device_type, batch["stresses"], batch["times"], batch["mask"]
        )
        components = centered_log_rate_components(
            torch.ones_like(physics_rate), physics_rate, batch["mask"]
        )
        for index, item in enumerate(dataset.series):
            sample_id = str(item.sample_id)
            group_id, prefix_end_index = parse_prefix_sample_id(sample_id)
            valid = batch["mask"][index]
            rate_mask = components["rate_mask"][index]
            point_count = int(valid.sum())
            positive_count = int(rate_mask.sum())
            normalized_values = physics_rate[index][valid]
            raw_values = raw_rate[index][valid]
            prior_centered = components["prior_centered"][index][rate_mask]
            prior_variance = float(components["prior_variance"][index])
            above_clamp_count = int(
                ((physics_rate[index] > LOG_RATE_CLAMP) & rate_mask).sum()
            )
            if positive_count:
                centered_range = float(
                    prior_centered.max() - prior_centered.min()
                )
            else:
                centered_range = math.nan
            stress = batch["stresses"][index][valid]
            stress_0 = stress[:, 0]
            stress_1 = stress[:, 1]
            rows.append({
                "device_type": device_type,
                "sample_id": sample_id,
                "group_id": group_id,
                "prefix_end_index": prefix_end_index,
                "prefix_ordinal": ordinals[sample_id],
                "target": float(item.target),
                "point_count": point_count,
                "positive_rate_point_count": positive_count,
                "positive_rate_fraction": positive_count / point_count,
                "above_log_clamp_point_count": above_clamp_count,
                "above_log_clamp_fraction": (
                    above_clamp_count / positive_count if positive_count else 0.0
                ),
                "raw_rate_min": float(raw_values.min()),
                "raw_rate_max": float(raw_values.max()),
                "raw_rate_mean": float(raw_values.mean()),
                "normalized_rate_mean": float(normalized_values.mean()),
                "centered_log_rate_variance": prior_variance,
                "centered_log_rate_standard_deviation": math.sqrt(prior_variance),
                "centered_log_rate_range": centered_range,
                "nondegenerate_centered_log_shape": (
                    positive_count >= 2
                    and prior_variance > SHAPE_VARIANCE_TOLERANCE
                ),
                "stress_0_mean": float(stress_0.mean()),
                "stress_0_range": float(stress_0.max() - stress_0.min()),
                "stress_1_mean": float(stress_1.mean()),
                "stress_1_range": float(stress_1.max() - stress_1.min()),
                "capacitor_applied_voltage_proxy_v": (
                    float(stress_0.mean()) * 14.0
                    if device_type == "capacitor"
                    else None
                ),
                "in_cp_udm_strict_test": (device_type, sample_id) in strict_test_ids,
                "in_cp_udm_ablation_mainmatch_test": (
                    (device_type, sample_id) in mainmatch_test_ids
                ),
            })
    return rows


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else math.nan


def summarize_prior_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    scopes = (
        ("all_cached_prefixes", lambda row: True),
        ("cp_udm_strict_test_prefixes", lambda row: row["in_cp_udm_strict_test"]),
        (
            "cp_udm_ablation_mainmatch_test_prefixes",
            lambda row: row["in_cp_udm_ablation_mainmatch_test"],
        ),
    )
    summaries = []
    devices = sorted({str(row["device_type"]) for row in rows})
    for scope, selector in scopes:
        scope_rows = [row for row in rows if selector(row)]
        for device_type in devices:
            device_rows = [
                row for row in scope_rows if row["device_type"] == device_type
            ]
            ordinals = sorted({int(row["prefix_ordinal"]) for row in device_rows})
            for prefix_ordinal in ["all", *ordinals]:
                selected = (
                    device_rows
                    if prefix_ordinal == "all"
                    else [
                        row
                        for row in device_rows
                        if int(row["prefix_ordinal"]) == prefix_ordinal
                    ]
                )
                variances = [float(row["centered_log_rate_variance"]) for row in selected]
                finite_variances = [value for value in variances if math.isfinite(value)]
                nondegenerate_count = sum(
                    bool(row["nondegenerate_centered_log_shape"])
                    for row in selected
                )
                summaries.append({
                    "scope": scope,
                    "device_type": device_type,
                    "prefix_ordinal": prefix_ordinal,
                    "prefix_count": len(selected),
                    "group_count": len({str(row["group_id"]) for row in selected}),
                    "nondegenerate_prefix_count": nondegenerate_count,
                    "nondegenerate_prefix_rate": (
                        nondegenerate_count / len(selected) if selected else math.nan
                    ),
                    "mean_centered_log_rate_variance": _mean(variances),
                    "median_centered_log_rate_variance": (
                        median(finite_variances) if finite_variances else math.nan
                    ),
                    "maximum_centered_log_rate_variance": (
                        max(finite_variances) if finite_variances else math.nan
                    ),
                    "mean_centered_log_rate_range": _mean(
                        row["centered_log_rate_range"] for row in selected
                    ),
                    "mean_positive_rate_fraction": _mean(
                        row["positive_rate_fraction"] for row in selected
                    ),
                    "mean_above_log_clamp_fraction": _mean(
                        row["above_log_clamp_fraction"] for row in selected
                    ),
                })
    return summaries


def _series_index(datasets: Mapping[str, object]) -> dict[tuple[str, str], object]:
    index = {}
    for device_type, dataset in datasets.items():
        for item in dataset.series:
            key = (device_type, str(item.sample_id))
            if key in index:
                raise ValueError(f"duplicate cached series key: {key}")
            index[key] = item
    return index


def _checkpoint_specs(
    strict_checkpoint_dir: Path, mainmatch_checkpoint_dir: Path
) -> list[dict]:
    specs = []
    for seed in EXPECTED_SEEDS:
        specs.append({
            "experiment": "cp_udm_strict",
            "variant": "cp_udm_net",
            "seed": seed,
            "path": strict_checkpoint_dir / f"cp_udm_net_seed_{seed}.pt",
        })
    for variant in EXPECTED_MAINMATCH_CONFIGURATIONS:
        for seed in EXPECTED_SEEDS:
            specs.append({
                "experiment": "cp_udm_ablation_mainmatch",
                "variant": variant,
                "seed": seed,
                "path": mainmatch_checkpoint_dir / f"{variant}_seed{seed}.pt",
            })
    return specs


@torch.no_grad()
def audit_checkpoints(
    specs: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    series_index: Mapping[tuple[str, str], object],
    prior_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict], dict, list[dict]]:
    prediction_index = {
        (
            str(row["experiment"]),
            str(row["variant"]),
            int(row["seed"]),
            str(row["device_type"]),
            str(row["sample_id"]),
        ): row
        for row in prediction_rows
    }
    prior_index = {
        (str(row["device_type"]), str(row["sample_id"])): row
        for row in prior_rows
    }
    prefix_rows = []
    composite_checks = {}
    provenance = []
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        if not config.use_latent_dynamics:
            raise ValueError(f"checkpoint is not a CP-UDM model: {path}")
        model = CausalPhysicsUDMNet(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        experiment = str(spec["experiment"])
        variant = str(spec["variant"])
        seed = int(spec["seed"])
        selected_predictions = [
            row
            for row in prediction_rows
            if row["experiment"] == experiment
            and row["variant"] == variant
            and int(row["seed"]) == seed
        ]
        if len(selected_predictions) != 108:
            raise ValueError(
                f"expected 108 predictions for {experiment}/{variant}/{seed}, "
                f"found {len(selected_predictions)}"
            )
        for device_type in ("capacitor", "igbt", "mosfet"):
            device_predictions = [
                row
                for row in selected_predictions
                if row["device_type"] == device_type
            ]
            items = [
                series_index[(device_type, str(row["sample_id"]))]
                for row in device_predictions
            ]
            batch = collate_degradation_series(items)
            output = model(
                device_type,
                batch["responses"],
                batch["stresses"],
                batch["times"],
                batch["mask"],
            )
            components = centered_log_rate_components(
                output["damage_rate"], output["physics_rate"], output["mask"]
            )
            point_count = int(components["count"].sum())
            finite = torch.isfinite(components["shape_mse"])
            weighted_mse = float(
                (
                    components["shape_mse"][finite]
                    * components["count"][finite]
                ).sum()
                / components["count"][finite].sum().clamp_min(1)
            )
            composite_physics = float(
                degradation_model_loss(output, batch["target"], config)["physics"]
            )
            difference = abs(weighted_mse - composite_physics)
            if difference > 2e-6:
                raise ValueError(
                    "recomputed centered-log MSE disagrees with composite physics "
                    f"loss for {experiment}/{variant}/{seed}/{device_type}: {difference}"
                )
            check_key = f"{experiment}|{variant}|seed={seed}|{device_type}"
            composite_checks[check_key] = {
                "positive_rate_point_count": point_count,
                "recomputed_point_weighted_mse": weighted_mse,
                "composite_loss_physics_component": composite_physics,
                "absolute_difference": difference,
            }
            recomputed_predictions = output["prediction"].detach().cpu().tolist()
            for index, (source_row, recomputed_prediction) in enumerate(
                zip(device_predictions, recomputed_predictions)
            ):
                sample_id = str(source_row["sample_id"])
                lookup_key = (
                    experiment,
                    variant,
                    seed,
                    device_type,
                    sample_id,
                )
                saved_prediction = float(prediction_index[lookup_key]["prediction"])
                prediction_error = abs(saved_prediction - recomputed_prediction)
                if prediction_error > 2e-6:
                    raise ValueError(
                        f"checkpoint prediction mismatch for {lookup_key}: "
                        f"{prediction_error}"
                    )
                prior = prior_index[(device_type, sample_id)]
                prefix_rows.append({
                    "experiment": experiment,
                    "variant": variant,
                    "seed": seed,
                    "physics_loss_weight": float(config.physics_loss_weight),
                    "device_type": device_type,
                    "sample_id": sample_id,
                    "group_id": prior["group_id"],
                    "prefix_end_index": prior["prefix_end_index"],
                    "prefix_ordinal": prior["prefix_ordinal"],
                    "positive_rate_point_count": int(components["count"][index]),
                    "prior_centered_log_rate_variance": float(
                        components["prior_variance"][index]
                    ),
                    "learned_centered_log_rate_variance": float(
                        components["learned_variance"][index]
                    ),
                    "learned_vs_prior_centered_log_shape_mse": float(
                        components["shape_mse"][index]
                    ),
                    "saved_prediction": saved_prediction,
                    "recomputed_prediction": recomputed_prediction,
                    "prediction_absolute_error": prediction_error,
                })
        provenance.append({
            "experiment": experiment,
            "variant": variant,
            "seed": seed,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "physics_loss_weight": float(config.physics_loss_weight),
            "prediction_count": len(selected_predictions),
        })
    return prefix_rows, composite_checks, provenance


def _shape_summary_row(
    rows: Sequence[Mapping[str, object]],
    *,
    experiment: str,
    variant: str,
    aggregation_scope: str,
    seed: int | str,
    device_type: str,
    composite_checks: Mapping[str, Mapping[str, float]],
) -> dict:
    weights = [int(row["positive_rate_point_count"]) for row in rows]
    mse = [float(row["learned_vs_prior_centered_log_shape_mse"]) for row in rows]
    total_weight = sum(weights)
    weighted_mse = (
        sum(value * weight for value, weight in zip(mse, weights)) / total_weight
        if total_weight
        else math.nan
    )
    loss_component = None
    loss_difference = None
    if aggregation_scope == "seed_device":
        check = composite_checks[
            f"{experiment}|{variant}|seed={seed}|{device_type}"
        ]
        loss_component = check["composite_loss_physics_component"]
        loss_difference = abs(weighted_mse - loss_component)
    loss_weights = {float(row["physics_loss_weight"]) for row in rows}
    if len(loss_weights) != 1:
        raise ValueError("physics loss weight is inconsistent within summary subset")
    return {
        "experiment": experiment,
        "variant": variant,
        "aggregation_scope": aggregation_scope,
        "seed": seed,
        "device_type": device_type,
        "physics_loss_weight": next(iter(loss_weights)),
        "prefix_count": len(rows),
        "positive_rate_point_count": total_weight,
        "point_weighted_centered_log_shape_mse": weighted_mse,
        "mean_prefix_centered_log_shape_mse": _mean(mse),
        "maximum_prefix_centered_log_shape_mse": max(mse, default=math.nan),
        "mean_prior_centered_log_rate_variance": _mean(
            row["prior_centered_log_rate_variance"] for row in rows
        ),
        "mean_learned_centered_log_rate_variance": _mean(
            row["learned_centered_log_rate_variance"] for row in rows
        ),
        "maximum_prediction_absolute_error": max(
            (float(row["prediction_absolute_error"]) for row in rows),
            default=math.nan,
        ),
        "composite_loss_physics_component": loss_component,
        "mse_vs_composite_loss_absolute_difference": loss_difference,
    }


def summarize_shape_rows(
    rows: Sequence[Mapping[str, object]],
    composite_checks: Mapping[str, Mapping[str, float]],
) -> list[dict]:
    summaries = []
    for experiment, variant in sorted({
        (str(row["experiment"]), str(row["variant"])) for row in rows
    }):
        variant_rows = [
            row
            for row in rows
            if row["experiment"] == experiment and row["variant"] == variant
        ]
        seeds = sorted({int(row["seed"]) for row in variant_rows})
        devices = sorted({str(row["device_type"]) for row in variant_rows})
        for seed in seeds:
            seed_rows = [row for row in variant_rows if int(row["seed"]) == seed]
            for device_type in devices:
                selected = [
                    row for row in seed_rows if row["device_type"] == device_type
                ]
                summaries.append(_shape_summary_row(
                    selected,
                    experiment=experiment,
                    variant=variant,
                    aggregation_scope="seed_device",
                    seed=seed,
                    device_type=device_type,
                    composite_checks=composite_checks,
                ))
            summaries.append(_shape_summary_row(
                seed_rows,
                experiment=experiment,
                variant=variant,
                aggregation_scope="seed_all_devices",
                seed=seed,
                device_type="all",
                composite_checks=composite_checks,
            ))
        for device_type in devices:
            selected = [
                row for row in variant_rows if row["device_type"] == device_type
            ]
            summaries.append(_shape_summary_row(
                selected,
                experiment=experiment,
                variant=variant,
                aggregation_scope="all_seeds_device",
                seed="all",
                device_type=device_type,
                composite_checks=composite_checks,
            ))
        summaries.append(_shape_summary_row(
            variant_rows,
            experiment=experiment,
            variant=variant,
            aggregation_scope="all_seeds_all_devices",
            seed="all",
            device_type="all",
            composite_checks=composite_checks,
        ))
    return summaries


def capacitor_voltage_diagnostic(prior_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in prior_rows:
        if row["device_type"] != "capacitor":
            continue
        voltage = round(float(row["capacitor_applied_voltage_proxy_v"]))
        grouped[voltage].append(row)
    result = []
    for voltage, rows in sorted(grouped.items()):
        result.append({
            "applied_voltage_proxy_v": voltage,
            "prefix_count": len(rows),
            "mean_raw_rate": _mean(row["raw_rate_mean"] for row in rows),
            "mean_normalized_rate": _mean(
                row["normalized_rate_mean"] for row in rows
            ),
            "maximum_centered_log_rate_variance": max(
                float(row["centered_log_rate_variance"]) for row in rows
            ),
            "nondegenerate_prefix_count": sum(
                bool(row["nondegenerate_centered_log_shape"]) for row in rows
            ),
        })
    return result


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def build_report(payload: Mapping[str, object]) -> str:
    prior_summary = [
        row
        for row in payload["prior_summary"]
        if row["scope"] == "all_cached_prefixes"
        and row["prefix_ordinal"] == "all"
    ]
    test_summary = [
        row
        for row in payload["prior_summary"]
        if row["scope"] == "cp_udm_strict_test_prefixes"
        and row["prefix_ordinal"] == "all"
    ]
    shape_summary = [
        row
        for row in payload["shape_mse_summary"]
        if row["aggregation_scope"] == "all_seeds_all_devices"
    ]
    lines = [
        "# Physical-prior identifiability audit",
        "",
        "## Protocol",
        "",
        "This is a no-retraining audit. The stress-only physical rate is computed "
        "from every cached prefix by the existing CP-UDM implementation. Existing "
        "checkpoints are then loaded in evaluation mode to recover their per-time-step "
        "learned `damage_rate`; recomputed endpoint predictions must match the saved "
        "CSV values before a learned-vs-prior shape MSE is accepted.",
        "",
        "For each sequence, the training loss clamps positive rates at `1e-6`, takes "
        "logs, subtracts the valid-point mean separately from learned and physical "
        "rates, and averages their squared residual. The prior is called "
        f"nondegenerate only when its centered-log variance exceeds "
        f"`{SHAPE_VARIANCE_TOLERANCE}`.",
        "",
        "## Input-prior identifiability: all cached prefixes",
        "",
        "| Device | Prefixes | Nondegenerate | Rate | Mean variance | Max variance |",
        "|:--|---:|---:|---:|---:|---:|",
    ]
    for row in prior_summary:
        lines.append(
            f"| {row['device_type']} | {row['prefix_count']} | "
            f"{row['nondegenerate_prefix_count']} | "
            f"{_pct(row['nondegenerate_prefix_rate'])} | "
            f"{float(row['mean_centered_log_rate_variance']):.6g} | "
            f"{float(row['maximum_centered_log_rate_variance']):.6g} |"
        )
    lines.extend([
        "",
        "## Input-prior identifiability: shared strict/main-match test subset",
        "",
        "| Device | Prefixes | Nondegenerate | Rate | Mean variance | Max variance |",
        "|:--|---:|---:|---:|---:|---:|",
    ])
    for row in test_summary:
        lines.append(
            f"| {row['device_type']} | {row['prefix_count']} | "
            f"{row['nondegenerate_prefix_count']} | "
            f"{_pct(row['nondegenerate_prefix_rate'])} | "
            f"{float(row['mean_centered_log_rate_variance']):.6g} | "
            f"{float(row['maximum_centered_log_rate_variance']):.6g} |"
        )
    lines.extend([
        "",
        "The strict and main-match artifacts contain the same 108 test prefixes, so "
        "their input-prior identifiability values are identical. Configuration and "
        "seed do not alter this deterministic input prior.",
        "",
        "## Learned-vs-prior centered-log shape MSE",
        "",
        "| Experiment | Variant | Physics weight | Prefixes | Point-weighted MSE | Max prefix MSE | Max prediction replay error |",
        "|:--|:--|---:|---:|---:|---:|---:|",
    ])
    for row in shape_summary:
        lines.append(
            f"| {row['experiment']} | {row['variant']} | "
            f"{float(row['physics_loss_weight']):.3g} | {row['prefix_count']} | "
            f"{float(row['point_weighted_centered_log_shape_mse']):.6f} | "
            f"{float(row['maximum_prefix_centered_log_shape_mse']):.6f} | "
            f"{float(row['maximum_prediction_absolute_error']):.3g} |"
        )
    lines.extend([
        "",
        "Every seed-device recomputation is also compared directly with the `physics` "
        "component returned by `degradation_model_loss`; the maximum discrepancy is "
        f"`{float(payload['validation']['maximum_mse_vs_composite_loss_difference']):.3g}`.",
        "",
        "## Capacitor limitation",
        "",
        "Within each capacitor prefix, the current prior depends only on the first "
        "stress channel, which is a sequence-constant applied-voltage ratio. The raw "
        "rate amplitude changes between voltage levels, but division by each prefix's "
        "mean and subsequent centered-log comparison remove every positive sequence "
        "scale. Its centered-log target is therefore exactly constant (zero variance) "
        "for every capacitor prefix. On this dataset the capacitor physics term is "
        "equivalent to a constant-rate-shape regularizer on the learned rate; it cannot "
        "encode the relative damage-rate magnitude of 10 V, 12 V, and 14 V conditions.",
        "",
        "This limitation is about identifiability of the implemented training target, "
        "not evidence that voltage has no physical effect. Endpoint CSV fields alone "
        "would not identify learned rate shape; the MSE above is available only because "
        "the saved checkpoints and cached sequences allowed deterministic forward "
        "replay. No target labels or held-out predictions were used to invent a "
        "time-resolved rate.",
        "",
        "Per-prefix prior metrics are in `physics_prior_prefix_metrics.csv`; checkpoint "
        "shape comparisons are in `physics_shape_prefix_mse.csv`; both summary tables "
        "and full provenance are available in the companion CSV and JSON files.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/public/nasa/processed"),
    )
    parser.add_argument(
        "--strict-predictions",
        type=Path,
        default=Path(
            "experiments/unified_degradation/results/cp_udm_strict/predictions.csv"
        ),
    )
    parser.add_argument(
        "--mainmatch-predictions",
        type=Path,
        default=Path(
            "experiments/unified_degradation/results/cp_udm_ablation_mainmatch/"
            "cp_udm_ablation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--strict-checkpoint-dir",
        type=Path,
        default=Path(
            "experiments/unified_degradation/results/cp_udm_strict/checkpoints"
        ),
    )
    parser.add_argument(
        "--mainmatch-checkpoint-dir",
        type=Path,
        default=Path(
            "experiments/unified_degradation/results/cp_udm_ablation_mainmatch/"
            "checkpoints"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/unified_degradation/results/"
            "physics_prior_identifiability_audit"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_metadata, metadata_validation = load_cached_sample_metadata(
        args.processed_dir
    )
    strict_rows, strict_validation = read_selected_predictions(
        args.strict_predictions,
        experiment="cp_udm_strict",
        variant_field="model",
        expected_variants=("cp_udm_net",),
        expected_seeds=EXPECTED_SEEDS,
        cached_metadata=cached_metadata,
    )
    mainmatch_rows, mainmatch_validation = read_selected_predictions(
        args.mainmatch_predictions,
        experiment="cp_udm_ablation_mainmatch",
        variant_field="configuration",
        expected_variants=EXPECTED_MAINMATCH_CONFIGURATIONS,
        expected_seeds=EXPECTED_SEEDS,
        cached_metadata=cached_metadata,
    )
    strict_test_ids = {
        (str(row["device_type"]), str(row["sample_id"])) for row in strict_rows
    }
    mainmatch_test_ids = {
        (str(row["device_type"]), str(row["sample_id"]))
        for row in mainmatch_rows
    }
    if strict_test_ids != mainmatch_test_ids:
        raise ValueError("strict and main-match test sample identities differ")

    datasets, dataset_provenance = load_processed_datasets(args.processed_dir)
    reference_path = args.strict_checkpoint_dir / "cp_udm_net_seed_17.pt"
    reference_checkpoint = torch.load(reference_path, map_location="cpu")
    reference_model = CausalPhysicsUDMNet(reference_checkpoint["config"])
    reference_model.load_state_dict(reference_checkpoint["state_dict"])
    reference_model.eval()

    prior_rows = build_prior_prefix_rows(
        reference_model,
        datasets,
        strict_test_ids,
        mainmatch_test_ids,
    )
    prior_summary = summarize_prior_rows(prior_rows)
    prediction_rows = strict_rows + mainmatch_rows
    shape_rows, composite_checks, checkpoint_provenance = audit_checkpoints(
        _checkpoint_specs(args.strict_checkpoint_dir, args.mainmatch_checkpoint_dir),
        prediction_rows,
        _series_index(datasets),
        prior_rows,
    )
    shape_summary = summarize_shape_rows(shape_rows, composite_checks)
    max_prediction_error = max(
        float(row["prediction_absolute_error"]) for row in shape_rows
    )
    max_loss_difference = max(
        float(check["absolute_difference"])
        for check in composite_checks.values()
    )
    voltage_diagnostic = capacitor_voltage_diagnostic(prior_rows)
    payload = {
        "schema_version": 1,
        "protocol": {
            "name": "no_retraining_physical_prior_identifiability_audit",
            "log_rate_clamp": LOG_RATE_CLAMP,
            "shape_variance_nondegeneracy_tolerance": SHAPE_VARIANCE_TOLERANCE,
            "centered_log_shape": (
                "log(rate_clamped) minus its valid-point within-sequence mean"
            ),
            "shape_mse": (
                "mean squared difference between learned and physical centered-log "
                "rates over points where physics_rate > 0"
            ),
        },
        "availability": {
            "prediction_csv_contains_time_resolved_damage_rate": False,
            "endpoint_fields_are_not_used_as_rate_shape": True,
            "learned_rate_source": (
                "deterministic evaluation-mode forward replay of existing checkpoints"
            ),
            "models_retrained": False,
        },
        "validation": {
            "cached_metadata": metadata_validation,
            "processed_datasets": dataset_provenance,
            "prediction_sources": [strict_validation, mainmatch_validation],
            "strict_and_mainmatch_test_ids_identical": True,
            "test_prefix_count": len(strict_test_ids),
            "checkpoint_count": len(checkpoint_provenance),
            "reference_prior_checkpoint": {
                "path": reference_path.as_posix(),
                "sha256": _sha256(reference_path),
            },
            "maximum_prediction_replay_absolute_error": max_prediction_error,
            "maximum_mse_vs_composite_loss_difference": max_loss_difference,
            "composite_loss_checks": composite_checks,
        },
        "interpretation_boundary": {
            "scale_invariance": (
                "Within-sequence centering removes every positive sequence-level "
                "multiplicative rate scale when clamp activation is unchanged."
            ),
            "capacitor": (
                "All capacitor prior shapes are constant in time; the loss is a "
                "constant-rate-shape regularizer and cannot express cross-voltage "
                "damage-rate magnitude."
            ),
            "not_a_calibrated_failure_law": True,
        },
        "capacitor_voltage_amplitude_diagnostic": voltage_diagnostic,
        "prior_summary": prior_summary,
        "shape_mse_summary": shape_summary,
        "checkpoint_provenance": checkpoint_provenance,
        "counts": {
            "cached_prefix_rows": len(prior_rows),
            "checkpoint_prefix_rows": len(shape_rows),
            "prior_summary_rows": len(prior_summary),
            "shape_summary_rows": len(shape_summary),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.output_dir / "physics_prior_prefix_metrics.csv",
        PRIOR_PREFIX_FIELDS,
        prior_rows,
    )
    _write_csv(
        args.output_dir / "physics_prior_summary.csv",
        PRIOR_SUMMARY_FIELDS,
        prior_summary,
    )
    _write_csv(
        args.output_dir / "physics_shape_prefix_mse.csv",
        SHAPE_PREFIX_FIELDS,
        shape_rows,
    )
    _write_csv(
        args.output_dir / "physics_shape_mse_summary.csv",
        SHAPE_SUMMARY_FIELDS,
        shape_summary,
    )
    with (args.output_dir / "physics_prior_identifiability_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    (args.output_dir / "physics_prior_identifiability_audit.md").write_text(
        build_report(payload), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "cached_prefix_count": len(prior_rows),
        "checkpoint_count": len(checkpoint_provenance),
        "checkpoint_prefix_count": len(shape_rows),
        "maximum_prediction_replay_absolute_error": max_prediction_error,
        "maximum_mse_vs_composite_loss_difference": max_loss_difference,
    }, indent=2))


if __name__ == "__main__":
    main()
