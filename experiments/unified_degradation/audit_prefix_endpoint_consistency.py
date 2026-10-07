"""Audit endpoint consistency across independently constructed run prefixes.

This is deliberately different from the within-prefix trajectory monotonicity
metric reported by ``evaluate_model``.  Here, rows sharing an original NASA run
are ordered by the truncation endpoint encoded in ``sample_id`` and the final
``prediction`` values from two adjacent prefixes are compared.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EXPECTED_SEEDS = (17, 42, 73)
EXPECTED_MAINMATCH_CONFIGURATIONS = (
    "full",
    "no_physics",
    "no_forecast",
    "no_ranking_endpoint",
    "no_alignment",
)
PREFIX_SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<group_id>.+)::prefix-(?P<prefix_end_index>[0-9]+)$"
)
PAIR_FIELDS = (
    "experiment",
    "variant",
    "seed",
    "device_type",
    "group_id",
    "earlier_sample_id",
    "later_sample_id",
    "earlier_prefix_end_index",
    "later_prefix_end_index",
    "earlier_target",
    "later_target",
    "earlier_prediction",
    "later_prediction",
    "signed_prediction_change",
    "decrease_magnitude",
    "is_decrease",
)
SUMMARY_FIELDS = (
    "experiment",
    "variant",
    "aggregation_scope",
    "seed",
    "device_type",
    "group_seed_count",
    "adjacent_pair_count",
    "decrease_pair_count",
    "adjacent_endpoint_decrease_rate",
    "group_seed_with_decrease_count",
    "group_seed_decrease_rate",
    "maximum_adjacent_endpoint_decrease",
    "mean_decrease_magnitude_when_positive",
)


def parse_prefix_sample_id(sample_id: str) -> tuple[str, int]:
    """Recover the exact original group and truncation endpoint from an ID."""
    match = PREFIX_SAMPLE_ID_PATTERN.fullmatch(str(sample_id).strip())
    if match is None:
        raise ValueError(
            "sample_id does not match the loader contract "
            f"'<group>::prefix-<integer>': {sample_id!r}"
        )
    group_id = match.group("group_id")
    prefix_end_index = int(match.group("prefix_end_index"))
    if prefix_end_index <= 0:
        raise ValueError(f"prefix endpoint must be positive: {sample_id!r}")
    return group_id, prefix_end_index


def _finite_float(value: object, *, field: str, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} in {context}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field} in {context}: {value!r}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _CachedDataProxy:
    """Metadata-only stand-in for local experiment dataclasses in pickle files."""


class MetadataOnlyUnpickler(pickle.Unpickler):
    _PROXIED_CLASSES = {
        ("experiments.unified_degradation.nasa_loaders", "LoadedDataset"),
        ("app.services.unified_degradation_data", "DegradationSeries"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self._PROXIED_CLASSES:
            return _CachedDataProxy
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("numpy._core."):
                legacy_module = "numpy.core." + module.removeprefix("numpy._core.")
                return super().find_class(legacy_module, name)
            raise


def load_cached_sample_metadata(processed_dir: Path) -> tuple[dict, dict]:
    """Load the experiment cache and validate its explicit group-ID contract."""
    metadata: dict[tuple[str, str], dict] = {}
    cache_files = []
    for device_type in ("capacitor", "igbt", "mosfet"):
        path = processed_dir / f"{device_type}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"required processed dataset cache is missing: {path}")
        with path.open("rb") as handle:
            dataset = MetadataOnlyUnpickler(handle).load()
        if len(dataset.series) != len(dataset.groups):
            raise ValueError(f"series/group length mismatch in {path}")
        for item, cached_group in zip(dataset.series, dataset.groups):
            sample_id = str(item.sample_id)
            parsed_group, prefix_end_index = parse_prefix_sample_id(sample_id)
            if parsed_group != str(cached_group):
                raise ValueError(
                    f"sample ID group {parsed_group!r} disagrees with cached group "
                    f"{cached_group!r} in {path}"
                )
            if str(item.device_type) != device_type:
                raise ValueError(
                    f"cached device {item.device_type!r} disagrees with {device_type!r}"
                )
            key = (device_type, sample_id)
            if key in metadata:
                raise ValueError(f"duplicate cached sample metadata: {key}")
            metadata[key] = {
                "group_id": parsed_group,
                "prefix_end_index": prefix_end_index,
                "target": float(item.target),
            }
        cache_files.append({
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "series_count": len(dataset.series),
            "group_count": len(set(map(str, dataset.groups))),
        })
    return metadata, {
        "method": (
            "anchored sample_id parsing, cross-checked against each cached "
            "DegradationSeries and its parallel dataset.groups entry"
        ),
        "cache_files": cache_files,
        "cached_sample_count": len(metadata),
    }


def read_selected_predictions(
    path: Path,
    *,
    experiment: str,
    variant_field: str,
    expected_variants: Sequence[str],
    expected_seeds: Sequence[int],
    cached_metadata: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[list[dict], dict]:
    """Read and fully validate one fixed-protocol prediction artifact."""
    required = {
        variant_field,
        "seed",
        "device_type",
        "sample_id",
        "target",
        "prediction",
    }
    rows: list[dict] = []
    total_rows = 0
    ignored_rows = 0
    expected_variants = tuple(expected_variants)
    expected_seed_set = set(expected_seeds)
    seen_keys = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for line_number, source_row in enumerate(reader, start=2):
            total_rows += 1
            variant = str(source_row[variant_field]).strip()
            if variant not in expected_variants:
                ignored_rows += 1
                continue
            context = f"{path}:{line_number}"
            try:
                seed = int(source_row["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid seed in {context}: {source_row['seed']!r}") from exc
            if seed not in expected_seed_set:
                raise ValueError(
                    f"unexpected seed {seed} for selected variant {variant!r} in {context}"
                )
            device_type = str(source_row["device_type"]).strip()
            sample_id = str(source_row["sample_id"]).strip()
            group_id, prefix_end_index = parse_prefix_sample_id(sample_id)
            cache_key = (device_type, sample_id)
            cached = cached_metadata.get(cache_key)
            if cached is None:
                raise ValueError(
                    f"prediction sample is absent from processed metadata cache: {cache_key}"
                )
            if (
                str(cached["group_id"]) != group_id
                or int(cached["prefix_end_index"]) != prefix_end_index
            ):
                raise ValueError(f"parsed prefix metadata disagrees with cache for {cache_key}")
            target = _finite_float(source_row["target"], field="target", context=context)
            cached_target = float(cached["target"])
            if not math.isclose(target, cached_target, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"prediction target {target} disagrees with cached target "
                    f"{cached_target} for {cache_key}"
                )
            prediction = _finite_float(
                source_row["prediction"], field="prediction", context=context
            )
            unique_key = (variant, seed, device_type, sample_id)
            if unique_key in seen_keys:
                raise ValueError(f"duplicate selected prediction row: {unique_key}")
            seen_keys.add(unique_key)
            rows.append({
                "experiment": experiment,
                "variant": variant,
                "seed": seed,
                "device_type": device_type,
                "sample_id": sample_id,
                "group_id": group_id,
                "prefix_end_index": prefix_end_index,
                "target": target,
                "prediction": prediction,
            })

    combinations = {(row["variant"], row["seed"]) for row in rows}
    expected_combinations = {
        (variant, seed) for variant in expected_variants for seed in expected_seeds
    }
    if combinations != expected_combinations:
        raise ValueError(
            f"selected protocol combinations differ from expectation in {path}: "
            f"missing={sorted(expected_combinations - combinations)}, "
            f"unexpected={sorted(combinations - expected_combinations)}"
        )

    reference_samples = None
    sample_count_by_combination = {}
    for combination in sorted(expected_combinations):
        sample_map = {
            (row["device_type"], row["sample_id"]): row["target"]
            for row in rows
            if (row["variant"], row["seed"]) == combination
        }
        if reference_samples is None:
            reference_samples = sample_map
        elif sample_map != reference_samples:
            raise ValueError(
                f"sample identities or targets are not paired for {combination} in {path}"
            )
        sample_count_by_combination[f"{combination[0]}|seed={combination[1]}"] = len(
            sample_map
        )

    return rows, {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "columns": list(reader.fieldnames or ()),
        "total_row_count": total_rows,
        "selected_row_count": len(rows),
        "ignored_row_count": ignored_rows,
        "variant_field": variant_field,
        "expected_variants": list(expected_variants),
        "expected_seeds": list(expected_seeds),
        "sample_count_by_variant_seed": sample_count_by_combination,
        "all_variant_seed_sample_sets_identical": True,
        "all_selected_rows_matched_cache": True,
    }


def build_adjacent_prefix_pairs(
    rows: Iterable[Mapping[str, object]], *, decrease_tolerance: float = 0.0
) -> list[dict]:
    """Compare final predictions for adjacent truncations of each original run."""
    if decrease_tolerance < 0:
        raise ValueError("decrease_tolerance must be non-negative")
    grouped = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["variant"]),
            int(row["seed"]),
            str(row["device_type"]),
            str(row["group_id"]),
        )
        grouped[key].append(dict(row))

    pairs = []
    for key, group_rows in sorted(grouped.items()):
        group_rows.sort(key=lambda row: int(row["prefix_end_index"]))
        endpoints = [int(row["prefix_end_index"]) for row in group_rows]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError(f"duplicate parsed prefix endpoint for group key {key}: {endpoints}")
        if len(group_rows) < 2:
            raise ValueError(f"fewer than two prefixes for group key {key}")
        targets = [float(row["target"]) for row in group_rows]
        if any(later <= earlier for earlier, later in zip(targets, targets[1:])):
            raise ValueError(
                f"targets do not increase with truncation endpoint for group key {key}: "
                f"{targets}"
            )
        for earlier, later in zip(group_rows, group_rows[1:]):
            earlier_prediction = float(earlier["prediction"])
            later_prediction = float(later["prediction"])
            signed_change = later_prediction - earlier_prediction
            decrease_magnitude = max(-signed_change, 0.0)
            pairs.append({
                "experiment": key[0],
                "variant": key[1],
                "seed": key[2],
                "device_type": key[3],
                "group_id": key[4],
                "earlier_sample_id": str(earlier["sample_id"]),
                "later_sample_id": str(later["sample_id"]),
                "earlier_prefix_end_index": int(earlier["prefix_end_index"]),
                "later_prefix_end_index": int(later["prefix_end_index"]),
                "earlier_target": float(earlier["target"]),
                "later_target": float(later["target"]),
                "earlier_prediction": earlier_prediction,
                "later_prediction": later_prediction,
                "signed_prediction_change": signed_change,
                "decrease_magnitude": decrease_magnitude,
                "is_decrease": decrease_magnitude > decrease_tolerance,
            })
    return pairs


def _summarize_subset(
    pairs: Sequence[Mapping[str, object]],
    *,
    experiment: str,
    variant: str,
    aggregation_scope: str,
    seed: int | str,
    device_type: str,
) -> dict:
    positive_decreases = [
        float(row["decrease_magnitude"])
        for row in pairs
        if float(row["decrease_magnitude"]) > 0.0
    ]
    classified_decreases = [row for row in pairs if row["is_decrease"]]
    groups = {
        (int(row["seed"]), str(row["device_type"]), str(row["group_id"]))
        for row in pairs
    }
    groups_with_decrease = {
        (int(row["seed"]), str(row["device_type"]), str(row["group_id"]))
        for row in pairs
        if row["is_decrease"]
    }
    pair_count = len(pairs)
    decrease_count = len(classified_decreases)
    return {
        "experiment": experiment,
        "variant": variant,
        "aggregation_scope": aggregation_scope,
        "seed": seed,
        "device_type": device_type,
        "group_seed_count": len(groups),
        "adjacent_pair_count": pair_count,
        "decrease_pair_count": decrease_count,
        "adjacent_endpoint_decrease_rate": decrease_count / pair_count if pair_count else 0.0,
        "group_seed_with_decrease_count": len(groups_with_decrease),
        "group_seed_decrease_rate": (
            len(groups_with_decrease) / len(groups) if groups else 0.0
        ),
        "maximum_adjacent_endpoint_decrease": max(positive_decreases, default=0.0),
        "mean_decrease_magnitude_when_positive": (
            sum(positive_decreases) / len(positive_decreases)
            if positive_decreases
            else 0.0
        ),
    }


def summarize_pairs(pairs: Sequence[Mapping[str, object]]) -> list[dict]:
    """Return seed/device and pooled summaries without averaging ratios."""
    summaries = []
    experiment_variants = sorted({
        (str(row["experiment"]), str(row["variant"])) for row in pairs
    })
    for experiment, variant in experiment_variants:
        selected = [
            row
            for row in pairs
            if row["experiment"] == experiment and row["variant"] == variant
        ]
        seeds = sorted({int(row["seed"]) for row in selected})
        devices = sorted({str(row["device_type"]) for row in selected})
        for seed in seeds:
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            for device in devices:
                subset = [
                    row for row in seed_rows if str(row["device_type"]) == device
                ]
                summaries.append(_summarize_subset(
                    subset,
                    experiment=experiment,
                    variant=variant,
                    aggregation_scope="seed_device",
                    seed=seed,
                    device_type=device,
                ))
            summaries.append(_summarize_subset(
                seed_rows,
                experiment=experiment,
                variant=variant,
                aggregation_scope="seed_all_devices",
                seed=seed,
                device_type="all",
            ))
        for device in devices:
            subset = [row for row in selected if str(row["device_type"]) == device]
            summaries.append(_summarize_subset(
                subset,
                experiment=experiment,
                variant=variant,
                aggregation_scope="all_seeds_device",
                seed="all",
                device_type=device,
            ))
        summaries.append(_summarize_subset(
            selected,
            experiment=experiment,
            variant=variant,
            aggregation_scope="all_seeds_all_devices",
            seed="all",
            device_type="all",
        ))
    return summaries


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_rate(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def build_markdown_report(payload: Mapping[str, object]) -> str:
    summaries = list(payload["summaries"])
    strict_seed_rows = [
        row
        for row in summaries
        if row["experiment"] == "cp_udm_strict"
        and row["variant"] == "cp_udm_net"
        and row["aggregation_scope"] == "seed_all_devices"
    ]
    mainmatch_rows = [
        row
        for row in summaries
        if row["experiment"] == "cp_udm_ablation_mainmatch"
        and row["aggregation_scope"] == "all_seeds_all_devices"
    ]
    pooled_device_rows = [
        row
        for row in summaries
        if row["aggregation_scope"] == "all_seeds_device"
    ]
    lines = [
        "# Cross-prefix endpoint consistency audit",
        "",
        "## Scope and estimand",
        "",
        "This audit compares the saved final `prediction` for adjacent truncation "
        "prefixes of the same original NASA run. The original run ID and truncation "
        "endpoint are recovered only from the anchored loader-defined "
        "`<group>::prefix-<integer>` sample ID, then cross-checked against the local "
        "processed dataset cache. No model is trained or re-evaluated.",
        "",
        "The adjacent endpoint decrease rate is `count(p[i+1] < p[i]) / "
        "count(adjacent prefix pairs)`. The maximum decrease is "
        "`max(p[i] - p[i+1], 0)` over the same pairs. The configured comparison "
        f"tolerance is `{payload['metric_definition']['decrease_tolerance']}`.",
        "",
        "This estimand is not the within-prefix trajectory monotonic violation rate "
        "(MVR). MVR checks consecutive latent trajectory states produced during one "
        "forward pass. This audit instead compares separate forward passes made from "
        "different truncations of one raw run.",
        "",
        "## CP-UDM strict result by seed",
        "",
        "| Seed | Group-seeds | Adjacent pairs | Decreases | Decrease rate | Max decrease |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strict_seed_rows:
        lines.append(
            f"| {row['seed']} | {row['group_seed_count']} | "
            f"{row['adjacent_pair_count']} | {row['decrease_pair_count']} | "
            f"{_format_rate(row['adjacent_endpoint_decrease_rate'])} | "
            f"{float(row['maximum_adjacent_endpoint_decrease']):.6f} |"
        )
    lines.extend([
        "",
        "## Main-match ablation result pooled across seeds",
        "",
        "| Configuration | Group-seeds | Adjacent pairs | Decreases | Decrease rate | Max decrease |",
        "|:--|---:|---:|---:|---:|---:|",
    ])
    for row in mainmatch_rows:
        lines.append(
            f"| {row['variant']} | {row['group_seed_count']} | "
            f"{row['adjacent_pair_count']} | {row['decrease_pair_count']} | "
            f"{_format_rate(row['adjacent_endpoint_decrease_rate'])} | "
            f"{float(row['maximum_adjacent_endpoint_decrease']):.6f} |"
        )
    lines.extend([
        "",
        "## Device-level pooled diagnostics",
        "",
        "| Experiment | Variant | Device | Adjacent pairs | Decreases | Decrease rate | Max decrease |",
        "|:--|:--|:--|---:|---:|---:|---:|",
    ])
    for row in pooled_device_rows:
        lines.append(
            f"| {row['experiment']} | {row['variant']} | {row['device_type']} | "
            f"{row['adjacent_pair_count']} | {row['decrease_pair_count']} | "
            f"{_format_rate(row['adjacent_endpoint_decrease_rate'])} | "
            f"{float(row['maximum_adjacent_endpoint_decrease']):.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The positive-rate integration guarantees nondecreasing latent states only "
        "inside one fixed observed prefix. It does not guarantee monotonic final "
        "predictions across independently truncated prefixes; causal attention also "
        "only blocks future-token interactions within one fixed prefix. The saved final "
        "prediction is an equal-weight blend of a direct stage head and the dynamic "
        "endpoint. In addition, response scale uses the median of the entire observed "
        "prefix, time features and local increments are normalized by that prefix's "
        "endpoint duration, and every prefix is independently resampled. Extending a "
        "prefix therefore changes the representation of earlier observations. Any "
        "cross-prefix decrease reported here is compatible with zero within-prefix MVR "
        "and must not be described as a violation of the structural integration "
        "guarantee.",
        "",
        "The detailed adjacent comparisons are in `prefix_endpoint_pairs.csv`; all "
        "aggregation scopes are in `prefix_endpoint_summary.csv`, and machine-readable "
        "validation details are in `prefix_endpoint_audit.json`.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--processed-dir",
        type=Path,
        default=Path("data/public/nasa/processed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/unified_degradation/results/"
            "prefix_endpoint_consistency_audit"
        ),
    )
    parser.add_argument("--decrease-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_metadata, cache_validation = load_cached_sample_metadata(args.processed_dir)
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

    strict_sample_set = {
        (row["device_type"], row["sample_id"], row["target"]) for row in strict_rows
    }
    mainmatch_sample_set = {
        (row["device_type"], row["sample_id"], row["target"])
        for row in mainmatch_rows
    }
    if strict_sample_set != mainmatch_sample_set:
        raise ValueError("strict and main-match artifacts do not use the same test samples")

    all_rows = strict_rows + mainmatch_rows
    pairs = build_adjacent_prefix_pairs(
        all_rows, decrease_tolerance=args.decrease_tolerance
    )
    summaries = summarize_pairs(pairs)
    prefix_count_distribution = Counter()
    grouped_prefixes = defaultdict(set)
    for row in all_rows:
        grouped_prefixes[
            (
                row["experiment"],
                row["variant"],
                row["seed"],
                row["device_type"],
                row["group_id"],
            )
        ].add(row["prefix_end_index"])
    for endpoints in grouped_prefixes.values():
        prefix_count_distribution[len(endpoints)] += 1

    payload = {
        "schema_version": 1,
        "audited_field": "prediction",
        "excluded_from_estimand": "dynamic_endpoint and within-prefix trajectory",
        "metric_definition": {
            "ordering": "integer truncation endpoint parsed from sample_id",
            "comparison_unit": "adjacent prefixes within a variant-seed-device-group",
            "decrease": "later_prediction < earlier_prediction - decrease_tolerance",
            "decrease_tolerance": args.decrease_tolerance,
            "adjacent_endpoint_decrease_rate": "decrease_pair_count / adjacent_pair_count",
            "maximum_adjacent_endpoint_decrease": "max(earlier_prediction - later_prediction, 0)",
        },
        "scope_boundary": {
            "different_from_mvr": (
                "MVR compares consecutive latent trajectory states inside one fixed "
                "prefix; this audit compares final predictions from separate prefixes."
            ),
            "no_cross_prefix_structural_guarantee": (
                "The final prediction includes a direct stage head, and prefix-wide "
                "response scaling, endpoint-normalized time features, and independent "
                "prefix resampling change the input representation across truncations."
            ),
        },
        "validation": {
            "cached_metadata": cache_validation,
            "sources": [strict_validation, mainmatch_validation],
            "strict_and_mainmatch_sample_sets_identical": True,
            "prefix_count_distribution_per_variant_seed_group": dict(
                sorted(prefix_count_distribution.items())
            ),
        },
        "summary_fields": list(SUMMARY_FIELDS),
        "summaries": summaries,
        "largest_decreases": sorted(
            (row for row in pairs if row["is_decrease"]),
            key=lambda row: float(row["decrease_magnitude"]),
            reverse=True,
        )[:20],
        "pair_count": len(pairs),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "prefix_endpoint_pairs.csv", PAIR_FIELDS, pairs)
    _write_csv(
        args.output_dir / "prefix_endpoint_summary.csv", SUMMARY_FIELDS, summaries
    )
    with (args.output_dir / "prefix_endpoint_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    (args.output_dir / "prefix_endpoint_audit.md").write_text(
        build_markdown_report(payload), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "selected_prediction_count": len(all_rows),
        "adjacent_pair_count": len(pairs),
        "decrease_pair_count": sum(bool(row["is_decrease"]) for row in pairs),
    }, indent=2))


if __name__ == "__main__":
    main()
