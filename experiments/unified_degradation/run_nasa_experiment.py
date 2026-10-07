"""Run the reproducible three-device NASA degradation experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from experiments.unified_degradation.experiment_runner import (
    DatasetSplit,
    StressLevelSplit,
    audit_nasa_stress_levels,
    build_loaders,
    evaluate_stress_level_holdout,
    evaluate_model,
    group_conformal_metrics,
    leave_one_stress_level_out_splits,
    model_config_for,
    conformal_metrics,
    run_duration_only_baseline,
    run_random_forest_baseline,
    run_response_only_baseline,
    save_experiment_outputs,
    split_groups,
    train_model,
)
from experiments.unified_degradation.nasa_loaders import (
    load_capacitor_dataset,
    load_igbt_dataset,
    load_mosfet_dataset,
)
from app.services.unified_degradation_model import build_ablation_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/public/nasa")
    parser.add_argument("--output-dir", default="experiments/unified_degradation/results/nasa_real")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="17,42,73")
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--rebuild-cache", action="store_true")
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument(
        "--stress-loso",
        action="store_true",
        help="run only the group-level leave-one-stress-level-out protocol",
    )
    parser.add_argument(
        "--stress-loso-output-dir",
        default="experiments/unified_degradation/results/nasa_stress_loso",
    )
    protocol.add_argument(
        "--cp-ablation",
        action="store_true",
        help="run only the paired confirmatory CP-UDM ablation protocol",
    )
    parser.add_argument(
        "--cp-ablation-output-dir",
        default=(
            "experiments/unified_degradation/results/"
            "cp_udm_ablation_confirmatory"
        ),
    )
    parser.add_argument("--cp-ablation-patience", type=int, default=15)
    return parser.parse_args()


CP_ABLATION_CONFIG_NAMES = (
    "full",
    "no_physics",
    "no_forecast",
    "no_ranking_endpoint",
    "no_alignment",
)
CP_ABLATION_SCOPES = ("overall", "capacitor", "igbt", "mosfet")
CP_ABLATION_METRICS = (
    "mae",
    "spearman",
    "trajectory_monotonic_violation_rate",
    "physics_rate_spearman",
)
CP_ABLATION_SPLIT_SEED = 42


def build_confirmatory_cp_ablation_configs(base):
    """Build the five paired CP-UDM configurations without mutating ``base``."""
    if not base.use_latent_dynamics:
        raise ValueError("confirmatory CP-UDM ablation requires use_latent_dynamics=True")
    return {
        "full": replace(base),
        "no_physics": replace(base, physics_loss_weight=0.0),
        "no_forecast": replace(base, forecast_loss_weight=0.0),
        "no_ranking_endpoint": replace(
            base,
            ranking_loss_weight=0.0,
            endpoint_consistency_weight=0.0,
        ),
        "no_alignment": replace(
            base,
            alignment_loss_weight=0.0,
            domain_loss_weight=0.0,
            domain_adversarial_strength=0.0,
        ),
    }


def add_prediction_metadata(rows, model_name: str, seed: int):
    return [{"model": model_name, "seed": seed, **row} for row in rows]


def load_cached_dataset(cache_path: Path, loader, rebuild: bool):
    if cache_path.exists() and not rebuild:
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    dataset = loader()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def generate_summary(output_dir: Path, metrics: dict, datasets: dict, splits: dict):
    rows = []
    for experiment, seed_results in metrics.items():
        for seed, result in seed_results.items():
            overall = result["overall"]
            rows.append((
                experiment,
                seed,
                overall["mae"],
                overall["rmse"],
                overall["r2"],
                overall["spearman"],
            ))
    rows.sort(key=lambda row: row[2])
    lines = [
        "# NASA 真实加速老化数据统一退化实验结果",
        "",
        "## 数据与划分",
        "",
    ]
    for name, dataset in datasets.items():
        split = splits[name]
        lines.append(
            f"- **{name}**：{len(set(dataset.groups))} 个独立运行/器件组，"
            f"{len(dataset.series)} 条固定长度前缀序列；训练/验证/测试组数为 "
            f"{len(split.train_groups)}/{len(split.validation_groups)}/{len(split.test_groups)}。"
        )
    lines.extend([
        "",
        "所有窗口均按原始器件或运行文件分组划分，同一原始曲线不会同时进入训练集和测试集。"
        "每个前缀被重采样到相同长度，模型不能通过序列点数推断标签。",
        "",
        "## 总体结果",
        "",
        "| 模型 | 随机种子 | MAE | RMSE | R² | Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, seed, mae, rmse, r2, rho in rows:
        lines.append(f"| {name} | {seed} | {mae:.4f} | {rmse:.4f} | {r2:.4f} | {rho:.4f} |")
    lines.extend([
        "",
        "## 结果解释边界",
        "",
        "本实验预测的是各公开试验记录中的相对退化阶段，不直接等同于真实现场剩余寿命或失效概率。"
        "通用性的证据来自跨器件联合训练、分器件测试、器件专用模型对照和模块消融，"
        "而不是把不同器件的物理参数强行解释为同一个量。",
        "",
    ])
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(output_dir: Path, metrics: dict):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    names, values = [], []
    for experiment, seed_results in metrics.items():
        maes = [result["overall"]["mae"] for result in seed_results.values()]
        names.append(experiment)
        values.append(float(np.mean(maes)))
    order = np.argsort(values)
    names = [names[index] for index in order]
    values = [values[index] for index in order]
    figure, axis = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    axis.barh(names, values, color="#2E74B5")
    axis.set_xlabel("Test MAE (lower is better)")
    axis.set_title("NASA accelerated-aging experiment")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "model_mae_comparison.png", dpi=180)
    plt.close(figure)


def _stress_fold_id(split: StressLevelSplit) -> str:
    physical_level = f"{split.held_out_physical_level:.3f}".rstrip("0").rstrip(".")
    return (
        f"{split.device_type}_{split.stress_name}_{physical_level}{split.physical_unit}"
        .lower()
        .replace(".", "p")
    )


def _annotate_stress_predictions(
    rows: list[dict],
    split: StressLevelSplit,
    fold_id: str,
    model_name: str,
    seed: int,
) -> list[dict]:
    return [
        {
            "fold_id": fold_id,
            "model": model_name,
            "seed": seed,
            "stress_name": split.stress_name,
            "held_out_level": split.held_out_level,
            "held_out_physical_level": split.held_out_physical_level,
            "physical_unit": split.physical_unit,
            **row,
        }
        for row in rows
    ]


def _add_stress_metadata(metrics: dict, split: StressLevelSplit) -> dict:
    result = dict(metrics)
    result["stress_holdout"] = {
        "device_type": split.device_type,
        "stress_index": split.stress_index,
        "stress_name": split.stress_name,
        "held_out_level": split.held_out_level,
        "held_out_physical_level": split.held_out_physical_level,
        "physical_unit": split.physical_unit,
        "train_group_count": len(split.train_groups),
        "validation_group_count": len(split.validation_groups),
        "calibration_group_count": len(split.calibration_groups),
        "test_group_count": len(split.test_groups),
        "excluded_group_count": len(split.excluded_groups),
    }
    return result


def _format_metric(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.4f}" if np.isfinite(numeric) else "NA"


def _sample_statistics(values) -> dict:
    finite = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite.append(numeric)
    return {
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "sample_sd": (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        ),
        "n": len(finite),
    }


def _cp_ablation_scope(result: dict, scope: str) -> dict:
    if scope == "overall":
        return result["overall"]
    return result["devices"][scope]


def summarize_cp_ablation_metrics(metrics: dict) -> tuple[dict, dict]:
    """Compute seed aggregates and paired ablation-minus-full differences."""
    aggregate = {}
    for configuration in CP_ABLATION_CONFIG_NAMES:
        seed_results = metrics.get(configuration, {})
        if not seed_results:
            continue
        aggregate[configuration] = {}
        for scope in CP_ABLATION_SCOPES:
            aggregate[configuration][scope] = {
                metric: _sample_statistics(
                    _cp_ablation_scope(result, scope).get(metric, float("nan"))
                    for result in seed_results.values()
                )
                for metric in CP_ABLATION_METRICS
            }

    paired = {}
    full_results = metrics.get("full", {})
    for configuration in CP_ABLATION_CONFIG_NAMES[1:]:
        seed_results = metrics.get(configuration, {})
        common_seeds = sorted(
            set(full_results) & set(seed_results),
            key=lambda value: int(value),
        )
        if not common_seeds:
            continue
        paired[configuration] = {}
        for scope in CP_ABLATION_SCOPES:
            paired[configuration][scope] = {}
            for metric in CP_ABLATION_METRICS:
                by_seed = {
                    seed: float(
                        _cp_ablation_scope(seed_results[seed], scope)[metric]
                        - _cp_ablation_scope(full_results[seed], scope)[metric]
                    )
                    for seed in common_seeds
                }
                paired[configuration][scope][metric] = {
                    **_sample_statistics(by_seed.values()),
                    "by_seed": by_seed,
                }
    return aggregate, paired


def _format_mean_sd(statistics: dict) -> str:
    mean = _format_metric(statistics.get("mean"))
    sample_sd = _format_metric(statistics.get("sample_sd"))
    return f"{mean} +/- {sample_sd}"


def generate_cp_ablation_summary(output_dir: Path, payload: dict):
    protocol = payload["protocol"]
    aggregate = payload.get("aggregate_statistics", {})
    paired = payload.get("paired_deltas_vs_full", {})
    lines = [
        "# CP-UDM configuration-based paired confirmatory ablation",
        "",
        "## Protocol",
        "",
        f"- Group split seed: {protocol['split_seed']} (one split reused by every configuration and training seed).",
        f"- Training seeds: {', '.join(str(seed) for seed in protocol['training_seeds'])}.",
        f"- Maximum epochs / early-stopping patience: {protocol['epochs_requested']} / {protocol['early_stopping_patience']}.",
        "- All configurations jointly train capacitor, IGBT, and MOSFET data. For a given training seed, model initialization and data-loader randomness are reset to that same seed for every configuration.",
        "",
        "## Mean +/- sample SD across training seeds",
        "",
    ]
    for scope in CP_ABLATION_SCOPES:
        lines.extend([
            f"### {scope}",
            "",
            "| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |",
            "|---|---:|---:|---:|---:|",
        ])
        for configuration in CP_ABLATION_CONFIG_NAMES:
            if configuration not in aggregate:
                continue
            values = aggregate[configuration][scope]
            lines.append(
                f"| {configuration} | {_format_mean_sd(values['mae'])} | "
                f"{_format_mean_sd(values['spearman'])} | "
                f"{_format_mean_sd(values['trajectory_monotonic_violation_rate'])} | "
                f"{_format_mean_sd(values['physics_rate_spearman'])} |"
            )
        lines.append("")

    lines.extend([
        "## Paired deltas relative to full",
        "",
        "Every delta below is `ablation - full` for the same training seed. Therefore a positive MAE delta means that removing the component degraded accuracy; the other metrics retain their natural signed interpretation.",
        "",
        "| Configuration | Scope | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for configuration in CP_ABLATION_CONFIG_NAMES[1:]:
        if configuration not in paired:
            continue
        for scope in CP_ABLATION_SCOPES:
            values = paired[configuration][scope]
            lines.append(
                f"| {configuration} | {scope} | {_format_mean_sd(values['mae'])} | "
                f"{_format_mean_sd(values['spearman'])} | "
                f"{_format_mean_sd(values['trajectory_monotonic_violation_rate'])} | "
                f"{_format_mean_sd(values['physics_rate_spearman'])} |"
            )
    lines.extend([
        "",
        "## Per-seed paired deltas",
        "",
        "| Configuration | Scope | Seed | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for configuration in CP_ABLATION_CONFIG_NAMES[1:]:
        if configuration not in paired:
            continue
        for scope in CP_ABLATION_SCOPES:
            values = paired[configuration][scope]
            seeds = values["mae"]["by_seed"]
            for seed in sorted(seeds, key=int):
                lines.append(
                    f"| {configuration} | {scope} | {seed} | "
                    f"{_format_metric(values['mae']['by_seed'][seed])} | "
                    f"{_format_metric(values['spearman']['by_seed'][seed])} | "
                    f"{_format_metric(values['trajectory_monotonic_violation_rate']['by_seed'][seed])} | "
                    f"{_format_metric(values['physics_rate_spearman']['by_seed'][seed])} |"
                )
    lines.extend([
        "",
        "## Metric interpretation boundary",
        "",
        "`physics_rate_spearman` is a cross-sample pooled diagnostic: within each device it pools valid time points from all test prefixes before computing Spearman correlation, while the overall value is the finite mean of the three device-level correlations. It is not the same estimand as the training physics loss, which compares the learned and prior rate shapes after centering their log rates separately within each sequence. Consequently this diagnostic must not be interpreted as a direct test-loss surrogate.",
        "",
    ])
    report_path = output_dir / "cp_udm_ablation_report.md"
    temporary_path = output_dir / ".cp_udm_ablation_report.md.tmp"
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    temporary_path.replace(report_path)


def _write_json_atomic(path: Path, value):
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_cp_ablation_outputs(
    output_dir: Path,
    payload: dict,
    predictions: list[dict],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate, paired = summarize_cp_ablation_metrics(payload["metrics"])
    payload["aggregate_statistics"] = aggregate
    payload["paired_deltas_vs_full"] = paired

    metrics_payload = {
        key: value for key, value in payload.items() if key != "histories"
    }
    _write_json_atomic(
        output_dir / "cp_udm_ablation_metrics.json", metrics_payload
    )
    _write_json_atomic(
        output_dir / "cp_udm_ablation_histories.json",
        payload.get("histories", {}),
    )
    if predictions:
        predictions_path = output_dir / "cp_udm_ablation_predictions.csv"
        temporary_path = output_dir / ".cp_udm_ablation_predictions.csv.tmp"
        fieldnames = [
            "configuration",
            "seed",
            "device_type",
            "sample_id",
            "target",
            "prediction",
            "variance",
            "dynamic_endpoint",
        ]
        with temporary_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)
        temporary_path.replace(predictions_path)
    generate_cp_ablation_summary(output_dir, payload)


def run_cp_udm_ablation_experiment(
    datasets: dict,
    output_dir: Path,
    seeds: list[int],
    epochs: int = 80,
    physics_weight: float = 0.05,
    patience: int = 15,
) -> dict:
    """Run the fixed-split, paired confirmatory CP-UDM ablation."""
    required_devices = set(CP_ABLATION_SCOPES[1:])
    if set(datasets) != required_devices:
        raise ValueError(
            "CP-UDM ablation requires exactly capacitor, igbt, and mosfet datasets"
        )
    if not seeds:
        raise ValueError("CP-UDM ablation requires at least one training seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("CP-UDM ablation training seeds must be unique")
    if epochs < 1:
        raise ValueError("CP-UDM ablation epochs must be positive")
    if patience < 1:
        raise ValueError("CP-UDM ablation patience must be positive")

    # This mapping is deliberately constructed exactly once and passed unchanged
    # to every run, which makes all configuration comparisons group-paired.
    splits = {
        name: split_groups(dataset, seed=CP_ABLATION_SPLIT_SEED)
        for name, dataset in datasets.items()
    }
    test_loaders = build_loaders(datasets, splits, "test", 64, False)
    base = model_config_for(
        datasets,
        physics_loss_weight=physics_weight,
        variant="cp_udm",
    )
    configurations = build_confirmatory_cp_ablation_configs(base)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "histories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": {
            "name": "paired_confirmatory_cp_udm_ablation",
            "split_seed": CP_ABLATION_SPLIT_SEED,
            "training_seeds": list(seeds),
            "epochs_requested": epochs,
            "early_stopping_patience": patience,
            "physics_loss_weight": physics_weight,
            "expected_run_count": len(configurations) * len(seeds),
            "devices": list(CP_ABLATION_SCOPES[1:]),
            "pairing": (
                "identical group split and repeated initialization/data-loader "
                "seed for every configuration"
            ),
        },
        "metric_definitions": {
            "paired_delta": "ablation_minus_full_for_the_same_training_seed",
            "mae_delta": "positive_means_component_removal_degraded_accuracy",
            "physics_rate_spearman": (
                "cross-sample pooled test diagnostic; not the same estimand as "
                "the per-sequence centered-log-rate shape training loss"
            ),
        },
        "configurations": {
            name: asdict(config) for name, config in configurations.items()
        },
        "splits": {name: asdict(split) for name, split in splits.items()},
        "metrics": {},
        "histories": {},
        "history_files": {},
    }
    all_predictions = []
    save_cp_ablation_outputs(output_dir, payload, all_predictions)

    serialized_splits = {name: asdict(split) for name, split in splits.items()}
    for configuration, config in configurations.items():
        for seed in seeds:
            print(
                f"[cp-ablation] configuration={configuration} seed={seed}",
                flush=True,
            )
            model, history = train_model(
                datasets,
                splits,
                config,
                seed,
                epochs=epochs,
                patience=patience,
            )
            result, rows = evaluate_model(model, test_loaders)
            seed_key = str(seed)
            payload["metrics"].setdefault(configuration, {})[seed_key] = result
            payload["histories"].setdefault(configuration, {})[seed_key] = history
            all_predictions.extend(
                {
                    "configuration": configuration,
                    "seed": seed,
                    **row,
                }
                for row in rows
            )

            history_path = history_dir / f"{configuration}_seed{seed}.json"
            _write_json_atomic(history_path, history)
            payload["history_files"].setdefault(configuration, {})[seed_key] = str(
                history_path.relative_to(output_dir)
            )
            checkpoint_path = checkpoint_dir / f"{configuration}_seed{seed}.pt"
            temporary_checkpoint = checkpoint_path.with_name(
                f".{checkpoint_path.name}.tmp"
            )
            torch.save(
                {
                    "configuration": configuration,
                    "config": config,
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "split_seed": CP_ABLATION_SPLIT_SEED,
                    "splits": serialized_splits,
                },
                temporary_checkpoint,
            )
            temporary_checkpoint.replace(checkpoint_path)
            save_cp_ablation_outputs(output_dir, payload, all_predictions)

    expected_seeds = {str(seed) for seed in seeds}
    for configuration in CP_ABLATION_CONFIG_NAMES:
        observed_seeds = set(payload["metrics"].get(configuration, {}))
        if observed_seeds != expected_seeds:
            raise RuntimeError(
                f"incomplete paired results for {configuration}: "
                f"expected {sorted(expected_seeds)}, got {sorted(observed_seeds)}"
            )
    return payload


def generate_stress_loso_summary(output_dir: Path, payload: dict):
    lines = [
        "# NASA 分组留一应力水平确认性实验",
        "",
        "## 协议审计",
        "",
    ]
    for device_type, audit in payload["audits"].items():
        if audit["supported"]:
            protocol = audit["protocol"]
            levels = ", ".join(
                f"{level * protocol['physical_scale']:.3g}{protocol['physical_unit']} "
                f"({len(groups)} groups)"
                for level, groups in audit["groups_by_level"]
            )
            lines.append(
                f"- **{device_type}**：支持 {protocol['stress_name']} 留一水平；{levels}；"
                f"排除 {len(audit['excluded_groups'])} 组。"
            )
        else:
            lines.append(f"- **{device_type}**：不支持。{audit['reason']}")
    lines.extend([
        "",
        "## 逐折结果",
        "",
        "| 数据集 | 留出应力 | seed | train/val/cal/test groups | 模型 | MAE | RMSE | R² | Spearman | PICP90 | MPIW90 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in payload["folds"]:
        split = run["split"]
        counts = (
            f"{len(split['train_groups'])}/{len(split['validation_groups'])}/"
            f"{len(split['calibration_groups'])}/"
            f"{len(split['test_groups'])}"
        )
        held_out = (
            f"{split['held_out_physical_level']:.3g}{split['physical_unit']}"
        )
        for model_name, metrics in run["metrics"].items():
            overall = metrics["overall"]
            conformal = metrics.get("group_conformal_90", {})
            lines.append(
                f"| {split['device_type']} | {held_out} | {run['seed']} | {counts} | "
                f"{model_name} | {_format_metric(overall.get('mae'))} | "
                f"{_format_metric(overall.get('rmse'))} | "
                f"{_format_metric(overall.get('r2'))} | "
                f"{_format_metric(overall.get('spearman'))} | "
                f"{_format_metric(conformal.get('picp'))} | "
                f"{_format_metric(conformal.get('mpiw'))} |"
            )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 所有前缀以原始运行/器件 group 为单位进入且只进入 train、validation、calibration、test 或 excluded 之一；测试集是完整未见应力水平。",
        "- capacitor 检验外加电压迁移；MOSFET 仅检验供电电压迁移，不应表述为未见热应力泛化。",
        "- IGBT 缓存没有重复、离散且组内稳定的外加应力水平，因此不生成 LOSO 结果。",
        "- MOSFET 两个方向的样本极不平衡，尤其低电压留出时只剩少量高电压训练组，必须逐方向报告，不能仅报告折间均值。",
        "- PICP/MPIW 使用独立 calibration groups，并以每个 group 的最大前缀残差为 conformal score；由于测试应力发生分布迁移，90% 仅是目标水平，不构成有限样本覆盖保证。",
        "- 预测目标仍是公开记录内的相对观测阶段；duration-only 是标签捷径诊断，不是可部署的健康估计方法。",
        "",
    ])
    (output_dir / "stress_loso_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def save_stress_loso_outputs(
    output_dir: Path,
    payload: dict,
    predictions: list[dict],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "stress_loso_metrics.json"
    metrics_temp_path = output_dir / ".stress_loso_metrics.json.tmp"
    metrics_temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    metrics_temp_path.replace(metrics_path)
    if predictions:
        fieldnames = [
            "fold_id",
            "model",
            "seed",
            "device_type",
            "stress_name",
            "held_out_level",
            "held_out_physical_level",
            "physical_unit",
            "sample_id",
            "target",
            "prediction",
            "variance",
            "dynamic_endpoint",
        ]
        predictions_path = output_dir / "stress_loso_predictions.csv"
        predictions_temp_path = output_dir / ".stress_loso_predictions.csv.tmp"
        with predictions_temp_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)
        predictions_temp_path.replace(predictions_path)
    generate_stress_loso_summary(output_dir, payload)


def run_stress_loso_experiment(
    datasets: dict,
    output_dir: Path,
    seeds: list[int],
    epochs: int,
    physics_weight: float,
) -> dict:
    """Run the confirmatory group-level stress generalization protocol."""
    if not seeds:
        raise ValueError("stress LOSO requires at least one training seed")
    if epochs < 1:
        raise ValueError("stress LOSO epochs must be positive")
    audits = {
        name: audit_nasa_stress_levels(dataset)
        for name, dataset in datasets.items()
    }
    if audits["igbt"].supported:
        raise RuntimeError("IGBT must not be treated as discrete stress-level data")
    for device_type in ("capacitor", "mosfet"):
        if not audits[device_type].supported:
            raise RuntimeError(audits[device_type].reason)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    payload = {
        "protocol": {
            "name": "group_level_leave_one_stress_level_out",
            "split_seed": 42,
            "training_seeds": seeds,
            "epochs_requested": epochs,
            "early_stopping_patience": 15,
            "physics_loss_weight": physics_weight,
            "conformal_alpha": 0.10,
            "conformal_score": "maximum_absolute_prefix_residual_per_group",
        },
        "audits": {name: asdict(audit) for name, audit in audits.items()},
        "folds": [],
    }
    baseline_functions = {
        "duration_only": run_duration_only_baseline,
        "response_only": run_response_only_baseline,
        "random_forest": run_random_forest_baseline,
    }
    for device_type in ("capacitor", "mosfet"):
        dataset = datasets[device_type]
        subset = {device_type: dataset}
        for split in leave_one_stress_level_out_splits(dataset, seed=42):
            fold_id = _stress_fold_id(split)
            split_mapping = {device_type: split}
            calibration_mapping = {
                device_type: DatasetSplit(
                    train_groups=split.train_groups,
                    validation_groups=split.validation_groups,
                    test_groups=split.calibration_groups,
                )
            }
            for seed in seeds:
                print(
                    f"[stress-loso] {fold_id} seed={seed} "
                    f"groups={len(split.train_groups)}/{len(split.validation_groups)}/"
                    f"{len(split.calibration_groups)}/"
                    f"{len(split.test_groups)}",
                    flush=True,
                )
                fold_record = {
                    "fold_id": fold_id,
                    "seed": seed,
                    "split": asdict(split),
                    "metrics": {},
                    "histories": {},
                }
                for model_name, baseline in baseline_functions.items():
                    metrics, rows = baseline(subset, split_mapping, seed)
                    _calibration_metrics, calibration_rows = baseline(
                        subset, calibration_mapping, seed
                    )
                    metrics = _add_stress_metadata(metrics, split)
                    metrics["group_conformal_90"] = group_conformal_metrics(
                        calibration_rows,
                        rows,
                        split.calibration_groups,
                        split.test_groups,
                        alpha=0.10,
                    )
                    fold_record["metrics"][model_name] = metrics
                    all_predictions.extend(_annotate_stress_predictions(
                        rows, split, fold_id, model_name, seed
                    ))

                for model_name, variant in (
                    ("legacy_udm_net", "legacy"),
                    ("cp_udm_net", "cp_udm"),
                ):
                    print(f"[stress-loso] training {model_name}", flush=True)
                    config = model_config_for(
                        subset,
                        physics_loss_weight=physics_weight,
                        variant=variant,
                    )
                    model, history = train_model(
                        subset,
                        split_mapping,
                        config,
                        seed,
                        epochs=epochs,
                    )
                    metrics, rows = evaluate_stress_level_holdout(
                        model, dataset, split
                    )
                    _calibration_metrics, calibration_rows = evaluate_model(
                        model,
                        build_loaders(
                            subset,
                            split_mapping,
                            "calibration",
                            64,
                            False,
                        ),
                    )
                    metrics["group_conformal_90"] = group_conformal_metrics(
                        calibration_rows,
                        rows,
                        split.calibration_groups,
                        split.test_groups,
                        alpha=0.10,
                    )
                    fold_record["metrics"][model_name] = metrics
                    fold_record["histories"][model_name] = history
                    all_predictions.extend(_annotate_stress_predictions(
                        rows, split, fold_id, model_name, seed
                    ))
                    torch.save(
                        {
                            "config": config,
                            "state_dict": model.state_dict(),
                            "seed": seed,
                            "split": asdict(split),
                        },
                        checkpoint_dir / f"{fold_id}_{model_name}_seed{seed}.pt",
                    )
                payload["folds"].append(fold_record)
                save_stress_loso_outputs(output_dir, payload, all_predictions)
    return payload


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if args.cp_ablation:
        selected_output_dir = args.cp_ablation_output_dir
    elif args.stress_loso:
        selected_output_dir = args.stress_loso_output_dir
    else:
        selected_output_dir = args.output_dir
    output_dir = Path(selected_output_dir).resolve()
    for path, label in ((data_root, "data root"), (output_dir, "output directory")):
        if path.drive.upper() != "D:":
            raise ValueError(f"{label} must be on D: for this experiment, got {path}")
    temp_dir = data_root.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for variable in ("TEMP", "TMP", "MPLCONFIGDIR", "TORCH_HOME"):
        os.environ[variable] = str(temp_dir / variable.lower())
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    cache_dir = data_root / "processed"
    datasets = {
        "capacitor": load_cached_dataset(
            cache_dir / "capacitor.pkl",
            lambda: load_capacitor_dataset(data_root / "capacitor"),
            args.rebuild_cache,
        ),
        "igbt": load_cached_dataset(
            cache_dir / "igbt.pkl",
            lambda: load_igbt_dataset(
                data_root / "igbt" / "extracted" / "IGBTAgingData_04022009" / "Data"
            ),
            args.rebuild_cache,
        ),
        "mosfet": load_cached_dataset(
            cache_dir / "mosfet.pkl",
            lambda: load_mosfet_dataset(data_root / "mosfet"),
            args.rebuild_cache,
        ),
    }
    if args.cp_ablation:
        payload = run_cp_udm_ablation_experiment(
            datasets,
            output_dir,
            seeds,
            epochs=args.epochs,
            physics_weight=args.physics_weight,
            patience=args.cp_ablation_patience,
        )
        print(json.dumps({
            "output_dir": str(output_dir),
            "protocol": payload["protocol"],
            "completed_run_count": sum(
                len(seed_results)
                for seed_results in payload["metrics"].values()
            ),
        }, ensure_ascii=False, indent=2))
        return
    if args.stress_loso:
        payload = run_stress_loso_experiment(
            datasets,
            output_dir,
            seeds,
            epochs=args.epochs,
            physics_weight=args.physics_weight,
        )
        print(json.dumps({
            "output_dir": str(output_dir),
            "protocol": payload["protocol"],
            "fold_count": len(payload["folds"]),
            "igbt_supported": payload["audits"]["igbt"]["supported"],
        }, ensure_ascii=False, indent=2))
        return

    splits = {name: split_groups(dataset, seed=42) for name, dataset in datasets.items()}
    test_loaders = build_loaders(datasets, splits, "test", 64, False)
    metrics: dict[str, dict[str, dict]] = {}
    predictions = []
    histories = {}

    for seed in seeds:
        duration_metrics, duration_predictions = run_duration_only_baseline(datasets, splits, seed)
        metrics.setdefault("duration_only", {})[str(seed)] = duration_metrics
        predictions.extend(add_prediction_metadata(duration_predictions, "duration_only", seed))

        response_metrics, response_predictions = run_response_only_baseline(datasets, splits, seed)
        metrics.setdefault("response_only", {})[str(seed)] = response_metrics
        predictions.extend(add_prediction_metadata(response_predictions, "response_only", seed))

        rf_metrics, rf_predictions = run_random_forest_baseline(datasets, splits, seed)
        metrics.setdefault("random_forest", {})[str(seed)] = rf_metrics
        predictions.extend(add_prediction_metadata(rf_predictions, "random_forest", seed))

        config = model_config_for(datasets, physics_loss_weight=args.physics_weight)
        model, history = train_model(datasets, splits, config, seed, epochs=args.epochs)
        joint_metrics, joint_predictions = evaluate_model(model, test_loaders)
        metrics.setdefault("udm_net_joint", {})[str(seed)] = joint_metrics
        predictions.extend(add_prediction_metadata(joint_predictions, "udm_net_joint", seed))
        histories[f"udm_net_joint_seed_{seed}"] = history
        torch.save({"config": config, "state_dict": model.state_dict()}, output_dir / "checkpoints" / f"udm_net_joint_seed_{seed}.pt")

        cp_config = model_config_for(
            datasets,
            physics_loss_weight=args.physics_weight,
            variant="cp_udm",
        )
        cp_model, cp_history = train_model(
            datasets, splits, cp_config, seed, epochs=args.epochs
        )
        cp_metrics, cp_predictions = evaluate_model(cp_model, test_loaders)
        _validation_metrics, validation_predictions = evaluate_model(
            cp_model,
            build_loaders(datasets, splits, "validation", 64, False),
        )
        cp_metrics["conformal_90"] = conformal_metrics(
            validation_predictions,
            cp_predictions,
            alpha=0.10,
        )
        metrics.setdefault("cp_udm_net", {})[str(seed)] = cp_metrics
        predictions.extend(add_prediction_metadata(cp_predictions, "cp_udm_net", seed))
        histories[f"cp_udm_net_seed_{seed}"] = cp_history
        torch.save(
            {"config": cp_config, "state_dict": cp_model.state_dict()},
            output_dir / "checkpoints" / f"cp_udm_net_seed_{seed}.pt",
        )

    base = model_config_for(datasets, physics_loss_weight=args.physics_weight)
    for ablation_name, config in build_ablation_configs(base).items():
        if ablation_name == "full_udm_net":
            continue
        model, history = train_model(datasets, splits, config, 42, epochs=args.epochs)
        result, rows = evaluate_model(model, test_loaders)
        metrics.setdefault(ablation_name, {})["42"] = result
        predictions.extend(add_prediction_metadata(rows, ablation_name, 42))
        histories[ablation_name] = history

    for device_type, dataset in datasets.items():
        subset = {device_type: dataset}
        subset_splits = {device_type: splits[device_type]}
        model, history = train_model(
            subset,
            subset_splits,
            model_config_for(subset, physics_loss_weight=args.physics_weight),
            42,
            epochs=args.epochs,
        )
        result, rows = evaluate_model(
            model,
            build_loaders(subset, subset_splits, "test", 64, False),
        )
        name = f"device_only_{device_type}"
        metrics.setdefault(name, {})["42"] = result
        predictions.extend(add_prediction_metadata(rows, name, 42))
        histories[name] = history

    save_experiment_outputs(output_dir, metrics, predictions, splits, histories)
    generate_summary(output_dir, metrics, datasets, splits)
    plot_metrics(output_dir, metrics)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "datasets": {name: {"groups": len(set(data.groups)), "series": len(data.series)} for name, data in datasets.items()},
        "experiments": list(metrics),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
