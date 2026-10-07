from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from app.services.unified_degradation_data import DegradationSeries
from app.services.unified_degradation_model import UnifiedDegradationConfig
from experiments.unified_degradation.experiment_runner import (
    DatasetSplit,
    audit_nasa_stress_levels,
    evaluate_stress_level_holdout,
    group_conformal_metrics,
    leave_one_stress_level_out_splits,
    split_groups,
)
from experiments.unified_degradation.nasa_loaders import LoadedDataset, _prefix_series
from experiments.unified_degradation.run_nasa_experiment import (
    CP_ABLATION_CONFIG_NAMES,
    build_confirmatory_cp_ablation_configs,
    run_cp_udm_ablation_experiment,
    summarize_cp_ablation_metrics,
)


def _capacitor_dataset(groups_by_level, unstable_groups=()):
    series, groups = [], []
    unstable_groups = set(unstable_groups)
    for level, level_groups in groups_by_level.items():
        for group in level_groups:
            for prefix, target in enumerate((0.5, 1.0), start=1):
                stresses = np.zeros((4, 8), dtype=np.float32)
                stresses[:, 0] = level
                if group in unstable_groups and target == 1.0:
                    stresses[:, 0] = np.linspace(level - 0.02, level + 0.02, 4)
                series.append(DegradationSeries(
                    "capacitor",
                    f"{group}::prefix-{prefix}",
                    np.arange(4, dtype=np.float32),
                    stresses,
                    np.full((4, 1), target, dtype=np.float32),
                    target,
                ))
                groups.append(group)
    return LoadedDataset("capacitor", ("response",), (1,), series, groups, [])


class _StageModelStub(torch.nn.Module):
    def forward(self, device_type, responses, stresses, times, mask):
        batch_size, sequence_length = times.shape
        return {
            "prediction": responses[:, 0, 0] * 0.8,
            "variance": torch.full((batch_size,), 0.04),
            "trajectory": torch.zeros((batch_size, sequence_length)),
            "mask": mask,
        }


class ExperimentProtocolTests(unittest.TestCase):
    def test_prefix_resampling_removes_sequence_length_leakage(self):
        times = np.arange(100, dtype=np.float32)
        stresses = np.ones((100, 8), dtype=np.float32)
        responses = np.column_stack([times, times ** 2]).astype(np.float32)
        prefixes = _prefix_series(
            "test_device",
            "run-1",
            times,
            stresses,
            responses,
            fractions=(0.25, 0.5, 1.0),
        )
        self.assertEqual([len(item.times) for item in prefixes], [32, 32, 32])
        self.assertEqual([round(item.target, 2) for item in prefixes], [0.24, 0.49, 1.0])

    def test_group_split_is_disjoint(self):
        series, groups = [], []
        for group_index in range(8):
            group = f"device-{group_index}"
            for prefix in range(3):
                series.append(DegradationSeries(
                    "test_device",
                    f"{group}-{prefix}",
                    [0, 1],
                    [[0] * 8, [0] * 8],
                    [[1], [2]],
                    prefix / 2,
                ))
                groups.append(group)
        dataset = LoadedDataset(
            "test_device", ("response",), (1,), series, groups, []
        )
        split = split_groups(dataset, seed=42)
        train = set(split.train_groups)
        validation = set(split.validation_groups)
        test = set(split.test_groups)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, set(groups))

    def test_stress_level_audit_counts_original_groups_once(self):
        levels = {
            10.0 / 14.0: ("v10-a", "v10-b", "v10-c"),
            12.0 / 14.0: ("v12-a", "v12-b", "v12-c"),
            1.0: ("v14-a", "v14-b", "v14-c"),
        }
        audit = audit_nasa_stress_levels(_capacitor_dataset(levels))
        self.assertTrue(audit.supported)
        self.assertEqual([len(groups) for _, groups in audit.groups_by_level], [3, 3, 3])
        self.assertEqual(len(audit.group_summaries), 9)
        self.assertEqual(audit.excluded_groups, ())

    def test_leave_one_stress_level_out_splits_are_group_disjoint(self):
        levels = {
            10.0 / 14.0: tuple(f"v10-{index}" for index in range(4)),
            12.0 / 14.0: tuple(f"v12-{index}" for index in range(4)),
            1.0: tuple(f"v14-{index}" for index in range(4)),
        }
        dataset = _capacitor_dataset(levels)
        audit = audit_nasa_stress_levels(dataset)
        expected_test_groups = dict(audit.groups_by_level)
        folds = leave_one_stress_level_out_splits(dataset, seed=42)
        self.assertEqual(folds, leave_one_stress_level_out_splits(dataset, seed=42))
        self.assertEqual(len(folds), 3)
        for fold in folds:
            train = set(fold.train_groups)
            validation = set(fold.validation_groups)
            calibration = set(fold.calibration_groups)
            test = set(fold.test_groups)
            parts = (train, validation, calibration, test)
            self.assertFalse(any(
                left & right
                for index, left in enumerate(parts)
                for right in parts[index + 1:]
            ))
            self.assertEqual(test, set(expected_test_groups[fold.held_out_level]))
            self.assertEqual(train | validation | calibration | test, set(dataset.groups))

    def test_mixed_stress_group_is_excluded_from_every_fold(self):
        levels = {
            10.0 / 14.0: ("v10-a", "v10-b", "v10-c", "v10-mixed"),
            12.0 / 14.0: ("v12-a", "v12-b", "v12-c"),
            1.0: ("v14-a", "v14-b", "v14-c"),
        }
        dataset = _capacitor_dataset(levels, unstable_groups={"v10-mixed"})
        audit = audit_nasa_stress_levels(dataset)
        self.assertIn("v10-mixed", audit.excluded_groups)
        summary = next(row for row in audit.group_summaries if row.group == "v10-mixed")
        self.assertEqual(summary.exclusion_reason, "mixed_or_unstable_stress")
        for fold in leave_one_stress_level_out_splits(dataset, seed=7):
            assigned = set(
                fold.train_groups
                + fold.validation_groups
                + fold.calibration_groups
                + fold.test_groups
            )
            self.assertNotIn("v10-mixed", assigned)
            self.assertIn("v10-mixed", fold.excluded_groups)

    def test_igbt_discrete_stress_holdout_is_explicitly_unsupported(self):
        base = _capacitor_dataset({1.0: ("run-a", "run-b")})
        dataset = LoadedDataset(
            "igbt", base.response_names, base.response_directions,
            base.series, base.groups, base.source_files,
        )
        audit = audit_nasa_stress_levels(dataset)
        self.assertFalse(audit.supported)
        self.assertIn("does not support", audit.reason)
        with self.assertRaisesRegex(ValueError, "does not support"):
            leave_one_stress_level_out_splits(dataset, seed=42)

    def test_stress_holdout_evaluation_scores_only_held_out_groups(self):
        levels = {
            10.0 / 14.0: ("v10-a", "v10-b", "v10-c"),
            12.0 / 14.0: ("v12-a", "v12-b", "v12-c"),
            1.0: ("v14-a", "v14-b", "v14-c"),
        }
        dataset = _capacitor_dataset(levels)
        fold = leave_one_stress_level_out_splits(dataset, seed=42)[0]
        metrics, predictions = evaluate_stress_level_holdout(
            _StageModelStub(), dataset, fold, batch_size=2
        )
        predicted_groups = {
            row["sample_id"].split("::", maxsplit=1)[0]
            for row in predictions
        }
        self.assertEqual(predicted_groups, set(fold.test_groups))
        self.assertEqual(len(predictions), 2 * len(fold.test_groups))
        self.assertEqual(
            metrics["stress_holdout"]["held_out_level"], fold.held_out_level
        )
        self.assertTrue(all(
            row["stress_name"] == "applied_voltage" for row in predictions
        ))
        self.assertEqual(
            metrics["stress_holdout"]["calibration_group_count"],
            len(fold.calibration_groups),
        )

    def test_group_conformal_uses_one_worst_case_score_per_group(self):
        calibration = [
            {"sample_id": "cal-a::prefix-1", "target": 0.2, "prediction": 0.1},
            {"sample_id": "cal-a::prefix-2", "target": 0.5, "prediction": 0.45},
            {"sample_id": "cal-b::prefix-1", "target": 0.3, "prediction": 0.1},
            {"sample_id": "cal-b::prefix-2", "target": 0.8, "prediction": 0.75},
        ]
        test = [
            {"sample_id": "test-a::prefix-1", "target": 0.2, "prediction": 0.15},
            {"sample_id": "test-a::prefix-2", "target": 0.9, "prediction": 0.6},
        ]
        metrics = group_conformal_metrics(
            calibration,
            test,
            ("cal-a", "cal-b"),
            ("test-a",),
            alpha=0.1,
        )
        self.assertAlmostEqual(metrics["radius"], 0.2)
        self.assertEqual(metrics["calibration_group_count"], 2)
        self.assertEqual(metrics["picp"], metrics["coverage"])
        self.assertEqual(metrics["mpiw"], metrics["mean_width"])


class ConfirmatoryCPAblationTests(unittest.TestCase):
    @staticmethod
    def _base_config():
        return UnifiedDegradationConfig(
            response_dims={"capacitor": 1, "igbt": 1, "mosfet": 1},
            response_directions={
                "capacitor": (1,),
                "igbt": (1,),
                "mosfet": (1,),
            },
            stress_dim=8,
            d_model=8,
            nhead=2,
            num_layers=1,
            use_latent_dynamics=True,
            physics_loss_weight=0.05,
            forecast_loss_weight=0.10,
            ranking_loss_weight=0.10,
            endpoint_consistency_weight=0.10,
            alignment_loss_weight=0.05,
            domain_loss_weight=0.10,
            domain_adversarial_strength=0.20,
        )

    @staticmethod
    def _result(value):
        metric_values = {
            "mae": float(value),
            "spearman": float(value),
            "trajectory_monotonic_violation_rate": float(value),
            "physics_rate_spearman": float(value),
        }
        return {
            "overall": dict(metric_values),
            "devices": {
                device_type: dict(metric_values)
                for device_type in ("capacitor", "igbt", "mosfet")
            },
        }

    def test_confirmatory_configs_change_only_declared_weights(self):
        base = self._base_config()
        base_values = asdict(base)
        configs = build_confirmatory_cp_ablation_configs(base)

        self.assertEqual(tuple(configs), CP_ABLATION_CONFIG_NAMES)
        self.assertIsNot(configs["full"], base)
        self.assertEqual(asdict(base), base_values)
        expected_changes = {
            "full": set(),
            "no_physics": {"physics_loss_weight"},
            "no_forecast": {"forecast_loss_weight"},
            "no_ranking_endpoint": {
                "ranking_loss_weight",
                "endpoint_consistency_weight",
            },
            "no_alignment": {
                "alignment_loss_weight",
                "domain_loss_weight",
                "domain_adversarial_strength",
            },
        }
        for name, config in configs.items():
            values = asdict(config)
            changed = {
                field_name
                for field_name, original in base_values.items()
                if values[field_name] != original
            }
            self.assertEqual(changed, expected_changes[name])
            self.assertTrue(config.use_latent_dynamics)
        self.assertEqual(configs["no_alignment"].alignment_loss_weight, 0.0)
        self.assertEqual(configs["no_alignment"].domain_loss_weight, 0.0)
        self.assertEqual(configs["no_alignment"].domain_adversarial_strength, 0.0)

    def test_summary_uses_sample_sd_and_same_seed_paired_deltas(self):
        metrics = {
            "full": {
                "17": self._result(1.0),
                "42": self._result(2.0),
                "73": self._result(3.0),
            },
            "no_physics": {
                "17": self._result(2.0),
                "42": self._result(4.0),
                "73": self._result(6.0),
            },
        }
        aggregate, paired = summarize_cp_ablation_metrics(metrics)

        full_mae = aggregate["full"]["overall"]["mae"]
        self.assertEqual(full_mae["n"], 3)
        self.assertAlmostEqual(full_mae["mean"], 2.0)
        self.assertAlmostEqual(full_mae["sample_sd"], 1.0)
        delta = paired["no_physics"]["overall"]["mae"]
        self.assertEqual(delta["by_seed"], {"17": 1.0, "42": 2.0, "73": 3.0})
        self.assertAlmostEqual(delta["mean"], 2.0)
        self.assertAlmostEqual(delta["sample_sd"], 1.0)

    def test_runner_reuses_one_split_and_writes_all_fifteen_runs(self):
        datasets = {
            device_type: SimpleNamespace(
                device_type=device_type,
                response_names=("response",),
                response_directions=(1,),
            )
            for device_type in ("capacitor", "igbt", "mosfet")
        }
        split_objects = {}
        train_calls = []

        def fake_split(dataset, seed):
            self.assertEqual(seed, 42)
            split = DatasetSplit(
                train_groups=(f"{dataset.device_type}-train",),
                validation_groups=(f"{dataset.device_type}-validation",),
                test_groups=(f"{dataset.device_type}-test",),
            )
            split_objects[dataset.device_type] = split
            return split

        class TrainedModel:
            def __init__(self, config, seed, run_index):
                self.config = config
                self.seed = seed
                self.run_index = run_index

            def state_dict(self):
                return {"weight": torch.tensor([float(self.run_index)])}

        def fake_train(
            received_datasets,
            received_splits,
            config,
            seed,
            epochs,
            patience,
        ):
            train_calls.append({
                "datasets": received_datasets,
                "splits": received_splits,
                "config": config,
                "seed": seed,
                "epochs": epochs,
                "patience": patience,
            })
            return TrainedModel(config, seed, len(train_calls)), [
                {"epoch": 1, "validation_mae": 0.25}
            ]

        def fake_evaluate(model, _loaders):
            value = model.run_index / 100.0
            return self._result(value), [{
                "device_type": "capacitor",
                "sample_id": f"sample-{model.run_index}",
                "target": 0.5,
                "prediction": value,
                "variance": 0.01,
                "dynamic_endpoint": value,
            }]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with (
                patch(
                    "experiments.unified_degradation.run_nasa_experiment.split_groups",
                    side_effect=fake_split,
                ) as split_mock,
                patch(
                    "experiments.unified_degradation.run_nasa_experiment.build_loaders",
                    return_value=object(),
                ) as loader_mock,
                patch(
                    "experiments.unified_degradation.run_nasa_experiment.train_model",
                    side_effect=fake_train,
                ),
                patch(
                    "experiments.unified_degradation.run_nasa_experiment.evaluate_model",
                    side_effect=fake_evaluate,
                ),
            ):
                payload = run_cp_udm_ablation_experiment(
                    datasets,
                    output_dir,
                    seeds=[17, 42, 73],
                )

            self.assertEqual(split_mock.call_count, 3)
            loader_mock.assert_called_once()
            self.assertEqual(len(train_calls), 15)
            shared_splits = train_calls[0]["splits"]
            self.assertTrue(all(call["splits"] is shared_splits for call in train_calls))
            self.assertEqual(shared_splits, split_objects)
            self.assertTrue(all(call["epochs"] == 80 for call in train_calls))
            self.assertTrue(all(call["patience"] == 15 for call in train_calls))
            self.assertTrue(all(call["config"].use_latent_dynamics for call in train_calls))
            self.assertEqual(
                [(call["seed"]) for call in train_calls[:3]],
                [17, 42, 73],
            )
            self.assertEqual(tuple(payload["metrics"]), CP_ABLATION_CONFIG_NAMES)
            self.assertEqual(len(list((output_dir / "checkpoints").glob("*.pt"))), 15)
            self.assertEqual(len(list((output_dir / "histories").glob("*.json"))), 15)
            for filename in (
                "cp_udm_ablation_metrics.json",
                "cp_udm_ablation_histories.json",
                "cp_udm_ablation_predictions.csv",
                "cp_udm_ablation_report.md",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            saved = json.loads(
                (output_dir / "cp_udm_ablation_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["protocol"]["split_seed"], 42)
            self.assertEqual(saved["protocol"]["expected_run_count"], 15)
            self.assertNotIn("histories", saved)
            report = (output_dir / "cp_udm_ablation_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("cross-sample pooled diagnostic", report)
            self.assertIn("not the same estimand", report)


if __name__ == "__main__":
    unittest.main()
