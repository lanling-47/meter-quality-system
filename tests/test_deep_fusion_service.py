import unittest
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from app.services.deep_fusion_service import (
    BATCH_FEATURE_NAMES,
    MGQFormerConfig,
    MGQFormerNet,
    MGQFormerPackage,
    _batch_consistency_loss,
    _gradient_importance,
    _group_holdout_indices,
    _profile_to_batch_features,
    _reliability_from_batch_matrix,
    _trusted_batch_group_ids,
)


class ReliabilityAwareFusionTests(unittest.TestCase):
    def test_unlinked_batch_cannot_change_meter_prediction(self):
        torch.manual_seed(5)
        config = MGQFormerConfig(
            single_dim=4,
            batch_dim=len(BATCH_FEATURE_NAMES),
            d_model=8,
            nhead=2,
            num_layers=1,
            dropout=0.0,
        )
        model = MGQFormerNet(config).eval()
        meter = torch.tensor([[0.2, -0.1, 0.4, 0.7]])
        first_batch = torch.zeros(1, len(BATCH_FEATURE_NAMES))
        second_batch = torch.full((1, len(BATCH_FEATURE_NAMES)), 99.0)
        unlinked = torch.tensor([[0.0, 0.5, 0.05, 0.0, 0.0]])
        with torch.no_grad():
            first = model(meter, first_batch, unlinked)
            second = model(meter, second_batch, unlinked)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_beta_binomial_uncertainty_decreases_with_sample_count(self):
        small = SimpleNamespace(
            sample_count=2,
            high_risk_count=1,
            high_risk_rate=0.5,
            avg_final_error_max=0.0,
            max_final_error_max=0.0,
            avg_error_degradation_rate=0.0,
            max_clock_drift_rate=0.0,
            max_rs485_total_errors=0,
            confidence=0.8,
        )
        large = SimpleNamespace(**{**small.__dict__, "sample_count": 20, "high_risk_count": 10})
        small_vector = _profile_to_batch_features(small, mapping_confidence=1.0)
        large_vector = _profile_to_batch_features(large, mapping_confidence=1.0)
        variance_column = BATCH_FEATURE_NAMES.index("posterior_high_risk_variance")
        self.assertLess(large_vector[variance_column], small_vector[variance_column])

    def test_reliability_matrix_preserves_zero_mapping_confidence(self):
        batch = np.zeros((2, len(BATCH_FEATURE_NAMES)), dtype=np.float32)
        batch[:, BATCH_FEATURE_NAMES.index("posterior_high_risk_mean")] = 0.5
        batch[1, BATCH_FEATURE_NAMES.index("mapping_confidence")] = 0.9
        batch[1, BATCH_FEATURE_NAMES.index("batch_available")] = 1.0
        reliability = _reliability_from_batch_matrix(batch, BATCH_FEATURE_NAMES)
        self.assertEqual(float(reliability[0, 3]), 0.0)
        self.assertEqual(float(reliability[0, 4]), 0.0)
        self.assertAlmostEqual(float(reliability[1, 3]), 0.9, places=6)

    def test_group_holdout_keeps_trusted_batch_together(self):
        y = np.asarray([0, 1] * 6)
        meta = [
            {"meter_no": index, "batch_key": f"b{index // 2}", "confidence": 0.8}
            for index in range(len(y))
        ]
        train_idx, validation_idx, scope = _group_holdout_indices(y, meta, seed=42)
        train_batches = {meta[index]["batch_key"] for index in train_idx}
        validation_batches = {meta[index]["batch_key"] for index in validation_idx}
        self.assertFalse(train_batches & validation_batches)
        self.assertTrue(scope.startswith("group_holdout"))

    def test_consistency_compares_batch_aggregate_not_each_meter(self):
        probability = torch.tensor([0.2, 0.8, 0.9])
        reliability = torch.tensor([
            [2.0, 0.5, 0.01, 1.0, 1.0],
            [2.0, 0.5, 0.01, 1.0, 1.0],
            [2.0, 0.1, 0.01, 0.0, 0.0],
        ])
        group_ids = torch.tensor([0, 0, -1])
        loss = _batch_consistency_loss(probability, reliability, group_ids)
        # BCE at the aggregate p=mu=0.5 equals log(2). A pointwise loss on
        # p={0.2, 0.8} would be larger and would wrongly shrink both meters.
        self.assertAlmostEqual(float(loss), float(np.log(2.0)), places=6)

    def test_consistency_strength_scales_with_uniform_mapping_confidence(self):
        probability = torch.tensor([0.2, 0.8])
        full_reliability = torch.tensor([
            [2.0, 0.5, 0.01, 1.0, 1.0],
            [2.0, 0.5, 0.01, 1.0, 1.0],
        ])
        half_reliability = full_reliability.clone()
        half_reliability[:, 3] = 0.5
        group_ids = torch.tensor([0, 0])

        full_loss = _batch_consistency_loss(
            probability, full_reliability, group_ids
        )
        half_loss = _batch_consistency_loss(
            probability, half_reliability, group_ids
        )

        self.assertAlmostEqual(float(half_loss), 0.5 * float(full_loss), places=6)

    def test_consistency_strength_scales_at_tiny_positive_confidence(self):
        full_probability = torch.tensor(
            [0.2, 0.8], dtype=torch.float64, requires_grad=True
        )
        tiny_probability = full_probability.detach().clone().requires_grad_(True)
        full_reliability = torch.tensor([
            [2.0, 0.5, 0.01, 1.0, 1.0],
            [2.0, 0.5, 0.01, 1.0, 1.0],
        ], dtype=torch.float64)
        tiny_reliability = full_reliability.clone()
        tiny_reliability[:, 3] = 1e-9
        group_ids = torch.tensor([0, 0])

        full_loss = _batch_consistency_loss(
            full_probability, full_reliability, group_ids
        )
        tiny_loss = _batch_consistency_loss(
            tiny_probability, tiny_reliability, group_ids
        )
        full_loss.backward()
        tiny_loss.backward()

        self.assertAlmostEqual(
            float((tiny_loss / full_loss).detach()), 1e-9, places=15
        )
        torch.testing.assert_close(
            tiny_probability.grad,
            full_probability.grad * 1e-9,
            rtol=1e-12,
            atol=1e-15,
        )

    def test_consistency_is_zero_without_trusted_mapping(self):
        probability = torch.tensor([0.2, 0.8], requires_grad=True)
        reliability = torch.tensor([
            [2.0, 0.5, 0.01, 0.0, 1.0],
            [2.0, 0.5, 0.01, 0.0, 1.0],
        ])
        group_ids = torch.tensor([0, 0])

        loss = _batch_consistency_loss(probability, reliability, group_ids)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        torch.testing.assert_close(probability.grad, torch.zeros_like(probability))

    def test_trusted_batch_group_ids_exclude_unlinked_samples(self):
        meta = [
            {"batch_key": "A", "confidence": 0.8, "batch_available": True},
            {"batch_key": "A", "confidence": 0.6, "batch_available": True},
            {"batch_key": "B", "confidence": 0.0, "batch_available": True},
            {"batch_key": None, "confidence": 1.0, "batch_available": False},
        ]
        np.testing.assert_array_equal(_trusted_batch_group_ids(meta), [0, 0, -1, -1])

    def test_masked_rule_feature_has_zero_reported_importance(self):
        torch.manual_seed(7)
        feature_names = ["safe_feature", "final_error_max"]
        config = MGQFormerConfig(
            single_dim=len(feature_names),
            d_model=8,
            nhead=2,
            num_layers=1,
            dropout=0.0,
        )
        X_single = np.asarray([[0.1, 3.0], [0.4, 7.0]], dtype=np.float32)
        X_model = X_single.copy()
        X_model[:, 1] = 0.0
        X_batch = np.zeros((2, len(BATCH_FEATURE_NAMES)), dtype=np.float32)
        package = MGQFormerPackage(
            model=MGQFormerNet(config),
            single_scaler=StandardScaler().fit(X_model),
            batch_scaler=StandardScaler().fit(X_batch),
            feature_names=feature_names,
            batch_feature_names=BATCH_FEATURE_NAMES,
            config=config.__dict__,
            masked_feature_indices=[1],
        )
        importance = _gradient_importance(package, X_single, X_batch, feature_names)
        by_name = {item["name"]: item["importance"] for item in importance}
        self.assertEqual(by_name["final_error_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
