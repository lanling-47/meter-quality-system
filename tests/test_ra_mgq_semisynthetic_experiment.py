import unittest

import numpy as np
import torch

from app.services.deep_fusion_service import BATCH_FEATURE_NAMES, MGQFormerConfig, MGQFormerNet
from experiments.ra_mgq.semisynthetic_confirmatory import (
    GeneratorConfig,
    beta_posterior,
    consistency_confidence_budget_audit,
    generate_semisynthetic_dataset,
    perturb_batch_mappings,
    posterior_contraction_audit,
)


class SemiSyntheticRAMGQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_semisynthetic_dataset(
            GeneratorConfig(batch_count=24, meters_per_batch=8)
        )

    def test_batch_split_is_disjoint_and_exhaustive(self):
        split_sets = [set(values) for values in self.dataset.split_batches.values()]
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])
        self.assertEqual(len(set.union(*split_sets)), 24)
        for split, batches in self.dataset.split_batches.items():
            indices = self.dataset.indices(split)
            self.assertEqual(
                set(self.dataset.batch_ids[indices].tolist()), set(batches)
            )

    def test_beta_posterior_contracts_at_fixed_fraction(self):
        rows = posterior_contraction_audit()
        variances = [row["posterior_variance"] for row in rows]
        self.assertTrue(all(left > right for left, right in zip(variances, variances[1:])))
        mean, variance = beta_posterior(20, 10)
        self.assertEqual(mean, 0.5)
        self.assertGreater(variance, 0.0)

    def test_consistency_confidence_budget_scales_loss_and_gradient(self):
        audit = consistency_confidence_budget_audit()
        self.assertEqual(
            [row["mapping_confidence"] for row in audit["rows"]],
            [1.0, 0.5, 0.1, 1e-9, 0.0],
        )
        for row in audit["rows"]:
            self.assertAlmostEqual(
                row["loss_ratio_to_c1"], row["mapping_confidence"], places=12
            )
            self.assertAlmostEqual(
                row["gradient_l1_ratio_to_c1"],
                row["mapping_confidence"],
                places=12,
            )
            self.assertAlmostEqual(
                row["gradient_l2_ratio_to_c1"],
                row["mapping_confidence"],
                places=12,
            )
            self.assertAlmostEqual(
                row["probability_gradient_l1_ratio_to_c1"],
                row["mapping_confidence"],
                places=12,
            )
        zero = audit["rows"][-1]
        self.assertEqual(zero["consistency_loss_abs"], 0.0)
        self.assertEqual(zero["gradient_l1_abs"], 0.0)
        self.assertTrue(all(value == 0.0 for value in zero["gradient_abs_per_meter"]))
        self.assertEqual(zero["probability_gradient_l1_abs"], 0.0)
        self.assertTrue(audit["passed_linear_budget_scaling"])
        self.assertTrue(audit["zero_confidence_loss_exactly_zero"])
        self.assertTrue(audit["zero_confidence_gradient_exactly_zero"])
        self.assertTrue(
            audit["tiny_positive_confidence_preserves_linear_scaling"]
        )

    def test_mapping_perturbation_has_exact_rate_and_no_self_donor(self):
        indices = self.dataset.indices("test")
        batch, observed, corrupted = perturb_batch_mappings(
            self.dataset,
            indices,
            error_rate=0.5,
            scenario="confidence_aware",
            seed=101,
        )
        expected = int(round(0.5 * len(indices)))
        self.assertEqual(int(corrupted.sum()), expected)
        self.assertTrue(np.all(observed[corrupted] != self.dataset.batch_ids[indices][corrupted]))
        confidence = batch[:, BATCH_FEATURE_NAMES.index("mapping_confidence")]
        np.testing.assert_allclose(confidence[corrupted], 0.5)
        np.testing.assert_allclose(confidence[~corrupted], 1.0)

    def test_overconfident_control_keeps_corrupted_links_trusted(self):
        indices = self.dataset.indices("test")
        batch, _, corrupted = perturb_batch_mappings(
            self.dataset,
            indices,
            error_rate=0.3,
            scenario="overconfident",
            seed=101,
        )
        confidence = batch[:, BATCH_FEATURE_NAMES.index("mapping_confidence")]
        np.testing.assert_allclose(confidence[corrupted], 1.0)

    def test_production_gate_is_exactly_invariant_at_zero_confidence(self):
        torch.manual_seed(11)
        config = MGQFormerConfig(
            single_dim=4,
            d_model=8,
            nhead=2,
            num_layers=1,
            dropout=0.0,
        )
        model = MGQFormerNet(config).eval()
        meter = torch.randn(5, 4)
        first_batch = torch.randn(5, len(BATCH_FEATURE_NAMES))
        second_batch = torch.randn(5, len(BATCH_FEATURE_NAMES)) * 100.0
        reliability = torch.zeros(5, 5)
        with torch.no_grad():
            first = model(meter, first_batch, reliability)
            second = model(meter, second_batch, reliability)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
