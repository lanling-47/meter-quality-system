import unittest

import numpy as np

from experiments.ra_mgq.beta_binomial_calibration_audit import (
    AuditConfig,
    beta_posterior_mean_interval,
    expected_binary_scores,
    run_audit,
    wilson_interval,
)


class BetaBinomialCalibrationAuditTests(unittest.TestCase):
    def test_intervals_are_ordered_and_bounded(self):
        counts = np.arange(9)
        mean, lower, upper = beta_posterior_mean_interval(
            counts, sample_count=8, alpha=2.0, beta=8.0
        )
        self.assertTrue(np.all((0.0 < mean) & (mean < 1.0)))
        self.assertTrue(np.all((0.0 <= lower) & (lower <= mean)))
        self.assertTrue(np.all((mean <= upper) & (upper <= 1.0)))

        raw_lower, raw_upper = wilson_interval(counts, sample_count=8)
        self.assertTrue(np.all((0.0 <= raw_lower) & (raw_lower <= raw_upper)))
        self.assertTrue(np.all(raw_upper <= 1.0))

    def test_expected_scores_match_binary_outcome_expectation(self):
        probability = np.asarray([0.2, 0.7])
        estimate = np.asarray([0.3, 0.6])
        brier, nll = expected_binary_scores(probability, estimate)
        expected_brier = probability * (1.0 - estimate) ** 2 + (
            1.0 - probability
        ) * estimate**2
        expected_nll = -(
            probability * np.log(estimate)
            + (1.0 - probability) * np.log1p(-estimate)
        )
        np.testing.assert_allclose(brier, expected_brier)
        np.testing.assert_allclose(nll, expected_nll)

    def test_fixed_protocol_is_reproducible_and_has_required_controls(self):
        config = AuditConfig(
            seeds=(17, 42),
            sample_sizes=(1, 8, 64),
            batches_per_size_per_seed=4_000,
        )
        first = run_audit(config)
        second = run_audit(config)
        self.assertEqual(first, second)
        self.assertEqual(
            first["evidence_label"],
            "SEMI-SYNTHETIC / MONTE CARLO EVIDENCE ONLY",
        )
        self.assertIn("beta_oracle", first["estimators"])
        self.assertIn("beta_mismatched", first["estimators"])

        lookup = {
            (row["sample_count"], row["estimator"]): row
            for row in first["summary"]
        }
        self.assertLess(
            lookup[(1, "beta_oracle")]["brier_mean"],
            lookup[(1, "raw_rate_wilson")]["brier_mean"],
        )
        self.assertGreater(
            lookup[(1, "beta_mismatched")]["brier_mean"],
            lookup[(1, "beta_uniform")]["brier_mean"],
        )
        self.assertLess(
            lookup[(64, "beta_oracle")]["interval_width_mean"],
            lookup[(1, "beta_oracle")]["interval_width_mean"],
        )


if __name__ == "__main__":
    unittest.main()
