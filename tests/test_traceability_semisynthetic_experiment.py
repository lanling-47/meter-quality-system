import unittest
from dataclasses import replace

import numpy as np

from experiments.ra_mgq.traceability_semisynthetic_confirmatory import (
    ALL_CANDIDATES,
    LOW_SUPPORT_CANDIDATE,
    POST_OUTCOME_CANDIDATE,
    GeneratorConfig,
    RunConfig,
    assemble_matrix,
    counterfactual_score,
    fit_risk_model,
    generate_traceability_dataset,
    rank_counterfactual_candidates,
    shuffled_event_timestamps,
)


class TraceabilitySemiSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_traceability_dataset(
            GeneratorConfig(production_group_count=36, meters_per_group=12)
        )

    def test_production_group_split_is_disjoint_balanced_and_exhaustive(self):
        split_sets = [set(values) for values in self.dataset.split_groups.values()]
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])
        self.assertEqual(len(set.union(*split_sets)), 36)
        for split, groups in self.dataset.split_groups.items():
            indices = self.dataset.indices(split)
            self.assertEqual(set(self.dataset.group_ids[indices]), set(groups))
            roots = self.dataset.true_roots[indices]
            counts = [np.sum(roots == root) for root in set(roots)]
            self.assertEqual(len(set(counts)), 1)

    def test_event_timestamps_are_auditable_and_leak_is_strictly_later(self):
        leak_index = ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)
        self.assertTrue(np.all(
            self.dataset.event_timestamps[:, leak_index]
            >= self.dataset.outcome_timestamps
        ))
        upstream = np.delete(self.dataset.event_timestamps, leak_index, axis=1)
        self.assertTrue(np.all(upstream < self.dataset.outcome_timestamps[:, None]))

    def test_counterfactual_score_matches_declared_equation(self):
        deltas = np.asarray((0.4, -0.2, 0.1))
        weights = np.asarray((1.0, 0.5, 0.25))
        result = counterfactual_score(deltas, weights, overlap=0.8, kappa=2.0)
        expected_neff = weights.sum() ** 2 / np.square(weights).sum()
        expected_delta = (1.0 * 0.4 + 0.5 * 0.0 + 0.25 * 0.1) / weights.sum()
        expected = expected_delta * expected_neff / (expected_neff + 2.0) * 0.8
        self.assertAlmostEqual(result["effective_sample_size"], expected_neff, places=10)
        self.assertAlmostEqual(result["score"], expected, places=10)

    def test_post_outcome_value_injection_is_exactly_filtered_from_model_input(self):
        test = self.dataset.indices("test")
        before = assemble_matrix(
            self.dataset, test, True, self.dataset.event_timestamps
        )
        changed_values = self.dataset.event_values.copy()
        leak_index = ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)
        changed_values[:, leak_index] += 1_000_000.0
        injected = replace(self.dataset, event_values=changed_values)
        after = assemble_matrix(injected, test, True, injected.event_timestamps)
        np.testing.assert_array_equal(before, after)

    def test_low_support_is_rejected_and_post_outcome_is_excluded(self):
        run = RunConfig(
            seeds=(17,), trees=16, reference_draws=4, bootstrap_replicates=20
        )
        model = fit_risk_model(
            self.dataset,
            seed=17,
            run=run,
            temporal_filter=True,
            timestamps=self.dataset.event_timestamps,
        )
        group = self.dataset.split_groups["test"][0]
        rows = rank_counterfactual_candidates(
            self.dataset, model, group, run, "proposed_temporal_cf"
        )
        by_candidate = {row["candidate"]: row for row in rows}
        self.assertEqual(
            by_candidate[LOW_SUPPORT_CANDIDATE]["status"],
            "evidence_insufficient_overlap",
        )
        self.assertLess(
            by_candidate[LOW_SUPPORT_CANDIDATE]["overlap"], run.min_overlap
        )
        self.assertIsNone(by_candidate[LOW_SUPPORT_CANDIDATE]["rank"])
        self.assertEqual(
            by_candidate[POST_OUTCOME_CANDIDATE]["status"],
            "excluded_post_outcome",
        )
        self.assertIsNone(by_candidate[POST_OUTCOME_CANDIDATE]["rank"])

    def test_timestamp_shuffle_exposes_result_derived_event(self):
        shuffled = shuffled_event_timestamps(self.dataset, seed=101)
        leak_index = ALL_CANDIDATES.index(POST_OUTCOME_CANDIDATE)
        leak_before = (
            self.dataset.event_timestamps[:, leak_index]
            < self.dataset.outcome_timestamps
        )
        leak_after = shuffled[:, leak_index] < self.dataset.outcome_timestamps
        self.assertFalse(np.any(leak_before))
        self.assertGreater(float(leak_after.mean()), 0.5)
        self.assertLess(float(leak_after.mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
