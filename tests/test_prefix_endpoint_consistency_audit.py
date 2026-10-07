import unittest

from experiments.unified_degradation.audit_prefix_endpoint_consistency import (
    build_adjacent_prefix_pairs,
    parse_prefix_sample_id,
    summarize_pairs,
)


def _row(sample_id, target, prediction):
    group_id, prefix_end_index = parse_prefix_sample_id(sample_id)
    return {
        "experiment": "test_experiment",
        "variant": "full",
        "seed": 17,
        "device_type": "capacitor",
        "sample_id": sample_id,
        "group_id": group_id,
        "prefix_end_index": prefix_end_index,
        "target": target,
        "prediction": prediction,
    }


class PrefixEndpointConsistencyAuditTests(unittest.TestCase):
    def test_parser_preserves_nested_group_delimiters(self):
        self.assertEqual(
            parse_prefix_sample_id("ES12.mat::ES12C4::prefix-0028"),
            ("ES12.mat::ES12C4", 28),
        )

    def test_parser_rejects_ambiguous_or_missing_prefix_metadata(self):
        for sample_id in ("run-1", "run-1::prefix-late", "::prefix-0001"):
            with self.subTest(sample_id=sample_id):
                with self.assertRaises(ValueError):
                    parse_prefix_sample_id(sample_id)

    def test_pairs_are_sorted_by_endpoint_and_measure_final_prediction_drops(self):
        rows = [
            _row("run-1::prefix-0030", 0.9, 0.55),
            _row("run-1::prefix-0010", 0.2, 0.40),
            _row("run-1::prefix-0020", 0.6, 0.35),
        ]
        pairs = build_adjacent_prefix_pairs(rows)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["earlier_prefix_end_index"], 10)
        self.assertEqual(pairs[0]["later_prefix_end_index"], 20)
        self.assertAlmostEqual(pairs[0]["signed_prediction_change"], -0.05)
        self.assertAlmostEqual(pairs[0]["decrease_magnitude"], 0.05)
        self.assertTrue(pairs[0]["is_decrease"])
        self.assertFalse(pairs[1]["is_decrease"])

        pooled = next(
            row
            for row in summarize_pairs(pairs)
            if row["aggregation_scope"] == "all_seeds_all_devices"
        )
        self.assertEqual(pooled["adjacent_pair_count"], 2)
        self.assertEqual(pooled["decrease_pair_count"], 1)
        self.assertAlmostEqual(pooled["adjacent_endpoint_decrease_rate"], 0.5)
        self.assertAlmostEqual(pooled["maximum_adjacent_endpoint_decrease"], 0.05)
        self.assertEqual(pooled["group_seed_with_decrease_count"], 1)

    def test_positive_tolerance_is_applied_only_to_decrease_classification(self):
        rows = [
            _row("run-1::prefix-0010", 0.2, 0.5),
            _row("run-1::prefix-0020", 0.8, 0.4999999),
        ]
        pair = build_adjacent_prefix_pairs(rows, decrease_tolerance=1e-6)[0]
        self.assertGreater(pair["decrease_magnitude"], 0.0)
        self.assertFalse(pair["is_decrease"])
        pooled = next(
            row
            for row in summarize_pairs([pair])
            if row["aggregation_scope"] == "all_seeds_all_devices"
        )
        self.assertEqual(pooled["decrease_pair_count"], 0)
        self.assertGreater(pooled["maximum_adjacent_endpoint_decrease"], 0.0)

    def test_duplicate_numeric_endpoint_is_rejected(self):
        rows = [
            _row("run-1::prefix-01", 0.2, 0.3),
            _row("run-1::prefix-001", 0.4, 0.5),
        ]
        with self.assertRaises(ValueError):
            build_adjacent_prefix_pairs(rows)


if __name__ == "__main__":
    unittest.main()
