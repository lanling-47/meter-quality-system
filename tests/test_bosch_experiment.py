"""Synthetic fixtures for Bosch audit and chronological response protocol."""
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from openpyxl import Workbook

from app.services.bosch_observation_adapter import build_bosch_observation
from experiments.bosch_gbh2_24.run_bosch_experiment import (
    PARAMS, as_float, load_mes, load_repairs, parse_mes_timestamp,
    reconstruct_response,
)


class BoschExperimentTests(unittest.TestCase):
    def test_timestamp_nanoseconds(self):
        self.assertEqual(parse_mes_timestamp("05-AUG-25 01.58.44.752209000 PM ASIA/SHANGHAI"), datetime(2025, 8, 5, 13, 58, 44, 752209))
        self.assertIsNone(parse_mes_timestamp("bad"))

    def test_numeric_invalid(self):
        for value in [None, "bad", "nan", "inf"]:
            self.assertIsNone(as_float(value))
        self.assertEqual(as_float("0"), 0)

    def test_mask_preserves_zero_and_codes(self):
        obs = build_bosch_observation({"WP7LeakageCurrent1": 0, "WP7Volt": 0, "WP4ImpactEnergy": 80}, {"WP7Volt": "0"})
        self.assertTrue(obs["fields"]["WP7LeakageCurrent1"]["physical_model_mask"])
        self.assertFalse(obs["fields"]["WP7Volt"]["physical_model_mask"])
        self.assertFalse(obs["fields"]["WP4ImpactEnergy"]["physical_model_mask"])
        self.assertFalse(obs["fields"]["WP3RunInCurrent"]["available"])
        self.assertEqual(obs["fields"]["WP7Volt"]["raw_state_code"], "0")
        self.assertIsNone(obs["damage_target"])

    def test_master_sheet_not_filtered_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair.xlsx"
            wb = Workbook()
            filtered = wb.active
            filtered.title = "2025.08生产日期"
            master = wb.create_sheet("所有生产日期")
            row = ["id", "model", "country", None, "serial", None, "part", "repair", "08.2025", None, "08.2026", None, None]
            for sheet in [filtered, master]:
                sheet.append(["header"] * 13)
                sheet.append(row)
            master.append(["Overall Result"] + [None] * 12)
            wb.save(path)
            result = load_repairs(path)
            self.assertEqual(result["rows"], 1)
            self.assertEqual(len(result["serials"]), 1)

    def test_completeness_is_per_test_not_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mes.xlsx"
            wb = Workbook()
            sheet = wb.active
            sheet.append(["header"] * 23)
            for index, param in enumerate(PARAMS):
                row = [None] * 23
                row[0:3] = ["type", "tool", "test1" if index < 8 else "test2"]
                row[4], row[6], row[7], row[10], row[12] = "serial", "05-AUG-25 01.00.00.000000000 PM ASIA/SHANGHAI", param, 1, 1
                sheet.append(row)
            wb.save(path)
            result = load_mes(path)
            self.assertEqual(result["complete_16_parameter_serials"], 1)
            self.assertEqual(result["complete_16_parameter_tests"], 0)

    def test_first_cycle_and_chronological_split(self):
        events = {}
        for day in range(10):
            values = {"WP2RunInHalfVolt": 110., "WP2RunInFullVolt": 220., "WP2RunInHalfCurrent": 2. + day / 100, "WP2RunInFullCurrent": 3. + day / 100, "WP3RunInCurrent": 2.5 + day / 100}
            events[str(day)] = {"serial": str(day), "date": datetime(2025, 8, 1) + timedelta(days=day), "type": "train-type" if day < 8 else "new-type", "values": values, "states": {}, "counts": Counter({key:1 for key in values})}
        events["repeat"] = dict(events["0"], date=datetime(2025, 8, 20))
        with tempfile.TemporaryDirectory() as tmp:
            result = reconstruct_response(events, Path(tmp))
            self.assertEqual(result["independent_eligible_devices"], 10)
            self.assertEqual(result["split_counts"], {"train":6, "validation":2, "test":2})
            self.assertEqual(result["test_serial_overlap"], 0)
            self.assertEqual(result["test_unseen_types"], ["new-type"])
            self.assertEqual(result["retested_devices"], 1)


if __name__ == "__main__":
    unittest.main()
