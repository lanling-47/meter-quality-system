"""Data-isolation and saved-inference checks for the meter forecast branch."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.unified_degradation.run_meter_neural import (
    MeterResidualNet, ROOT, features, neural_fit, neural_predict, read_excel,
)


class MeterForecastTests(unittest.TestCase):
    def test_raw_data_and_historical_prefix(self):
        ids, y, dates, channels = read_excel(next((ROOT / 'data/raw').glob('单相表*.xlsx')))
        self.assertEqual(y.shape, (29, 6, 10))
        self.assertEqual(len(set(ids)), 29)
        changed = y.copy()
        changed[:, 3:] = 999
        np.testing.assert_array_equal(features(y[:, :3]), features(changed[:, :3]))
        with self.assertRaises(ValueError):
            features(y)

    def test_training_statistics_and_validation_isolation(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(18)
        x, y = rng.normal(size=(10, 30)), rng.normal(size=(10, 30)) * .1
        config = dict(width=8, structured=True, shrinkage=.1)
        a = neural_fit(x, y, config, 42, (x[:2], y[:2]), epochs=1)
        b = neural_fit(x, y, config, 42, (x[:2] * 100, y[:2] * 100), epochs=1)
        for aa, bb in zip(a[:1], b[:1]):
            for key in aa.state_dict():
                torch.testing.assert_close(aa.state_dict()[key], bb.state_dict()[key])
        np.testing.assert_array_equal(a[1], x.mean(0))
        np.testing.assert_array_equal(a[3], np.median(y, axis=0))

    def test_checkpoint_roundtrip(self):
        rng = np.random.default_rng(42)
        x, y = rng.normal(size=(8, 30)), rng.normal(size=(8, 30)) * .1
        for structured in (True, False):
            config = dict(width=8, structured=structured, shrinkage=.1)
            fit = neural_fit(x, y, config, 42, epochs=3)
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / 'model.pt'
                torch.save(fit[0].state_dict(), p)
                restored = MeterResidualNet(8, structured)
                restored.load_state_dict(torch.load(p, weights_only=True))
                np.testing.assert_array_equal(neural_predict(fit, x),
                                              neural_predict((restored, *fit[1:4]), x))


if __name__ == '__main__':
    unittest.main()
