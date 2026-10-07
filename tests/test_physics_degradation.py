import unittest

import numpy as np

from app.services.physics_degradation import (
    capacitor_damage_rate,
    integrate_damage,
    physical_damage_stage,
    response_damage_proxy,
)


class PhysicsDegradationTests(unittest.TestCase):
    def test_arrhenius_voltage_rate_increases_with_stress(self):
        low = capacitor_damage_rate(np.array([320.0]), np.array([1.0]))[0]
        high = capacitor_damage_rate(np.array([320.0]), np.array([1.2]))[0]
        self.assertGreater(high, low)

    def test_integrated_damage_is_normalized_and_monotone(self):
        damage = integrate_damage([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        self.assertAlmostEqual(float(damage[-1]), 1.0)
        self.assertTrue(np.all(np.diff(damage) >= 0.0))

    def test_physics_stage_and_response_proxy_are_bounded(self):
        stresses = np.column_stack([np.linspace(0.7, 1.0, 8), np.ones(8), np.zeros((8, 6))])
        times = np.arange(8, dtype=float)
        responses = np.column_stack([np.linspace(1.0, 1.3, 8), np.ones(8), np.linspace(0.1, 0.8, 8)])
        stage = physical_damage_stage("igbt", stresses, times)
        proxy = response_damage_proxy("igbt", responses)
        self.assertTrue(np.all((stage >= 0.0) & (stage <= 1.0)))
        self.assertTrue(np.all((proxy >= 0.0) & (proxy <= 1.0)))


if __name__ == "__main__":
    unittest.main()
