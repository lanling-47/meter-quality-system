import unittest

import numpy as np

from app.services.unified_degradation_data import (
    DegradationSeries,
    SameDeviceSeriesDataset,
    collate_degradation_series,
)
from app.services.unified_degradation_features import (
    ResponseSpec,
    aggregate_degradation_evidence,
    extract_degradation_features,
)


class UnifiedFeatureTests(unittest.TestCase):
    def test_positive_and_negative_directions_share_degradation_semantics(self):
        times = [0, 10, 20, 30]
        rising_bad = extract_degradation_features(
            times,
            [10.0, 11.0, 12.0, 13.0],
            ResponseSpec("resistance", degradation_direction=1),
        )
        falling_bad = extract_degradation_features(
            times,
            [100.0, 95.0, 90.0, 85.0],
            ResponseSpec("capacity", degradation_direction=-1),
        )
        self.assertGreater(rising_bad.final_relative_drift, 0.0)
        self.assertGreater(falling_bad.final_relative_drift, 0.0)
        self.assertAlmostEqual(rising_bad.monotonicity, 1.0)
        self.assertAlmostEqual(falling_bad.monotonicity, 1.0)

    def test_threshold_and_aggregate_evidence(self):
        feature = extract_degradation_features(
            [0, 1, 2, 3],
            [0.1, 0.2, 0.6, 0.8],
            ResponseSpec("meter_error", warning_threshold=0.5),
        )
        evidence = aggregate_degradation_evidence([feature])
        self.assertAlmostEqual(feature.threshold_exceedance_ratio, 0.5)
        self.assertGreater(evidence["health_risk_index"], 0.0)
        self.assertLessEqual(evidence["health_risk_index"], 1.0)

    def test_collate_pads_variable_length_sequences(self):
        items = [
            DegradationSeries("capacitor", "c1", [0, 1], [[10], [10]], [[1], [2]], 0.2),
            DegradationSeries("capacitor", "c2", [0, 1, 2], [[12], [12], [12]], [[1], [2], [3]], 0.8),
        ]
        batch = collate_degradation_series(items)
        self.assertEqual(tuple(batch["responses"].shape), (2, 3, 1))
        self.assertEqual(batch["mask"].tolist(), [[True, True, False], [True, True, True]])


try:
    import torch
    from app.services.unified_degradation_model import (
        CausalPhysicsUDMNet,
        UnifiedDegradationConfig,
        UnifiedDegradationNet,
        build_ablation_configs,
        degradation_model_loss,
        train_multidevice_epoch,
    )
    from torch.utils.data import DataLoader
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class UnifiedModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = UnifiedDegradationConfig(
            response_dims={"capacitor": 2, "mosfet": 3},
            response_directions={"capacitor": (-1, 1), "mosfet": (1, 1, 1)},
            stress_dim=3,
            d_model=16,
            nhead=4,
            num_layers=1,
            dropout=0.1,
        )
        self.model = UnifiedDegradationNet(self.config)

    def test_forward_and_composite_loss(self):
        responses = torch.tensor([
            [[100.0, 0.10], [97.0, 0.12], [94.0, 0.15], [90.0, 0.19]],
            [[100.0, 0.10], [99.0, 0.11], [98.0, 0.12], [97.0, 0.13]],
        ])
        stresses = torch.tensor([
            [[10.0, 25.0, 0.4]] * 4,
            [[12.0, 25.0, 0.4]] * 4,
        ])
        times = torch.tensor([[0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0]])
        mask = torch.ones(2, 4, dtype=torch.bool)
        output = self.model("capacitor", responses, stresses, times, mask)
        losses = degradation_model_loss(output, torch.tensor([0.8, 0.3]), self.config)
        self.assertEqual(tuple(output["prediction"].shape), (2,))
        self.assertEqual(tuple(output["trajectory"].shape), (2, 4))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(self.model.response_adapters["capacitor"].network[1].weight.grad)

    def test_mc_uncertainty_returns_all_components(self):
        self.model.eval()
        responses = torch.tensor([[[1.0, 0.1], [0.9, 0.2], [0.8, 0.3]]])
        stresses = torch.tensor([[[10.0, 25.0, 0.5]] * 3])
        times = torch.tensor([[0.0, 1.0, 2.0]])
        result = self.model.predict_with_uncertainty(
            "capacitor", responses, stresses, times, mc_samples=3
        )
        self.assertIn("epistemic_variance", result)
        self.assertIn("aleatoric_variance", result)
        self.assertGreaterEqual(float(result["total_variance"][0]), 0.0)

    def test_research_ablation_configs_disable_one_component_at_a_time(self):
        configs = build_ablation_configs(self.config)
        self.assertTrue(configs["full_udm_net"].use_stress_conditioning)
        self.assertFalse(configs["without_stress_conditioning"].use_stress_conditioning)
        self.assertEqual(configs["without_monotonicity"].monotonicity_loss_weight, 0.0)
        self.assertEqual(configs["without_domain_alignment"].domain_loss_weight, 0.0)

    def test_causal_physics_variant_activates_auxiliary_losses(self):
        config = UnifiedDegradationConfig(
            response_dims={"capacitor": 2},
            response_directions={"capacitor": (-1, 1)},
            stress_dim=3,
            d_model=16,
            nhead=4,
            num_layers=1,
            use_latent_dynamics=True,
        )
        model = CausalPhysicsUDMNet(config)
        responses = torch.tensor([[[100.0, 0.10]], [[98.0, 0.12]]])
        stresses = torch.ones(2, 1, 3)
        times = torch.zeros(2, 1)
        output = model("capacitor", responses, stresses, times)
        self.assertEqual(tuple(output["forecast"].shape), (2, 0, 2))
        losses = degradation_model_loss(output, torch.tensor([0.2, 0.8]), config)
        for name in ("forecast", "ranking", "endpoint", "alignment"):
            self.assertIn(name, losses)
        self.assertGreater(float(losses["endpoint"].detach()), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(model.damage_rate_heads["capacitor"][1].weight.grad)

    def test_causal_physics_multidevice_epoch_uses_synchronized_gradients(self):
        config = UnifiedDegradationConfig(
            response_dims={"capacitor": 2, "mosfet": 3},
            response_directions={"capacitor": (-1, 1), "mosfet": (1, 1, 1)},
            stress_dim=3,
            d_model=8,
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_latent_dynamics=True,
        )
        model = CausalPhysicsUDMNet(config)

        def make_series(device_type, response_dim, offset):
            result = []
            for index, stage in enumerate((0.2, 0.5, 0.8, 0.95)):
                length = 3 + index % 2
                times = np.arange(length, dtype=np.float32)
                stresses = np.stack([
                    np.linspace(0.4 + offset, 0.7 + offset, length),
                    np.linspace(0.8, 1.1, length),
                    np.ones(length),
                ], axis=1)
                response_columns = [
                    1.0 + (column + 1) * stage * times / max(length - 1, 1)
                    for column in range(response_dim)
                ]
                responses = np.stack(response_columns, axis=1)
                result.append(DegradationSeries(
                    device_type,
                    f"{device_type}-{index}",
                    times,
                    stresses,
                    responses,
                    stage,
                ))
            return result

        loaders = {
            "capacitor": DataLoader(
                SameDeviceSeriesDataset(make_series("capacitor", 2, 0.0)),
                batch_size=2,
                collate_fn=collate_degradation_series,
            ),
            "mosfet": DataLoader(
                SameDeviceSeriesDataset(make_series("mosfet", 3, 0.1)),
                batch_size=2,
                collate_fn=collate_degradation_series,
            ),
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        parameter_before = model.alignment_projection[1].weight.detach().clone()
        metrics = train_multidevice_epoch(model, loaders, optimizer)

        self.assertEqual(
            set(metrics),
            {"total", "data", "monotonicity", "smoothness", "domain", "physics",
             "forecast", "ranking", "endpoint", "alignment"},
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        for name in ("forecast", "ranking", "endpoint", "alignment"):
            self.assertGreater(metrics[name], 0.0)
        self.assertFalse(torch.equal(parameter_before, model.alignment_projection[1].weight))


if __name__ == "__main__":
    unittest.main()
