import unittest

try:
    import torch

    from app.services.unified_degradation_model import (
        UnifiedDegradationConfig,
        degradation_model_loss,
    )
    from experiments.unified_degradation.audit_physics_prior_identifiability import (
        centered_log_rate_components,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PhysicsPriorIdentifiabilityAuditTests(unittest.TestCase):
    def test_composite_physics_component_is_invariant_to_per_sequence_scale(self):
        config = UnifiedDegradationConfig(response_dims={"test": 1}, stress_dim=1)
        mask = torch.ones(2, 4, dtype=torch.bool)
        physics_rate = torch.tensor([
            [0.8, 1.1, 1.7, 2.4],
            [2.0, 1.6, 1.2, 0.9],
        ])
        output = {
            "prediction": torch.tensor([0.3, 0.7]),
            "variance": torch.tensor([0.2, 0.2]),
            "trajectory": torch.tensor([
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
            ]),
            "mask": mask,
            "domain_logits": torch.zeros(2, 1),
            "domain_target": torch.zeros(2, dtype=torch.long),
            "physics_rate": physics_rate,
            "damage_rate": torch.tensor([
                [0.7, 1.0, 1.4, 2.1],
                [1.8, 1.5, 1.4, 1.0],
            ]),
        }
        base = degradation_model_loss(output, torch.tensor([0.4, 0.6]), config)
        scaled_output = dict(output)
        scaled_output["physics_rate"] = physics_rate * torch.tensor(
            [[3.5], [0.25]]
        )
        self.assertTrue(torch.all(scaled_output["physics_rate"] > 1e-6))
        scaled = degradation_model_loss(
            scaled_output, torch.tensor([0.4, 0.6]), config
        )
        self.assertTrue(torch.allclose(base["physics"], scaled["physics"], atol=1e-6))
        self.assertTrue(torch.allclose(base["total"], scaled["total"], atol=1e-6))

    def test_constant_prior_has_zero_centered_log_shape_variance(self):
        learned = torch.tensor([[0.8, 1.0, 1.2, 1.4]])
        prior = torch.full((1, 4), 2.5)
        components = centered_log_rate_components(
            learned, prior, torch.ones_like(prior, dtype=torch.bool)
        )
        self.assertAlmostEqual(float(components["prior_variance"][0]), 0.0)
        self.assertGreater(float(components["learned_variance"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
