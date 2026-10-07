"""Stress-aware multi-device degradation model.

UDM-Net implements four model properties required by the research design:

1. device-specific response adapters preserve heterogeneous measurements;
2. stress-conditioned shared temporal encoding learns common degradation laws;
3. monotonic and smoothness losses inject degradation-process constraints;
4. domain-adversarial alignment and heteroscedastic output support transfer and
   uncertainty-aware decisions.

One mini-batch contains one device type. Training alternates mini-batches from
capacitor, IGBT, MOSFET, meter, or other registered datasets so the temporal
encoder is shared while response adapters and output heads remain device-aware.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle


def _validate_device_name(name: str) -> None:
    if not name or "." in name:
        raise ValueError("device names must be non-empty and cannot contain '.'")


@dataclass
class UnifiedDegradationConfig:
    response_dims: dict[str, int]
    stress_dim: int
    response_directions: dict[str, tuple[int, ...]] = field(default_factory=dict)
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.15
    use_stress_conditioning: bool = True
    domain_loss_weight: float = 0.10
    monotonicity_loss_weight: float = 0.25
    smoothness_loss_weight: float = 0.05
    physics_loss_weight: float = 0.05
    domain_adversarial_strength: float = 0.20
    # The causal physics-guided variant is opt-in so existing deployments and
    # serialized legacy checkpoints remain compatible.
    use_latent_dynamics: bool = False
    forecast_loss_weight: float = 0.10
    ranking_loss_weight: float = 0.10
    endpoint_consistency_weight: float = 0.10
    alignment_loss_weight: float = 0.05
    alignment_temperature: float = 0.25
    include_absolute_duration: bool = False

    def __post_init__(self):
        if not self.response_dims:
            raise ValueError("response_dims cannot be empty")
        if self.stress_dim <= 0:
            raise ValueError("stress_dim must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.alignment_temperature <= 0:
            raise ValueError("alignment_temperature must be positive")
        for field_name in (
            "domain_loss_weight",
            "monotonicity_loss_weight",
            "smoothness_loss_weight",
            "physics_loss_weight",
            "forecast_loss_weight",
            "ranking_loss_weight",
            "endpoint_consistency_weight",
            "alignment_loss_weight",
            "domain_adversarial_strength",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for name, dim in self.response_dims.items():
            _validate_device_name(name)
            if dim <= 0:
                raise ValueError(f"response dimension for {name} must be positive")
            directions = self.response_directions.get(name, (1,) * dim)
            if len(directions) != dim or any(v not in (-1, 1) for v in directions):
                raise ValueError(f"invalid response directions for {name}")


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


def gradient_reverse(x: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReversal.apply(x, strength)


class DeviceResponseAdapter(nn.Module):
    def __init__(self, response_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(response_dim),
            nn.Linear(response_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, degradation_response: torch.Tensor) -> torch.Tensor:
        return self.network(degradation_response)


class UnifiedDegradationNet(nn.Module):
    """Unified Degradation Model Network (UDM-Net)."""

    def __init__(self, config: UnifiedDegradationConfig):
        super().__init__()
        self.config = config
        self.device_names = list(config.response_dims)
        self.device_to_index = {name: idx for idx, name in enumerate(self.device_names)}

        self.response_adapters = nn.ModuleDict({
            name: DeviceResponseAdapter(dim, config.d_model, config.dropout)
            for name, dim in config.response_dims.items()
        })
        self.device_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 2),
            )
            for name in config.response_dims
        })
        # Normalized trajectory statistics are complemented by raw statistics.
        # The latter are the same information family used by the classical
        # baseline, but learned jointly instead of being split into trees.
        summary_dim = lambda response_dim: response_dim * 7 + config.stress_dim * 2 + 1
        self.summary_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(summary_dim(dim)),
                nn.Linear(summary_dim(dim), config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            for name, dim in config.response_dims.items()
        })
        self.summary_output_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 2),
            )
            for name in config.response_dims
        })
        for name, dim in config.response_dims.items():
            directions = config.response_directions.get(name, (1,) * dim)
            self.register_buffer(f"direction_{name}", torch.tensor(directions, dtype=torch.float32))

        self.stress_encoder = nn.Sequential(
            nn.LayerNorm(config.stress_dim),
            nn.Linear(config.stress_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.stress_film = nn.Linear(config.d_model, config.d_model * 2)
        self.time_encoder = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(config.d_model, 1)
        self.trajectory_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.domain_classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, len(self.device_names)),
        )

    def _direction(self, device_type: str, reference: torch.Tensor) -> torch.Tensor:
        if device_type not in self.response_adapters:
            raise KeyError(f"unregistered device type: {device_type}")
        return getattr(self, f"direction_{device_type}").to(reference)

    @staticmethod
    def _time_features(times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        start = times[:, :1]
        elapsed = torch.clamp(times - start, min=0.0)
        duration = elapsed.masked_fill(~mask, 0.0).amax(dim=1, keepdim=True).clamp_min(1e-6)
        elapsed = elapsed / duration
        delta = torch.zeros_like(elapsed)
        delta[:, 1:] = torch.clamp(times[:, 1:] - times[:, :-1], min=0.0) / duration
        return torch.stack([elapsed, delta], dim=-1)

    def _normalize_response(self, device_type: str, responses: torch.Tensor) -> torch.Tensor:
        baseline = responses[:, :1, :]
        median_scale = responses.detach().abs().median(dim=1, keepdim=True).values
        scale = torch.maximum(baseline.abs(), median_scale).clamp_min(1e-6)
        direction = self._direction(device_type, responses).view(1, 1, -1)
        return direction * (responses - baseline) / scale

    def _raw_physics_rate(self, device_type: str, stresses: torch.Tensor) -> torch.Tensor:
        """Compute an uncalibrated, stress-only rate proxy.

        Material constants are deliberately fixed to broad literature-scale
        values.  The result is used only for relative rate shape; it is never
        interpreted as a calibrated lifetime or failure probability.
        """
        if device_type == "capacitor":
            voltage_ratio = stresses[..., 0].clamp(0.0, 4.0)
            temperature = 298.15 + 40.0 * voltage_ratio
            rate = torch.exp(-0.30 / (8.617333262e-5 * temperature)) * voltage_ratio.pow(2.0)
        elif device_type in {"igbt", "mosfet"}:
            temperature = (273.15 + stresses[..., 0].clamp(-0.2, 1.5) * 300.0).clamp(200.0, 700.0)
            delta_temperature = (stresses[..., 0] * 300.0 - 25.0).abs().clamp_min(1.0)
            exponent = 1.6 if device_type == "igbt" else 1.8
            rate = torch.exp(-0.42 / (8.617333262e-5 * temperature)) * delta_temperature.pow(exponent) * stresses[..., 1].clamp(0.0, 4.0).pow(1.5)
        else:
            return torch.zeros_like(stresses[..., 0])
        return rate

    def _physics_prior(self, device_type: str, stresses: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a normalized stress-only damage trajectory for legacy use."""
        rate = self._raw_physics_rate(device_type, stresses)
        elapsed = torch.zeros_like(times)
        elapsed[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(0.0)
        increments = rate * elapsed * mask.float()
        cumulative = torch.cumsum(increments, dim=1)
        final = cumulative.masked_fill(~mask, 0.0).amax(dim=1, keepdim=True).clamp_min(1e-6)
        return (cumulative / final).masked_fill(~mask, 0.0).clamp(0.0, 1.0)

    def forward(
        self,
        device_type: str,
        responses: torch.Tensor,
        stresses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one same-device padded batch.

        Shapes: responses ``[B,T,R_device]``, stresses ``[B,T,S]``, times
        ``[B,T]`` and mask ``[B,T]`` where ``True`` marks real observations.
        """
        if responses.ndim != 3 or stresses.ndim != 3 or times.ndim != 2:
            raise ValueError("expected responses/stresses/times with ranks 3/3/2")
        if responses.shape[:2] != stresses.shape[:2] or responses.shape[:2] != times.shape:
            raise ValueError("batch and time dimensions must match")
        if stresses.shape[-1] != self.config.stress_dim:
            raise ValueError("stress feature dimension does not match config")
        expected_response_dim = self.config.response_dims.get(device_type)
        if expected_response_dim is None or responses.shape[-1] != expected_response_dim:
            raise ValueError("response feature dimension does not match device config")
        if mask is None:
            mask = torch.ones_like(times, dtype=torch.bool)
        else:
            mask = mask.bool()

        degradation = self._normalize_response(device_type, responses)
        response_token = self.response_adapters[device_type](degradation)
        if self.config.use_stress_conditioning:
            stress_token = self.stress_encoder(stresses)
            gamma, beta = self.stress_film(stress_token).chunk(2, dim=-1)
            conditioned = response_token * (1.0 + 0.1 * torch.tanh(gamma)) + beta
        else:
            conditioned = response_token
        conditioned = conditioned + self.time_encoder(self._time_features(times, mask))

        device_idx = self.device_to_index[device_type]
        device_ids = torch.full((responses.shape[0],), device_idx, dtype=torch.long, device=responses.device)
        encoded = self.temporal_encoder(conditioned, src_key_padding_mask=~mask)

        pool_logits = self.pool_score(encoded).squeeze(-1).masked_fill(~mask, -1e9)
        pool_weights = torch.softmax(pool_logits, dim=1)
        representation = torch.sum(encoded * pool_weights.unsqueeze(-1), dim=1)

        # Preserve the low-dimensional trajectory information that classical
        # baselines exploit: final state, mean, variability, stress summary and
        # the absolute duration of the observed prefix.  The Transformer still
        # models temporal interactions; this branch prevents small datasets from
        # losing the most informative degradation statistics in attention pooling.
        valid_count = mask.sum(dim=1).clamp_min(1)
        batch_index = torch.arange(responses.shape[0], device=responses.device)
        last_index = valid_count - 1
        last_response = degradation[batch_index, last_index]
        mask_float = mask.unsqueeze(-1).float()
        response_mean = (degradation * mask_float).sum(dim=1) / valid_count.unsqueeze(-1)
        response_var = (((degradation - response_mean.unsqueeze(1)) * mask_float).pow(2).sum(dim=1)
                        / valid_count.unsqueeze(-1)).sqrt()
        raw_last_response = responses[batch_index, last_index]
        raw_response_mean = (responses * mask_float).sum(dim=1) / valid_count.unsqueeze(-1)
        raw_response_var = (((responses - raw_response_mean.unsqueeze(1)) * mask_float).pow(2).sum(dim=1)
                            / valid_count.unsqueeze(-1)).sqrt()
        raw_response_delta = raw_last_response - responses[:, :1, :].squeeze(1)
        stress_mean = (stresses * mask_float).sum(dim=1) / valid_count.unsqueeze(-1)
        stress_max = stresses.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
        duration = times.masked_fill(~mask, 0.0).amax(dim=1, keepdim=True) - times[:, :1]
        duration = torch.log1p(duration.clamp_min(0.0))
        # Keep the legacy summary shape for checkpoint compatibility, while
        # allowing leakage-safe experiments to zero the absolute-duration
        # channel explicitly.
        if not self.config.include_absolute_duration:
            duration = torch.zeros_like(duration)
        summary = torch.cat(
            [last_response, response_mean, response_var,
             raw_last_response, raw_response_mean, raw_response_var, raw_response_delta,
             stress_mean, stress_max, duration], dim=-1
        )
        summary_token = self.summary_heads[device_type](summary)
        representation = representation + summary_token

        head_output = self.device_heads[device_type](representation)
        # A parallel summary head makes the low-dimensional degradation signal
        # directly predictive while the temporal head captures residual dynamics.
        head_output = head_output + 0.5 * self.summary_output_heads[device_type](summary_token)
        prediction = torch.sigmoid(head_output[:, 0])
        variance = F.softplus(head_output[:, 1]) + 1e-6
        trajectory = torch.sigmoid(self.trajectory_head(encoded).squeeze(-1)).masked_fill(~mask, 0.0)
        physics_prior = self._physics_prior(device_type, stresses, times, mask)
        reversed_representation = gradient_reverse(
            representation,
            self.config.domain_adversarial_strength,
        )
        domain_logits = self.domain_classifier(reversed_representation)
        return {
            "prediction": prediction,
            "variance": variance,
            "trajectory": trajectory,
            "physics_prior": physics_prior,
            "representation": representation,
            "domain_logits": domain_logits,
            "domain_target": device_ids,
            "attention": pool_weights,
            "normalized_degradation": degradation,
            "mask": mask,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        device_type: str,
        responses: torch.Tensor,
        stresses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        mc_samples: int = 20,
    ) -> dict[str, torch.Tensor]:
        """Estimate aleatoric and epistemic uncertainty with MC dropout.

        The method belongs to the base class so both the legacy and causal
        variants expose the same inference contract.
        """
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least 2")
        was_training = self.training
        self.train()
        predictions, variances = [], []
        for _ in range(mc_samples):
            output = self(device_type, responses, stresses, times, mask)
            predictions.append(output["prediction"])
            variances.append(output["variance"])
        if not was_training:
            self.eval()
        prediction_stack = torch.stack(predictions)
        variance_stack = torch.stack(variances)
        return {
            "prediction_mean": prediction_stack.mean(dim=0),
            "epistemic_variance": prediction_stack.var(dim=0, unbiased=False),
            "aleatoric_variance": variance_stack.mean(dim=0),
            "total_variance": prediction_stack.var(dim=0, unbiased=False) + variance_stack.mean(dim=0),
        }


class CausalPhysicsUDMNet(UnifiedDegradationNet):
    """Causal Physics-guided Unified Degradation Network (CP-UDM-Net).

    The legacy UDM-Net regresses a prefix stage from a bidirectional sequence
    representation.  This variant makes the latent damage state the central
    object: a positive, stress-conditioned damage rate is integrated over the
    observed path, so the trajectory is monotone by construction.  The stage
    head is trained jointly with next-step response forecasting and ordinal
    prefix ranking.  No absolute prefix duration is used as a summary feature.
    """

    def __init__(self, config: UnifiedDegradationConfig):
        super().__init__(config)
        if not config.use_latent_dynamics:
            raise ValueError("CausalPhysicsUDMNet requires use_latent_dynamics=True")

        self.damage_rate_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 1),
            )
            for name in config.response_dims
        })
        self.forecast_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, dim),
            )
            for name, dim in config.response_dims.items()
        })
        self.stage_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 2),
            )
            for name in config.response_dims
        })
        self.duration_blind_summary_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(dim * 5 + config.stress_dim * 2),
                nn.Linear(dim * 5 + config.stress_dim * 2, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            for name, dim in config.response_dims.items()
        })
        self.alignment_projection = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )

    @staticmethod
    def _masked_slope(values: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fit a differentiable least-squares slope on local normalized time."""
        time_features = UnifiedDegradationNet._time_features(times, mask)
        x = time_features[..., 0]
        weights = mask.float()
        count = weights.sum(dim=1, keepdim=True).clamp_min(2.0)
        x_mean = (x * weights).sum(dim=1, keepdim=True) / count
        y_mean = (values * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        numerator = ((x - x_mean).unsqueeze(-1) * (values - y_mean) * weights.unsqueeze(-1)).sum(dim=1)
        denominator = ((x - x_mean).pow(2) * weights).sum(dim=1, keepdim=True).clamp_min(1e-6)
        return numerator / denominator

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _physics_rate_from_prior(
        self,
        device_type: str,
        stresses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a sequence-normalized stress dose rate for weak supervision.

        Normalizing by the valid-sequence mean removes the unknown material
        scale and prevents the prefix endpoint from being forced to one.  The
        learned damage-rate head remains responsible for absolute stage scale.
        """
        raw_rate = self._raw_physics_rate(device_type, stresses).masked_fill(~mask, 0.0)
        valid = mask.float()
        mean_rate = (raw_rate * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (raw_rate / mean_rate.clamp_min(1e-6)).masked_fill(~mask, 0.0)

    def forward(
        self,
        device_type: str,
        responses: torch.Tensor,
        stresses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if responses.ndim != 3 or stresses.ndim != 3 or times.ndim != 2:
            raise ValueError("expected responses/stresses/times with ranks 3/3/2")
        if responses.shape[:2] != stresses.shape[:2] or responses.shape[:2] != times.shape:
            raise ValueError("batch and time dimensions must match")
        if stresses.shape[-1] != self.config.stress_dim:
            raise ValueError("stress feature dimension does not match config")
        expected_response_dim = self.config.response_dims.get(device_type)
        if expected_response_dim is None or responses.shape[-1] != expected_response_dim:
            raise ValueError("response feature dimension does not match device config")
        mask = torch.ones_like(times, dtype=torch.bool) if mask is None else mask.bool()

        degradation = self._normalize_response(device_type, responses)
        response_token = self.response_adapters[device_type](degradation)
        if self.config.use_stress_conditioning:
            stress_token = self.stress_encoder(stresses)
            gamma, beta = self.stress_film(stress_token).chunk(2, dim=-1)
            conditioned = response_token * (1.0 + 0.1 * torch.tanh(gamma)) + beta
        else:
            conditioned = response_token
        conditioned = conditioned + self.time_encoder(self._time_features(times, mask))

        encoded = self.temporal_encoder(
            conditioned,
            mask=self._causal_mask(conditioned.shape[1], conditioned.device),
            src_key_padding_mask=~mask,
        )
        valid_count = mask.sum(dim=1).clamp_min(1)
        batch_index = torch.arange(responses.shape[0], device=responses.device)
        last_index = valid_count - 1
        last_hidden = encoded[batch_index, last_index]
        mask_float = mask.unsqueeze(-1).float()
        response_mean = (degradation * mask_float).sum(dim=1) / valid_count.unsqueeze(-1)
        response_var = (((degradation - response_mean.unsqueeze(1)) * mask_float).pow(2).sum(dim=1)
                        / valid_count.unsqueeze(-1)).sqrt()
        raw_last = responses[batch_index, last_index]
        raw_delta = raw_last - responses[:, :1, :].squeeze(1)
        slope = self._masked_slope(degradation, times, mask)
        stress_mean = (stresses * mask_float).sum(dim=1) / valid_count.unsqueeze(-1)
        stress_max = stresses.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
        summary = torch.cat([
            degradation[batch_index, last_index], response_mean, response_var,
            raw_delta, slope, stress_mean, stress_max,
        ], dim=-1)
        representation = last_hidden + self.duration_blind_summary_heads[device_type](summary)

        # Positive rate + integration is the model's latent damage state. The
        # local time increments are normalized within the observed prefix, so
        # the network cannot recover the target from absolute duration alone.
        rate = F.softplus(self.damage_rate_heads[device_type](encoded).squeeze(-1))
        local_delta = self._time_features(times, mask)[..., 1]
        increments = rate * local_delta * mask.float()
        cumulative_damage = torch.cumsum(increments, dim=1)
        trajectory = (1.0 - torch.exp(-cumulative_damage)).masked_fill(~mask, 0.0)
        dynamic_endpoint = trajectory[batch_index, last_index]

        stage_output = self.stage_heads[device_type](representation)
        direct_stage = torch.sigmoid(stage_output[:, 0])
        prediction = 0.5 * direct_stage + 0.5 * dynamic_endpoint
        variance = F.softplus(stage_output[:, 1]) + 1e-6
        # Keep an empty one-step forecast's last dimension equal to the
        # device response dimension.  Returning ``d_model`` here makes the
        # masked loss fail for valid single-observation sequences.
        forecast = self.forecast_heads[device_type](encoded[:, :-1]) if encoded.shape[1] > 1 else self.forecast_heads[device_type](encoded[:, :0, :])
        forecast_mask = mask[:, 1:] & mask[:, :-1] if encoded.shape[1] > 1 else mask[:, :0]
        forecast_target = degradation[:, 1:] if encoded.shape[1] > 1 else degradation[:, :0]

        device_idx = self.device_to_index[device_type]
        device_ids = torch.full((responses.shape[0],), device_idx, dtype=torch.long, device=responses.device)
        reversed_representation = gradient_reverse(representation, self.config.domain_adversarial_strength)
        domain_logits = self.domain_classifier(reversed_representation)
        stage_embedding = F.normalize(self.alignment_projection(representation), dim=-1)
        physics_prior = self._physics_prior(device_type, stresses, times, mask)
        physics_rate = self._physics_rate_from_prior(device_type, stresses, times, mask)

        pool_logits = self.pool_score(encoded).squeeze(-1).masked_fill(~mask, -1e9)
        pool_weights = torch.softmax(pool_logits, dim=1)
        return {
            "prediction": prediction,
            "variance": variance,
            "trajectory": trajectory,
            "dynamic_endpoint": dynamic_endpoint,
            "damage_rate": rate.masked_fill(~mask, 0.0),
            "physics_prior": physics_prior,
            "physics_rate": physics_rate,
            "forecast": forecast,
            "forecast_target": forecast_target,
            "forecast_mask": forecast_mask,
            "representation": representation,
            "stage_embedding": stage_embedding,
            "domain_logits": domain_logits,
            "domain_target": device_ids,
            "attention": pool_weights,
            "normalized_degradation": degradation,
            "mask": mask,
        }

def _pairwise_stage_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    embedding: torch.Tensor | None = None,
    margin: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return order and representation-distance losses for one mini-batch.

    Prefix labels are ordinal rather than categorical.  A pairwise objective is
    therefore useful for the latent-dynamics variant, while remaining a no-op
    for batches with fewer than two distinct stages.  The embedding term keeps
    the optional stage representation geometrically aligned with stage gaps.
    """
    prediction = prediction.reshape(-1)
    target = target.detach().reshape(-1)
    if prediction.numel() < 2:
        zero = prediction.new_tensor(0.0)
        return zero, zero

    target_delta = target.unsqueeze(1) - target.unsqueeze(0)
    prediction_delta = prediction.unsqueeze(1) - prediction.unsqueeze(0)
    pair_mask = target_delta.abs() > 1e-6
    signed_delta = prediction_delta * target_delta.sign()
    ranking_loss = (
        F.relu(float(margin) - signed_delta)[pair_mask].mean()
        if pair_mask.any()
        else prediction.new_tensor(0.0)
    )

    if embedding is None or embedding.shape[0] < 2:
        return ranking_loss, prediction.new_tensor(0.0)
    embedding = F.normalize(embedding, dim=-1)
    upper = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
    upper = upper & pair_mask
    if not upper.any():
        return ranking_loss, prediction.new_tensor(0.0)
    embedding_distance = torch.cdist(embedding, embedding, p=2)
    target_distance = target_delta.abs()
    alignment_loss = F.smooth_l1_loss(
        embedding_distance[upper], target_distance[upper], reduction="mean"
    )
    return ranking_loss, alignment_loss


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 mean that safely handles an empty padded sequence mask."""
    mask = mask.bool()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1).expand_as(prediction)
    if not mask.any():
        return prediction.new_tensor(0.0)
    return F.smooth_l1_loss(prediction[mask], target[mask], reduction="mean")


def cross_device_stage_alignment_loss(
    embeddings: list[torch.Tensor],
    targets: list[torch.Tensor],
    device_ids: list[torch.Tensor],
    temperature: float = 0.25,
    margin: float = 0.50,
) -> torch.Tensor:
    """Align latent stages across devices without collapsing device identity.

    Pairs from different devices with nearby relative stages are pulled
    together, while pairs with a large stage gap are separated by a margin.
    The soft weight avoids an arbitrary binary definition of ``same stage``
    and makes the objective usable when only relative progress labels exist.
    """
    if len(embeddings) < 2:
        if embeddings:
            return embeddings[0].new_tensor(0.0)
        return torch.tensor(0.0)
    embedding = F.normalize(torch.cat(embeddings, dim=0), dim=-1)
    target = torch.cat([item.reshape(-1) for item in targets], dim=0).detach()
    domains = torch.cat([item.reshape(-1) for item in device_ids], dim=0)
    if embedding.shape[0] < 2:
        return embedding.new_tensor(0.0)
    pair_mask = torch.triu(
        torch.ones((embedding.shape[0], embedding.shape[0]), dtype=torch.bool, device=embedding.device),
        diagonal=1,
    )
    pair_mask &= domains.unsqueeze(1) != domains.unsqueeze(0)
    if not pair_mask.any():
        return embedding.new_tensor(0.0)
    distances = torch.cdist(embedding, embedding, p=2)
    stage_gap = (target.unsqueeze(1) - target.unsqueeze(0)).abs()
    close_weight = torch.exp(-stage_gap / max(float(temperature), 1e-6))
    pair_loss = (
        close_weight * distances.pow(2)
        + (1.0 - close_weight) * F.relu(float(margin) - distances).pow(2)
    )
    return pair_loss[pair_mask].mean()


def degradation_model_loss(
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: UnifiedDegradationConfig,
) -> dict[str, torch.Tensor]:
    """Compute legacy losses and the opt-in latent-dynamics objectives.

    The original UDM output does not contain the optional keys and therefore
    follows the original loss exactly.  ``CausalPhysicsUDMNet`` emits those
    keys, activating forecast, ordinal ranking, endpoint consistency,
    representation alignment, and shape-normalized physics-rate losses.
    """
    prediction = output["prediction"]
    variance = output["variance"]
    target = target.float().reshape_as(prediction)
    data_loss = 0.5 * (torch.log(variance) + (target - prediction).pow(2) / variance)
    data_loss = data_loss.mean()

    trajectory = output["trajectory"]
    mask = output["mask"].bool()
    pair_mask = mask[:, 1:] & mask[:, :-1]
    backwards = F.relu(trajectory[:, :-1] - trajectory[:, 1:])
    monotonicity_loss = (
        backwards[pair_mask].mean() if pair_mask.any() else prediction.new_tensor(0.0)
    )

    if trajectory.shape[1] >= 3:
        second_difference = trajectory[:, 2:] - 2.0 * trajectory[:, 1:-1] + trajectory[:, :-2]
        triple_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        smoothness_loss = (
            second_difference[triple_mask].abs().mean()
            if triple_mask.any()
            else prediction.new_tensor(0.0)
        )
    else:
        smoothness_loss = prediction.new_tensor(0.0)

    domain_loss = F.cross_entropy(output["domain_logits"], output["domain_target"])
    physics_prior = output.get("physics_prior")
    physics_rate = output.get("physics_rate")
    if physics_rate is not None and output.get("damage_rate") is not None:
        # The prior and learned rate have arbitrary absolute scales.  Compare
        # their within-sequence log-rate shape instead of forcing an uncalibrated
        # numerical scale onto the network.
        learned_rate = output["damage_rate"].clamp_min(1e-6)
        prior_rate = physics_rate.clamp_min(1e-6)
        rate_mask = mask & (physics_rate > 0.0)
        if rate_mask.any():
            learned_log = torch.log(learned_rate)
            prior_log = torch.log(prior_rate)
            count = rate_mask.sum(dim=1, keepdim=True).clamp_min(1)
            learned_mean = (learned_log * rate_mask.float()).sum(dim=1, keepdim=True) / count
            prior_mean = (prior_log * rate_mask.float()).sum(dim=1, keepdim=True) / count
            residual = (learned_log - learned_mean) - (prior_log - prior_mean)
            physics_loss = residual[rate_mask].pow(2).mean()
        else:
            physics_loss = prediction.new_tensor(0.0)
    elif physics_prior is not None:
        # Legacy UDM compatibility path.
        physics_mask = mask.float()
        physics_loss = ((trajectory - physics_prior).pow(2) * physics_mask).sum() / physics_mask.sum().clamp_min(1.0)
    else:
        physics_loss = prediction.new_tensor(0.0)

    forecast_loss = prediction.new_tensor(0.0)
    if output.get("forecast") is not None and output.get("forecast_target") is not None:
        forecast_loss = _masked_smooth_l1(
            output["forecast"],
            output["forecast_target"].detach(),
            output.get("forecast_mask", mask[:, : output["forecast"].shape[1]]),
        )

    ranking_loss = prediction.new_tensor(0.0)
    alignment_loss = prediction.new_tensor(0.0)
    endpoint_loss = prediction.new_tensor(0.0)
    if output.get("dynamic_endpoint") is not None:
        dynamic_endpoint = output["dynamic_endpoint"]
        endpoint_loss = F.smooth_l1_loss(dynamic_endpoint, target, reduction="mean")
        ranking_loss, alignment_loss = _pairwise_stage_losses(
            dynamic_endpoint,
            target,
            output.get("stage_embedding"),
        )

    total = (
        data_loss
        + config.monotonicity_loss_weight * monotonicity_loss
        + config.smoothness_loss_weight * smoothness_loss
        + config.domain_loss_weight * domain_loss
        + config.physics_loss_weight * physics_loss
        + config.forecast_loss_weight * forecast_loss
        + config.ranking_loss_weight * ranking_loss
        + config.endpoint_consistency_weight * endpoint_loss
        + config.alignment_loss_weight * alignment_loss
    )
    return {
        "total": total,
        "data": data_loss,
        "monotonicity": monotonicity_loss,
        "smoothness": smoothness_loss,
        "domain": domain_loss,
        "physics": physics_loss,
        "forecast": forecast_loss,
        "ranking": ranking_loss,
        "endpoint": endpoint_loss,
        "alignment": alignment_loss,
    }


def train_multidevice_epoch(
    model: UnifiedDegradationNet,
    loaders: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Alternate device-specific batches while updating the shared encoder."""
    model.train()
    totals = {key: 0.0 for key in (
        "total", "data", "monotonicity", "smoothness", "domain", "physics",
        "forecast", "ranking", "endpoint", "alignment",
    )}
    steps = 0
    # Balance device contributions.  The public datasets have very different
    # numbers of runs; iterating each loader once would let the largest dataset
    # dominate the shared encoder.  Cycling shorter loaders gives every device
    # the same number of parameter updates per epoch.
    loader_items = list(loaders.items())
    if not loader_items:
        raise ValueError("loaders produced no batches")
    steps_per_device = max(len(loader) for _, loader in loader_items)

    # The causal variant updates the shared encoder once per synchronized
    # multi-device step.  This is important: computing an alignment loss after
    # separate optimizer steps would compare representations from different
    # parameter states and would not implement cross-device supervision.
    if model.config.use_latent_dynamics:
        iterators = {name: cycle(loader) for name, loader in loader_items}
        for _ in range(steps_per_device):
            optimizer.zero_grad()
            batch_outputs = []
            batch_losses = []
            stage_targets = []
            stage_domains = []
            for device_type, _loader in loader_items:
                batch = next(iterators[device_type])
                responses = batch["responses"].to(device)
                stresses = batch["stresses"].to(device)
                times = batch["times"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)
                output = model(device_type, responses, stresses, times, mask)
                losses = degradation_model_loss(output, target, model.config)
                batch_outputs.append(output)
                batch_losses.append(losses)
                stage_targets.append(target)
                stage_domains.append(output["domain_target"])

            local_total = torch.stack([item["total"] for item in batch_losses]).mean()
            alignment = cross_device_stage_alignment_loss(
                [item["stage_embedding"] for item in batch_outputs],
                stage_targets,
                stage_domains,
                temperature=model.config.alignment_temperature,
            )
            total = local_total + model.config.alignment_loss_weight * alignment
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key in totals:
                if key == "total":
                    value = total
                elif key == "alignment":
                    value = torch.stack([
                        item.get("alignment", total.new_tensor(0.0)) for item in batch_losses
                    ]).mean() + alignment
                else:
                    value = torch.stack([
                        item.get(key, total.new_tensor(0.0)) for item in batch_losses
                    ]).mean()
                totals[key] += float(value.detach().cpu())
            steps += 1
        if steps == 0:
            raise ValueError("loaders produced no batches")
        return {key: value / steps for key, value in totals.items()}

    for device_type, loader in loader_items:
        for batch_index, batch in zip(range(steps_per_device), cycle(loader)):
            responses = batch["responses"].to(device)
            stresses = batch["stresses"].to(device)
            times = batch["times"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            output = model(device_type, responses, stresses, times, mask)
            losses = degradation_model_loss(output, target, model.config)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses.get(key, losses["total"].new_tensor(0.0)).detach().cpu())
            steps += 1
    if steps == 0:
        raise ValueError("loaders produced no batches")
    return {key: value / steps for key, value in totals.items()}


def get_unified_degradation_architecture() -> dict:
    return {
        "name": "CP-UDM-Net",
        "full_name": "Causal Physics-guided Unified Degradation Network",
        "innovations": [
            "器件专用响应适配器保留电表、电容、IGBT、MOSFET等测量特性",
            "上三角注意力掩码阻断固定观测前缀内的后续Token交互；前缀级归一化不等同于严格流式状态",
            "正值损伤速率积分使潜在退化轨迹在结构上单调",
            "下一时刻预测、前缀排序和端点一致性构成可证伪弱监督",
            "应力物理项只约束相对速率形状，不伪称已标定失效机理",
            "跨器件阶段对齐、异方差与MC Dropout支持迁移及不确定性复核",
        ],
        "training": "同步采样多器件批次，联合优化器件任务损失与跨器件阶段对齐",
    }


def build_ablation_configs(base: UnifiedDegradationConfig) -> dict[str, UnifiedDegradationConfig]:
    """Return standard ablations for estimating each component's contribution."""
    if base.use_latent_dynamics:
        return {
            "full_cp_udm_net": replace(base),
            "cp_without_stress_conditioning": replace(base, use_stress_conditioning=False),
            "cp_without_physics_rate": replace(base, physics_loss_weight=0.0),
            "cp_without_forecasting": replace(base, forecast_loss_weight=0.0),
            "cp_without_ranking": replace(base, ranking_loss_weight=0.0),
            "cp_without_cross_device_alignment": replace(
                base,
                alignment_loss_weight=0.0,
                domain_loss_weight=0.0,
                domain_adversarial_strength=0.0,
            ),
        }
    return {
        "full_udm_net": replace(base),
        "without_stress_conditioning": replace(base, use_stress_conditioning=False),
        "without_monotonicity": replace(base, monotonicity_loss_weight=0.0),
        "without_domain_alignment": replace(
            base,
            domain_loss_weight=0.0,
            domain_adversarial_strength=0.0,
        ),
        "without_process_constraints": replace(
            base,
            monotonicity_loss_weight=0.0,
            smoothness_loss_weight=0.0,
        ),
    }
