"""Monte Carlo calibration audit for batch-level Beta--Binomial evidence.

This experiment is deliberately independent from the application database and
the RA-MGQ neural network.  It checks when conjugate shrinkage helps estimate a
latent batch anomaly probability and when a misspecified prior hurts.  It is
SEMI-SYNTHETIC / MONTE CARLO EVIDENCE ONLY, not enterprise validation and not a
claim that the mature Beta--Binomial model is itself novel.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import norm


ESTIMATOR_NAMES = (
    "raw_rate_wilson",
    "beta_uniform",
    "beta_oracle",
    "beta_mismatched",
)


@dataclass(frozen=True)
class AuditConfig:
    seeds: tuple[int, ...] = (17, 42, 73)
    sample_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    batches_per_size_per_seed: int = 20_000
    true_alpha: float = 2.0
    true_beta: float = 8.0
    mismatched_alpha: float = 8.0
    mismatched_beta: float = 2.0
    interval_level: float = 0.90
    mc_confidence_level: float = 0.95
    nll_clip: float = 1e-6

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one Monte Carlo seed is required")
        if not self.sample_sizes or any(value < 1 for value in self.sample_sizes):
            raise ValueError("sample sizes must be positive integers")
        if self.batches_per_size_per_seed < 2:
            raise ValueError("at least two batches per size and seed are required")
        for name in (
            "true_alpha",
            "true_beta",
            "mismatched_alpha",
            "mismatched_beta",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError("interval_level must lie in (0, 1)")
        if not 0.0 < self.mc_confidence_level < 1.0:
            raise ValueError("mc_confidence_level must lie in (0, 1)")
        if not 0.0 < self.nll_clip < 0.5:
            raise ValueError("nll_clip must lie in (0, 0.5)")


def beta_posterior_mean_interval(
    positive_count: np.ndarray,
    sample_count: int,
    alpha: float,
    beta: float,
    interval_level: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return posterior means and equal-tailed intervals for latent risk."""
    counts = np.asarray(positive_count, dtype=np.float64)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if np.any((counts < 0.0) | (counts > sample_count)):
        raise ValueError("positive counts must lie in [0, sample_count]")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")

    posterior_alpha = alpha + counts
    posterior_beta = beta + float(sample_count) - counts
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    tail = (1.0 - interval_level) / 2.0
    lower = beta_distribution.ppf(tail, posterior_alpha, posterior_beta)
    upper = beta_distribution.ppf(1.0 - tail, posterior_alpha, posterior_beta)
    return posterior_mean, lower, upper


def wilson_interval(
    positive_count: np.ndarray,
    sample_count: int,
    interval_level: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Wilson score intervals paired with the unsmoothed raw rate."""
    counts = np.asarray(positive_count, dtype=np.float64)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if np.any((counts < 0.0) | (counts > sample_count)):
        raise ValueError("positive counts must lie in [0, sample_count]")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")

    z_value = float(norm.ppf(0.5 + interval_level / 2.0))
    rate = counts / float(sample_count)
    denominator = 1.0 + z_value**2 / sample_count
    center = (rate + z_value**2 / (2.0 * sample_count)) / denominator
    radius = (
        z_value
        * np.sqrt(
            rate * (1.0 - rate) / sample_count
            + z_value**2 / (4.0 * sample_count**2)
        )
        / denominator
    )
    return np.clip(center - radius, 0.0, 1.0), np.clip(center + radius, 0.0, 1.0)


def expected_binary_scores(
    latent_probability: np.ndarray,
    estimate: np.ndarray,
    nll_clip: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a fresh Bernoulli outcome in expectation, conditional on its risk.

    Using the conditional expectation removes unnecessary next-outcome sampling
    noise while retaining the standard one-step-ahead Brier and log scores.
    """
    probability = np.asarray(latent_probability, dtype=np.float64)
    prediction = np.asarray(estimate, dtype=np.float64)
    if probability.shape != prediction.shape:
        raise ValueError("latent probabilities and estimates must have equal shapes")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("latent probabilities must lie in [0, 1]")

    brier = probability * (1.0 - prediction) ** 2 + (1.0 - probability) * prediction**2
    clipped = np.clip(prediction, nll_clip, 1.0 - nll_clip)
    nll = -(
        probability * np.log(clipped)
        + (1.0 - probability) * np.log1p(-clipped)
    )
    return brier, nll


def _new_moments() -> dict[str, float | int]:
    return {"count": 0, "sum": 0.0, "sum_squares": 0.0}


def _update_moments(moments: dict[str, float | int], values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    moments["count"] = int(moments["count"]) + int(array.size)
    moments["sum"] = float(moments["sum"]) + float(array.sum(dtype=np.float64))
    moments["sum_squares"] = float(moments["sum_squares"]) + float(
        np.square(array).sum(dtype=np.float64)
    )


def _moment_summary(
    moments: dict[str, float | int],
    confidence_level: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float, float, float]:
    count = int(moments["count"])
    total = float(moments["sum"])
    total_squares = float(moments["sum_squares"])
    if count < 2:
        raise ValueError("Monte Carlo confidence intervals require at least two batches")
    mean = total / count
    variance = max((total_squares - total * total / count) / (count - 1), 0.0)
    standard_error = math.sqrt(variance / count)
    z_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    lower = mean - z_value * standard_error
    upper = mean + z_value * standard_error
    if lower_bound is not None:
        lower = max(lower, lower_bound)
        upper = max(upper, lower_bound)
    if upper_bound is not None:
        lower = min(lower, upper_bound)
        upper = min(upper, upper_bound)
    return float(mean), float(lower), float(upper), float(standard_error)


def _estimates_and_intervals(
    positive_count: np.ndarray,
    sample_count: int,
    config: AuditConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    raw_estimate = positive_count.astype(np.float64) / float(sample_count)
    raw_lower, raw_upper = wilson_interval(
        positive_count, sample_count, config.interval_level
    )
    output = {"raw_rate_wilson": (raw_estimate, raw_lower, raw_upper)}
    prior_parameters = {
        "beta_uniform": (1.0, 1.0),
        "beta_oracle": (config.true_alpha, config.true_beta),
        "beta_mismatched": (config.mismatched_alpha, config.mismatched_beta),
    }
    for estimator, (alpha, beta) in prior_parameters.items():
        output[estimator] = beta_posterior_mean_interval(
            positive_count,
            sample_count,
            alpha,
            beta,
            config.interval_level,
        )
    return output


def run_audit(config: AuditConfig = AuditConfig()) -> dict[str, Any]:
    """Run the fixed Monte Carlo protocol and return serializable results."""
    config.validate()
    metric_names = (
        "brier",
        "nll",
        "latent_probability_mse",
        "latent_probability_bias",
        "interval_coverage",
        "interval_width",
    )
    pooled: dict[tuple[int, str], dict[str, dict[str, float | int]]] = {}
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        for sample_count in config.sample_sizes:
            rng = np.random.default_rng(
                np.random.SeedSequence([20260903, int(seed), int(sample_count)])
            )
            latent_probability = rng.beta(
                config.true_alpha,
                config.true_beta,
                size=config.batches_per_size_per_seed,
            )
            positive_count = rng.binomial(sample_count, latent_probability)
            estimates = _estimates_and_intervals(
                positive_count, sample_count, config
            )

            for estimator in ESTIMATOR_NAMES:
                estimate, lower, upper = estimates[estimator]
                brier, nll = expected_binary_scores(
                    latent_probability, estimate, config.nll_clip
                )
                samples = {
                    "brier": brier,
                    "nll": nll,
                    "latent_probability_mse": np.square(
                        estimate - latent_probability
                    ),
                    "latent_probability_bias": estimate - latent_probability,
                    "interval_coverage": (
                        (lower <= latent_probability) & (latent_probability <= upper)
                    ).astype(np.float64),
                    "interval_width": upper - lower,
                }
                row: dict[str, Any] = {
                    "seed": int(seed),
                    "sample_count": int(sample_count),
                    "estimator": estimator,
                    "batch_count": int(config.batches_per_size_per_seed),
                }
                key = (sample_count, estimator)
                pooled.setdefault(
                    key, {name: _new_moments() for name in metric_names}
                )
                for metric, values in samples.items():
                    row[metric] = float(np.mean(values))
                    _update_moments(pooled[key][metric], values)
                per_seed.append(row)

    summary: list[dict[str, Any]] = []
    for sample_count in config.sample_sizes:
        for estimator in ESTIMATOR_NAMES:
            key = (sample_count, estimator)
            row = {
                "sample_count": int(sample_count),
                "estimator": estimator,
                "batch_count": int(
                    config.batches_per_size_per_seed * len(config.seeds)
                ),
            }
            for metric in metric_names:
                bounds = (
                    (0.0, 1.0)
                    if metric in {"brier", "latent_probability_mse", "interval_coverage", "interval_width"}
                    else ((0.0, None) if metric == "nll" else (None, None))
                )
                mean, low, high, standard_error = _moment_summary(
                    pooled[key][metric],
                    config.mc_confidence_level,
                    lower_bound=bounds[0],
                    upper_bound=bounds[1],
                )
                row[f"{metric}_mean"] = mean
                row[f"{metric}_mc_ci_low"] = low
                row[f"{metric}_mc_ci_high"] = high
                row[f"{metric}_mc_standard_error"] = standard_error
            summary.append(row)

    return {
        "evidence_label": "SEMI-SYNTHETIC / MONTE CARLO EVIDENCE ONLY",
        "novelty_boundary": (
            "The audit evaluates prior sensitivity; it does not present the mature "
            "Beta--Binomial conjugate model as a methodological innovation."
        ),
        "protocol": {
            **asdict(config),
            "true_prior_mean": config.true_alpha
            / (config.true_alpha + config.true_beta),
            "mismatched_prior_mean": config.mismatched_alpha
            / (config.mismatched_alpha + config.mismatched_beta),
            "score_target": (
                "expected proper score for one fresh Bernoulli outcome conditional "
                "on each simulated latent batch probability"
            ),
            "interval_target": "latent batch anomaly probability",
            "raw_interval": "two-sided Wilson score interval",
            "beta_interval": "equal-tailed posterior credible interval",
            "mc_interval": (
                "normal Monte Carlo confidence interval from independent simulated "
                "batch-level contributions pooled across seeds"
            ),
        },
        "estimators": {
            "raw_rate_wilson": {
                "point": "k/n",
                "interval": "Wilson score interval",
                "role": "unsmoothed baseline",
            },
            "beta_uniform": {
                "prior": "Beta(1, 1)",
                "role": "weak uniform prior used by the current prototype",
            },
            "beta_oracle": {
                "prior": f"Beta({config.true_alpha:g}, {config.true_beta:g})",
                "role": "generator-matched upper-bound comparator; not deployable",
            },
            "beta_mismatched": {
                "prior": (
                    f"Beta({config.mismatched_alpha:g}, "
                    f"{config.mismatched_beta:g})"
                ),
                "role": "deliberately mean-inverted prior-misspecification control",
            },
        },
        "per_seed_metrics": per_seed,
        "summary": summary,
    }


def _format_interval(row: dict[str, Any], metric: str, digits: int = 5) -> str:
    return (
        f"{row[f'{metric}_mean']:.{digits}f} "
        f"[{row[f'{metric}_mc_ci_low']:.{digits}f}, "
        f"{row[f'{metric}_mc_ci_high']:.{digits}f}]"
    )


def build_report(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    lookup = {
        (row["sample_count"], row["estimator"]): row
        for row in payload["summary"]
    }
    lines = [
        "# Beta--Binomial batch posterior calibration audit",
        "",
        "**SEMI-SYNTHETIC / MONTE CARLO EVIDENCE ONLY**",
        "",
        "This audit tests when batch-level shrinkage improves probability estimation",
        "and when a wrong prior damages it. It does not use enterprise meter data, and",
        "it does not claim that the mature Beta--Binomial conjugate model is novel.",
        "",
        "## Protocol",
        "",
        f"- Latent batch risks: `pi_b ~ Beta({protocol['true_alpha']:g}, {protocol['true_beta']:g})` "
        f"(mean {protocol['true_prior_mean']:.3f}).",
        "- Inspection counts: `k_b | pi_b ~ Binomial(n_b, pi_b)`.",
        f"- Sample sizes: {', '.join(str(value) for value in protocol['sample_sizes'])}.",
        f"- Seeds: {', '.join(str(value) for value in protocol['seeds'])}; "
        f"{protocol['batches_per_size_per_seed']:,} independent batches per size and seed.",
        f"- Point scores are expected one-step-ahead Bernoulli Brier/NLL values.",
        f"- Coverage targets the latent `pi_b`; raw rates use a Wilson "
        f"{100 * protocol['interval_level']:.0f}% interval and Beta estimators use "
        "equal-tailed posterior credible intervals.",
        f"- Brackets report {100 * protocol['mc_confidence_level']:.0f}% Monte Carlo "
        "confidence intervals from independent batch-level contributions.",
        "",
        "## Estimators",
        "",
        "- `raw_rate_wilson`: unsmoothed `k/n` point estimate with a Wilson interval.",
        "- `beta_uniform`: `Beta(1,1)` prior, matching the current lightweight prototype.",
        f"- `beta_oracle`: generator-matched `Beta({protocol['true_alpha']:g}, "
        f"{protocol['true_beta']:g})`; an unavailable upper-bound comparator.",
        f"- `beta_mismatched`: deliberately wrong `Beta({protocol['mismatched_alpha']:g}, "
        f"{protocol['mismatched_beta']:g})`, whose mean "
        f"{protocol['mismatched_prior_mean']:.3f} reverses the generator mean.",
        "",
        "## Results",
        "",
        "| n | Estimator | Brier [95% MC CI] | NLL [95% MC CI] | 90% coverage [95% MC CI] | Width [95% MC CI] |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for sample_count in protocol["sample_sizes"]:
        for estimator in ESTIMATOR_NAMES:
            row = lookup[(sample_count, estimator)]
            lines.append(
                f"| {sample_count} | {estimator} | "
                f"{_format_interval(row, 'brier')} | "
                f"{_format_interval(row, 'nll')} | "
                f"{_format_interval(row, 'interval_coverage', 4)} | "
                f"{_format_interval(row, 'interval_width', 4)} |"
            )

    smallest = protocol["sample_sizes"][0]
    largest = protocol["sample_sizes"][-1]
    raw_small = lookup[(smallest, "raw_rate_wilson")]
    uniform_small = lookup[(smallest, "beta_uniform")]
    oracle_small = lookup[(smallest, "beta_oracle")]
    mismatch_small = lookup[(smallest, "beta_mismatched")]
    raw_large = lookup[(largest, "raw_rate_wilson")]
    uniform_large = lookup[(largest, "beta_uniform")]
    oracle_large = lookup[(largest, "beta_oracle")]
    mismatch_large = lookup[(largest, "beta_mismatched")]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- At `n={smallest}`, the matched prior changes Brier by "
        f"{oracle_small['brier_mean'] - raw_small['brier_mean']:+.5f} relative to raw "
        f"`k/n`; the uniform prior changes it by "
        f"{uniform_small['brier_mean'] - raw_small['brier_mean']:+.5f}.",
        f"- At `n={smallest}`, the deliberately mismatched prior changes Brier by "
        f"{mismatch_small['brier_mean'] - uniform_small['brier_mean']:+.5f} relative "
        "to the uniform prior. This is the required negative control: shrinkage is",
        "  beneficial only when its prior is sufficiently compatible with the target batches.",
        f"- At `n={largest}`, the Brier range across the three Beta estimators is "
        f"{max(uniform_large['brier_mean'], oracle_large['brier_mean'], mismatch_large['brier_mean']) - min(uniform_large['brier_mean'], oracle_large['brier_mean'], mismatch_large['brier_mean']):.5f}; "
        "inspection evidence increasingly dominates prior choice.",
        f"- Raw `k/n` NLL is especially poor for tiny samples because `k=0` and `k=n` "
        f"produce boundary probabilities; clipping is used only to keep the log score finite.",
        f"- The matched-prior interval contracts from width "
        f"{oracle_small['interval_width_mean']:.4f} at `n={smallest}` to "
        f"{oracle_large['interval_width_mean']:.4f} at `n={largest}` while its coverage "
        "can be checked directly against the nominal 0.90 target.",
        "",
        "These results validate a limited statistical statement about shrinkage and prior",
        "sensitivity under the declared generator. They do not validate cross-batch meter",
        "prediction, mapping-confidence gating, enterprise generalization, or causal tracing.",
        "",
    ])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "summary.csv", payload["summary"])
    _write_csv(output_dir / "per_seed_metrics.csv", payload["per_seed_metrics"])
    (output_dir / "report.md").write_text(build_report(payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/ra_mgq/results/beta_binomial_calibration_audit"),
    )
    parser.add_argument("--batches-per-size-per-seed", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AuditConfig(
        batches_per_size_per_seed=args.batches_per_size_per_seed
    )
    payload = run_audit(config)
    write_outputs(payload, args.output_dir)
    print(payload["evidence_label"])
    print(
        f"Wrote {len(payload['summary'])} summary rows and "
        f"{len(payload['per_seed_metrics'])} seed-level rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
