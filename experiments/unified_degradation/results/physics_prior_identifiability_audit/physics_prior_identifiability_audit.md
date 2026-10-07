# Physical-prior identifiability audit

## Protocol

This is a no-retraining audit. The stress-only physical rate is computed from every cached prefix by the existing CP-UDM implementation. Existing checkpoints are then loaded in evaluation mode to recover their per-time-step learned `damage_rate`; recomputed endpoint predictions must match the saved CSV values before a learned-vs-prior shape MSE is accepted.

For each sequence, the training loss clamps positive rates at `1e-6`, takes logs, subtracts the valid-point mean separately from learned and physical rates, and averages their squared residual. The prior is called nondegenerate only when its centered-log variance exceeds `1e-12`.

## Input-prior identifiability: all cached prefixes

| Device | Prefixes | Nondegenerate | Rate | Mean variance | Max variance |
|:--|---:|---:|---:|---:|---:|
| capacitor | 138 | 0 | 0.00% | 0 | 0 |
| igbt | 42 | 42 | 100.00% | 6.68897 | 10.0701 |
| mosfet | 246 | 246 | 100.00% | 2.70153 | 17.4047 |

## Input-prior identifiability: shared strict/main-match test subset

| Device | Prefixes | Nondegenerate | Rate | Mean variance | Max variance |
|:--|---:|---:|---:|---:|---:|
| capacitor | 36 | 0 | 0.00% | 0 | 0 |
| igbt | 12 | 12 | 100.00% | 6.31395 | 10.0701 |
| mosfet | 60 | 60 | 100.00% | 1.56223 | 7.87531 |

The strict and main-match artifacts contain the same 108 test prefixes, so their input-prior identifiability values are identical. Configuration and seed do not alter this deterministic input prior.

## Learned-vs-prior centered-log shape MSE

| Experiment | Variant | Physics weight | Prefixes | Point-weighted MSE | Max prefix MSE | Max prediction replay error |
|:--|:--|---:|---:|---:|---:|---:|
| cp_udm_ablation_mainmatch | full | 0.01 | 324 | 1.328032 | 8.609554 | 1.19e-07 |
| cp_udm_ablation_mainmatch | no_alignment | 0.01 | 324 | 1.342179 | 8.748966 | 1.19e-07 |
| cp_udm_ablation_mainmatch | no_forecast | 0.01 | 324 | 1.338002 | 8.587170 | 1.79e-07 |
| cp_udm_ablation_mainmatch | no_physics | 0 | 324 | 1.770104 | 8.847033 | 1.19e-07 |
| cp_udm_ablation_mainmatch | no_ranking_endpoint | 0.01 | 324 | 1.295976 | 8.086506 | 1.79e-07 |
| cp_udm_strict | cp_udm_net | 0.01 | 324 | 1.328032 | 8.609554 | 1.19e-07 |

Every seed-device recomputation is also compared directly with the `physics` component returned by `degradation_model_loss`; the maximum discrepancy is `4.77e-07`.

## Capacitor limitation

Within each capacitor prefix, the current prior depends only on the first stress channel, which is a sequence-constant applied-voltage ratio. The raw rate amplitude changes between voltage levels, but division by each prefix's mean and subsequent centered-log comparison remove every positive sequence scale. Its centered-log target is therefore exactly constant (zero variance) for every capacitor prefix. On this dataset the capacitor physics term is equivalent to a constant-rate-shape regularizer on the learned rate; it cannot encode the relative damage-rate magnitude of 10 V, 12 V, and 14 V conditions.

This limitation is about identifiability of the implemented training target, not evidence that voltage has no physical effect. Endpoint CSV fields alone would not identify learned rate shape; the MSE above is available only because the saved checkpoints and cached sequences allowed deterministic forward replay. No target labels or held-out predictions were used to invent a time-resolved rate.

Per-prefix prior metrics are in `physics_prior_prefix_metrics.csv`; checkpoint shape comparisons are in `physics_shape_prefix_mse.csv`; both summary tables and full provenance are available in the companion CSV and JSON files.
