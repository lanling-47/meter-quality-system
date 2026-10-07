# CP-UDM configuration-based paired confirmatory ablation

## Protocol

- Group split seed: 42 (one split reused by every configuration and training seed).
- Training seeds: 17.
- Maximum epochs / early-stopping patience: 2 / 15.
- All configurations jointly train capacitor, IGBT, and MOSFET data. For a given training seed, model initialization and data-loader randomness are reset to that same seed for every configuration.

## Mean +/- sample SD across training seeds

### overall

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.2483 +/- NA | 0.0750 +/- NA | 0.0000 +/- NA | 0.6235 +/- NA |
| no_physics | 0.2479 +/- NA | 0.1229 +/- NA | 0.0000 +/- NA | -0.0910 +/- NA |
| no_forecast | 0.2484 +/- NA | 0.0683 +/- NA | 0.0000 +/- NA | 0.6192 +/- NA |
| no_ranking_endpoint | 0.2485 +/- NA | 0.0150 +/- NA | 0.0000 +/- NA | 0.6223 +/- NA |
| no_alignment | 0.2480 +/- NA | 0.0276 +/- NA | 0.0000 +/- NA | 0.6311 +/- NA |

### capacitor

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.2249 +/- NA | 0.8406 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_physics | 0.2247 +/- NA | 0.8608 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_forecast | 0.2248 +/- NA | 0.8428 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_ranking_endpoint | 0.2251 +/- NA | 0.8303 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_alignment | 0.2244 +/- NA | 0.8513 +/- NA | 0.0000 +/- NA | NA +/- NA |

### igbt

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.2268 +/- NA | 0.7391 +/- NA | 0.0000 +/- NA | 0.8613 +/- NA |
| no_physics | 0.2271 +/- NA | 0.6970 +/- NA | 0.0000 +/- NA | -0.2852 +/- NA |
| no_forecast | 0.2267 +/- NA | 0.7391 +/- NA | 0.0000 +/- NA | 0.8710 +/- NA |
| no_ranking_endpoint | 0.2268 +/- NA | 0.7391 +/- NA | 0.0000 +/- NA | 0.8779 +/- NA |
| no_alignment | 0.2261 +/- NA | 0.9142 +/- NA | 0.0000 +/- NA | 0.8516 +/- NA |

### mosfet

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.2667 +/- NA | -0.2236 +/- NA | 0.0000 +/- NA | 0.3858 +/- NA |
| no_physics | 0.2660 +/- NA | -0.1628 +/- NA | 0.0000 +/- NA | 0.1033 +/- NA |
| no_forecast | 0.2669 +/- NA | -0.2309 +/- NA | 0.0000 +/- NA | 0.3673 +/- NA |
| no_ranking_endpoint | 0.2669 +/- NA | -0.2249 +/- NA | 0.0000 +/- NA | 0.3667 +/- NA |
| no_alignment | 0.2665 +/- NA | -0.1888 +/- NA | 0.0000 +/- NA | 0.4105 +/- NA |

## Paired deltas relative to full

Every delta below is `ablation - full` for the same training seed. Therefore a positive MAE delta means that removing the component degraded accuracy; the other metrics retain their natural signed interpretation.

| Configuration | Scope | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |
|---|---|---:|---:|---:|---:|
| no_physics | overall | -0.0004 +/- NA | 0.0479 +/- NA | 0.0000 +/- NA | -0.7145 +/- NA |
| no_physics | capacitor | -0.0002 +/- NA | 0.0202 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_physics | igbt | 0.0003 +/- NA | -0.0420 +/- NA | 0.0000 +/- NA | -1.1465 +/- NA |
| no_physics | mosfet | -0.0007 +/- NA | 0.0608 +/- NA | 0.0000 +/- NA | -0.2825 +/- NA |
| no_forecast | overall | 0.0001 +/- NA | -0.0067 +/- NA | 0.0000 +/- NA | -0.0044 +/- NA |
| no_forecast | capacitor | -0.0001 +/- NA | 0.0022 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_forecast | igbt | -0.0000 +/- NA | 0.0000 +/- NA | 0.0000 +/- NA | 0.0098 +/- NA |
| no_forecast | mosfet | 0.0002 +/- NA | -0.0073 +/- NA | 0.0000 +/- NA | -0.0185 +/- NA |
| no_ranking_endpoint | overall | 0.0002 +/- NA | -0.0600 +/- NA | 0.0000 +/- NA | -0.0012 +/- NA |
| no_ranking_endpoint | capacitor | 0.0002 +/- NA | -0.0104 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_ranking_endpoint | igbt | 0.0001 +/- NA | 0.0000 +/- NA | 0.0000 +/- NA | 0.0166 +/- NA |
| no_ranking_endpoint | mosfet | 0.0002 +/- NA | -0.0013 +/- NA | 0.0000 +/- NA | -0.0191 +/- NA |
| no_alignment | overall | -0.0003 +/- NA | -0.0474 +/- NA | 0.0000 +/- NA | 0.0075 +/- NA |
| no_alignment | capacitor | -0.0005 +/- NA | 0.0106 +/- NA | 0.0000 +/- NA | NA +/- NA |
| no_alignment | igbt | -0.0006 +/- NA | 0.1751 +/- NA | 0.0000 +/- NA | -0.0097 +/- NA |
| no_alignment | mosfet | -0.0002 +/- NA | 0.0348 +/- NA | 0.0000 +/- NA | 0.0247 +/- NA |

## Per-seed paired deltas

| Configuration | Scope | Seed | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |
|---|---|---:|---:|---:|---:|---:|
| no_physics | overall | 17 | -0.0004 | 0.0479 | 0.0000 | -0.7145 |
| no_physics | capacitor | 17 | -0.0002 | 0.0202 | 0.0000 | NA |
| no_physics | igbt | 17 | 0.0003 | -0.0420 | 0.0000 | -1.1465 |
| no_physics | mosfet | 17 | -0.0007 | 0.0608 | 0.0000 | -0.2825 |
| no_forecast | overall | 17 | 0.0001 | -0.0067 | 0.0000 | -0.0044 |
| no_forecast | capacitor | 17 | -0.0001 | 0.0022 | 0.0000 | NA |
| no_forecast | igbt | 17 | -0.0000 | 0.0000 | 0.0000 | 0.0098 |
| no_forecast | mosfet | 17 | 0.0002 | -0.0073 | 0.0000 | -0.0185 |
| no_ranking_endpoint | overall | 17 | 0.0002 | -0.0600 | 0.0000 | -0.0012 |
| no_ranking_endpoint | capacitor | 17 | 0.0002 | -0.0104 | 0.0000 | NA |
| no_ranking_endpoint | igbt | 17 | 0.0001 | 0.0000 | 0.0000 | 0.0166 |
| no_ranking_endpoint | mosfet | 17 | 0.0002 | -0.0013 | 0.0000 | -0.0191 |
| no_alignment | overall | 17 | -0.0003 | -0.0474 | 0.0000 | 0.0075 |
| no_alignment | capacitor | 17 | -0.0005 | 0.0106 | 0.0000 | NA |
| no_alignment | igbt | 17 | -0.0006 | 0.1751 | 0.0000 | -0.0097 |
| no_alignment | mosfet | 17 | -0.0002 | 0.0348 | 0.0000 | 0.0247 |

## Metric interpretation boundary

`physics_rate_spearman` is a cross-sample pooled diagnostic: within each device it pools valid time points from all test prefixes before computing Spearman correlation, while the overall value is the finite mean of the three device-level correlations. It is not the same estimand as the training physics loss, which compares the learned and prior rate shapes after centering their log rates separately within each sequence. Consequently this diagnostic must not be interpreted as a direct test-loss surrogate.
