# Cross-prefix endpoint consistency audit

## Scope and estimand

This audit compares the saved final `prediction` for adjacent truncation prefixes of the same original NASA run. The original run ID and truncation endpoint are recovered only from the anchored loader-defined `<group>::prefix-<integer>` sample ID, then cross-checked against the local processed dataset cache. No model is trained or re-evaluated.

The adjacent endpoint decrease rate is `count(p[i+1] < p[i]) / count(adjacent prefix pairs)`. The maximum decrease is `max(p[i] - p[i+1], 0)` over the same pairs. The configured comparison tolerance is `0.0`.

This estimand is not the within-prefix trajectory monotonic violation rate (MVR). MVR checks consecutive latent trajectory states produced during one forward pass. This audit instead compares separate forward passes made from different truncations of one raw run.

## CP-UDM strict result by seed

| Seed | Group-seeds | Adjacent pairs | Decreases | Decrease rate | Max decrease |
|---:|---:|---:|---:|---:|---:|
| 17 | 18 | 90 | 25 | 27.78% | 0.069919 |
| 42 | 18 | 90 | 27 | 30.00% | 0.077399 |
| 73 | 18 | 90 | 26 | 28.89% | 0.122460 |

## Main-match ablation result pooled across seeds

| Configuration | Group-seeds | Adjacent pairs | Decreases | Decrease rate | Max decrease |
|:--|---:|---:|---:|---:|---:|
| full | 54 | 270 | 78 | 28.89% | 0.122460 |
| no_alignment | 54 | 270 | 81 | 30.00% | 0.128892 |
| no_forecast | 54 | 270 | 77 | 28.52% | 0.105542 |
| no_physics | 54 | 270 | 78 | 28.89% | 0.131657 |
| no_ranking_endpoint | 54 | 270 | 72 | 26.67% | 0.107451 |

## Device-level pooled diagnostics

| Experiment | Variant | Device | Adjacent pairs | Decreases | Decrease rate | Max decrease |
|:--|:--|:--|---:|---:|---:|---:|
| cp_udm_ablation_mainmatch | full | capacitor | 90 | 3 | 3.33% | 0.019087 |
| cp_udm_ablation_mainmatch | full | igbt | 30 | 0 | 0.00% | 0.000000 |
| cp_udm_ablation_mainmatch | full | mosfet | 150 | 75 | 50.00% | 0.122460 |
| cp_udm_ablation_mainmatch | no_alignment | capacitor | 90 | 3 | 3.33% | 0.016064 |
| cp_udm_ablation_mainmatch | no_alignment | igbt | 30 | 0 | 0.00% | 0.000000 |
| cp_udm_ablation_mainmatch | no_alignment | mosfet | 150 | 78 | 52.00% | 0.128892 |
| cp_udm_ablation_mainmatch | no_forecast | capacitor | 90 | 3 | 3.33% | 0.019183 |
| cp_udm_ablation_mainmatch | no_forecast | igbt | 30 | 0 | 0.00% | 0.000000 |
| cp_udm_ablation_mainmatch | no_forecast | mosfet | 150 | 74 | 49.33% | 0.105542 |
| cp_udm_ablation_mainmatch | no_physics | capacitor | 90 | 3 | 3.33% | 0.012731 |
| cp_udm_ablation_mainmatch | no_physics | igbt | 30 | 0 | 0.00% | 0.000000 |
| cp_udm_ablation_mainmatch | no_physics | mosfet | 150 | 75 | 50.00% | 0.131657 |
| cp_udm_ablation_mainmatch | no_ranking_endpoint | capacitor | 90 | 3 | 3.33% | 0.038438 |
| cp_udm_ablation_mainmatch | no_ranking_endpoint | igbt | 30 | 0 | 0.00% | 0.000000 |
| cp_udm_ablation_mainmatch | no_ranking_endpoint | mosfet | 150 | 69 | 46.00% | 0.107451 |
| cp_udm_strict | cp_udm_net | capacitor | 90 | 3 | 3.33% | 0.019087 |
| cp_udm_strict | cp_udm_net | igbt | 30 | 0 | 0.00% | 0.000000 |
| cp_udm_strict | cp_udm_net | mosfet | 150 | 75 | 50.00% | 0.122460 |

## Interpretation boundary

The positive-rate integration guarantees nondecreasing latent states only inside one fixed observed prefix. It does not guarantee monotonic final predictions across independently truncated prefixes; causal attention also only blocks future-token interactions within one fixed prefix. The saved final prediction is an equal-weight blend of a direct stage head and the dynamic endpoint. In addition, response scale uses the median of the entire observed prefix, time features and local increments are normalized by that prefix's endpoint duration, and every prefix is independently resampled. Extending a prefix therefore changes the representation of earlier observations. Any cross-prefix decrease reported here is compatible with zero within-prefix MVR and must not be described as a violation of the structural integration guarantee.

The detailed adjacent comparisons are in `prefix_endpoint_pairs.csv`; all aggregation scopes are in `prefix_endpoint_summary.csv`, and machine-readable validation details are in `prefix_endpoint_audit.json`.
