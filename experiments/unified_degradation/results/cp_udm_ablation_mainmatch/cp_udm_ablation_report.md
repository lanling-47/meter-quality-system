# CP-UDM configuration-based paired confirmatory ablation

## Protocol

- Group split seed: 42 (one split reused by every configuration and training seed).
- Training seeds: 17, 42, 73.
- Maximum epochs / early-stopping patience: 40 / 15.
- All configurations jointly train capacitor, IGBT, and MOSFET data. For a given training seed, model initialization and data-loader randomness are reset to that same seed for every configuration.

## Mean +/- sample SD across training seeds

### overall

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.1740 +/- 0.0013 | 0.6159 +/- 0.0225 | 0.0000 +/- 0.0000 | 0.2925 +/- 0.0460 |
| no_physics | 0.1790 +/- 0.0012 | 0.5966 +/- 0.0140 | 0.0000 +/- 0.0000 | 0.2217 +/- 0.1003 |
| no_forecast | 0.1798 +/- 0.0028 | 0.5876 +/- 0.0150 | 0.0000 +/- 0.0000 | 0.3309 +/- 0.0547 |
| no_ranking_endpoint | 0.1764 +/- 0.0037 | 0.6012 +/- 0.0105 | 0.0000 +/- 0.0000 | 0.3506 +/- 0.0227 |
| no_alignment | 0.1765 +/- 0.0037 | 0.6020 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.3342 +/- 0.0342 |

### capacitor

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.0604 +/- 0.0039 | 0.9634 +/- 0.0182 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_physics | 0.0649 +/- 0.0070 | 0.9496 +/- 0.0308 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_forecast | 0.0715 +/- 0.0030 | 0.9566 +/- 0.0113 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_ranking_endpoint | 0.0634 +/- 0.0075 | 0.9627 +/- 0.0120 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_alignment | 0.0624 +/- 0.0046 | 0.9536 +/- 0.0181 | 0.0000 +/- 0.0000 | NA +/- NA |

### igbt

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.1242 +/- 0.0045 | 0.9831 +/- 0.0081 | 0.0000 +/- 0.0000 | 0.7588 +/- 0.0054 |
| no_physics | 0.1346 +/- 0.0250 | 0.9702 +/- 0.0195 | 0.0000 +/- 0.0000 | 0.7221 +/- 0.0373 |
| no_forecast | 0.1367 +/- 0.0154 | 0.9807 +/- 0.0070 | 0.0000 +/- 0.0000 | 0.7671 +/- 0.0050 |
| no_ranking_endpoint | 0.1269 +/- 0.0117 | 0.9714 +/- 0.0040 | 0.0000 +/- 0.0000 | 0.7666 +/- 0.0202 |
| no_alignment | 0.1247 +/- 0.0254 | 0.9784 +/- 0.0040 | 0.0000 +/- 0.0000 | 0.7773 +/- 0.0049 |

### mosfet

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.2522 +/- 0.0041 | 0.3466 +/- 0.0738 | 0.0000 +/- 0.0000 | -0.1738 +/- 0.0883 |
| no_physics | 0.2563 +/- 0.0022 | 0.2895 +/- 0.0033 | 0.0000 +/- 0.0000 | -0.2788 +/- 0.1701 |
| no_forecast | 0.2535 +/- 0.0011 | 0.2824 +/- 0.0205 | 0.0000 +/- 0.0000 | -0.1054 +/- 0.1137 |
| no_ranking_endpoint | 0.2540 +/- 0.0028 | 0.2842 +/- 0.0319 | 0.0000 +/- 0.0000 | -0.0653 +/- 0.0403 |
| no_alignment | 0.2554 +/- 0.0036 | 0.2884 +/- 0.0169 | 0.0000 +/- 0.0000 | -0.1088 +/- 0.0712 |

## Paired deltas relative to full

Every delta below is `ablation - full` for the same training seed. Therefore a positive MAE delta means that removing the component degraded accuracy; the other metrics retain their natural signed interpretation.

| Configuration | Scope | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |
|---|---|---:|---:|---:|---:|
| no_physics | overall | 0.0050 +/- 0.0025 | -0.0192 +/- 0.0324 | 0.0000 +/- 0.0000 | -0.0708 +/- 0.1220 |
| no_physics | capacitor | 0.0046 +/- 0.0056 | -0.0138 +/- 0.0128 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_physics | igbt | 0.0103 +/- 0.0204 | -0.0128 +/- 0.0222 | 0.0000 +/- 0.0000 | -0.0367 +/- 0.0380 |
| no_physics | mosfet | 0.0041 +/- 0.0063 | -0.0571 +/- 0.0746 | 0.0000 +/- 0.0000 | -0.1050 +/- 0.2084 |
| no_forecast | overall | 0.0058 +/- 0.0026 | -0.0283 +/- 0.0366 | 0.0000 +/- 0.0000 | 0.0384 +/- 0.0705 |
| no_forecast | capacitor | 0.0111 +/- 0.0033 | -0.0069 +/- 0.0078 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_forecast | igbt | 0.0125 +/- 0.0198 | -0.0023 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0083 +/- 0.0084 |
| no_forecast | mosfet | 0.0012 +/- 0.0051 | -0.0642 +/- 0.0770 | 0.0000 +/- 0.0000 | 0.0685 +/- 0.1467 |
| no_ranking_endpoint | overall | 0.0023 +/- 0.0031 | -0.0146 +/- 0.0245 | 0.0000 +/- 0.0000 | 0.0581 +/- 0.0628 |
| no_ranking_endpoint | capacitor | 0.0030 +/- 0.0106 | -0.0007 +/- 0.0094 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_ranking_endpoint | igbt | 0.0027 +/- 0.0114 | -0.0117 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0078 +/- 0.0228 |
| no_ranking_endpoint | mosfet | 0.0018 +/- 0.0057 | -0.0623 +/- 0.0820 | 0.0000 +/- 0.0000 | 0.1085 +/- 0.1256 |
| no_alignment | overall | 0.0025 +/- 0.0035 | -0.0139 +/- 0.0284 | 0.0000 +/- 0.0000 | 0.0418 +/- 0.0770 |
| no_alignment | capacitor | 0.0020 +/- 0.0072 | -0.0098 +/- 0.0080 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_alignment | igbt | 0.0004 +/- 0.0299 | -0.0047 +/- 0.0040 | 0.0000 +/- 0.0000 | 0.0185 +/- 0.0059 |
| no_alignment | mosfet | 0.0032 +/- 0.0077 | -0.0582 +/- 0.0594 | 0.0000 +/- 0.0000 | 0.0650 +/- 0.1535 |

## Per-seed paired deltas

| Configuration | Scope | Seed | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |
|---|---|---:|---:|---:|---:|---:|
| no_physics | overall | 17 | 0.0078 | -0.0494 | 0.0000 | -0.0237 |
| no_physics | overall | 42 | 0.0031 | -0.0234 | 0.0000 | 0.0206 |
| no_physics | overall | 73 | 0.0039 | 0.0150 | 0.0000 | -0.2093 |
| no_physics | capacitor | 17 | -0.0002 | -0.0057 | 0.0000 | NA |
| no_physics | capacitor | 42 | 0.0108 | -0.0286 | 0.0000 | NA |
| no_physics | capacitor | 73 | 0.0032 | -0.0071 | 0.0000 | NA |
| no_physics | igbt | 17 | 0.0148 | 0.0000 | 0.0000 | -0.0094 |
| no_physics | igbt | 42 | -0.0120 | -0.0385 | 0.0000 | -0.0206 |
| no_physics | igbt | 73 | 0.0281 | 0.0000 | 0.0000 | -0.0801 |
| no_physics | mosfet | 17 | 0.0112 | -0.1348 | 0.0000 | -0.0380 |
| no_physics | mosfet | 42 | 0.0016 | -0.0505 | 0.0000 | 0.0618 |
| no_physics | mosfet | 73 | -0.0005 | 0.0140 | 0.0000 | -0.3386 |
| no_forecast | overall | 17 | 0.0067 | -0.0666 | 0.0000 | 0.0669 |
| no_forecast | overall | 42 | 0.0079 | -0.0246 | 0.0000 | 0.0901 |
| no_forecast | overall | 73 | 0.0028 | 0.0064 | 0.0000 | -0.0419 |
| no_forecast | capacitor | 17 | 0.0073 | -0.0128 | 0.0000 | NA |
| no_forecast | capacitor | 42 | 0.0129 | 0.0019 | 0.0000 | NA |
| no_forecast | capacitor | 73 | 0.0131 | -0.0097 | 0.0000 | NA |
| no_forecast | igbt | 17 | 0.0028 | 0.0070 | 0.0000 | 0.0129 |
| no_forecast | igbt | 42 | 0.0353 | 0.0000 | 0.0000 | -0.0015 |
| no_forecast | igbt | 73 | -0.0006 | -0.0140 | 0.0000 | 0.0134 |
| no_forecast | mosfet | 17 | 0.0070 | -0.1523 | 0.0000 | 0.1210 |
| no_forecast | mosfet | 42 | -0.0007 | -0.0304 | 0.0000 | 0.1817 |
| no_forecast | mosfet | 73 | -0.0026 | -0.0098 | 0.0000 | -0.0972 |
| no_ranking_endpoint | overall | 17 | 0.0005 | -0.0425 | 0.0000 | 0.1306 |
| no_ranking_endpoint | overall | 42 | 0.0006 | 0.0037 | 0.0000 | 0.0246 |
| no_ranking_endpoint | overall | 73 | 0.0059 | -0.0051 | 0.0000 | 0.0193 |
| no_ranking_endpoint | capacitor | 17 | -0.0084 | 0.0025 | 0.0000 | NA |
| no_ranking_endpoint | capacitor | 42 | 0.0049 | 0.0067 | 0.0000 | NA |
| no_ranking_endpoint | capacitor | 73 | 0.0126 | -0.0113 | 0.0000 | NA |
| no_ranking_endpoint | igbt | 17 | -0.0103 | 0.0000 | 0.0000 | 0.0112 |
| no_ranking_endpoint | igbt | 42 | 0.0072 | -0.0210 | 0.0000 | -0.0166 |
| no_ranking_endpoint | igbt | 73 | 0.0111 | -0.0140 | 0.0000 | 0.0287 |
| no_ranking_endpoint | mosfet | 17 | 0.0080 | -0.1570 | 0.0000 | 0.2499 |
| no_ranking_endpoint | mosfet | 42 | -0.0033 | -0.0153 | 0.0000 | 0.0657 |
| no_ranking_endpoint | mosfet | 73 | 0.0008 | -0.0146 | 0.0000 | 0.0100 |
| no_alignment | overall | 17 | 0.0034 | -0.0407 | 0.0000 | 0.1304 |
| no_alignment | overall | 42 | 0.0054 | -0.0169 | 0.0000 | 0.0040 |
| no_alignment | overall | 73 | -0.0013 | 0.0159 | 0.0000 | -0.0091 |
| no_alignment | capacitor | 17 | -0.0060 | -0.0021 | 0.0000 | NA |
| no_alignment | capacitor | 42 | 0.0043 | -0.0093 | 0.0000 | NA |
| no_alignment | capacitor | 73 | 0.0078 | -0.0180 | 0.0000 | NA |
| no_alignment | igbt | 17 | -0.0109 | 0.0000 | 0.0000 | 0.0198 |
| no_alignment | igbt | 42 | 0.0344 | -0.0070 | 0.0000 | 0.0121 |
| no_alignment | igbt | 73 | -0.0222 | -0.0070 | 0.0000 | 0.0237 |
| no_alignment | mosfet | 17 | 0.0119 | -0.1172 | 0.0000 | 0.2409 |
| no_alignment | mosfet | 42 | 0.0003 | -0.0590 | 0.0000 | -0.0041 |
| no_alignment | mosfet | 73 | -0.0026 | 0.0016 | 0.0000 | -0.0418 |

## Metric interpretation boundary

`physics_rate_spearman` is a cross-sample pooled diagnostic: within each device it pools valid time points from all test prefixes before computing Spearman correlation, while the overall value is the finite mean of the three device-level correlations. It is not the same estimand as the training physics loss, which compares the learned and prior rate shapes after centering their log rates separately within each sequence. Consequently this diagnostic must not be interpreted as a direct test-loss surrogate.
