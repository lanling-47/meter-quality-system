# CP-UDM configuration-based paired confirmatory ablation

## Protocol

- Group split seed: 42 (one split reused by every configuration and training seed).
- Training seeds: 17, 42, 73.
- Maximum epochs / early-stopping patience: 80 / 15.
- All configurations jointly train capacitor, IGBT, and MOSFET data. For a given training seed, model initialization and data-loader randomness are reset to that same seed for every configuration.

## Mean +/- sample SD across training seeds

### overall

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.1767 +/- 0.0016 | 0.5447 +/- 0.0076 | 0.0000 +/- 0.0000 | 0.4890 +/- 0.1244 |
| no_physics | 0.1746 +/- 0.0063 | 0.6084 +/- 0.0208 | 0.0000 +/- 0.0000 | 0.2343 +/- 0.1021 |
| no_forecast | 0.1769 +/- 0.0037 | 0.5775 +/- 0.0230 | 0.0000 +/- 0.0000 | 0.4883 +/- 0.0651 |
| no_ranking_endpoint | 0.1747 +/- 0.0022 | 0.5401 +/- 0.0177 | 0.0000 +/- 0.0000 | 0.5540 +/- 0.0541 |
| no_alignment | 0.1795 +/- 0.0098 | 0.5480 +/- 0.0264 | 0.0000 +/- 0.0000 | 0.5152 +/- 0.0435 |

### capacitor

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.0765 +/- 0.0061 | 0.9587 +/- 0.0103 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_physics | 0.0637 +/- 0.0078 | 0.9503 +/- 0.0314 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_forecast | 0.0629 +/- 0.0092 | 0.9601 +/- 0.0076 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_ranking_endpoint | 0.0759 +/- 0.0009 | 0.9580 +/- 0.0080 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_alignment | 0.0809 +/- 0.0210 | 0.9544 +/- 0.0112 | 0.0000 +/- 0.0000 | NA +/- NA |

### igbt

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.1093 +/- 0.0241 | 0.9749 +/- 0.0222 | 0.0000 +/- 0.0000 | 0.7765 +/- 0.0273 |
| no_physics | 0.1231 +/- 0.0284 | 0.9621 +/- 0.0222 | 0.0000 +/- 0.0000 | 0.7210 +/- 0.0364 |
| no_forecast | 0.1273 +/- 0.0234 | 0.9702 +/- 0.0195 | 0.0000 +/- 0.0000 | 0.7805 +/- 0.0273 |
| no_ranking_endpoint | 0.1046 +/- 0.0158 | 0.9831 +/- 0.0081 | 0.0000 +/- 0.0000 | 0.7756 +/- 0.0241 |
| no_alignment | 0.1030 +/- 0.0178 | 0.9632 +/- 0.0213 | 0.0000 +/- 0.0000 | 0.7748 +/- 0.0334 |

### mosfet

| Configuration | MAE | Spearman | Trajectory MVR | Physics-rate Spearman |
|---|---:|---:|---:|---:|
| full | 0.2502 +/- 0.0034 | 0.1596 +/- 0.0187 | 0.0000 +/- 0.0000 | 0.2015 +/- 0.2224 |
| no_physics | 0.2515 +/- 0.0062 | 0.3206 +/- 0.0546 | 0.0000 +/- 0.0000 | -0.2525 +/- 0.1696 |
| no_forecast | 0.2552 +/- 0.0068 | 0.2390 +/- 0.0557 | 0.0000 +/- 0.0000 | 0.1962 +/- 0.1048 |
| no_ranking_endpoint | 0.2480 +/- 0.0025 | 0.1245 +/- 0.0437 | 0.0000 +/- 0.0000 | 0.3324 +/- 0.0862 |
| no_alignment | 0.2541 +/- 0.0039 | 0.1673 +/- 0.0297 | 0.0000 +/- 0.0000 | 0.2557 +/- 0.0771 |

## Paired deltas relative to full

Every delta below is `ablation - full` for the same training seed. Therefore a positive MAE delta means that removing the component degraded accuracy; the other metrics retain their natural signed interpretation.

| Configuration | Scope | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |
|---|---|---:|---:|---:|---:|
| no_physics | overall | -0.0021 +/- 0.0067 | 0.0636 +/- 0.0263 | 0.0000 +/- 0.0000 | -0.2547 +/- 0.1945 |
| no_physics | capacitor | -0.0129 +/- 0.0115 | -0.0083 +/- 0.0241 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_physics | igbt | 0.0138 +/- 0.0139 | -0.0128 +/- 0.0222 | 0.0000 +/- 0.0000 | -0.0554 +/- 0.0541 |
| no_physics | mosfet | 0.0013 +/- 0.0031 | 0.1610 +/- 0.0420 | 0.0000 +/- 0.0000 | -0.4540 +/- 0.3404 |
| no_forecast | overall | 0.0002 +/- 0.0030 | 0.0327 +/- 0.0260 | 0.0000 +/- 0.0000 | -0.0007 +/- 0.0636 |
| no_forecast | capacitor | -0.0137 +/- 0.0031 | 0.0014 +/- 0.0055 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_forecast | igbt | 0.0180 +/- 0.0278 | -0.0047 +/- 0.0081 | 0.0000 +/- 0.0000 | 0.0040 +/- 0.0089 |
| no_forecast | mosfet | 0.0050 +/- 0.0056 | 0.0794 +/- 0.0435 | 0.0000 +/- 0.0000 | -0.0054 +/- 0.1305 |
| no_ranking_endpoint | overall | -0.0020 +/- 0.0028 | -0.0047 +/- 0.0162 | 0.0000 +/- 0.0000 | 0.0650 +/- 0.0724 |
| no_ranking_endpoint | capacitor | -0.0007 +/- 0.0069 | -0.0007 +/- 0.0065 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_ranking_endpoint | igbt | -0.0047 +/- 0.0151 | 0.0082 +/- 0.0272 | 0.0000 +/- 0.0000 | -0.0009 +/- 0.0091 |
| no_ranking_endpoint | mosfet | -0.0022 +/- 0.0018 | -0.0352 +/- 0.0383 | 0.0000 +/- 0.0000 | 0.1308 +/- 0.1377 |
| no_alignment | overall | 0.0029 +/- 0.0114 | 0.0033 +/- 0.0280 | 0.0000 +/- 0.0000 | 0.0262 +/- 0.1227 |
| no_alignment | capacitor | 0.0043 +/- 0.0243 | -0.0043 +/- 0.0172 | 0.0000 +/- 0.0000 | NA +/- NA |
| no_alignment | igbt | -0.0063 +/- 0.0403 | -0.0117 +/- 0.0233 | 0.0000 +/- 0.0000 | -0.0017 +/- 0.0169 |
| no_alignment | mosfet | 0.0038 +/- 0.0008 | 0.0077 +/- 0.0452 | 0.0000 +/- 0.0000 | 0.0541 +/- 0.2458 |

## Per-seed paired deltas

| Configuration | Scope | Seed | Delta MAE | Delta Spearman | Delta MVR | Delta physics-rate Spearman |
|---|---|---:|---:|---:|---:|---:|
| no_physics | overall | 17 | -0.0096 | 0.0907 | 0.0000 | -0.3446 |
| no_physics | overall | 42 | 0.0032 | 0.0381 | 0.0000 | -0.0316 |
| no_physics | overall | 73 | 0.0002 | 0.0621 | 0.0000 | -0.3880 |
| no_physics | capacitor | 17 | -0.0242 | 0.0105 | 0.0000 | NA |
| no_physics | capacitor | 42 | -0.0012 | -0.0355 | 0.0000 | NA |
| no_physics | capacitor | 73 | -0.0131 | 0.0000 | 0.0000 | NA |
| no_physics | igbt | 17 | -0.0022 | 0.0000 | 0.0000 | -0.0547 |
| no_physics | igbt | 42 | 0.0215 | -0.0385 | 0.0000 | -0.0017 |
| no_physics | igbt | 73 | 0.0222 | 0.0000 | 0.0000 | -0.1098 |
| no_physics | mosfet | 17 | -0.0022 | 0.2065 | 0.0000 | -0.6345 |
| no_physics | mosfet | 42 | 0.0022 | 0.1239 | 0.0000 | -0.0614 |
| no_physics | mosfet | 73 | 0.0038 | 0.1526 | 0.0000 | -0.6662 |
| no_forecast | overall | 17 | -0.0031 | 0.0435 | 0.0000 | -0.0679 |
| no_forecast | overall | 42 | 0.0009 | 0.0516 | 0.0000 | 0.0586 |
| no_forecast | overall | 73 | 0.0028 | 0.0031 | 0.0000 | 0.0073 |
| no_forecast | capacitor | 17 | -0.0101 | 0.0067 | 0.0000 | NA |
| no_forecast | capacitor | 42 | -0.0158 | 0.0018 | 0.0000 | NA |
| no_forecast | capacitor | 73 | -0.0152 | -0.0043 | 0.0000 | NA |
| no_forecast | igbt | 17 | -0.0070 | 0.0000 | 0.0000 | 0.0111 |
| no_forecast | igbt | 42 | 0.0479 | -0.0140 | 0.0000 | 0.0069 |
| no_forecast | igbt | 73 | 0.0131 | 0.0000 | 0.0000 | -0.0059 |
| no_forecast | mosfet | 17 | 0.0019 | 0.0755 | 0.0000 | -0.1468 |
| no_forecast | mosfet | 42 | 0.0016 | 0.1247 | 0.0000 | 0.1103 |
| no_forecast | mosfet | 73 | 0.0114 | 0.0379 | 0.0000 | 0.0205 |
| no_ranking_endpoint | overall | 17 | -0.0047 | -0.0073 | 0.0000 | -0.0077 |
| no_ranking_endpoint | overall | 42 | 0.0009 | 0.0127 | 0.0000 | 0.1371 |
| no_ranking_endpoint | overall | 73 | -0.0022 | -0.0194 | 0.0000 | 0.0655 |
| no_ranking_endpoint | capacitor | 17 | -0.0086 | 0.0060 | 0.0000 | NA |
| no_ranking_endpoint | capacitor | 42 | 0.0028 | -0.0009 | 0.0000 | NA |
| no_ranking_endpoint | capacitor | 73 | 0.0037 | -0.0071 | 0.0000 | NA |
| no_ranking_endpoint | igbt | 17 | -0.0154 | 0.0385 | 0.0000 | -0.0113 |
| no_ranking_endpoint | igbt | 42 | 0.0127 | 0.0000 | 0.0000 | 0.0033 |
| no_ranking_endpoint | igbt | 73 | -0.0112 | -0.0140 | 0.0000 | 0.0054 |
| no_ranking_endpoint | mosfet | 17 | -0.0003 | -0.0590 | 0.0000 | -0.0042 |
| no_ranking_endpoint | mosfet | 42 | -0.0026 | 0.0090 | 0.0000 | 0.2710 |
| no_ranking_endpoint | mosfet | 73 | -0.0039 | -0.0555 | 0.0000 | 0.1257 |
| no_alignment | overall | 17 | -0.0026 | 0.0194 | 0.0000 | -0.0465 |
| no_alignment | overall | 42 | 0.0159 | -0.0291 | 0.0000 | 0.1679 |
| no_alignment | overall | 73 | -0.0047 | 0.0194 | 0.0000 | -0.0427 |
| no_alignment | capacitor | 17 | -0.0161 | 0.0106 | 0.0000 | NA |
| no_alignment | capacitor | 42 | 0.0312 | -0.0003 | 0.0000 | NA |
| no_alignment | capacitor | 73 | -0.0020 | -0.0232 | 0.0000 | NA |
| no_alignment | igbt | 17 | 0.0060 | 0.0035 | 0.0000 | 0.0149 |
| no_alignment | igbt | 42 | 0.0263 | -0.0385 | 0.0000 | -0.0012 |
| no_alignment | igbt | 73 | -0.0513 | 0.0000 | 0.0000 | -0.0188 |
| no_alignment | mosfet | 17 | 0.0038 | -0.0158 | 0.0000 | -0.1080 |
| no_alignment | mosfet | 42 | 0.0047 | -0.0209 | 0.0000 | 0.3370 |
| no_alignment | mosfet | 73 | 0.0030 | 0.0598 | 0.0000 | -0.0666 |

## Metric interpretation boundary

`physics_rate_spearman` is a cross-sample pooled diagnostic: within each device it pools valid time points from all test prefixes before computing Spearman correlation, while the overall value is the finite mean of the three device-level correlations. It is not the same estimand as the training physics loss, which compares the learned and prior rate shapes after centering their log rates separately within each sequence. Consequently this diagnostic must not be interpreted as a direct test-loss surrogate.
