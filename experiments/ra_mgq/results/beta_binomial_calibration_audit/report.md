# Beta--Binomial batch posterior calibration audit

**SEMI-SYNTHETIC / MONTE CARLO EVIDENCE ONLY**

This audit tests when batch-level shrinkage improves probability estimation
and when a wrong prior damages it. It does not use enterprise meter data, and
it does not claim that the mature Beta--Binomial conjugate model is novel.

## Protocol

- Latent batch risks: `pi_b ~ Beta(2, 8)` (mean 0.200).
- Inspection counts: `k_b | pi_b ~ Binomial(n_b, pi_b)`.
- Sample sizes: 1, 2, 4, 8, 16, 32, 64.
- Seeds: 17, 42, 73; 20,000 independent batches per size and seed.
- Point scores are expected one-step-ahead Bernoulli Brier/NLL values.
- Coverage targets the latent `pi_b`; raw rates use a Wilson 90% interval and Beta estimators use equal-tailed posterior credible intervals.
- Brackets report 95% Monte Carlo confidence intervals from independent batch-level contributions.

## Estimators

- `raw_rate_wilson`: unsmoothed `k/n` point estimate with a Wilson interval.
- `beta_uniform`: `Beta(1,1)` prior, matching the current lightweight prototype.
- `beta_oracle`: generator-matched `Beta(2, 8)`; an unavailable upper-bound comparator.
- `beta_mismatched`: deliberately wrong `Beta(8, 2)`, whose mean 0.800 reverses the generator mean.

## Results

| n | Estimator | Brier [95% MC CI] | NLL [95% MC CI] | 90% coverage [95% MC CI] | Width [95% MC CI] |
|---:|---|---:|---:|---:|---:|
| 1 | raw_rate_wilson | 0.29060 [0.28863, 0.29256] | 4.01473 [3.98754, 4.04193] | 0.8942 [0.8917, 0.8966] | 0.7301 [0.7301, 0.7301] |
| 1 | beta_uniform | 0.20798 [0.20732, 0.20863] | 0.60689 [0.60553, 0.60826] | 0.9018 [0.8994, 0.9042] | 0.7511 [0.7511, 0.7511] |
| 1 | beta_oracle | 0.15894 [0.15837, 0.15952] | 0.49711 [0.49577, 0.49844] | 0.8985 [0.8961, 0.9009] | 0.3698 [0.3696, 0.3700] |
| 1 | beta_mismatched | 0.45588 [0.45538, 0.45637] | 1.15490 [1.15372, 1.15608] | 0.0114 [0.0105, 0.0122] | 0.4072 [0.4070, 0.4074] |
| 2 | raw_rate_wilson | 0.21834 [0.21717, 0.21950] | 2.22124 [2.20351, 2.23896] | 0.9154 [0.9132, 0.9177] | 0.6278 [0.6271, 0.6284] |
| 2 | beta_uniform | 0.18967 [0.18902, 0.19033] | 0.56523 [0.56381, 0.56666] | 0.9035 [0.9012, 0.9059] | 0.6477 [0.6473, 0.6481] |
| 2 | beta_oracle | 0.15738 [0.15682, 0.15795] | 0.49276 [0.49143, 0.49409] | 0.8987 [0.8963, 0.9011] | 0.3539 [0.3536, 0.3541] |
| 2 | beta_mismatched | 0.40762 [0.40719, 0.40806] | 1.03547 [1.03442, 1.03651] | 0.0181 [0.0170, 0.0192] | 0.4129 [0.4126, 0.4131] |
| 4 | raw_rate_wilson | 0.18175 [0.18097, 0.18253] | 1.29612 [1.28617, 1.30607] | 0.9221 [0.9199, 0.9242] | 0.5075 [0.5067, 0.5083] |
| 4 | beta_uniform | 0.17309 [0.17248, 0.17370] | 0.52641 [0.52503, 0.52780] | 0.9086 [0.9063, 0.9109] | 0.5210 [0.5203, 0.5216] |
| 4 | beta_oracle | 0.15559 [0.15503, 0.15614] | 0.48741 [0.48610, 0.48872] | 0.9010 [0.8986, 0.9034] | 0.3273 [0.3270, 0.3276] |
| 4 | beta_mismatched | 0.33955 [0.33917, 0.33994] | 0.88163 [0.88076, 0.88250] | 0.0336 [0.0322, 0.0351] | 0.4061 [0.4060, 0.4063] |
| 8 | raw_rate_wilson | 0.16375 [0.16313, 0.16436] | 0.81428 [0.80869, 0.81987] | 0.9166 [0.9144, 0.9188] | 0.3896 [0.3888, 0.3903] |
| 8 | beta_uniform | 0.16139 [0.16083, 0.16195] | 0.49872 [0.49739, 0.50004] | 0.9105 [0.9083, 0.9128] | 0.3962 [0.3956, 0.3969] |
| 8 | beta_oracle | 0.15360 [0.15306, 0.15414] | 0.48117 [0.47988, 0.48246] | 0.8996 [0.8972, 0.9020] | 0.2884 [0.2881, 0.2888] |
| 8 | beta_mismatched | 0.26477 [0.26439, 0.26514] | 0.72466 [0.72387, 0.72544] | 0.0710 [0.0689, 0.0730] | 0.3721 [0.3720, 0.3722] |
| 16 | raw_rate_wilson | 0.15443 [0.15388, 0.15498] | 0.57622 [0.57356, 0.57889] | 0.9111 [0.9088, 0.9133] | 0.2880 [0.2874, 0.2885] |
| 16 | beta_uniform | 0.15379 [0.15325, 0.15432] | 0.47951 [0.47821, 0.48080] | 0.9088 [0.9065, 0.9111] | 0.2906 [0.2900, 0.2911] |
| 16 | beta_oracle | 0.15092 [0.15039, 0.15146] | 0.47279 [0.47150, 0.47407] | 0.8996 [0.8972, 0.9020] | 0.2394 [0.2391, 0.2398] |
| 16 | beta_mismatched | 0.20409 [0.20367, 0.20452] | 0.59890 [0.59802, 0.59978] | 0.1462 [0.1434, 0.1490] | 0.3079 [0.3078, 0.3079] |
| 32 | raw_rate_wilson | 0.15021 [0.14969, 0.15074] | 0.49103 [0.48961, 0.49246] | 0.9049 [0.9026, 0.9073] | 0.2094 [0.2089, 0.2098] |
| 32 | beta_uniform | 0.15006 [0.14953, 0.15058] | 0.46916 [0.46788, 0.47043] | 0.9032 [0.9008, 0.9056] | 0.2102 [0.2098, 0.2106] |
| 32 | beta_oracle | 0.14911 [0.14859, 0.14963] | 0.46678 [0.46551, 0.46805] | 0.8983 [0.8959, 0.9007] | 0.1884 [0.1881, 0.1888] |
| 32 | beta_mismatched | 0.16954 [0.16907, 0.17002] | 0.52177 [0.52074, 0.52281] | 0.2723 [0.2688, 0.2759] | 0.2316 [0.2315, 0.2318] |
| 64 | raw_rate_wilson | 0.14757 [0.14706, 0.14809] | 0.46581 [0.46456, 0.46706] | 0.9026 [0.9002, 0.9050] | 0.1498 [0.1495, 0.1501] |
| 64 | beta_uniform | 0.14753 [0.14702, 0.14804] | 0.46168 [0.46042, 0.46294] | 0.9015 [0.8991, 0.9039] | 0.1501 [0.1498, 0.1504] |
| 64 | beta_oracle | 0.14725 [0.14674, 0.14776] | 0.46095 [0.45969, 0.46221] | 0.8998 [0.8974, 0.9022] | 0.1416 [0.1414, 0.1419] |
| 64 | beta_mismatched | 0.15384 [0.15334, 0.15433] | 0.48157 [0.48042, 0.48273] | 0.4341 [0.4301, 0.4380] | 0.1640 [0.1638, 0.1641] |

## Interpretation

- At `n=1`, the matched prior changes Brier by -0.13165 relative to raw `k/n`; the uniform prior changes it by -0.08262.
- At `n=1`, the deliberately mismatched prior changes Brier by +0.24790 relative to the uniform prior. This is the required negative control: shrinkage is
  beneficial only when its prior is sufficiently compatible with the target batches.
- At `n=64`, the Brier range across the three Beta estimators is 0.00658; inspection evidence increasingly dominates prior choice.
- Raw `k/n` NLL is especially poor for tiny samples because `k=0` and `k=n` produce boundary probabilities; clipping is used only to keep the log score finite.
- The matched-prior interval contracts from width 0.3698 at `n=1` to 0.1416 at `n=64` while its coverage can be checked directly against the nominal 0.90 target.

These results validate a limited statistical statement about shrinkage and prior
sensitivity under the declared generator. They do not validate cross-batch meter
prediction, mapping-confidence gating, enterprise generalization, or causal tracing.
