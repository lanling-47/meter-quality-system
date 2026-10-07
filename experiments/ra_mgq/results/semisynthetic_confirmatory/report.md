# RA-MGQ semi-synthetic confirmatory experiment

> SEMI-SYNTHETIC EVIDENCE ONLY. This report does not contain enterprise
> production data and must not be presented as field generalization evidence.

## Fixed generator and protocol

For batch `b` and meter `i`, the generator draws independent `B_b, U_bi ~ N(0,1)`
and samples `Y_bi` from `sigmoid(-0.35 + 1.25 B_b + 1.10 U_bi +
0.35 tanh(B_b U_bi))`. Meter features are noisy measurements of `U_bi`;
batch evidence comes from a separate binomial inspection and a Beta(1,1)
posterior. The dataset and batch split are generated once; seeds 17/42/73
change model initialization and minibatch order only.

- Batches: 72 total, 48/12/12 train/validation/test
- Meters: 1728 total
- All preprocessing statistics are fitted on training batches only.
- Error mappings are injected only after training and only into test inputs.

## Beta posterior contraction

| Inspection n | Positive k | Posterior mean | Posterior variance |
|---:|---:|---:|---:|
| 2 | 1 | 0.5000 | 0.050000 |
| 4 | 2 | 0.5000 | 0.035714 |
| 8 | 4 | 0.5000 | 0.022727 |
| 16 | 8 | 0.5000 | 0.013158 |
| 32 | 16 | 0.5000 | 0.007143 |
| 64 | 32 | 0.5000 | 0.003731 |

## Consistency confidence budget audit

The fixed two-batch fixture calls the production `_batch_consistency_loss`
directly. Uniform confidence leaves each within-batch aggregate unchanged;
because confidence is excluded from the final normalization, the absolute
loss and its training-path logit gradient should scale linearly with confidence.

| Mapping confidence | Absolute loss | Loss / c=1 | Gradient L1 | Gradient / c=1 |
|---:|---:|---:|---:|---:|
| 1 | 0.7617747789 | 1.0000000000 | 0.1833829201 | 1.0000000000 |
| 0.5 | 0.3808873894 | 0.5000000000 | 0.0916914601 | 0.5000000000 |
| 0.1 | 0.0761774779 | 0.1000000000 | 0.0183382920 | 0.1000000000 |
| 1e-09 | 0.0000000008 | 0.0000000010 | 0.0000000002 | 0.0000000010 |
| 0 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 |

Maximum loss/gradient ratio errors are 0/4.16e-17. At zero confidence, exact zero loss/gradient are True/True. This audit checks
the fixed-logit gradient budget of the auxiliary batch-supervision path.
It does not claim that the end-to-end model output is linear in confidence:
the learned fusion gate and the remaining training objectives are nonlinear.

## Clean held-out-batch performance

| Model | AUROC | AUPRC | Brier | ECE | Batch Brier | Within-batch variance |
|---|---:|---:|---:|---:|---:|---:|
| direct_fusion | 0.7589 +/- 0.0107 | 0.7641 +/- 0.0209 | 0.2042 +/- 0.0055 | 0.0761 +/- 0.0119 | 0.0066 +/- 0.0009 | 0.0250 +/- 0.0070 |
| meter_only | 0.6172 +/- 0.0050 | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0800 +/- 0.0176 | 0.0405 +/- 0.0001 | 0.0278 +/- 0.0070 |
| ra_mgq | 0.7525 +/- 0.0036 | 0.7533 +/- 0.0103 | 0.2075 +/- 0.0030 | 0.0728 +/- 0.0194 | 0.0080 +/- 0.0023 | 0.0323 +/- 0.0090 |
| ra_mgq_no_consistency | 0.7558 +/- 0.0099 | 0.7656 +/- 0.0164 | 0.2078 +/- 0.0038 | 0.0857 +/- 0.0168 | 0.0084 +/- 0.0025 | 0.0310 +/- 0.0074 |

## Mapping-error perturbation

`confidence_aware` assigns each corrupted link confidence `1-error_rate`;
`overconfident` leaves corrupted links at confidence 1.0. The latter is a
negative control: structural gating cannot repair a falsely trusted mapping.

| Scenario | Error rate | Model | AUPRC | Brier | Batch Brier | ECE |
|---|---:|---|---:|---:|---:|---:|
| confidence_aware | 0% | direct_fusion | 0.7641 +/- 0.0209 | 0.2042 +/- 0.0055 | 0.0066 +/- 0.0009 | 0.0761 +/- 0.0119 |
| confidence_aware | 10% | direct_fusion | 0.7487 +/- 0.0169 | 0.2090 +/- 0.0044 | 0.0075 +/- 0.0010 | 0.0798 +/- 0.0064 |
| confidence_aware | 30% | direct_fusion | 0.6899 +/- 0.0158 | 0.2290 +/- 0.0041 | 0.0114 +/- 0.0008 | 0.1094 +/- 0.0060 |
| confidence_aware | 50% | direct_fusion | 0.6561 +/- 0.0153 | 0.2543 +/- 0.0021 | 0.0173 +/- 0.0010 | 0.1489 +/- 0.0032 |
| overconfident | 0% | direct_fusion | 0.7641 +/- 0.0209 | 0.2042 +/- 0.0055 | 0.0066 +/- 0.0009 | 0.0761 +/- 0.0119 |
| overconfident | 10% | direct_fusion | 0.7483 +/- 0.0170 | 0.2092 +/- 0.0044 | 0.0075 +/- 0.0010 | 0.0790 +/- 0.0074 |
| overconfident | 30% | direct_fusion | 0.6917 +/- 0.0113 | 0.2298 +/- 0.0048 | 0.0113 +/- 0.0010 | 0.1115 +/- 0.0103 |
| overconfident | 50% | direct_fusion | 0.6544 +/- 0.0141 | 0.2567 +/- 0.0027 | 0.0177 +/- 0.0012 | 0.1471 +/- 0.0090 |
| confidence_aware | 0% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| confidence_aware | 10% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| confidence_aware | 30% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| confidence_aware | 50% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| overconfident | 0% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| overconfident | 10% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| overconfident | 30% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| overconfident | 50% | meter_only | 0.6042 +/- 0.0102 | 0.2450 +/- 0.0041 | 0.0405 +/- 0.0001 | 0.0800 +/- 0.0176 |
| confidence_aware | 0% | ra_mgq | 0.7533 +/- 0.0103 | 0.2075 +/- 0.0030 | 0.0080 +/- 0.0023 | 0.0728 +/- 0.0194 |
| confidence_aware | 10% | ra_mgq | 0.7424 +/- 0.0091 | 0.2130 +/- 0.0039 | 0.0093 +/- 0.0024 | 0.0765 +/- 0.0263 |
| confidence_aware | 30% | ra_mgq | 0.6857 +/- 0.0109 | 0.2253 +/- 0.0014 | 0.0126 +/- 0.0024 | 0.0954 +/- 0.0087 |
| confidence_aware | 50% | ra_mgq | 0.6813 +/- 0.0142 | 0.2453 +/- 0.0056 | 0.0185 +/- 0.0026 | 0.1236 +/- 0.0256 |
| overconfident | 0% | ra_mgq | 0.7533 +/- 0.0103 | 0.2075 +/- 0.0030 | 0.0080 +/- 0.0023 | 0.0728 +/- 0.0194 |
| overconfident | 10% | ra_mgq | 0.7414 +/- 0.0090 | 0.2135 +/- 0.0039 | 0.0093 +/- 0.0024 | 0.0769 +/- 0.0261 |
| overconfident | 30% | ra_mgq | 0.6812 +/- 0.0096 | 0.2282 +/- 0.0016 | 0.0129 +/- 0.0021 | 0.1031 +/- 0.0067 |
| overconfident | 50% | ra_mgq | 0.6604 +/- 0.0106 | 0.2585 +/- 0.0076 | 0.0197 +/- 0.0025 | 0.1483 +/- 0.0211 |
| confidence_aware | 0% | ra_mgq_no_consistency | 0.7656 +/- 0.0164 | 0.2078 +/- 0.0038 | 0.0084 +/- 0.0025 | 0.0857 +/- 0.0168 |
| confidence_aware | 10% | ra_mgq_no_consistency | 0.7519 +/- 0.0147 | 0.2136 +/- 0.0044 | 0.0097 +/- 0.0027 | 0.0835 +/- 0.0292 |
| confidence_aware | 30% | ra_mgq_no_consistency | 0.6943 +/- 0.0177 | 0.2273 +/- 0.0035 | 0.0125 +/- 0.0023 | 0.1139 +/- 0.0181 |
| confidence_aware | 50% | ra_mgq_no_consistency | 0.6859 +/- 0.0175 | 0.2468 +/- 0.0055 | 0.0183 +/- 0.0025 | 0.1423 +/- 0.0223 |
| overconfident | 0% | ra_mgq_no_consistency | 0.7656 +/- 0.0164 | 0.2078 +/- 0.0038 | 0.0084 +/- 0.0025 | 0.0857 +/- 0.0168 |
| overconfident | 10% | ra_mgq_no_consistency | 0.7513 +/- 0.0149 | 0.2139 +/- 0.0043 | 0.0097 +/- 0.0027 | 0.0855 +/- 0.0318 |
| overconfident | 30% | ra_mgq_no_consistency | 0.6904 +/- 0.0151 | 0.2300 +/- 0.0037 | 0.0127 +/- 0.0017 | 0.1192 +/- 0.0017 |
| overconfident | 50% | ra_mgq_no_consistency | 0.6678 +/- 0.0123 | 0.2598 +/- 0.0063 | 0.0194 +/- 0.0023 | 0.1568 +/- 0.0169 |

## Structural and trade-off audits

| Model | Seed | Maximum logit change at zero confidence |
|---|---:|---:|
| meter_only | 17 | 0 |
| ra_mgq_no_consistency | 17 | 0 |
| ra_mgq | 17 | 0 |
| meter_only | 42 | 0 |
| ra_mgq_no_consistency | 42 | 0 |
| ra_mgq | 42 | 0 |
| meter_only | 73 | 0 |
| ra_mgq_no_consistency | 73 | 0 |
| ra_mgq | 73 | 0 |

At 50% confidence-aware mapping error, direct-fusion Brier changes by +0.0501, while RA-MGQ changes by +0.0378.
Adding aggregate consistency changes clean batch Brier by -0.0004, AUPRC by -0.0123, and within-batch prediction variance by +0.0013.
These are descriptive semi-synthetic effects, not pre-declared universal
performance guarantees.

## Interpretation boundary

This experiment can falsify implementation-level claims about posterior
contraction, zero-confidence invariance, sensitivity to mapping corruption,
and the aggregate-consistency trade-off. It cannot establish performance on
real meters, real production batches, or real causal root causes. In
particular, the overconfident negative control must be retained when reporting
robustness: RA-MGQ depends on provenance confidence being informative.
