# Event-time counterfactual traceability confirmation

> **SEMI-SYNTHETIC EVIDENCE ONLY.** No enterprise production row is used.
> The results test implementation mechanisms; they do not identify a Pearl-style
> causal effect and do not establish field or enterprise generalization.

## Frozen protocol

- Production groups: 120 total; 78/18/24 train/validation/test.
- Meters: 2880 total; groups never cross splits.
- Five balanced upstream causes and a balanced no-fault reference stratum are
  generated. Ranking is evaluated only on incident test groups. A station
  sensor is a nuisance,
  a two-meter coating marker is a low-support control, and a rework code is
  generated from the result at a strictly later timestamp.
- Random forests use training groups only. Platt calibration uses validation
  groups only. Signed thermal/calibration offsets enter the risk model as
  deviations from their frozen nominal targets. Conditional references are
  qualified training meters inside a training-only robust nominal band, matched
  first on product model and shift.
- The proposed score is weighted positive counterfactual risk reduction times
  N_eff/(N_eff+kappa) times empirical common support. Candidates below frozen
  N_eff/reference/overlap thresholds are marked evidence-insufficient.
- Fixed model/randomization seeds: (17, 42, 73); batch
  bootstrap replicates: 2000.

## Held-out production-group root ranking

Each cell is seed mean +/- seed standard deviation [production-group bootstrap 95% CI].

| Method | Top-1 | Top-3 | MRR | NDCG |
|---|---:|---:|---:|---:|
| proposed_temporal_cf | 0.650 +/- 0.087 [0.450, 0.833] | 0.967 +/- 0.029 [0.900, 1.000] | 0.787 +/- 0.042 [0.662, 0.912] | 0.841 +/- 0.030 [0.744, 0.928] |
| marginal_correlation | 0.650 +/- 0.000 [0.450, 0.850] | 0.700 +/- 0.000 [0.500, 0.900] | 0.729 +/- 0.000 [0.562, 0.887] | 0.793 +/- 0.000 [0.671, 0.914] |
| static_unfiltered_cf | 0.017 +/- 0.029 [0.000, 0.050] | 0.817 +/- 0.029 [0.650, 0.967] | 0.441 +/- 0.020 [0.388, 0.487] | 0.583 +/- 0.014 [0.542, 0.621] |
| time_shuffled_cf | 0.017 +/- 0.029 [0.000, 0.050] | 0.850 +/- 0.087 [0.717, 0.967] | 0.439 +/- 0.004 [0.389, 0.482] | 0.581 +/- 0.002 [0.540, 0.615] |
| random_eligible | 0.133 +/- 0.153 [0.067, 0.200] | 0.433 +/- 0.144 [0.333, 0.533] | 0.376 +/- 0.088 [0.316, 0.435] | 0.526 +/- 0.065 [0.481, 0.572] |

## Paired differences from the proposed method

Positive values favor `proposed_temporal_cf`; intervals resample test production groups.

| Baseline | Top-1 difference [95% CI] | MRR difference [95% CI] |
|---|---:|---:|
| marginal_correlation | +0.000 [-0.250, +0.267] | +0.058 [-0.113, +0.240] |
| static_unfiltered_cf | +0.633 [+0.433, +0.833] | +0.346 [+0.252, +0.431] |
| time_shuffled_cf | +0.633 [+0.433, +0.817] | +0.349 [+0.250, +0.439] |
| random_eligible | +0.517 [+0.300, +0.717] | +0.411 [+0.262, +0.550] |

`marginal_correlation` uses only a training-set point-biserial association
times the current group deviation. `static_unfiltered_cf` admits every
timestamp. `time_shuffled_cf` permutes timestamps within each meter before
applying the same filter. `random_eligible` samples only candidates that pass
the proposed eligibility checks.

## Negative controls and integrity

- Exact post-outcome injection/deletion invariance: True (max prediction/eligible-score deltas 0/0; max rank delta 0).
- Low/common-support coating candidates all rejected: True (maximum overlap 0.000, threshold 0.150; maximum N_eff 2.000, threshold 6.000).
- Proposed post-outcome candidates all excluded: True.
- Static unfiltered leakage candidate Top-1 rate: 0.950.
- After timestamp shuffling, leakage candidate eligibility/Top-1 rates: 0.917/0.917.
- Split disjointness: True; ranking keys unique: True; prediction keys unique: True; expected row counts complete: True.

## Interpretation boundary

The experiment can support or falsify structural statements about temporal
exclusion, confidence/uncertainty weighting, effective-sample shrinkage, common
support, and controlled ranking under this known generator. A high ranking score
is still model sensitivity, not an identified intervention effect. The static
and shuffled controls may attain deceptively strong outcome prediction by using
the result-derived rework code; that is leakage evidence, not a useful gain.
Real production traceability remains unverified until timestamp provenance,
identity links, repair/replace outcomes, and blind engineer adjudication are
available on genuinely held-out enterprise batches.
