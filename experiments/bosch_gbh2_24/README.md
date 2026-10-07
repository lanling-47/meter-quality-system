# Bosch GBH 2-24 industrial-object experiment

This experiment connects the Bosch GBH 2-24 MES test results with the repair workbook through the serial number. It is an object-adapter and traceability experiment for the unified degradation framework.

The experiment deliberately keeps the raw Bosch workbooks outside Git. Set the following variables or pass the two paths explicitly:

```powershell
$env:BOSCH_GBH2_MES = 'C:\path\MES 生产记录.xlsx'
$env:BOSCH_GBH2_REPAIR = 'C:\path\GBH2-24 2026.08 维修记录.xlsx'
python experiments/bosch_gbh2_24/run_bosch_experiment.py
```

The script writes aggregate outputs to `experiments/bosch_gbh2_24/results/`:

- `summary.json`: reproducible counts and linkage quality flags;
- `parameter_summary.csv`: parameter stage, inferred meaning, unit, state and raw range audit;
- `repair_part_counts.csv`: repair-part distribution without serial numbers;
- `repair_model_mapping.csv`: MES tool names among matched serials;
- `report.md`: compact report used by the thesis section.
- `response_metrics.csv`: held-out same-cycle WP3 reconstruction MAE/RMSE;
- `response_stratified.csv`: aggregate held-out errors by tool and WP2 operating bands;
- `response_pair_summary.csv`: WP2-to-WP3 current correlation and paired difference summaries;
- `fitted_models.json`: training-only preprocessing and frozen ridge coefficients;
- `response_reconstruction.pdf/.png`: aggregate experimental figure;
- `response_relationship.pdf/.png`: held-out WP2/WP3 current relationship figure.

The parameter mapping follows the Bosch test description supplied with the data: WP2 half/full-voltage run-in, WP3 post-run test, WP4 impact performance, WP5 clutch torque, WP6 no-load test and WP7 high-voltage leakage test. `WP1DummyCurrent10`, `WP7Similarity`, `WP7Volt` and `WP7Current` remain diagnostic-only until the Bosch test dictionary confirms their semantics and scaling. WP4 is retained as a proxy but masked from physical modeling because m/ss is not an energy unit. Raw state codes are preserved, not interpreted as pass/fail.

## Frozen response reconstruction protocol

Use only each device's earliest test; require positive finite, singly recorded WP2 half/full voltages and currents and WP3 current. Exclude 67 devices, leaving 10,843. Assign whole observed calendar days to approximately 60/20/20 chronological day groups, yielding 6,183/2,473/2,187 devices. Use type, WP2 voltage, optionally WP2 current; never WP3/6/7 response, retest count or repairs as inputs. Standardization and type vocabulary are fitted on training only. Select ridge alpha from [0.01,0.1,1,10,100] by validation MAE; no test refit. Compare against unchanged WP2 full current and type+voltage ridge. Bootstrap 1,000 paired test-device error differences with seed 20261007. This interval is conditional on the observed cohort and ignores potential within-day/lot correlation; it is not a population or cross-factory guarantee. The protocol was specified before the initial Bosch test score was inspected; this is one internal holdout, not prospective validation.

The full ridge achieves 0.059046 A test MAE, versus 0.184318 A for type+voltage and 0.410741 A for unchanged current. This establishes same-cycle association, not a damage-rate model. A stage number is metadata, not verified damaging exposure or duration. Of 442 retested devices, 425 have all tests on one day; do not treat these as lifetime trajectories.

Repairs are read only from the master sheet, excluding the footer and filtered views. There are 3,101 records/2,789 serials and 65 exact MES matches; 58/64 August manufacturing serials match. Serial matching is descriptive and still requires manufacturing-month/model audits. The repair workbook does not establish complete follow-up for every MES serial number, so a missing repair record is not treated as a negative label. There is no validated lifetime, failure-probability, cross-object transfer or deployed web integration claim.

Dependencies: Python, openpyxl, numpy, matplotlib. Verification: `python -m unittest discover -s tests -p test_bosch_experiment.py -v`. Raw XLSX are read-only; the committed outputs do not contain serial numbers or QR URLs.
