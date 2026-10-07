# Gated residual exploratory experiment

Run from the `meter-quality-system` directory with the project's conda Python:

```powershell
& .\.conda\python.exe -m experiments.unified_degradation.run_gated_residual --epochs 120
```

The script reads cached NASA prefixes from `data/public/nasa/processed`, uses
group split seed 42, trains three neural initializations (17, 42, 73), and
evaluates the existing 400-tree random forest on exactly the same groups and
prefix features. Imputation and scaling are fitted on training groups only;
validation groups select the neural checkpoint. The output is
`results/gated_residual_exploratory.json`.

Observed neural test MAE: 0.1960, 0.1893, 0.1899. Paired RF test MAE:
0.1677, 0.1698, 0.1687. This is a negative, single-split exploratory result.
The test groups have now been inspected; further model selection needs new
outer group splits and a final untouched holdout. Neither this public-device
stage task nor its scores establish future smart-meter error prediction.
