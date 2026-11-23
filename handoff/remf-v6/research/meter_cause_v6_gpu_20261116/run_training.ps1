$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\biye\outputs\meter_v6_gpu_dependencies'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\Users\jay\anaconda3\envs\yolov9\python.exe' -u (Join-Path $PSScriptRoot 'train.py') 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'training.log')
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& 'D:\Users\jay\anaconda3\envs\yolov9\python.exe' -u (Join-Path $PSScriptRoot 'evaluate.py') 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'evaluation.log')
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
