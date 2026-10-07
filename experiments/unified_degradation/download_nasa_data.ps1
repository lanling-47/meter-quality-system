param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)

$ErrorActionPreference = 'Stop'
$target = Join-Path $ProjectRoot 'data\public\nasa'
$tempDir = Join-Path $ProjectRoot 'data\public\tmp'
New-Item -ItemType Directory -Force -Path $target, $tempDir | Out-Null
$env:TEMP = $tempDir
$env:TMP = $tempDir

$datasets = @(
    @{
        Name = 'capacitor_electrical_stress.zip'
        Url = 'https://phm-datasets.s3.amazonaws.com/NASA/12.+Capacitor+Electrical+Stress.zip'
        Size = 5038942729
    },
    @{
        Name = 'igbt_accelerated_aging.zip'
        Url = 'https://phm-datasets.s3.amazonaws.com/NASA/8.+IGBT+Accelerated+Aging.zip'
        Size = 240502381
    },
    @{
        Name = 'mosfet_thermal_overstress.zip'
        Url = 'https://phm-datasets.s3.amazonaws.com/NASA/13.+MOSFET+Thermal+Overstress+Aging.zip'
        Size = 7849865909
    }
)

foreach ($dataset in $datasets) {
    $output = Join-Path $target $dataset.Name
    & curl.exe -L --fail --retry 8 --retry-delay 5 --continue-at - --output $output $dataset.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($dataset.Name)"
    }
    $actual = (Get-Item -LiteralPath $output).Length
    if ($actual -ne $dataset.Size) {
        throw "Size mismatch for $($dataset.Name): expected $($dataset.Size), got $actual"
    }
    Write-Host "Verified $($dataset.Name): $actual bytes"
}
