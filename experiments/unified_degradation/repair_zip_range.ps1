param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [long]$Start,
    [Parameter(Mandatory = $true)]
    [long]$End,
    [int]$Segments = 8
)

$ErrorActionPreference = 'Stop'
$archive = [System.IO.Path]::GetFullPath($ArchivePath)
if (-not $archive.StartsWith('D:\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Archive must be on D: $archive"
}
if ($Start -lt 0 -or $End -lt $Start) {
    throw "Invalid repair range: $Start-$End"
}

$partDirectory = "$archive.repair-$Start-$End"
New-Item -ItemType Directory -Force -Path $partDirectory | Out-Null
$length = $End - $Start + 1
$segmentSize = [long][Math]::Ceiling($length / [double]$Segments)
$processes = @()
$parts = @()
for ($index = 0; $index -lt $Segments; $index++) {
    $partStart = $Start + $index * $segmentSize
    if ($partStart -gt $End) { break }
    $partEnd = [Math]::Min($End, $partStart + $segmentSize - 1)
    $partPath = Join-Path $partDirectory ("part-{0:D2}-{1}-{2}.bin" -f $index, $partStart, $partEnd)
    $parts += [pscustomobject]@{ Path = $partPath; Start = $partStart; End = $partEnd }
    $arguments = @(
        '-sS', '-L', '--fail', '--retry', '8', '--retry-delay', '3',
        '--range', "$partStart-$partEnd", '--output', $partPath, $Url
    )
    $processes += Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -NoNewWindow -PassThru
}

$processes | Wait-Process
foreach ($process in $processes) {
    if ($process.ExitCode -ne 0) {
        throw "Range repair download failed with curl exit code $($process.ExitCode)"
    }
}

$output = [System.IO.File]::Open($archive, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $parts) {
        $expected = $part.End - $part.Start + 1
        $actual = (Get-Item -LiteralPath $part.Path).Length
        if ($actual -ne $expected) {
            throw "Repair segment size mismatch: expected $expected, got $actual"
        }
        $output.Seek($part.Start, [System.IO.SeekOrigin]::Begin) | Out-Null
        $input = [System.IO.File]::OpenRead($part.Path)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally {
    $output.Dispose()
}

Remove-Item -LiteralPath $partDirectory -Recurse -Force
Write-Host "Repaired range $Start-$End in $archive"
