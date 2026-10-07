param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [long]$ExpectedSize,
    [int]$Segments = 6
)

$ErrorActionPreference = 'Stop'
$resolvedPath = [System.IO.Path]::GetFullPath($Path)
if (-not $resolvedPath.StartsWith('D:\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Download path must be on D: $resolvedPath"
}

$prefixLength = (Get-Item -LiteralPath $resolvedPath).Length
if ($prefixLength -ge $ExpectedSize) {
    throw "Existing file is not a partial prefix: $prefixLength >= $ExpectedSize"
}

$partDirectory = "$resolvedPath.parts"
New-Item -ItemType Directory -Force -Path $partDirectory | Out-Null
$remaining = $ExpectedSize - $prefixLength
$segmentSize = [long][Math]::Ceiling($remaining / [double]$Segments)
$processes = @()
$parts = @()

for ($index = 0; $index -lt $Segments; $index++) {
    $start = $prefixLength + $index * $segmentSize
    if ($start -ge $ExpectedSize) { break }
    $end = [Math]::Min($ExpectedSize - 1, $start + $segmentSize - 1)
    $partPath = Join-Path $partDirectory ("part-{0:D2}-{1}-{2}.bin" -f $index, $start, $end)
    $parts += [pscustomobject]@{ Path = $partPath; Start = $start; End = $end }
    $arguments = @(
        '-L', '--fail', '--retry', '8', '--retry-delay', '3',
        '--range', "$start-$end", '--output', $partPath, $Url
    )
    $processes += Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -NoNewWindow -PassThru
}

$processes | Wait-Process
foreach ($process in $processes) {
    if ($process.ExitCode -ne 0) {
        throw "Segment download failed with curl exit code $($process.ExitCode)"
    }
}

foreach ($part in $parts) {
    $expectedPartSize = $part.End - $part.Start + 1
    $actualPartSize = (Get-Item -LiteralPath $part.Path).Length
    if ($actualPartSize -ne $expectedPartSize) {
        throw "Segment size mismatch for $($part.Path): expected $expectedPartSize, got $actualPartSize"
    }
}

$assembledPath = "$resolvedPath.assembled"
$output = [System.IO.File]::Open($assembledPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
    $input = [System.IO.File]::OpenRead($resolvedPath)
    try { $input.CopyTo($output) } finally { $input.Dispose() }
    foreach ($part in $parts) {
        $input = [System.IO.File]::OpenRead($part.Path)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally {
    $output.Dispose()
}

$actualSize = (Get-Item -LiteralPath $assembledPath).Length
if ($actualSize -ne $ExpectedSize) {
    throw "Assembled size mismatch: expected $ExpectedSize, got $actualSize"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($assembledPath)
try {
    foreach ($entry in $archive.Entries) {
        $stream = $entry.Open()
        try {
            $buffer = New-Object byte[] (4MB)
            while ($stream.Read($buffer, 0, $buffer.Length) -gt 0) {}
        } finally {
            $stream.Dispose()
        }
    }
} finally {
    $archive.Dispose()
}

$backupPath = "$resolvedPath.partial-backup"
Move-Item -LiteralPath $resolvedPath -Destination $backupPath
Move-Item -LiteralPath $assembledPath -Destination $resolvedPath
Remove-Item -LiteralPath $backupPath -Force
Remove-Item -LiteralPath $partDirectory -Recurse -Force
Write-Host "Verified complete ZIP: $resolvedPath ($ExpectedSize bytes)"
