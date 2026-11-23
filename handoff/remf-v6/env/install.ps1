param([string]$Python, [string]$EnvDir = (Join-Path (Split-Path $PSScriptRoot) '.venv'))
$ErrorActionPreference = 'Stop'
if ($Python) { $launcher = $Python; $args32 = @() }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $launcher = 'py'; $args32 = @('-3.11') }
else { $launcher = 'python'; $args32 = @() }
& $launcher @args32 -c "import sys,struct; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Install 64-bit Python 3.11 for Windows first.'"
if ($LASTEXITCODE) { throw 'Python 3.11 x64 is required.' }
& $launcher @args32 -m venv $EnvDir
if ($LASTEXITCODE) { throw 'Could not create venv.' }
$exe = Join-Path $EnvDir 'Scripts/python.exe'
& $exe -m pip install --disable-pip-version-check --no-index --find-links (Join-Path $PSScriptRoot 'wheels') -r (Join-Path $PSScriptRoot 'requirements-cpu.txt')
if ($LASTEXITCODE) { throw 'Offline install failed.' }
& $exe -m pip check
if ($LASTEXITCODE) { throw 'Dependency check failed.' }
& $exe -c "import torch,numpy,fastapi; print('Ready:', torch.__version__, numpy.__version__, fastapi.__version__)"
exit $LASTEXITCODE
