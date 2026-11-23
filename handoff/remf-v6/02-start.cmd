@echo off
setlocal
cd /d "%~dp0system"
if not defined REMF_PYTHON set "REMF_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined REMF_PORT set "REMF_PORT=8092"
if not exist "%REMF_PYTHON%" (
  echo Run 01-install.cmd first.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:%REMF_PORT% in your browser. Stop with Ctrl+C.
"%REMF_PYTHON%" -m uvicorn remf_main:app --host 127.0.0.1 --port %REMF_PORT%
if errorlevel 1 exit /b 1
