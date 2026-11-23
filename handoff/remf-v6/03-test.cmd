@echo off
setlocal
cd /d "%~dp0system"
if not defined REMF_PYTHON set "REMF_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "remf_acceptance" mkdir "remf_acceptance"
set "REMF_STATE_DIR=%TEMP%\remf-test-%RANDOM%-%RANDOM%"
set "PYTHONIOENCODING=utf-8"
"%REMF_PYTHON%" -m pytest -p no:cacheprovider -q tests/test_remf_demo.py --junitxml=remf_acceptance/integration.xml
if errorlevel 1 exit /b 1
"%REMF_PYTHON%" tests/extra_acceptance.py
if errorlevel 1 exit /b 1
echo Reports in system\remf_acceptance
