@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0env\install.ps1"
if errorlevel 1 pause
