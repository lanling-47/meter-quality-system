@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo 制作 Windows 离线依赖包 offline_packages_windows
echo ========================================

python --version >nul 2>&1
if errorlevel 1 (
  echo [错误] 未检测到可用 Python。
  echo 当前系统可能只有 Microsoft Store 的 python 占位符。
  echo 请先安装 Python 3.11 x64，并勾选 Add Python to PATH。
  pause
  exit /b 1
)

if not exist offline_packages_windows mkdir offline_packages_windows
python -m pip install --upgrade pip wheel setuptools
python -m pip download -r requirements.txt -d offline_packages_windows

echo.
echo Windows 离线依赖包制作完成：offline_packages_windows
pause
