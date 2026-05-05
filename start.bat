@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo 智能电表质量管控系统 - Windows 离线一键启动
echo ========================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python。请先安装 Python 3.11 x64。
    echo 如果已准备安装包，请运行 offline_install 目录中的 python 安装程序。
    pause
    exit /b 1
)

if not exist "venv\" (
    echo 首次运行，正在创建虚拟环境...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 正在从 offline_packages_windows 离线安装依赖...
    if not exist "offline_packages_windows\" (
        echo [错误] offline_packages_windows 目录不存在。
        echo 请先在有网电脑运行 make_offline_windows.bat。
        pause
        exit /b 1
    )
    pip install --no-index --find-links=offline_packages_windows -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo 系统启动中...
echo 访问地址: http://localhost:8090
echo 默认账号: admin
echo 默认密码: admin123
echo 关闭此窗口将停止服务。

echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8090
pause
