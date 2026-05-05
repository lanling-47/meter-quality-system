#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "========================================"
echo "智能电表质量管控系统 - Linux 离线一键启动"
echo "========================================"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[错误] 未检测到 Python3，请先安装 Python 3.10+"
  exit 1
fi

if [ ! -d venv ]; then
  echo "首次运行，正在创建虚拟环境..."
  python3 -m venv venv
  source venv/bin/activate
  echo "正在从 offline_packages_linux 离线安装依赖..."
  if [ ! -d offline_packages_linux ]; then
    echo "[错误] offline_packages_linux 目录不存在。"
    echo "请先在有网 Linux/WSL 电脑运行 make_offline_linux.sh。"
    exit 1
  fi
  pip install --no-index --find-links=offline_packages_linux -r requirements.txt
else
  source venv/bin/activate
fi

echo "系统启动中..."
echo "访问地址: http://localhost:8090"
echo "默认账号: admin"
echo "默认密码: admin123"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8090
