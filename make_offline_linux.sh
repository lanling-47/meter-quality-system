#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p offline_packages_linux
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip download -r requirements.txt -d offline_packages_linux
echo "Linux 离线依赖包已生成：offline_packages_linux"
