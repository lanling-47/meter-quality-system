#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p offline_packages_windows
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip download -r requirements.txt -d offline_packages_windows \
  --platform win_amd64 \
  --python-version 3.11 \
  --implementation cp \
  --abi cp311 \
  --only-binary=:all:
echo "Windows Python 3.11 x64 离线依赖包已生成：offline_packages_windows"
