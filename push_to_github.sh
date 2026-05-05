#!/bin/bash
# 清理离线包并推送到 GitHub

cd /mnt/e/work

echo "=== 1. 从 Git 缓存中移除离线包目录 ==="
git rm -r --cached offline_packages_windows offline_packages_linux offline_install 2>/dev/null || echo "部分目录未被跟踪"

echo ""
echo "=== 2. 添加 .gitignore 更改 ==="
git add .gitignore

echo ""
echo "=== 3. 提交更改 ==="
git commit -m "添加离线包到gitignore，移除大文件"

echo ""
echo "=== 4. 添加或更新远程仓库 ==="
git remote add origin https://github.com/lanling-47/meter-quality-system.git 2>/dev/null || \
git remote set-url origin https://github.com/lanling-47/meter-quality-system.git

echo ""
echo "=== 5. 推送到 GitHub dev 分支 ==="
git push -u origin dev

echo ""
echo "=== 推送完成 ==="
