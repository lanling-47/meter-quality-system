#!/bin/bash
# 下载 CDN 资源到本地 static 目录
# 运行: bash download_assets.sh

STATIC_DIR="static"
mkdir -p "$STATIC_DIR/css" "$STATIC_DIR/js"

echo "下载 Bootstrap CSS..."
curl -sL "https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" -o "$STATIC_DIR/css/bootstrap.min.css"

echo "下载 Bootstrap Icons CSS..."
curl -sL "https://cdn.bootcdn.net/ajax/libs/bootstrap-icons/1.11.0/font/bootstrap-icons.min.css" -o "$STATIC_DIR/css/bootstrap-icons.min.css"

echo "下载 ECharts JS..."
curl -sL "https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js" -o "$STATIC_DIR/js/echarts.min.js"

echo "下载完成！"
ls -la "$STATIC_DIR/css/" "$STATIC_DIR/js/"
