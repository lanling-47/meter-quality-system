#!/usr/bin/env python3
"""下载 CDN 资源到本地 static 目录"""
import os
import urllib.request

STATIC_DIR = "static"
os.makedirs(f"{STATIC_DIR}/css", exist_ok=True)
os.makedirs(f"{STATIC_DIR}/js", exist_ok=True)

resources = [
    ("https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css", f"{STATIC_DIR}/css/bootstrap.min.css"),
    ("https://cdn.bootcdn.net/ajax/libs/bootstrap-icons/1.11.0/font/bootstrap-icons.min.css", f"{STATIC_DIR}/css/bootstrap-icons.min.css"),
    ("https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js", f"{STATIC_DIR}/js/echarts.min.js"),
]

for url, path in resources:
    print(f"下载 {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"  -> {path}")

print("\n下载完成!")
