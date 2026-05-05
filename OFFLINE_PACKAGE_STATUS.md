# 离线包制作状态

## 已完成

### Windows 离线部署包

目录：

```text
offline_packages_windows/
```

状态：已完成。

统计：

```text
依赖文件数量：62
依赖包大小：约 378.31 MB
Python 安装包：offline_install/python-3.11.9-amd64.exe，约 25 MB
```

Windows 厂方断网电脑可直接使用：

```text
start.bat
```

首次运行会：

```text
创建 venv
从 offline_packages_windows 离线安装依赖
启动 localhost:8090
```

## 未完全完成

### Linux 离线部署包

目录：

```text
offline_packages_linux/
```

当前仅完成部分包：

```text
lightgbm-4.5.0-py3-none-manylinux_2_28_x86_64.whl
numpy-2.4.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
scipy-1.17.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

原因：当前 WSL 多次出现连接超时/命令中断，无法在 Linux 原生环境完整下载。Windows 交叉下载 Linux wheel 时，部分包的平台标签不匹配，例如：

```text
lightgbm==4.5.0
pandas==2.3.3
shap==0.46.0
```

因此 Linux 包暂不建议交付使用。

## Linux 包补全方式

在一台有网络且 Linux/WSL 正常的电脑上执行：

```bash
cd /mnt/e/work
bash make_offline_linux.sh
```

完成后应生成完整的：

```text
offline_packages_linux/
```

## 推荐交付

如果厂方电脑是 Windows，当前已经可以交付：

```text
源码目录
offline_packages_windows/
offline_install/python-3.11.9-amd64.exe
start.bat
meter_quality.db
ml_models/
data/raw/
```

打包为：

```text
meter-quality-system-offline-windows.zip
```
