# 智能电表质量管控系统 - 离线一键部署说明

本系统支持在厂方断网电脑上离线部署和一键启动。

## 1. 离线部署包目录结构

交付给厂方时，建议包含以下内容：

```text
meter-quality-system/
├── app/
├── data/
├── templates/
├── ml_models/
├── main.py
├── config.py
├── database.py
├── requirements.txt
├── meter_quality.db
├── start.bat
├── start.sh
├── make_offline_package.bat
├── make_offline_package.sh
├── offline_packages/
└── offline_install/
```

其中：

- `start.bat`：Windows 一键启动脚本；
- `start.sh`：Linux/WSL 一键启动脚本；
- `offline_packages/`：离线 Python 依赖包；
- `offline_install/`：可放 Python 安装包；
- `meter_quality.db`：SQLite 数据库；
- `ml_models/`：模型文件目录；
- `data/raw/`：Excel 原始数据目录。

## 2. 在有网络电脑上制作离线依赖包

### Windows

双击运行：

```text
make_offline_package.bat
```

### Linux/WSL

```bash
chmod +x make_offline_package.sh
./make_offline_package.sh
```

执行后会生成：

```text
offline_packages/
```

里面包含所有离线安装需要的 `.whl` / `.tar.gz` 依赖包。

## 3. 准备 Python 安装包

厂方电脑如果没有 Python，需要提前下载 Python 安装程序。

推荐版本：Python 3.11.x

Windows 安装包下载地址：

```text
https://www.python.org/downloads/windows/
```

下载后放入：

```text
offline_install/
```

例如：

```text
offline_install/python-3.11.9-amd64.exe
```

安装时注意勾选：

```text
Add Python to PATH
```

## 4. 厂方电脑离线部署步骤

### Windows

1. 将整个项目目录复制到厂方电脑，例如：

```text
D:\meter-quality-system\
```

2. 如果没有 Python，先安装：

```text
offline_install/python-3.11.9-amd64.exe
```

3. 双击运行：

```text
start.bat
```

首次运行会自动完成：

```text
创建 venv 虚拟环境
离线安装依赖
启动 Web 服务
```

4. 浏览器访问：

```text
http://localhost:8090
```

默认账号：

```text
admin
```

默认密码：

```text
admin123
```

### Linux/WSL

```bash
cd meter-quality-system
chmod +x start.sh
./start.sh
```

## 5. 数据和模型持久化

系统使用相对路径，不依赖开发电脑的绝对路径。

默认路径如下：

```text
数据库：项目目录/meter_quality.db
原始数据：项目目录/data/raw/
模型目录：项目目录/ml_models/
监听目录：项目上级目录/watch_dir/
```

如果项目复制到：

```text
D:\meter-quality-system\
```

则自动变为：

```text
D:\meter-quality-system\meter_quality.db
D:\meter-quality-system\data\raw\
D:\meter-quality-system\ml_models\
D:\watch_dir\
```

## 6. 常见问题

### 6.1 端口 8090 被占用

Windows：

```cmd
netstat -ano | findstr :8090
taskkill /F /PID 进程ID
```

Linux/WSL：

```bash
lsof -i :8090
kill 进程ID
```

### 6.2 提示找不到 Python

请先安装 Python，并确认安装时勾选：

```text
Add Python to PATH
```

### 6.3 离线依赖安装失败

检查目录是否存在：

```text
offline_packages/
```

如果不存在，需要在有网络电脑上运行：

```text
make_offline_package.bat
```

或：

```bash
./make_offline_package.sh
```

## 7. 交付建议

最终给厂方的压缩包建议命名为：

```text
meter-quality-system-offline.zip
```

压缩包内必须包含：

```text
源码
requirements.txt
start.bat
start.sh
offline_packages/
offline_install/ 可选
meter_quality.db
ml_models/
data/raw/
```

这样厂方电脑断网情况下也可以直接运行。
