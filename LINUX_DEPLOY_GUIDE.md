# Linux 系统快速部署启动指南

## 1. 克隆项目

```bash
# 克隆 dev 分支
git clone -b dev https://github.com/lanling-47/meter-quality-system.git

# 进入项目目录
cd meter-quality-system
```

## 2. 检查 Python 版本

```bash
python3 --version
```

要求：Python 3.10 或以上版本。

如果版本过低，需要先安装 Python 3.11：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# CentOS/RHEL
sudo yum install python3.11 python3.11-devel
```

## 3. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

如果安装较慢，可以使用国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 5. 初始化数据库

首次运行会自动创建数据库和表结构。

如果需要导入测试数据，可以将 Excel 文件放入：

```bash
data/raw/
```

## 6. 启动系统

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8090
```

或者使用提供的启动脚本：

```bash
chmod +x start.sh
./start.sh
```

## 7. 访问系统

打开浏览器访问：

```
http://localhost:8090
```

或者从其他机器访问（将 IP 替换为实际 IP）：

```
http://服务器IP:8090
```

默认登录账号：

```
账号：admin
密码：admin123
```

## 8. 后台运行（可选）

如果需要后台运行，使用 nohup：

```bash
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8090 > uvicorn.log 2>&1 &
```

查看日志：

```bash
tail -f uvicorn.log
```

停止服务：

```bash
pkill -f "uvicorn main:app"
```

## 9. 使用 systemd 服务（推荐生产环境）

创建服务文件：

```bash
sudo nano /etc/systemd/system/meter-quality.service
```

内容：

```ini
[Unit]
Description=Meter Quality System
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/path/to/meter-quality-system
Environment="PATH=/path/to/meter-quality-system/venv/bin"
ExecStart=/path/to/meter-quality-system/venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8090
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl start meter-quality
sudo systemctl enable meter-quality
sudo systemctl status meter-quality
```

## 10. 常见问题

### 10.1 端口被占用

```bash
# 查看占用 8090 端口的进程
lsof -i :8090

# 杀死进程
kill -9 进程ID
```

### 10.2 权限问题

```bash
# 确保有执行权限
chmod +x start.sh
chmod +x make_offline_linux.sh
```

### 10.3 依赖安装失败

某些依赖需要系统库支持：

```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev libpq-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel postgresql-devel
```

### 10.4 数据库文件权限

```bash
# 确保数据库文件可写
chmod 644 meter_quality.db
```

## 11. 验证部署

访问以下页面确认系统正常：

```
http://localhost:8090/login          登录页
http://localhost:8090/setup          部署向导
http://localhost:8090/dashboard      质量看板
http://localhost:8090/factory-fault  出厂故障预测
```

## 12. 数据导入

### 方式一：上传 Excel

进入 `/import` 页面，选择 Excel 文件上传。

### 方式二：批量导入

将 Excel 文件放入：

```bash
data/raw/
```

然后在页面点击批量导入。

### 方式三：手动录入

在 `/import` 页面选择手动录入。

## 13. 模型训练

进入 `/factory-fault` 页面，点击"训练本地模型"。

前提条件：
- 至少 5 条出厂检测数据
- 至少 2 条正常样本
- 至少 2 条故障样本

## 14. 数据备份

定期备份以下文件：

```bash
meter_quality.db
ml_models/
data/raw/
```

备份命令：

```bash
tar -czf backup-$(date +%Y%m%d).tar.gz meter_quality.db ml_models/ data/raw/
```

## 15. 更新系统

```bash
cd meter-quality-system
git pull origin dev
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart meter-quality
```

## 16. 完整部署脚本

创建一键部署脚本 `deploy.sh`：

```bash
#!/bin/bash
set -e

echo "=== 智能电表质量管控系统 Linux 部署 ==="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误：未安装 Python3"
    exit 1
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 启动系统
echo "启动系统..."
echo "访问地址: http://localhost:8090"
echo "默认账号: admin"
echo "默认密码: admin123"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8090
```

使用：

```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 快速启动命令总结

```bash
# 1. 克隆项目
git clone -b dev https://github.com/lanling-47/meter-quality-system.git
cd meter-quality-system

# 2. 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 启动系统
python3 -m uvicorn main:app --host 0.0.0.0 --port 8090

# 4. 访问系统
# http://localhost:8090
# admin / admin123
```
