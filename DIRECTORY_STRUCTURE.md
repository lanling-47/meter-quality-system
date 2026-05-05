# 项目目录结构说明

## 核心程序文件

```text
main.py                     FastAPI 主程序入口
config.py                   全局配置文件
database.py                 数据库连接配置
requirements.txt            Python 依赖列表
meter_quality.db            SQLite 本地数据库
```

## 业务代码目录

```text
app/                        后端业务逻辑
├── services/               业务服务层
├── models/                 数据模型
└── routers/                路由控制器

templates/                  Jinja2 页面模板
static/                     静态资源（CSS/JS/图片）
data/                       数据目录
├── raw/                    原始 Excel 批量导入目录
ml_models/                  本地训练模型存储目录
```

## Windows 离线部署文件

```text
start.bat                                       Windows 一键启动脚本
make_offline_windows.bat                        Windows 离线包制作脚本
offline_packages_windows/                       Windows 离线依赖包目录（62个.whl文件）
offline_install/                                Python 离线安装包目录
└── python-3.11.9-amd64.exe                     Python 3.11 Windows 安装包
WINDOWS_OFFLINE_INSTALL_AND_USER_GUIDE.md       Windows 离线安装与使用指南
```

## Linux 离线部署文件

```text
start.sh                                Linux/WSL 一键启动脚本
make_offline_linux.sh                   Linux 离线包制作脚本
make_offline_windows_from_linux.sh      从 Linux 交叉下载 Windows 包脚本
offline_packages_linux/                 Linux 离线依赖包目录（未完整）
```

## Docker 部署文件

```text
Dockerfile                  Docker 镜像构建文件
docker-compose.yml          Docker Compose 编排文件
.dockerignore               Docker 构建忽略文件
```

## 文档文件

```text
WINDOWS_OFFLINE_INSTALL_AND_USER_GUIDE.md   Windows 离线安装与使用指南（简明版）
README_OFFLINE.md                           离线部署包制作指南
OFFLINE_PACKAGE_STATUS.md                   离线包制作状态说明
DEPLOY.md                                   通用部署文档
PROJECT_MEMORY.md                           项目记忆文档
```

## Git 相关

```text
.git/                       Git 版本控制目录
.gitignore                  Git 忽略文件配置
```

## 目录用途说明

### 数据持久化目录

```text
meter_quality.db            所有业务数据（出厂检测、生产追溯、老化试验等）
ml_models/                  训练好的本地模型文件
data/raw/                   原始 Excel 数据文件
```

这三个目录/文件需要定期备份。

### 离线部署目录

```text
offline_packages_windows/   Windows 厂方断网电脑使用
offline_packages_linux/     Linux 厂方断网电脑使用（未完整）
offline_install/            Python 离线安装包
```

### 启动脚本

```text
start.bat                   Windows 双击启动
start.sh                    Linux/WSL 执行启动
```

### 离线包制作脚本

```text
make_offline_windows.bat                在有网 Windows 电脑制作 Windows 离线包
make_offline_linux.sh                   在有网 Linux/WSL 制作 Linux 离线包
make_offline_windows_from_linux.sh      从 Linux 交叉下载 Windows 包（备用）
```

## 厂方交付建议

### Windows 厂方电脑

打包以下内容：

```text
app/
data/
templates/
static/
ml_models/
offline_packages_windows/
offline_install/
main.py
config.py
database.py
requirements.txt
meter_quality.db
start.bat
WINDOWS_OFFLINE_INSTALL_AND_USER_GUIDE.md
OFFLINE_PACKAGE_STATUS.md
```

压缩为：

```text
meter-quality-system-offline-windows.zip
```

### Linux 厂方电脑

Linux 离线包目前未完整制作，需要在有网 Linux/WSL 环境补全：

```bash
bash make_offline_linux.sh
```

## 不需要交付的文件

以下文件仅用于开发和构建，不需要交付给厂方：

```text
.git/
.gitignore
.dockerignore
Dockerfile
docker-compose.yml
PROJECT_MEMORY.md
DEPLOY.md
make_offline_*.sh
make_offline_*.bat
README_OFFLINE.md
```

## 当前状态

```text
✅ Windows 离线包已完成（62个依赖，约378MB）
✅ Python 离线安装包已下载（约25MB）
✅ Windows 离线安装指南已完成
⚠️  Linux 离线包未完整（仅3个依赖）
✅ 系统代码完整
✅ 测试数据已导入
✅ 本地模型已训练
```

## 目录大小参考

```text
offline_packages_windows/       约 378 MB
offline_install/                约 25 MB
meter_quality.db                约 220 KB
ml_models/                      约 1-10 MB（取决于训练模型数量）
app/                            约 1 MB
templates/                      约 500 KB
static/                         约 2 MB
```

总计 Windows 离线交付包大小：

```text
约 400-450 MB
```
