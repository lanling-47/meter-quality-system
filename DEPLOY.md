# 智能电表全周期质量管控系统 · Docker 部署手册

## 系统概览

- **服务端口**：8090
- **访问地址**：`http://服务器IP:8090`
- **数据持久化**：数据库 + 训练模型保存在 Docker Volume `meter_storage` 中，容器升级/重启不丢失
- **原始数据**：挂载宿主机的 `./data/raw/` 目录，只读挂载

---

## 快速部署（推荐：Docker Compose）

### 前提条件

- Docker Engine ≥ 20.10
- Docker Compose ≥ 2.0（或 docker-compose v1.29+）
- 可用内存 ≥ 1GB，磁盘 ≥ 5GB

### 第一步：准备数据文件

将以下 3 个 Excel 文件放入 `data/raw/` 目录：

```
data/raw/
├── 晶振和互感器10个样品的检验数据26.3.11.xlsx
├── 样品出厂检测和数字化电路板生产过程数据26.xlsx
└── 单相表-20版正面布局的耐久性（可靠性）试验数据26.3.6.xlsx
```

### 第二步：构建并启动

```bash
# 构建镜像（首次需要，约 5-10 分钟，需要网络）
bash deploy.sh build

# 启动服务
bash deploy.sh start

# 查看状态
bash deploy.sh status
```

访问 `http://localhost:8090` 即可打开系统界面。

---

## 工厂内网离线部署

如工厂服务器**无法访问互联网**，在有网络的机器构建后导出：

### 有网络机器（开发机）

```bash
# 1. 构建镜像
bash deploy.sh build

# 2. 导出为 tar.gz（约 1-2GB）
bash deploy.sh export
# 生成文件：meter-quality-system-1.0.0.tar.gz
```

### 工厂服务器（无网络）

```bash
# 1. 将 tar.gz 文件和整个项目目录传到工厂服务器
scp meter-quality-system-1.0.0.tar.gz factory-server:/opt/meter-quality/
scp -r . factory-server:/opt/meter-quality/

# 2. 在工厂服务器上导入镜像
cd /opt/meter-quality
docker load < meter-quality-system-1.0.0.tar.gz

# 3. 启动服务
bash deploy.sh start
```

---

## 常用运维命令

```bash
bash deploy.sh logs     # 查看实时日志
bash deploy.sh stop     # 停止服务
bash deploy.sh restart  # 重启服务
bash deploy.sh status   # 查看运行状态
```

或直接用 Docker Compose：

```bash
docker-compose up -d          # 启动
docker-compose down           # 停止
docker-compose logs -f        # 日志
docker-compose pull           # 拉取最新镜像（有镜像仓库时使用）
```

---

## 数据持久化说明

| 目录 | 类型 | 说明 |
|------|------|------|
| `/app/storage/meter_quality.db` | Docker Volume | SQLite 数据库，保存所有测量数据和质量链路 |
| `/app/storage/ml_models/` | Docker Volume | 训练好的 AI 模型文件（pkl） |
| `/app/data/raw/` | 宿主机挂载 | 原始 Excel 数据文件（只读） |

> 升级系统只需重新构建镜像，Volume 中的数据库和模型不受影响。

---

## 端口修改

如 8090 端口被占用，编辑 `docker-compose.yml` 中的 ports 配置：

```yaml
ports:
  - "9000:8090"   # 改为 9000 端口
```

---

## 系统功能模块

| 路由 | 功能 |
|------|------|
| `/dashboard` | 质量看板（KPI + 风险分布） |
| `/import` | 数据导入 / 手动录入 / 数据增强 / AI训练 |
| `/predict` | 质量风险预测 |
| `/rootcause` | 根因追溯（SHAP 解释） |
| `/degradation` | 退化趋势分析 |
| `/anomaly` | 多维用电异常检测 |

---

## 技术栈

- **后端**：Python 3.11 + FastAPI + SQLAlchemy + SQLite
- **AI**：LightGBM + SMOTE + SHAP
- **前端**：Bootstrap 5 + ECharts
- **容器**：Docker + Docker Compose
