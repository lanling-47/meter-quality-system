# Docker 一键部署指南

## 1. 前置要求

确保已安装 Docker 和 Docker Compose：

```bash
# 检查 Docker
docker --version

# 检查 Docker Compose
docker-compose --version
```

如果未安装，请参考：
- Docker: https://docs.docker.com/engine/install/
- Docker Compose: https://docs.docker.com/compose/install/

---

## 2. 一键启动（最简单）

```bash
# 克隆项目
git clone -b dev https://github.com/lanling-47/meter-quality-system.git
cd meter-quality-system

# 一键启动
docker-compose up -d
```

启动后访问：

```
http://localhost:8090
```

登录：

```
账号：admin
密码：admin123
```

---

## 3. 详细部署步骤

### 3.1 克隆项目

```bash
git clone -b dev https://github.com/lanling-47/meter-quality-system.git
cd meter-quality-system
```

### 3.2 构建镜像

```bash
docker-compose build
```

或者单独构建：

```bash
docker build -t meter-quality-system:latest .
```

### 3.3 启动容器

```bash
# 后台启动
docker-compose up -d

# 前台启动（查看日志）
docker-compose up
```

### 3.4 查看状态

```bash
# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看最近 100 行日志
docker-compose logs --tail=100
```

---

## 4. 常用操作

### 4.1 停止服务

```bash
docker-compose stop
```

### 4.2 重启服务

```bash
docker-compose restart
```

### 4.3 停止并删除容器

```bash
docker-compose down
```

### 4.4 停止并删除容器和数据卷

```bash
# ⚠️ 警告：会删除所有数据
docker-compose down -v
```

### 4.5 进入容器

```bash
docker-compose exec meter-quality bash
```

### 4.6 查看容器日志

```bash
# 实时查看
docker-compose logs -f meter-quality

# 查看最近 50 行
docker-compose logs --tail=50 meter-quality
```

---

## 5. 数据持久化

Docker 部署使用数据卷持久化存储：

```yaml
volumes:
  meter_storage:/app/storage
```

数据存储位置：

```
/app/storage/meter_quality.db      # SQLite 数据库
/app/storage/ml_models/            # 训练的模型文件
```

即使容器删除重建，数据也不会丢失。

---

## 6. 数据导入

### 方式一：通过 Web 界面上传

访问 `http://localhost:8090/import` 上传 Excel 文件。

### 方式二：挂载本地目录

将 Excel 文件放入项目的 `data/raw/` 目录：

```bash
cp your_data.xlsx meter-quality-system/data/raw/
```

然后在 Web 界面点击批量导入。

### 方式三：复制到容器

```bash
docker cp your_data.xlsx meter-quality-system:/app/data/raw/
```

---

## 7. 端口配置

默认端口：`8090`

如需修改，编辑 `docker-compose.yml`：

```yaml
ports:
  - "9000:8090"  # 宿主机端口:容器端口
```

然后重启：

```bash
docker-compose down
docker-compose up -d
```

---

## 8. 环境变量配置

可以在 `docker-compose.yml` 中修改环境变量：

```yaml
environment:
  - DB_PATH=/app/storage/meter_quality.db
  - ML_MODELS_DIR=/app/storage/ml_models
  - LOGIN_USER=admin
  - LOGIN_PASSWORD=admin123
  - WATCH_INTERVAL_SEC=300
```

---

## 9. 数据备份

### 9.1 备份数据卷

```bash
# 创建备份目录
mkdir -p backups

# 备份数据卷
docker run --rm \
  -v meter-quality-system_meter_storage:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/meter-storage-$(date +%Y%m%d).tar.gz -C /data .
```

### 9.2 恢复数据卷

```bash
# 恢复数据
docker run --rm \
  -v meter-quality-system_meter_storage:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/meter-storage-20260504.tar.gz -C /data
```

### 9.3 直接复制数据库

```bash
# 备份数据库
docker cp meter-quality-system:/app/storage/meter_quality.db ./backup/

# 恢复数据库
docker cp ./backup/meter_quality.db meter-quality-system:/app/storage/
```

---

## 10. 更新系统

```bash
# 1. 拉取最新代码
git pull origin dev

# 2. 重新构建镜像
docker-compose build

# 3. 重启容器
docker-compose down
docker-compose up -d
```

---

## 11. 生产环境部署建议

### 11.1 使用私有镜像仓库

```yaml
services:
  meter-quality:
    image: registry.yourcompany.com/meter-quality-system:1.0.0
```

### 11.2 配置资源限制

```yaml
services:
  meter-quality:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 11.3 配置重启策略

```yaml
services:
  meter-quality:
    restart: always
```

### 11.4 配置日志轮转

```yaml
services:
  meter-quality:
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
```

---

## 12. 健康检查

容器自带健康检查，每 30 秒检查一次：

```bash
# 查看健康状态
docker-compose ps
```

健康检查端点：

```
http://localhost:8090/dashboard
```

---

## 13. 多容器部署（可选）

如果需要使用 PostgreSQL 替代 SQLite：

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: meter_quality
      POSTGRES_USER: meter
      POSTGRES_PASSWORD: your_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  meter-quality:
    image: meter-quality-system:latest
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql://meter:your_password@postgres:5432/meter_quality

volumes:
  postgres_data:
  meter_storage:
```

---

## 14. 常见问题

### 14.1 容器启动失败

查看日志：

```bash
docker-compose logs meter-quality
```

### 14.2 端口被占用

修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "9000:8090"
```

### 14.3 数据丢失

确保使用了数据卷，不要使用 `docker-compose down -v`。

### 14.4 权限问题

```bash
# 修复权限
docker-compose exec meter-quality chown -R root:root /app/storage
```

---

## 15. 完整部署脚本

创建 `docker-deploy.sh`：

```bash
#!/bin/bash
set -e

echo "=== Docker 一键部署智能电表质量管控系统 ==="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误：未安装 Docker"
    exit 1
fi

# 检查 Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "错误：未安装 Docker Compose"
    exit 1
fi

# 构建镜像
echo "构建 Docker 镜像..."
docker-compose build

# 启动容器
echo "启动容器..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查状态
echo "检查容器状态..."
docker-compose ps

echo ""
echo "=== 部署完成 ==="
echo "访问地址: http://localhost:8090"
echo "默认账号: admin"
echo "默认密码: admin123"
echo ""
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose stop"
echo "重启服务: docker-compose restart"
```

使用：

```bash
chmod +x docker-deploy.sh
./docker-deploy.sh
```

---

## 16. 快速命令总结

```bash
# 一键启动
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose stop

# 重启服务
docker-compose restart

# 删除容器（保留数据）
docker-compose down

# 删除容器和数据
docker-compose down -v

# 进入容器
docker-compose exec meter-quality bash

# 备份数据
docker cp meter-quality-system:/app/storage/meter_quality.db ./backup/

# 更新系统
git pull && docker-compose build && docker-compose up -d
```

---

## 17. 验证部署

部署成功后，访问以下页面验证：

```
http://localhost:8090/login          登录页
http://localhost:8090/setup          部署向导
http://localhost:8090/dashboard      质量看板
http://localhost:8090/factory-fault  出厂故障预测
http://localhost:8090/production     生产追溯
http://localhost:8090/degradation    老化分析
```

---

## 18. 优势

Docker 部署的优势：

✅ 环境一致性：开发、测试、生产环境完全一致  
✅ 快速部署：一条命令启动整个系统  
✅ 易于迁移：打包镜像即可迁移到任何服务器  
✅ 资源隔离：不影响宿主机其他服务  
✅ 易于扩展：可以轻松添加数据库、缓存等服务  
✅ 数据持久化：使用数据卷保证数据安全  
✅ 自动重启：容器异常退出会自动重启  
✅ 健康检查：自动监控服务状态  

---

## 19. 与传统部署对比

| 特性 | Docker 部署 | 传统部署 |
|---|---|---|
| 部署时间 | 5 分钟 | 30+ 分钟 |
| 环境依赖 | 无需安装 Python | 需要安装 Python 和所有依赖 |
| 环境一致性 | 完全一致 | 可能存在差异 |
| 迁移难度 | 极简单 | 较复杂 |
| 资源隔离 | 完全隔离 | 共享宿主机环境 |
| 回滚 | 秒级 | 需要重新部署 |

---

推荐使用 Docker 部署，特别是在生产环境。
