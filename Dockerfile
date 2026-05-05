# ─────────────────────────────────────────────────────────────────────────────
# 智能电表全周期质量管控系统 · Docker 镜像
# 基于 Python 3.11-slim，轻量生产级镜像
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# 维护者
LABEL maintainer="hujie09" \
      version="1.0.0" \
      description="智能电表全周期质量管控工程系统"

# ── 系统依赖 ─────────────────────────────────────────────────────────────────
# libomp: LightGBM 必须依赖; libgomp1: scikit-learn 多线程; curl: 健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 工作目录 ─────────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python 依赖（先复制 requirements，充分利用 Docker 层缓存）────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── 复制应用代码 ──────────────────────────────────────────────────────────────
COPY . .

# ── 创建数据卷挂载目录 ────────────────────────────────────────────────────────
# /app/data        : 存放原始 Excel 数据文件（可从宿主机挂载）
# /app/storage     : 数据库 + 训练好的模型文件（持久化，不会随容器销毁）
RUN mkdir -p /app/data/raw /app/storage/ml_models

# 将数据库和模型目录重定向到持久化存储目录
# config.py 会读取环境变量 DB_PATH / ML_MODELS_DIR 来覆盖默认路径
ENV DB_PATH=/app/storage/meter_quality.db \
    ML_MODELS_DIR=/app/storage/ml_models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── 暴露端口 ──────────────────────────────────────────────────────────────────
EXPOSE 8090

# ── 健康检查（容器启动 30s 后每 30s 检查一次） ────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8090/dashboard || exit 1

# ── 启动命令 ──────────────────────────────────────────────────────────────────
# workers=1 避免 SQLite 多进程写冲突；生产环境如换 PostgreSQL 可增加 workers
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8090", \
     "--workers", "1", \
     "--access-log"]
