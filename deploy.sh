#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 智能电表全周期质量管控系统 · 一键部署脚本
# 使用方式：bash deploy.sh [build|start|stop|restart|logs|status|export]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE_NAME="meter-quality-system"
IMAGE_TAG="1.0.0"
CONTAINER_NAME="meter-quality-system"
PORT=8090

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

check_docker() {
    command -v docker &>/dev/null || error "未找到 docker，请先安装 Docker"
}

cmd=${1:-help}

case "$cmd" in
# ── 构建镜像 ───────────────────────────────────────────────────────────────
build)
    check_docker
    info "构建 Docker 镜像：${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -t "${IMAGE_NAME}:latest" .
    info "✅ 镜像构建完成"
    docker images | grep "${IMAGE_NAME}"
    ;;

# ── 启动服务（推荐使用 docker-compose） ────────────────────────────────────
start)
    check_docker
    # 检测可用的 compose 命令
    COMPOSE_CMD=""
    if command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    fi
    # docker compose (v2 内置插件) 检测
    if docker compose version &>/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    fi

    if [ -n "$COMPOSE_CMD" ]; then
        info "使用 ${COMPOSE_CMD} 启动..."
        $COMPOSE_CMD up -d
    else
        info "单容器方式启动（未找到 docker compose）..."
        # 创建持久化 volume
        docker volume create meter_storage 2>/dev/null || true
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
        docker run -d \
            --name "${CONTAINER_NAME}" \
            --restart unless-stopped \
            -p "${PORT}:8090" \
            -v meter_storage:/app/storage \
            -v "$(pwd)/data/raw:/app/data/raw:ro" \
            -e DB_PATH=/app/storage/meter_quality.db \
            -e ML_MODELS_DIR=/app/storage/ml_models \
            "${IMAGE_NAME}:latest"
    fi
    info "✅ 服务已启动，访问 http://localhost:${PORT}"
    ;;

# ── 停止服务 ──────────────────────────────────────────────────────────────
stop)
    check_docker
    if command -v docker-compose &>/dev/null || docker compose version &>/dev/null 2>&1; then
        docker-compose down 2>/dev/null || docker compose down
    else
        docker stop "${CONTAINER_NAME}" && docker rm "${CONTAINER_NAME}"
    fi
    info "✅ 服务已停止"
    ;;

# ── 重启 ──────────────────────────────────────────────────────────────────
restart)
    bash "$0" stop || true
    sleep 2
    bash "$0" start
    ;;

# ── 查看日志 ──────────────────────────────────────────────────────────────
logs)
    check_docker
    docker logs -f --tail=100 "${CONTAINER_NAME}"
    ;;

# ── 服务状态 ──────────────────────────────────────────────────────────────
status)
    check_docker
    echo ""
    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    # 健康检查
    if curl -sf "http://localhost:${PORT}/dashboard" &>/dev/null; then
        info "✅ 系统运行正常，访问：http://localhost:${PORT}"
    else
        warn "系统未响应，请检查容器日志"
    fi
    ;;

# ── 导出镜像（用于离线迁移到工厂内网） ────────────────────────────────────
export)
    check_docker
    OUTPUT="meter-quality-system-${IMAGE_TAG}.tar.gz"
    info "导出镜像到 ${OUTPUT}（可能需要几分钟）..."
    docker save "${IMAGE_NAME}:${IMAGE_TAG}" | gzip > "${OUTPUT}"
    SIZE=$(du -sh "${OUTPUT}" | cut -f1)
    info "✅ 导出完成：${OUTPUT}（${SIZE}）"
    info "工厂侧导入命令："
    echo "    docker load < ${OUTPUT}"
    echo "    bash deploy.sh start"
    ;;

# ── 帮助 ──────────────────────────────────────────────────────────────────
*)
    echo ""
    echo "用法：bash deploy.sh <命令>"
    echo ""
    echo "  build    构建 Docker 镜像"
    echo "  start    启动服务（首次需先 build）"
    echo "  stop     停止服务"
    echo "  restart  重启服务"
    echo "  logs     查看实时日志"
    echo "  status   查看服务状态"
    echo "  export   导出镜像 tar.gz（用于内网迁移）"
    echo ""
    ;;
esac
