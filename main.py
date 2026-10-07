"""FastAPI 应用入口"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from database import init_db
from app.routers import dashboard, data_import, risk_predict, root_cause, degradation, anomaly_detect
from app.routers import production, ablation, notify, retrain, data_review, factory_fault, system, data_management

app = FastAPI(
    title="智能电表全周期质量管控系统",
    description="基于多源数据融合与AI的智能电表质量风险预测、根因追溯与用电异常检测平台",
    version="1.0.0"
)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板 (添加uru参数支持异步上下文)
templates = Jinja2Templates(directory="templates")


class LoginRequiredMiddleware(BaseHTTPMiddleware):
    """阶段一单机部署的轻量访问保护。"""

    async def dispatch(self, request: Request, call_next):
        public_prefixes = ("/login", "/static", "/docs", "/openapi.json", "/redoc")
        if request.url.path.startswith(public_prefixes):
            return await call_next(request)
        if request.cookies.get("meter_quality_login") == "ok":
            return await call_next(request)
        # API calls must receive a machine-readable error. Redirecting a POST
        # to the login form turns it into an unrelated validation error.
        accepts = request.headers.get("accept", "")
        if request.method != "GET" or "application/json" in accepts or "text/event-stream" in accepts:
            return JSONResponse(
                {
                    "success": False,
                    "code": "AUTH_REQUIRED",
                    "error": "登录已失效，请重新登录后再操作。",
                    "login_url": "/login",
                },
                status_code=401,
            )
        return RedirectResponse(url="/login")


app.add_middleware(LoginRequiredMiddleware)

# 注册路由
app.include_router(system.router)
app.include_router(dashboard.router)
app.include_router(data_import.router)
app.include_router(data_management.router)
app.include_router(risk_predict.router)
app.include_router(root_cause.router)
app.include_router(degradation.router)
app.include_router(anomaly_detect.router)
app.include_router(production.router)
app.include_router(factory_fault.router)
app.include_router(ablation.router)
app.include_router(notify.router)
app.include_router(retrain.router)
app.include_router(data_review.router)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库 + 启动文件监控服务"""
    import asyncio
    init_db()
    print("数据库初始化完成")
    # 自动导入数据（如数据库为空）
    try:
        from app.services.etl_service import auto_import_on_startup
        result = auto_import_on_startup()
        if result:
            print(f"数据自动导入完成：{result}")
    except Exception as e:
        print(f"数据自动导入失败（可稍后手动导入）：{e}")

    # 启动文件监控服务（生产线实时采集）
    try:
        from app.services.file_watcher_service import get_watcher
        from database import SessionLocal
        watcher = get_watcher()
        watcher.set_db_factory(SessionLocal)
        asyncio.create_task(watcher.start())
        print("文件监控服务已启动")
    except Exception as e:
        print(f"文件监控服务启动失败：{e}")


@app.get("/")
async def root():
    return RedirectResponse(url="/setup")
