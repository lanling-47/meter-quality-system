"""FastAPI 应用入口"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from database import init_db
from app.routers import dashboard, data_import, risk_predict, root_cause, degradation, anomaly_detect

app = FastAPI(
    title="智能电表全周期质量管控系统",
    description="基于多源数据融合与AI的智能电表质量风险预测、根因追溯与用电异常检测平台",
    version="1.0.0"
)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板
templates = Jinja2Templates(directory="templates")

# 注册路由
app.include_router(dashboard.router)
app.include_router(data_import.router)
app.include_router(risk_predict.router)
app.include_router(root_cause.router)
app.include_router(degradation.router)
app.include_router(anomaly_detect.router)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    init_db()
    print("✅ 数据库初始化完成")
    # 自动导入数据（如数据库为空）
    try:
        from app.services.etl_service import auto_import_on_startup
        result = auto_import_on_startup()
        if result:
            print(f"✅ 数据自动导入完成：{result}")
    except Exception as e:
        print(f"⚠️  数据自动导入失败（可稍后手动导入）：{e}")


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")
