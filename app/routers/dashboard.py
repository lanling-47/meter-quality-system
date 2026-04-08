"""质量看板路由"""
import json
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import QualityChain, CrystalData, CTData, FactoryTest, AgingTest, ModelVersion

    # KPI 统计
    total = db.query(QualityChain).count()
    high_risk = db.query(QualityChain).filter_by(quality_label=1).count()
    normal = total - high_risk
    crystal_count = db.query(CrystalData).count()
    factory_count = db.query(FactoryTest).count()
    aging_count = db.query(AgingTest).count()

    # 平均值
    chains = db.query(QualityChain).all()
    avg_ppm = None
    avg_ct = None
    if chains:
        ppms = [c.crystal_ppm for c in chains if c.crystal_ppm is not None]
        cts = [c.ct_ratio_err for c in chains if c.ct_ratio_err is not None]
        avg_ppm = round(sum(ppms) / len(ppms), 2) if ppms else None
        avg_ct = round(sum(cts) / len(cts), 2) if cts else None

    # 激活模型信息
    active_model = db.query(ModelVersion).filter_by(is_active=True).first()

    # 质量链路数据（用于图表）
    chain_list = []
    for c in chains:
        chain_list.append({
            "meter_no": c.meter_no,
            "quality_label": c.quality_label or 0,
            "final_error_max": float(c.final_error_max) if c.final_error_max is not None else None,
            "clock_drift_rate": float(c.clock_drift_rate) if c.clock_drift_rate is not None else None,
            "predicted_risk": float(c.predicted_risk) if c.predicted_risk is not None else None,
            "quality_reason": c.quality_reason or "",
        })

    # 风险分布饼图数据
    pie_data = [
        {"name": "正常", "value": normal},
        {"name": "高风险", "value": high_risk},
    ]

    # 误差分布柱状图
    bar_meters = [c["meter_no"] for c in chain_list]
    bar_errors = [round(c["final_error_max"] or 0, 4) for c in chain_list]
    bar_colors = ["#e74c3c" if c["quality_label"] == 1 else "#2ecc71" for c in chain_list]

    stats = {
        "total": total,
        "high_risk": high_risk,
        "normal": normal,
        "crystal_count": crystal_count,
        "factory_count": factory_count,
        "aging_count": aging_count,
        "avg_ppm": avg_ppm,
        "avg_ct": avg_ct,
    }

    return templates.TemplateResponse(request, "dashboard.html", {
        "stats": stats,
        "chain_list": chain_list,
        "pie_data": json.dumps(pie_data, ensure_ascii=False),
        "bar_meters": json.dumps(bar_meters),
        "bar_errors": json.dumps(bar_errors),
        "bar_colors": json.dumps(bar_colors),
        "active_model": active_model,
        "page": "dashboard",
    })
