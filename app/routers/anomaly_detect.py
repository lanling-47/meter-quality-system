"""用电异常检测路由"""
import json
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/anomaly", response_class=HTMLResponse)
async def anomaly_page(request: Request, db: Session = Depends(get_db)):
    from app.services.anomaly_service import get_power_data, get_energy_comparison
    data = get_power_data(db)
    energy_cmp = get_energy_comparison(db)
    return templates.TemplateResponse(request, "anomaly_detect.html", {
        "anomaly_data": data,
        "chart_data_json": json.dumps(data.get("chart_data", {}), ensure_ascii=False),
        "anomaly_meters_json": json.dumps(data.get("anomaly_meters", []), ensure_ascii=False, default=str),
        "energy_cmp_json": json.dumps(energy_cmp.get("data", []), ensure_ascii=False, default=str),
        "thresholds_json": json.dumps(data.get("thresholds", {}), ensure_ascii=False),
        "page": "anomaly",
    })


@router.get("/anomaly/meter/{meter_no}")
async def get_meter_anomaly(meter_no: int, db: Session = Depends(get_db)):
    from app.services.anomaly_service import get_power_trend
    result = get_power_trend(meter_no, db)
    return JSONResponse(result)


@router.get("/anomaly/energy/compare")
async def get_energy_compare(db: Session = Depends(get_db)):
    """获取全部电表电量对比数据"""
    from app.services.anomaly_service import get_energy_comparison
    result = get_energy_comparison(db)
    return JSONResponse(result)
