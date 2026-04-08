"""根因追溯路由"""
import json
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/rootcause", response_class=HTMLResponse)
async def rootcause_page(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import QualityChain
    chains = db.query(QualityChain).all()
    meter_list = [{
        "meter_no": c.meter_no,
        "quality_label": c.quality_label,
        "quality_reason": c.quality_reason,
        "predicted_risk": round(c.predicted_risk or 0, 3),
    } for c in chains]

    # 全局特征重要度
    from app.services.shap_service import get_all_shap_summary
    summary = get_all_shap_summary(db)

    return templates.TemplateResponse(request, "root_cause.html", {
        "meter_list": meter_list,
        "meter_list_json": json.dumps(meter_list, ensure_ascii=False, default=str),
        "global_importance": json.dumps(summary.get("feature_importance", []), ensure_ascii=False),
        "page": "rootcause",
    })


@router.get("/rootcause/{meter_no}")
async def explain_meter(meter_no: int, db: Session = Depends(get_db)):
    """获取单个电表的 SHAP 解释"""
    from app.services.shap_service import explain_meter
    result = explain_meter(meter_no, db)
    return JSONResponse(result)
