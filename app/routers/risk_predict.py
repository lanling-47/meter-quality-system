"""风险预测路由"""
import json
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import QualityChain, ModelVersion
    chains = db.query(QualityChain).all()
    active_model = db.query(ModelVersion).filter_by(is_active=True).first()

    chain_list = [{
        "meter_no": c.meter_no,
        "quality_label": c.quality_label,
        "predicted_risk": round(c.predicted_risk or 0, 3),
        "predicted_label": c.predicted_label,
        "crystal_ppm": c.crystal_ppm,
        "ct_ratio_err": c.ct_ratio_err,
        "quality_reason": c.quality_reason,
    } for c in chains]

    return templates.TemplateResponse(request, "risk_predict.html", {
        "chain_list": chain_list,
        "chain_list_json": json.dumps(chain_list, ensure_ascii=False, default=str),
        "active_model": active_model,
        "page": "predict",
    })


@router.post("/predict/single")
async def predict_single(request: Request, db: Session = Depends(get_db)):
    """单次预测 API"""
    from app.services.predict_service import predict_single
    body = await request.json()
    try:
        result = predict_single(body, db)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/predict/batch")
async def predict_batch(request: Request, db: Session = Depends(get_db)):
    """批量重新预测所有电表"""
    from app.services.predict_service import predict_single
    from app.models.db_models import QualityChain
    chains = db.query(QualityChain).all()
    results = []
    for c in chains:
        features = {
            "crystal_ppm": c.crystal_ppm or 9.0,
            "ct_ratio_err": c.ct_ratio_err or 0.5,
            "clock_accuracy_sd": c.clock_accuracy_sd or 0.03,
            "factory_fault": c.factory_fault,
            "initial_error_max": c.initial_error_max or 0.0,
            "final_error_max": c.final_error_max or 0.0,
            "error_degradation_rate": c.error_degradation_rate or 0.0,
            "clock_drift_rate": c.clock_drift_rate or 0.0,
            "rs485_total_errors": c.rs485_total_errors or 0,
        }
        r = predict_single(features, db)
        c.predicted_risk = r.get("risk_prob", 0)
        c.predicted_label = r.get("risk_label", 0)
        results.append({"meter_no": c.meter_no, **r})
    db.commit()
    return JSONResponse({"success": True, "results": results})
