"""出厂故障预测路由：阶段一单机导入版核心预测入口。"""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/factory-fault", response_class=HTMLResponse)
async def factory_fault_page(request: Request, db: Session = Depends(get_db)):
    from app.services.factory_fault_service import get_overview, recent_predictions, FACTORY_FEATURE_LABELS
    return templates.TemplateResponse(request, "factory_fault.html", {
        "page": "factory_fault",
        "overview": get_overview(db),
        "recent_predictions": recent_predictions(db, limit=30),
        "feature_labels": FACTORY_FEATURE_LABELS,
    })


@router.post("/factory-fault/predict")
async def predict_factory_fault(request: Request, db: Session = Depends(get_db)):
    from app.services.factory_fault_service import predict_factory_fault
    body = await request.json()
    return JSONResponse(predict_factory_fault(body, db))


@router.get("/factory-fault/predict/{meter_id}")
async def predict_factory_fault_by_meter(meter_id: str, db: Session = Depends(get_db)):
    from app.services.factory_fault_service import predict_by_meter
    result = predict_by_meter(meter_id, db)
    return JSONResponse(result, status_code=200 if result.get("success") else 404)


@router.post("/factory-fault/train")
async def train_factory_fault(db: Session = Depends(get_db)):
    from app.services.factory_fault_service import train_factory_fault_model
    result = train_factory_fault_model(db)
    return JSONResponse(result, status_code=200 if result.get("success") else 400)


@router.get("/factory-fault/overview")
async def factory_fault_overview(db: Session = Depends(get_db)):
    from app.services.factory_fault_service import get_overview
    return JSONResponse(get_overview(db))
