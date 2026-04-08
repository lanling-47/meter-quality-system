"""退化分析路由"""
import json
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/degradation", response_class=HTMLResponse)
async def degradation_page(request: Request, db: Session = Depends(get_db)):
    from app.services.degradation_service import get_all_meters_degradation_summary
    summaries = get_all_meters_degradation_summary(db)
    return templates.TemplateResponse(request, "degradation.html", {
        "summaries": summaries,
        "summaries_json": json.dumps(summaries, ensure_ascii=False, default=str),
        "page": "degradation",
    })


@router.get("/degradation/{meter_no}")
async def get_degradation(meter_no: int, db: Session = Depends(get_db)):
    from app.services.degradation_service import get_meter_degradation
    result = get_meter_degradation(meter_no, db)
    return JSONResponse(result)
