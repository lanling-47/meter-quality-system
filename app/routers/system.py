"""登录、退出与阶段一部署向导路由。"""
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.cookies.get("meter_quality_login") == "ok":
        return RedirectResponse(url="/setup")
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def do_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    import config as cfg
    if username == cfg.LOGIN_USER and password == cfg.LOGIN_PASSWORD:
        response = RedirectResponse(url="/setup", status_code=303)
        response.set_cookie(
            "meter_quality_login",
            "ok",
            max_age=cfg.LOGIN_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        response.set_cookie("meter_quality_user", username, max_age=cfg.LOGIN_COOKIE_MAX_AGE, samesite="lax")
        return response
    return templates.TemplateResponse(request, "login.html", {"error": "账号或密码错误"}, status_code=401)


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("meter_quality_login")
    response.delete_cookie("meter_quality_user")
    return response


@router.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request, db: Session = Depends(get_db)):
    import config as cfg
    from app.models.db_models import CrystalData, CTData, FactoryTest, AgingTest, MeterQualityTrace, ModelVersion, DataIngestionLog

    stats = {
        "crystal": db.query(CrystalData).count(),
        "ct": db.query(CTData).count(),
        "factory": db.query(FactoryTest).count(),
        "aging": db.query(AgingTest).count(),
        "production_trace": db.query(MeterQualityTrace).count(),
        "models": db.query(ModelVersion).count(),
        "ingestion_logs": db.query(DataIngestionLog).count(),
    }
    readiness = {
        "data_dir_exists": cfg.DATA_DIR.exists(),
        "watch_dir_exists": cfg.WATCH_DIR.exists(),
        "db_parent_exists": cfg.DB_PATH.parent.exists(),
        "model_dir_exists": cfg.ML_MODELS_DIR.exists(),
        "has_factory_samples": stats["factory"] >= 5,
        "has_production_trace": stats["production_trace"] > 0,
    }
    return templates.TemplateResponse(request, "setup.html", {
        "page": "setup",
        "stats": stats,
        "readiness": readiness,
        "paths": {
            "db_path": str(cfg.DB_PATH),
            "data_dir": str(cfg.DATA_DIR),
            "watch_dir": str(cfg.WATCH_DIR),
            "model_dir": str(cfg.ML_MODELS_DIR),
        },
    })


@router.get("/templates/download/{template_type}")
async def download_template(template_type: str):
    from app.services.template_service import build_template_workbook
    from urllib.parse import quote
    workbook, filename = build_template_workbook(template_type)

    import io
    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    # HTTP headers must be latin-1 encodable in Starlette. Use RFC 5987
    # filename* encoding so Chinese template filenames download correctly.
    encoded_filename = quote(filename.encode("utf-8"))
    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"{template_type}_template.xlsx\"; "
            f"filename*=UTF-8''{encoded_filename}"
        )
    }
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@router.get("/templates/list")
async def list_templates():
    from app.services.template_service import TEMPLATE_DEFINITIONS
    return {"items": [v for v in TEMPLATE_DEFINITIONS.values()]}
