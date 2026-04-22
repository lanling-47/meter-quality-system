"""数据采集审核中心路由"""
import json
from datetime import datetime
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/data-review", response_class=HTMLResponse)
async def data_review_page(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import DataIngestionLog, QualityLabelReview, ModelVersion
    import config as cfg

    pending_reviews = db.query(QualityLabelReview).filter_by(review_status="pending").count()
    confirmed_not_trained = db.query(QualityLabelReview).filter_by(
        review_status="confirmed", added_to_train=False
    ).count()
    active_model = db.query(ModelVersion).filter_by(is_active=True).first()
    total_logs = db.query(DataIngestionLog).count()

    return templates.TemplateResponse(request, "data_review.html", {
        "page": "data_review",
        "watch_dir": str(cfg.WATCH_DIR),
        "interval_sec": cfg.WATCH_INTERVAL_SEC,
        "threshold": cfg.AUTO_RETRAIN_THRESHOLD,
        "pending_reviews": pending_reviews,
        "confirmed_not_trained": confirmed_not_trained,
        "active_model_tag": active_model.version_tag if active_model else "无",
        "active_model_auc": active_model.auc_score if active_model else None,
        "total_logs": total_logs,
    })


@router.get("/data-review/logs")
async def get_ingestion_logs(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    from app.models.db_models import DataIngestionLog
    total = db.query(DataIngestionLog).count()
    logs = db.query(DataIngestionLog).order_by(
        DataIngestionLog.parsed_at.desc()
    ).offset((page - 1) * page_size).limit(page_size).all()
    return JSONResponse({
        "total": total,
        "page": page,
        "items": [l.to_dict() for l in logs],
    })


@router.get("/data-review/queue")
async def get_review_queue(
    request: Request,
    status: str = "pending",
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    from app.models.db_models import QualityLabelReview
    q = db.query(QualityLabelReview).filter_by(review_status=status)
    total = q.count()
    items = q.order_by(QualityLabelReview.created_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    return JSONResponse({
        "total": total,
        "page": page,
        "items": [i.to_dict() for i in items],
    })


@router.post("/data-review/confirm")
async def confirm_review(request: Request, db: Session = Depends(get_db)):
    """
    人工确认/拒绝标注
    body: { ids: [1,2,3], action: "confirm"|"reject", manual_label: 0|1, note: "" }
    """
    body = await request.json()
    ids = body.get("ids", [])
    action = body.get("action", "confirm")
    manual_label = body.get("manual_label")  # None 表示保留自动标注
    note = body.get("note", "")
    reviewer = body.get("reviewer", "admin")

    from app.models.db_models import QualityLabelReview
    updated = 0
    for rid in ids:
        review = db.query(QualityLabelReview).filter_by(id=rid).first()
        if not review:
            continue
        review.review_status = "confirmed" if action == "confirm" else "rejected"
        review.reviewed_by = reviewer
        review.review_note = note
        review.reviewed_at = datetime.utcnow()
        if manual_label is not None:
            review.manual_label = int(manual_label)
        elif review.auto_label is not None:
            review.manual_label = review.auto_label  # 接受自动标注
        updated += 1

    # 同时更新关联的采集日志状态
    from app.models.db_models import DataIngestionLog
    log_ids = {r.ingestion_log_id for r in db.query(QualityLabelReview).filter(
        QualityLabelReview.id.in_(ids)
    ).all() if r.ingestion_log_id}
    for lid in log_ids:
        log = db.query(DataIngestionLog).filter_by(id=lid).first()
        if log:
            log.status = "reviewed"
            log.reviewed_at = datetime.utcnow()
            log.reviewed_by = reviewer

    db.commit()
    return JSONResponse({"success": True, "updated": updated})


@router.post("/data-review/batch-confirm-all")
async def batch_confirm_all(request: Request, db: Session = Depends(get_db)):
    """批量确认所有 pending 审核（接受自动标注结果）"""
    from app.models.db_models import QualityLabelReview
    pending = db.query(QualityLabelReview).filter_by(review_status="pending").all()
    for r in pending:
        r.review_status = "confirmed"
        r.manual_label = r.auto_label
        r.reviewed_at = datetime.utcnow()
        r.reviewed_by = "batch_auto"
    db.commit()
    return JSONResponse({"success": True, "confirmed": len(pending)})
