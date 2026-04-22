"""
通知中心 + 文件监控管理路由
- GET  /notify/stream        SSE 实时通知流
- GET  /notify/watcher-status  监控服务状态
- POST /notify/scan-now      手动触发立即扫描
- POST /notify/upload        手动上传文件（进入解析流程）
"""
import json
from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()


@router.get("/notify/stream")
async def notify_stream(request: Request):
    """SSE 实时通知流，前端长连接订阅"""
    from app.services.file_watcher_service import notify_event_stream
    return StreamingResponse(
        notify_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/notify/watcher-status")
async def watcher_status(request: Request, db: Session = Depends(get_db)):
    """获取文件监控服务状态"""
    import config as cfg
    from app.models.db_models import DataIngestionLog, QualityLabelReview
    from app.services.file_watcher_service import get_watcher

    watcher = get_watcher()
    pending_reviews = db.query(QualityLabelReview).filter_by(review_status="pending").count()
    recent_logs = db.query(DataIngestionLog).order_by(
        DataIngestionLog.parsed_at.desc()
    ).limit(5).all()

    return JSONResponse({
        "running": watcher._running,
        "watch_dir": str(cfg.WATCH_DIR),
        "interval_sec": cfg.WATCH_INTERVAL_SEC,
        "seen_files": len(watcher._seen_hashes),
        "pending_reviews": pending_reviews,
        "recent_logs": [l.to_dict() for l in recent_logs],
        "auto_retrain_threshold": cfg.AUTO_RETRAIN_THRESHOLD,
    })


@router.post("/notify/scan-now")
async def scan_now(request: Request, db: Session = Depends(get_db)):
    """手动触发立即扫描（无需等待轮询间隔）"""
    from app.services.file_watcher_service import get_watcher
    watcher = get_watcher()
    try:
        summary = await watcher.scan_now()
        return JSONResponse({"success": True, "summary": summary})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.post("/notify/upload")
async def upload_and_parse(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    手动上传文件直接进入解析流程
    支持 xlsx / xls / csv
    """
    import hashlib
    import tempfile
    import shutil
    from pathlib import Path
    from app.services.file_watcher_service import (
        infer_data_type, md5_of_file, parse_and_upsert, auto_label_new_records
    )
    from app.models.db_models import DataIngestionLog

    # 保存到临时文件
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    file_hash = md5_of_file(tmp_path)
    data_type = infer_data_type(file.filename)

    # 检查是否已导入
    existing = db.query(DataIngestionLog).filter_by(file_hash=file_hash).first()
    if existing:
        tmp_path.unlink(missing_ok=True)
        return JSONResponse({
            "success": False,
            "message": f"文件已存在（hash 重复），上次导入于 {existing.parsed_at}",
            "log_id": existing.id,
        })

    try:
        log_entry = DataIngestionLog(
            file_name=file.filename,
            file_hash=file_hash,
            data_type=data_type,
            status="pending",
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        record_count, new_count = parse_and_upsert(tmp_path, data_type, db)
        labeled = auto_label_new_records(new_count, data_type, log_entry.id, db)

        log_entry.record_count = record_count
        log_entry.new_count = new_count
        log_entry.status = "pending"
        db.commit()

        return JSONResponse({
            "success": True,
            "file_name": file.filename,
            "data_type": data_type,
            "record_count": record_count,
            "new_count": new_count,
            "labeled_count": labeled,
            "log_id": log_entry.id,
            "message": f"解析完成：{new_count} 条新记录，{labeled} 条进入审核队列",
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)
