"""
增量训练 & 持续学习路由

策略（人工审核后触发）：
  - 新增标注数据 < AUTO_RETRAIN_THRESHOLD 条 → DQN 在线学习（轻量）
  - 新增标注数据 ≥ AUTO_RETRAIN_THRESHOLD 条 → 全量重训练 MGQ-Former

API 列表：
  POST /retrain/incremental  基于审核标注数据触发增量训练（DQN 或 全量）
  POST /retrain/full         强制全量重训练
  GET  /retrain/status       训练状态 & 待处理数据量
  GET  /retrain/stream       SSE 流式进度（复用训练进度队列）
"""
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
import config as cfg

router = APIRouter()


@router.get("/retrain/status")
async def retrain_status(request: Request, db: Session = Depends(get_db)):
    """返回训练状态与待处理数据量"""
    from app.models.db_models import (
        QualityLabelReview, DataIngestionLog, ModelVersion, QualityChain
    )

    pending_reviews = db.query(QualityLabelReview).filter_by(review_status="pending").count()
    confirmed_not_trained = db.query(QualityLabelReview).filter_by(
        review_status="confirmed", added_to_train=False
    ).count()

    active_model = db.query(ModelVersion).filter_by(is_active=True).first()
    total_train_samples = db.query(QualityChain).count()

    # 未处理的采集日志
    pending_logs = db.query(DataIngestionLog).filter_by(status="pending").count()

    should_retrain = confirmed_not_trained >= cfg.AUTO_RETRAIN_THRESHOLD
    retrain_strategy = "full" if confirmed_not_trained >= cfg.AUTO_RETRAIN_THRESHOLD else "dqn_only"

    return JSONResponse({
        "pending_reviews": pending_reviews,
        "confirmed_not_trained": confirmed_not_trained,
        "pending_ingestion_logs": pending_logs,
        "total_train_samples": total_train_samples,
        "auto_retrain_threshold": cfg.AUTO_RETRAIN_THRESHOLD,
        "should_retrain": should_retrain,
        "retrain_strategy": retrain_strategy,
        "active_model": {
            "version_tag": active_model.version_tag if active_model else None,
            "model_type": active_model.model_type if active_model else None,
            "auc_score": active_model.auc_score if active_model else None,
            "created_at": active_model.created_at.strftime("%Y-%m-%d %H:%M:%S") if active_model else None,
        } if active_model else None,
    })


@router.post("/retrain/incremental")
async def retrain_incremental(request: Request, db: Session = Depends(get_db)):
    """
    基于已审核的标注数据触发增量训练

    自动选择策略：
      - 数据量 < threshold → DQN 批量学习
      - 数据量 ≥ threshold → 全量重训练 MGQ-Former
    """
    from app.models.db_models import QualityLabelReview, QualityChain
    from app.services.predict_service import load_active_model
    import numpy as np

    confirmed = db.query(QualityLabelReview).filter_by(
        review_status="confirmed", added_to_train=False
    ).all()

    if not confirmed:
        return JSONResponse({"success": False, "message": "没有待训练的已确认数据"})

    count = len(confirmed)
    strategy = "full" if count >= cfg.AUTO_RETRAIN_THRESHOLD else "dqn_only"

    if strategy == "dqn_only":
        # DQN 批量反馈学习
        learned = 0
        try:
            from app.services.predict_service import learn_from_feedback
            for review in confirmed:
                if review.manual_label is None:
                    continue
                features_snapshot = {}
                if review.feature_snapshot:
                    feat_list = json.loads(review.feature_snapshot)
                    from app.services.predict_service import FEATURE_NAMES
                    features_snapshot = dict(zip(FEATURE_NAMES, feat_list))
                result = learn_from_feedback(features_snapshot, review.manual_label, db)
                if result.get("success"):
                    review.added_to_train = True
                    learned += 1
            db.commit()
        except Exception as e:
            return JSONResponse({"success": False, "error": f"DQN 学习失败: {e}"})

        return JSONResponse({
            "success": True,
            "strategy": "dqn_only",
            "learned_count": learned,
            "message": f"DQN 在线学习完成，{learned} 条数据已学习",
        })

    else:
        # 全量重训练：将已确认数据写入 QualityChain，然后触发训练
        added_to_chain = _merge_confirmed_to_chain(confirmed, db)

        # 标记为已加入训练集
        for review in confirmed:
            review.added_to_train = True
        db.commit()

        # 异步触发训练
        asyncio.create_task(_async_full_retrain(cfg.RETRAIN_MODEL_TYPE, db))

        return JSONResponse({
            "success": True,
            "strategy": "full_retrain",
            "confirmed_count": count,
            "added_to_chain": added_to_chain,
            "message": f"已将 {added_to_chain} 条确认数据加入训练集，{cfg.RETRAIN_MODEL_TYPE.upper()} 全量重训练已在后台启动",
        })


@router.post("/retrain/full")
async def retrain_full(request: Request, db: Session = Depends(get_db)):
    """强制触发全量重训练（不依赖新标注，使用现有全量数据）"""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    model_type = body.get("model_type", cfg.RETRAIN_MODEL_TYPE) if isinstance(body, dict) else cfg.RETRAIN_MODEL_TYPE

    asyncio.create_task(_async_full_retrain(model_type, db))
    return JSONResponse({
        "success": True,
        "model_type": model_type,
        "message": f"{model_type.upper()} 全量重训练已在后台启动，可通过 /train/stream 监听进度",
    })


@router.get("/retrain/stream")
async def retrain_stream(request: Request, db: Session = Depends(get_db)):
    """SSE 流式训练进度（复用 predict_service 的训练事件流）"""
    from app.services.predict_service import training_event_stream
    return StreamingResponse(
        training_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── 内部辅助函数 ──────────────────────────────────────────────────────────

def _merge_confirmed_to_chain(confirmed_reviews: list, db) -> int:
    """
    将人工确认的标注数据合并到 QualityChain
    （以 review 中的 feature_snapshot 为基础重建）
    """
    from app.models.db_models import QualityChain
    from app.services.predict_service import FEATURE_NAMES

    added = 0
    # 获取当前最大 meter_no，新数据从此接续
    max_no_result = db.query(QualityChain.meter_no).order_by(QualityChain.meter_no.desc()).first()
    next_no = (max_no_result[0] + 1) if max_no_result else 10001

    for review in confirmed_reviews:
        if review.manual_label is None:
            continue

        feat_dict = {}
        if review.feature_snapshot:
            try:
                feat_list = json.loads(review.feature_snapshot)
                feat_dict = dict(zip(FEATURE_NAMES, feat_list))
            except Exception:
                pass

        # 构建 QualityChain 记录
        chain = QualityChain(
            meter_no=next_no,
            quality_label=int(review.manual_label),
            quality_reason=f"采集自生产线（barcode={review.barcode}，人工审核）",
            crystal_ppm=feat_dict.get("crystal_ppm", 9.0),
            ct_ratio_err=feat_dict.get("ct_ratio_err", 0.5),
            clock_accuracy_sd=feat_dict.get("clock_accuracy_sd", 0.018),
            factory_fault=bool(feat_dict.get("factory_fault", 0)),
            initial_error_max=feat_dict.get("initial_error_max", 0.0),
            final_error_max=feat_dict.get("final_error_max", 0.0),
            error_degradation_rate=feat_dict.get("error_degradation_rate", 0.0),
            clock_drift_rate=feat_dict.get("clock_drift_rate", 0.0),
            rs485_total_errors=int(feat_dict.get("rs485_total_errors", 0)),
            is_augmented=False,  # 真实采集数据
        )
        db.add(chain)
        next_no += 1
        added += 1

    db.commit()
    return added


async def _async_full_retrain(model_type: str, db):
    """后台异步全量重训练"""
    from app.services.predict_service import train_model
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: train_model(db, model_type=model_type))
    except Exception as e:
        from app.services.predict_service import _emit
        _emit("error", {"message": f"重训练失败: {e}"})
