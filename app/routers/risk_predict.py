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
    """批量重新预测所有电表（融合生产过程特征）"""
    from app.services.predict_service import predict_single
    from app.models.db_models import QualityChain
    from app.services.data_linkage_service import (
        build_quality_feature_snapshots,
        features_from_snapshot,
        snapshot_for_meter,
    )
    chains = db.query(QualityChain).all()

    linkage_stats = build_quality_feature_snapshots(db, include_synthetic_demo=False)
    results = []
    for c in chains:
        snapshot = snapshot_for_meter(db, c.meter_no)
        if snapshot is not None:
            features = features_from_snapshot(snapshot)
            features["batch_key"] = snapshot.batch_key
            features["order_no"] = snapshot.order_no
        else:
            features = {
                "crystal_ppm": c.crystal_ppm or 9.0,
                "ct_ratio_err": c.ct_ratio_err or 0.5,
                "clock_accuracy_sd": c.clock_accuracy_sd or 0.03,
                "factory_fault": c.factory_fault,
                "initial_error_max": c.initial_error_max or 0.0,
                "final_error_max": c.final_error_max or 0.0,
                "error_degradation_rate": c.error_degradation_rate or 0.0,
                "clock_drift_rate": c.clock_drift_rate or 0.0,
                "clock_daily_error_max": c.clock_daily_error_max,
                "error_variability_max": c.error_variability_max,
                "error_consistency_max": c.error_consistency_max,
                "energy_balance_ratio": c.energy_balance_ratio,
                "energy_peak_ratio": c.energy_peak_ratio,
                "rs485_total_errors": c.rs485_total_errors or 0,
            }
        r = predict_single(features, db)
        c.predicted_risk = r.get("risk_prob", 0)
        c.predicted_label = r.get("risk_label", 0)
        results.append({
            "meter_no": c.meter_no,
            "barcode": snapshot.barcode if snapshot else None,
            "batch_key": snapshot.batch_key if snapshot else None,
            "source": snapshot.source if snapshot else "legacy",
            "confidence": snapshot.confidence if snapshot else 0.0,
            **r,
        })
    db.commit()
    return JSONResponse({"success": True, "linkage": linkage_stats, "results": results})


# ─── DQN 在线学习 API ──────────────────────────────────────────

@router.get("/drl/stats")
async def drl_stats(request: Request, db: Session = Depends(get_db)):
    """获取 DQN 智能体统计信息"""
    from app.services.predict_service import get_drl_stats
    return JSONResponse(get_drl_stats(db))


@router.post("/drl/feedback")
async def drl_feedback(request: Request, db: Session = Depends(get_db)):
    """
    DQN 反馈学习接口
    
    当电表实际质量被确认后，调用此接口让 DQN 智能体学习
    """
    from app.services.predict_service import learn_from_feedback
    body = await request.json()
    meter_no = body.get("meter_no")
    true_label = body.get("true_label")  # 1=高风险, 0=正常
    
    if meter_no:
        from app.models.db_models import QualityChain
        chain = db.query(QualityChain).filter_by(meter_no=meter_no).first()
        if chain:
            features = {
                "crystal_ppm": chain.crystal_ppm or 9.0,
                "ct_ratio_err": chain.ct_ratio_err or 0.5,
                "clock_accuracy_sd": chain.clock_accuracy_sd or 0.03,
                "factory_fault": chain.factory_fault,
                "initial_error_max": chain.initial_error_max or 0.0,
                "final_error_max": chain.final_error_max or 0.0,
                "error_degradation_rate": chain.error_degradation_rate or 0.0,
                "clock_drift_rate": chain.clock_drift_rate or 0.0,
                "clock_daily_error_max": chain.clock_daily_error_max,
                "error_variability_max": chain.error_variability_max,
                "error_consistency_max": chain.error_consistency_max,
                "energy_balance_ratio": chain.energy_balance_ratio,
                "energy_peak_ratio": chain.energy_peak_ratio,
                "rs485_total_errors": chain.rs485_total_errors or 0,
                # 生产过程特征（默认0）
                "prod_ccd_retry": 0.0,
                "prod_func_retry": 0.0,
                "prod_reverify_retry": 0.0,
                "prod_total_retry": 0.0,
                "prod_smt_aoi_defect": 0.0,
                "prod_wave_aoi_defect": 0.0,
                "prod_repaired": 0.0,
                "prod_repair_count": 0.0,
                "prod_leak_current_max": 0.0,
                "prod_quality_risk_score": 0.0,
            }
            result = learn_from_feedback(features, true_label, db)
            # 同时更新数据库标签
            if result.get("success"):
                chain.quality_label = true_label
                db.commit()
            return JSONResponse(result)
    
    return JSONResponse({"success": False, "message": "电表不存在"})


@router.post("/drl/reset")
async def drl_reset(request: Request, db: Session = Depends(get_db)):
    """重置 DQN 智能体"""
    from app.services.predict_service import reset_drl_agent
    return JSONResponse(reset_drl_agent())


@router.get("/drl/history")
async def drl_history(request: Request, db: Session = Depends(get_db)):
    """获取 DQN 学习历史"""
    from app.services.drl_agent import get_agent
    
    agent = get_agent()
    stats = agent.get_agent_stats()
    
    return JSONResponse({
        "total_steps": stats["total_steps"],
        "epsilon": round(stats["epsilon"], 4),
        "buffer_size": stats["buffer_size"],
        "avg_reward": round(stats["avg_reward"], 2),
        "reward_std": round(stats["reward_std"], 2),
    })
