"""数据导入 & 模型管理路由"""
import asyncio
import json
from pathlib import Path
from fastapi import APIRouter, Request, Depends, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _get_db_stats(db):
    from app.models.db_models import (
        AgingTest,
        CTData,
        CrystalData,
        FactoryTest,
        ProductionProcessProfile,
        QualityChain,
        ReliabilitySampleAfterRecord,
        ReliabilitySampleProductionRecord,
        ReliabilitySampleProfile,
    )
    return {
        "crystal": db.query(CrystalData).count(),
        "ct": db.query(CTData).count(),
        "factory": db.query(FactoryTest).count(),
        "process_profile": db.query(ProductionProcessProfile).count(),
        "aging": db.query(AgingTest).count(),
        "chain": db.query(QualityChain).count(),
        "reliability_production": db.query(ReliabilitySampleProductionRecord).count(),
        "reliability_after": db.query(ReliabilitySampleAfterRecord).count(),
        "reliability_profile": db.query(ReliabilitySampleProfile).count(),
    }


@router.get("/import", response_class=HTMLResponse)
async def import_page(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import ModelVersion
    import config as cfg
    stats = _get_db_stats(db)
    model_versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    mv_list = [{
        "version_tag": m.version_tag,
        "model_type": m.model_type,
        "auc_score": m.auc_score,
        "f1_score": m.f1_score,
        "n_samples": m.n_samples,
        "is_active": m.is_active,
        "created_at": str(m.created_at)[:16] if m.created_at else "--",
    } for m in model_versions]

    return templates.TemplateResponse(request, "data_import.html", {
        "stats": stats,
        "model_versions": mv_list,
        "watch_dir": str(cfg.WATCH_DIR),
        "data_dir": str(cfg.DATA_DIR),
        "page": "import",
    })


# ── 数据导入 API ──────────────────────────────────────────────────────

@router.post("/import/excel")
async def import_excel(request: Request, db: Session = Depends(get_db)):
    """从 data/raw/ 目录批量导入 Excel 数据"""
    from app.services.etl_service import import_all_data
    try:
        stats = import_all_data(db)
        return JSONResponse({"success": True, "stats": stats})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})


@router.post("/import/reliability-sample")
async def import_reliability_sample(request: Request, db: Session = Depends(get_db)):
    """导入 X202201001 可靠性样机生产数据和试验后数据。"""
    from app.services.reliability_sample_service import import_reliability_sample_data

    try:
        stats = import_reliability_sample_data(db)
        return JSONResponse({"success": not stats.get("errors"), "stats": stats})
    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "message": str(e), "trace": traceback.format_exc()})


@router.post("/import/factory")
async def import_factory_manual(request: Request, db: Session = Depends(get_db)):
    """手动录入出厂检测记录"""
    from app.models.db_models import FactoryTest
    body = await request.json()
    meter_id = body.get("meter_id")
    if not meter_id:
        return JSONResponse({"success": False, "message": "meter_id 不能为空"})
    existing = db.query(FactoryTest).filter_by(meter_id=meter_id).first()
    if existing:
        for k, v in body.items():
            if hasattr(existing, k) and v is not None:
                setattr(existing, k, v)
    else:
        ft = FactoryTest(**{k: v for k, v in body.items() if hasattr(FactoryTest, k)})
        db.add(ft)
    db.commit()
    return JSONResponse({"success": True, "message": f"出厂记录 {meter_id} 已保存"})


# ── 训练 API ─────────────────────────────────────────────────────────

@router.post("/train")
async def train_model_api(request: Request, db: Session = Depends(get_db)):
    """触发模型训练（同步返回结果）"""
    from app.services.predict_service import train_model
    try:
        result = train_model(db, model_type="mgqformer")
        return JSONResponse({"success": True, "result": {
            k: v for k, v in result.items()
            if k not in ("roc_data", "train_loss_curve", "cv_auc_curve", "feature_importance")
        }})
    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "trace": traceback.format_exc()})


@router.get("/train/stream")
async def train_stream(request: Request, model_type: str = "mgqformer", db: Session = Depends(get_db)):
    """SSE 端点：流式推送训练进度，支持 model_type=mgqformer|tabnet|lightgbm"""
    from fastapi.responses import StreamingResponse
    from app.services.predict_service import training_event_stream, train_model

    async def run_train():
        """在后台线程中运行训练"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: train_model(db, model_type=model_type))

    # 启动训练任务（不等待完成）
    asyncio.create_task(run_train())

    # 返回 SSE 流
    return StreamingResponse(
        training_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ── 模型管理 API ──────────────────────────────────────────────────────

@router.get("/models")
async def list_models(db: Session = Depends(get_db)):
    """获取模型版本列表"""
    from app.models.db_models import ModelVersion
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return JSONResponse([{
        "version_tag": m.version_tag,
        "model_type": m.model_type,
        "auc_score": round(m.auc_score, 3) if m.auc_score else None,
        "f1_score": round(m.f1_score, 3) if m.f1_score else None,
        "n_samples": m.n_samples,
        "is_active": m.is_active,
        "created_at": str(m.created_at)[:16] if m.created_at else "--",
    } for m in versions])


@router.get("/models/mgqformer/architecture")
async def mgqformer_architecture():
    """Return MGQ-Former architecture metadata for UI and thesis export."""
    from app.services.deep_fusion_service import get_mgqformer_architecture
    return JSONResponse(get_mgqformer_architecture())


@router.post("/models/activate/{version_tag}")
async def activate_model(version_tag: str, db: Session = Depends(get_db)):
    """激活指定模型版本"""
    from app.services.predict_service import activate_model_version
    result = activate_model_version(version_tag, db)
    return JSONResponse(result)


@router.post("/models/upload")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form(default="外部上传"),
    db: Session = Depends(get_db)
):
    """上传外部 pkl 模型文件并激活"""
    from app.services.predict_service import upload_and_activate_model
    content = await file.read()
    result = upload_and_activate_model(content, model_type, db)
    return JSONResponse(result)


@router.post("/import/crystal")
async def import_crystal_manual(request: Request, db: Session = Depends(get_db)):
    """手动录入晶振检测记录"""
    from app.models.db_models import CrystalData
    body = await request.json()
    record = CrystalData(
        batch_no=body.get("batch_no", ""),
        sample_no=body.get("sample_no", 1),
        model="HC-49SMD",
        center_freq_hz=body.get("center_freq_hz"),
        freq_error_ppm=body.get("freq_error_ppm"),
        load_cap_pf=body.get("load_cap_pf"),
        is_qualified=body.get("is_qualified", True),
    )
    db.add(record)
    try:
        db.commit()
        return JSONResponse({"success": True, "message": f"晶振记录已保存（批次{body.get('batch_no')}，样品{body.get('sample_no')}）"})
    except Exception as e:
        db.rollback()
        return JSONResponse({"success": False, "message": f"保存失败：{e}"})


@router.post("/import/aging")
async def import_aging_manual(request: Request, db: Session = Depends(get_db)):
    """手动录入老化试验记录"""
    from app.models.db_models import AgingTest
    body = await request.json()
    meter_no = body.get("meter_no")
    date_label = body.get("test_date_label")
    if not meter_no or not date_label:
        return JSONResponse({"success": False, "message": "meter_no 和 test_date_label 不能为空"})

    from config import AGING_TIME_MAP
    test_hours = body.get("test_hours") or AGING_TIME_MAP.get(date_label, 0)
    error_max = body.get("error_max")
    clock_offset = body.get("clock_offset_s", 0)
    rs485_errors = body.get("rs485_errors", 0)

    quality_label = 0
    quality_reason = ""
    if error_max and error_max > 0.5:
        quality_label = 1
        quality_reason += "误差超限;"
    if abs(clock_offset) > 10:
        quality_label = 1
        quality_reason += "时钟超限;"

    existing = db.query(AgingTest).filter_by(meter_no=meter_no, test_date_label=date_label).first()
    if existing:
        existing.test_hours = test_hours
        existing.error_max = error_max
        existing.clock_offset_s = clock_offset
        existing.rs485_errors = rs485_errors
        existing.quality_label = quality_label
        existing.quality_reason = quality_reason
    else:
        record = AgingTest(
            meter_no=meter_no,
            test_date_label=date_label,
            test_hours=test_hours,
            test_condition="手动录入",
            basic_errors_json="{}",
            error_max=error_max,
            clock_offset_s=clock_offset,
            rs485_errors=rs485_errors,
            quality_label=quality_label,
            quality_reason=quality_reason,
        )
        db.add(record)
    try:
        db.commit()
        return JSONResponse({"success": True, "message": f"老化试验记录已保存（电表#{meter_no}，{date_label}）"})
    except Exception as e:
        db.rollback()
        return JSONResponse({"success": False, "message": f"保存失败：{e}"})


@router.post("/import/upload_excel")
async def upload_excel_file(
    request: Request,
    file: UploadFile = File(...),
    data_type: str = Form(default="factory"),
    db: Session = Depends(get_db)
):
    """上传 Excel 文件解析导入"""
    import tempfile, os
    from app.services.etl_service import (
        parse_crystal_data,
        parse_ct_data,
        parse_factory_test,
        parse_factory_process_profile,
        parse_aging_test,
        upsert_production_process_profiles,
    )
    from app.models.db_models import CrystalData, CTData, FactoryTest, AgingTest

    content = await file.read()
    suffix = Path(file.filename).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        count = 0
        if data_type == "crystal":
            records = parse_crystal_data(tmp_path)
            for r in records:
                db.add(CrystalData(**r))
            count = len(records)
        elif data_type == "factory":
            records = parse_factory_test(tmp_path)
            for r in records:
                existing = db.query(FactoryTest).filter_by(meter_id=r["meter_id"]).first()
                if existing:
                    for k, v in r.items():
                        if k != "meter_id" and hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    db.add(FactoryTest(**r))
            count = len(records)
            count += upsert_production_process_profiles(db, parse_factory_process_profile(tmp_path))
        elif data_type == "aging":
            records = parse_aging_test(tmp_path)
            for r in records:
                db.add(AgingTest(**r))
            count = len(records)
        db.commit()
        return JSONResponse({"success": True, "message": f"上传解析完成，共导入 {count} 条记录"})
    except Exception as e:
        db.rollback()
        return JSONResponse({"success": False, "message": f"解析失败：{e}"})
    finally:
        tmp_path.unlink(missing_ok=True)


@router.delete("/models/{version_tag}")
async def delete_model(version_tag: str, db: Session = Depends(get_db)):
    """删除指定模型版本（不可删除激活中的模型）"""
    from app.models.db_models import ModelVersion
    mv = db.query(ModelVersion).filter_by(version_tag=version_tag).first()
    if not mv:
        return JSONResponse({"success": False, "message": "版本不存在"})
    if mv.is_active:
        return JSONResponse({"success": False, "message": "不能删除当前激活模型，请先激活其他版本"})
    if mv.model_path and Path(mv.model_path).exists():
        Path(mv.model_path).unlink(missing_ok=True)
    db.delete(mv)
    db.commit()
    return JSONResponse({"success": True, "message": f"已删除 {version_tag}"})


# ── 数据增强 API ──────────────────────────────────────────────────────

@router.post("/import/augment")
async def augment_data(request: Request, db: Session = Depends(get_db)):
    """触发数据增强：基于真实分布生成合成样本（幂等操作）"""
    from app.services.data_augment import apply_augmented_data
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    n = int(body.get("n", 200))
    seed = int(body.get("seed", 42))
    try:
        result = apply_augmented_data(db, n=n, seed=seed)
        return JSONResponse({"success": True, "stats": result})
    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "trace": traceback.format_exc()})


@router.get("/import/augment/stats")
async def augment_stats(db: Session = Depends(get_db)):
    """获取当前数据集的真实/增强样本统计"""
    from app.models.db_models import QualityChain as QC
    total = db.query(QC).count()
    real = db.query(QC).filter(QC.is_augmented == False).count()
    aug = db.query(QC).filter(QC.is_augmented == True).count()
    high_risk = db.query(QC).filter(QC.quality_label == 1).count()
    high_risk_real = db.query(QC).filter(QC.quality_label == 1, QC.is_augmented == False).count()
    high_risk_aug = db.query(QC).filter(QC.quality_label == 1, QC.is_augmented == True).count()
    return JSONResponse({
        "total": total,
        "real_count": real,
        "aug_count": aug,
        "high_risk_total": high_risk,
        "high_risk_real": high_risk_real,
        "high_risk_aug": high_risk_aug,
        "normal_total": total - high_risk,
        "has_augmented": aug > 0,
    })
