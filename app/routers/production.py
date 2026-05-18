"""
生产过程数据相关路由：
  - /production          生产追溯页
  - /production/import   导入生产数据（POST）
  - /production/trace    全工序追溯查询（分页/按条码）
  - /production/trace/{barcode}  单表全链路详情
  - /production/repair-analysis  维修闭环分析页
  - /production/repair-stats     维修统计 JSON API
  - /production/propagation      缺陷传播路径 JSON API
  - /production/overview-stats   生产概况 KPI JSON
"""
from fastapi import APIRouter, Request, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# ── 页面路由 ─────────────────────────────────────────────────────────────

@router.get("/production", response_class=HTMLResponse)
async def production_page(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import MeterQualityTrace, ProductionInspection
    total_trace = db.query(MeterQualityTrace).count()
    total_insp = db.query(ProductionInspection).count()
    return templates.TemplateResponse(request, "production_trace.html", {
        "page": "production",
        "total_trace": total_trace,
        "total_insp": total_insp,
    })


@router.get("/repair-analysis", response_class=HTMLResponse)
async def repair_analysis_page(request: Request, db: Session = Depends(get_db)):
    from app.models.db_models import RepairRecord
    total_repairs = db.query(RepairRecord).count()
    return templates.TemplateResponse(request, "repair_analysis.html", {
        "page": "repair",
        "total_repairs": total_repairs,
    })


# ── 数据导入 ──────────────────────────────────────────────────────────────

@router.post("/production/import")
async def import_production(request: Request, db: Session = Depends(get_db)):
    """导入电表生产数据文件夹中的所有数据"""
    from app.services.production_etl_service import import_production_data
    try:
        stats = import_production_data(db)
        return JSONResponse({"success": True, "stats": stats})
    except Exception as e:
        import traceback
        return JSONResponse({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        })


# ── 概况 KPI ─────────────────────────────────────────────────────────────

@router.get("/production/overview-stats")
async def overview_stats(db: Session = Depends(get_db)):
    """生产概况 KPI 数据"""
    from app.models.db_models import (
        BatchQualityProfile, MeterQualityTrace, ProductionInspection, AoiDefect,
        RepairRecord, BoardTestDetail
    )
    total_trace = db.query(MeterQualityTrace).count()
    total_insp = db.query(ProductionInspection).count()
    aoi_total = db.query(AoiDefect).count()
    repair_total = db.query(RepairRecord).count()
    board_fail_total = db.query(BoardTestDetail).count()

    if total_trace > 0:
        from sqlalchemy import func, case
        from app.models.db_models import MeterQualityTrace as MQT
        pass_count = db.query(MQT).filter(MQT.final_pass == True).count()
        high_risk_count = db.query(MQT).filter(MQT.quality_risk_score >= 0.5).count()
        repaired_count = db.query(MQT).filter(MQT.repaired == True).count()
        aoi_defect_count = db.query(MQT).filter(
            (MQT.smt_aoi_defect == True) | (MQT.wave_aoi_defect == True)
        ).count()
        # 各工序不合格率
        from app.models.db_models import ProductionInspection as PI
        from sqlalchemy import func as sqlfunc
        process_stats = {}
        for proc in ['CCD', '功能测试', '耐压', '复校']:
            total_p = db.query(PI).filter(PI.process == proc).count()
            fail_p = db.query(PI).filter(PI.process == proc, PI.result == '不合格').count()
            process_stats[proc] = {
                'total': total_p,
                'fail': fail_p,
                'fail_rate': round(fail_p / total_p * 100, 2) if total_p > 0 else 0,
            }
    else:
        pass_count = high_risk_count = repaired_count = aoi_defect_count = 0
        process_stats = {}

    return JSONResponse({
        'total_trace': total_trace,
        'total_insp': total_insp,
        'pass_count': pass_count,
        'pass_rate': round(pass_count / total_trace * 100, 1) if total_trace > 0 else 0,
        'high_risk_count': high_risk_count,
        'repaired_count': repaired_count,
        'aoi_defect_count': aoi_defect_count,
        'aoi_total': aoi_total,
        'repair_total': repair_total,
        'board_fail_total': board_fail_total,
        'process_stats': process_stats,
    })


# ── 追溯列表 ─────────────────────────────────────────────────────────────

@router.get("/production/batch-profiles")
async def batch_profiles(db: Session = Depends(get_db)):
    """List batch-level quality profiles built from sampled aging tests."""
    from app.models.db_models import BatchQualityProfile
    profiles = db.query(BatchQualityProfile).order_by(
        BatchQualityProfile.high_risk_rate.desc(),
        BatchQualityProfile.updated_at.desc(),
    ).all()
    return JSONResponse(jsonable_encoder({
        "total": len(profiles),
        "items": [p.to_dict() for p in profiles],
    }))


@router.get("/production/linkage/status")
async def linkage_status(db: Session = Depends(get_db)):
    """Show how many rows are linked at barcode, batch, and sampled-aging levels."""
    from app.models.db_models import BatchQualityProfile, MeterIdentity, QualityFeatureSnapshot
    source_counts = {}
    for row in db.query(MeterIdentity.source).all():
        source = row[0] or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
    snapshot_counts = {}
    for row in db.query(QualityFeatureSnapshot.source).all():
        source = row[0] or "unknown"
        snapshot_counts[source] = snapshot_counts.get(source, 0) + 1
    return JSONResponse({
        "identity_total": db.query(MeterIdentity).count(),
        "identity_by_source": source_counts,
        "snapshot_total": db.query(QualityFeatureSnapshot).count(),
        "snapshot_by_source": snapshot_counts,
        "batch_profile_total": db.query(BatchQualityProfile).count(),
    })


@router.post("/production/linkage/build")
async def build_quality_linkage(db: Session = Depends(get_db), include_demo: bool = False):
    """Rebuild meter identity links and fused feature snapshots."""
    from app.services.data_linkage_service import build_quality_feature_snapshots
    try:
        stats = build_quality_feature_snapshots(db, include_synthetic_demo=include_demo)
        return JSONResponse({"success": True, "stats": stats})
    except Exception as e:
        db.rollback()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.get("/production/trace")
async def trace_list(
    db: Session = Depends(get_db),
    page: int = 1,
    page_size: int = 50,
    barcode: str = "",
    order_no: str = "",
    result_filter: str = "",   # "pass" | "fail" | "high_risk"
):
    """分页获取全工序追溯列表"""
    from app.models.db_models import MeterQualityTrace as MQT
    query = db.query(MQT)
    if barcode:
        query = query.filter(MQT.barcode.contains(barcode))
    if order_no:
        query = query.filter(MQT.order_no.contains(order_no))
    if result_filter == 'pass':
        query = query.filter(MQT.final_pass == True)
    elif result_filter == 'fail':
        query = query.filter(MQT.final_pass == False)
    elif result_filter == 'high_risk':
        query = query.filter(MQT.quality_risk_score >= 0.5)

    total = query.count()
    records = query.order_by(MQT.quality_risk_score.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    return JSONResponse({
        'total': total,
        'page': page,
        'page_size': page_size,
        'items': [r.to_dict() for r in records],
    })


# ── 单表详情 ─────────────────────────────────────────────────────────────

@router.get("/production/trace/{barcode}")
async def trace_detail(barcode: str, db: Session = Depends(get_db)):
    """获取单块电表全链路追溯详情"""
    from app.models.db_models import (
        BatchQualityProfile, MeterQualityTrace, ProductionInspection, AoiDefect,
        RepairRecord, BoardTestDetail
    )
    import json as _json

    trace = db.query(MeterQualityTrace).filter_by(barcode=barcode).first()
    batch_profile = None
    if trace and trace.order_no:
        batch_profile = db.query(BatchQualityProfile).filter_by(batch_key=trace.order_no).first()
        if batch_profile is None:
            batch_profile = db.query(BatchQualityProfile).filter_by(order_no=trace.order_no).first()

    # 工序检测明细
    inspections = db.query(ProductionInspection).filter_by(barcode=barcode).order_by(
        ProductionInspection.process, ProductionInspection.test_count
    ).all()

    # AOI缺陷
    aoi = db.query(AoiDefect).filter_by(barcode=barcode).all()

    # 维修记录
    repairs = db.query(RepairRecord).filter_by(barcode=barcode).all()

    # 单板测试不合格
    board_fails = db.query(BoardTestDetail).filter_by(barcode=barcode).all()

    def _dt(v):
        return str(v)[:19] if v else None

    return JSONResponse(jsonable_encoder({
        'trace': trace.to_dict() if trace else None,
        'batch_profile': batch_profile.to_dict() if batch_profile else None,
        'inspections': [{
            'process': r.process,
            'test_count': r.test_count,
            'result': r.result,
            'test_time': _dt(r.test_time),
            'device_no': r.device_no,
            'remark': r.remark,
            'leak_current': r.leak_current,
        } for r in inspections],
        'aoi_defects': [{
            'aoi_type': r.aoi_type,
            'defect_point': r.defect_point,
            'defect_desc': r.defect_desc,
            'quality_risk': r.quality_risk,
            'detect_time': _dt(r.detect_time),
        } for r in aoi],
        'repairs': [{
            'fault_name': r.fault_name,
            'cause_name': r.cause_name,
            'fault_location': r.fault_location,
            'fault_process': r.fault_process,
            'fault_time': _dt(r.fault_time),
            'repair_time': _dt(r.repair_time),
        } for r in repairs],
        'board_test_fails': [{
            'test_item': r.test_item,
            'result': r.result,
            'test_data': r.test_data,
            'test_time': _dt(r.test_time),
        } for r in board_fails],
    }))


# ── 维修分析 ─────────────────────────────────────────────────────────────

@router.get("/production/repair-stats")
async def repair_stats(db: Session = Depends(get_db)):
    """维修闭环统计分析"""
    from app.services.production_etl_service import get_repair_analysis
    try:
        result = get_repair_analysis(db)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({'error': str(e)})


@router.get("/production/propagation")
async def defect_propagation(db: Session = Depends(get_db)):
    """缺陷跨工序传播路径分析"""
    from app.services.production_etl_service import get_process_defect_propagation
    try:
        result = get_process_defect_propagation(db)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({'error': str(e)})
