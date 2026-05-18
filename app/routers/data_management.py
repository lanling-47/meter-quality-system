"""Enterprise data management center.

Provides database-backed CRUD pages so operators can enter and maintain
full-lifecycle quality data one record at a time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Boolean, DateTime, Float, Integer, Text, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db
from app.models import db_models as models


router = APIRouter()
templates = Jinja2Templates(directory="templates")


DATASETS = {
    "meter_identity": {
        "name": "电表身份映射",
        "stage": "基础主数据",
        "model": models.MeterIdentity,
        "description": "维护电表编号、生产条码、订单号、批次号之间的映射关系。",
        "search": ["barcode", "normalized_barcode", "order_no", "batch_no", "product_model"],
    },
    "crystal_data": {
        "name": "晶振来料检测",
        "stage": "来料质量",
        "model": models.CrystalData,
        "description": "逐条录入晶振批次抽检数据。",
        "search": ["batch_no", "model"],
    },
    "ct_data": {
        "name": "互感器来料检测",
        "stage": "来料质量",
        "model": models.CTData,
        "description": "逐条录入互感器批次抽检数据。",
        "search": ["batch_no", "model"],
    },
    "factory_test": {
        "name": "出厂检测",
        "stage": "出厂质量",
        "model": models.FactoryTest,
        "description": "维护每块电表的出厂检测与故障标签。",
        "search": ["meter_id", "crystal_batch", "ct_batch", "fault_type"],
    },
    "aging_test": {
        "name": "老化试验",
        "stage": "抽检可靠性",
        "model": models.AgingTest,
        "description": "录入抽检电表在不同老化时间点的退化数据。",
        "search": ["meter_no", "test_date_label", "test_condition", "quality_reason"],
    },
    "production_inspection": {
        "name": "生产工序检测",
        "stage": "生产过程",
        "model": models.ProductionInspection,
        "description": "维护 CCD、耐压、功能、复校等工序检测记录。",
        "search": ["barcode", "order_no", "station_no", "process", "result", "device_no"],
    },
    "aoi_defect": {
        "name": "AOI 缺陷",
        "stage": "生产过程",
        "model": models.AoiDefect,
        "description": "维护贴片 AOI 与波峰焊 AOI 缺陷记录。",
        "search": ["barcode", "order_no", "defect_point", "defect_desc", "process_no", "aoi_type"],
    },
    "repair_record": {
        "name": "维修记录",
        "stage": "维修闭环",
        "model": models.RepairRecord,
        "description": "维护故障现象、原因、位置、工序和维修时间。",
        "search": ["barcode", "order_no", "fault_name", "cause_name", "fault_location", "fault_process"],
    },
    "board_test_detail": {
        "name": "单板测试明细",
        "stage": "生产过程",
        "model": models.BoardTestDetail,
        "description": "维护按测试项目展开的单板检测明细。",
        "search": ["barcode", "order_no", "scheme_name", "test_item", "result", "device_no"],
    },
    "production_process_profile": {
        "name": "生产过程画像",
        "stage": "生产过程",
        "model": models.ProductionProcessProfile,
        "description": "维护 SMT、回流焊、在线测试、波峰焊等批次级工艺参数画像。",
        "search": ["batch_key", "source_file", "note"],
    },
    "reliability_sample_production_record": {
        "name": "可靠性样机生产数据",
        "stage": "抽检可靠性",
        "model": models.ReliabilitySampleProductionRecord,
        "description": "维护 X202201001 可靠性样机在生产修调/检测阶段的多次误差、通讯、起动、日计时等明细记录。",
        "search": ["sample_id", "source_file", "test_item", "operation_result", "operator"],
    },
    "reliability_sample_after_record": {
        "name": "可靠性样机试验后数据",
        "stage": "抽检可靠性",
        "model": models.ReliabilitySampleAfterRecord,
        "description": "维护 X202201001 可靠性样机试验前多次测量与温度循环试验后误差/偏移对比明细。",
        "search": ["sample_id", "source_file", "test_item", "current_point", "power_factor"],
    },
    "reliability_sample_profile": {
        "name": "可靠性样机质量画像",
        "stage": "融合追溯",
        "model": models.ReliabilitySampleProfile,
        "description": "按电子批号聚合生产数据与试验后数据，形成模型使用的批次/样机级可靠性风险特征。",
        "search": ["sample_id", "batch_key", "source", "note"],
    },
    "meter_quality_trace": {
        "name": "单表质量追溯汇总",
        "stage": "融合追溯",
        "model": models.MeterQualityTrace,
        "description": "维护每块电表的全工序汇总风险记录。",
        "search": ["barcode", "order_no"],
    },
    "quality_chain": {
        "name": "质量链路特征",
        "stage": "融合追溯",
        "model": models.QualityChain,
        "description": "维护面向模型训练的单表质量链路特征。",
        "search": ["meter_no", "crystal_batch", "ct_batch", "factory_fault_type", "quality_reason"],
    },
    "batch_quality_profile": {
        "name": "批次抽检质量画像",
        "stage": "融合追溯",
        "model": models.BatchQualityProfile,
        "description": "维护由抽检老化样本聚合形成的批次可靠性画像。",
        "search": ["batch_key", "order_no", "batch_no", "source", "note"],
    },
    "quality_feature_snapshot": {
        "name": "融合特征快照",
        "stage": "建模快照",
        "model": models.QualityFeatureSnapshot,
        "description": "查看和维护模型训练用的24维基础融合快照，训练时再拼接生产工艺画像和可靠性样机画像形成46维单表输入。",
        "search": ["meter_no", "barcode", "order_no", "batch_key", "source", "quality_reason"],
    },
}


READONLY_FIELDS = {"id", "created_at", "updated_at"}


def _dataset_or_404(dataset: str) -> dict[str, Any] | None:
    return DATASETS.get(dataset)


def _column_type(column) -> str:
    col_type = column.type
    if isinstance(col_type, Boolean):
        return "boolean"
    if isinstance(col_type, Integer):
        return "integer"
    if isinstance(col_type, Float):
        return "float"
    if isinstance(col_type, DateTime):
        return "datetime"
    if isinstance(col_type, Text):
        return "text"
    return "string"


def _schema(dataset: str, cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg["model"]
    fields = []
    for column in model.__table__.columns:
        field_type = _column_type(column)
        readonly = column.name in READONLY_FIELDS
        fields.append({
            "name": column.name,
            "label": column.comment or column.name,
            "type": field_type,
            "required": bool(not column.nullable and not column.primary_key and column.default is None),
            "readonly": readonly,
            "primary_key": bool(column.primary_key),
        })
    return {
        "key": dataset,
        "name": cfg["name"],
        "stage": cfg["stage"],
        "description": cfg["description"],
        "fields": fields,
    }


def _parse_value(value: Any, field_type: str):
    if value == "" or value is None:
        return None
    if field_type == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "是", "合格", "通过"}
    if field_type == "integer":
        return int(value)
    if field_type == "float":
        return float(value)
    if field_type == "datetime":
        if isinstance(value, datetime):
            return value
        text = str(value).strip().replace("T", " ")
        if len(text) == 10:
            text += " 00:00:00"
        return datetime.fromisoformat(text)
    return value


def _apply_payload(row, schema: dict[str, Any], payload: dict[str, Any]):
    for field in schema["fields"]:
        if field["readonly"]:
            continue
        name = field["name"]
        if name not in payload:
            continue
        setattr(row, name, _parse_value(payload.get(name), field["type"]))


def _row_to_dict(row) -> dict[str, Any]:
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


@router.get("/data-management", response_class=HTMLResponse)
async def data_management_page(request: Request):
    return templates.TemplateResponse(request, "data_management.html", {
        "page": "data_management",
    })


@router.get("/data-management/api/datasets")
async def list_datasets(db: Session = Depends(get_db)):
    items = []
    for key, cfg in DATASETS.items():
        count = db.query(cfg["model"]).count()
        item = {k: v for k, v in _schema(key, cfg).items() if k != "fields"}
        item["count"] = count
        items.append(item)
    return JSONResponse({"items": items})


@router.get("/data-management/api/{dataset}/schema")
async def dataset_schema(dataset: str):
    cfg = _dataset_or_404(dataset)
    if not cfg:
        return JSONResponse({"success": False, "message": "未知数据表"}, status_code=404)
    return JSONResponse(_schema(dataset, cfg))


@router.get("/data-management/api/{dataset}")
async def list_records(
    dataset: str,
    db: Session = Depends(get_db),
    page: int = 1,
    page_size: int = 20,
    q: str = "",
):
    cfg = _dataset_or_404(dataset)
    if not cfg:
        return JSONResponse({"success": False, "message": "未知数据表"}, status_code=404)

    model = cfg["model"]
    query = db.query(model)
    if q:
        filters = []
        for field in cfg.get("search", []):
            column = getattr(model, field, None)
            if column is not None:
                filters.append(column.cast(Text).contains(q))
        if filters:
            query = query.filter(or_(*filters))

    total = query.count()
    rows = query.order_by(model.id.desc()).offset(max(page - 1, 0) * page_size).limit(page_size).all()
    return JSONResponse(jsonable_encoder({
        "success": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_row_to_dict(row) for row in rows],
    }))


@router.post("/data-management/api/{dataset}")
async def create_record(dataset: str, request: Request, db: Session = Depends(get_db)):
    cfg = _dataset_or_404(dataset)
    if not cfg:
        return JSONResponse({"success": False, "message": "未知数据表"}, status_code=404)
    payload = await request.json()
    schema = _schema(dataset, cfg)
    row = cfg["model"]()
    try:
        _apply_payload(row, schema, payload)
        db.add(row)
        db.commit()
        db.refresh(row)
        return JSONResponse(jsonable_encoder({"success": True, "item": _row_to_dict(row)}))
    except (ValueError, TypeError) as exc:
        db.rollback()
        return JSONResponse({"success": False, "message": f"字段格式错误：{exc}"}, status_code=400)
    except IntegrityError as exc:
        db.rollback()
        return JSONResponse({"success": False, "message": f"数据唯一性或必填约束不满足：{exc.orig}"}, status_code=400)


@router.put("/data-management/api/{dataset}/{record_id}")
async def update_record(dataset: str, record_id: int, request: Request, db: Session = Depends(get_db)):
    cfg = _dataset_or_404(dataset)
    if not cfg:
        return JSONResponse({"success": False, "message": "未知数据表"}, status_code=404)
    row = db.query(cfg["model"]).filter_by(id=record_id).first()
    if row is None:
        return JSONResponse({"success": False, "message": "记录不存在"}, status_code=404)
    payload = await request.json()
    schema = _schema(dataset, cfg)
    try:
        _apply_payload(row, schema, payload)
        db.commit()
        db.refresh(row)
        return JSONResponse(jsonable_encoder({"success": True, "item": _row_to_dict(row)}))
    except (ValueError, TypeError) as exc:
        db.rollback()
        return JSONResponse({"success": False, "message": f"字段格式错误：{exc}"}, status_code=400)
    except IntegrityError as exc:
        db.rollback()
        return JSONResponse({"success": False, "message": f"数据唯一性或必填约束不满足：{exc.orig}"}, status_code=400)


@router.delete("/data-management/api/{dataset}/{record_id}")
async def delete_record(dataset: str, record_id: int, db: Session = Depends(get_db)):
    cfg = _dataset_or_404(dataset)
    if not cfg:
        return JSONResponse({"success": False, "message": "未知数据表"}, status_code=404)
    row = db.query(cfg["model"]).filter_by(id=record_id).first()
    if row is None:
        return JSONResponse({"success": False, "message": "记录不存在"}, status_code=404)
    db.delete(row)
    db.commit()
    return JSONResponse({"success": True})


@router.post("/data-management/api/rebuild-linkage")
async def rebuild_linkage(db: Session = Depends(get_db)):
    from app.services.data_linkage_service import build_quality_feature_snapshots

    try:
        stats = build_quality_feature_snapshots(db, include_synthetic_demo=False)
        return JSONResponse({"success": True, "stats": stats})
    except Exception as exc:
        db.rollback()
        return JSONResponse({"success": False, "message": str(exc)}, status_code=500)
