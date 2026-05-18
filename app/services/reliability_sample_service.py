"""ETL for X202201001 reliability sample production and after-test workbooks."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import base64
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import openpyxl
from sqlalchemy.orm import Session

from app.models.db_models import (
    ReliabilitySampleAfterRecord,
    ReliabilitySampleProductionRecord,
    ReliabilitySampleProfile,
)


DEFAULT_PRODUCTION_XLS = Path(
    r"C:\Users\jay\xwechat_files\wxid_8gc78r122mz322_f6a4\msg\file\2025-11\3、生产数据-X202201001可靠性样机误差数据.xls"
)
DEFAULT_AFTER_XLS = Path(
    r"C:\Users\jay\xwechat_files\wxid_8gc78r122mz322_f6a4\msg\file\2025-11\4、试验后数据-X202201001可靠性样机误差数据分析.xls"
)

RELIABILITY_FEATURE_NAMES = [
    "rel_prod_error_abs_max",
    "rel_prod_error_abs_avg",
    "rel_prod_error_abs_std",
    "rel_prod_bad_rate",
    "rel_prod_test_date_count",
    "rel_after_error_abs_max",
    "rel_after_error_abs_avg",
    "rel_after_deviation_abs_max",
    "rel_after_deviation_abs_avg",
    "rel_timing_after_abs",
    "rel_risk_score",
    "rel_sample_count",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", "").strip()
    return re.sub(r"\s+", "", text)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        num = float(str(value).strip())
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _safe_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            pass
    return None


def _xlsx_path(path: Path) -> Path:
    """Return an xlsx path, converting legacy xls via local Excel COM if needed."""
    if path.suffix.lower() == ".xlsx":
        return path

    tmp_dir = Path(tempfile.mkdtemp(prefix="meter_rel_xls_"))
    copied_xls = tmp_dir / "input.xls"
    converted = tmp_dir / "input.xlsx"
    shutil.copy2(path, copied_xls)

    try:
        import win32com.client  # type: ignore

        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        workbook = excel.Workbooks.Open(str(copied_xls))
        workbook.SaveAs(str(converted), 51)
        workbook.Close(False)
        excel.Quit()
        return converted
    except Exception:
        # Fallback to late-bound COM available in pywin32-free environments.
        import subprocess

        powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if powershell is None:
            system_root = os.environ.get("SystemRoot", r"C:\Windows")
            candidate = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
            if candidate.exists():
                powershell = str(candidate)
        if powershell is None:
            raise RuntimeError("未找到 powershell.exe，无法将旧版 .xls 转换为 .xlsx")

        def ps_quote(p: Path) -> str:
            return "'" + str(p).replace("'", "''") + "'"

        ps = (
            "$ProgressPreference='SilentlyContinue'; "
            "$ErrorActionPreference='Stop'; "
            "$excel = New-Object -ComObject Excel.Application; "
            "$excel.Visible = $false; $excel.DisplayAlerts = $false; "
            "$wb = $null; "
            "try { "
            f"$wb = $excel.Workbooks.Open({ps_quote(copied_xls)}); "
            f"$wb.SaveAs({ps_quote(converted)}, 51); "
            "$wb.Close($false); "
            "} finally { "
            "if ($wb -ne $null) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wb) }; "
            "$excel.Quit(); "
            "[void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel); "
            "[GC]::Collect(); [GC]::WaitForPendingFinalizers(); "
            "}"
        )
        encoded = base64.b64encode(ps.encode("utf-16le")).decode("ascii")
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
            check=True,
        )
        return converted


def _cleanup_converted_path(original: Path, converted: Path) -> None:
    if original.suffix.lower() == ".xlsx":
        return
    parent = converted.parent
    if parent.name.startswith("meter_rel_xls_"):
        shutil.rmtree(parent, ignore_errors=True)


def parse_production_workbook(path: Path) -> list[dict]:
    xlsx = _xlsx_path(path)
    try:
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        next(rows, None)  # requirement row
        next(rows, None)  # header row

        current_sample = ""
        records: list[dict] = []
        for row in rows:
            vals = list(row) + [None] * 13
            if vals[0] not in (None, ""):
                current_sample = str(vals[0]).strip()
            if not current_sample:
                continue

            test_item = _clean_text(vals[2])
            operation_result = str(vals[11]).strip() if vals[11] is not None else ""
            if not test_item and not operation_result:
                continue

            raw = {
                "station": vals[1],
                "test_item": vals[2],
                "status": vals[3],
                "phase_sequence": vals[4],
                "phase_element": vals[5],
                "current_point": vals[6],
                "power_factor": vals[7],
                "error_value": vals[8],
                "operator": vals[9],
                "operation_time": vals[10],
                "operation_result": vals[11],
                "actual_value": vals[12],
            }
            records.append({
                "sample_id": current_sample,
                "source_file": path.name,
                "test_item": test_item,
                "status": str(vals[3]).strip() if vals[3] is not None else None,
                "phase_sequence": str(vals[4]).strip() if vals[4] is not None else None,
                "phase_element": str(vals[5]).strip() if vals[5] is not None else None,
                "current_point": str(vals[6]).strip() if vals[6] is not None else None,
                "power_factor": str(vals[7]).strip() if vals[7] is not None else None,
                "error_value": _safe_float(vals[8]),
                "actual_value": str(vals[12]).strip() if vals[12] is not None else None,
                "operator": str(vals[9]).strip() if vals[9] is not None else None,
                "operation_time": _safe_dt(vals[10]),
                "operation_result": operation_result or None,
                "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
            })
        wb.close()
        return records
    finally:
        _cleanup_converted_path(path, xlsx)


def parse_after_workbook(path: Path) -> list[dict]:
    xlsx = _xlsx_path(path)
    try:
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        records: list[dict] = []
        current_sample = ""
        current_section = ""

        for row in ws.iter_rows(values_only=True):
            vals = list(row) + [None] * 16
            first = vals[0]
            if first not in (None, ""):
                text = str(first).strip()
                if re.fullmatch(r"\d{8,}", text):
                    current_sample = text
                    current_section = "data"
                elif text.startswith("电子批号"):
                    current_section = "header"

            if not current_sample:
                continue

            test_item = _clean_text(vals[1])
            if test_item not in {"误差", "日计时误差"}:
                continue

            pre_values = [_safe_float(vals[i]) for i in range(4, 9)]
            pre_values = [v for v in pre_values if v is not None]
            current_point = str(vals[2]).strip() if vals[2] is not None else ""
            power_factor = str(vals[3]).strip() if vals[3] is not None else ""
            after_value = _safe_float(vals[12])
            raw = {
                "row": vals[:16],
                "section": current_section,
            }
            records.append({
                "sample_id": current_sample,
                "source_file": path.name,
                "test_item": test_item,
                "current_point": current_point or None,
                "power_factor": power_factor or None,
                "pre_values_json": json.dumps(pre_values, ensure_ascii=False),
                "pre_avg": _safe_float(vals[9]),
                "pre_min": _safe_float(vals[10]),
                "pre_max": _safe_float(vals[11]),
                "after_value": after_value,
                "after_min_deviation": _safe_float(vals[13]),
                "after_max_deviation": _safe_float(vals[14]),
                "after_avg_deviation": _safe_float(vals[15]),
                "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
            })
        wb.close()
        return records
    finally:
        _cleanup_converted_path(path, xlsx)


def build_reliability_profiles(db: Session) -> int:
    db.query(ReliabilitySampleProfile).delete()
    db.flush()

    prod_rows = db.query(ReliabilitySampleProductionRecord).all()
    after_rows = db.query(ReliabilitySampleAfterRecord).all()
    prod_by_sample: dict[str, list[ReliabilitySampleProductionRecord]] = defaultdict(list)
    after_by_sample: dict[str, list[ReliabilitySampleAfterRecord]] = defaultdict(list)
    for row in prod_rows:
        prod_by_sample[row.sample_id].append(row)
    for row in after_rows:
        after_by_sample[row.sample_id].append(row)

    sample_ids = sorted(set(prod_by_sample) | set(after_by_sample))
    for sample_id in sample_ids:
        prod = prod_by_sample.get(sample_id, [])
        after = after_by_sample.get(sample_id, [])
        prod_errors = [r.error_value for r in prod if r.test_item == "误差" and r.error_value is not None]
        prod_abs = [abs(v) for v in prod_errors]
        prod_dates = {r.operation_time.date() for r in prod if r.operation_time is not None}
        prod_bad = sum(1 for r in prod if r.operation_result == "不合格")

        after_errors = [r.after_value for r in after if r.test_item == "误差" and r.after_value is not None]
        after_abs = [abs(v) for v in after_errors]
        deviations = [
            abs(v)
            for r in after
            for v in (r.after_min_deviation, r.after_max_deviation, r.after_avg_deviation)
            if v is not None
        ]
        timing_abs = max(
            [abs(r.after_value) for r in after if r.test_item == "日计时误差" and r.after_value is not None],
            default=0.0,
        )
        prod_bad_rate = prod_bad / max(len(prod), 1)
        risk = min(
            1.0,
            max(after_abs or [0.0]) / 1.0 * 0.35
            + max(deviations or [0.0]) / 0.5 * 0.25
            + timing_abs / 0.5 * 0.20
            + prod_bad_rate * 0.20,
        )
        label = 1 if risk >= 0.55 or max(after_abs or [0.0]) >= 0.8 or timing_abs >= 0.35 else 0

        db.add(ReliabilitySampleProfile(
            sample_id=sample_id,
            production_record_count=len(prod),
            production_error_count=len(prod_errors),
            production_error_abs_max=max(prod_abs, default=0.0),
            production_error_abs_avg=mean(prod_abs) if prod_abs else 0.0,
            production_error_abs_std=pstdev(prod_abs) if len(prod_abs) > 1 else 0.0,
            production_bad_count=prod_bad,
            production_test_date_count=len(prod_dates),
            after_error_count=len(after_errors),
            after_error_abs_max=max(after_abs, default=0.0),
            after_error_abs_avg=mean(after_abs) if after_abs else 0.0,
            after_deviation_abs_max=max(deviations, default=0.0),
            after_deviation_abs_avg=mean(deviations) if deviations else 0.0,
            timing_after_abs=timing_abs,
            reliability_risk_score=round(risk, 4),
            quality_label=label,
            note="Aggregated from X202201001 production repeated-test and after-test analysis workbooks.",
        ))
    db.commit()
    return len(sample_ids)


def import_reliability_sample_data(
    db: Session,
    production_path: Path | None = None,
    after_path: Path | None = None,
) -> dict:
    production_path = production_path or DEFAULT_PRODUCTION_XLS
    after_path = after_path or DEFAULT_AFTER_XLS

    stats = {
        "production_records": 0,
        "after_records": 0,
        "profiles": 0,
        "errors": [],
    }

    try:
        db.query(ReliabilitySampleProductionRecord).delete()
        db.query(ReliabilitySampleAfterRecord).delete()
        db.query(ReliabilitySampleProfile).delete()
        db.commit()
    except Exception as exc:
        db.rollback()
        stats["errors"].append(f"清理旧可靠性样机数据失败: {exc}")
        return stats

    if production_path.exists():
        try:
            prod_records = parse_production_workbook(production_path)
            db.bulk_save_objects([ReliabilitySampleProductionRecord(**r) for r in prod_records])
            db.commit()
            stats["production_records"] = len(prod_records)
        except Exception as exc:
            db.rollback()
            stats["errors"].append(f"生产数据解析失败: {exc}")
    else:
        stats["errors"].append(f"生产数据文件不存在: {production_path}")

    if after_path.exists():
        try:
            after_records = parse_after_workbook(after_path)
            db.bulk_save_objects([ReliabilitySampleAfterRecord(**r) for r in after_records])
            db.commit()
            stats["after_records"] = len(after_records)
        except Exception as exc:
            db.rollback()
            stats["errors"].append(f"试验后数据解析失败: {exc}")
    else:
        stats["errors"].append(f"试验后数据文件不存在: {after_path}")

    try:
        stats["profiles"] = build_reliability_profiles(db)
    except Exception as exc:
        db.rollback()
        stats["errors"].append(f"可靠性样机画像构建失败: {exc}")

    return stats


def _features_from_profile(profile: ReliabilitySampleProfile | None) -> list[float]:
    if profile is None:
        return [0.0 for _ in RELIABILITY_FEATURE_NAMES]
    prod_bad_rate = profile.production_bad_count / max(profile.production_record_count or 0, 1)
    sample_count = 1.0 if profile.sample_id else 0.0
    return [
        float(profile.production_error_abs_max or 0.0),
        float(profile.production_error_abs_avg or 0.0),
        float(profile.production_error_abs_std or 0.0),
        float(prod_bad_rate),
        float(profile.production_test_date_count or 0),
        float(profile.after_error_abs_max or 0.0),
        float(profile.after_error_abs_avg or 0.0),
        float(profile.after_deviation_abs_max or 0.0),
        float(profile.after_deviation_abs_avg or 0.0),
        float(profile.timing_after_abs or 0.0),
        float(profile.reliability_risk_score or 0.0),
        sample_count,
    ]


def get_reliability_feature_matrix(db: Session, n_rows: int) -> tuple[list[str], list[float]]:
    profiles = db.query(ReliabilitySampleProfile).order_by(ReliabilitySampleProfile.sample_id).all()
    if not profiles:
        return RELIABILITY_FEATURE_NAMES, [0.0 for _ in RELIABILITY_FEATURE_NAMES]

    values = np.asarray([_features_from_profile(p) for p in profiles], dtype=float)
    # Current source lacks a trusted meter_no/barcode bridge to aging samples, so
    # use a batch-level mean profile as reliability evidence for the training set.
    return RELIABILITY_FEATURE_NAMES, values.mean(axis=0).tolist()
