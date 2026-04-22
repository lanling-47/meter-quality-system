"""
生产过程数据 ETL 服务
解析「电表生产数据」文件夹下的10个文件：
  - CCD数据.xlsx          → production_inspection (process=CCD)
  - 耐压测试数据.xlsx       → production_inspection (process=耐压)
  - 功能测试数据.xlsx       → production_inspection (process=功能测试)
  - 复校数据.xlsx           → production_inspection (process=复校)
  - 单板测试数据.xlsx       → board_test_detail
  - 维修数据.xlsx           → repair_record
  - 贴片AOI不合格数据.xls  → aoi_defect (aoi_type=smt)
  - 选择性波峰焊AOI.xls    → aoi_defect (aoi_type=wave)
  - 贴片AOI的故障.xls      → 仅作为字典，映射质量隐患描述

同时构建 meter_quality_trace（全工序追溯汇总）
"""
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from database import SessionLocal
from app.models.db_models import (
    ProductionInspection, AoiDefect, RepairRecord,
    BoardTestDetail, MeterQualityTrace,
)

# ── 生产数据目录（从 config 读取，或使用默认路径）────────────────────────
try:
    import config as _cfg
    PRODUCTION_DATA_DIR = _cfg.PRODUCTION_DATA_DIR
except Exception:
    PRODUCTION_DATA_DIR = Path(__file__).parent.parent.parent.parent / "电表生产数据"


def _safe_dt(val) -> Optional[datetime]:
    """将 pandas Timestamp 或字符串转为 datetime"""
    if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'NaT'):
        return None
    try:
        if isinstance(val, datetime):
            return val
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None and str(val) != 'nan' else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# 1. 解析工序检测数据（CCD / 耐压 / 功能测试 / 复校）
# ══════════════════════════════════════════════════════════════════════════
def parse_inspection(filepath: Path, process: str) -> list[dict]:
    """解析四类工序检测文件，返回记录列表"""
    df = pd.read_excel(filepath, engine='openpyxl')
    df.columns = df.columns.str.strip()
    records = []
    for _, row in df.iterrows():
        barcode = str(row.get('ATXM(条形码)', '') or '').strip()
        if not barcode or barcode in ('nan', ''):
            continue
        remark_val = row.get('BZ（不合格项备注）', None)
        remark = str(remark_val).strip() if remark_val and str(remark_val) != 'nan' else None
        leak_val = row.get('LDL（漏电流）', None)

        records.append({
            'barcode': barcode,
            'order_no': str(row.get('DDH（订单号）', '') or '').strip(),
            'station_no': str(row.get('BWH（表位号）', '') or '').strip(),
            'process': process,
            'test_count': int(row.get('JYCS（测试次数）', 1) or 1),
            'result': str(row.get('JYJL（结论）', '') or '').strip(),
            'test_time': _safe_dt(row.get('JYRQ（时间）')),
            'device_no': str(row.get('JYZZ（设备编号）', '') or '').strip(),
            'remark': remark,
            'leak_current': _safe_float(leak_val),
        })
    return records


# ══════════════════════════════════════════════════════════════════════════
# 2. 解析 AOI 故障字典
# ══════════════════════════════════════════════════════════════════════════
def _load_aoi_fault_dict(filepath: Path) -> dict:
    """返回 {故障代码: 质量隐患描述}"""
    try:
        df = pd.read_excel(filepath, engine='xlrd')
        result = {}
        for _, row in df.iterrows():
            code = str(row.iloc[0] or '').strip()
            risk = str(row.iloc[2] or '').strip() if len(row) > 2 else ''
            if code and code != 'nan':
                result[code] = risk
        return result
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════
# 3. 解析 AOI 不合格数据
# ══════════════════════════════════════════════════════════════════════════
def parse_aoi_defects(filepath: Path, aoi_type: str, fault_dict: dict) -> list[dict]:
    engine = 'xlrd' if str(filepath).endswith('.xls') else 'openpyxl'
    df = pd.read_excel(filepath, engine=engine)
    df.columns = df.columns.str.strip()
    records = []
    for _, row in df.iterrows():
        barcode = str(row.get('TXM（条形码）', '') or '').strip()
        if not barcode or barcode == 'nan':
            continue
        defect_desc = str(row.get('GZMS（故障描述）', '') or '').strip()
        quality_risk = fault_dict.get(defect_desc, '')
        records.append({
            'barcode': barcode,
            'order_no': str(row.get('DDH（订单号）', '') or '').strip(),
            'device_no': str(row.get('SBBH（设备编号）', '') or '').strip(),
            'defect_point': str(row.get('GZD（故障点）', '') or '').strip(),
            'defect_desc': defect_desc,
            'process_no': str(row.get('DQGX（当前工序）', '') or '').strip(),
            'detect_time': _safe_dt(row.get('SJ（时间）')),
            'aoi_type': aoi_type,
            'quality_risk': quality_risk or None,
        })
    return records


# ══════════════════════════════════════════════════════════════════════════
# 4. 解析维修记录
# ══════════════════════════════════════════════════════════════════════════
def parse_repair_records(filepath: Path) -> list[dict]:
    df = pd.read_excel(filepath, engine='openpyxl')
    df.columns = df.columns.str.strip()
    records = []
    for _, row in df.iterrows():
        barcode = str(row.get('TXM（条形码）', '') or '').strip()
        if not barcode or barcode == 'nan':
            continue
        records.append({
            'barcode': barcode,
            'order_no': str(row.get('PCBM（订单号）', '') or '').strip(),
            'fault_code': str(row.get('GZXXDM（故障现象代码）', '') or '').strip(),
            'fault_name': str(row.get('GZXXMC（故障现象名称）', '') or '').strip(),
            'cause_code': str(row.get('GZYYDM（故障原因代码）', '') or '').strip(),
            'cause_name': str(row.get('GZYYMC（故障原因名称）', '') or '').strip(),
            'fault_location': str(row.get('GZDD（故障地点）', '') or '').strip(),
            'fault_process': str(row.get('GZGX（故障工序）', '') or '').strip(),
            'fault_time': _safe_dt(row.get('GZJLSJ（故障记录时间）')),
            'repair_time': _safe_dt(row.get('WXJLSJ（维修记录时间）')),
        })
    return records


# ══════════════════════════════════════════════════════════════════════════
# 5. 解析单板测试数据（大文件，仅写入失败记录，节省空间）
# ══════════════════════════════════════════════════════════════════════════
def parse_board_test(filepath: Path, failed_only: bool = True) -> list[dict]:
    """
    解析单板测试数据。
    failed_only=True 时只写入不合格 (Fail/FAIL/NG) 记录，减少数据量。
    """
    df = pd.read_excel(filepath, engine='openpyxl')
    df.columns = df.columns.str.strip()
    records = []
    for _, row in df.iterrows():
        barcode = str(row.get('CS_TXM（条形码）', '') or '').strip()
        if not barcode or barcode == 'nan':
            continue
        result_val = str(row.get('CS_JG（结论）', '') or '').strip().upper()
        if failed_only and result_val in ('OK', 'PASS', '合格', '成功'):
            continue
        records.append({
            'barcode': barcode,
            'order_no': str(row.get('DDH（订单号）', '') or '').strip(),
            'scheme_name': str(row.get('CS_XMM（方案名称）', '') or '').strip(),
            'seq_no': int(row.get('CS_XH（顺序号）', 0) or 0),
            'test_item': str(row.get('CS_XM（测试项目）', '') or '').strip(),
            'result': result_val,
            'test_data': str(row.get('CS_DATA（测试数据）', '') or '').strip(),
            'test_time': _safe_dt(row.get('CS_DATE（测试时间）')),
            'device_no': str(row.get('GZBH（设备编号）', '') or '').strip(),
        })
    return records


# ══════════════════════════════════════════════════════════════════════════
# 6. 构建全工序追溯汇总表
# ══════════════════════════════════════════════════════════════════════════
def build_quality_trace(db: Session) -> list[dict]:
    """
    按条形码聚合四类工序检测+AOI+维修数据，
    构建每块电表的全工序质量追溯记录。
    """
    from sqlalchemy import text
    from collections import defaultdict

    # 取所有工序检测记录
    insp_all = db.query(ProductionInspection).all()
    # 按 barcode + process 分组
    meter_insp: dict = defaultdict(lambda: defaultdict(list))
    for r in insp_all:
        meter_insp[r.barcode][r.process].append(r)

    # AOI缺陷
    aoi_all = db.query(AoiDefect).all()
    meter_aoi: dict = defaultdict(list)
    for r in aoi_all:
        meter_aoi[r.barcode].append(r)

    # 维修
    repair_all = db.query(RepairRecord).all()
    meter_repair: dict = defaultdict(list)
    for r in repair_all:
        meter_repair[r.barcode].append(r)

    traces = []
    all_barcodes = set(meter_insp.keys()) | set(meter_aoi.keys()) | set(meter_repair.keys())

    for bc in all_barcodes:
        proc = meter_insp.get(bc, {})

        # ── CCD ──────────────────────────────────────────────────────
        ccd_recs = proc.get('CCD', [])
        ccd_pass = None
        ccd_retry = 0
        if ccd_recs:
            # 最终一次结论
            last_ccd = sorted(ccd_recs, key=lambda x: x.test_count)[-1]
            ccd_pass = last_ccd.result in ('合格', '成功')
            ccd_retry = max(r.test_count for r in ccd_recs) - 1

        # ── 耐压 ──────────────────────────────────────────────────────
        hipot_recs = proc.get('耐压', [])
        hipot_pass = None
        leak_current_max = None
        if hipot_recs:
            last_hipot = sorted(hipot_recs, key=lambda x: x.test_count)[-1]
            hipot_pass = last_hipot.result in ('合格',)
            leaks = [r.leak_current for r in hipot_recs if r.leak_current is not None]
            leak_current_max = max(leaks) if leaks else None

        # ── 功能测试 ──────────────────────────────────────────────────
        func_recs = proc.get('功能测试', [])
        func_pass = None
        func_retry = 0
        func_fail_items = []
        if func_recs:
            last_func = sorted(func_recs, key=lambda x: x.test_count)[-1]
            func_pass = last_func.result in ('合格',)
            func_retry = max(r.test_count for r in func_recs) - 1
            for r in func_recs:
                if r.remark and r.remark.strip():
                    for item in r.remark.split(';'):
                        item = item.strip()
                        if item and item not in func_fail_items:
                            func_fail_items.append(item)

        # ── 复校 ──────────────────────────────────────────────────────
        reverify_recs = proc.get('复校', [])
        reverify_pass = None
        reverify_retry = 0
        reverify_fail_items = []
        if reverify_recs:
            last_rv = sorted(reverify_recs, key=lambda x: x.test_count)[-1]
            reverify_pass = last_rv.result in ('合格',)
            reverify_retry = max(r.test_count for r in reverify_recs) - 1
            for r in reverify_recs:
                if r.remark and r.remark.strip():
                    for item in r.remark.split(';'):
                        item = item.strip()
                        if item and item not in reverify_fail_items:
                            reverify_fail_items.append(item)

        # ── AOI ───────────────────────────────────────────────────────
        aoi_recs = meter_aoi.get(bc, [])
        smt_defects = [r for r in aoi_recs if r.aoi_type == 'smt']
        wave_defects = [r for r in aoi_recs if r.aoi_type == 'wave']

        # ── 维修 ──────────────────────────────────────────────────────
        repair_recs = meter_repair.get(bc, [])

        # ── 获取订单号 ─────────────────────────────────────────────────
        order_no = ''
        for p_recs in proc.values():
            if p_recs and p_recs[0].order_no:
                order_no = p_recs[0].order_no
                break

        # ── 质量风险评分（启发式，综合多工序失败次数）────────────────────
        risk_score = 0.0
        if ccd_pass is False:
            risk_score += 0.25
        if ccd_retry > 0:
            risk_score += min(0.1 * ccd_retry, 0.2)
        if func_pass is False:
            risk_score += 0.25
        if func_retry > 0:
            risk_score += min(0.1 * func_retry, 0.2)
        if reverify_pass is False:
            risk_score += 0.25
        if smt_defects:
            risk_score += 0.15
        if wave_defects:
            risk_score += 0.15
        if repair_recs:
            risk_score += 0.2
        risk_score = round(min(risk_score, 1.0), 4)

        final_pass = (
            (ccd_pass is not False) and
            (hipot_pass is not False) and
            (func_pass is not False) and
            (reverify_pass is not False) and
            not repair_recs
        )

        total_retry = (ccd_retry or 0) + (func_retry or 0) + (reverify_retry or 0)

        traces.append({
            'barcode': bc,
            'order_no': order_no,
            'ccd_pass': ccd_pass,
            'ccd_retry_count': ccd_retry,
            'hipot_pass': hipot_pass,
            'leak_current_max': leak_current_max,
            'func_pass': func_pass,
            'func_retry_count': func_retry,
            'func_fail_items': json.dumps(func_fail_items, ensure_ascii=False) if func_fail_items else None,
            'reverify_pass': reverify_pass,
            'reverify_retry_count': reverify_retry,
            'reverify_fail_items': json.dumps(reverify_fail_items, ensure_ascii=False) if reverify_fail_items else None,
            'smt_aoi_defect': bool(smt_defects),
            'smt_aoi_defect_types': json.dumps(list({r.defect_desc for r in smt_defects}), ensure_ascii=False) if smt_defects else None,
            'wave_aoi_defect': bool(wave_defects),
            'wave_aoi_defect_types': json.dumps(list({r.defect_desc for r in wave_defects}), ensure_ascii=False) if wave_defects else None,
            'repaired': bool(repair_recs),
            'repair_count': len(repair_recs),
            'fault_processes': json.dumps(list({r.fault_process for r in repair_recs}), ensure_ascii=False) if repair_recs else None,
            'fault_names': json.dumps([r.fault_name for r in repair_recs], ensure_ascii=False) if repair_recs else None,
            'final_pass': final_pass,
            'total_retry_count': total_retry,
            'quality_risk_score': risk_score,
        })

    return traces


# ══════════════════════════════════════════════════════════════════════════
# 7. 主导入函数
# ══════════════════════════════════════════════════════════════════════════
def import_production_data(db: Session, data_dir: Path = None) -> dict:
    """
    从「电表生产数据」目录导入所有生产过程数据。
    幂等操作：先清空相关表，再重新写入。
    """
    if data_dir is None:
        data_dir = PRODUCTION_DATA_DIR

    stats = {
        'inspection': 0,
        'aoi_defect': 0,
        'repair': 0,
        'board_test': 0,
        'trace': 0,
        'errors': [],
    }

    # ── 清空旧数据 ──────────────────────────────────────────────────────
    try:
        db.query(MeterQualityTrace).delete()
        db.query(ProductionInspection).delete()
        db.query(AoiDefect).delete()
        db.query(RepairRecord).delete()
        db.query(BoardTestDetail).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        stats['errors'].append(f'清空旧数据失败: {e}')
        return stats

    # ── 加载 AOI 故障字典 ──────────────────────────────────────────────
    fault_dict_path = data_dir / '贴片AOI的故障.xls'
    fault_dict = _load_aoi_fault_dict(fault_dict_path)

    # ── 工序检测文件 ───────────────────────────────────────────────────
    inspection_files = [
        ('CCD数据.xlsx', 'CCD'),
        ('耐压测试数据.xlsx', '耐压'),
        ('功能测试数据.xlsx', '功能测试'),
        ('复校数据.xlsx', '复校'),
    ]
    for filename, process in inspection_files:
        fp = data_dir / filename
        if not fp.exists():
            stats['errors'].append(f'文件不存在: {filename}')
            continue
        try:
            records = parse_inspection(fp, process)
            batch = []
            for r in records:
                batch.append(ProductionInspection(**r))
                if len(batch) >= 1000:
                    db.bulk_save_objects(batch)
                    db.commit()
                    batch = []
            if batch:
                db.bulk_save_objects(batch)
                db.commit()
            stats['inspection'] += len(records)
        except Exception as e:
            db.rollback()
            stats['errors'].append(f'{filename}: {e}')

    # ── AOI 缺陷 ────────────────────────────────────────────────────────
    aoi_files = [
        ('贴片AOI不合格数据.xls', 'smt'),
        ('选择性波峰焊AOI.xls', 'wave'),
    ]
    for filename, aoi_type in aoi_files:
        fp = data_dir / filename
        if not fp.exists():
            stats['errors'].append(f'文件不存在: {filename}')
            continue
        try:
            records = parse_aoi_defects(fp, aoi_type, fault_dict)
            db.bulk_save_objects([AoiDefect(**r) for r in records])
            db.commit()
            stats['aoi_defect'] += len(records)
        except Exception as e:
            db.rollback()
            stats['errors'].append(f'{filename}: {e}')

    # ── 维修记录 ────────────────────────────────────────────────────────
    repair_fp = data_dir / '维修数据.xlsx'
    if repair_fp.exists():
        try:
            records = parse_repair_records(repair_fp)
            db.bulk_save_objects([RepairRecord(**r) for r in records])
            db.commit()
            stats['repair'] += len(records)
        except Exception as e:
            db.rollback()
            stats['errors'].append(f'维修数据.xlsx: {e}')
    else:
        stats['errors'].append('文件不存在: 维修数据.xlsx')

    # ── 单板测试（仅不合格记录）────────────────────────────────────────
    board_fp = data_dir / '单板测试数据.xlsx'
    if board_fp.exists():
        try:
            records = parse_board_test(board_fp, failed_only=True)
            batch = []
            for r in records:
                batch.append(BoardTestDetail(**r))
                if len(batch) >= 500:
                    db.bulk_save_objects(batch)
                    db.commit()
                    batch = []
            if batch:
                db.bulk_save_objects(batch)
                db.commit()
            stats['board_test'] += len(records)
        except Exception as e:
            db.rollback()
            stats['errors'].append(f'单板测试数据.xlsx: {e}')
    else:
        stats['errors'].append('文件不存在: 单板测试数据.xlsx')

    # ── 构建追溯汇总 ────────────────────────────────────────────────────
    try:
        traces = build_quality_trace(db)
        batch = []
        for r in traces:
            batch.append(MeterQualityTrace(**r))
            if len(batch) >= 500:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []
        if batch:
            db.bulk_save_objects(batch)
            db.commit()
        stats['trace'] = len(traces)
    except Exception as e:
        db.rollback()
        stats['errors'].append(f'追溯汇总构建失败: {e}')

    return stats


# ══════════════════════════════════════════════════════════════════════════
# 8. 为 LightGBM 提供扩充特征（跨工序质量风险特征）
# ══════════════════════════════════════════════════════════════════════════
PRODUCTION_FEATURE_NAMES = [
    'prod_ccd_retry',          # CCD重测次数
    'prod_func_retry',         # 功能测试重测次数
    'prod_reverify_retry',     # 复校重测次数
    'prod_total_retry',        # 总重测次数
    'prod_smt_aoi_defect',     # 贴片AOI是否有缺陷 0/1
    'prod_wave_aoi_defect',    # 波峰焊AOI是否有缺陷 0/1
    'prod_repaired',           # 是否经过维修 0/1
    'prod_repair_count',       # 维修次数
    'prod_leak_current_max',   # 最大漏电流（耐压）
    'prod_quality_risk_score', # 综合生产质量风险分
]


def get_production_features_for_barcode(barcode: str, db: Session) -> dict:
    """
    根据条形码获取生产过程特征，用于扩充 LightGBM 特征向量。
    如无数据则返回默认值（0）。
    """
    trace = db.query(MeterQualityTrace).filter_by(barcode=barcode).first()
    if trace is None:
        return {k: 0.0 for k in PRODUCTION_FEATURE_NAMES}
    return {
        'prod_ccd_retry': float(trace.ccd_retry_count or 0),
        'prod_func_retry': float(trace.func_retry_count or 0),
        'prod_reverify_retry': float(trace.reverify_retry_count or 0),
        'prod_total_retry': float(trace.total_retry_count or 0),
        'prod_smt_aoi_defect': 1.0 if trace.smt_aoi_defect else 0.0,
        'prod_wave_aoi_defect': 1.0 if trace.wave_aoi_defect else 0.0,
        'prod_repaired': 1.0 if trace.repaired else 0.0,
        'prod_repair_count': float(trace.repair_count or 0),
        'prod_leak_current_max': float(trace.leak_current_max or 0.0),
        'prod_quality_risk_score': float(trace.quality_risk_score or 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════
# 9. 维修闭环分析（方向4）
# ══════════════════════════════════════════════════════════════════════════
def get_repair_analysis(db: Session) -> dict:
    """
    维修闭环分析：
    - 故障现象分布
    - 故障原因分布
    - 工序→故障现象映射
    - 复校不合格项→维修关联
    """
    repairs = db.query(RepairRecord).all()
    if not repairs:
        return {'error': '暂无维修数据，请先导入'}

    from collections import Counter

    fault_name_dist = Counter(r.fault_name for r in repairs if r.fault_name)
    cause_name_dist = Counter(r.cause_name for r in repairs if r.cause_name)
    process_dist = Counter(r.fault_process for r in repairs if r.fault_process)

    # 工序→故障现象映射
    process_fault_map: dict = {}
    for r in repairs:
        proc = r.fault_process or '未知'
        fault = r.fault_name or '未知'
        if proc not in process_fault_map:
            process_fault_map[proc] = Counter()
        process_fault_map[proc][fault] += 1

    # 分析每个工序的TOP故障
    process_top_faults = {}
    for proc, counter in process_fault_map.items():
        process_top_faults[proc] = [
            {'fault': k, 'count': v}
            for k, v in counter.most_common(5)
        ]

    # 复校失败 → 维修关联分析
    reverify_fail_barcodes = {
        r.barcode
        for r in db.query(ProductionInspection).filter_by(process='复校').all()
        if r.result == '不合格'
    }
    repaired_barcodes = {r.barcode for r in repairs}
    reverify_then_repair = reverify_fail_barcodes & repaired_barcodes

    # AOI缺陷 → 功能测试失败关联
    aoi_defect_barcodes = {r.barcode for r in db.query(AoiDefect).all()}
    func_fail_barcodes = {
        r.barcode
        for r in db.query(ProductionInspection).filter_by(process='功能测试').all()
        if r.result == '不合格'
    }
    aoi_then_func_fail = aoi_defect_barcodes & func_fail_barcodes

    return {
        'total_repairs': len(repairs),
        'unique_meters_repaired': len(repaired_barcodes),
        'fault_name_dist': [{'name': k, 'count': v} for k, v in fault_name_dist.most_common(15)],
        'cause_name_dist': [{'name': k, 'count': v} for k, v in cause_name_dist.most_common(10)],
        'process_dist': [{'process': k, 'count': v} for k, v in process_dist.most_common()],
        'process_top_faults': process_top_faults,
        'reverify_fail_count': len(reverify_fail_barcodes),
        'reverify_then_repair_count': len(reverify_then_repair),
        'reverify_repair_rate': round(
            len(reverify_then_repair) / len(reverify_fail_barcodes) * 100, 1
        ) if reverify_fail_barcodes else 0,
        'aoi_defect_count': len(aoi_defect_barcodes),
        'aoi_then_func_fail_count': len(aoi_then_func_fail),
        'aoi_func_fail_rate': round(
            len(aoi_then_func_fail) / len(aoi_defect_barcodes) * 100, 1
        ) if aoi_defect_barcodes else 0,
    }


def get_process_defect_propagation(db: Session) -> dict:
    """
    缺陷传播路径分析：
    分析"某工序曾有不合格记录"→"后续工序也曾有不合格记录"的传播概率
    """
    # 按条形码聚合：是否在某工序出现过不合格
    insp_all = db.query(ProductionInspection).all()
    from collections import defaultdict
    barcode_fail: dict = defaultdict(set)  # barcode → {fail_process, ...}
    barcode_processes: dict = defaultdict(set)  # barcode → {process, ...}

    for r in insp_all:
        barcode_processes[r.barcode].add(r.process)
        if r.result == '不合格':
            barcode_fail[r.barcode].add(r.process)

    processes = ['CCD', '功能测试', '耐压', '复校']
    propagation = []

    for p1 in processes:
        for p2 in processes:
            if p1 == p2:
                continue
            # 在 p1 曾不合格 且 也有 p2 检测记录 的电表
            p1_fail_barcodes = [
                bc for bc, fails in barcode_fail.items()
                if p1 in fails and p2 in barcode_processes[bc]
            ]
            # 其中 p2 也曾不合格
            p1_fail_then_p2_fail = [
                bc for bc in p1_fail_barcodes
                if p2 in barcode_fail.get(bc, set())
            ]
            if p1_fail_barcodes:
                propagation.append({
                    'from_process': p1,
                    'to_process': p2,
                    'base_count': len(p1_fail_barcodes),
                    'propagated_count': len(p1_fail_then_p2_fail),
                    'propagation_rate': round(
                        len(p1_fail_then_p2_fail) / len(p1_fail_barcodes) * 100, 1
                    ),
                })

    # 整体：曾出现不合格的电表比率
    overall = {}
    for proc in processes:
        all_bc = [bc for bc, procs in barcode_processes.items() if proc in procs]
        fail_bc = [bc for bc in all_bc if proc in barcode_fail.get(bc, set())]
        overall[proc] = {
            'total': len(all_bc),
            'fail': len(fail_bc),
            'fail_rate': round(len(fail_bc) / len(all_bc) * 100, 2) if all_bc else 0,
        }

    return {
        'propagation': sorted(propagation, key=lambda x: x['propagation_rate'], reverse=True),
        'overall_fail_rates': overall,
    }
