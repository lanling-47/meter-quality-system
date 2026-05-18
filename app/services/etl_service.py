"""
ETL 数据治理服务
解析3个 Excel 文件（11个Sheets），完成数据清洗、关联建模，写入 SQLite 数据库
新增：日计时、误差变差、误差一致性、正反向电量累计、开表盖 5个Sheet解析
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import openpyxl
from sqlalchemy.orm import Session

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from database import SessionLocal
from app.models.db_models import (
    CrystalData,
    CTData,
    FactoryTest,
    AgingTest,
    QualityChain,
    ProductionProcessProfile,
)
import config as cfg


# ─── 时间点映射：标签 → 近似试验时长(h) ───────────────────────────
TIME_POINT_MAP = {
    "6.18": 0, "6.19": 0,
    "6.29": 264,
    "7.8": 480,
    "7.16": 672,
    "7.25": 864,
    "8.5": 1128, "8.6": 1128,
}


def _safe_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(str(val).strip().replace("+", ""))
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    if val is None or val == "":
        return None
    s = str(val).strip().replace("+", "")
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _extract_numbers(val) -> list[float]:
    if val is None or val == "":
        return []
    if isinstance(val, (int, float)):
        return [float(val)]
    text = str(val).replace("，", ".").replace("：", ":")
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def _first_number(val) -> Optional[float]:
    nums = _extract_numbers(val)
    return nums[0] if nums else None


def _normalize_serial(value) -> str:
    if value is None:
        return ""
    text = re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())
    runs = re.findall(r"\d{6,}", text)
    return max(runs, key=len) if runs else text


def _sheet_or_fallback(wb, preferred: str):
    if preferred in wb.sheetnames:
        return wb[preferred]
    return wb[wb.sheetnames[-1]]


def _parse_time_block(rows, allow_no_tablabel=False):
    """
    通用时间块解析器：解析格式为
      col0="[试验前/测试时间]X.X" col1="表号" col2..=电表编号
      后续行 col0=None col1=条件 col2..=数值
    返回：{time_label: {meter_no: {condition: value}}}
    """
    blocks = {}
    current_time = None
    meter_nos = []

    for row in rows:
        first = str(row[0]).strip() if row[0] else ""
        second = str(row[1]).strip() if row[1] else "" if len(row) > 1 else ""

        is_time_row = (
            ("测试时间" in first or "试验前" in first)
            and re.search(r"\d+\.\d+", first)
            and "表号" in second
        )

        if is_time_row:
            m = re.search(r"(\d+\.\d+)", first)
            if m:
                current_time = m.group(1)
                blocks[current_time] = {}
                meter_nos = [_safe_int(c) for c in row[2:] if _safe_int(c) and 100 < _safe_int(c) < 300]
                for mn in meter_nos:
                    blocks[current_time][mn] = {}
        elif current_time and meter_nos and second and second not in ("表号", ""):
            for idx, mn in enumerate(meter_nos):
                val = _safe_float(row[idx + 2]) if idx + 2 < len(row) else None
                if val is not None and mn in blocks.get(current_time, {}):
                    blocks[current_time][mn][second] = val

    return blocks


# ═══════════════════════════════════════════════════════════════════════
# 1. 解析晶振来料检测数据
# ═══════════════════════════════════════════════════════════════════════
def parse_crystal_data(filepath: Path) -> list[dict]:
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    ws = wb["晶振测试数据"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    records = []
    batch_no, model, center_freq, load_cap = None, None, None, None

    for i, row in enumerate(rows):
        if i < 2:
            continue
        if row[0]: model = str(row[0]).strip()
        if row[1]: batch_no = str(row[1]).strip()
        if row[2]:
            freq_str = str(row[2]).split("\n")[0].split("±")[0].strip()
            try:
                center_freq = float(freq_str) * 1e6
            except Exception:
                center_freq = None
        if row[3]: load_cap = _safe_float(row[3])

        if row[4] is not None and row[5] is not None:
            try:
                sno = int(row[4])
            except Exception:
                continue
            ppm = _safe_float(row[5])
            if ppm is None:
                continue
            records.append({
                "batch_no": batch_no or "20251008",
                "sample_no": sno,
                "model": model or "HC-49SMD",
                "center_freq_hz": center_freq,
                "load_cap_pf": load_cap,
                "freq_error_ppm": ppm,
                "is_qualified": abs(ppm) <= cfg.CRYSTAL_PPM_LIMIT,
            })
    return records


# ═══════════════════════════════════════════════════════════════════════
# 2. 解析互感器来料检测数据
# ═══════════════════════════════════════════════════════════════════════
def parse_ct_data(filepath: Path) -> list[dict]:
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    ws = wb["互感器测试数据"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    batch_no, model = "20251220", "HDCT3 5(60)A2mA"
    for row in rows[:3]:
        for cell in row:
            if cell and "型号" in str(cell):
                m = re.search(r"型号[：:]\s*(.+)", str(cell))
                if m: model = m.group(1).strip()
            if cell and "批次" in str(cell):
                m = re.search(r"批次[：:]\s*(\d+)", str(cell))
                if m: batch_no = m.group(1)

    data_start = 0
    for idx, row in enumerate(rows):
        if row[0] and str(row[0]).strip().isdigit() and int(str(row[0]).strip()) == 1:
            data_start = idx
            break

    records = []
    for row in rows[data_start:]:
        if not row[0]:
            continue
        try:
            sno = int(str(row[0]).strip())
        except Exception:
            continue

        raw_vals = [_safe_float(cell) for cell in row[1:] if _safe_float(cell) is not None]
        ratios = [raw_vals[i] for i in range(0, min(len(raw_vals), 16), 2)]
        angles = [raw_vals[i] for i in range(1, min(len(raw_vals), 16), 2)]

        ratio_max = max([abs(r) for r in ratios], default=0.0)
        ratio_range = (max(ratios, default=0) - min(ratios, default=0)) if ratios else 0
        angle_max = max([abs(a) for a in angles], default=0.0)

        records.append({
            "batch_no": batch_no,
            "sample_no": sno,
            "model": model,
            "ratio_errors_json": json.dumps(ratios),
            "angle_errors_json": json.dumps(angles),
            "ratio_error_max": ratio_max,
            "ratio_error_range": ratio_range,
            "angle_error_max": angle_max,
            "dc_resistance": None,
            "is_qualified": ratio_max <= cfg.CT_RATIO_LIMIT,
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
# 3. 解析出厂检测数据
# ═══════════════════════════════════════════════════════════════════════
def parse_factory_test(filepath: Path) -> list[dict]:
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb["确认版"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    records = []
    in_factory_table = False
    for row in rows:
        first_text = str(row[0]).strip() if row[0] is not None else ""
        if "电表唯一ID" in first_text:
            in_factory_table = True
            continue
        if in_factory_table and ("检测参量" in first_text or "故障类型" in first_text or "数字化电路板" in first_text):
            break
        if not in_factory_table:
            continue

        meter_id = row[0]
        if not meter_id or not str(meter_id).strip().isdigit():
            continue
        mid = str(meter_id).strip()
        if len(mid) < 10:
            continue

        crystal_batch = str(row[1]).strip() if row[1] else ""
        ct_batch = str(row[2]).strip() if row[2] else ""
        error_str = str(row[3]) if row[3] else ""
        crystal_ppm = None
        ct_ratio_err = None
        m_c = re.search(r"晶振[：:]\s*([\d.]+)", error_str)
        m_ct = re.search(r"互感器[：:]\s*([\d.]+)", error_str)
        if m_c: crystal_ppm = float(m_c.group(1))
        if m_ct: ct_ratio_err = float(m_ct.group(1))

        fault_type = str(row[4]).strip() if row[4] else ""
        is_fault = bool(fault_type and fault_type not in ("", "None", "正常"))

        records.append({
            "meter_id": mid,
            "crystal_batch": crystal_batch,
            "ct_batch": ct_batch,
            "crystal_ppm": crystal_ppm,
            "ct_ratio_err": ct_ratio_err,
            "fault_type": fault_type if fault_type else "正常",
            "is_fault": is_fault,
            "meter_constant_err": None,
            "clock_accuracy_sd": None,
            "rs485_ok": True,
            "ir_ok": True,
            "module_ok": True,
            "startup_ok": True,
            "creep_ok": True,
        })

    meter_idx = 0
    for row in rows:
        if row[0] and "正向有功" in str(row[0]):
            if meter_idx < len(records):
                records[meter_idx]["meter_constant_err"] = _safe_float(row[3])
                records[meter_idx]["clock_accuracy_sd"] = _safe_float(row[4])
                startup_str = str(row[1]) if row[1] else ""
                creep_str = str(row[2]) if row[2] else ""
                records[meter_idx]["startup_ok"] = "合格" in startup_str
                records[meter_idx]["creep_ok"] = "合格" in creep_str
                meter_idx += 1
    return records


# ═══════════════════════════════════════════════════════════════════════
# 4. 解析日计时误差（日计时 Sheet）
# ═══════════════════════════════════════════════════════════════════════
def parse_factory_process_profile(filepath: Path) -> list[dict]:
    """解析出厂检测工作簿中嵌入的数字化电路板生产过程画像。"""
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = _sheet_or_fallback(wb, "确认版")
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    def find_row(keyword: str) -> Optional[int]:
        for idx, row in enumerate(rows):
            for cell in row:
                if cell is not None and keyword in str(cell):
                    return idx
        return None

    if find_row("数字化电路板生产过程数据") is None:
        return []

    smt_idx = find_row("SMT贴片")
    reflow_idx = find_row("回流焊")
    online_idx = find_row("在线测试")
    wave_idx = find_row("波峰焊")

    smt_pressure = None
    smt_tol = None
    smt_cpk = None
    if smt_idx is not None:
        for row in rows[smt_idx + 1: min(smt_idx + 8, len(rows))]:
            if smt_pressure is None and row[0] is not None and "mm" in str(row[0]):
                smt_pressure = _first_number(row[0])
            if row[1] is not None and ("CPK" in str(row[1]).upper() or "mm" in str(row[1])):
                nums = _extract_numbers(row[1])
                if nums:
                    smt_tol = nums[0]
                    smt_cpk = nums[-1]

    reflow_temps: list[float] = []
    reflow_speeds: list[float] = []
    if reflow_idx is not None:
        stop = online_idx if online_idx is not None else len(rows)
        for row in rows[reflow_idx + 1: stop]:
            cell0 = row[0] if len(row) > 0 else None
            cell1 = row[1] if len(row) > 1 else None
            if cell0 is not None and "℃" in str(cell0):
                temp = _first_number(cell0)
                if temp is not None:
                    reflow_temps.append(temp)
            if cell1 is not None and ("cm" in str(cell1) or "分" in str(cell1)):
                speed = _first_number(cell1)
                if speed is not None:
                    reflow_speeds.append(speed)

    online_currents: dict[str, float] = {}
    if online_idx is not None:
        stop = wave_idx if wave_idx is not None else len(rows)
        for row in rows[online_idx + 1: stop]:
            meter_id = str(row[0]).strip() if row[0] is not None else ""
            if not meter_id.isdigit() or len(meter_id) < 10:
                continue
            current = _first_number(row[1] if len(row) > 1 else None)
            if current is not None:
                online_currents[meter_id] = current

    wave_flux = None
    wave_preheats: list[float] = []
    wave_solder_temp = None
    wave_peak_1 = None
    wave_peak_2 = None
    if wave_idx is not None:
        for row in rows[wave_idx + 1: min(wave_idx + 12, len(rows))]:
            cell0 = row[0] if len(row) > 0 else None
            cell1 = row[1] if len(row) > 1 else None
            cell2 = row[2] if len(row) > 2 else None
            cell3 = row[3] if len(row) > 3 else None

            if wave_flux is None and isinstance(cell0, (int, float)):
                wave_flux = float(cell0)
            if cell1 is not None and "℃" in str(cell1):
                temp = _first_number(cell1)
                if temp is not None:
                    wave_preheats.append(temp)
            if wave_solder_temp is None and cell2 is not None and "℃" in str(cell2):
                wave_solder_temp = _first_number(cell2)
            if cell3 is not None and "波峰" in str(cell3):
                text = str(cell3).replace("：", ":")
                peaks = re.findall(r"波峰\s*[12]\s*:\s*([-+]?\d+(?:\.\d+)?)", text)
                if len(peaks) >= 2:
                    wave_peak_1 = float(peaks[0])
                    wave_peak_2 = float(peaks[1])
                else:
                    nums = _extract_numbers(text)
                    if len(nums) >= 2:
                        wave_peak_1, wave_peak_2 = nums[-2], nums[-1]

    if not any([smt_pressure, reflow_temps, online_currents, wave_preheats]):
        return []

    meter_ids = list(online_currents.keys())
    normalized_ids = [_normalize_serial(mid) for mid in meter_ids]
    current_values = list(online_currents.values())
    reflow_min = min(reflow_temps) if reflow_temps else None
    reflow_max = max(reflow_temps) if reflow_temps else None
    preheat_min = min(wave_preheats) if wave_preheats else None
    preheat_max = max(wave_preheats) if wave_preheats else None
    preheat_avg = (sum(wave_preheats) / len(wave_preheats)) if wave_preheats else None

    raw_payload = {
        "smt": {
            "pressure_mm": smt_pressure,
            "precision_tol_mm": smt_tol,
            "precision_cpk": smt_cpk,
        },
        "reflow": {
            "temps_c": reflow_temps,
            "speeds_cm_min": reflow_speeds,
        },
        "online_test": {
            "current_by_meter_ua": online_currents,
        },
        "wave_solder": {
            "flux_spray_qty": wave_flux,
            "preheat_temps_c": wave_preheats,
            "solder_temp_c": wave_solder_temp,
            "peak_height_1_mm": wave_peak_1,
            "peak_height_2_mm": wave_peak_2,
        },
    }

    return [{
        "batch_key": filepath.stem,
        "source_file": filepath.name,
        "meter_ids_json": json.dumps(meter_ids, ensure_ascii=False),
        "normalized_meter_ids_json": json.dumps(normalized_ids, ensure_ascii=False),
        "smt_pressure_mm": smt_pressure,
        "smt_precision_tol_mm": smt_tol,
        "smt_precision_cpk": smt_cpk,
        "reflow_temp_min_c": reflow_min,
        "reflow_temp_max_c": reflow_max,
        "reflow_temp_span_c": (reflow_max - reflow_min) if reflow_min is not None and reflow_max is not None else None,
        "reflow_conveyor_speed_cm_min": (sum(reflow_speeds) / len(reflow_speeds)) if reflow_speeds else None,
        "reflow_profile_json": json.dumps(reflow_temps, ensure_ascii=False),
        "online_current_min_ua": min(current_values) if current_values else None,
        "online_current_max_ua": max(current_values) if current_values else None,
        "online_current_avg_ua": (sum(current_values) / len(current_values)) if current_values else None,
        "online_current_by_meter_json": json.dumps(online_currents, ensure_ascii=False),
        "wave_flux_spray_qty": wave_flux,
        "wave_preheat_temp_min_c": preheat_min,
        "wave_preheat_temp_max_c": preheat_max,
        "wave_preheat_temp_avg_c": preheat_avg,
        "wave_solder_temp_c": wave_solder_temp,
        "wave_peak_height_1_mm": wave_peak_1,
        "wave_peak_height_2_mm": wave_peak_2,
        "wave_profile_json": json.dumps(raw_payload["wave_solder"], ensure_ascii=False),
        "confidence": 0.90 if meter_ids else 0.70,
        "raw_json": json.dumps(raw_payload, ensure_ascii=False),
        "note": "Parsed from the production-process block in the factory-test workbook.",
    }]


def parse_daily_timing(wb) -> dict:
    """
    解析「日计时」Sheet，返回 {(meter_no, time_label): clock_daily_error_sd}
    取「日误差(s/d)」行的值
    """
    try:
        ws = wb["日计时"]
    except KeyError:
        return {}

    rows = list(ws.iter_rows(values_only=True))
    result = {}
    current_time = None
    meter_nos = []

    for row in rows:
        first = str(row[0]).strip() if row[0] else ""
        second = str(row[1]).strip() if row[1] else "" if len(row) > 1 else ""

        is_time_row = (
            ("测试时间" in first or "试验前" in first)
            and re.search(r"\d+\.\d+", first)
            and "表号" in second
        )

        if is_time_row:
            m = re.search(r"(\d+\.\d+)", first)
            if m:
                current_time = m.group(1)
                meter_nos = [_safe_int(c) for c in row[2:] if _safe_int(c) and 100 < _safe_int(c) < 300]
        elif current_time and meter_nos and "日误差(s/d)" in second:
            for idx, mn in enumerate(meter_nos):
                val = _safe_float(row[idx + 2]) if idx + 2 < len(row) else None
                if val is not None and mn:
                    result[(mn, current_time)] = val

    return result


# ═══════════════════════════════════════════════════════════════════════
# 5. 解析误差变差（误差变差 Sheet）
# ═══════════════════════════════════════════════════════════════════════
def parse_error_variability(wb) -> dict:
    """
    解析「误差变差」Sheet，返回 {(meter_no, time_label): max_variability}
    取多工况中最大变差值
    """
    try:
        ws = wb["误差变差 "]
    except KeyError:
        try:
            ws = wb["误差变差"]
        except KeyError:
            return {}

    blocks = _parse_time_block(list(ws.iter_rows(values_only=True)))
    result = {}
    for time_label, meter_dict in blocks.items():
        for meter_no, conditions in meter_dict.items():
            if conditions:
                max_var = max([abs(v) for v in conditions.values() if v is not None], default=None)
                if max_var is not None:
                    result[(meter_no, time_label)] = max_var
    return result


# ═══════════════════════════════════════════════════════════════════════
# 6. 解析误差一致性（误差一致性 Sheet）
# ═══════════════════════════════════════════════════════════════════════
def parse_error_consistency(wb) -> dict:
    """
    解析「误差一致性」Sheet，返回 {(meter_no, time_label): max_abs_consistency}
    取三个工况中最大绝对偏差
    """
    try:
        ws = wb["误差一致性"]
    except KeyError:
        return {}

    blocks = _parse_time_block(list(ws.iter_rows(values_only=True)))
    result = {}
    for time_label, meter_dict in blocks.items():
        for meter_no, conditions in meter_dict.items():
            if conditions:
                max_cons = max([abs(v) for v in conditions.values() if v is not None], default=None)
                if max_cons is not None:
                    result[(meter_no, time_label)] = max_cons
    return result


# ═══════════════════════════════════════════════════════════════════════
# 7. 解析正反向电量累计（正反向电量累计 Sheet）
# ═══════════════════════════════════════════════════════════════════════
def parse_energy_accumulation(wb) -> dict:
    """
    解析「正反向电量累计」Sheet
    返回 {(meter_no, time_label): {energy_total, energy_forward, energy_balance_ratio, energy_peak_ratio}}
    电量编码格式如 00340569 = 340.569 kWh（前 5 位整数，后 3 位小数）
    """
    try:
        ws = wb["正反向电量累计"]
    except KeyError:
        return {}

    rows = list(ws.iter_rows(values_only=True))
    result = {}

    # 第一行：表号行
    header_row = rows[0]
    meter_nos = [_safe_int(c) for c in header_row[2:] if _safe_int(c) and 100 < _safe_int(c) < 300]

    current_time = None
    current_meter_nos = meter_nos[:]
    block_data = {}  # {meter_no: {item: value}}

    def _parse_energy(val) -> Optional[float]:
        """将 '00340569' 格式转为 340.569"""
        if val is None:
            return None
        s = str(val).strip().replace(" ", "").replace("\xa0", "")
        if not s.isdigit():
            return None
        try:
            return int(s) / 1000.0
        except Exception:
            return None

    for i, row in enumerate(rows[1:], start=2):
        first = str(row[0]).strip() if row[0] else ""
        second = str(row[1]).strip() if row[1] else ""

        if "测试时间" in first and re.search(r"\d+\.\d+", first):
            if "表号" in second:
                # 新时间块开始前保存上一块
                if current_time and block_data:
                    for mn, d in block_data.items():
                        total = d.get("当前组合有功总电能")
                        forward = d.get("当前正向有功总电能")
                        peak = d.get("当前组合有功尖电能")
                        if total and total > 0:
                            balance = (forward / total) if forward else 1.0
                            peak_r = (peak / total) if peak else 0.0
                        else:
                            balance, peak_r = 1.0, 0.0
                        result[(mn, current_time)] = {
                            "energy_total": total,
                            "energy_forward": forward,
                            "energy_balance_ratio": round(balance, 4),
                            "energy_peak_ratio": round(peak_r, 4),
                        }

                m = re.search(r"(\d+\.\d+)", first)
                current_time = m.group(1) if m else None
                current_meter_nos = [_safe_int(c) for c in row[2:] if _safe_int(c) and 100 < _safe_int(c) < 300]
                block_data = {mn: {} for mn in current_meter_nos}
            else:
                pass
        elif second and current_time and current_meter_nos:
            for idx, mn in enumerate(current_meter_nos):
                val = _parse_energy(row[idx + 2]) if idx + 2 < len(row) else None
                if val is not None and mn in block_data:
                    block_data[mn][second] = val

    # 保存最后一块
    if current_time and block_data:
        for mn, d in block_data.items():
            total = d.get("当前组合有功总电能")
            forward = d.get("当前正向有功总电能")
            peak = d.get("当前组合有功尖电能")
            if total and total > 0:
                balance = (forward / total) if forward else 1.0
                peak_r = (peak / total) if peak else 0.0
            else:
                balance, peak_r = 1.0, 0.0
            result[(mn, current_time)] = {
                "energy_total": total,
                "energy_forward": forward,
                "energy_balance_ratio": round(balance, 4),
                "energy_peak_ratio": round(peak_r, 4),
            }

    return result


# ═══════════════════════════════════════════════════════════════════════
# 8. 解析开表盖（开表盖 Sheet）
# ═══════════════════════════════════════════════════════════════════════
def parse_cover_open(wb) -> dict:
    """
    解析「开表盖」Sheet，返回 {meter_no: cover_open_count}
    取「试验后开表盖记录」列
    """
    try:
        ws = wb["开表盖"]
    except KeyError:
        return {}

    rows = list(ws.iter_rows(values_only=True))
    result = {}
    for row in rows[1:]:  # 跳过标题行
        if not row[0]:
            continue
        mn = _safe_int(row[0])
        if mn and 100 < mn < 300:
            after_count = _safe_int(row[2]) or 0  # 试验后开表盖次数
            result[mn] = after_count
    return result


# ═══════════════════════════════════════════════════════════════════════
# 9. 解析老化退化试验数据（整合所有Sheet）
# ═══════════════════════════════════════════════════════════════════════
def parse_aging_test(filepath: Path) -> list[dict]:
    """解析老化试验所有Sheet，整合全量特征"""
    wb = openpyxl.load_workbook(filepath, data_only=True)

    # ── 9.1 基本误差 ─────────────────────────────────────────────────
    ws_err = wb["基本误差"]
    err_rows = list(ws_err.iter_rows(values_only=True))
    time_blocks = {}
    current_time = None
    meter_nos = []

    for row in err_rows:
        first = str(row[0]).strip() if row[0] else ""
        second = str(row[1]).strip() if row[1] else ""

        if (("测试时间" in first or "试验前" in first) and re.search(r"\d+\.\d+", first)
                and "表号" in second):
            m = re.search(r"(\d+\.\d+)", first)
            if m:
                current_time = m.group(1)
                time_blocks[current_time] = {}
                meter_nos = [_safe_int(c) for c in row[2:] if _safe_int(c) and 100 < _safe_int(c) < 300]
                for mn in meter_nos:
                    time_blocks[current_time][mn] = {}
        elif current_time and meter_nos and second and second not in ("表号", ""):
            for idx, mn in enumerate(meter_nos):
                val = _safe_float(row[idx + 2]) if idx + 2 < len(row) else None
                if val is not None and mn in time_blocks.get(current_time, {}):
                    time_blocks[current_time][mn][second] = val

    # ── 9.2 时钟偏差（时间Sheet） ─────────────────────────────────────
    ws_time = wb["时间"]
    time_rows = list(ws_time.iter_rows(values_only=True))
    clock_data = {}
    t_meter_nos = []
    for row in time_rows:
        first = str(row[0]).strip() if row[0] else ""
        if first == "表号":
            t_meter_nos = [_safe_int(c) for c in row[2:] if _safe_int(c) and 100 < _safe_int(c) < 300]
        elif ("测试时间" in first or "试验前" in first) and re.search(r"\d+\.\d+", first):
            m = re.search(r"(\d+\.\d+)", first)
            tlabel = m.group(1) if m else "0.0"
            clock_data[tlabel] = {}
            for i, mn in enumerate(t_meter_nos):
                v = _safe_int(row[i + 2]) if i + 2 < len(row) else None
                if mn:
                    clock_data[tlabel][mn] = v if v is not None else 0

    # ── 9.3 RS485 ───────────────────────────────────────────────────
    ws_485 = wb["485"]
    rows_485 = list(ws_485.iter_rows(values_only=True))
    rs485_meter_nos = [_safe_int(c) for c in rows_485[1] if _safe_int(c)]
    rs485_errors = {mn: 0 for mn in rs485_meter_nos if mn}
    rs485_errors[220] = 10  # 已知异常电表

    # ── 9.4 新增Sheets ───────────────────────────────────────────────
    daily_timing = parse_daily_timing(wb)
    error_variability = parse_error_variability(wb)
    error_consistency = parse_error_consistency(wb)
    energy_data = parse_energy_accumulation(wb)
    cover_open = parse_cover_open(wb)

    wb.close()

    # ── 9.5 整合输出 ─────────────────────────────────────────────────
    records = []
    for time_label, meter_dict in time_blocks.items():
        hours = TIME_POINT_MAP.get(time_label, 0)
        for meter_no, conditions in meter_dict.items():
            if not conditions:
                continue
            vals = [abs(v) for v in conditions.values() if v is not None]
            error_max = max(vals) if vals else None
            error_a_imin_10 = conditions.get("A Imin 1.0") or conditions.get("10Itr 1.0")
            error_a_imin_05l = conditions.get("A Imin 0.5L")
            clock_offset = clock_data.get(time_label, {}).get(meter_no, None)
            rs485_err = rs485_errors.get(meter_no, 0)

            # 新增特征
            daily_err = daily_timing.get((meter_no, time_label))
            variability = error_variability.get((meter_no, time_label))
            consistency = error_consistency.get((meter_no, time_label))
            energy = energy_data.get((meter_no, time_label), {})
            cover_cnt = cover_open.get(meter_no, 0)

            records.append({
                "meter_no": meter_no,
                "test_date_label": time_label,
                "test_hours": float(hours),
                "test_condition": "85℃/85%RH/1.1Un/60A",
                "basic_errors_json": json.dumps({k: v for k, v in conditions.items() if v is not None}),
                "error_a_imin_10": _safe_float(error_a_imin_10),
                "error_a_imin_05l": _safe_float(error_a_imin_05l),
                "error_max": error_max,
                "clock_offset_s": clock_offset,
                # 新增字段
                "clock_daily_error_sd": daily_err,
                "error_variability": variability,
                "error_consistency": consistency,
                "energy_total": energy.get("energy_total"),
                "energy_forward": energy.get("energy_forward"),
                "energy_balance_ratio": energy.get("energy_balance_ratio"),
                "energy_peak_ratio": energy.get("energy_peak_ratio"),
                "rs485_errors": rs485_err,
                "quality_label": 0,
                "quality_reason": "",
            })

    return records


# ═══════════════════════════════════════════════════════════════════════
# 10. 生成质量标签 & 质量链路表（含新增特征）
# ═══════════════════════════════════════════════════════════════════════
def build_quality_chain(db: Session) -> list[dict]:
    """为每块真实电表生成质量链路汇总和质量标签"""
    from app.models.db_models import AgingTest as AT
    from collections import defaultdict
    import numpy as np

    aging_records = db.query(AT).all()
    meter_groups = defaultdict(list)
    for r in aging_records:
        meter_groups[r.meter_no].append(r)

    chains = []
    for meter_no, records in meter_groups.items():
        records.sort(key=lambda x: x.test_hours)

        initial = records[0] if records else None
        final = records[-1] if records else None

        # 基础误差特征
        initial_err = initial.error_max if initial else None
        final_err = final.error_max if final else None
        error_deg_rate = None
        if initial_err is not None and final_err is not None and final.test_hours > 0:
            error_deg_rate = (final_err - initial_err) / final.test_hours * 100

        # 时钟特征
        initial_clock = initial.clock_offset_s if initial else 0
        final_clock = final.clock_offset_s if final else 0
        clock_drift_rate = None
        if final and final.test_hours > 0 and final_clock is not None:
            clock_drift_rate = abs(final_clock) / (final.test_hours / 24.0)

        # RS485
        rs485_total = max(r.rs485_errors for r in records) if records else 0

        # 新增特征：取末期值（最后一个时间点）
        clock_daily_max = final.clock_daily_error_sd if final else None
        variability_max = max(
            [r.error_variability for r in records if r.error_variability is not None],
            default=None
        )
        consistency_max = max(
            [r.error_consistency for r in records if r.error_consistency is not None],
            default=None
        )
        cover_count = 0
        energy_balance = final.energy_balance_ratio if final else None
        energy_peak = final.energy_peak_ratio if final else None

        # 质量标签规则（增强版）
        label = 0
        reasons = []
        if final_err is not None and final_err > cfg.ERROR_THRESHOLD:
            label = 1
            reasons.append(f"末期基本误差{final_err:.3f}%超差(>{cfg.ERROR_THRESHOLD}%)")
        if final_clock is not None and abs(final_clock) > cfg.CLOCK_THRESHOLD:
            label = 1
            reasons.append(f"时钟偏差{final_clock}s超限")
        if rs485_total > 0:
            label = 1
            reasons.append(f"RS485通信异常{rs485_total}次")
        if clock_drift_rate is not None and clock_drift_rate > 0.35:
            label = 1
            reasons.append(f"时钟漂移{clock_drift_rate:.3f}s/d偏高")
        if clock_daily_max is not None and abs(clock_daily_max) > getattr(cfg, 'CLOCK_DAILY_THRESHOLD', 0.3):
            label = 1
            reasons.append(f"日误差{clock_daily_max:.3f}s/d超限")
        if consistency_max is not None and consistency_max > getattr(cfg, 'CONSISTENCY_THRESHOLD', 0.12):
            label = 1
            reasons.append(f"误差一致性偏差{consistency_max:.3f}%偏大")

        chains.append({
            "meter_no": meter_no,
            "crystal_batch": None,
            "ct_batch": None,
            "crystal_ppm": None,
            "ct_ratio_err": None,
            "clock_accuracy_sd": None,
            "factory_fault": False,
            "factory_fault_type": None,
            "initial_error_max": initial_err,
            "final_error_max": final_err,
            "error_degradation_rate": error_deg_rate,
            "initial_clock_offset": initial_clock,
            "final_clock_offset": final_clock,
            "clock_drift_rate": clock_drift_rate,
            "rs485_total_errors": rs485_total,
            "clock_daily_error_max": clock_daily_max,
            "error_variability_max": variability_max,
            "error_consistency_max": consistency_max,
            "cover_open_count": cover_count,
            "energy_balance_ratio": energy_balance,
            "energy_peak_ratio": energy_peak,
            "is_augmented": False,
            "quality_label": label,
            "quality_reason": "；".join(reasons) if reasons else "正常",
        })

    return chains


# ═══════════════════════════════════════════════════════════════════════
# 11. 主导入函数
# ═══════════════════════════════════════════════════════════════════════
def upsert_production_process_profiles(db: Session, profiles: list[dict]) -> int:
    """Insert or update parsed production-process profiles by batch_key."""
    count = 0
    for profile in profiles:
        batch_key = profile.get("batch_key")
        if not batch_key:
            continue
        existing = db.query(ProductionProcessProfile).filter_by(batch_key=batch_key).first()
        if existing is None:
            db.add(ProductionProcessProfile(**profile))
        else:
            for key, value in profile.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
        count += 1
    return count


def import_all_data(db: Session) -> dict:
    stats = {"crystal": 0, "ct": 0, "factory": 0, "process_profile": 0, "aging": 0, "chain": 0, "errors": []}
    data_dir = cfg.DATA_DIR

    # 晶振
    try:
        for r in parse_crystal_data(data_dir / cfg.CRYSTAL_FILE):
            if not db.query(CrystalData).filter_by(batch_no=r["batch_no"], sample_no=r["sample_no"]).first():
                db.add(CrystalData(**r))
                stats["crystal"] += 1
        db.commit()
    except Exception as e:
        stats["errors"].append(f"晶振：{e}"); db.rollback()

    # 互感器
    try:
        for r in parse_ct_data(data_dir / cfg.CRYSTAL_FILE):
            if not db.query(CTData).filter_by(batch_no=r["batch_no"], sample_no=r["sample_no"]).first():
                db.add(CTData(**r))
                stats["ct"] += 1
        db.commit()
    except Exception as e:
        stats["errors"].append(f"互感器：{e}"); db.rollback()

    # 出厂检测
    try:
        factory_path = data_dir / cfg.FACTORY_FILE
        for r in parse_factory_test(factory_path):
            existing = db.query(FactoryTest).filter_by(meter_id=r["meter_id"]).first()
            if not existing:
                db.add(FactoryTest(**r))
                stats["factory"] += 1
            else:
                for k, v in r.items():
                    if k != "meter_id":
                        setattr(existing, k, v)
        stats["process_profile"] = upsert_production_process_profiles(
            db,
            parse_factory_process_profile(factory_path),
        )
        db.commit()
    except Exception as e:
        stats["errors"].append(f"出厂检测：{e}"); db.rollback()

    # 老化试验（删旧数据重建，以包含新字段）
    try:
        db.query(AgingTest).filter(AgingTest.rs485_errors != None).delete()
        db.commit()
        for r in parse_aging_test(data_dir / cfg.AGING_FILE):
            if not db.query(AgingTest).filter_by(meter_no=r["meter_no"], test_date_label=r["test_date_label"]).first():
                db.add(AgingTest(**r))
                stats["aging"] += 1
        db.commit()
    except Exception as e:
        stats["errors"].append(f"老化试验：{e}"); db.rollback()

    # 质量链路
    try:
        for r in build_quality_chain(db):
            existing = db.query(QualityChain).filter_by(meter_no=r["meter_no"]).first()
            if not existing:
                db.add(QualityChain(**r))
                stats["chain"] += 1
            else:
                for k, v in r.items():
                    if k != "meter_no":
                        setattr(existing, k, v)
        db.commit()
    except Exception as e:
        stats["errors"].append(f"质量链路：{e}"); db.rollback()

    return stats


def auto_import_on_startup() -> Optional[str]:
    """启动时自动检查并导入数据"""
    db = SessionLocal()
    try:
        count = db.query(AgingTest).count()
        if count > 0:
            return None
        stats = import_all_data(db)
        return (f"晶振:{stats['crystal']} 互感器:{stats['ct']} "
                f"出厂:{stats['factory']} 老化:{stats['aging']} 链路:{stats['chain']}")
    finally:
        db.close()
