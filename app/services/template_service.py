"""标准 Excel 模板生成服务。"""
from __future__ import annotations

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

TEMPLATE_DEFINITIONS = {
    "meter_base": {
        "type": "meter_base",
        "name": "电表基础信息模板",
        "filename": "01_电表基础信息模板.xlsx",
        "description": "用于建立电表编号、条形码、订单和生产批次的基础映射。",
        "headers": ["meter_id", "barcode", "order_no", "batch_no", "product_model", "production_date"],
        "example": ["M202604280001", "ATXM202604280001", "DDH20260401", "BATCH202604", "DDSYxxx", "2026-04-28"],
    },
    "component_binding": {
        "type": "component_binding",
        "name": "元器件绑定关系模板",
        "filename": "02_元器件绑定关系模板.xlsx",
        "description": "用于说明每块电表使用的晶振、互感器、PCB、芯片批次。",
        "headers": ["meter_id", "barcode", "crystal_batch", "ct_batch", "pcb_batch", "chip_batch"],
        "example": ["M202604280001", "ATXM202604280001", "X202604A", "CT202604B", "PCB202604", "CHIP202604"],
    },
    "crystal": {
        "type": "crystal",
        "name": "晶振检测数据模板",
        "filename": "03_晶振检测数据模板.xlsx",
        "description": "用于导入晶振来料检测参数。",
        "headers": ["batch_no", "sample_no", "model", "center_freq_hz", "freq_error_ppm", "load_cap_pf", "is_qualified", "supplier"],
        "example": ["X202604A", 1, "HC-49SMD", 32768, 8.5, 12.5, "是", "供应商A"],
    },
    "ct": {
        "type": "ct",
        "name": "互感器检测数据模板",
        "filename": "04_互感器检测数据模板.xlsx",
        "description": "用于导入互感器来料检测参数。",
        "headers": ["batch_no", "sample_no", "model", "ratio_error_max", "ratio_error_range", "angle_error_max", "dc_resistance", "is_qualified", "supplier"],
        "example": ["CT202604B", 1, "HDCT3 5(60)A2mA", 0.35, 0.12, 8.0, 0.02, "是", "供应商B"],
    },
    "production": {
        "type": "production",
        "name": "生产过程检测数据模板",
        "filename": "05_生产过程检测数据模板.xlsx",
        "description": "用于导入 CCD、耐压、功能测试、复校等工序检测记录。",
        "headers": ["barcode", "order_no", "process", "test_count", "result", "test_time", "device_no", "station_no", "remark", "leak_current"],
        "example": ["ATXM202604280001", "DDH20260401", "功能测试", 1, "合格", "2026-04-28 10:30:00", "DEV-01", "S01", "", 0.12],
    },
    "aoi": {
        "type": "aoi",
        "name": "AOI缺陷数据模板",
        "filename": "06_AOI缺陷数据模板.xlsx",
        "description": "用于导入贴片 AOI、波峰焊 AOI 缺陷记录。",
        "headers": ["barcode", "order_no", "aoi_type", "defect_point", "defect_desc", "process_no", "detect_time", "device_no"],
        "example": ["ATXM202604280001", "DDH20260401", "smt", "R12", "虚焊", "贴片AOI", "2026-04-28 09:20:00", "AOI-01"],
    },
    "repair": {
        "type": "repair",
        "name": "维修记录数据模板",
        "filename": "07_维修记录数据模板.xlsx",
        "description": "用于导入故障维修闭环记录。",
        "headers": ["barcode", "order_no", "fault_name", "cause_name", "fault_location", "fault_process", "fault_time", "repair_time", "repair_result"],
        "example": ["ATXM202604280001", "DDH20260401", "通信异常", "焊接不良", "通信模块", "功能测试", "2026-04-28 11:00:00", "2026-04-28 11:30:00", "已修复"],
    },
    "factory": {
        "type": "factory",
        "name": "出厂检测数据模板",
        "filename": "08_出厂检测数据模板.xlsx",
        "description": "用于导入出厂检测结果，是出厂故障预测模型的标签来源。",
        "headers": ["meter_id", "barcode", "test_time", "is_fault", "fault_type", "final_result", "meter_constant_err", "clock_accuracy_sd", "rs485_ok", "ir_ok", "module_ok", "startup_ok", "creep_ok"],
        "example": ["M202604280001", "ATXM202604280001", "2026-04-28 14:00:00", "否", "正常", "合格", 0.02, 0.018, "是", "是", "是", "是", "是"],
    },
    "aging": {
        "type": "aging",
        "name": "1000h老化测试数据模板",
        "filename": "09_1000h老化测试数据模板.xlsx",
        "description": "用于导入抽检电表的 1000h 加压老化测试数据。",
        "headers": ["meter_id", "barcode", "test_date_label", "test_hours", "test_condition", "error_max", "clock_offset_s", "clock_daily_error_sd", "rs485_errors", "energy_total", "energy_balance_ratio", "energy_peak_ratio", "quality_label", "quality_reason"],
        "example": ["M202604280001", "ATXM202604280001", "1000h", 1000, "85℃/85%RH", 0.32, 5, 0.12, 0, 123.4, 1.01, 0.24, 0, "正常"],
    },
}


def build_template_workbook(template_type: str) -> tuple[Workbook, str]:
    if template_type not in TEMPLATE_DEFINITIONS:
        template_type = "factory"
    definition = TEMPLATE_DEFINITIONS[template_type]
    wb = Workbook()
    ws = wb.active
    ws.title = "数据模板"

    title = definition["name"]
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(definition["headers"]))
    ws.cell(row=1, column=1, value=title)
    ws.cell(row=1, column=1).font = Font(bold=True, size=14, color="FFFFFF")
    ws.cell(row=1, column=1).fill = PatternFill("solid", fgColor="1A3A5C")
    ws.cell(row=1, column=1).alignment = Alignment(horizontal="center")

    ws.cell(row=2, column=1, value=definition["description"])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(definition["headers"]))

    header_fill = PatternFill("solid", fgColor="F0A500")
    for col, header in enumerate(definition["headers"], start=1):
        cell = ws.cell(row=4, column=col, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        ws.cell(row=5, column=col, value=definition["example"][col - 1])
        ws.column_dimensions[get_column_letter(col)].width = max(len(str(header)) + 4, 16)

    note_ws = wb.create_sheet("填写说明")
    note_ws.append(["字段", "说明"])
    for cell in note_ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1A3A5C")
    for header in definition["headers"]:
        note_ws.append([header, "请按字段含义填写；日期时间建议使用 YYYY-MM-DD HH:MM:SS 格式"])
    note_ws.column_dimensions["A"].width = 28
    note_ws.column_dimensions["B"].width = 70
    return wb, definition["filename"]
