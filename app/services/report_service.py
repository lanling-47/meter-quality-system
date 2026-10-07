"""Generate plain-language quality disposition reports for shop-floor operators."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


FONT = "Arial Unicode MS"
BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
LIGHT_RED = "FCE4D6"


def _set_font(run, size: float = 10.5, bold: bool = False, color: str | None = None):
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _shade(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell(cell, text: str, *, bold: bool = False, fill: str | None = None, align=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    _set_font(run, 9.5, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if fill:
        _shade(cell, fill)


def _heading(doc: Document, text: str, level: int = 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(text)
    _set_font(run, 13 if level == 1 else 11, bold=True, color=BLUE)
    return p


def _paragraph(doc: Document, text: str, *, bold_prefix: str | None = None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    if bold_prefix and text.startswith(bold_prefix):
        _set_font(p.add_run(bold_prefix), bold=True)
        _set_font(p.add_run(text[len(bold_prefix):]))
    else:
        _set_font(p.add_run(text))
    return p


def _effective_risk(chain) -> tuple[float, str]:
    """Prefer a model score, but never hide a rule-based quality alarm."""
    predicted = chain.predicted_risk
    if predicted is None:
        return (0.8, "规则预警") if chain.quality_label else (0.0, "正常")

    score = max(float(predicted), 0.7 if chain.quality_label else 0.0)
    return score, "模型预警" if chain.predicted_label is not None else "规则预警"


def _action_for_chain(chain) -> str:
    actions = []
    if (chain.rs485_total_errors or 0) > 0:
        actions.append("复测RS485通信；检查接口、通信模块及相关焊点")
    if (chain.final_error_max or 0) > 0.8 or (chain.error_consistency_max or 0) > 0.12:
        actions.append("复测计量误差并核查老化退化曲线")
    if (chain.clock_daily_error_max or 0) > 0.3 or (chain.clock_drift_rate or 0) > 0.3:
        actions.append("复测时钟准确度；核查晶振及其批次记录")
    if chain.factory_fault:
        actions.append("核对出厂故障、复校和维修闭环记录")
    return "；".join(actions) if actions else "复核原始检测记录；必要时安排抽检复测"


def _issue_for_chain(chain) -> str:
    if chain.quality_reason and chain.quality_reason != "正常":
        return chain.quality_reason.replace(";", "；").rstrip("；")
    if chain.factory_fault:
        return "出厂检测有故障记录"
    return "系统提示复核该表的检测记录"


def generate_quality_analysis_report(db) -> tuple[bytes, str]:
    """Create a one-click, plain-language disposition sheet for operators."""
    from app.models.db_models import ModelVersion, QualityChain

    chains = db.query(QualityChain).order_by(QualityChain.meter_no).all()
    model = db.query(ModelVersion).filter_by(is_active=True).order_by(ModelVersion.created_at.desc()).first()

    ranked = sorted(chains, key=lambda item: _effective_risk(item)[0], reverse=True)
    high = [item for item in ranked if _effective_risk(item)[0] >= 0.7]
    medium = [item for item in ranked if 0.4 <= _effective_risk(item)[0] < 0.7]
    low = [item for item in ranked if _effective_risk(item)[0] < 0.4]
    now = datetime.now()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(12)
    _set_font(title.add_run("智能电表现场质量处置单"), 18, bold=True, color=BLUE)

    meta = doc.add_table(rows=2, cols=4)
    meta.style = "Table Grid"
    meta_data = [
        ("处置单编号", f"QD-{now.strftime('%Y%m%d-%H%M%S')}", "生成时间", now.strftime("%Y-%m-%d %H:%M")),
        ("本次检查电表", f"{len(chains)} 块", "当前判定方式", model.version_tag if model else "规则预警（未训练模型）"),
    ]
    for row, values in zip(meta.rows, meta_data):
        for idx, value in enumerate(values):
            _set_cell(row.cells[idx], value, bold=idx % 2 == 0, fill=LIGHT_BLUE if idx % 2 == 0 else None)

    _heading(doc, "一、先看这里：本班需要做什么")
    result_table = doc.add_table(rows=1, cols=3)
    result_table.style = "Table Grid"
    headers = ["立即复测 / 处置", "安排复核", "可按常规抽检"]
    values = [f"{len(high)} 块", f"{len(medium)} 块", f"{len(low)} 块"]
    for cell, value in zip(result_table.rows[0].cells, headers):
        _set_cell(cell, value, bold=True, fill=LIGHT_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER)
    row = result_table.add_row()
    for cell, value in zip(row.cells, values):
        _set_cell(cell, value, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    _paragraph(doc, f"操作顺序：先处理下表中的 {len(high)} 块，再安排 {len(medium)} 块复核。处理完成后，在系统中复查记录。")

    _heading(doc, "二、立即处理清单")
    if not high:
        _paragraph(doc, "本次没有需要立即处置的电表。请按常规抽检继续生产。")
    else:
        risk_table = doc.add_table(rows=1, cols=4)
        risk_table.style = "Table Grid"
        risk_headers = ["序号", "电表编号", "发现的问题", "现在怎么做"]
        for cell, value in zip(risk_table.rows[0].cells, risk_headers):
            _set_cell(cell, value, bold=True, fill=LIGHT_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER)
        for index, chain in enumerate(high, start=1):
            row = risk_table.add_row()
            values = [
                str(index),
                str(chain.meter_no),
                _issue_for_chain(chain),
                _action_for_chain(chain),
            ]
            for col, value in enumerate(values):
                _set_cell(row.cells[col], value, fill=LIGHT_RED if col in (1, 2) else None)

    _heading(doc, "三、处理完成后")
    _paragraph(doc, "□ 已完成复测  □ 已隔离不合格品  □ 已通知班组长 / 质量人员  □ 已记录处理结果")
    _paragraph(doc, "说明：本处置单用于安排复测和隔离，不替代最终检验结论。")

    _heading(doc, "四、签字确认")
    sign = doc.add_table(rows=2, cols=3)
    sign.style = "Table Grid"
    for cell, value in zip(sign.rows[0].cells, ["编制", "复核", "批准"]):
        _set_cell(cell, value, bold=True, fill=LIGHT_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER)
    for cell in sign.rows[1].cells:
        _set_cell(cell, "签字：                 日期：", align=WD_ALIGN_PARAGRAPH.LEFT)

    output = BytesIO()
    doc.save(output)
    filename = f"智能电表现场质量处置单_{now.strftime('%Y%m%d_%H%M%S')}.docx"
    return output.getvalue(), filename
