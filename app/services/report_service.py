"""Generate auditable quality-risk analysis reports for production management."""
from __future__ import annotations

from collections import Counter
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
LIGHT_YELLOW = "FFF2CC"


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


def _risk_level(probability: float | None) -> str:
    p = probability or 0.0
    if p >= 0.7:
        return "高风险"
    if p >= 0.4:
        return "中风险"
    return "低风险"


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


def generate_quality_analysis_report(db) -> tuple[bytes, str]:
    """Create a management-facing Word report from current prediction results."""
    from app.models.db_models import BatchQualityProfile, ModelVersion, QualityChain, QualityFeatureSnapshot

    chains = db.query(QualityChain).order_by(QualityChain.meter_no).all()
    model = db.query(ModelVersion).filter_by(is_active=True).order_by(ModelVersion.created_at.desc()).first()
    snapshots = db.query(QualityFeatureSnapshot).all()
    batch_profiles = db.query(BatchQualityProfile).all()

    ranked = sorted(chains, key=lambda item: item.predicted_risk or 0.0, reverse=True)
    high = [item for item in ranked if (item.predicted_risk or 0.0) >= 0.7]
    medium = [item for item in ranked if 0.4 <= (item.predicted_risk or 0.0) < 0.7]
    low = [item for item in ranked if (item.predicted_risk or 0.0) < 0.4]
    source_counts = Counter(item.source or "unknown" for item in snapshots)
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
    _set_font(title.add_run("智能电表质量风险分析报告"), 18, bold=True, color=BLUE)

    meta = doc.add_table(rows=3, cols=4)
    meta.style = "Table Grid"
    meta_data = [
        ("报告编号", f"QRA-{now.strftime('%Y%m%d-%H%M%S')}", "生成时间", now.strftime("%Y-%m-%d %H:%M")),
        ("分析对象", "当前已完成预测的质量链路数据", "分析样本数", f"{len(chains)} 条"),
        ("当前模型", model.version_tag if model else "规则引擎（未训练模型）", "报告状态", "系统自动生成，待质量负责人审核"),
    ]
    for row, values in zip(meta.rows, meta_data):
        for idx, value in enumerate(values):
            _set_cell(row.cells[idx], value, bold=idx % 2 == 0, fill=LIGHT_BLUE if idx % 2 == 0 else None)

    _heading(doc, "一、分析结果概览")
    result_table = doc.add_table(rows=1, cols=4)
    result_table.style = "Table Grid"
    headers = ["高风险", "中风险", "低风险", "高风险占比"]
    values = [str(len(high)), str(len(medium)), str(len(low)), f"{(len(high) / len(chains) * 100) if chains else 0:.1f}%"]
    for cell, value in zip(result_table.rows[0].cells, headers):
        _set_cell(cell, value, bold=True, fill=LIGHT_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER)
    row = result_table.add_row()
    for cell, value in zip(row.cells, values):
        _set_cell(cell, value, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    _paragraph(doc, "处理顺序：先处理高风险，再处理中风险；低风险按常规抽检执行。")

    _heading(doc, "二、高风险处置清单")
    if not high:
        _paragraph(doc, "暂无预测为高风险的记录。")
    else:
        risk_table = doc.add_table(rows=1, cols=5)
        risk_table.style = "Table Grid"
        risk_headers = ["序号", "电表编号", "风险概率", "发现的问题", "立即处理"]
        for cell, value in zip(risk_table.rows[0].cells, risk_headers):
            _set_cell(cell, value, bold=True, fill=LIGHT_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER)
        for index, chain in enumerate(high[:12], start=1):
            row = risk_table.add_row()
            values = [
                str(index),
                str(chain.meter_no),
                f"{(chain.predicted_risk or 0) * 100:.2f}%",
                chain.quality_reason or "系统未记录明确线索",
                _action_for_chain(chain),
            ]
            for col, value in enumerate(values):
                _set_cell(row.cells[col], value, fill=LIGHT_RED if col == 2 else None)
        if len(high) > 12:
            _paragraph(doc, f"注：高风险共 {len(high)} 条，本页展示风险最高的前12条。其余记录请在系统风险列表中查看。")

    _heading(doc, "三、审核结论")
    _paragraph(doc, "请根据复测结果勾选：□ 高风险记录已复测并完成处置  □ 需扩大同批抽检  □ 需转质量工程师进一步分析")
    _paragraph(doc, "说明：本报告用于安排复查和处置，不替代检验结论。")
    if source_counts.get("sampled_aging_unlinked", 0):
        _paragraph(doc, "本次含抽检老化数据，其结果用于提示批次风险，不能直接作为单表最终结论。")

    _heading(doc, "四、签核")
    sign = doc.add_table(rows=2, cols=3)
    sign.style = "Table Grid"
    for cell, value in zip(sign.rows[0].cells, ["编制", "复核", "批准"]):
        _set_cell(cell, value, bold=True, fill=LIGHT_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER)
    for cell in sign.rows[1].cells:
        _set_cell(cell, "签字：                 日期：", align=WD_ALIGN_PARAGRAPH.LEFT)

    output = BytesIO()
    doc.save(output)
    filename = f"智能电表质量风险分析报告_{now.strftime('%Y%m%d_%H%M%S')}.docx"
    return output.getvalue(), filename
