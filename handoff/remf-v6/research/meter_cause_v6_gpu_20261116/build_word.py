"""Build the report with the Codex bundled Python runtime."""
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT=Path(__file__).resolve().parent
doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.7)
sec.left_margin=sec.right_margin=Inches(.72)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    style=doc.styles[name]
    style.font.name='Microsoft YaHei'
    style._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    style.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(10)
doc.styles['Normal'].paragraph_format.line_spacing=1.08
doc.styles['Normal'].paragraph_format.space_after=Pt(5)
for name,size in [('Title',21),('Heading 1',15),('Heading 2',12)]:
    doc.styles[name].font.size=Pt(size)
    doc.styles[name].paragraph_format.space_before=Pt(12)
    doc.styles[name].paragraph_format.space_after=Pt(6)

head=sec.header.paragraphs[0]
head.text='电表根因分析实验  /  v6 样本'
head.style=doc.styles['Normal']; head.runs[0].font.size=Pt(8)
foot=sec.footer.paragraphs[0]; foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
foot.add_run('训练与测试报告  ·  ')
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); foot._p.append(field)
for r in foot.runs:r.font.size=Pt(8)

def table(lines):
    parsed=[[x.strip() for x in l.strip().strip('|').split('|')] for l in lines]
    parsed=[r for r in parsed if not all(re.fullmatch('[: -]+',c or '-') for c in r)]
    t=doc.add_table(rows=0,cols=len(parsed[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    n=len(parsed[0]); total=7.06
    widths=[total/n]*n
    if n>=4:widths=[2.0]+[(total-2)/(n-1)]*(n-1)
    for col,w in zip(t.columns,widths):col.width=Inches(w)
    for i,values in enumerate(parsed):
        cells=t.add_row().cells
        for j,(cell,value) in enumerate(zip(cells,values)):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text=value
            for p in cell.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                p.paragraph_format.line_spacing=1.05
                for r in p.runs:r.font.size=Pt(9);r.bold=i==0
            pr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'60');el.set(qn('w:type'),'dxa');margins.append(el)
            pr.append(margins)
            if i==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EEEEEE');pr.append(shade)
        rowpr=t.rows[-1]._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit');rowpr.append(cant)
        if i==0:rowpr.append(OxmlElement('w:tblHeader'))
    borders=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
    t._tbl.tblPr.append(borders)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)

lines=(ROOT/'模型训练与测试报告.md').read_text(encoding='utf-8').splitlines()
i=0
while i<len(lines):
    s=lines[i].strip()
    if not s:i+=1;continue
    if s.startswith('|'):
        block=[]
        while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
        table(block);continue
    if s.startswith('!['):
        path=re.search(r'\]\((.+)\)',s).group(1)
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        width=5.5 if '置信度' in path else 6.4
        p.add_run().add_picture(str(ROOT/path),width=Inches(width))
        p.paragraph_format.space_after=Pt(4)
    elif s.startswith('# '):doc.add_paragraph(s[2:].replace(' GPU ',' GPU '),style='Title')
    elif s.startswith('### '):doc.add_heading(s[4:],level=2)
    elif s.startswith('## '):doc.add_heading(s[3:],level=1)
    else:
        # Keep paths readable without Markdown syntax.
        s=s.replace('`','').replace('**','')
        doc.add_paragraph(s)
    i+=1
for element in list(doc.styles.element.iter())+list(doc.element.iter()):
    if element.tag==qn('w:pBdr'):
        element.getparent().remove(element)
out=ROOT/'电表根因模型训练与测试报告.docx'
doc.save(out)
print(out)
