# -*- coding: utf-8 -*-
import csv, io, zipfile
from openpyxl import load_workbook
from .schema import ALIASES, FIELDS, FEATURES, OBS, clean_row

def mapped_headers(header):
    keys=[ALIASES.get(str(h).strip(),str(h).strip()) if h is not None else '' for h in header]
    used=[k for k in keys if k]
    if len(used)!=len(set(used)):raise ValueError('存在重复列标题或中英文重复映射列')
    return keys

def worksheet_rows(ws):
    iterator=ws.iter_rows(values_only=True)
    keys=mapped_headers(next(iterator,()))
    if 'sample_id' not in keys:raise ValueError(ws.title+'缺少电表编号列')
    rows=[]
    for number, vals in enumerate(iterator,2):
        if not any(v is not None and v!='' for v in vals):continue
        if len(rows)>=20000:raise ValueError('单次最多导入 20000 台')
        row={k:v for k,v in zip(keys,vals) if k}
        row['_line']=f'{ws.title} 第 {number} 行';rows.append(row)
    return rows, keys

def parse_file(data, name):
    if len(data)>30*1024*1024:raise ValueError('文件超过 30 MB 上限')
    if name.lower().endswith('.csv'):
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:raise ValueError('CSV 请保存为 UTF-8 编码')
        reader=csv.reader(io.StringIO(text));keys=mapped_headers(next(reader,[]));rows=[]
        for number, values in enumerate(reader,2):
            if not any(values):continue
            if len(rows)>=20000:raise ValueError('单次最多导入 20000 台')
            if len(values)!=len(keys):raise ValueError(f'第 {number} 行列数与表头不一致')
            rows.append({**dict(zip(keys,values)),'_line':f'第 {number} 行'})
        return rows, ['CSV 总表'], keys
    if not name.lower().endswith('.xlsx'):raise ValueError('仅支持 .xlsx 或 UTF-8 .csv 文件')
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if len(z.infolist())>2000 or sum(i.file_size for i in z.infolist())>250*1024*1024:
                raise ValueError('Excel 解压大小超过安全处理上限')
        wb=load_workbook(io.BytesIO(data),read_only=True,data_only=False,keep_links=False)
    except (zipfile.BadZipFile, KeyError, OSError) as exc:raise ValueError('无法解析 Excel 文件，请检查文件是否损坏') from exc
    try:
        if '生产与出厂总表' in wb.sheetnames:
            rows,keys=worksheet_rows(wb['生产与出厂总表'])
            return rows,['生产与出厂总表'],keys
        # A flat template may have a user-selected sheet name. Never accept post-test evidence sheets.
        forbidden=('复测','维修','确认','诊断检查','审计','真值','潜在')
        split_names=['样本主表','model_inputs','制造参数','出厂检测']
        split_present=sum(name in wb.sheetnames for name in split_names)>1
        for ws in wb:
            if any(word in ws.title for word in forbidden):continue
            if split_present and ws.title in split_names:continue
            head=mapped_headers(next(ws.iter_rows(max_row=1,values_only=True),()))
            if set(['sample_id','batch_id']+FEATURES)<=set(head):
                rows,keys=worksheet_rows(ws);return rows,[ws.title],keys
        # Split form: only explicitly named T0 tables are eligible for joining.
        merged={};sheets=[];allkeys=set()
        for name in split_names:
            if name not in wb.sheetnames:continue
            rows,keys=worksheet_rows(wb[name]);sheets.append(name);allkeys.update(keys);seen=set()
            for r in rows:
                sid=r.get('sample_id')
                if not isinstance(sid,str) or not sid.strip():raise ValueError(r['_line']+'：电表编号须为文本')
                sid=sid.strip()
                if sid in seen:raise ValueError(name+'存在重复编号：'+sid)
                seen.add(sid);previous=merged.setdefault(sid,{})
                for k,v in r.items():
                    if k not in FIELDS+['sample_id','batch_id','comm_trials','lcd_expected_segments']:continue
                    if k in previous and previous[k]!=v:raise ValueError(sid+'跨表字段冲突：'+k)
                    previous[k]=v
                previous['_line']=sid+'（按编号关联）'
        if not merged:raise ValueError('未找到完整生产与出厂总表，或样本主表 / model_inputs / 出厂检测工作表')
        if len(merged)>20000:raise ValueError('单次最多导入 20000 台')
        return list(merged.values()),sheets,sorted(allkeys)
    finally:wb.close()

def validate_rows(raw):
    records=[];errors=[];seen=set()
    for i,row in enumerate(raw,2):
        try:
            normalized={ALIASES.get(str(k).strip(),str(k).strip()):v for k,v in row.items()}
            manufacturing_only=all(normalized.get(k) is None or str(normalized.get(k)).strip()=='' for k in OBS)
            record=clean_row(row,manufacturing_only=manufacturing_only)
            sid=record['sample_id']
            if sid in seen:raise ValueError('文件内电表编号重复：'+sid)
            seen.add(sid);records.append(record)
        except ValueError as exc:
            errors.append({'row':row.get('_line',f'第 {i} 行'),'message':str(exc)})
    if not raw:errors.append({'row':'文件','message':'没有数据记录'})
    return records,errors
