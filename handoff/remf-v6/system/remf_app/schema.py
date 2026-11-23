# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'model_bundle'
META = json.loads((BUNDLE / 'preprocessing.json').read_text(encoding='utf-8'))
FEATURES = META['feature_columns']
OBS = META['observation_columns']
FIELDS = FEATURES + OBS
MANIFEST = json.loads((BUNDLE / 'workbook_manifest.json').read_text(encoding='utf-8'))
LABELS = dict(zip(MANIFEST[0]['fields'], MANIFEST[0]['headers']))
ALIASES = {**{k:k for k in LABELS}, **{v:k for k,v in LABELS.items()}}
ALIASES.update({'电表编号':'sample_id', '批次编号':'batch_id'})
GROUPS = [
    ('回流焊与装配', FEATURES[:12] + [FEATURES[29], FEATURES[30]]),
    ('CT 元件抽检', FEATURES[12:24]),
    ('计量校准', FEATURES[24:28]),
    ('清洁与来料', [FEATURES[28]] + FEATURES[31:]),
    ('出厂检测', OBS),
]
CAUSES = [
    dict(code='CT_CALIBRATION', name='CT 与计量校准不匹配', component='电流互感器 CT', stage='计量校准', family='元件与校准', check='核对逐台 CT 比差、角差与增益及相位补偿；复测不同电流和功率因数下的误差。'),
    dict(code='SAMPLING_SOLDER', name='采样回路焊接异常', component='采样电阻与连接焊点', stage='回流焊接', family='生产工艺', check='检查采样回路焊点润湿、虚焊及接触电阻；记录局部补焊前后的同工况复测结果。'),
    dict(code='PCB_RESIDUE', name='PCB 残留与漏电', component='PCB 高阻节点', stage='清洗与防护', family='生产工艺', check='复查板面清洁度、绝缘电阻及高阻节点污染；比较受控清洁前后的计量误差。'),
    dict(code='RS485_JOINT', name='RS485 连接焊接异常', component='RS485 接口焊点', stage='接口焊接', family='生产工艺', check='检查接口焊点与连接连续性，并在固定通信参数下复测，区分连接故障与器件故障。'),
    dict(code='LCD_CONTACT', name='LCD 接触与装配异常', component='LCD 连接触点', stage='显示装配', family='生产工艺', check='检查压接、触点和装配位置；记录缺段位置是否随接触复位改变。'),
    dict(code='POWER_SUPPLY', name='供电电路异常', component='电源模块与稳压器件', stage='电源来料与装联', family='元件与装联', check='测量供电电压、纹波和负载裕量；核对计量、通信与显示是否存在共同异常。'),
    dict(code='RS485_DEVICE', name='RS485 器件异常', component='RS485 收发器', stage='通信器件来料', family='元件', check='测量收发器供电及差分信号；在排除接口连接问题后进行替换复测。'),
    dict(code='LCD_DRIVER', name='LCD 驱动异常', component='LCD 驱动器件', stage='显示器件来料', family='元件', check='核对驱动输出、供电与段码映射；排除触点问题后进行器件替换验证。'),
]

def clean_row(raw, manufacturing_only=False):
    row = {}
    for key, value in raw.items():
        k = ALIASES.get(str(key).strip(), str(key).strip())
        if k in row and row[k] != value:
            raise ValueError('重复或冲突列：' + k)
        row[k] = value
    sid = row.get('sample_id')
    if not isinstance(sid, str) or not sid.strip() or len(sid.strip()) > 100:
        raise ValueError('电表编号必须为 1～100 位文本；Excel 请将编号列设为文本以保留前导零')
    bid = row.get('batch_id')
    if not isinstance(bid, str) or not bid.strip() or len(bid.strip()) > 100:
        raise ValueError('批次编号必须为 1～100 位文本')
    required = FEATURES if manufacturing_only else FIELDS
    missing = [LABELS.get(k,k) for k in required if row.get(k) is None or str(row[k]).strip()=='']
    if missing:
        raise ValueError('缺少必填数据：' + '、'.join(missing[:6]) + ('等' if len(missing)>6 else ''))
    values = {}
    for k in required:
        try:
            if isinstance(row[k], bool): raise ValueError()
            v = float(row[k])
            if not math.isfinite(v): raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(LABELS.get(k,k) + '必须为有限数值，不支持百分号文本或公式')
        if k in ['comm_failures','lcd_missing_segments']:
            cap = 200 if k=='comm_failures' else 42
            if v != int(v) or not 0 <= v <= cap:
                raise ValueError(LABELS[k] + f'必须是 0～{cap} 的整数')
        if (k.endswith('_proxy') or k=='aoi_wetting_fraction_sim') and not 0 <= v <= 1:
            raise ValueError(LABELS[k] + '必须在 0～1 之间')
        if k.startswith('reflow_zone_') and not 0 <= v <= 400:
            raise ValueError(LABELS[k] + '超出 0～400℃ 输入范围，请核对单位')
        if k=='conveyor_reading_sim_cm_min' and not 0 < v <= 1000:
            raise ValueError('输送速度必须大于 0 且不超过 1000 cm/min')
        if k=='cleaning_ionic_measurement_sim_ug_cm2' and v < 0:
            raise ValueError('离子测量值不能为负数')
        values[k] = v
    for key, expected in ([] if manufacturing_only else [('comm_trials',200),('lcd_expected_segments',42)]):
        if key in row and row[key] not in (None,''):
            try: valid = float(row[key])==expected
            except (TypeError,ValueError): valid = False
            if not valid: raise ValueError(f'{LABELS[key]}必须为 {expected}，否则不匹配当前模型测试协议')
    return {'sample_id':sid.strip(), 'batch_id':bid.strip(), 'values':values}

def inspect_values(values):
    if not all(k in values for k in OBS):
        return dict(ready=False, pending=True, failed=False, domains=[], checks=[])
    checks=[]
    for i,k in enumerate(OBS):
        limit = 1.5 if i<2 else 1.0 if i<10 else 10 if i==10 else 0
        value=values[k]
        fail=abs(value)>limit if i<10 else value>limit
        checks.append(dict(key=k, name=LABELS[k], value=value, limit=limit, failed=fail,
                           domain='计量' if i<10 else '通信' if i==10 else '显示', unit='%' if i<10 else '次' if i==10 else '段'))
    domains=list(dict.fromkeys(c['domain'] for c in checks if c['failed']))
    return dict(ready=True, pending=False, failed=bool(domains), domains=domains, checks=checks)

def schema():
    rp=BUNDLE/'risk_preprocessing.json'
    risk=json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {}
    def describe(k):
        out={'key':k,'label':LABELS[k]}
        if k in risk.get('features',[]):
            i=risk['features'].index(k)
            out['training_min']=risk['training_min'][i];out['training_max']=risk['training_max'][i]
        return out
    return {'groups':[{'name':name,'fields':[describe(k) for k in keys]} for name,keys in GROUPS],
            'features':34,'observations':12,'causes':CAUSES,
            'protocol':'首次计量误差限值：Imin ±1.5%，其余 ±1.0%；通信 200 次中失败超过 10 次；LCD 42 段中缺段 > 0。仅适用于本 V6 展示协议。'}
