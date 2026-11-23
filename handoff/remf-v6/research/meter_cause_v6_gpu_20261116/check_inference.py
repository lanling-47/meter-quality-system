"""Verify the delivered inference boundary using actual exported CSVs."""
from pathlib import Path
import json,subprocess,sys
import pandas as pd
from data import SOURCE,dump

ROOT=Path(__file__).resolve().parent
folder=ROOT/'interface_check';folder.mkdir(exist_ok=True)
meta=json.loads((ROOT/'preprocessing.json').read_text(encoding='utf-8'))
base=pd.read_csv(SOURCE/'training_dataset_10000.csv',encoding='utf-8-sig')
obs=pd.read_csv(SOURCE/'出厂检测.csv',encoding='utf-8-sig').set_index('sample_id')
examples=pd.concat([base[(base.split=='test')&(base.y_any_fail==0)].head(2),base[(base.split=='test')&(base.y_any_fail==1)].head(4)]).copy()
f=examples[['sample_id']+meta['feature_columns']].copy()
for col in meta['observation_columns']:f[col]=obs.reindex(f.sample_id)[col].to_numpy()
f.to_csv(folder/'input.csv',index=False,encoding='utf-8-sig')
perturbed=f.copy();perturbed['confirmed_cause_code']='LCD_DRIVER';perturbed['repair_target']='POWER_SUPPLY';perturbed['repair_success']=True
perturbed.to_csv(folder/'future_fields.csv',index=False,encoding='utf-8-sig')
f.drop(columns=[meta['feature_columns'][0]]).to_csv(folder/'missing.csv',index=False,encoding='utf-8-sig')
checks=[]
selection=json.loads((ROOT/'selection_frozen.json').read_text(encoding='utf-8'))
for model in dict.fromkeys([selection['selected_model'],selection['selected_neural']]):
    for source in ['input','future_fields']:
        subprocess.run([sys.executable,str(ROOT/'predict.py'),str(folder/(source+'.csv')),str(folder/(model+'_'+source+'.csv')),'--model',model],check=True)
    a=pd.read_csv(folder/(model+'_input.csv'));b=pd.read_csv(folder/(model+'_future_fields.csv'))
    pd.testing.assert_frame_equal(a,b)
    checks.append({'check':model+' future confirmation/repair fields do not affect inference','passed':True})
    assert a.iloc[:2].status.eq('首次检测通过不强制归因').all()
    assert a.iloc[:2][['top1','top2','top3']].isna().all().all()
    assert a.iloc[2:].top1.notna().all()
    checks.append({'check':model+' normal meters not forced into causes; abnormal meters ranked','passed':True})
    invalid=subprocess.run([sys.executable,str(ROOT/'predict.py'),str(folder/'missing.csv'),str(folder/'should_not_exist.csv'),'--model',model],capture_output=True,text=True,encoding='utf-8')
    assert invalid.returncode!=0 and '缺少首次诊断所需字段' in invalid.stderr
    checks.append({'check':model+' missing required fields rejected','passed':True})
dump(ROOT/'接口验收.json',{'passed':True,'checks':checks,'scope':'CSV boundary, normal routing, future field leakage; not system deployment'})
print('INFERENCE BOUNDARY CHECK PASSED')
