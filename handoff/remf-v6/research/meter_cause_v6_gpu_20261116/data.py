from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=Path(json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))['source'])
CAUSES=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))['causes']

def read(name): return pd.read_csv(SOURCE/name,encoding='utf-8-sig')
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw_obs(df,cols):
    v=df[cols].to_numpy(np.float32).copy(); v[:,-2]/=200; v[:,-1]/=42
    return v

def prepare(save=True):
    contract=json.loads((SOURCE/'model_contract.json').read_text(encoding='utf-8'))
    feat=contract['primary_features']; obscols=contract['diagnosis_observations']
    base=read('training_dataset_10000.csv')
    inspection=read('出厂检测.csv').set_index('sample_id').reindex(base.sample_id)
    x=base[feat].to_numpy(np.float32); o=raw_obs(inspection,obscols)
    tr=(base.split=='train').to_numpy()
    xm=x[tr].mean(0); xs=x[tr].std(0); xs[xs<1e-6]=1
    om=o[tr].mean(0); os=o[tr].std(0); os[os<1e-6]=1
    conf=read('故障确认.csv').set_index('sample_id')
    fault=base[base.y_any_fail==1].copy().reset_index(drop=True)
    cf=conf.reindex(fault.sample_id)
    fault['label']=cf.confirmed_cause_code.map({c:i for i,c in enumerate(CAUSES)}).fillna(-1).to_numpy(int)
    fault['confirmation_status']=cf.confirmation_status.to_numpy()
    fault['confirmed_cause_code']=cf.confirmed_cause_code.fillna('').to_numpy()
    assert ((fault.label>=0)==cf.eligible_single_cause_label.to_numpy(bool)).all()
    assert len(base)==10000 and len(fault)==1000 and (fault.label>=0).sum()==292
    idx=base.set_index('sample_id').index.get_indexer(fault.sample_id)
    x=(x[idx]-xm)/xs; o=(o[idx]-om)/os
    repeats=read('复测记录.csv')
    pre=repeats[repeats.repair_attempt==0].groupby('sample_id')[obscols].mean()
    post=repeats[repeats.repair_attempt==1].groupby('sample_id')[obscols].mean()
    repairs=read('维修记录.csv'); first=repairs[repairs.attempt==1].set_index('sample_id').reindex(fault.sample_id)
    delta=raw_obs(pre.reindex(fault.sample_id),obscols)-raw_obs(post.reindex(fault.sample_id),obscols)
    action=first.repair_target.map({c:i for i,c in enumerate(CAUSES)}).to_numpy(int)
    usable=first.co_intervention_target.fillna('').eq('').to_numpy()
    auxtrain=(fault.split=='train').to_numpy()&usable
    rs=delta[auxtrain].std(0); rs[rs<.01]=.01
    weights=np.bincount(fault.loc[(fault.split=='train')&(fault.label>=0),'label'],minlength=8).astype(float)
    assert (weights>0).all()
    weights=1/np.sqrt(weights); weights/=weights.mean()
    identities=read('来源映射.csv').set_index('sample_id').reindex(base.sample_id)
    isolation={}
    for col in ['batch_id','source_background_meter','ct_source_sample','crystal_source_sample']:
        vals=base[col].to_numpy() if col=='batch_id' else identities[col].to_numpy()
        sets={s:set(vals[(base.split==s).to_numpy()]) for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]: assert not sets[a]&sets[b],col
        isolation[col]={s:len(v) for s,v in sets.items()}
    assert np.isfinite(x).all() and np.isfinite(o).all() and np.isfinite(delta).all()
    counts=fault[fault.label>=0].groupby(['split','confirmed_cause_code']).size().unstack(fill_value=0)
    meta={'feature_columns':feat,'observation_columns':obscols,'cause_codes':CAUSES,
          'x_mean':xm.tolist(),'x_scale':xs.tolist(),'o_mean':om.tolist(),'o_scale':os.tolist(),
          'response_scale':rs.tolist(),'class_weight':weights.tolist(),'source_isolation':isolation,
          'confirmed_counts':counts.to_dict('index'),
          'aux_first_repairs':{s:int(((fault.split==s).to_numpy()&usable).sum()) for s in ['train','validation','test']},
          'feature_time':'T0后，诊断检查复测维修前','no_simulator_audit_tables_read':True}
    if save:
        dump(ROOT/'preprocessing.json',meta)
        counts.to_csv(ROOT/'标签分布.csv',encoding='utf-8-sig')
    return dict(frame=fault,x=x,o=o,y=fault.label.to_numpy(),delta=delta/rs,
                action=action,aux_usable=usable,weights=weights.astype(np.float32),meta=meta)
