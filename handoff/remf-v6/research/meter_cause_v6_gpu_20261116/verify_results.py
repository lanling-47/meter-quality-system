"""Independent output checks, including recomputing metrics from per-unit probabilities."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score
from data import prepare,sha,dump,SOURCE

ROOT=Path(__file__).resolve().parent

def main():
    d=prepare(False); frame=d['frame']; checks=[]
    def check(name,value):
        checks.append({'check':name,'passed':bool(value)})
    frozen=json.loads((ROOT/'experiment_frozen.json').read_text(encoding='utf-8'))
    selection=json.loads((ROOT/'selection_frozen.json').read_text(encoding='utf-8'))
    env=json.loads((ROOT/'environment.json').read_text(encoding='utf-8'))
    check('CUDA RTX4060 training recorded',env['cuda_available'] and '4060' in env['gpu'] and env['peak_cuda_allocated_bytes']>0)
    check('Conda interpreter used','anaconda3' in env['executable'] and 'envs' in env['executable'])
    check('24 neural trainings completed',env['neural_trainings']==24)
    check('protocol hash unchanged',sha(ROOT/'protocol.json')==frozen['protocol_sha256'])
    check('training code frozen',all(sha(ROOT/name)==h for name,h in frozen['code_sha256'].items()))
    check('source CSVs unchanged',all(sha(SOURCE/name)==h for name,h in frozen['source_csv_sha256'].items()))
    check('all checkpoint hashes match',all(sha(ROOT/'checkpoints'/name)==h for name,h in selection['checkpoints'].items()))
    pred=pd.read_csv(ROOT/'测试逐台候选原因.csv')
    metrics=pd.read_csv(ROOT/'三种子集成测试指标.csv')
    causes=d['meta']['cause_codes']; cols=['p_'+c for c in causes]
    count=0
    for row in metrics.itertuples():
        pp=pred[(pred.model==row.model)&(pred.confirmation_status=='单因证据支持')]
        y=pp.label.map({c:i for i,c in enumerate(causes)}).to_numpy(int); p=pp[cols].to_numpy()
        check(row.model+' probability range and normalization',np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5))
        check(row.model+' independent macro F1 recomputation',abs(f1_score(y,p.argmax(1),labels=np.arange(8),average='macro',zero_division=0)-row.macro_f1_8)<1e-9)
        check(row.model+' independent accuracy recomputation',abs(accuracy_score(y,p.argmax(1))-row.accuracy)<1e-9)
        check(row.model+' 180 fault predictions 51 known labels',len(pred[pred.model==row.model])==180 and len(pp)==51)
        count+=1
    val=pd.read_csv(ROOT/'validation_metrics.csv').groupby('model').macro_f1_8.mean()
    check('model selected using validation only',selection['selected_model']==val.idxmax())
    history=json.loads((ROOT/'training_history.json').read_text(encoding='utf-8'))
    check('finite training losses',all(np.isfinite([r['loss'],r['ce'],r['response_loss'],r['validation_nll']]).all() for run in history for r in run['history']))
    check('no >300 epochs',all(len(run['history'])<=300 for run in history))
    check('unconfirmed not assigned supervision',len(frame[frame.label>=0])==292 and frame.loc[frame.confirmation_status!='单因证据支持','label'].eq(-1).all())
    result={'passed':all(c['passed'] for c in checks),'passed_count':sum(c['passed'] for c in checks),
            'total':len(checks),'checks':checks,'model_families':count,
            'scope':'Software/numeric reproducibility; not real-enterprise validation'}
    dump(ROOT/'验收结果.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
    assert result['passed']

if __name__=='__main__': main()
