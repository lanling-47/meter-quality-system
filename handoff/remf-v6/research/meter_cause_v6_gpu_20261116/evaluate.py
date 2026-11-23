"""Frozen evaluation. Never tunes a model or threshold against test labels."""
from pathlib import Path
import os,sys,json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.metrics import f1_score,classification_report,confusion_matrix
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from data import prepare,dump,sha,CAUSES
from models import make_model
from metrics import evaluate,temperature

def main():
    torch.set_num_threads(4)
    frozen=json.loads((ROOT/'selection_frozen.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/'checkpoints'/name)==digest for name,digest in frozen['checkpoints'].items())
    d=prepare(save=False); frame=d['frame']; y=d['y']; known=y>=0
    va=(frame.split=='validation').to_numpy()&known; te=(frame.split=='test').to_numpy()&known
    alltest=(frame.split=='test').to_numpy()
    x=torch.tensor(d['x'],device='cuda'); o=torch.tensor(d['o'],device='cuda')
    a=np.column_stack((d['x'],d['o']))
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    rows=[]; ensemble={}; predictions=[]; response_rows=[]
    for name in protocol['models']:
        probabilities=[]; response_predictions=[]
        if name.startswith(('logistic','rf','hist','majority')):
            net=joblib.load(ROOT/'checkpoints'/f'{name}.joblib')
            p=net.predict_proba(a); probabilities.append(p)
            rows.append({'model':name,'seed':17,'split':'test',**evaluate(y[te],p[te])})
        else:
            for seed in protocol['seeds']:
                net=make_model(name).cuda()
                net.load_state_dict(torch.load(ROOT/'checkpoints'/f'{name}_s{seed}.pt',map_location='cuda',weights_only=False)['state_dict'])
                net.eval()
                with torch.no_grad():
                    output=net(x,o); p=output['logits'].softmax(-1).mean(1).cpu().numpy()
                    if name.startswith('remf'):
                        r=output['response'][torch.arange(len(frame),device='cuda'),torch.tensor(d['action'],device='cuda',dtype=torch.long)].cpu().numpy()
                        response_predictions.append(r)
                probabilities.append(p)
                rows.append({'model':name,'seed':seed,'split':'test',**evaluate(y[te],p[te])})
        mean=np.mean(probabilities,axis=0); ensemble[name]=mean
        for i in np.where(alltest)[0]:
            order=np.argsort(-mean[i]); row={'sample_id':frame.iloc[i].sample_id,'batch_id':frame.iloc[i].batch_id,
                'model':name,'label':frame.iloc[i].confirmed_cause_code,'confirmation_status':frame.iloc[i].confirmation_status,
                'top1':CAUSES[order[0]],'top2':CAUSES[order[1]],'top3':CAUSES[order[2]],'confidence':float(mean[i,order[0]])}
            row.update({f'p_{c}':float(mean[i,j]) for j,c in enumerate(CAUSES)}); predictions.append(row)
        if response_predictions:
            r=np.mean(response_predictions,axis=0); allowed=alltest&d['aux_usable']; raw=d['delta']*np.array(d['meta']['response_scale'])
            predicted=r*np.array(d['meta']['response_scale'])
            train=(frame.split=='train').to_numpy()&d['aux_usable']
            means=np.stack([raw[train&(d['action']==c)].mean(0) for c in range(8)])
            prior=means[d['action']]
            for label,ids,actions in [('metering',np.arange(10),[0,1,2,5]),('communication',np.array([10]),[3,5,6]),('display',np.array([11]),[4,5,7])]:
                sel=allowed&np.isin(d['action'],actions)
                mult=200 if label=='communication' else 42 if label=='display' else 1
                response_rows.append({'model':name,'domain':label,'n':int(sel.sum()),
                    'recorded_response_mae':float(np.abs(predicted[sel][:,ids]-raw[sel][:,ids]).mean()*mult),
                    'training_action_mean_mae':float(np.abs(prior[sel][:,ids]-raw[sel][:,ids]).mean()*mult)})
    pd.DataFrame(rows).to_csv(ROOT/'test_metrics_by_seed.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(ROOT/'测试逐台候选原因.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(response_rows).to_csv(ROOT/'维修响应辅助任务.csv',index=False,encoding='utf-8-sig')
    ensemble_rows=[]
    for name,p in ensemble.items():
        cal=frozen['calibration'][name]; cp=temperature(p,cal['temperature'])
        m=evaluate(y[te],p[te]); cm=evaluate(y[te],cp[te]); retained=cp[te].max(1)>=cal['threshold']
        ensemble_rows.append({'model':name,**m,'calibrated_nll':cm['nll'],'calibrated_ece':cm['ece'],
             'selective_retained':int(retained.sum()),'selective_coverage':float(retained.mean()),
             'selective_accuracy':float((cp[te].argmax(1)[retained]==y[te][retained]).mean()) if retained.any() else None})
    pd.DataFrame(ensemble_rows).to_csv(ROOT/'三种子集成测试指标.csv',index=False,encoding='utf-8-sig')
    selected=frozen['selected_model']; selectedneural=frozen['selected_neural']; p=ensemble[selected]
    reports={name:classification_report(y[te],pp[te].argmax(1),labels=np.arange(8),target_names=CAUSES,output_dict=True,zero_division=0) for name,pp in ensemble.items()}
    dump(ROOT/'per_class_reports.json',reports)
    confusion_matrix(y[te],p[te].argmax(1),labels=np.arange(8))
    pd.DataFrame(confusion_matrix(y[te],p[te].argmax(1),labels=np.arange(8)),index=CAUSES,columns=CAUSES).to_csv(ROOT/'选定模型混淆矩阵.csv',encoding='utf-8-sig')
    cal=frozen['calibration'][selected]; cp=temperature(p,cal['temperature']); candidate=[]
    for i in np.where(alltest)[0]:
        order=np.argsort(-cp[i]); retained=cp[i].max()>=cal['threshold']
        candidate.append({'sample_id':frame.iloc[i].sample_id,'confirmation_status':frame.iloc[i].confirmation_status,
            'evidence_label':frame.iloc[i].confirmed_cause_code,'top1':CAUSES[order[0]],'top2':CAUSES[order[1]],'top3':CAUSES[order[2]],
            'calibrated_confidence':float(cp[i,order[0]]),'recommendation':'优先人工复核' if retained else '证据不足需人工排查',
            'not_enterprise_confirmed':True})
    pd.DataFrame(candidate).to_csv(ROOT/'选定模型全部故障候选.csv',index=False,encoding='utf-8-sig')
    # Bootstrap by held-out batch, with the same resample for paired comparisons.
    rng=np.random.default_rng(2026111621); batches=frame.loc[te,'batch_id'].to_numpy(); unique=np.unique(batches); yy=y[te]
    bootstrap={name:[] for name in ensemble}; differences=[]
    for _ in range(2000):
        sample=rng.choice(unique,len(unique),replace=True)
        ids=np.concatenate([np.where(batches==b)[0] for b in sample])
        for name,pp in ensemble.items():
            bootstrap[name].append(float(f1_score(yy[ids],pp[te][ids].argmax(1),labels=np.arange(8),average='macro',zero_division=0)))
        differences.append(bootstrap[selectedneural][-1]-bootstrap['rf_combined'][-1])
    cis={name:{'p025':float(np.quantile(v,.025)),'p975':float(np.quantile(v,.975))} for name,v in bootstrap.items()}
    difference={'comparison':selectedneural+' minus rf_combined','p025':float(np.quantile(differences,.025)),
        'p975':float(np.quantile(differences,.975)),'note':'fixed 8 labels; rare classes absent from some resamples; exploratory interval'}
    multi=alltest&(frame.confirmation_status=='多因素证据支持').to_numpy(); mrows=[]
    for i in np.where(multi)[0]:
        labels=set(frame.iloc[i].confirmed_cause_code.split(';')); order=np.argsort(-p[i])
        top2={CAUSES[j] for j in order[:2]}; top3={CAUSES[j] for j in order[:3]}
        mrows.append({'sample_id':frame.iloc[i].sample_id,'labels':';'.join(sorted(labels)),
            'top2_all_covered':labels<=top2,'top3_all_covered':labels<=top3,'top3_any_covered':bool(labels&top3)})
    pd.DataFrame(mrows).to_csv(ROOT/'多因病例候选覆盖.csv',index=False,encoding='utf-8-sig')
    # Input missingness challenge: use train means, never refit on test.
    robustness=[]; rng=np.random.default_rng(2026111622)
    for rate in [.1,.3]:
        missing=rng.random(a.shape)<rate; modified=a.copy(); modified[missing]=0
        preds=[]
        if selected.startswith(('logistic','rf','hist','majority')):
            preds=[joblib.load(ROOT/'checkpoints'/f'{selected}.joblib').predict_proba(modified)]
        else:
            for seed in protocol['seeds']:
                net=make_model(selected).cuda();net.load_state_dict(torch.load(ROOT/'checkpoints'/f'{selected}_s{seed}.pt',map_location='cuda',weights_only=False)['state_dict']);net.eval()
                with torch.no_grad(): preds.append(net(torch.tensor(modified[:,:34],device='cuda'),torch.tensor(modified[:,34:],device='cuda'))['logits'].softmax(-1).mean(1).cpu().numpy())
        robustness.append({'missing_fraction':rate,'imputation':'training_mean_no_retraining',**evaluate(y[te],np.mean(preds,axis=0)[te])})
    pd.DataFrame(robustness).to_csv(ROOT/'缺失输入压力测试.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(ROOT/'evaluation_probabilities.npz',**ensemble)
    output={'selected_model':selected,'selected_neural':selectedneural,'test_single_n':int(te.sum()),
        'test_fault_n':int(alltest.sum()),'test_metering3_n':int((te&(y<3)).sum()),'bootstrap_by_batch':cis,
        'paired_neural_vs_rf':difference,'multi_n':int(multi.sum()),
        'multi_top3_all_covered':float(np.mean([z['top3_all_covered'] for z in mrows])) if mrows else None,
        'test_single_batches':int(len(unique)),'source_truth_used':False,
        'note':'Known-class evidence subset only; no accuracy claim for unconfirmed cases or enterprise data'}
    dump(ROOT/'evaluation.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
