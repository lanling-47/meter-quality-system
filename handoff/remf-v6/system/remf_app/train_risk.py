# -*- coding: utf-8 -*-
"""Train a separate preinspection model; never consume T0 readings or repair labels."""
import json, hashlib, sys, time, copy, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'data'
OUT=ROOT.parents[1]/'runs'/('risk_'+time.strftime('%Y%m%d_%H%M%S'))
BUNDLE=OUT/'model_bundle'
BUNDLE.mkdir(parents=True,exist_ok=False)
shutil.copy2(ROOT/'model_bundle'/'risk_model.py',BUNDLE)
sys.path.insert(0,str(BUNDLE))
from risk_model import RiskMLP

def main():
    torch.set_num_threads(4)
    data=pd.read_csv(SOURCE/'training_dataset_10000.csv')
    features=json.loads((SOURCE/'model_contract.json').read_text(encoding='utf-8'))['primary_features']
    raw=data[features].to_numpy(np.float32)
    masks={s:(data.split==s).to_numpy() for s in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(data.loc[masks[a],'batch_id'])&set(data.loc[masks[b],'batch_id'])
    bits=data[['y_metering_fail','y_communication_fail','y_display_fail']].to_numpy(np.int64)
    y=(bits*np.array([1,2,4])).sum(1)
    assert np.array_equal(y>0,data.y_any_fail.to_numpy().astype(bool))
    mean=raw[masks['train']].mean(0);scale=raw[masks['train']].std(0);scale[scale<1e-6]=1
    x=torch.tensor((raw-mean)/scale);target=torch.tensor(y,dtype=torch.long)
    tr=np.where(masks['train'])[0];va=np.where(masks['validation'])[0]
    histories=[];valps=[];testps=[];models=[]
    # All hyperparameters declared before test evaluation; no test feedback tuning.
    protocol=dict(seeds=[17,43,101],lr=.001,weight_decay=.03,max_epochs=180,patience=25,batch_size=256,
        selection='minimum validation joint NLL',architecture='34-32-16-8 GELU dropout',
        targets='8 joint combinations of metering/communication/display; derived marginals',
        test_status='existing V6 explored holdout; not a fresh external blind test',device='cpu')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for seed in protocol['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=RiskMLP()
        optimizer=torch.optim.AdamW(net.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        best=1e9;best_epoch=0;best_state=None;history=[]
        for epoch in range(protocol['max_epochs']):
            net.train();losses=[]
            order=rng.permutation(tr)
            for start in range(0,len(order),protocol['batch_size']):
                ids=order[start:start+protocol['batch_size']]
                optimizer.zero_grad();loss=nn.functional.cross_entropy(net(x[ids]),target[ids]);loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
            net.eval()
            with torch.inference_mode():val_loss=float(nn.functional.cross_entropy(net(x[va]),target[va]))
            history.append(dict(epoch=epoch+1,train_nll=float(np.mean(losses)),validation_nll=val_loss))
            if val_loss<best-1e-6:best=val_loss;best_epoch=epoch+1;best_state=copy.deepcopy(net.state_dict())
            if epoch+1-best_epoch>=protocol['patience']:break
        net.load_state_dict(best_state);net.eval();models.append(net)
        torch.save({'state_dict':best_state,'seed':seed,'best_epoch':best_epoch},BUNDLE/f'risk_mlp_s{seed}.pt')
        with torch.inference_mode():valps.append(net(x[va]).softmax(-1).numpy())
        histories.append({'seed':seed,'best_epoch':best_epoch,'history':history})
        print(f'seed={seed} best_epoch={best_epoch} val_nll={best:.6f}',flush=True)
    pval=np.mean(valps,axis=0)
    def temperature(p,t):
        logits=np.log(np.clip(p,1e-10,1))/t;logits-=logits.max(1,keepdims=True)
        pp=np.exp(logits);return pp/pp.sum(1,keepdims=True)
    t=float(min(np.linspace(.5,2.5,101),key=lambda t:log_loss(y[va],temperature(pval,t),labels=np.arange(8))))
    meta=dict(features=features,mean=mean.tolist(),scale=scale.tolist(),temperature=t,seeds=protocol['seeds'],
        training_min=raw[tr].min(0).tolist(),training_max=raw[tr].max(0).tolist(),
        quantile_01=np.quantile(raw[tr],.01,axis=0).tolist(),quantile_99=np.quantile(raw[tr],.99,axis=0).tolist(),
        source_sha256=hashlib.sha256((SOURCE/'training_dataset_10000.csv').read_bytes()).hexdigest(),
        split_counts={s:int(m.sum()) for s,m in masks.items()},
        input_time='before first factory inspection; no detection, repair or confirmation fields',
        model='Manufacturing Risk MLP',version='risk-v1-20261119',device='cpu',
        scope='synthetic sample distribution, not enterprise prevalence or an intervention effect')
    # Freeze preprocessing, temperature and selected checkpoints before evaluating holdout.
    (BUNDLE/'risk_preprocessing.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    te=np.where(masks['test'])[0]
    with torch.inference_mode():ptest=temperature(np.mean([n(x[te]).softmax(-1).numpy() for n in models],axis=0),t)
    combos=np.array([[(k&1)>0,(k&2)>0,(k&4)>0] for k in range(8)],dtype=float)
    marginal=ptest@combos;pred=np.column_stack((1-ptest[:,0],marginal))
    targets=np.column_stack((y[te]>0,bits[te]))
    train_targets=np.column_stack((y[tr]>0,bits[tr]));prior=train_targets.mean(0)
    metrics=[]
    for j,name in enumerate(['any','metering','communication','display']):
        yy=targets[:,j];pp=pred[:,j]
        metrics.append(dict(target=name,n=len(yy),positives=int(yy.sum()),prevalence=float(yy.mean()),
            roc_auc=float(roc_auc_score(yy,pp)),average_precision=float(average_precision_score(yy,pp)),
            brier=float(brier_score_loss(yy,pp)),prior_brier=float(brier_score_loss(yy,np.full(len(yy),prior[j]))),
            mean_probability=float(pp.mean())))
    report=dict(protocol=protocol,metrics=metrics,test_joint_nll=float(log_loss(y[te],ptest,labels=np.arange(8))),temperature=t,
        limitations=['Only synthetic manufacturing proxies; no demonstrated real enterprise predictive performance',
            'Changed inputs show model sensitivity, not experimentally identified causal effects',
            '10% fault prevalence is set by sampling; absolute probabilities must be recalibrated for enterprise data'])
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (BUNDLE/'risk_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    pd.DataFrame(metrics).to_csv(OUT/'测试指标.csv',index=False,encoding='utf-8-sig')
    records=pd.DataFrame(pred,columns=['any','metering','communication','display']);records.insert(0,'sample_id',data.loc[te,'sample_id'].to_numpy())
    records.to_csv(OUT/'test_predictions.csv',index=False)
    names=['risk_model.py','risk_preprocessing.json','risk_evaluation.json']+[f'risk_mlp_s{s}.pt' for s in protocol['seeds']]
    (BUNDLE/'risk_hashes.json').write_text(json.dumps({name:hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() for name in names},indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
