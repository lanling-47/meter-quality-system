"""Train/validate only. Test predictions are deferred until selection_frozen.json exists."""
from pathlib import Path
import os,json,random,time,sys,platform,subprocess
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
import scipy,sklearn,matplotlib
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score,log_loss
from sklearn.utils.class_weight import compute_sample_weight
import joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from models import make_model,domain_mask
from data import prepare,dump,sha,SOURCE,CAUSES
from metrics import evaluate,calibrate

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def probabilities(model,x,o):
    model.eval()
    with torch.no_grad(): return model(x,o)['logits'].softmax(-1).mean(1).cpu().numpy()

def main():
    assert torch.cuda.is_available(),'CUDA required; refusing CPU fallback'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.cuda.reset_peak_memory_stats()
    probe=torch.randn(256,256,device='cuda'); probe=probe@probe; torch.cuda.synchronize()
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':True,
         'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
         'gpu_probe_mean':float(probe.mean()),'training_device':'cuda:0',
         'resolved_packages':{m.__name__:{'version':m.__version__,'path':m.__file__} for m in [np,pd,scipy,sklearn,matplotlib,torch]},
         'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
    dump(ROOT/'environment.json',env)
    (ROOT/'requirements-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    data=prepare(); frame=data['frame']; protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    hashes={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file() and p.suffix=='.csv' and '仅审计' not in p.name}
    dump(ROOT/'experiment_frozen.json',{'protocol_sha256':sha(ROOT/'protocol.json'),
        'code_sha256':{name:sha(ROOT/name) for name in ['data.py','models.py','metrics.py','train.py']},
        'source_csv_sha256':hashes,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'test_feedback_used_for_training':False})
    keep=(data['y']>=0); tr=((frame.split=='train').to_numpy()&keep); va=((frame.split=='validation').to_numpy()&keep)
    aux=((frame.split=='train').to_numpy()&data['aux_usable'])
    x=torch.tensor(data['x'],device='cuda'); o=torch.tensor(data['o'],device='cuda')
    y=torch.tensor(data['y'],device='cuda',dtype=torch.long); delta=torch.tensor(data['delta'],device='cuda')
    action=torch.tensor(data['action'],device='cuda',dtype=torch.long); weights=torch.tensor(data['weights'],device='cuda')
    tx,vo=x[tr],o[tr]; vx,vobs=x[va],o[va]; ty=y[tr]; vy=data['y'][va]
    ax,ao=x[aux],o[aux]; ad=delta[aux]; aa=action[aux]
    checkpoints=ROOT/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    rows=[]; allval={}; trainrecords=[]
    for name in protocol['models']:
        if name.startswith(('logistic','rf','hist','majority')): continue
        pp=[]
        for seed in protocol['seeds']:
            seed_all(seed); net=make_model(name).cuda()
            optimizer=torch.optim.AdamW(net.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
            lossfn=nn.CrossEntropyLoss(weight=weights)
            best=(-1.,float('-inf')); beststate=None; bestepoch=0; history=[]; start=time.perf_counter()
            for epoch in range(protocol['max_epochs']):
                net.train(); optimizer.zero_grad(set_to_none=True)
                result=net(tx,vo); logits=result['logits']
                ce=lossfn(logits.reshape(-1,8),ty[:,None].expand(-1,logits.shape[1]).reshape(-1))
                response_loss=torch.zeros((),device='cuda')
                use_response=name.startswith('remf')
                if use_response:
                    response=net(ax,ao)['response'][torch.arange(len(aa),device='cuda'),aa]
                    active=net.mask[aa]
                    diff=nn.functional.smooth_l1_loss(response,ad,reduction='none')*active
                    # Mean each observed domain separately, preventing 10 metering dimensions from dominating.
                    domain_losses=[]
                    for sl in [slice(0,10),slice(10,11),slice(11,12)]:
                        m=active[:,sl]; valid=m.sum(1)>0
                        if valid.any(): domain_losses.append((diff[:,sl].sum(1)[valid]/m.sum(1)[valid]).mean())
                    response_loss=torch.stack(domain_losses).mean()
                loss=ce+protocol['response_loss_weight']*response_loss
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5); optimizer.step()
                vp=probabilities(net,vx,vobs)
                vf=float(f1_score(vy,vp.argmax(1),labels=np.arange(8),average='macro',zero_division=0))
                vn=float(log_loss(vy,vp,labels=np.arange(8)))
                history.append({'epoch':epoch+1,'loss':float(loss.detach()),'ce':float(ce.detach()),
                                'response_loss':float(response_loss.detach()),'validation_macro_f1':vf,'validation_nll':vn})
                score=(vf,-vn)
                if score>best:
                    best=score; bestepoch=epoch+1
                    beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                if epoch+1-bestepoch>=protocol['patience']: break
            net.load_state_dict(beststate); torch.cuda.synchronize()
            vp=probabilities(net,vx,vobs); pp.append(vp)
            checkpoint=checkpoints/f'{name}_s{seed}.pt'
            torch.save({'model':name,'seed':seed,'state_dict':beststate,'best_epoch':bestepoch,
                        'preprocessing':data['meta'],'cause_codes':CAUSES},checkpoint)
            # Verify serialised parameters reload and reproduce validation predictions on GPU.
            loaded=make_model(name).cuda(); loaded.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict'])
            np.testing.assert_allclose(probabilities(loaded,vx,vobs),vp,atol=1e-6,rtol=1e-5)
            row={'model':name,'seed':seed,'split':'validation',**evaluate(vy,vp),
                 'epochs':len(history),'best_epoch':bestepoch,'seconds':time.perf_counter()-start,
                 'parameters':sum(p.numel() for p in net.parameters()),'device':str(next(net.parameters()).device)}
            rows.append(row); trainrecords.append({'model':name,'seed':seed,'history':history})
            print(json.dumps({k:row[k] for k in ['model','seed','macro_f1_8','best_epoch','seconds','device']}),flush=True)
            del net,loaded; torch.cuda.empty_cache()
        allval[name]=np.mean(pp,axis=0)
    a=np.column_stack((data['x'],data['o']))
    baselines={'logistic_combined':LogisticRegression(C=1.,class_weight='balanced',max_iter=2000),
        'rf_combined':RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=17,n_jobs=4),
        'hist_combined':HistGradientBoostingClassifier(max_iter=200,max_depth=3,learning_rate=.05,l2_regularization=5,random_state=17),
        'majority_prior':DummyClassifier(strategy='prior')}
    for name,net in baselines.items():
        start=time.perf_counter()
        if name.startswith('hist'): net.fit(a[tr],data['y'][tr],sample_weight=compute_sample_weight('balanced',data['y'][tr]))
        else: net.fit(a[tr],data['y'][tr])
        vp=net.predict_proba(a[va]); allval[name]=vp
        joblib.dump(net,checkpoints/f'{name}.joblib')
        row={'model':name,'seed':17,'split':'validation',**evaluate(vy,vp),'seconds':time.perf_counter()-start,'device':'cpu'}
        rows.append(row); print(json.dumps({k:row[k] for k in ['model','macro_f1_8','seconds']}),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'validation_metrics.csv',index=False,encoding='utf-8-sig')
    dump(ROOT/'training_history.json',trainrecords)
    avg=pd.DataFrame(rows).groupby('model').macro_f1_8.mean().to_dict()
    neural=[name for name in avg if name.startswith(('mlp','tabm','grouped','remf'))]
    selected=max(avg,key=avg.get); selectedneural=max(neural,key=avg.get)
    calibration={name:calibrate(vy,p) for name,p in allval.items()}
    np.savez_compressed(ROOT/'validation_probabilities.npz',**allval)
    dump(ROOT/'selection_frozen.json',{'selection_basis':'mean validation macroF1 across predeclared seeds; no test scores',
        'validation_mean_macro_f1':avg,'selected_model':selected,'selected_neural':selectedneural,
        'calibration':calibration,'checkpoints':{p.name:sha(p) for p in checkpoints.iterdir()},
        'timestamp':time.strftime('%Y-%m-%d %H:%M:%S %z')})
    env['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(); env['neural_trainings']=len(neural)*3
    dump(ROOT/'environment.json',env)
    print('TRAINING COMPLETE; selection frozen. Run evaluate.py separately.',flush=True)

if __name__=='__main__': main()
