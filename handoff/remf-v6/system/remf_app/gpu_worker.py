# -*- coding: utf-8 -*-
"""Persistent JSONL inference worker in the existing Conda CUDA environment."""
import sys, json, hashlib, os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def main():
    bundle=Path(sys.argv[1])
    sys.path.insert(0,str(bundle))
    import numpy as np
    import torch
    from models import make_model
    torch.set_num_threads(4)
    meta=json.loads((bundle/'preprocessing.json').read_text(encoding='utf-8'))
    frozen=json.loads((bundle/'selection_frozen.json').read_text(encoding='utf-8'))
    code=json.loads((bundle/'experiment_frozen.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((bundle/'models.py').read_bytes()).hexdigest()==code['code_sha256']['models.py'], '模型代码校验失败'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    nets=[]; hashes={}
    for seed in [17,43,101]:
        filename=f'remf_v6_s{seed}.pt'
        h=hashlib.sha256((bundle/filename).read_bytes()).hexdigest()
        assert h==frozen['checkpoints'][filename], '模型权重校验失败'
        ck=torch.load(bundle/filename,map_location=device,weights_only=False)
        assert ck['preprocessing']==meta, '预处理与权重不匹配'
        net=make_model('remf_v6').to(device)
        net.load_state_dict(ck['state_dict']); net.eval(); nets.append(net); hashes[filename]=h
    calibration=frozen['calibration']['remf_v6']
    risk_nets=[];risk_meta=None
    if (bundle/'risk_hashes.json').exists():
        hashes_risk=json.loads((bundle/'risk_hashes.json').read_text(encoding='utf-8'))
        for filename,digest in hashes_risk.items():
            assert hashlib.sha256((bundle/filename).read_bytes()).hexdigest()==digest, '预测模型校验失败'
        from risk_model import RiskMLP
        risk_meta=json.loads((bundle/'risk_preprocessing.json').read_text(encoding='utf-8'))
        for seed in risk_meta['seeds']:
            net=RiskMLP().to(device)
            net.load_state_dict(torch.load(bundle/f'risk_mlp_s{seed}.pt',map_location=device,weights_only=False)['state_dict'])
            net.eval();risk_nets.append(net)
    print(json.dumps({'ready':True,'device':device,'hardware':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        'torch':torch.__version__,'hashes':hashes,'temperature':calibration['temperature'],'risk_ready':bool(risk_nets)}),flush=True)
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line); values=request['values']
            if request.get('task')=='preinspection':
                assert risk_nets, '检测前预测模型尚未就绪'
                raw=np.array([[values[k] for k in risk_meta['features']]],dtype=np.float32)
                xx=((raw-np.array(risk_meta['mean'],dtype=np.float32))/np.array(risk_meta['scale'],dtype=np.float32)).astype(np.float32)
                with torch.inference_mode():
                    p=np.mean([n(torch.tensor(xx,device=device)).softmax(-1).cpu().numpy() for n in risk_nets],axis=0)[0]
                z=np.log(np.clip(p,1e-10,1))/risk_meta['temperature'];z-=z.max();joint=np.exp(z);joint/=joint.sum()
                bits=np.array([[(k&1)>0,(k&2)>0,(k&4)>0] for k in range(8)],dtype=float)
                predictions=np.concatenate(([1-joint[0]],joint@bits))
                print(json.dumps({'id':request['id'],'probabilities':predictions.tolist(),'joint':joint.tolist(),
                    'out_of_range':[k for i,k in enumerate(risk_meta['features']) if raw[0,i]<risk_meta['training_min'][i] or raw[0,i]>risk_meta['training_max'][i]],
                    'tail_features':[k for i,k in enumerate(risk_meta['features']) if raw[0,i]<risk_meta['quantile_01'][i] or raw[0,i]>risk_meta['quantile_99'][i]],
                    'model_version':risk_meta['version']},allow_nan=False),flush=True)
                continue
            x=np.array([[values[k] for k in meta['feature_columns']]],dtype=np.float64)
            o=np.array([[values[k] for k in meta['observation_columns']]],dtype=np.float64)
            o[:,-2]/=200; o[:,-1]/=42
            x=((x-meta['x_mean'])/meta['x_scale']).astype(np.float32)
            o=((o-meta['o_mean'])/meta['o_scale']).astype(np.float32)
            assert np.isfinite(x).all() and np.isfinite(o).all()
            with torch.inference_mode():
                p=np.mean([n(torch.tensor(x,device=device),torch.tensor(o,device=device))['logits'].softmax(-1).mean(1).cpu().numpy() for n in nets],axis=0)[0]
            z=np.log(np.clip(p,1e-10,1))/calibration['temperature']; z-=z.max(); p=np.exp(z); p/=p.sum()
            print(json.dumps({'id':request['id'],'probabilities':p.tolist(),
                'low_confidence':bool(p.max()<calibration['threshold']), 'temperature':calibration['temperature'],
                'threshold':calibration['threshold']},allow_nan=False),flush=True)
        except Exception as exc:
            print(json.dumps({'id':request.get('id'),'error':str(exc)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
