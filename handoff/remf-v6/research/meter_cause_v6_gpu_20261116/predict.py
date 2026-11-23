"""CSV cause ranking at T0. Strictly requires all 46 observed feature columns."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch,joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from models import make_model
from metrics import temperature

def main():
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('output')
    parser.add_argument('--model');parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    selection=json.loads((ROOT/'selection_frozen.json').read_text(encoding='utf-8'))
    meta=json.loads((ROOT/'preprocessing.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    name=args.model or selection['selected_model'];f=pd.read_csv(args.input,encoding='utf-8-sig')
    required=meta['feature_columns']+meta['observation_columns']
    missing=set(required)-set(f)
    if missing: raise ValueError('缺少首次诊断所需字段: '+', '.join(sorted(missing)))
    x=f[meta['feature_columns']].to_numpy(np.float32); o=f[meta['observation_columns']].to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(o).all(): raise ValueError('存在空值或非有限值；请先补齐或明确处理缺失策略')
    o[:,-2]/=200;o[:,-1]/=42
    # Route only physically abnormal meters to cause ranking; this is not a learned preinspection predictor.
    abnormal=(np.abs(o[:,:10])>np.array([1.5,1.5]+[1.]*8)).any(1)|(o[:,10]>.05)|(o[:,11]>0)
    x=(x-np.array(meta['x_mean']))/np.array(meta['x_scale'])
    o=(o-np.array(meta['o_mean']))/np.array(meta['o_scale'])
    a=np.column_stack((x,o)).astype(np.float32)
    if name.startswith(('logistic','rf','hist','majority')):
        p=joblib.load(ROOT/'checkpoints'/f'{name}.joblib').predict_proba(a)
    else:
        ps=[]
        for seed in protocol['seeds']:
            net=make_model(name).to(args.device);net.load_state_dict(torch.load(ROOT/'checkpoints'/f'{name}_s{seed}.pt',map_location=args.device,weights_only=False)['state_dict']);net.eval()
            with torch.no_grad():ps.append(net(torch.tensor(a[:,:34],device=args.device),torch.tensor(a[:,34:],device=args.device))['logits'].softmax(-1).mean(1).cpu().numpy())
        p=np.mean(ps,axis=0)
    cal=selection['calibration'][name];p=temperature(p,cal['temperature']);rows=[]
    for i in range(len(f)):
        order=np.argsort(-p[i]);r={'sample_id':str(f.iloc[i].get('sample_id',i)), 'model':name}
        r['status']='优先人工复核' if abnormal[i] and p[i].max()>=cal['threshold'] else '证据不足需人工排查' if abnormal[i] else '首次检测通过不强制归因'
        for k in range(3):r[f'top{k+1}']=meta['cause_codes'][order[k]] if abnormal[i] else ''
        r['model_confidence']=float(p[i].max()) if abnormal[i] else None
        r['scope']='模拟证据训练，仅供辅助；分值不是实际原因发生概率'
        rows.append(r)
    pd.DataFrame(rows).to_csv(args.output,index=False,encoding='utf-8-sig')
    print('Saved',args.output)

if __name__=='__main__':main()
