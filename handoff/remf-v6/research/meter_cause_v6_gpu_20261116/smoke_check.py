from pathlib import Path
import sys,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import torch,numpy as np
from data import prepare,dump
from models import make_model,GROUPS,domain_mask

d=prepare(); checks=[]
assert torch.cuda.is_available()
assert set(sum(GROUPS,[]))==set(range(34))
assert len(sum(GROUPS,[]))==34
assert tuple(domain_mask().shape)==(8,12)
assert d['frame'].loc[d['y']>=0].groupby('split').size().to_dict()=={'train':187,'validation':54,'test':51}
for name in ['mlp_combined','tabm_combined','remf_v6','remf_no_groups','remf_no_mask']:
    net=make_model(name).cuda(); x=torch.randn(6,34,device='cuda'); o=torch.randn(6,12,device='cuda')
    out=net(x,o); assert out['logits'].shape[0]==6 and out['logits'].shape[-1]==8
    loss=out['logits'].square().mean()
    if 'response' in out:
        assert out['response'].shape==(6,8,12)
        loss+=out['response'].square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    checks.append({'model':name,'cuda_forward_backward':True,'parameters':sum(p.numel() for p in net.parameters())})
dump(Path(__file__).resolve().parent/'smoke_check.json',{'passed':True,'gpu':torch.cuda.get_device_name(0),'checks':checks})
print('SMOKE CHECK PASSED',torch.cuda.get_device_name(0),flush=True)
