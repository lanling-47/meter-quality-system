"""V6 cause diagnosis. Engineering groups/masks are explicit priors, not causal discovery."""
import torch
from torch import nn

GROUPS = [list(range(11)) + [30], list(range(12,24)), list(range(24,28)), [28], [11,29], [31,32,33]]
GROUP_NAMES = ['thermal_paste', 'ct', 'compensation', 'cleanliness', 'assembly', 'incoming_margin']

def block(n, h, out, drop=.15):
    return nn.Sequential(nn.Linear(n,h), nn.GELU(), nn.Dropout(drop), nn.Linear(h,out))

def domain_mask():
    a=torch.zeros(8,12); a[:3,:10]=1; a[3,10]=1; a[4,11]=1
    a[5,:]=1; a[6,10]=1; a[7,11]=1
    return a

class Plain(nn.Module):
    def __init__(self, name):
        super().__init__(); self.name=name
        n=34 if name=='mlp_manufacturing' else 12 if name=='mlp_observations' else 46
        if name.startswith('tabm'):
            import tabm
            import rtdl_num_embeddings
            self.net=tabm.TabM.make(n_num_features=n,cat_cardinalities=None,d_out=8,
                num_embeddings=rtdl_num_embeddings.LinearReLUEmbeddings(n,8),
                n_blocks=2,d_block=64,dropout=.15,k=8)
        else:
            self.net=nn.Sequential(nn.Linear(n,64), nn.GELU(), nn.Dropout(.2),
                                   nn.Linear(64,32), nn.GELU(), nn.Dropout(.15), nn.Linear(32,8))
    def forward(self,x,o):
        z=x if self.name=='mlp_manufacturing' else o if self.name=='mlp_observations' else torch.cat((x,o),-1)
        y=self.net(z)
        return {'logits':y.unsqueeze(1) if y.ndim==2 else y}

class Fusion(nn.Module):
    def __init__(self,name):
        super().__init__(); self.grouped=name!='remf_no_groups'
        self.enc=nn.ModuleList(block(len(g),24,12) for g in GROUPS)
        self.flat=block(34,64,72)
        self.gates=nn.Parameter(torch.zeros(8,6))
        self.cause=nn.ModuleList(block(72,32,16) for _ in range(8))
        self.observation=block(12,32,16)
        self.head=nn.ModuleList(block(16*3+12,32,1) for _ in range(8))
        self.response=nn.ModuleList(block(16*2+12,32,12,0.) for _ in range(8))
        self.register_buffer('mask',domain_mask() if name!='remf_no_mask' else torch.ones(8,12))
    def forward(self,x,o):
        g=torch.stack([e(x[:,idx]) for e,idx in zip(self.enc,GROUPS)],1) if self.grouped else self.flat(x).reshape(-1,6,12)
        w=6*self.gates.softmax(-1)
        z=[e((g*w[c][None,:,None]).flatten(1)) for c,e in enumerate(self.cause)]
        obs=self.observation(o)
        logits=torch.cat([h(torch.cat((z[c],obs,z[c]*obs,o),-1)) for c,h in enumerate(self.head)],-1)
        response=torch.stack([h(torch.cat((z[c],obs,o),-1)) for c,h in enumerate(self.response)],1)*self.mask
        return {'logits':logits.unsqueeze(1),'response':response}

def make_model(name):
    return Plain(name) if name.startswith(('mlp','tabm')) else Fusion(name)
