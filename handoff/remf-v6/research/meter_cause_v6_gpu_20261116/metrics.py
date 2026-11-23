import numpy as np
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,log_loss

def softmax(a):
    a=a-a.max(-1,keepdims=True); p=np.exp(a); return p/p.sum(-1,keepdims=True)

def temperature(p,t): return softmax(np.log(np.clip(p,1e-10,1))/t)

def evaluate(y,p):
    pred=p.argmax(1); order=np.argsort(-p,axis=1)
    confidence=p.max(1); correct=(pred==y)
    ece=0.
    for lo in np.linspace(0,1,11)[:-1]:
        keep=(confidence>=lo)&(confidence<lo+.1+1e-10)
        if keep.any(): ece+=keep.mean()*abs(correct[keep].mean()-confidence[keep].mean())
    meter=y<3
    return dict(n=len(y),macro_f1_8=float(f1_score(y,pred,labels=np.arange(8),average='macro',zero_division=0)),
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        top2=float((order[:,:2]==y[:,None]).any(1).mean()),top3=float((order[:,:3]==y[:,None]).any(1).mean()),
        macro_f1_metering3=float(f1_score(y[meter],p[meter,:3].argmax(1),labels=np.arange(3),average='macro',zero_division=0)),
        n_metering3=int(meter.sum()),nll=float(log_loss(y,p,labels=np.arange(8))),
        brier=float(((p-np.eye(8)[y])**2).sum(1).mean()),ece=float(ece))

def calibrate(y,p):
    temps=np.exp(np.linspace(np.log(.5),np.log(5),101))
    t=float(min(temps,key=lambda t: log_loss(y,temperature(p,t),labels=np.arange(8))))
    pp=temperature(p,t); conf=pp.max(1); correct=pp.argmax(1)==y
    eligible=[]
    for threshold in np.unique(conf):
        keep=conf>=threshold
        if keep.sum()>=10 and correct[keep].mean()>=.8:
            eligible.append((int(keep.sum()),float(threshold),float(correct[keep].mean())))
    threshold=max(eligible,key=lambda z:(z[0],-z[1]))[1] if eligible else 1.000001
    return {'temperature':t,'threshold':threshold,'target_validation_precision':.8,
        'min_validation_retained':10,'eligible_threshold_found':bool(eligible),
        'validation_retained':int((conf>=threshold).sum()),
        'validation_retained_accuracy':float(correct[conf>=threshold].mean()) if (conf>=threshold).any() else None}
