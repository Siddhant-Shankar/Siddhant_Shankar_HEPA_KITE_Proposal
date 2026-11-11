"""Shared-HEAD experiment (reliability scout). Is one hazard head across domains what makes a
single conformal calibration transfer?  Backbones: shared / separate (same pretrain as exp_shared_eps).
Heads (trained on cached frozen-encoder embeddings; encoder in eval mode, i.e. no dropout noise):
  own   : one head per dataset (replicates the existing protocol)
  joint : ONE head trained on all 5 datasets' labels
  lodo  : head trained on 4 datasets, applied zero-shot to the 5th (shared backbone only)
Saves per-window CDFs in the same npz layout as replay.py.
"""
import sys, argparse, copy, time, numpy as np, torch
sys.path.insert(0,'/home/claude/HEPA')
from exp_shared_eps import load, build_model, pretrain, activate, sample_cuts, pad
from hepa.utils import get_context, set_seed
from hepa.training.finetune import EventDataset
from hepa.training.losses import weighted_bce_loss
from hepa.evaluation.surface import build_label_surface
from torch.utils.data import ConcatDataset
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)
ap=argparse.ArgumentParser()
ap.add_argument("--seed",type=int,default=42); ap.add_argument("--pre-epochs",type=int,default=20)
ap.add_argument("--cuts",type=int,default=1024); ap.add_argument("--ft-epochs",type=int,default=10)
ap.add_argument("--ft-cap",type=int,default=2048); ap.add_argument("--test-windows",type=int,default=1200)
ap.add_argument("--out",default="npz_head"); ap.add_argument("--backbones",default="shared,separate")
ap.add_argument("--arms",default="own,joint,lodo")
a=ap.parse_args()
NAMES="FD001,FD003,MBA,PSM,ETTm1".split(",")
data={n:load(n) for n in NAMES}
HZ={n:torch.tensor(data[n][0]["horizons"],dtype=torch.float32) for n in NAMES}

def windows(seed):
    """Same numpy RNG stream as exp_shared_eps.run -> identical train/val/test windows when caps match."""
    rng=np.random.RandomState(seed); W={}
    for n in NAMES:
        b,_=data[n]; ctx,kmax=get_context(n),max(b["horizons"])
        sample_cuts(b["pretrain_seqs"],1024,ctx,64,rng)
        def pick(ents,cap):
            items=[]
            for ei,e in enumerate(ents):
                if len(e["test"])<=129: continue
                d=EventDataset(e["test"],e["labels"],max_context=ctx,stride=1,max_future=kmax)
                items+=[(ei,t,float(d._tte[t])) for t in d.starts]
            idx=rng.choice(len(items),size=min(cap,len(items)),replace=False)
            return [items[i] for i in idx]
        tr=pick(b["ft_train"],a.ft_cap); va=pick(b["ft_val"],a.ft_cap//3)
        wins=[]
        for ei,e in enumerate(b["ft_test"]):
            T=len(e["test"])
            if T<=129: continue
            tte=EventDataset._compute_tte(np.asarray(e["labels"],dtype=np.int32),kmax)
            wins+=[(ei,t,float(tte[t])) for t in range(128,T-1)]
        sel=rng.choice(len(wins),size=min(a.test_windows,len(wins)),replace=False)
        W[n]=dict(tr=tr,va=va,te=[wins[i] for i in sorted(sel)])
    return W

@torch.no_grad()
def embed(model,n,ents,wins):
    b,norm=data[n]; ctx=get_context(n); activate(model,n,norm); model.eval(); out=[]
    for i in range(0,len(wins),128):
        c,cm=pad([ents[ei]["test"][max(0,t-ctx):t] for ei,t,_ in wins[i:i+128]])
        out.append(model.encoder(c,cm))
    return torch.cat(out)

def cdf(pred,head,h,hz):
    B,d=h.shape; K=len(hz)
    hp=pred(h.unsqueeze(1).expand(B,K,d).reshape(B*K,d), hz.unsqueeze(0).expand(B,K).reshape(B*K)).view(B,K,d)
    return 1-torch.cumprod(1-torch.sigmoid(head(hp)),dim=-1)

def train_head(pred,head,E,src,restore_best,seed):
    """E[n] = dict(tr=(H,tte), va=(H,tte)). One head on all datasets in src."""
    pred,head=copy.deepcopy(pred),copy.deepcopy(head)
    for p in list(pred.parameters())+list(head.parameters()): p.requires_grad=True
    params=list(pred.parameters())+list(head.parameters())
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0.01); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.ft_epochs)
    pw={}
    for n in src:
        y=build_label_surface(E[n]["tr"][1].unsqueeze(1),HZ[n]).squeeze(1); pos=float(y.sum()); neg=y.numel()-pos
        pw[n]=torch.tensor(float(min(max(neg/max(pos,1.0),1.0),1000.0)))
    rng=np.random.RandomState(seed); best,bs,wait=float("inf"),None,0
    for ep in range(a.ft_epochs):
        batches=[]
        for n in src:
            perm=rng.permutation(len(E[n]["tr"][0])); batches+=[(n,perm[i:i+64]) for i in range(0,len(perm),64)]
        rng.shuffle(batches)
        for n,idx in batches:
            H,tte=E[n]["tr"]; p=cdf(pred,head,H[idx],HZ[n]); y=build_label_surface(tte[idx].unsqueeze(1),HZ[n]).squeeze(1)
            loss=weighted_bce_loss(p,y,pos_weight=pw[n]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            vl=[]
            for n in src:
                H,tte=E[n]["va"]; ls=[]
                for i in range(0,len(H),256):
                    p=cdf(pred,head,H[i:i+256],HZ[n]); y=build_label_surface(tte[i:i+256].unsqueeze(1),HZ[n]).squeeze(1)
                    ls.append(weighted_bce_loss(p,y,pos_weight=pw[n]).item())
                vl.append(np.mean(ls))
            v=float(np.mean(vl))
        if v<best: best,bs,wait=v,(copy.deepcopy(pred.state_dict()),copy.deepcopy(head.state_dict())),0
        else:
            wait+=1
            if restore_best and wait>=5: break
    if restore_best and bs is not None: pred.load_state_dict(bs[0]); head.load_state_dict(bs[1])
    return pred,head

@torch.no_grad()
def predict(pred,head,E,n):
    H,tte=E[n]["te"]; ps=[cdf(pred,head,H[i:i+256],HZ[n]) for i in range(0,len(H),256)]
    p=torch.cat(ps).numpy(); y=build_label_surface(tte.unsqueeze(1),HZ[n]).squeeze(1).numpy(); return p,y

def save(tag,preds,W):
    S={}; aur={}
    for n,(p,y) in preds.items():
        pc=np.clip(p,1e-6,1-1e-6)
        S[n+"_p"]=p.astype(np.float32); S[n+"_y"]=y.astype(np.int8)
        S[n+"_ent"]=np.array([w[0] for w in W[n]["te"]]); S[n+"_t"]=np.array([w[1] for w in W[n]["te"]]); S[n+"_tte"]=np.array([w[2] for w in W[n]["te"]])
        S[n+"_hent"]=(-(pc*np.log(pc)+(1-pc)*np.log(1-pc))).mean(1); S[n+"_brier"]=((p-y)**2).mean(1); S[n+"_eps"]=np.zeros(len(p))
        aur[n]=roc_auc_score(y.ravel(),p.ravel()) if 0<y.mean()<1 else float("nan")
    np.savez_compressed(f"{a.out}/{tag}_d256_s{a.seed}.npz",**S)
    print(f"RESULT {tag} s{a.seed} hAUROC "+" ".join(f"{n}={aur[n]:.3f}" for n in NAMES)+f" mean={np.mean(list(aur.values())):.3f}",flush=True)

import os; os.makedirs(a.out,exist_ok=True)
W=windows(a.seed)
for bb in a.backbones.split(","):
    t0=time.time(); set_seed(a.seed); E={}; heads={}
    groups=[NAMES] if bb=="shared" else [[n] for n in NAMES]
    for grp in groups:
        m=build_model({n:data[n][0]["n_channels"] for n in grp},256,2)
        pretrain(m,data,grp,a.pre_epochs,a.cuts,a.seed)
        for n in grp:
            b=data[n][0]; E[n]={}
            for k,ents in [("tr",b["ft_train"]),("va",b["ft_val"]),("te",b["ft_test"])]:
                E[n][k]=(embed(m,n,ents,W[n][k]), torch.tensor([w[2] for w in W[n][k]]))
            heads[n]=(copy.deepcopy(m.predictor),copy.deepcopy(m.event_head))
    print(f"[{bb}] pretrain+embed {time.time()-t0:.0f}s",flush=True); t0=time.time()
    preds={}
    for n in NAMES:
        pr,hd=train_head(*heads[n],E,[n],restore_best=(data[n][1]!="none"),seed=a.seed); preds[n]=predict(pr,hd,E,n)
    save(f"{bb}-own",preds,W); print(f"[{bb}] own heads {time.time()-t0:.0f}s",flush=True); t0=time.time()
    pr,hd=train_head(*heads[NAMES[0]],E,NAMES,restore_best=True,seed=a.seed)   # init: pretrained predictor (FD001's for separate)
    save(f"{bb}-joint",{n:predict(pr,hd,E,n) for n in NAMES},W); print(f"[{bb}] joint head {time.time()-t0:.0f}s",flush=True); t0=time.time()
    if bb=="shared" and "lodo" in a.arms:
        preds={}
        for n in NAMES:
            src=[x for x in NAMES if x!=n]; pr,hd=train_head(*heads[src[0]],E,src,restore_best=True,seed=a.seed); preds[n]=predict(pr,hd,E,n)
            a_out=a.out; a.out=a.out+"_lodo"; os.makedirs(a.out,exist_ok=True)   # head_{-n} applied to ALL domains (sources = calibration)
            save(f"shared-lodoH{n}",{x:predict(pr,hd,E,x) for x in NAMES},W); a.out=a_out
        save(f"{bb}-lodo",preds,W); print(f"[{bb}] lodo heads {time.time()-t0:.0f}s",flush=True)
