"""Recover (entity, t) for every saved test window by replaying the numpy RNG stream of
exp_shared_eps.run (it never depends on model outputs), verify against saved labels, and
write compact npz files. No training."""
import sys, json, glob, os, numpy as np, torch
sys.path.insert(0,'/home/claude/HEPA')
from exp_shared_eps import load, sample_cuts, event_loader
from hepa.utils import get_context
from hepa.training.finetune import EventDataset
from hepa.evaluation.surface import build_label_surface
NAMES="FD001,FD003,MBA,PSM,ETTm1".split(",")
data={n:load(n) for n in NAMES}
def replay(seed, ft_cap, test_windows):
    rng=np.random.RandomState(seed); out={}
    for n in NAMES:
        b,norm=data[n]; ctx,kmax=get_context(n),max(b["horizons"])
        sample_cuts(b["pretrain_seqs"],1024,ctx,64,rng)
        event_loader(b["ft_train"],ctx,kmax,ft_cap,rng,True)
        event_loader(b["ft_val"],ctx,kmax,ft_cap//3,rng,False)
        wins=[]
        for ei,e in enumerate(b["ft_test"]):
            T=len(e["test"])
            if T<=129: continue
            tte=EventDataset._compute_tte(np.asarray(e["labels"],dtype=np.int32),kmax)
            for t in range(128,T-1): wins.append((ei,t,tte[t]))
        sel=rng.choice(len(wins),size=min(test_windows,len(wins)),replace=False)
        out[n]=[wins[i] for i in sorted(sel)]
    return out
cache={}
for f in sorted(glob.glob('/home/claude/HEPA/results/*.json')):
    J=json.load(open(f)); a=J["args"]; rows=J["rows"]
    key=(a["seed"],a["ft_cap"],a["test_windows"])
    if key not in cache: cache[key]=replay(*key)
    W=cache[key]; tag=os.path.basename(f)[:-5]; save={}; ok=True
    for n in NAMES:
        r=[x for x in rows if x["ds"]==n]; w=W[n]
        hz=torch.tensor(data[n][0]["horizons"],dtype=torch.float32)
        y=build_label_surface(torch.tensor([x[2] for x in w]).unsqueeze(1),hz).squeeze(1).numpy()
        yv=np.array([x["yv"] for x in r])
        match = (len(r)==len(w)) and np.array_equal(y,yv)
        ok&=match
        save[n+"_p"]=np.array([x["p"] for x in r],dtype=np.float32); save[n+"_y"]=yv.astype(np.int8)
        save[n+"_ent"]=np.array([x[0] for x in w]); save[n+"_t"]=np.array([x[1] for x in w])
        save[n+"_tte"]=np.array([float(x[2]) for x in w])
        save[n+"_hent"]=np.array([x["hent"] for x in r]); save[n+"_brier"]=np.array([x["brier"] for x in r]); save[n+"_eps"]=np.array([x["eps"] for x in r])
    print(tag,a["seed"],a["d_model"],'n',len(rows),'labels_match_replay',ok,flush=True)
    if ok: np.savez_compressed(f'npz/{tag}.npz',**save)
