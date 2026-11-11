"""Conformal risk control (CRC) + selective prediction on saved HEPA survival CDFs. No training.
Task B  horizon-wise label sets C_lam(p)={y:|y-p|<=lam}; window loss = fraction of horizons whose label is not covered.
Task A  safe horizon L_tau = #{k: p_k<tau}; claim "no event within next L steps"; miss if event by horizon L.
alpha = 0.10 for both. CRC rule: (n*Rhat+1)/(n+1) <= alpha.
"""
import numpy as np, json, glob, sys, itertools
NAMES="FD001,FD003,MBA,PSM,ETTm1".split(",")
ALPHA=0.10
G=np.round(np.arange(0,1.0001,0.005),4)
RUNS={"separate":[42,123,456],"shared":[42,123,456],"separateTR":[42,123],"sharedTR":[42,123]}
NPZ="npz"; OUT="conformal_results.json"
if len(sys.argv)>1:
    NPZ=sys.argv[1]; OUT=sys.argv[2]; RUNS={}
    for f in sorted(glob.glob(NPZ+"/*.npz")):
        import os; c,sd=os.path.basename(f)[:-4].split("_d256_s"); RUNS.setdefault(c,[]).append(int(sd))
def prep(z,n):
    p=z[n+"_p"].astype(np.float64); y=z[n+"_y"].astype(np.int64); K=p.shape[1]
    s=np.abs(y-p)
    B=np.stack([(s>g).mean(1) for g in G],1)                  # loss, nonincreasing in lam
    def dec(g):
        return ((p<=g)|(p>=1-g)) if g<0.5 else ((p<1-g)|(p>g))
    D=np.stack([dec(g).mean(1) for g in G],1)                 # singleton-set rate
    A=np.zeros((len(p),len(G))); Lf=np.zeros_like(A)
    for j,g in enumerate(G):
        L=(p<g).sum(1); Lf[:,j]=L/K
        idx=np.clip(L-1,0,K-1); A[:,j]=(L>=1)&(y[np.arange(len(p)),idx]==1)
    return dict(B=B,D=D,A=A,Lf=Lf,ent=z[n+"_ent"],t=z[n+"_t"],hent=z[n+"_hent"],brier=z[n+"_brier"],n=len(p),K=K,
                pos=float(y.max(1).mean()))
def crcB(Bc,w=None):
    n=len(Bc); r=(n*Bc.mean(0)+1)/(n+1) if w is None else None
    ok=np.where(r<=ALPHA)[0]; return ok[0] if len(ok) else len(G)-1
def crcA(Ac):
    n=len(Ac); r=(n*Ac.mean(0)+1)/(n+1)
    ok=np.where(r<=ALPHA)[0]; return ok[-1] if len(ok) else 0
def groups(d,n,nblk=10):
    if n.startswith("FD"): return d["ent"]
    t=d["t"]; edges=np.linspace(t.min(),t.max()+1,nblk+1); return np.searchsorted(edges,t,side="right")-1
def evalB(d,j,idx): return 1-d["B"][idx,j].mean(), d["D"][idx,j].mean()
def evalA(d,j,idx): return 1-d["A"][idx,j].mean(), d["Lf"][idx,j].mean()
out={}
for cond,seeds in RUNS.items():
    for seed in seeds:
        z=np.load(f"{NPZ}/{cond}_d256_s{seed}.npz"); P={n:prep(z,n) for n in NAMES}; res={}
        rng=np.random.RandomState(0)
        for n in NAMES:
            d=P[n]; N=d["n"]; g=groups(d,n); ug=np.unique(g); r={"n":N,"n_groups":len(ug),"frac_event_in_horizon":d["pos"]}
            # oracle (in-sample) thresholds: the threshold each domain "needs"
            jB=np.where(d["B"].mean(0)<=ALPHA)[0][0]; jA=np.where(d["A"].mean(0)<=ALPHA)[0][-1]
            r["oracle_lam"]=float(G[jB]); r["oracle_tau"]=float(G[jA])
            for mode in ["random","grouped"]:
                cb,db,ca,la=[],[],[],[]
                for rep in range(200):
                    if mode=="random":
                        perm=rng.permutation(N); ci,ti=perm[:N//2],perm[N//2:]
                    else:
                        pg=rng.permutation(ug); cg=pg[:len(ug)//2]; m=np.isin(g,cg); ci,ti=np.where(m)[0],np.where(~m)[0]
                    j=crcB(d["B"][ci]); c,e=evalB(d,j,ti); cb.append(c); db.append(e)
                    j=crcA(d["A"][ci]); c,e=evalA(d,j,ti); ca.append(c); la.append(e)
                r[mode]=dict(B_cov=float(np.mean(cb)),B_cov_sd=float(np.std(cb)),B_cov_min=float(np.min(cb)),B_viol=float(np.mean(np.array(cb)<1-ALPHA-0.02)),B_dec=float(np.mean(db)),
                             A_cov=float(np.mean(ca)),A_cov_sd=float(np.std(ca)),A_cov_min=float(np.min(ca)),A_viol=float(np.mean(np.array(ca)<1-ALPHA-0.02)),A_L=float(np.mean(la)))
            # past -> future (first half of time / first half of entities calibrates)
            if n.startswith("FD"): m=d["ent"]<np.median(ug)
            else: m=d["t"]<np.median(d["t"])
            ci,ti=np.where(m)[0],np.where(~m)[0]
            j=crcB(d["B"][ci]); c,e=evalB(d,j,ti); j2=crcA(d["A"][ci]); c2,e2=evalA(d,j2,ti)
            r["past_future"]=dict(B_cov=float(c),B_dec=float(e),A_cov=float(c2),A_L=float(e2))
            # LODO: calibrate on the 4 other domains pooled; and conservative max-of-source-domain thresholds
            src=[m_ for m_ in NAMES if m_!=n]; allidx=np.arange(N)
            j=crcB(np.concatenate([P[m_]["B"] for m_ in src])); c,e=evalB(d,j,allidx)
            j2=crcA(np.concatenate([P[m_]["A"] for m_ in src])); c2,e2=evalA(d,j2,allidx)
            r["lodo_pooled"]=dict(B_cov=float(c),B_dec=float(e),B_lam=float(G[j]),A_cov=float(c2),A_L=float(e2),A_tau=float(G[j2]))
            j=max(crcB(P[m_]["B"]) for m_ in src); c,e=evalB(d,j,allidx)
            j2=min(crcA(P[m_]["A"]) for m_ in src); c2,e2=evalA(d,j2,allidx)
            r["lodo_worst"]=dict(B_cov=float(c),B_dec=float(e),B_lam=float(G[j]),A_cov=float(c2),A_L=float(e2),A_tau=float(G[j2]))
            # in-domain full-sample reference efficiency at oracle threshold
            r["oracle_B_dec"]=float(d["D"][:,jB].mean()); r["oracle_A_L"]=float(d["Lf"][:,jA].mean())
            # selective prediction: accept if hent<=theta; theta = largest s.t. source selective Brier <= r*
            sel={}
            for rstar in [0.02,0.05,0.10]:
                def pick(h,b):
                    o=np.argsort(h); cum=np.cumsum(b[o])/np.arange(1,len(o)+1)
                    ok=np.where(cum<=rstar)[0]; return (h[o][ok[-1]] if len(ok) else -1.0)
                hs=np.concatenate([P[m_]["hent"] for m_ in src]); bs=np.concatenate([P[m_]["brier"] for m_ in src])
                th=pick(hs,bs); acc=d["hent"]<=th
                # in-domain reference: threshold from random half, test on other half
                sel[str(rstar)]=dict(theta=float(th),tgt_cov=float(acc.mean()),tgt_risk=(float(d["brier"][acc].mean()) if acc.any() else None),
                                     src_cov=float((hs<=th).mean()))
            r["selective"]=sel; r["brier_all"]=float(d["brier"].mean())
            from scipy.stats import spearmanr
            r["spearman_hent_brier"]=float(spearmanr(d["hent"],d["brier"]).correlation)
            res[n]=r
        out[f"{cond}_s{seed}"]=res
json.dump(out,open(OUT,"w"),indent=1)
print("saved",OUT)
