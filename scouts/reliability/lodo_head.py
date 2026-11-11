"""True unseen-domain test: head trained on 4 domains' labels (shared backbone), conformal calibration on those
4 domains' test windows, applied zero-shot to the 5th. alpha=0.10."""
import numpy as np, json, glob, os
NAMES="FD001,FD003,MBA,PSM,ETTm1".split(","); ALPHA=0.10; G=np.round(np.arange(0,1.0001,0.005),4)
def prep(z,n):
    p=z[n+"_p"].astype(np.float64); y=z[n+"_y"].astype(np.int64); K=p.shape[1]; s=np.abs(y-p)
    B=np.stack([(s>g).mean(1) for g in G],1)
    D=np.stack([(((p<=g)|(p>=1-g)) if g<0.5 else ((p<1-g)|(p>g))).mean(1) for g in G],1)
    A=np.zeros((len(p),len(G))); Lf=np.zeros_like(A)
    for j,g in enumerate(G):
        L=(p<g).sum(1); Lf[:,j]=L/K; A[:,j]=(L>=1)&(y[np.arange(len(p)),np.clip(L-1,0,K-1)]==1)
    return dict(B=B,D=D,A=A,Lf=Lf)
out={}
for seed in [42,123]:
    for n in NAMES:
        f=f"npz_head_lodo/shared-lodoH{n}_d256_s{seed}.npz"
        if not os.path.exists(f): continue
        z=np.load(f); P={m:prep(z,m) for m in NAMES}; src=[m for m in NAMES if m!=n]
        Bc=np.concatenate([P[m]["B"] for m in src]); Ac=np.concatenate([P[m]["A"] for m in src]); N=len(Bc)
        jB=np.where((N*Bc.mean(0)+1)/(N+1)<=ALPHA)[0][0]; jA=np.where((N*Ac.mean(0)+1)/(N+1)<=ALPHA)[0][-1]
        d=P[n]
        out[f"{n}_s{seed}"]=dict(B_cov=float(1-d["B"][:,jB].mean()),B_dec=float(d["D"][:,jB].mean()),A_cov=float(1-d["A"][:,jA].mean()),A_L=float(d["Lf"][:,jA].mean()),
            src_B_cov={m:float(1-P[m]["B"][:,jB].mean()) for m in src})
json.dump(out,open("lodo_head_results.json","w"),indent=1)
for k in ["B_cov","B_dec","A_cov","A_L"]:
    print(k," ".join(f"{n}={np.mean([out[x][k] for x in out if x.startswith(n+'_')]):.3f}±{np.std([out[x][k] for x in out if x.startswith(n+'_')]):.3f}" for n in NAMES),
          "| mean|cov-.9|=" + (f"{np.mean([abs(out[x][k]-0.9) for x in out]):.3f}" if "cov" in k else "-"))
