import numpy as np, glob, os
from sklearn.metrics import roc_auc_score
NAMES="FD001,FD003,MBA,PSM,ETTm1".split(",")
def per_h(p,y):
    a=[roc_auc_score(y[:,k],p[:,k]) for k in range(p.shape[1]) if 0<y[:,k].mean()<1]; return np.mean(a)
for f in sorted(glob.glob("npz/s*_d256_s*.npz"))+sorted(glob.glob("npz_head/*.npz")):
    z=np.load(f); o=[]
    for n in NAMES:
        p,y=z[n+"_p"],z[n+"_y"]; o.append(f"{n} pooled={roc_auc_score(y.ravel(),p.ravel()):.3f} perH={per_h(p,y):.3f} brier={z[n+'_brier'].mean():.3f}")
    print(os.path.basename(f)); print("   "+" | ".join(o))
