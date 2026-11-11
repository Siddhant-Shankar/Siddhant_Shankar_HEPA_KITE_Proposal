import json, numpy as np, sys
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else "conformal_results.json")); NAMES="FD001,FD003,MBA,PSM,ETTm1".split(",")
conds=sorted({k.rsplit("_s",1)[0] for k in R}, key=lambda c:(len(c),c))
def ms(v): v=[x for x in v if x is not None]; return f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "  n/a   "
def get(c,n,path):
    o=[]
    for k in R:
        if k.rsplit("_s",1)[0]!=c: continue
        x=R[k][n]
        for p in path: x=x[p]
        o.append(x)
    return o
x=R[list(R)[0]]
print("windows/groups/frac-with-event:",{n:(x[n]["n"],x[n]["n_groups"],round(x[n]["frac_event_in_horizon"],3)) for n in NAMES})
for title,path in [("oracle lam (B)",["oracle_lam"]),("oracle tau (A)",["oracle_tau"]),("mean Brier",["brier_all"]),("Spearman hent~brier",["spearman_hent_brier"]),
  ("B cov random",["random","B_cov"]),("B cov-sd-over-splits random",["random","B_cov_sd"]),("B cov grouped",["grouped","B_cov"]),("B cov-sd-over-splits grouped",["grouped","B_cov_sd"]),("B min cov grouped",["grouped","B_cov_min"]),("B P(cov<0.88) grouped",["grouped","B_viol"]),
  ("B decisive in-domain grouped",["grouped","B_dec"]),("B cov past->future",["past_future","B_cov"]),
  ("B cov LODO pooled",["lodo_pooled","B_cov"]),("B decisive LODO pooled",["lodo_pooled","B_dec"]),("B lam LODO pooled",["lodo_pooled","B_lam"]),
  ("B cov LODO worst-src",["lodo_worst","B_cov"]),("B decisive LODO worst-src",["lodo_worst","B_dec"]),
  ("A cov random",["random","A_cov"]),("A cov grouped",["grouped","A_cov"]),("A cov-sd grouped",["grouped","A_cov_sd"]),("A P(cov<0.88) grouped",["grouped","A_viol"]),("A L/K in-domain grouped",["grouped","A_L"]),("A cov past->future",["past_future","A_cov"]),
  ("A cov LODO pooled",["lodo_pooled","A_cov"]),("A L/K LODO pooled",["lodo_pooled","A_L"]),("A cov LODO worst-src",["lodo_worst","A_cov"]),("A L/K LODO worst-src",["lodo_worst","A_L"]),
  ("SEL r*=.05 target risk",["selective","0.05","tgt_risk"]),("SEL r*=.05 target cov",["selective","0.05","tgt_cov"]),("SEL r*=.10 target risk",["selective","0.1","tgt_risk"]),("SEL r*=.10 target cov",["selective","0.1","tgt_cov"]),("SEL r*=.02 target risk",["selective","0.02","tgt_risk"]),("SEL r*=.02 target cov",["selective","0.02","tgt_cov"])]:
    print("\n== "+title+"  (mean±sd over seeds)")
    print(f"{'cond':11s}"+"".join(f"{n:>13s}" for n in NAMES)+"   | mean|gap| / spread")
    for c in conds:
        vals=[get(c,n,path) for n in NAMES]
        extra=""
        if "cov" in title and "sd" not in title and "SEL" not in title and "min" not in title and "P(" not in title:
            ns=len(vals[0]); gaps=[np.mean([abs(vals[i][s]-0.9) for i in range(5)]) for s in range(ns)]; worst=[min(vals[i][s] for i in range(5)) for s in range(ns)]
            extra=f"   | mean|cov-.9|={np.mean(gaps):.3f}±{np.std(gaps):.3f} worst={np.mean(worst):.3f}±{np.std(worst):.3f}"
        if "oracle" in title:
            ns=len(vals[0]); sp=[np.ptp([vals[i][s] for i in range(5)]) for s in range(ns)]; extra=f"   | range across domains={np.mean(sp):.3f}±{np.std(sp):.3f}"
        print(f"{c:11s}"+"".join(f"{ms(v):>13s}" for v in vals)+extra)
