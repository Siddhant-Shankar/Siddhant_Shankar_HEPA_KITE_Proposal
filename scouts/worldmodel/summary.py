import glob, json, re
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
R = "/home/claude/scout/worldmodel/results"
DS = ["FD001", "FD003", "MBA", "PSM", "ETTm1"]; PR = ["8+8", "16+16", "32+32"]
def ece(p, y, nb=15):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum((b == i).mean() * abs(p[b == i].mean() - y[b == i].mean()) for i in range(nb) if (b == i).any()))
D = {}
for f in sorted(glob.glob(R + "/*.json")):
    c, s = re.match(r".*/(\w+)_d256_s(\d+)\.json", f).groups(); D[(c, int(s))] = json.load(open(f))
conds = ["separate", "shared", "shared_cons"]; seeds = sorted({s for _, s in D})
def ms(v):
    v = np.array([x for x in v if x == x], float)
    return "n/a" if len(v) == 0 else (f"{v.mean():.3f}±{v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}(1 seed)")
def met(J, ds):
    rs = [r for r in J["rows"] if r["ds"] == ds]
    P = np.array([r["p"] for r in rs]); Y = np.array([r["yv"] for r in rs]); lo = int(.75 * P.shape[1])
    return {"auroc": roc_auc_score(Y.ravel(), P.ravel()), "auroc_long": roc_auc_score(Y[:, lo:].ravel(), P[:, lo:].ravel()),
            "brier": ((P - Y) ** 2).mean(), "brier_long": ((P[:, lo:] - Y[:, lo:]) ** 2).mean(),
            "ece": ece(P.ravel(), Y.ravel()), "ece_long": ece(P[:, lo:].ravel(), Y[:, lo:].ravel())}
M = {k: {ds: met(J, ds) for ds in DS} for k, J in D.items()}
print("seeds", seeds)
print("\nQ2: per-dataset metrics, mean±sd over seeds; then paired diff cons-shared per seed")
for m in ["auroc", "auroc_long", "brier", "brier_long", "ece", "ece_long"]:
    print("--", m)
    for c in conds:
        print(f"  {c:12s}", " ".join(f"{ds}={ms([M[(c, s)][ds][m] for s in seeds if (c, s) in M])}" for ds in DS),
              "MEAN5=" + ms([np.mean([M[(c, s)][ds][m] for ds in DS]) for s in seeds if (c, s) in M]))
    print("  diff cons-shared", " ".join(f"{ds}=" + "/".join(f"{M[('shared_cons', s)][ds][m] - M[('shared', s)][ds][m]:+.3f}" for s in seeds) for ds in DS),
          "MEAN5=" + "/".join(f"{np.mean([M[('shared_cons', s)][ds][m] - M[('shared', s)][ds][m] for ds in DS]):+.3f}" for s in seeds))
    print("  seed spread |s42-s123| shared", " ".join(f"{ds}={abs(M[('shared', seeds[0])][ds][m] - M[('shared', seeds[-1])][ds][m]):.3f}" for ds in DS))
print("\nQ1: ratios (mean over 3 pairs), mean±sd over seeds")
RAT = [("viol_z", "unrel_z"), ("viol_c", "unrel_c"), ("hsens_z", "unrel_z"), ("tdrift_z", "unrel_z"), ("tgt_gap_z", "tgt_unrel_z"), ("haz_full", "haz_full_unrel"), ("rev", "rev_unrel"), ("naive_comp_z", "unrel_z")]
for c in conds:
    for ds in DS + ["ALL"]:
        o = []
        for a, b in RAT:
            v = []
            for s in seeds:
                if (c, s) not in D: continue
                cs = D[(c, s)]["cons_summary"]
                x = [cs[d]["pairs"][p][a] / cs[d]["pairs"][p][b] for d in (DS if ds == "ALL" else [ds]) for p in PR if a in cs[d]["pairs"][p]]
                if x: v.append(np.mean(x))
            o.append(f"{a}/{b.split('_')[-1] if 'unrel' in b else b}={ms(v)}")
        print(f"  {c:12s} {ds:6s}", "  ".join(o))
print("  cos_c (centered cosine, same endpoint) vs unrelated, ALL datasets/pairs:")
for c in conds:
    for k in ["cos_c", "cos_c_unrel"]:
        print(f"    {c:12s} {k} =", ms([np.mean([D[(c, s)]["cons_summary"][d]["pairs"][p][k] for d in DS for p in PR]) for s in seeds if (c, s) in D and "cos_c" in D[(c, s)]["cons_summary"]["FD001"]["pairs"]["8+8"]]),
              " per-ds:", " ".join(f"{d}={ms([np.mean([D[(c, s)]['cons_summary'][d]['pairs'][p][k] for p in PR]) for s in seeds if (c, s) in D and 'cos_c' in D[(c, s)]['cons_summary'][d]['pairs']['8+8']])}" for d in DS))
print("\nQ3: Spearman(signal, per-window Brier): pooled | within-dataset mean | per dataset")
for c in conds:
    print("--", c)
    for sg in ["hent", "hazf", "haz", "rev", "lat_c", "lat_z", "lat_n", "eps"]:
        po, wi, per = [], [], {d: [] for d in DS}
        for s in seeds:
            if (c, s) not in D or sg not in D[(c, s)]["rows"][0]: continue
            rows = D[(c, s)]["rows"]
            po.append(spearmanr([r[sg] for r in rows], [r["brier"] for r in rows]).correlation)
            w = []
            for d in DS:
                rs = [r for r in rows if r["ds"] == d]
                rho = spearmanr([r[sg] for r in rs], [r["brier"] for r in rs]).correlation; per[d].append(rho); w.append(rho)
            wi.append(np.nanmean(w))
        print(f"  {sg:6s} pooled={ms(po)}  within={ms(wi)}  ", " ".join(f"{d}={ms(per[d])}" for d in DS))
print("\ntiming (s):", {f"{c}_s{s}": {k: v for k, v in D[(c, s)]["timing"].items() if k.startswith("pre")} for (c, s) in D if c != "separate"})
print("target-collapse diag: tgt_unrel_z at 16+16:", {f"{c}_s{s}": {d: round(D[(c, s)]["cons_summary"][d]["pairs"]["16+16"]["tgt_unrel_z"], 3) for d in DS} for (c, s) in D})
