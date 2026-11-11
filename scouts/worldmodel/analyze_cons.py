import glob, json, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R = "/home/claude/scout/worldmodel/results"
SIG = ["lat_c", "lat_z", "lat_n", "haz", "hazf", "rev", "hent", "eps"]
DS = ["FD001", "FD003", "MBA", "PSM", "ETTm1"]

def ece(p, y, nb=15):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0
    for i in range(nb):
        m = b == i
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def auc(y, p):
    return float(roc_auc_score(y.ravel(), p.ravel())) if 0 < y.mean() < 1 else float("nan")

res = defaultdict(dict)
for f in sorted(glob.glob(R + "/*.json")):
    cond, s = re.match(r".*/(\w+)_d256_s(\d+)\.json", f).groups()
    J = json.load(open(f)); rows = J["rows"]
    out = {"timing": J["timing"], "cons": J["cons_summary"], "ds": {}}
    for ds in DS:
        rs = [r for r in rows if r["ds"] == ds]
        P = np.array([r["p"] for r in rs]); Y = np.array([r["yv"] for r in rs]); K = P.shape[1]
        lo = int(0.75 * K)
        d = {"auroc": auc(Y, P), "auroc_long": auc(Y[:, lo:], P[:, lo:]), "auroc_last": auc(Y[:, -1], P[:, -1]),
             "brier": float(((P - Y) ** 2).mean()), "brier_long": float(((P[:, lo:] - Y[:, lo:]) ** 2).mean()),
             "ece": ece(P.ravel(), Y.ravel()), "ece_long": ece(P[:, lo:].ravel(), Y[:, lo:].ravel())}
        br = np.array([r["brier"] for r in rs])
        for sg in SIG:
            d["rho_" + sg] = (float(spearmanr([r[sg] for r in rs], br).correlation) if sg in rs[0] else float("nan"))
        out["ds"][ds] = d
    br = np.array([r["brier"] for r in rows])
    out["pooled"] = {sg: (float(spearmanr([r[sg] for r in rows], br).correlation) if sg in rows[0] else float("nan")) for sg in SIG}
    res[cond][int(s)] = out

def ms(v):
    v = np.asarray(v, float)
    k = int(np.sum(~np.isnan(v)))
    if k == 0: return "nan"
    return f"{np.nanmean(v):.4f}±{np.nanstd(v, ddof=1):.4f}" if k > 1 else f"{np.nanmean(v):.4f}(n=1)"

print("seeds:", {c: sorted(v) for c, v in res.items()})
print("\n== Q2 forecasting metrics (mean±sd over seeds) ==")
for met in ["auroc", "auroc_long", "auroc_last", "brier", "brier_long", "ece", "ece_long"]:
    print(f"-- {met}")
    for c in res:
        vals = {ds: [res[c][s]["ds"][ds][met] for s in sorted(res[c])] for ds in DS}
        mean5 = [np.nanmean([res[c][s]["ds"][ds][met] for ds in DS]) for s in sorted(res[c])]
        print(f"  {c:12s} " + "  ".join(f"{ds}={ms(vals[ds])}" for ds in DS) + f"  MEAN5={ms(mean5)}")
if "shared" in res and "shared_cons" in res:
    print("\n== paired diff shared_cons - shared per seed ==")
    for met in ["auroc", "auroc_long", "auroc_last", "brier", "brier_long", "ece", "ece_long"]:
        ss = sorted(set(res["shared"]) & set(res["shared_cons"]))
        print(f"  {met:11s} " + "  ".join(f"{ds}=" + ",".join(f"{res['shared_cons'][s]['ds'][ds][met]-res['shared'][s]['ds'][ds][met]:+.4f}" for s in ss) for ds in DS))

print("\n== Q1 consistency violation (per pair; mean over seeds) ==")
keys = ["viol_c", "unrel_c", "cos_c", "cos_c_unrel", "viol_z", "unrel_z", "tgt_gap_z", "tgt_unrel_z", "hsens_z", "tdrift_z", "naive_comp_z", "viol_n", "unrel_n", "tgt_gap_n", "tgt_unrel_n", "predB_err_n", "ctx_shift_n", "ctx_unrel_n", "haz_full", "haz_full_unrel", "rev", "rev_unrel"]
for c in res:
    for pair in ["8+8", "16+16", "32+32"]:
        print(f"-- {c} pair {pair}")
        for ds in DS:
            print(f"  {ds:6s} " + " ".join(f"{k}={ms([res[c][s]['cons'][ds]['pairs'][pair].get(k, np.nan) for s in sorted(res[c])])}" for k in keys))
print("\n== Q1 ratios viol/unrel (z), tgt_gap/tgt_unrel (z), haz_full/unrel; avg over 3 pairs, per seed then mean±sd ==")
for c in res:
    for ds in DS + ["ALL"]:
        o = []
        for a, b in [("viol_c", "unrel_c"), ("viol_z", "unrel_z"), ("tgt_gap_z", "tgt_unrel_z"), ("viol_n", "unrel_n"), ("tgt_gap_n", "tgt_unrel_n"), ("haz_full", "haz_full_unrel"), ("hsens_z", "unrel_z"), ("tdrift_z", "unrel_z"), ("naive_comp_z", "unrel_z")]:
            v = [np.mean([res[c][s]["cons"][d]["pairs"][p].get(a, np.nan) / res[c][s]["cons"][d]["pairs"][p].get(b, np.nan) for p in ["8+8", "16+16", "32+32"] for d in (DS if ds == "ALL" else [ds])]) for s in sorted(res[c])]
            o.append(f"{a}/{b}={ms(v)}")
        print(f"  {c:12s} {ds:6s} " + "  ".join(o))

print("\n== Q3 Spearman(signal, per-window Brier) ==")
for c in res:
    print(f"-- {c}")
    for sg in SIG:
        within = [np.mean([res[c][s]["ds"][ds]["rho_" + sg] for ds in DS]) for s in sorted(res[c])]
        print(f"  {sg:6s} pooled={ms([res[c][s]['pooled'][sg] for s in sorted(res[c])])}  within-mean={ms(within)}  " +
              " ".join(f"{ds}={ms([res[c][s]['ds'][ds]['rho_' + sg] for s in sorted(res[c])])}" for ds in DS))
print("\n== timing ==")
for c in res:
    for s in sorted(res[c]): print(c, s, res[c][s]["timing"])
