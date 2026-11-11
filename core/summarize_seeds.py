"""Aggregate analyze.summarize over seeds and model sizes."""

import glob
import json
import re
from collections import defaultdict

import numpy as np

from analyze import SIGNALS, summarize

agg = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob("results/*.json")):
    cond, d, s = re.match(r"results/(\w+?)_d(\d+)_s(\d+)\.json", f).groups()
    rows = json.load(open(f))["rows"]
    r = summarize(rows)
    key = (cond, int(d))
    for sig in SIGNALS:
        agg[key][f"pooled_rho_{sig}"].append(r["pooled"][sig]["rho"])
        agg[key][f"gain_{sig}"].append(r["pooled"][sig]["gain"])
        agg[key][f"within_rho_{sig}"].append(np.mean([v[f"rho_{sig}"] for v in r["per_ds"].values()]))
    agg[key]["mean_auroc"].append(np.mean([v["auroc"] for v in r["per_ds"].values()]))
    agg[key]["tcos"].append(np.mean([v["tcos"] for v in r["per_ds"].values()]))
    agg[key]["seeds"].append(int(s))


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():+.3f}" + (f"±{v.std(ddof=1):.3f}" if len(v) > 1 else "      ")


out = {}
for key in sorted(agg, key=lambda k: (k[1], k[0])):
    a = agg[key]
    print(f"\n### {key[0]:8s} d={key[1]:3d}  seeds={a['seeds']}  mean AUROC={fmt(a['mean_auroc'])}  target cos={fmt(a['tcos'])}")
    print(f"  {'signal':6s} {'pooled rho':>14s} {'within rho':>14s} {'abstain gain':>14s}")
    for sig in SIGNALS:
        print(f"  {sig:6s} {fmt(a[f'pooled_rho_{sig}']):>14s} {fmt(a[f'within_rho_{sig}']):>14s} {fmt(a[f'gain_{sig}']):>14s}")
    out[f"{key[0]}_d{key[1]}"] = {k: list(map(float, v)) for k, v in a.items()}
json.dump(out, open("summary.json", "w"), indent=1)
