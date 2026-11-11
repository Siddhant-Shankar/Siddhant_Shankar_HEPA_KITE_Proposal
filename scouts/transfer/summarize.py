import glob, json, sys
from collections import defaultdict
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/scout/transfer/results"
rows = [r for f in sorted(glob.glob(d + "/*_s*.json")) for r in json.load(open(f))["rows"]]
by = defaultdict(list)
for r in rows:
    by[(r["heldout"], r["frac"], r["cond"])].append(r)
for H in sorted({k[0] for k in by}):
    for frac in sorted({k[1] for k in by if k[0] == H}, reverse=True):
        print(f"\n== held-out {H}  label frac {frac}  (train windows {by[(H,frac,'INDOM')][0]['n_train_windows']}, entities {by[(H,frac,'INDOM')][0]['n_train_entities']})")
        ref = {r["seed"]: r["auroc"] for r in by[(H, frac, "INDOM")]}
        for (h, f, c), rs in by.items():
            if h != H or f != frac: continue
            a = np.array([r["auroc"] for r in rs]); b = np.array([r["brier"] for r in rs])
            dlt = np.array([r["auroc"] - ref[r["seed"]] for r in rs])
            print(f"  {c:36s} AUROC {a.mean():.4f} ± {a.std(ddof=1) if len(a)>1 else float('nan'):.4f}  per-seed {[round(x,4) for x in a]}  "
                  f"Δ vs INDOM {dlt.mean():+.4f} {[round(x,4) for x in dlt]}  Brier {b.mean():.4f}  n={len(a)}")
