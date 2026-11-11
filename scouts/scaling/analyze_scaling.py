"""Aggregate h-AUROC / Brier vs params and vs number of pretraining domains. No training."""
import glob, json, re, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

PARAMS = {64: 0.159, 128: 0.573, 256: 2.161, 384: 4.765}
DS = ["FD001", "FD003", "MBA", "PSM", "ETTm1"]
# table[(cond, d, ds)][seed] = (auroc, brier)
T = defaultdict(dict)

def per_ds(rows):
    by = defaultdict(list)
    for r in rows: by[r["ds"]].append(r)
    out = {}
    for ds, rs in by.items():
        p = np.array([x for r in rs for x in r["p"]]); y = np.array([x for r in rs for x in r["yv"]])
        out[ds] = (float(roc_auc_score(y, p)), float(np.mean([r["brier"] for r in rs])))
    return out

for pat in ["/home/claude/HEPA/results/*.json", "/home/claude/scout/scaling/results/*.json"]:
    for f in sorted(glob.glob(pat)):
        m = re.match(r"(separate|shared)_d(\d+)_s(\d+)\.json", f.split("/")[-1])
        if not m: continue
        cond, d, s = m.group(1), int(m.group(2)), int(m.group(3))
        for ds, v in per_ds(json.load(open(f))["rows"]).items():
            T[(cond, d, ds)][s] = v
        cond5 = "N5" if cond == "shared" else "N1"
for f in sorted(glob.glob("/home/claude/scout/scaling/results/*.json")):
    name = f.split("/")[-1][:-5]
    j = json.load(open(f))
    if "res" not in j: continue
    m = re.match(r"dom(\d)_d(\d+)_s(\d+)", name)
    if m: cond = f"dom{m.group(1)}"
    elif name.startswith("sep_"): cond = "separate"
    elif name.startswith("shared5_d64_e40"): cond = "shared_e40"
    else: continue
    d, s = j["args"]["d_model"], j["args"]["seed"]
    for ds, v in j["res"].items():
        T[(cond, d, ds)][s] = (v["auroc"], v["brier"])

def ms(vals):
    v = np.asarray(vals, float)
    if len(v) == 0: return "   --   "
    return f"{v.mean():.3f}" + (f"±{v.std(ddof=1):.3f}" if len(v) > 1 else "      ") + f"(n={len(v)})"

def cell(cond, d, ds, k):
    return ms([v[k] for v in T.get((cond, d, ds), {}).values()])

def meanrow(cond, d, dss, k):
    seeds = set.intersection(*[set(T.get((cond, d, x), {})) for x in dss]) if all((cond, d, x) in T for x in dss) else set()
    return ms([np.mean([T[(cond, d, x)][s][k] for x in dss]) for s in sorted(seeds)])

for k, nm in [(0, "h-AUROC"), (1, "Brier")]:
    print(f"\n===== {nm} vs width (20 pretrain epochs, 1024 cuts/dataset/epoch) =====")
    print(f"{'cond':10s} {'d':>4s} {'params(M)':>9s} " + " ".join(f"{x:>17s}" for x in DS + ["mean5", "mean(FD001,MBA)"]))
    for cond in ["separate", "shared", "shared_e40"]:
        for d in [64, 128, 256, 384]:
            if not any((cond, d, x) in T for x in DS): continue
            print(f"{cond:10s} {d:4d} {PARAMS[d]:9.3f} " + " ".join(f"{cell(cond, d, x, k):>17s}" for x in DS)
                  + f" {meanrow(cond, d, DS, k):>17s} {meanrow(cond, d, ['FD001','MBA'], k):>17s}")
    print(f"\n----- paired gap shared - separate ({nm}), per seed then mean±sd -----")
    for d in [64, 128, 256]:
        line = f"d={d:3d} "
        for dss, lab in [([x], x) for x in DS] + [(DS, "mean5")]:
            seeds = sorted(set.intersection(*[set(T.get((c, d, x), {})) for c in ("separate", "shared") for x in dss]))
            g = [np.mean([T[("shared", d, x)][s][k] - T[("separate", d, x)][s][k] for x in dss]) for s in seeds]
            line += f" {lab}:{np.mean(g):+.3f}" + (f"±{np.std(g, ddof=1):.3f}" if len(g) > 1 else "") + f"(n={len(g)})"
        print(line)
    print(f"\n===== {nm} vs number of pretraining domains, d=128 (0.57M), eval on FD001 and MBA =====")
    print("N=1 separate; N=2 {FD001,MBA}; N=3 +FD003; N=5 +PSM,ETTm1 (the 'shared' runs)")
    for ds in ["FD001", "MBA"]:
        print(f"{ds:6s} " + "  ".join(f"N={n}: {cell(c, 128, ds, k)}" for n, c in [(1, "separate"), (2, "dom2"), (3, "dom3"), (5, "shared")]))
    print("paired vs N=1 (same seed):")
    for ds in ["FD001", "MBA"]:
        line = f"{ds:6s} "
        for n, c in [(2, "dom2"), (3, "dom3"), (5, "shared")]:
            seeds = sorted(set(T.get((c, 128, ds), {})) & set(T.get(("separate", 128, ds), {})))
            g = [T[(c, 128, ds)][s][k] - T[("separate", 128, ds)][s][k] for s in seeds]
            line += f" N={n}: " + (f"{np.mean(g):+.4f}" + (f"±{np.std(g, ddof=1):.4f}" if len(g) > 1 else "") + f"(n={len(g)})" if g else "--")
        print(line)

print("\n===== raw per-seed values (auroc, brier) =====")
for key in sorted(T):
    print(key, {s: (round(a, 4), round(b, 4)) for s, (a, b) in sorted(T[key].items())})
json.dump({"|".join(map(str, k)): v for k, v in T.items()}, open("/home/claude/scout/scaling/scaling_table.json", "w"), indent=1)
