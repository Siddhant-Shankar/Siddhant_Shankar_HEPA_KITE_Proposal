"""Build tables.md + numbers.json from out/*.json (no training)."""
import json, glob, subprocess
import numpy as np
from collections import defaultdict
subprocess.run(["python3", "agg.py"], stdout=subprocess.DEVNULL, cwd="/home/claude/phase2")
A = json.load(open("agg.json")); G, P = A["groups"], A["paired_vs_rand"]
rows = [json.load(open(f)) for f in sorted(glob.glob("out/*.json"))]
def f(x, nd=4):
    m, s = x
    return f"{m:.{nd}f} ± {s:.{nd}f}" if s == s else f"{m:.{nd}f} (n=1)"
def sg(x, nd=4):
    m, s = x
    return f"{m:+.{nd}f} ± {s:.{nd}f}" if s == s else f"{m:+.{nd}f} (n=1)"
L = []
def table(title, keys, paired=True, note=""):
    L.append(f"\n**{title}**\n")
    if note: L.append(note + "\n")
    L.append("| dataset | condition | d | n (seeds) | pooled AUROC | mean per-horizon AUROC (repo `h_auroc`) | horizon-only pooled | Δ pooled vs rand (paired) | Δ per-horizon vs rand (paired) | seeds better than rand (per-h) | target cos init → post | target eff. rank (post) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        if k not in G: 
            L.append(f"| {k} | NOT RUN |||||||||||"); continue
        g = G[k]; ds, cond, d, tag = k.split("|")
        p = P.get(k)
        dp = sg(p["d_pooled_mean_sd"]) + f" (n={p['n']})" if p else "—"
        dh = sg(p["d_h_auroc_mean_sd"]) + f" (n={p['n']})" if p else "—"
        w = f"{p['wins_h']}/{p['n']}" if p else "—"
        L.append(f"| {ds} | {cond}{tag} | {d} | {g['n']} ({','.join(map(str,g['seeds']))}) | {f(g['pooled_auroc'])} | {f(g['h_auroc'])} | {g['horizon_only_pooled'][0]:.4f} | {dp} | {dh} | {w} | {g['target_cos_init'][0]:.4f} → {g['target_cos_post'][0]:.4f} | {g['target_effrank_post'][0]:.1f} |")
DS5 = ["FD001", "FD003", "MBA", "PSM", "ETTm1"]
table("Table 1 — Step 1/2: authors' pipeline, PROTOCOL (50 pretrain epochs, 30 finetune epochs, d=256), pretrained vs random-init encoder",
      [f"{ds}|{c}|256|" for ds in DS5 for c in ("pre", "rand")])
table("Table 2 — Step 3a/3b: 2x2 encoder x predictor-init (finetune restarted from the same checkpoint and RNG state as the `pre` run)",
      [f"{ds}|{c}|256|" for ds in ("FD001", "FD003", "MBA") for c in ("pre", "preE_newP", "randE_preP", "rand")])
table("Table 3 — Step 3c/3d: target-side regularisers (tr = var/cov penalty also on targets; sig = OUR crude SIGReg-style Epps-Pulley penalty on targets, not the paper's)",
      [f"FD001|{c}|256|" for c in ("pre", "tr", "rand")] + [f"FD003|{c}|256|" for c in ("pre", "tr", "sig", "rand")] +
      [f"MBA|{c}|256|" for c in ("pre", "tr", "sig", "rand")] + [f"MBA|{c}|256|_c2400" for c in ("pre", "tr", "sig")])
table("Table 4 — Step 4: width sweep, pretrained vs random (separate per-dataset encoders)",
      [f"{ds}|{c}|{d}|" for ds in ("FD001", "FD003", "MBA") for d in (64, 128, 256) for c in ("pre", "rand")])
# per-seed target cosine / eff rank for regulariser conditions (means hide sign flips)
L.append("\n**Table 5 — per-seed embedding diagnostics after pretraining (512 fixed pretraining cuts)**\n")
L.append("| run | target cos | target eff. rank (centred) | context cos | context eff. rank | predictor-output cos |")
L.append("|---|---|---|---|---|---|")
for r in rows:
    if r["d_model"] != 256 or r["cond"] in ("preE_newP", "randE_preP"): continue
    s = r["stats_post_pretrain"]
    L.append(f"| {r['dataset']} {r['cond']}{r.get('tag','')} s{r['seed']} | {s['target_cos']:.4f} | {s['target_effrank']:.1f} | {s['context_cos']:.4f} | {s['context_effrank']:.1f} | {s['pred_cos']:.4f} |")
# horizon bands
L.append("\n**Table 6 — per-horizon AUROC by horizon band (mean over valid horizons in each third of the grid, then mean ± sd over seeds), d=256**\n")
L.append("| dataset | condition | n | short third | middle third | long third |")
L.append("|---|---|---|---|---|---|")
bands = {}
GG = defaultdict(list)
for r in rows:
    if r["d_model"] == 256 and r["cond"] in ("pre", "rand") and r.get("tag", "") == "": GG[(r["dataset"], r["cond"])].append(r)
for (ds, c), rs in sorted(GG.items()):
    B = []
    for r in rs:
        a = np.array(r["auroc_per_k"], float); K = len(a)
        B.append([np.nanmean(a[:K//3]), np.nanmean(a[K//3:2*K//3]), np.nanmean(a[2*K//3:])])
    B = np.array(B); m = B.mean(0); sd = B.std(0, ddof=1) if len(B) > 1 else [float("nan")]*3
    bands[f"{ds}|{c}"] = {"n": len(B), "mean": m.tolist(), "sd": [float(x) for x in sd]}
    L.append(f"| {ds} | {c} | {len(B)} | " + " | ".join(f"{m[i]:.4f} ± {sd[i]:.4f}" for i in range(3)) + " |")
# timings
L.append("\n**Table 7 — measured wall-clock per run (seconds; 1 torch thread, two runs sharing 2 CPU cores)**\n")
L.append("| dataset | condition | d | params | pretrain s | finetune s | eval s | finetune epochs run (last epoch index per seed) |")
L.append("|---|---|---|---|---|---|---|---|")
for k in sorted(G):
    ds, cond, d, tag = k.split("|")
    if cond not in ("pre", "rand"): continue
    g = G[k]
    L.append(f"| {ds} | {cond}{tag} | {d} | {g['n_params']/1e6:.2f}M | {g['t_pretrain_s']:.0f} | {g['t_finetune_s']:.0f} | {g['t_eval_s']:.0f} | {g['ft_final_epoch']} |")
open("tables.md", "w").write("\n".join(L) + "\n")
json.dump({"groups": G, "paired_vs_rand": P, "horizon_bands": bands,
           "notes": {"sd": "sample sd (ddof=1) over seeds", "h_auroc": "repo hepa.evaluation.h_auroc = mean per-horizon AUROC",
                     "pooled_auroc": "AUROC over all (window, horizon) cells", "raw": "/home/claude/phase2/out/*.json"}},
          open("numbers.json", "w"), indent=1)
print("\n".join(L))
