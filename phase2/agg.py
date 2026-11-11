"""Aggregate out/*.json -> table + numbers.json fragments. No training."""
import json, glob, sys
import numpy as np
from collections import defaultdict
rows = [json.load(open(f)) for f in sorted(glob.glob("/home/claude/phase2/out/*.json"))]
G = defaultdict(list)
for r in rows:
    G[(r["dataset"], r["cond"], r["d_model"], r.get("tag", ""))].append(r)
def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
out = {}
print(f"{'dataset':6s} {'cond':11s} {'d':>3s} {'tag':7s} n  pooled         hAUROC(perh)   honly_pool  tcos_init->post     effrank_t  wvar   ft_ep  t_pre  t_ft")
for k in sorted(G):
    rs = sorted(G[k], key=lambda r: r["seed"])
    f = lambda key: ms([r[key] for r in rs])
    p, h, hb = f("pooled_auroc"), f("h_auroc"), f("horizon_only_pooled_auroc")
    ti = ms([r["stats_init"]["target_cos"] for r in rs]); tp = ms([r["stats_post_pretrain"]["target_cos"] for r in rs])
    er = ms([r["stats_post_pretrain"]["target_effrank"] for r in rs])
    cc = ms([r["stats_post_pretrain"]["context_cos"] for r in rs])
    wv = f("within_horizon_var_share"); br = f("brier")
    name = "|".join(map(str, k))
    out[name] = {"n": len(rs), "seeds": [r["seed"] for r in rs], "pooled_auroc": p, "h_auroc": h, "horizon_only_pooled": hb,
                 "target_cos_init": ti, "target_cos_post": tp, "target_effrank_post": er, "context_cos_post": cc,
                 "within_horizon_var_share": wv, "brier": br,
                 "per_seed_pooled": [r["pooled_auroc"] for r in rs], "per_seed_h_auroc": [r["h_auroc"] for r in rs],
                 "ft_final_epoch": [r["ft_final_epoch"] for r in rs],
                 "t_pretrain_s": ms([r["t_pretrain_s"] for r in rs])[0], "t_finetune_s": ms([r["t_finetune_s"] for r in rs])[0],
                 "t_eval_s": ms([r["t_eval_s"] for r in rs])[0], "n_params": rs[0]["n_params"],
                 "enc_unchanged": [r["encoder_unchanged_from_init"] for r in rs]}
    print(f"{k[0]:6s} {k[1]:11s} {k[2]:3d} {k[3]:7s} {len(rs)}  {p[0]:.4f}±{p[1]:.4f}  {h[0]:.4f}±{h[1]:.4f}  {hb[0]:.4f}      {ti[0]:.4f}->{tp[0]:.4f}  {er[0]:6.1f}  {wv[0]:.3f}  "
          f"{[r['ft_final_epoch'] for r in rs]}  {out[name]['t_pretrain_s']:.0f}  {out[name]['t_finetune_s']:.0f}")
# paired differences vs rand (same dataset, d, seed)
print("\npaired (cond - rand), same seed:")
diffs = {}
for k in sorted(G):
    ds, cond, d, tag = k
    if cond == "rand": continue
    base = {r["seed"]: r for r in G.get((ds, "rand", d, ""), [])}
    pr = [(r, base[r["seed"]]) for r in sorted(G[k], key=lambda r: r["seed"]) if r["seed"] in base]
    if not pr: continue
    dp = [a["pooled_auroc"] - b["pooled_auroc"] for a, b in pr]; dh = [a["h_auroc"] - b["h_auroc"] for a, b in pr]
    name = "|".join(map(str, k))
    diffs[name] = {"n": len(pr), "seeds": [a["seed"] for a, _ in pr], "d_pooled": dp, "d_h_auroc": dh,
                   "d_pooled_mean_sd": ms(dp), "d_h_auroc_mean_sd": ms(dh), "wins_h": int(sum(x > 0 for x in dh)), "wins_pooled": int(sum(x > 0 for x in dp))}
    print(f"  {name:32s} n={len(pr)} d_pooled={ms(dp)[0]:+.4f}±{ms(dp)[1]:.4f} {['%+.4f'%x for x in dp]}  d_hAUROC={ms(dh)[0]:+.4f}±{ms(dh)[1]:.4f} {['%+.4f'%x for x in dh]}")
json.dump({"groups": out, "paired_vs_rand": diffs}, open("/home/claude/phase2/agg.json", "w"), indent=1)
