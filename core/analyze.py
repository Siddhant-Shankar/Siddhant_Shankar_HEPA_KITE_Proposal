"""Analyze per-window results from exp_shared_eps.py."""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SIGNALS = ["eps", "epsz", "epsc", "atyp", "hent"]
COVS = np.linspace(0.1, 1.0, 46)


def aurc(signal, brier):
    order = np.argsort(signal)  # keep lowest-signal (most trusted) first
    b = brier[order]
    n = len(b)
    risks = [b[: max(1, int(round(c * n)))].mean() for c in COVS]
    return float(np.mean(risks)), risks


def add_epsz(rows):
    """Horizon-adjusted eps: z-score each Delta_t column over all windows, then average
    the columns each window has. Removes the bias that windows close to an event only
    have short horizons (and short-horizon eps is smaller)."""
    E = np.array([[np.nan if v is None else v for v in r["eps_dt"]] for r in rows], dtype=float)
    mu, sd = np.nanmean(E, 0), np.nanstd(E, 0) + 1e-12
    Z = (E - mu) / sd
    z = np.nanmean(Z, 1)
    for r, v in zip(rows, z):
        r["epsz"] = float(v)


def summarize(rows):
    add_epsz(rows)
    for r in rows:
        r.setdefault("epsc", float("nan"))
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["ds"]].append(r)
    per_ds = {}
    for ds, rs in by.items():
        p = np.array([x for r in rs for x in r["p"]])
        y = np.array([x for r in rs for x in r["yv"]])
        eps = np.array([r["eps"] for r in rs])
        br = np.array([r["brier"] for r in rs])
        d = {"n": len(rs), "brier": float(br.mean()), "eps_mean": float(eps.mean()),
             "hcos": rs[0].get("hcos", float("nan")), "tcos": rs[0].get("tcos", float("nan"))}
        d["auroc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
        for s in SIGNALS:
            v = np.array([r[s] for r in rs])
            d[f"rho_{s}"] = float(spearmanr(v, br).correlation)
        per_ds[ds] = d
    out["per_ds"] = per_ds
    br = np.array([r["brier"] for r in rows])
    pooled = {}
    for s in SIGNALS:
        v = np.array([r[s] for r in rows])
        a, curve = aurc(v, br)
        pooled[s] = {"rho": float(spearmanr(v, br).correlation), "aurc": a, "curve": curve}
    rnd = float(br.mean())
    orc, orc_curve = aurc(br, br)
    pooled["random_aurc"] = rnd
    pooled["oracle_aurc"] = orc
    for s in SIGNALS:
        pooled[s]["gain"] = (rnd - pooled[s]["aurc"]) / (rnd - orc)
    out["pooled"] = pooled
    out["within_mean_rho_eps"] = float(np.mean([d["rho_eps"] for d in per_ds.values()]))
    out["within_mean_rho_epsz"] = float(np.mean([d["rho_epsz"] for d in per_ds.values()]))
    dsm = list(per_ds.values())
    out["between_rho_eps"] = float(spearmanr([d["eps_mean"] for d in dsm], [d["brier"] for d in dsm]).correlation)
    return out


def main(pattern="results/*.json"):
    res = {}
    for f in sorted(glob.glob(pattern)):
        rows = json.load(open(f))["rows"]
        tag = f.split("/")[-1][:-5]
        res[tag] = summarize(rows)
        s = res[tag]
        print(f"\n=== {tag}  (n={len(rows)})")
        for ds, d in s["per_ds"].items():
            print(f"  {ds:6s} AUROC={d['auroc']:.3f} Brier={d['brier']:.4f} eps={d['eps_mean']:.2e} "
                  f"cos(h)={d['hcos']:.3f} cos(h*)={d['tcos']:.3f} "
                  f"rho: eps={d['rho_eps']:+.3f} epsz={d['rho_epsz']:+.3f} epsc={d.get('rho_epsc',float('nan')):+.3f} atyp={d['rho_atyp']:+.3f} hent={d['rho_hent']:+.3f}")
        p = s["pooled"]
        print(f"  within-dataset mean rho(eps,Brier) = {s['within_mean_rho_eps']:+.3f}   epsz: {s['within_mean_rho_epsz']:+.3f}")
        print(f"  between-dataset rho(mean eps, mean Brier) = {s['between_rho_eps']:+.3f}  (n={len(s['per_ds'])} datasets, weak)")
        for sig in SIGNALS:
            print(f"  POOLED {sig:5s}: rho={p[sig]['rho']:+.3f}  AURC={p[sig]['aurc']:.4f}  gain={p[sig]['gain']:+.3f}")
        print(f"  random AURC={p['random_aurc']:.4f}  oracle AURC={p['oracle_aurc']:.4f}")
    json.dump(res, open("analysis.json", "w"))
    return res


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
