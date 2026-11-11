"""Scaling scout: pretrain ONE trunk on a chosen group of domains, finetune/evaluate on a subset.

Reuses the functions of /home/claude/HEPA/exp_shared_eps.py unchanged (imported, not edited).
Per-domain pretraining budget is fixed: `cuts` cuts per dataset per epoch for `pre_epochs` epochs,
so a trunk pretrained on N domains takes N x the gradient steps of a single-domain trunk.
"""
import argparse, copy, json, sys, time, io, contextlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/HEPA")
import exp_shared_eps as E
from hepa.utils import get_context, set_seed
from hepa.training.finetune import finetune
from sklearn.metrics import roc_auc_score


class Tee(io.StringIO):
    def __init__(self, real):
        super().__init__(); self.real = real
    def write(self, s):
        self.real.write(s); self.real.flush(); return super().write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--pre-epochs", type=int, default=20)
    ap.add_argument("--cuts", type=int, default=1024)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-cap", type=int, default=2048)
    ap.add_argument("--test-windows", type=int, default=1200)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/home/claude/scout/scaling/results")
    a = ap.parse_args()
    grp, ev = a.pretrain.split(","), a.eval.split(",")
    assert all(n in grp for n in ev)
    data = {n: E.load(n) for n in grp}
    set_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    t0 = time.time()
    model = E.build_model({n: data[n][0]["n_channels"] for n in grp}, a.d_model, a.n_layers)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[{a.tag}] pretrain on {grp}  params={nparams}", flush=True)
    tee = Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        E.pretrain(model, data, grp, a.pre_epochs, a.cuts, a.seed)
    curve = [float(x) for x in re.findall(r"loss=([0-9.]+)", tee.getvalue())]
    t_pre = time.time() - t0
    pre = copy.deepcopy(model)
    res = {}
    for n in ev:
        b, norm = data[n]
        ctx, kmax = get_context(n), max(b["horizons"])
        stats = E.pretrain_stats(pre, b, n, norm, rng)
        ft = copy.deepcopy(pre)
        E.activate(ft, n, norm)
        tr = E.event_loader(b["ft_train"], ctx, kmax, a.ft_cap, rng, True)
        va = E.event_loader(b["ft_val"], ctx, kmax, a.ft_cap // 3, rng, False)
        print(f"[{a.tag}] finetune {n}", flush=True)
        finetune(ft, tr, va, b["horizons"], lr=1e-3, weight_decay=0.01, n_epochs=a.ft_epochs,
                 patience=5, device="cpu", early_stop=(norm != "none"))
        rows = E.eval_windows(pre, ft, b, n, norm, a.test_windows, stats, rng)
        p = np.array([x for r in rows for x in r["p"]]); y = np.array([x for r in rows for x in r["yv"]])
        res[n] = {"auroc": float(roc_auc_score(y, p)), "brier": float(np.mean([r["brier"] for r in rows])),
                  "n": len(rows), "tcos": rows[0]["tcos"], "hcos": rows[0]["hcos"]}
        print(f"[{a.tag}] {n} AUROC={res[n]['auroc']:.4f} Brier={res[n]['brier']:.4f}", flush=True)
    out = {"args": vars(a), "params": nparams, "pretrain_loss": curve, "t_pretrain": round(t_pre, 1),
           "t_total": round(time.time() - t0, 1), "res": res}
    Path(a.out).mkdir(exist_ok=True)
    json.dump(out, open(Path(a.out) / f"{a.tag}.json", "w"), indent=1)
    print(f"[{a.tag}] saved  t_pre={t_pre:.0f}s t_total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
