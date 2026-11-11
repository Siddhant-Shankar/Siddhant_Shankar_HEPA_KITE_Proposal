"""World-model consistency pilot for HEPA's predictor g(h, dt).

Copies the protocol of /home/claude/HEPA/exp_shared_eps.py (imports its functions unchanged) and adds
  (1) an optional temporal-consistency loss in pretraining (conds ending in "_cons")
  (2) per-window consistency-violation measurements at eval time.

Why not g(g(h,d1),d2)?  g maps a CONTEXT-encoder state h_t to the TARGET-encoder embedding of the
interval (t, t+dt]. Its output is not a context state, so feeding it back into g is ill-typed.
Closest well-defined test (uses real data, all quantities available at time t):
    A = g(h_{t-d1}, d1+d2)   forecast made d1 steps ago for the endpoint t+d2
    B = g(h_t, d2)           forecast made now for the same endpoint
Caveat: the true targets differ (interval (t-d1,t+d2] vs (t,t+d2]); we report that true-target gap as the floor.
Probability-space version (exact Chapman-Kolmogorov on the finetuned survival surface):
    q_k = (P_{t-d1}(k+d1) - P_{t-d1}(d1)) / (1 - P_{t-d1}(d1))   vs   P_t(k)
"""
from __future__ import annotations
import argparse, copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/HEPA")
import exp_shared_eps as E
from hepa.training.finetune import EventDataset, finetune
from hepa.training.losses import vicreg_loss
from hepa.utils import get_context, set_seed

PAIRS = [(8, 8), (16, 16), (32, 32)]  # (d1, d2)


def sample_cuts_full(seqs, n, ctx, dt_max, rng):
    """Identical RNG consumption to E.sample_cuts, but also returns (seq, t)."""
    keys = list(seqs)
    lens = np.array([len(seqs[k]) for k in keys], dtype=float)
    pick = rng.choice(len(keys), size=n, p=lens / lens.sum())
    out = []
    for i in pick:
        seq = seqs[keys[i]]
        T = len(seq)
        hi = min(dt_max, T - 1)
        dt = max(1, min(int(math.exp(rng.uniform(0, math.log(hi)))), hi))
        t = int(rng.randint(1, T - dt + 1))
        out.append((seq[max(0, t - ctx):t], seq[t:t + dt], dt, seq, t))
    return out


def pretrain(model, data, names, epochs, cuts_per_ds, seed, alpha=0.04, bs=64, lam=0.0):
    rng = np.random.RandomState(seed)
    rng2 = np.random.RandomState(seed + 9999)  # only used by the consistency term
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        batches = []
        for n in names:
            b, norm = data[n]
            cuts = sample_cuts_full(b["pretrain_seqs"], cuts_per_ds, get_context(n), max(b["horizons"]), rng)
            for i in range(0, len(cuts), bs):
                batches.append((n, norm, cuts[i:i + bs]))
        rng.shuffle(batches)
        losses, closses = [], []
        for n, norm, cb in batches:
            E.activate(model, n, norm)
            c, cm = E.pad([x[0] for x in cb])
            tg, tm = E.pad([x[1] for x in cb])
            dt = torch.tensor([x[2] for x in cb], dtype=torch.float32)
            hp, ht = model.pretrain_forward(c, tg, dt, cm, tm)
            loss = vicreg_loss(hp, ht, alpha)
            if lam > 0:
                ok = [j for j, x in enumerate(cb) if x[2] >= 2]
                if ok:
                    ctx = get_context(n)
                    d1 = [int(rng2.randint(1, cb[j][2])) for j in ok]  # 1..dt-1
                    c2, cm2 = E.pad([cb[j][3][max(0, cb[j][4] + d - ctx):cb[j][4] + d] for j, d in zip(ok, d1)])
                    dt2 = torch.tensor([cb[j][2] - d for j, d in zip(ok, d1)], dtype=torch.float32)
                    with torch.no_grad():  # the later, better-informed forecast is the (stop-grad) target
                        hp2 = model.predictor(model.encoder(c2, cm2), dt2)
                    lc = F.l1_loss(F.normalize(hp[ok], dim=-1), F.normalize(hp2, dim=-1))
                    loss = loss + lam * lc
                    closses.append(lc.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"    pretrain ep{ep:02d} loss={np.mean(losses):.4f}" + (f" cons={np.mean(closses):.5f}" if closses else ""), flush=True)


@torch.no_grad()
def enc_all(model, ctxs, bs=128):
    out = []
    for i in range(0, len(ctxs), bs):
        c, cm = E.pad(ctxs[i:i + bs])
        out.append(model.encoder(c, cm))
    return torch.cat(out)


@torch.no_grad()
def tgt_all(model, segs, bs=128):
    out = []
    for i in range(0, len(segs), bs):
        c, cm = E.pad(segs[i:i + bs])
        out.append(model.target_encoder(c, cm))
    return torch.cat(out)


@torch.no_grad()
def cdf_from_h(model, h, hz, bs=64):
    out = []
    K = len(hz)
    for i in range(0, len(h), bs):
        hb = h[i:i + bs]
        B, d = hb.shape
        hp = model.predictor(hb.unsqueeze(1).expand(B, K, d).reshape(B * K, d),
                             hz.unsqueeze(0).expand(B, K).reshape(B * K)).view(B, K, d)
        lam = torch.sigmoid(model.event_head(hp))
        out.append(1 - torch.cumprod(1 - lam, dim=-1))
    return torch.cat(out).numpy()


@torch.no_grad()
def consistency_eval(pre, ft, b, n, norm, max_windows, stats, rng_state, rows, seed):
    """Adds per-window violation signals to rows (same windows as E.eval_windows) and returns dataset summary."""
    rng = np.random.RandomState(); rng.set_state(rng_state)
    ctx = get_context(n)
    hz = torch.tensor(b["horizons"], dtype=torch.float32)
    K = len(hz)
    wins = []
    for ei, e in enumerate(b["ft_test"]):
        T = len(e["test"])
        if T <= 129:
            continue
        for t in range(128, T - 1):
            wins.append((ei, t))
    sel = rng.choice(len(wins), size=min(max_windows, len(wins)), replace=False)
    wins = [wins[i] for i in sorted(sel)]
    N = len(wins)
    assert N == len(rows)
    for m in (pre, ft):
        E.activate(m, n, norm); m.eval()
    mu, sd, tmu, tsd = stats
    X = lambda ei: b["ft_test"][ei]["test"]
    zd = lambda a, c: ((a - c).abs() / sd).mean(1).numpy()
    zdt = lambda a, c: ((a - c).abs() / tsd).mean(1).numpy()
    nd = lambda a, c: (F.normalize(a, dim=-1) - F.normalize(c, dim=-1)).abs().mean(1).numpy()
    prng = np.random.RandomState(seed + 7)
    perm = prng.permutation(N)
    g = lambda h, d: pre.predictor(h, torch.full((len(h),), float(d)))

    H0 = enc_all(pre, [X(ei)[max(0, t - ctx):t] for ei, t in wins])
    P0 = cdf_from_h(ft, H0, hz)
    chk = float(np.abs(P0 - np.array([r["p"] for r in rows])).max())
    summ = {"n": N, "cdf_check_maxabs": chk, "pairs": {}}
    lat_c = np.zeros(N); lat_z = np.zeros(N); lat_n = np.zeros(N); haz = np.zeros(N); hazf = np.zeros(N); rev = np.zeros(N)
    for d1, d2 in PAIRS:
        Hb = enc_all(pre, [X(ei)[max(0, t - d1 - ctx):t - d1] for ei, t in wins])
        A, Bv = g(Hb, d1 + d2), g(H0, d2)
        vz, vn = zd(A, Bv), nd(A, Bv)
        lat_z += vz / len(PAIRS); lat_n += vn / len(PAIRS)
        s = {
            "viol_z": float(vz.mean()), "viol_n": float(vn.mean()),
            "unrel_z": float(zd(A, Bv[perm]).mean()), "unrel_n": float(nd(A, Bv[perm]).mean()),
            # decomposition: same context / different horizon, and same horizon / shifted context
            "hsens_z": float(zd(g(H0, d1 + d2), Bv).mean()), "hsens_n": float(nd(g(H0, d1 + d2), Bv).mean()),
            "tdrift_z": float(zd(g(Hb, d2), Bv).mean()), "tdrift_n": float(nd(g(Hb, d2), Bv).mean()),
            # ill-typed naive composition, for the record
            "naive_comp_z": float(zd(g(g(H0, d1), d2), g(H0, d1 + d2)).mean()),
            "naive_comp_n": float(nd(g(g(H0, d1), d2), g(H0, d1 + d2)).mean()),
            "ctx_shift_n": float(nd(Hb, H0).mean()), "ctx_unrel_n": float(nd(Hb, H0[perm]).mean()),
        }
        # horizon-mean-removed version: is the window-specific CONTENT of the two forecasts consistent?
        Ac, Bc = A - A.mean(0, keepdim=True), Bv - Bv.mean(0, keepdim=True)
        vc = zd(Ac, Bc)
        lat_c += vc / len(PAIRS)
        s.update({"viol_c": float(vc.mean()), "unrel_c": float(zd(Ac, Bc[perm]).mean()),
                  "cos_c": float(F.cosine_similarity(Ac, Bc, dim=-1).mean()),
                  "cos_c_unrel": float(F.cosine_similarity(Ac, Bc[perm], dim=-1).mean())})
        # floor: how different are the two TRUE targets (needs the future, diagnostic only)
        ok = [j for j, (ei, t) in enumerate(wins) if t + d2 <= len(X(ei))]
        if len(ok) > 1:
            TA = tgt_all(pre, [X(wins[j][0])[wins[j][1] - d1:wins[j][1] + d2] for j in ok])
            TB = tgt_all(pre, [X(wins[j][0])[wins[j][1]:wins[j][1] + d2] for j in ok])
            pp = prng.permutation(len(ok))
            s.update({"n_tgt": len(ok), "tgt_gap_z": float(zdt(TA, TB).mean()), "tgt_gap_n": float(nd(TA, TB).mean()),
                      "tgt_unrel_z": float(zdt(TA, TB[pp]).mean()), "tgt_unrel_n": float(nd(TA, TB[pp]).mean()),
                      "viol_n_sub": float(vn[ok].mean()),
                      "predA_err_n": float(nd(A[ok], TA).mean()), "predB_err_n": float(nd(Bv[ok], TB).mean())})
        # probability space (finetuned model): Chapman-Kolmogorov on the survival surface
        Pb = cdf_from_h(ft, Hb, hz)
        base = Pb[:, d1 - 1:d1]
        Q = (Pb[:, d1:] - base) / (1 - base + 1e-6)          # implied P(event in (t, t+k] | none by t), k=1..K-d1
        diff = np.abs(Q - P0[:, :K - d1])
        hv_pt, hv_full = diff[:, d2 - 1], diff.mean(1)
        rv = np.abs(Pb - P0).mean(1)                         # plain forecast revision between t-d1 and t
        haz += hv_pt / len(PAIRS); hazf += hv_full / len(PAIRS); rev += rv / len(PAIRS)
        s.update({"haz_pt": float(hv_pt.mean()), "haz_full": float(hv_full.mean()),
                  "haz_full_unrel": float(np.abs(Q - P0[perm][:, :K - d1]).mean()),
                  "rev": float(rv.mean()), "rev_unrel": float(np.abs(Pb - P0[perm]).mean())})
        summ["pairs"][f"{d1}+{d2}"] = s
    for j, r in enumerate(rows):
        r.update({"lat_c": float(lat_c[j]), "lat_z": float(lat_z[j]), "lat_n": float(lat_n[j]), "haz": float(haz[j]),
                  "hazf": float(hazf[j]), "rev": float(rev[j])})
    p0 = summ["pairs"]["16+16"]
    print(f"    {n}: cons 16+16 viol_z={p0['viol_z']:.3f} unrel_z={p0['unrel_z']:.3f} tgt_gap_z={p0.get('tgt_gap_z', float('nan')):.3f} "
          f"tgt_unrel_z={p0.get('tgt_unrel_z', float('nan')):.3f} hazf={p0['haz_full']:.4f} hazf_unrel={p0['haz_full_unrel']:.4f} cdfchk={chk:.1e}", flush=True)
    return summ


def run(cond, names, args, data):
    base, cons = cond.split("_")[0], cond.endswith("_cons")
    seed = args.seed
    set_seed(seed)
    rng = np.random.RandomState(seed)
    rows, timing, summ = [], {}, {}
    groups = [names] if base == "shared" else [[n] for n in names]
    for grp in groups:
        t0 = time.time()
        print(f"[{cond}] pretrain on {grp}", flush=True)
        model = E.build_model({n: data[n][0]["n_channels"] for n in grp}, args.d_model, args.n_layers)
        tp = time.time()
        pretrain(model, data, grp, args.pre_epochs, args.cuts, seed, lam=args.lam if cons else 0.0)
        timing["pre:" + ",".join(grp)] = round(time.time() - tp, 1)
        pre = copy.deepcopy(model)
        for n in grp:
            b, norm = data[n]
            ctx, kmax = get_context(n), max(b["horizons"])
            stats = E.pretrain_stats(pre, b, n, norm, rng)
            ft = copy.deepcopy(pre)
            E.activate(ft, n, norm)
            tr = E.event_loader(b["ft_train"], ctx, kmax, args.ft_cap, rng, True)
            va = E.event_loader(b["ft_val"], ctx, kmax, args.ft_cap // 3, rng, False)
            print(f"[{cond}] finetune {n}", flush=True)
            finetune(ft, tr, va, b["horizons"], lr=1e-3, weight_decay=0.01, n_epochs=args.ft_epochs,
                     patience=5, device="cpu", early_stop=(norm != "none"))
            st = rng.get_state()
            r = E.eval_windows(pre, ft, b, n, norm, args.test_windows, stats, rng)
            summ[n] = consistency_eval(pre, ft, b, n, norm, args.test_windows, stats, st, r, seed)
            for x in r:
                x["cond"] = cond
            rows += r
        timing[",".join(grp)] = round(time.time() - t0, 1)
    print(f"[{cond}] timing {timing}", flush=True)
    return rows, summ, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="FD001,FD003,MBA,PSM,ETTm1")
    ap.add_argument("--conds", default="separate,shared,shared_cons")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--pre-epochs", type=int, default=20)
    ap.add_argument("--cuts", type=int, default=1024)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-cap", type=int, default=2048)
    ap.add_argument("--test-windows", type=int, default=1200)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="/home/claude/scout/worldmodel/results")
    args = ap.parse_args()
    names = args.datasets.split(",")
    data = {n: E.load(n) for n in names}
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    for cond in args.conds.split(","):
        rows, summ, timing = run(cond, names, args, data)
        tag = f"{cond}_d{args.d_model}_s{args.seed}"
        with open(out / f"{tag}.json", "w") as fh:
            json.dump({"args": vars(args), "rows": rows, "cons_summary": summ, "timing": timing}, fh)
        print(f"saved {out/tag}.json  n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
