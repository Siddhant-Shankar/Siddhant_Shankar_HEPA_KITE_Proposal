"""Preliminary test of the Forgis proposal hypothesis on CPU.

Question: does per-window latent prediction error (eps) become a comparable,
pooled reliability signal for event-forecast error once one encoder is shared
across domains, compared with separate per-dataset encoders?

Conditions
  separate : one HEPA per dataset (paper protocol, smaller budget)
  shared   : one HEPA trunk for all datasets. Each dataset keeps its own input
             patch projection (channel counts differ: 14, 2, 25, 7); the
             Transformer layers, target encoder layers and predictor are shared.
             (KITE's variate-token encoder is not in the public repo, so this is
             the closest shared-encoder stand-in.)

Per test window we record
  eps      mean over Delta_t in {1,2,4,8,16,32,64} (where the future exists) of the
           pretraining alignment error  mean|normalize(pred) - normalize(target)|,
           computed with the PRETRAINED predictor (before finetuning changes it)
  atyp     forward proxy: ||standardize(h_pred)|| for the predicted future
           embedding vs the pretraining distribution (available at prediction time)
  brier    mean over horizons of (p - y)^2 from the finetuned survival CDF
  hent     baseline: mean binary entropy of the per-horizon CDF
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

from hepa.data import load_dataset
from hepa.data._common import global_zscore_bundle
from hepa.evaluation.surface import build_label_surface
from hepa.model import HEPA
from hepa.model.encoder import PatchEmbedding
from hepa.training.finetune import EventDataset, collate_event, finetune
from hepa.training.losses import vicreg_loss, vicreg_var_cov
from hepa.utils import PROTOCOL, get_context, get_norm_mode, set_seed

torch.set_num_threads(2)
EPS_DTS = [1, 2, 4, 8, 16, 32, 64]


class SwitchPatch(nn.Module):
    """Per-dataset input projection; the rest of the encoder is shared."""

    def __init__(self, channels: dict, patch: int, d: int):
        super().__init__()
        self.embeds = nn.ModuleDict({k: PatchEmbedding(c, patch, d) for k, c in channels.items()})
        self.active = next(iter(channels))

    def forward(self, x):
        return self.embeds[self.active](x)


def build_model(channels: dict, d_model: int, n_layers: int):
    m = HEPA(
        n_channels=1,
        patch_size=int(PROTOCOL["patch_size"]),
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=d_model,
        dropout=float(PROTOCOL["dropout"]),
        predictor_hidden=d_model,
        target_mode="joint_train",
    )
    m.encoder.patch_embed = SwitchPatch(channels, int(PROTOCOL["patch_size"]), d_model)
    m.target_encoder.patch_embed = SwitchPatch(channels, int(PROTOCOL["patch_size"]), d_model)
    m.target_encoder.patch_embed.load_state_dict(m.encoder.patch_embed.state_dict())
    return m


def activate(model, name, norm):
    model.encoder.patch_embed.active = name
    model.target_encoder.patch_embed.active = name
    model.encoder.norm_mode = norm
    model.target_encoder.norm_mode = norm


def load(name):
    b = load_dataset(name)
    norm = get_norm_mode(name)
    if norm == "none":
        b = global_zscore_bundle(b)
    return b, norm


def sample_cuts(seqs, n, ctx, dt_max, rng):
    """n random (context, target, dt) cuts, sequences weighted by length."""
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
        out.append((seq[max(0, t - ctx):t], seq[t:t + dt], dt))
    return out


def pad(arrs):
    B, L, C = len(arrs), max(len(a) for a in arrs), arrs[0].shape[1]
    x = torch.zeros(B, L, C)
    m = torch.ones(B, L, dtype=torch.bool)
    for i, a in enumerate(arrs):
        x[i, :len(a)] = torch.from_numpy(np.asarray(a, dtype=np.float32))
        m[i, :len(a)] = False
    return x, m


def pretrain(model, data, names, epochs, cuts_per_ds, seed, alpha=0.04, bs=64, target_reg=False):
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        batches = []
        for n in names:
            b, norm = data[n]
            cuts = sample_cuts(b["pretrain_seqs"], cuts_per_ds, get_context(n), max(b["horizons"]), rng)
            for i in range(0, len(cuts), bs):
                batches.append((n, norm, cuts[i:i + bs]))
        rng.shuffle(batches)
        losses = []
        for n, norm, cb in batches:
            activate(model, n, norm)
            c, cm = pad([x[0] for x in cb])
            tg, tm = pad([x[1] for x in cb])
            dt = torch.tensor([x[2] for x in cb], dtype=torch.float32)
            hp, ht = model.pretrain_forward(c, tg, dt, cm, tm)
            loss = vicreg_loss(hp, ht, alpha)
            if target_reg:
                # same variance-covariance penalty, applied to the target embeddings as well
                lv, lc, _ = vicreg_var_cov(ht)
                loss = loss + alpha * (lv + lc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"    pretrain ep{ep:02d} loss={np.mean(losses):.4f}", flush=True)


def event_loader(entities, ctx, kmax, cap, rng, shuffle):
    ds = [EventDataset(e["test"], e["labels"], max_context=ctx, stride=1, max_future=kmax)
          for e in entities if len(e["test"]) > 129]
    ds = ConcatDataset([d for d in ds if len(d) > 0])
    idx = rng.choice(len(ds), size=min(cap, len(ds)), replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=shuffle, collate_fn=collate_event)


@torch.no_grad()
def pretrain_stats(model, b, n, norm, rng, n_samp=1024):
    """Mean/std of predicted embeddings on pretraining cuts (for the atypicality proxy)."""
    activate(model, n, norm)
    model.eval()
    cuts = sample_cuts(b["pretrain_seqs"], n_samp, get_context(n), 64, rng)
    hs, ts = [], []
    for i in range(0, len(cuts), 128):
        cb = cuts[i:i + 128]
        c, cm = pad([x[0] for x in cb])
        tg, tm = pad([x[1] for x in cb])
        dt = torch.tensor([x[2] for x in cb], dtype=torch.float32)
        hs.append(model.predictor(model.encoder(c, cm), dt))
        ts.append(model.target_encoder(tg, tm))
    H, Tt = torch.cat(hs), torch.cat(ts)
    return H.mean(0), H.std(0) + 1e-4, Tt.mean(0), Tt.std(0) + 1e-4


@torch.no_grad()
def eval_windows(pre_model, ft_model, b, n, norm, max_windows, stats, rng):
    """Per-window eps, atypicality, Brier, hazard entropy on the test split."""
    ctx = get_context(n)
    hz = torch.tensor(b["horizons"], dtype=torch.float32)
    wins = []
    for ei, e in enumerate(b["ft_test"]):
        T = len(e["test"])
        if T <= 129:
            continue
        tte = EventDataset._compute_tte(np.asarray(e["labels"], dtype=np.int32), max(b["horizons"]))
        for t in range(128, T - 1):
            wins.append((ei, t, tte[t]))
    sel = rng.choice(len(wins), size=min(max_windows, len(wins)), replace=False)
    wins = [wins[i] for i in sorted(sel)]
    mu, sd, tmu, tsd = stats
    rows = []
    cos_list = []
    tcos_list = []
    for m in (pre_model, ft_model):
        activate(m, n, norm)
        m.eval()
    for i in range(0, len(wins), 64):
        wb = wins[i:i + 64]
        ctxs = [b["ft_test"][ei]["test"][max(0, t - ctx):t] for ei, t, _ in wb]
        c, cm = pad(ctxs)
        # finetuned survival CDF
        cdf = ft_model.finetune_forward(c, hz, cm).numpy()
        y = build_label_surface(torch.tensor([w[2] for w in wb]).unsqueeze(1), hz).squeeze(1).numpy()
        brier = ((cdf - y) ** 2).mean(1)
        pc = np.clip(cdf, 1e-6, 1 - 1e-6)
        hent = (-(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))).mean(1)
        # eps + atypicality from the pretrained model
        h_t = pre_model.encoder(c, cm)
        eps_sum = np.zeros(len(wb))
        eps_cnt = np.zeros(len(wb))
        eps_dt = np.full((len(wb), len(EPS_DTS)), np.nan)
        epsc_sum = np.zeros(len(wb))
        atyp_sum = np.zeros(len(wb))
        hn = F.normalize(h_t, dim=-1)
        cos_list.append(float((hn @ hn.t()).mean()))
        for di, dtv in enumerate(EPS_DTS):
            hp = pre_model.predictor(h_t, torch.full((len(wb),), float(dtv)))
            atyp_sum += (((hp - mu) / sd) ** 2).mean(1).sqrt().numpy()
            ok = [j for j, (ei, t, _) in enumerate(wb) if t + dtv <= len(b["ft_test"][ei]["test"])]
            if not ok:
                continue
            tg, tm = pad([b["ft_test"][wb[j][0]]["test"][wb[j][1]:wb[j][1] + dtv] for j in ok])
            ht = pre_model.target_encoder(tg, tm)
            e = (F.normalize(hp[ok], dim=-1) - F.normalize(ht, dim=-1)).abs().mean(1).numpy()
            if dtv == 16 and len(ok) > 1:
                tn = F.normalize(ht, dim=-1)
                tcos_list.append(float((tn @ tn.t()).mean()))
            # centered eps: standardize both sides with label-free pretraining stats so the
            # shared mean direction (which dominates cosine) does not swamp the residual
            ec = (((hp[ok] - mu) / sd) - ((ht - tmu) / tsd)).abs().mean(1).numpy()
            epsc_sum[ok] += ec
            eps_sum[ok] += e
            eps_cnt[ok] += 1
            eps_dt[ok, di] = e
        eps = eps_sum / np.maximum(eps_cnt, 1)
        epsc = epsc_sum / np.maximum(eps_cnt, 1)
        for j in range(len(wb)):
            rows.append({
                "ds": n, "eps": float(eps[j]), "epsc": float(epsc[j]), "atyp": float(atyp_sum[j] / len(EPS_DTS)),
                "brier": float(brier[j]), "hent": float(hent[j]),
                "y_any": int(y[j].max()), "eps_dt": [None if np.isnan(v) else float(v) for v in eps_dt[j]], "p": cdf[j].tolist(), "yv": y[j].tolist(),
            })
    print(f"    {n}: mean pairwise cosine h_t={np.mean(cos_list):.4f}  target h*={np.mean(tcos_list):.4f}", flush=True)
    for r in rows:
        r["hcos"] = float(np.mean(cos_list))
        r["tcos"] = float(np.mean(tcos_list))
    return rows


def run(cond, names, args, data):
    seed = args.seed
    set_seed(seed)
    rng = np.random.RandomState(seed)
    rows, timing = [], {}
    groups = [names] if cond == "shared" else [[n] for n in names]
    for grp in groups:
        t0 = time.time()
        print(f"[{cond}] pretrain on {grp}", flush=True)
        model = build_model({n: data[n][0]["n_channels"] for n in grp}, args.d_model, args.n_layers)
        pretrain(model, data, grp, args.pre_epochs, args.cuts, seed, target_reg=args.target_reg)
        pre = copy.deepcopy(model)
        for n in grp:
            b, norm = data[n]
            ctx, kmax = get_context(n), max(b["horizons"])
            stats = pretrain_stats(pre, b, n, norm, rng)
            ft = copy.deepcopy(pre)
            activate(ft, n, norm)
            tr = event_loader(b["ft_train"], ctx, kmax, args.ft_cap, rng, True)
            va = event_loader(b["ft_val"], ctx, kmax, args.ft_cap // 3, rng, False)
            print(f"[{cond}] finetune {n}", flush=True)
            finetune(ft, tr, va, b["horizons"], lr=1e-3, weight_decay=0.01, n_epochs=args.ft_epochs,
                     patience=5, device="cpu", early_stop=(norm != "none"))
            r = eval_windows(pre, ft, b, n, norm, args.test_windows, stats, rng)
            for x in r:
                x["cond"] = cond
            rows += r
        timing[",".join(grp)] = round(time.time() - t0, 1)
    print(f"[{cond}] timing {timing}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="FD001,FD003,MBA,PSM,ETTm1")
    ap.add_argument("--conds", default="separate,shared")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--cuts", type=int, default=1024)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-cap", type=int, default=2048)
    ap.add_argument("--test-windows", type=int, default=1200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--target-reg", action="store_true")
    args = ap.parse_args()
    names = args.datasets.split(",")
    data = {n: load(n) for n in names}
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for cond in args.conds.split(","):
        rows = run(cond, names, args, data)
        tag = f"{cond}{'TR' if args.target_reg else ''}_d{args.d_model}_s{args.seed}"
        with open(out / f"{tag}.json", "w") as fh:
            json.dump({"args": vars(args), "rows": rows}, fh)
        print(f"saved {out/tag}.json  n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
