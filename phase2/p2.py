"""Phase 2 wrapper around the authors' pipeline (scripts/train.py).

Mirrors scripts/train.py::main() statement by statement (same seeding order, same loaders, same
HEPA constructor, the repo's own pretrain()/finetune()/evaluate()), and adds:
  * conditions (--cond):
      pre     : repo pretrain() for --pre-epochs epochs            (== train.py)
      rand    : repo pretrain() with n_epochs=0                    (== train.py --pre-epochs 0)
      preE_newP : pretrain, then reset predictor (+event head) to their INIT weights before finetune
      randE_preP: pretrain, then reset encoder to its INIT weights before finetune
      tr      : pretrain with the var/cov penalty ALSO on target embeddings (as exp_shared_eps --target-reg)
      sig     : pretrain with OUR crude SIGReg-style (Epps-Pulley on random 1-D projections) penalty on targets
      frozen  : repo pretrain() with target_mode='frozen_target' (REPRODUCIBILITY.md: closest to the paper's run)
  * diagnostics: encoder-weights-unchanged check, target / context embedding mean pairwise cosine at init
    and after pretraining, timings, per-horizon AUROC, pooled AUROC, horizon-only baseline.
Nothing in /home/claude/HEPA is modified.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/claude/HEPA/scripts")
import train as T  # the authors' script (for _build_event_loader)  # noqa: E402
from hepa.data import load_dataset  # noqa: E402
from hepa.data._common import global_zscore_bundle  # noqa: E402
from hepa.evaluation import evaluate_probability_surface, h_auroc, per_horizon_auroc  # noqa: E402
from hepa.evaluation.surface import build_label_surface  # noqa: E402
from hepa.model import HEPA  # noqa: E402
from hepa.training.finetune import evaluate, finetune  # noqa: E402
from hepa.training.losses import vicreg_loss, vicreg_var_cov  # noqa: E402
from hepa.training.pretrain import PretrainDataset, collate_pretrain, pretrain, _eval_pretrain_loss  # noqa: E402
from hepa.utils import PROTOCOL, get_context, get_norm_mode, set_seed  # noqa: E402


def state_hash(module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def sigreg(h: torch.Tensor, gen: torch.Generator, n_proj: int = 64) -> torch.Tensor:
    """OUR crude SIGReg-style statistic (NOT the paper's implementation): Epps-Pulley distance between
    the empirical characteristic function of random 1-D projections of h and that of N(0,1)."""
    B, D = h.shape
    A = torch.randn(D, n_proj, generator=gen)
    A = A / A.norm(dim=0, keepdim=True)
    z = h @ A                                           # (B, M)
    t = torch.linspace(-4.0, 4.0, 17)
    zt = z.unsqueeze(-1) * t                            # (B, M, T)
    re, im = torch.cos(zt).mean(0), torch.sin(zt).mean(0)
    target = torch.exp(-0.5 * t ** 2)
    err = (re - target) ** 2 + im ** 2                  # (M, T)
    w = torch.exp(-0.5 * t ** 2)
    return torch.trapz(err * w, t, dim=-1).mean()


def pretrain_custom(model, train_loader, val_loader, lr, weight_decay, n_epochs, patience, alpha, kind, seed):
    """Line-for-line copy of hepa.training.pretrain.pretrain with one extra target-side loss term."""
    device = "cpu"
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    gen = torch.Generator().manual_seed(seed + 31337)
    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tgt, tgt_m, dt in train_loader:
            hp, ht = model.pretrain_forward(ctx, tgt, dt, ctx_m, tgt_m)
            loss = vicreg_loss(hp, ht, alpha=alpha)
            if kind == "tr":
                lv, lc, _ = vicreg_var_cov(ht)
                loss = loss + alpha * (lv + lc)
            elif kind == "sig":
                loss = loss + alpha * sigreg(ht, gen)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(losses))
        # validation on the repo's own loss (alignment + predictor var/cov) so early stopping is comparable
        val_loss = _eval_pretrain_loss(model, val_loader, alpha, device)
        with torch.no_grad():
            h_std = float(hp.std(dim=0).mean().item())
        print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  h_std={h_std:.3f}", flush=True)
        if h_std < 0.01:
            print("  COLLAPSED - aborting", flush=True)
            break
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def embed_stats(model, diag_loader):
    """Mean pairwise cosine (off-diagonal) of target embeddings h*, context embeddings h_t and predictor
    outputs on a fixed diagnostic set of pretraining cuts; plus effective rank of h*."""
    rng_state = torch.get_rng_state()   # DataLoader iteration draws a base seed; restore so the stream matches train.py
    model.eval()
    Ht, Hc, Hp = [], [], []
    for ctx, ctx_m, tgt, tgt_m, dt in diag_loader:
        h = model.encoder(ctx, ctx_m)
        Hc.append(h); Hp.append(model.predictor(h, dt)); Ht.append(model.target_encoder(tgt, tgt_m))
    out = {}
    for name, H in (("target", torch.cat(Ht)), ("context", torch.cat(Hc)), ("pred", torch.cat(Hp))):
        n = F.normalize(H, dim=-1)
        S = n @ n.t()
        N = S.shape[0]
        out[f"{name}_cos"] = float((S.sum() - S.diag().sum()) / (N * (N - 1)))
        Hc_ = H - H.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(Hc_)
        p = (sv ** 2) / (sv ** 2).sum()
        out[f"{name}_effrank"] = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
        out[f"{name}_std"] = float(H.std(0).mean())
    out["n_diag"] = int(N)
    torch.set_rng_state(rng_state)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond", default="pre")
    ap.add_argument("--pre-epochs", type=int, default=int(PROTOCOL["pre_epochs"]))
    ap.add_argument("--ft-epochs", type=int, default=int(PROTOCOL["ft_epochs"]))
    ap.add_argument("--d-model", type=int, default=int(PROTOCOL["d_model"]))
    ap.add_argument("--n-cuts", type=int, default=int(PROTOCOL["n_cuts"]))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="/home/claude/phase2/out")
    ap.add_argument("--save-surface", action="store_true")
    ap.add_argument("--save-ckpt", action="store_true", help="save init + pretrained weights + torch RNG state after pretraining")
    ap.add_argument("--from-ckpt", default="", help="skip pretraining; load the pretrained weights/RNG saved by a --save-ckpt 'pre' run")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cond = args.cond
    pre_epochs = 0 if cond == "rand" else args.pre_epochs
    d = args.d_model

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    set_seed(args.seed)
    bundle = load_dataset(args.dataset)
    norm_mode = get_norm_mode(args.dataset)
    if norm_mode == "none":
        bundle = global_zscore_bundle(bundle)
    horizons = bundle["horizons"]; n_channels = bundle["n_channels"]
    max_context = get_context(args.dataset)
    delta_t_max = max(horizons)
    train_pre = PretrainDataset(bundle["pretrain_seqs"], n_cuts=args.n_cuts, max_context=max_context,
                                delta_t_max=delta_t_max, delta_t_min=int(PROTOCOL["delta_t_min"]), seed=args.seed)
    val_seqs = {}
    for k, seq in bundle["pretrain_seqs"].items():
        cut = int(0.9 * len(seq))
        if len(seq) - cut >= 128:
            val_seqs[k] = seq[cut:]
    val_pre = PretrainDataset(val_seqs or bundle["pretrain_seqs"], n_cuts=10, max_context=max_context,
                              delta_t_max=delta_t_max, delta_t_min=int(PROTOCOL["delta_t_min"]), seed=args.seed + 10000)
    pre_loader = DataLoader(train_pre, batch_size=int(PROTOCOL["pre_batch"]), shuffle=True, collate_fn=collate_pretrain)
    val_loader = DataLoader(val_pre, batch_size=int(PROTOCOL["pre_batch"]), shuffle=False, collate_fn=collate_pretrain)
    ft_train = T._build_event_loader(bundle["ft_train"], int(PROTOCOL["ft_batch"]), max_context, delta_t_max, stride=4, shuffle=True)
    ft_val = T._build_event_loader(bundle["ft_val"], int(PROTOCOL["ft_batch"]), max_context, delta_t_max, stride=4, shuffle=False)
    ft_test = T._build_event_loader(bundle["ft_test"], int(PROTOCOL["ft_batch"]), max_context, delta_t_max, stride=1, shuffle=False)

    model = HEPA(n_channels=n_channels, patch_size=int(PROTOCOL["patch_size"]), d_model=d,
                 n_heads=int(PROTOCOL["n_heads"]), n_layers=int(PROTOCOL["n_layers"]),
                 d_ff=d if d != 256 else int(PROTOCOL["d_ff"]), dropout=float(PROTOCOL["dropout"]),
                 predictor_hidden=d if d != 256 else int(PROTOCOL["predictor_hidden"]),
                 target_mode="frozen_target" if cond == "frozen" else str(PROTOCOL["target_mode"]),
                 sync_interval_steps=int(PROTOCOL["sync_interval_steps"]), norm_mode=norm_mode)
    n_params = sum(p.numel() for p in model.parameters())
    init_state = copy.deepcopy(model.state_dict())
    enc_hash_init = state_hash(model.encoder)

    # diagnostic cuts: own RandomState, eval-mode forward -> does not touch the global RNG stream
    n_seq = len(bundle["pretrain_seqs"])
    diag_ds = PretrainDataset(bundle["pretrain_seqs"], n_cuts=max(1, -(-512 // n_seq)), max_context=max_context,
                              delta_t_max=delta_t_max, delta_t_min=1, seed=987654)
    diag_ds.samples = diag_ds.samples[:512]
    diag_loader = DataLoader(diag_ds, batch_size=128, shuffle=False, collate_fn=collate_pretrain)
    stats_init = embed_stats(model, diag_loader)

    print(f"[p2] {args.dataset} s{args.seed} cond={cond} d={d} pre_epochs={pre_epochs} n_cuts={args.n_cuts} "
          f"params={n_params/1e6:.2f}M pre_samples={len(train_pre)} ft_train={len(ft_train.dataset)} test={len(ft_test.dataset)}", flush=True)
    t0 = time.time()
    if args.from_ckpt:
        ck = torch.load(args.from_ckpt, weights_only=False)
        assert ck["enc_hash_init"] == enc_hash_init, "init mismatch vs checkpoint"
        model.load_state_dict(ck["post_state"])
        torch.set_rng_state(ck["rng_state"])     # same finetune shuffle/dropout stream as the 'pre' run
    elif cond in ("tr", "sig"):
        pretrain_custom(model, pre_loader, val_loader, float(PROTOCOL["pre_lr"]), float(PROTOCOL["pre_weight_decay"]),
                        pre_epochs, int(PROTOCOL["pre_patience"]), float(PROTOCOL["alpha"]), cond, args.seed)
    else:
        pretrain(model, pre_loader, val_loader, lr=float(PROTOCOL["pre_lr"]), weight_decay=float(PROTOCOL["pre_weight_decay"]),
                 n_epochs=pre_epochs, patience=int(PROTOCOL["pre_patience"]), alpha=float(PROTOCOL["alpha"]), device=device)
    t_pre = time.time() - t0
    if args.save_ckpt:
        Path("/home/claude/phase2/ckpt").mkdir(exist_ok=True)
        torch.save({"init_state": init_state, "post_state": copy.deepcopy(model.state_dict()),
                    "rng_state": torch.get_rng_state(), "enc_hash_init": enc_hash_init},
                   f"/home/claude/phase2/ckpt/{args.dataset}_{cond}_d{d}_s{args.seed}{args.tag}.pt")
    stats_post = embed_stats(model, diag_loader)

    if cond == "preE_newP":
        model.predictor.load_state_dict({k[len("predictor."):]: v for k, v in init_state.items() if k.startswith("predictor.")})
        model.event_head.load_state_dict({k[len("event_head."):]: v for k, v in init_state.items() if k.startswith("event_head.")})
    elif cond == "randE_preP":
        model.encoder.load_state_dict({k[len("encoder."):]: v for k, v in init_state.items() if k.startswith("encoder.")})
    enc_hash_ft = state_hash(model.encoder)

    t0 = time.time()
    ft_info = finetune(model, ft_train, ft_val, horizons, mode="pred_ft", lr=float(PROTOCOL["ft_lr"]),
                       weight_decay=float(PROTOCOL["ft_weight_decay"]), n_epochs=args.ft_epochs,
                       patience=int(PROTOCOL["ft_patience"]), device=device, early_stop=(norm_mode != "none"))
    t_ft = time.time() - t0
    assert state_hash(model.encoder) == enc_hash_ft, "encoder changed during finetune"
    t0 = time.time()
    surf = evaluate(model, ft_test, horizons, mode="pred_ft", device=device)
    t_ev = time.time() - t0
    P, Y = surf["p_surface"], surf["y_surface"]
    pooled = evaluate_probability_surface(P, Y)
    per_h = per_horizon_auroc(P, Y, horizons)
    h_aur = h_auroc(P, Y, horizons)

    # horizon-only baseline: training base rate per horizon, same for every window
    h_tensor = torch.tensor(horizons, dtype=torch.float32)
    ys = []
    for _c, _m, tte, _t in ft_train:
        ys.append(build_label_surface(tte.unsqueeze(1), h_tensor).squeeze(1).numpy())
    base = np.concatenate(ys).mean(0)                       # (K,)
    Pb = np.repeat(base[None, :], P.shape[0], axis=0)
    honly_pooled = float(roc_auc_score(Y.ravel(), Pb.ravel())) if 0 < Y.mean() < 1 else float("nan")
    # share of prediction variance that is within-horizon (across windows)
    var_tot = float(P.var())
    var_within = float(P.var(axis=0).mean())
    res = {
        "dataset": args.dataset, "seed": args.seed, "cond": cond, "d_model": d, "pre_epochs": pre_epochs,
        "from_ckpt": args.from_ckpt, "tag": args.tag, "n_cuts": args.n_cuts, "ft_epochs": args.ft_epochs, "n_params": n_params, "threads": args.threads,
        "n_pre_samples": len(train_pre), "n_ft_train": len(ft_train.dataset), "n_test": int(P.shape[0]),
        "pooled_auroc": pooled["auroc"], "pooled_auprc": pooled["auprc"], "h_auroc": h_aur,
        "n_valid_horizons": int(np.sum(~np.isnan(per_h["auroc_per_k"]))),
        "auroc_per_k": per_h["auroc_per_k"], "prevalence_per_k": per_h["prevalence_per_k"],
        "horizon_only_pooled_auroc": honly_pooled, "horizon_only_h_auroc": 0.5,
        "brier": float(((P - Y) ** 2).mean()), "within_horizon_var_share": var_within / var_tot if var_tot > 0 else float("nan"),
        "encoder_unchanged_from_init": enc_hash_ft == enc_hash_init, "enc_hash_init": enc_hash_init, "enc_hash_ft": enc_hash_ft,
        "stats_init": stats_init, "stats_post_pretrain": stats_post,
        "t_pretrain_s": round(t_pre, 1), "t_finetune_s": round(t_ft, 1), "t_eval_s": round(t_ev, 1),
        "ft_final_epoch": ft_info["final_epoch"], "ft_best_val": ft_info["best_val"],
    }
    tag = f"{args.dataset}_{cond}_d{d}_s{args.seed}{args.tag}"
    json.dump(res, open(out_dir / f"{tag}.json", "w"))
    if args.save_surface:
        np.savez_compressed(out_dir / f"{tag}_surface.npz", p=P.astype(np.float16), y=Y.astype(np.int8))
    print(f"[p2-result] {tag} pooled={pooled['auroc']:.4f} hAUROC={h_aur:.4f} honly_pooled={honly_pooled:.4f} "
          f"tcos {stats_init['target_cos']:.4f}->{stats_post['target_cos']:.4f} enc_unchanged={res['encoder_unchanged_from_init']} "
          f"t_pre={t_pre:.0f}s t_ft={t_ft:.0f}s t_ev={t_ev:.0f}s ft_ep={ft_info['final_epoch']}", flush=True)


if __name__ == "__main__":
    main()
