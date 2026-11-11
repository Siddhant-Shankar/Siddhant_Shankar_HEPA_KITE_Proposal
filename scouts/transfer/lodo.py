"""Leave-one-domain-out (LODO) transfer pilot for HEPA.

For a held-out domain H and seed s, compares on H's test split (h-AUROC, Brier):
  INDOM        encoder pretrained on H's own unlabeled data only (paper protocol), frozen; predictor+head finetuned
  RAND         random-init encoder, frozen; predictor+head finetuned
  <T>-ssl      trunk T frozen; ONLY the new H input projections (context + target encoder patch embeds) are
               trained with the JEPA objective on H's unlabeled pretrain split; then paper protocol finetune
  <T>-sup      trunk T frozen; new H input projection trained from LABELS together with predictor+head
               (H's unlabeled data never used)
with T in {LODO<N> = trunk pretrained on N other domains (never H), RANDT = random-init trunk (control)}.
Label fractions: 1.0 and 0.1 of the finetuning labels.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, "/home/claude/HEPA")
from exp_shared_eps import build_model, activate, load, sample_cuts, pad  # noqa: E402
from hepa.evaluation.surface import build_label_surface  # noqa: E402
from hepa.model.encoder import PatchEmbedding  # noqa: E402
from hepa.training.finetune import EventDataset, collate_event, _estimate_pos_weight  # noqa: E402
from hepa.training.losses import vicreg_loss, weighted_bce_loss  # noqa: E402
from hepa.utils import PROTOCOL, get_context, set_seed  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent


def pretrain(model, data, names, epochs, cuts_per_ds, seed, alpha=0.04, bs=64, lr=3e-4):
    """Same as exp_shared_eps.pretrain, with an lr argument; trains only requires_grad params."""
    rng = np.random.RandomState(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    last = None
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
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        last = float(np.mean(losses))
        if ep in (0, epochs - 1):
            print(f"    pretrain {names} ep{ep:02d} loss={last:.4f}", flush=True)
    return last


def get_trunk(sources, data, args):
    """Pretrained model on `sources` (cached on disk so trunks are shared between held-out runs)."""
    set_seed(args.seed)
    model = build_model({n: data[n][0]["n_channels"] for n in sources}, args.d_model, 2)
    if not sources:
        return model
    ck = HERE / "ckpt" / f"{'+'.join(sources)}_s{args.seed}_e{args.pre_epochs}_c{args.cuts}.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck))
        print(f"  loaded cached trunk {ck.name}", flush=True)
    else:
        t0 = time.time()
        pretrain(model, data, sources, args.pre_epochs, args.cuts, args.seed)
        ck.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ck)
        print(f"  pretrained {sources} in {time.time()-t0:.0f}s", flush=True)
    return model


def add_projection(model, H, n_ch, d, seed):
    """Fresh (random) input projection for the unseen domain; trunk untouched."""
    g = torch.Generator().manual_seed(seed + 7919)
    pe = PatchEmbedding(n_ch, int(PROTOCOL["patch_size"]), d)
    with torch.no_grad():
        for p in pe.parameters():
            if p.dim() > 1:
                bound = 1.0 / np.sqrt(p.shape[1])
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) * bound)
            else:
                p.zero_()
    model.encoder.patch_embed.embeds[H] = pe
    model.target_encoder.patch_embed.embeds[H] = copy.deepcopy(pe)
    return model


def freeze_all_but_projection(model, H):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.patch_embed.embeds[H].parameters():
        p.requires_grad = True
    for p in model.target_encoder.patch_embed.embeds[H].parameters():
        p.requires_grad = True


def make_loader(entities, ctx, kmax, cap, seed, shuffle, ent_frac=1.0):
    rng = np.random.RandomState(seed)
    ents = [e for e in entities if len(e["test"]) > 129]
    if ent_frac < 1.0 and len(ents) > 1:  # low-label: keep a subset of ENTITIES (e.g. engines)
        k = max(1, int(round(ent_frac * len(ents))))
        ents = [ents[i] for i in sorted(rng.choice(len(ents), size=k, replace=False))]
    ds = [EventDataset(e["test"], e["labels"], max_context=ctx, stride=1, max_future=kmax) for e in ents]
    ds = ConcatDataset([d for d in ds if len(d) > 0])
    idx = rng.choice(len(ds), size=min(cap, len(ds)), replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=shuffle, collate_fn=collate_event), len(idx), len(ents)


def cdf_forward(model, ctx, hz, mask, train_proj):
    if train_proj:
        h_t = model.encoder(ctx, mask)
    else:
        with torch.no_grad():
            h_t = model.encoder(ctx, mask)
    B, d = h_t.shape
    K = hz.shape[0]
    hp = model.predictor(h_t.unsqueeze(1).expand(B, K, d).reshape(B * K, d), hz.unsqueeze(0).expand(B, K).reshape(B * K))
    lam = torch.sigmoid(model.event_head(hp.view(B, K, d)))
    return 1 - torch.cumprod(1 - lam, dim=-1)


def finetune(model, H, tr, va, horizons, n_epochs, early_stop, train_proj, lr=1e-3, patience=5):
    """Mirror of hepa.training.finetune.finetune (pred_ft); optionally also trains H's input projection."""
    hz = torch.tensor(horizons, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    params = list(model.predictor.parameters()) + list(model.event_head.parameters())
    if train_proj:
        params += list(model.encoder.patch_embed.embeds[H].parameters())
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    pw = torch.tensor(_estimate_pos_weight(tr, hz))
    best, best_state, wait = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        for ctx, m, tte, _ in tr:
            cdf = cdf_forward(model, ctx, hz, m, train_proj)
            y = build_label_surface(tte.unsqueeze(1), hz).squeeze(1)
            loss = weighted_bce_loss(cdf, y, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        vl = []
        with torch.no_grad():
            for ctx, m, tte, _ in va:
                y = build_label_surface(tte.unsqueeze(1), hz).squeeze(1)
                vl.append(weighted_bce_loss(cdf_forward(model, ctx, hz, m, False), y, pos_weight=pw).item())
        vl = float(np.mean(vl))
        if vl < best:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if early_stop and wait >= patience:
                break
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    return ep + 1


@torch.no_grad()
def evaluate(model, b, H, norm, wins):
    ctx = get_context(H)
    hz = torch.tensor(b["horizons"], dtype=torch.float32)
    activate(model, H, norm)
    model.eval()
    P, Y = [], []
    for i in range(0, len(wins), 64):
        wb = wins[i:i + 64]
        c, cm = pad([b["ft_test"][ei]["test"][max(0, t - ctx):t] for ei, t, _ in wb])
        P.append(cdf_forward(model, c, hz, cm, False).numpy())
        Y.append(build_label_surface(torch.tensor([w[2] for w in wb]).unsqueeze(1), hz).squeeze(1).numpy())
    return np.concatenate(P), np.concatenate(Y)


def test_windows(b, n, seed):
    wins = []
    for ei, e in enumerate(b["ft_test"]):
        T = len(e["test"])
        if T <= 129:
            continue
        tte = EventDataset._compute_tte(np.asarray(e["labels"], dtype=np.int32), max(b["horizons"]))
        for t in range(128, T - 1):
            wins.append((ei, t, tte[t]))
    rng = np.random.RandomState(seed + 1)
    sel = rng.choice(len(wins), size=min(n, len(wins)), replace=False)
    return [wins[i] for i in sorted(sel)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--sources", required=True, help="';'-separated source sets, each ','-separated, e.g. 'FD001,MBA,PSM,ETTm1;PSM,ETTm1'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--cuts", type=int, default=1024)
    ap.add_argument("--adapt-epochs", type=int, default=5)
    ap.add_argument("--adapt-lr", type=float, default=1e-3)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-cap", type=int, default=2048)
    ap.add_argument("--fracs", default="1.0,0.1")
    ap.add_argument("--test-windows", type=int, default=1200)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    H, seed = args.heldout, args.seed
    source_sets = [s.split(",") for s in args.sources.split(";")]
    for ss in source_sets:
        assert H not in ss, "LEAKAGE: held-out domain in source set"
    names = sorted({n for ss in source_sets for n in ss} | {H})
    data = {n: load(n) for n in names}
    b, norm = data[H]
    ctx, kmax, nch = get_context(H), max(b["horizons"]), b["n_channels"]
    wins = test_windows(b, args.test_windows, seed)
    fracs = [float(f) for f in args.fracs.split(",")]

    # ---- frozen encoders to evaluate: name -> (model, train_proj_in_finetune)
    encoders = {}
    t0 = time.time()
    indom = get_trunk([H], data, args)
    encoders["INDOM"] = (indom, False)
    set_seed(seed)
    rand = build_model({H: nch}, args.d_model, 2)
    encoders["RAND"] = (rand, False)
    trunks = {f"LODO{len(ss)}[{'+'.join(ss)}]": get_trunk(ss, data, args) for ss in source_sets}
    set_seed(seed)
    trunks["RANDT"] = build_model({"_dummy": 1}, args.d_model, 2)
    adapt_loss = {}
    for tn, tm in trunks.items():
        tm = add_projection(tm, H, nch, args.d_model, seed)
        encoders[f"{tn}-sup"] = (copy.deepcopy(tm), True)
        ssl = copy.deepcopy(tm)
        freeze_all_but_projection(ssl, H)
        ta = time.time()
        adapt_loss[tn] = pretrain(ssl, data, [H], args.adapt_epochs, args.cuts, seed + 1, lr=args.adapt_lr)
        print(f"  ssl-adapt {tn}: {time.time()-ta:.0f}s final loss {adapt_loss[tn]:.4f}", flush=True)
        encoders[f"{tn}-ssl"] = (ssl, False)
    print(f"encoders ready in {time.time()-t0:.0f}s", flush=True)

    res = []
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    raw = {}
    for frac in fracs:
        cap = max(64, int(round(args.ft_cap * frac)))
        n_ep = args.ft_epochs if frac >= 1.0 else min(30, int(round(args.ft_epochs / frac)))
        for cn, (enc, tp) in encoders.items():
            t1 = time.time()
            set_seed(seed)
            m = copy.deepcopy(enc)
            activate(m, H, norm)
            tr, ntr, nent = make_loader(b["ft_train"], ctx, kmax, cap, seed, True, frac)
            va, nva, _ = make_loader(b["ft_val"], ctx, kmax, max(64, cap // 3), seed, False, 1.0)
            ep = finetune(m, H, tr, va, b["horizons"], n_ep, early_stop=(norm != "none"), train_proj=tp)
            P, Y = evaluate(m, b, H, norm, wins)
            r = {"heldout": H, "seed": seed, "cond": cn, "frac": frac, "n_train_windows": ntr, "n_train_entities": nent,
                 "ft_epochs_run": ep, "auroc": float(roc_auc_score(Y.ravel(), P.ravel())),
                 "brier": float(((P - Y) ** 2).mean()), "pos_rate": float(Y.mean()), "n_test": len(wins),
                 "adapt_loss": adapt_loss.get(cn.rsplit("-", 1)[0])}
            res.append(r)
            raw[f"{cn}|{frac}|p"] = P.astype(np.float32)
            raw[f"{cn}|{frac}|y"] = Y.astype(np.int8)
            print(f"  [{H} s{seed} frac={frac}] {cn:38s} AUROC={r['auroc']:.4f} brier={r['brier']:.4f} "
                  f"(ntr={ntr}, ent={nent}, ep={ep}, {time.time()-t1:.0f}s)", flush=True)
    tag = f"{H}_s{seed}"
    json.dump({"args": vars(args), "rows": res}, open(out / f"{tag}.json", "w"), indent=1)
    np.savez_compressed(out / f"{tag}_raw.npz", **raw)
    print(f"saved {out/tag}.json total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
