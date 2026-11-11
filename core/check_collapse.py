"""Check embedding diversity after pretraining with the repo's OWN pipeline and protocol.

Reuses scripts/train.py's pretraining setup unchanged (PROTOCOL defaults, 50 epochs,
early stopping) and reports mean pairwise cosine of context embeddings h_t and of
target embeddings h* on held-out pretraining cuts, at init and after training.
A value near 1.0 means every window maps to (almost) the same direction.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hepa.data import load_dataset
from hepa.data._common import global_zscore_bundle
from hepa.model import HEPA
from hepa.training.pretrain import PretrainDataset, collate_pretrain, pretrain
from hepa.utils import PROTOCOL, get_context, get_norm_mode, set_seed

torch.set_num_threads(2)


@torch.no_grad()
def diversity(model, loader):
    model.eval()
    hc, ht, hcc, htc = [], [], [], []
    H, T = [], []
    for ctx, cm, tgt, tm, dt in loader:
        H.append(model.encoder(ctx, cm))
        T.append(model.target_encoder(tgt, tm))
    H, T = torch.cat(H), torch.cat(T)

    def mcos(X):
        n = F.normalize(X, dim=-1)
        return float((n @ n.t()).mean())

    return {
        "cos_h": mcos(H), "cos_target": mcos(T),
        "cos_h_centered": mcos(H - H.mean(0)), "cos_target_centered": mcos(T - T.mean(0)),
        "target_norm_of_mean_over_mean_norm": float(T.mean(0).norm() / T.norm(dim=1).mean()),
    }


def main(name, epochs):
    set_seed(42)
    b = load_dataset(name)
    norm = get_norm_mode(name)
    if norm == "none":
        b = global_zscore_bundle(b)
    ctx, dtmax = get_context(name), max(b["horizons"])
    seqs = b["pretrain_seqs"]
    val = {k: v[int(0.9 * len(v)):] for k, v in seqs.items() if len(v) - int(0.9 * len(v)) >= 128}
    tr = DataLoader(PretrainDataset(seqs, n_cuts=int(PROTOCOL["n_cuts"]), max_context=ctx,
                                    delta_t_max=dtmax, seed=42),
                    batch_size=64, shuffle=True, collate_fn=collate_pretrain)
    va = DataLoader(PretrainDataset(val or seqs, n_cuts=10, max_context=ctx, delta_t_max=dtmax, seed=10042),
                    batch_size=64, shuffle=False, collate_fn=collate_pretrain)
    m = HEPA(n_channels=b["n_channels"], norm_mode=norm, target_mode=str(PROTOCOL["target_mode"]))
    print(name, "init   ", {k: round(v, 4) for k, v in diversity(m, va).items()}, flush=True)
    pretrain(m, tr, va, lr=3e-4, weight_decay=0.01, n_epochs=epochs, patience=10,
             alpha=float(PROTOCOL["alpha"]), device="cpu")
    print(name, "trained", {k: round(v, 4) for k, v in diversity(m, va).items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
