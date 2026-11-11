import sys, time
sys.path.insert(0, "/home/claude/HEPA/scripts")
import numpy as np
from hepa.data import load_dataset
from hepa.utils import get_context, PROTOCOL
from hepa.training.pretrain import PretrainDataset
import train as T
for n in sys.argv[1:]:
    try:
        t0=time.time(); b = load_dataset(n)
        ps = b["pretrain_seqs"]; hz=b["horizons"]
        pre = PretrainDataset(ps, n_cuts=40, max_context=get_context(n), delta_t_max=max(hz), delta_t_min=1, seed=0)
        s = {}
        for sp, st in (("ft_train",4),("ft_val",4),("ft_test",1)):
            try:
                s[sp] = len(T._build_event_loader(b[sp], 64, get_context(n), max(hz), st, False).dataset)
            except Exception as e:
                s[sp] = repr(e)
        print(n, "C", b["n_channels"], "K", len(hz), "ctx", get_context(n), "n_seqs", len(ps),
              "seq_len_tot", sum(len(v) for v in ps.values()), "pre_samples", len(pre), "pre_batches/ep", -(-len(pre)//64), s,
              "ents", {k: len(b[k]) for k in ("ft_train","ft_val","ft_test")}, f"{time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(n, "FAILED", repr(e), flush=True)
