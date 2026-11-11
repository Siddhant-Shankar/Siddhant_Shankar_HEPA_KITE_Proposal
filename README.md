# Does event-JEPA pretraining beat its own random initialisation?

CPU pilot experiments on top of the public [HEPA](https://github.com/Forgis-Labs/HEPA) code
(commit `686deb4`). They ask one question: **does HEPA's event-JEPA pretraining beat the same encoder left at
its random initialisation?** HEPA itself is not included or modified here. Every script imports it from a
local clone.

The one-page proposal is in [`proposal/Shankar_HEPA_KITE_Proposal.pdf`](proposal/Shankar_HEPA_KITE_Proposal.pdf) (LaTeX source and figure script alongside; the figure is built from `phase2/numbers.json`).

## Main result (`phase2/`)

The setup is `scripts/train.py` at its PROTOCOL defaults (50 pretrain / 30 finetune epochs, d=256, 2.16M
parameters), comparing `--pre-epochs 50` with `--pre-epochs 0`. The metric is mean per-horizon AUROC (the
repo's `h_auroc`). There are 3 seeds per cell, reported as mean ± sd, with paired differences.

| | FD001 | FD003 | MBA | PSM | ETTm1 |
|:--|--:|--:|--:|--:|--:|
| random encoder | .9237 ± .0020 | .9566 ± .0024 | .7986 ± .0512 | .6312 ± .0079 | .8120 ± .0099 |
| pretrained | .9012 ± .0068 | .9411 ± .0032 | .8196 ± .0098 | .6291 ± .0217 | .8091 ± .0109 |
| paired Δ | −.0225 ± .0063 | −.0155 ± .0056 | +.0210 ± .0430 | −.0022 ± .0139 | −.0029 ± .0172 |
| seeds where pretrained is ahead | 0/3 | 0/3 | 2/3 | 1/3 | 1/3 |

The full write-up is in [`phase2/REPORT.md`](phase2/REPORT.md). It covers the 2×2 encoder/predictor ablation,
target-side regularisers, the width sweep (d=64/128/256), the horizon-only baseline, GPU-hour estimates and
every caveat. Every figure is also in `phase2/numbers.json`, and the per-run JSONs are in `phase2/out/`.

## Caveats (read these before citing anything)

- The runs used a 2-core CPU, 5 of the paper's 14 benchmarks, and 3 seeds on a fixed train/test split.
- The public code is not the run behind the paper's table. It uses a VICReg-style regulariser on the
  predictor output where the paper uses SIGReg, a joint target where the paper uses a detached target, and
  50/30 epochs where the paper uses 100/50. Nothing here is a statement about the paper's reported numbers.
- Both clear losses for pretraining are on C-MAPSS. That data is globally z-scored with no RevIN, so a
  random projection keeps the degradation level.
- On MBA/PSM/ETTm1 the public script takes at most 50 pretraining optimiser steps.
- `sig` is our own crude Epps-Pulley penalty, not the paper's SIGReg.

## Layout

- `phase2/`: the controlled comparison. `p2.py` mirrors `scripts/train.py` and adds conditions and
  diagnostics. `agg.py`, `make_tables.py` and `build_report.py` produce the tables. `gpu_est.py` holds the
  GPU-hour extrapolation.
- `core/`: the earlier shared-vs-separate encoder and target-collapse experiments (`exp_shared_eps.py`,
  `check_collapse.py`, `analyze.py`, `summarize_seeds.py`). These were run from inside the HEPA checkout.
- `scouts/`: four short exploratory studies (`transfer/`, `worldmodel/`, `scaling/`, `reliability/`). Their
  numbers were run at reduced budget. Some did not reproduce in `phase2/`, which `phase2/REPORT.md` notes.
- `run_queued.sh`: a flock queue that serialises every run on a 2-core box.

## Running

Paths are hard-coded to the original workspace: HEPA at `/home/claude/HEPA` and data at
`/home/claude/data`. Either symlink your clone and data there or edit the paths. Then run, for example:

```
python phase2/p2.py --dataset FD001 --cond pre  --seed 42 --d-model 256
python phase2/p2.py --dataset FD001 --cond rand --seed 42 --d-model 256
```

Check `python phase2/p2.py -h` for the exact flags.
