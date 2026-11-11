# Phase 2 brief: the random-encoder finding

Read /home/claude/scout/BRIEF.md first. Its HARD RULES still apply (queue every run, no invented numbers,
no git/email, never edit /home/claude/HEPA). Work only in /home/claude/phase2/. You are the only agent
running now, so the queue is yours, but still use /home/claude/scout/run_queued.sh <tag> ... for every run.

## What Phase 1 found (four scouts; full reports in their folders)
- transfer (/home/claude/scout/transfer, lodo.py, results/): a FROZEN RANDOM-INIT encoder (predictor+head
  finetuned on top) beats the HEPA-pretrained encoder on FD003 and MBA in all 12 seed x label-budget cells
  (+.018 to +.156 pooled h-AUROC), at d=256, 15 pretrain epochs, alpha .04. On MBA the pretrained
  models' forecasts do not vary across windows at all (within-horizon share of prediction variance 0.000),
  so their pooled h-AUROC equals a horizon-only oracle (.8512).
- worldmodel (/home/claude/scout/worldmodel): g(h,dt)'s output is ~96-99% a horizon code; latent
  consistency is at unrelated-window level. Consistency loss killed.
- reliability (/home/claude/scout/reliability): pooled h-AUROC can hide below-chance per-horizon ranking
  (MBA zero-shot head: pooled .859 vs per-horizon .31). Cross-domain conformal killed.
- scaling (/home/claude/scout/scaling): shared-vs-separate gap does not shrink 0.16M -> 2.16M; separate
  saturates by 0.57M; d=384 no better. Inconclusive.
- baseline (existing): target embeddings collapse in direction (cosine 0.69 -> 0.9993 with the repo's own
  pretrain()); a target-side var/cov regulariser (--target-reg in exp_shared_eps.py) closes the
  shared/separate gap.

## The hypothesis to pressure-test (sharpen it if the evidence says so)
"In HEPA-style event-JEPA, the pretraining objective rather than model size determines whether pretraining
beats frozen random features: at the published recipe the pretrained encoder does not beat a random
encoder once the horizon prior is removed from the metric, de-collapsing the targets moves pretraining
above the random baseline, and widening the model without de-collapsing does not."
Refuted if: (1) at the paper's own budget and pipeline the pretrained encoder beats random on per-horizon
AUROC on most datasets; or (2) de-collapse does not beat random either; or (3) the pretrained-minus-random
gap closes with width alone.

## Work, in priority order (stop and write up if you run low on time)
Total compute budget about 3.5 hours of queued training. Hard stop: have REPORT.md written by 05:30 PST
(`date` in this container is PST). Use 3 seeds everywhere; 5 if time allows for step 1.

1. VERIFY with the authors' own pipeline (the most important step; this is the "too good to be true" flag).
   /home/claude/HEPA/scripts/train.py uses PROTOCOL (pre_epochs 50, paper finetune). Compare
   `--pre-epochs 50` (or the PROTOCOL default) against `--pre-epochs 0` (random encoder, same everything
   else) on FD001, FD003, MBA, PSM, ETTm1. Check first that pre_epochs 0 really leaves the encoder at
   init and runs the identical finetune path (read hepa/training/*.py). Time one run before committing;
   if 50 epochs is too slow on 2 cores, use the largest budget that fits and say so. Do NOT modify the
   repo; write a wrapper in your folder if you need extra outputs (per-horizon AUROC, saved surfaces).
   train.py already reports per_horizon_auroc - use it.
2. METRIC: for every condition report pooled h-AUROC, mean per-horizon AUROC, and a horizon-only baseline
   (predict the training base rate per horizon). Say which conclusions flip between metrics.
3. MECHANISM ABLATION (use exp_shared_eps.py / lodo.py machinery, copied into your folder):
   a. pretrained encoder + re-initialised predictor (is the harm in the encoder features or the
      collapsed predictor init?);
   b. random encoder + pretrained predictor (other half of the 2x2);
   c. pretrained with target-side regulariser (--target-reg) vs random;
   d. if cheap: a crude SIGReg-style target regulariser (e.g. isotropic-Gaussian moment matching on
      targets), clearly labelled as our approximation, not the paper's.
   Report target cosine for each condition.
4. SCALE: pretrained vs random at d=64, 128, 256 (separate encoders, 3 seeds) — does the gap close with width?
5. MORE DATASETS if time: try FD002/FD004 (raw files in /home/claude/data/CMAPSS); report if they don't load.

## Write /home/claude/phase2/REPORT.md
- Final hypothesis (one sentence) and what would refute it.
- Every table: mean ± sd over seeds, n stated, file paths to raw results. Negative results stated plainly.
- Verdict on each refutation condition.
- A 3-step plan CPU pilot -> single GPU -> foundation scale (FactoryNet, 14 datasets, KITE variate tokens),
  with concrete GPU-hour estimates derived from your measured CPU timings (say how you extrapolated).
- Why it matters for Forgis's ICML/NeurIPS 2027 TSFM/world-model papers (random-feature baseline +
  horizon-debiased metric as an evaluation protocol; ROCKET is the known precedent in TS classification).
- Caveats: CPU, budget, datasets, public code VCReg vs paper's SIGReg, anything that looks too good.
- Tone toward the authors: collaborative. This is about what the objective needs at scale, not "the code
  is broken".
Also save a compact machine-readable phase2/numbers.json of every figure in REPORT.md.
