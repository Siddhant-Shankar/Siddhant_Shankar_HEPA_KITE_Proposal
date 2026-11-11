# Shared brief for Forgis hypothesis scouts

## Goal
Siddhant (UIUC CS+Econ undergrad) is proposing a research direction to Pratik Jawahar at Forgis Labs,
co-author of KITE. Pratik asked for a one-page proposal with a clear hypothesis, "thinking in terms of
the fundamental architecture from HEPA/KITE scaled up to foundation model scales". Forgis targets
ICML/NeurIPS 2027 work on time-series foundation models (TSFMs) and world models.
We are scouting several hypotheses with cheap CPU pilots, to pick the most impressive one that
survives evidence.

## Papers (read the relevant parts; local copies exist)
- HEPA: arXiv:2605.11130 — local: /home/claude/HEPA/papers/hepa-2026/ and
  /home/claude/HEPA/artifacts/HEPA-small-technical-report.pdf. Code: /home/claude/HEPA (installed editable as `hepa`).
- KITE: arXiv:2610.12738 (fetch via WebFetch on https://arxiv.org/abs/2610.12738 or the /html/ version).
  KITE's variate-token encoder is NOT in the public repo.
- FactoryNet: arXiv:2605.09081 (Forgis's industrial corpus; not downloaded here).

HEPA in one paragraph: a causal Transformer encoder (d=256, 2 layers, 2.16M params) maps a context
window to h_t. A predictor g(h_t, Δt) predicts the target encoder's embedding of the future window
(L1 on L2-normalised embeddings + variance/covariance regulariser on the predictor output;
the paper text describes SIGReg). The encoder is then frozen and the predictor finetuned into per-horizon
hazards composing a monotone survival CDF p(event by horizon k). Metric: h-AUROC over (window, horizon) pairs,
plus Brier/ECE. The public code trains one encoder per dataset.

## What already exists (reuse it; do not redo it)
- Data in /home/claude/data, loaded with `HEPA_DATA_DIR=/home/claude/data` and
  `from hepa.data import load_dataset`. Available datasets that load cleanly:
  FD001, FD003 (C-MAPSS jet engines; FD002/FD004 raw files also present, untested), MBA (ECG), PSM (server metrics), ETTm1
  (electricity transformer). Other HEPA benchmarks are not downloaded and most hosts are blocked by
  the network allowlist; GitHub raw/clone works.
- /home/claude/HEPA/exp_shared_eps.py — the main experiment: separate vs shared encoder
  (`build_model`, `SwitchPatch` per-dataset input projection, `pretrain`, `event_loader`, `finetune`,
  `eval_windows` which saves per-window p (CDF), yv (labels), brier, eps, hent etc.). Flags:
  --datasets --conds separate,shared --seed --d-model --pre-epochs --cuts --ft-epochs --ft-cap
  --test-windows --target-reg --out. Defaults: 20 pretrain epochs, 1024 cuts/dataset/epoch, 10 ft epochs.
  One seed of separate+shared on all 5 datasets takes ~11-12 min on the 2 cores.
- /home/claude/HEPA/analyze.py, summarize_seeds.py — analysis helpers (per-dataset AUROC, Brier,
  Spearman, risk-coverage AURC).
- Existing results: /home/claude/HEPA/results/*.json (per-window rows) for separate/shared at
  d=256 (seeds 42,123,456), d=64 and d=128 (seed 42), and target-reg (TR) at d=256 (seeds 42,123).
  Param counts: d=64 → 0.16M, d=128 → 0.57M, d=256 → 2.16M.

Key numbers so far (mean h-AUROC over 5 datasets):
  separate .840±.002 (3 seeds), shared .823±.003 (3 seeds), sharedTR .839±.002 (2), separateTR .842±.002 (2).
  Per dataset (separate / shared): FD001 .961/.949, FD003 .954/.934, MBA .892/.883, PSM .585/.544, ETTm1 .808/.802.
  Target embeddings collapse in direction (mean pairwise cosine ≈0.999+), also with the repo's own pretrain().
  Pooled Spearman with per-window Brier: hazard entropy ≈+0.8, latent error eps ≈+0.3 (not a useful trust signal).
  Size sweep (shared, seed 42): mean AUROC .754 (0.16M), .827 (0.57M), .823 (2.16M); separate .828, .837, .840.

## HARD RULES
1. CPU is shared by several agents (2 cores total). EVERY training/eval run must go through the queue:
   `bash /home/claude/scout/run_queued.sh <your-tag> python3 -W ignore your_script.py ...`
   Never run training outside the queue. Keep each queued job short-ish (≤20 min) so others interleave.
   Long waits are expected; use Bash run_in_background for queued jobs and check back, or foreground
   with timeout up to 600000 ms. Your total compute budget is ~45 minutes of training time.
2. Work only in your own folder /home/claude/scout/<your-name>/. Do not edit files in /home/claude/HEPA
   (import from them; copy a file into your folder if you need to modify it).
3. Never invent, round in our favour, or cherry-pick numbers. Report spread over seeds (≥2 seeds; 3 if time
   allows). Negative results are valuable — report them plainly.
4. Smoke-test with a tiny budget first (pre-epochs 1, small caps) — still through the queue.
5. Do not touch git, GitHub, email, or anything outside your folder.

## Report format (your final message)
- Hypothesis (one falsifiable sentence, framed at foundation-model scale)
- Pilot design (what was run, datasets, seeds, budget, what would have killed it)
- Results table with mean ± sd over seeds, and the file paths of raw results
- Verdict: survived / killed / inconclusive, and why
- Scores 1–10 with one line each: (a) surprising/impressive to a KITE co-author, (b) relevance to Forgis's
  2027 TSFM/world-model papers, (c) feasibility of a strong result in ~1 week
- What a Phase-2 deep dive would run, and anything that looks too good to be true
