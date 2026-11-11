# Phase 2 report: the random-encoder finding

Generated Wed Nov 11 10:10:40 PST 2026 (container clock). Work folder `/home/claude/phase2/`. Raw per-run results:
`/home/claude/phase2/out/*.json` (one file per dataset x condition x width x seed; per-horizon AUROC lists
inside), saved probability surfaces for Step 1 in `out/*_surface.npz`, logs in `logs/`, machine-readable
summary in `numbers.json`. All ± are sample standard deviations (ddof=1) over seeds; n is stated in every
row. Every figure in the prose below is copied from the auto-generated tables.

## Status

Steps 1-4 complete with 3 seeds in every cell (108 runs). Step 5 (FD002/FD004) not run. Step 3d (`sig`) was not run on FD001. Not run and flagged as the first follow-up: `target_mode='frozen_target'`, the setting REPRODUCIBILITY.md says is closest to the run behind the paper's table. The session was interrupted at about 03:21; two queued chunks were killed then and their runs were redone from scratch (no partial outputs were used: a result file is written only when a run finishes). Completed queue time for this phase was 3.68 hours (sum of START-END intervals for `p2-*` tags in `/home/claude/scout/queue.log`), plus up to about 19 minutes in the two killed chunks; that is over the brief's 'about 3.5 hours'.

## Final hypothesis (sharpened; the original second clause did not survive)

"At 2M parameters and per-dataset data, the encoder produced by HEPA's public pretraining recipe is not a
better frozen feature extractor than the same encoder left at its random initialisation — on C-MAPSS it is
measurably worse, the loss sits in the encoder rather than the predictor, and it coincides with the target
embeddings collapsing to about one effective dimension; regularising the targets recovers random-encoder
level but does not exceed it — so whether event-JEPA pretraining adds anything over the untrained
architecture is an open question that only data and model scale, with a non-collapsing target, can answer."

Refuted if, at the paper's configuration of record on the full 14-dataset benchmark, the pretrained encoder
beats the random encoder on mean per-horizon AUROC by more than one seed-sd on most datasets; or if the
C-MAPSS deficit disappears under the detached-target setting that REPRODUCIBILITY.md says produced the
paper's table (not run here).

## What was run

- **Pipeline check.** `p2.py` mirrors `scripts/train.py` and calls the repo's own `pretrain()`, `finetune()`
  and `evaluate()`. On MBA seed 42 it reproduces `train.py` to the last printed digit in both modes
  (2 pretrain / 2 finetune epochs: `h_auroc` 0.7791432964475965 from both; 0 pretrain / 2 finetune epochs:
  0.7883432088835639 from both; files in `smoke/`). With `--pre-epochs 0` the pretraining loop body never
  executes, a hash of the encoder weights is unchanged from initialisation in every `rand` run
  (`encoder_unchanged_from_init: true`), and the finetune path is the same function call. The only
  difference downstream is the random-number stream (no pretraining batches are drawn).
- **Step 1/2.** `pre` (PROTOCOL: 50 pretrain epochs, alpha 0.04, joint target, 30 finetune epochs, d=256)
  versus `rand` on FD001, FD003, MBA, PSM, ETTm1, seeds 42/123/456. Pretrained FD001 gives mean
  per-horizon AUROC 0.9012 ± 0.0068, in line with the "~0.902" REPRODUCIBILITY.md says to expect at
  alpha 0.04, so the pipeline is behaving as the authors describe. No budget reduction was needed.
- **Step 3.** 2x2 of encoder (pretrained / init) x predictor (pretrained / init) on FD001, FD003, MBA,
  restarted from the saved `pre` checkpoint and RNG state; target var/cov regulariser (`tr`) on FD001,
  FD003, MBA; our crude SIGReg-style penalty (`sig`) on FD003 and MBA; and, for MBA only, the same three
  objectives with 2400 pretraining cuts instead of the pipeline's 40 (`_c2400`), because 40 cuts means at
  most 50 optimiser steps.
- **Step 4.** `pre` vs `rand` at d=64, 128, 256 (0.16M, 0.57M, 2.16M parameters) on FD001, FD003, MBA, 3 seeds, separate per-dataset encoders, same pipeline. PSM and ETTm1 were not swept.
- **Step 5.** Not run. FD002 and FD004 do load through `load_dataset` (156 and 149 pretraining engines,
  6175 and 7981 test windows; `sizes.py`), but no model was trained on them.

## Results in words

**Step 1 — pretrained vs random encoder, authors' pipeline (Table 1).**
- FD001: random better in 3/3 seeds on both metrics. Mean per-horizon AUROC 0.9237 ± 0.0020 (random) vs
  0.9012 ± 0.0068 (pretrained); paired difference -0.0225 ± 0.0063; pooled -0.0105 ± 0.0018.
- FD003: random better in 3/3 seeds on both metrics. 0.9566 ± 0.0024 vs 0.9411 ± 0.0032; paired
  -0.0155 ± 0.0056; pooled -0.0179 ± 0.0041.
- MBA: no detectable difference. Per-horizon +0.0210 ± 0.0430 in favour of pretrained (2/3 seeds), pooled
  -0.0045 ± 0.0042 against it (0/3 seeds). The random encoder's seed-sd on per-horizon AUROC is 0.0512.
- PSM: no detectable difference. Per-horizon -0.0022 ± 0.0139 (1/3 seeds), pooled -0.0000 ± 0.0026.
- ETTm1: no detectable difference. Per-horizon -0.0029 ± 0.0172 (1/3 seeds), pooled -0.0041 ± 0.0167.
- Across the 15 dataset x seed cells the pretrained encoder has the higher per-horizon AUROC in 4.
- The C-MAPSS deficit is concentrated at long horizons (Table 6): FD001 long-third 0.8606 ± 0.0107
  (pretrained) vs 0.9116 ± 0.0054 (random), while the short third is 0.9280 vs 0.9309.
- Effect sizes are smaller than Phase 1 reported (+0.018 to +0.156 pooled for FD003 and MBA at 15 epochs
  with 1024 cuts): here FD003 is 0.018 pooled and MBA at the public pipeline is 0.0045 pooled. MBA becomes
  large only when pretraining is given more cuts (below).
- Phase 1's statement that pretrained MBA forecasts do not vary across windows (within-horizon share of
  prediction variance 0.000) is NOT reproduced in the authors' pipeline: the share is 0.385 for the
  pretrained MBA runs (0.459 random), and 0.211 with 2400 cuts.

**Step 2 — metrics (Tables 1 and 6).**
- The repo's `h_auroc` is the mean per-horizon AUROC; "pooled" is the AUROC over all (window, horizon)
  cells. A horizon-only predictor (training base rate per horizon, identical for every window) scores 0.5
  per horizon by construction and, pooled: FD001 0.8579, FD003 0.7532, MBA 0.8522, ETTm1 0.6496,
  PSM 0.5351.
- Conclusions that flip with the metric: MBA pretrained-vs-random changes sign (pooled -0.0045, per-horizon
  +0.0210), both inside seed noise; FD003 `sig`-vs-random is +0.0004 pooled (1/3 seeds) but +0.0072
  per-horizon (3/3 seeds). The C-MAPSS conclusions do not flip.
- MBA shows why the pooled number needs the horizon-only row next to it: pretrained pooled AUROC is
  0.9293 against 0.8522 from horizons alone, while its per-horizon AUROC is 0.8196.

**Step 3a/3b — where the difference lives (Table 2).**
- On C-MAPSS it is in the encoder. Pretrained encoder + re-initialised predictor is as bad as fully
  pretrained (FD001 -0.0223 ± 0.0077, FD003 -0.0163 ± 0.0036 vs random, 0/3 seeds each). Random encoder +
  pretrained predictor is indistinguishable from fully random (FD001 -0.0008 ± 0.0047, FD003
  +0.0007 ± 0.0020).
- This agrees with the paper's own ablation that the predictor's initialisation does not matter; the sign
  of the encoder's contribution is what differs.
- MBA: all four cells are within noise of each other (differences vs random between -0.0131 and +0.0272
  with sds of 0.018-0.043).

**Step 3c/3d — target-side regularisers (Tables 3 and 5).**
- Collapse is real in rank terms: on C-MAPSS the centred effective rank of the target embeddings falls
  from about 13 at initialisation to 1.3 after pretraining (mean pairwise cosine 0.65-0.66 -> 0.999).
- Target var/cov (`tr`) removes most of the C-MAPSS deficit but does not beat random: FD001
  -0.0046 ± 0.0030 per-horizon (0/3 seeds), FD003 -0.0024 ± 0.0046 (1/3). Effective rank rises to 54 and 40;
  the cosine goes to 0.9998.
- Our SIGReg-style penalty (`sig`) on FD003: +0.0072 ± 0.0077 per-horizon (3/3 seeds), +0.0004 ± 0.0056
  pooled (1/3). It did not de-collapse the targets: per-seed centred effective rank is 1.0, 1.0 and 1.6 (Table 5), and
  the seed that carries the gain (456) is the one whose targets ended in a different geometry (mean pairwise
  cosine 0.0012 instead of 0.999). Whatever it does, it is not the mechanism the hypothesis named. Treat as
  unexplained.
- MBA with 2400 cuts (about 1900 optimiser steps instead of 50): pretraining is clearly harmful,
  -0.0998 ± 0.0470 per-horizon and -0.0407 ± 0.0053 pooled vs random (0/3 seeds on both). `tr` recovers part
  (-0.0627 ± 0.0364), `sig` most (-0.0054 ± 0.0598); neither beats random.
- MBA at the public 40 cuts: `tr` +0.0210 ± 0.0434 and `sig` +0.0023 ± 0.0312, both within noise.

**Step 4 — width.** (Table 4.)
- FD001: the paired per-horizon difference is -0.0087 ± 0.0146 at d=64 (1/3 seeds), -0.0109 ± 0.0072 at d=128
  (0/3), -0.0225 ± 0.0063 at d=256 (0/3). Pooled: -0.0125 ± 0.0060, -0.0082 ± 0.0010, -0.0105 ± 0.0018. No
  closing; on the per-horizon metric the point estimate grows with width.
- FD003: -0.0428 ± 0.0260 at d=64, -0.0197 ± 0.0025 at d=128, -0.0155 ± 0.0056 at d=256, 0/3 seeds at every
  width. Pooled: -0.0364 ± 0.0172, -0.0197 ± 0.0032, -0.0179 ± 0.0041. The gap narrows from d=64 to d=128 and
  is flat within noise from d=128 to d=256; it does not reach zero.
- MBA (public 40-cut pretraining): per-horizon -0.0130 ± 0.0135, -0.0084 ± 0.0142, +0.0210 ± 0.0430, all
  within noise. Pooled is negative at every width (-0.0068 ± 0.0054, -0.0062 ± 0.0012, -0.0045 ± 0.0042) and
  the pretrained encoder has the higher pooled AUROC in 0 of 9 MBA seed x width cells.
- Target collapse is present at every width (C-MAPSS target effective rank 1.1-1.3 after pretraining).

## Verdict on each refutation condition

1. *"At the paper's own budget and pipeline the pretrained encoder beats random on per-horizon AUROC on most
   datasets."* **Not met; this leg of the hypothesis survives.** Pretrained is worse on 2 of 5 datasets
   (3/3 seeds each) and indistinguishable on the other 3. Qualifier: "the paper's own pipeline" here is the
   public repo at its PROTOCOL defaults, which REPRODUCIBILITY.md says is not the run behind the paper's
   table.
2. *"De-collapse does not beat random either."* **Met; this leg of the original hypothesis is refuted as
   worded.** Target var/cov regularisation closes most of the gap to random and never exceeds it (0/3,
   1/3, 0/3 seeds on FD001, FD003, MBA-2400). The one condition that edges above random (`sig` on FD003,
   per-horizon only) did not de-collapse the targets. The hypothesis above has been rewritten accordingly:
   de-collapsing reaches parity with the untrained encoder, not superiority.
3. *"The pretrained-minus-random gap closes with width alone."* **Not met in the range tested; this leg survives.** On C-MAPSS the pretrained
   encoder is behind the random encoder at all three widths (17 of 18 seed x width cells on per-horizon
   AUROC, 18 of 18 pooled). FD003's gap shrinks between 0.16M and 0.57M parameters and then stops shrinking;
   FD001's does not shrink. Three widths over a 13x parameter range on two related datasets cannot say what
   happens at 40M.

## Things that look too good to be true (or too clean), and how far to trust them

- **Random beats pretrained in 6/6 C-MAPSS cells.** Real in this pipeline and bit-verified, but the effect
  is 0.015-0.023 AUROC, on 30 test engines, with a split that does not change across seeds. C-MAPSS is also
  the one family with global z-scoring and no RevIN, where a random projection keeps the degradation level.
  Do not generalise from it to "random beats pretrained".
- **`sig` beats random 3/3 on FD003.** One seed carries it (+0.0161; the other two are +0.0029 and +0.0026),
  the pooled metric shows nothing, the penalty is ours and untuned, and it did not produce isotropic
  targets. Not evidence that SIGReg fixes anything.
- **The MBA "pretraining hurts by 0.10" number** only appears when we change the pipeline (2400 cuts).
  At the public setting there is no difference. It is a dose-response hint from one dataset.
- **Single-stream pretraining is at most 50 steps in the public script.** If the authors' internal runs used
  more cuts, the PSM/ETTm1/MBA rows of Table 1 say little about their models.
- **Phase 1's larger gaps and the "0.000 variance share" did not carry over** to the authors' pipeline.

## Tables (auto-generated from /home/claude/phase2/out/*.json by make_tables.py)

**Table 1 — Step 1/2: authors' pipeline, PROTOCOL (50 pretrain epochs, 30 finetune epochs, d=256), pretrained vs random-init encoder**

| dataset | condition | d | n (seeds) | pooled AUROC | mean per-horizon AUROC (repo `h_auroc`) | horizon-only pooled | Δ pooled vs rand (paired) | Δ per-horizon vs rand (paired) | seeds better than rand (per-h) | target cos init → post | target eff. rank (post) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FD001 | pre | 256 | 3 (42,123,456) | 0.9583 ± 0.0021 | 0.9012 ± 0.0068 | 0.8579 | -0.0105 ± 0.0018 (n=3) | -0.0225 ± 0.0063 (n=3) | 0/3 | 0.6526 → 0.9990 | 1.3 |
| FD001 | rand | 256 | 3 (42,123,456) | 0.9689 ± 0.0012 | 0.9237 ± 0.0020 | 0.8579 | — | — | — | 0.6526 → 0.6526 | 13.2 |
| FD003 | pre | 256 | 3 (42,123,456) | 0.9519 ± 0.0023 | 0.9411 ± 0.0032 | 0.7532 | -0.0179 ± 0.0041 (n=3) | -0.0155 ± 0.0056 (n=3) | 0/3 | 0.6608 → 0.9991 | 1.3 |
| FD003 | rand | 256 | 3 (42,123,456) | 0.9698 ± 0.0018 | 0.9566 ± 0.0024 | 0.7532 | — | — | — | 0.6608 → 0.6608 | 13.4 |
| MBA | pre | 256 | 3 (42,123,456) | 0.9293 ± 0.0027 | 0.8196 ± 0.0098 | 0.8522 | -0.0045 ± 0.0042 (n=3) | +0.0210 ± 0.0430 (n=3) | 2/3 | 0.4660 → 0.9681 | 4.6 |
| MBA | rand | 256 | 3 (42,123,456) | 0.9337 ± 0.0062 | 0.7986 ± 0.0512 | 0.8522 | — | — | — | 0.4660 → 0.4660 | 6.1 |
| PSM | pre | 256 | 3 (42,123,456) | 0.6213 ± 0.0099 | 0.6291 ± 0.0217 | 0.5351 | -0.0000 ± 0.0026 (n=3) | -0.0022 ± 0.0139 (n=3) | 1/3 | 0.1533 → 0.4017 | 2.3 |
| PSM | rand | 256 | 3 (42,123,456) | 0.6213 ± 0.0107 | 0.6312 ± 0.0079 | 0.5351 | — | — | — | 0.1533 → 0.1533 | 19.5 |
| ETTm1 | pre | 256 | 3 (42,123,456) | 0.8210 ± 0.0110 | 0.8091 ± 0.0109 | 0.6496 | -0.0041 ± 0.0167 (n=3) | -0.0029 ± 0.0172 (n=3) | 1/3 | 0.4197 → 0.8423 | 7.7 |
| ETTm1 | rand | 256 | 3 (42,123,456) | 0.8251 ± 0.0085 | 0.8120 ± 0.0099 | 0.6496 | — | — | — | 0.4197 → 0.4197 | 12.4 |

**Table 2 — Step 3a/3b: 2x2 encoder x predictor-init (finetune restarted from the same checkpoint and RNG state as the `pre` run)**

| dataset | condition | d | n (seeds) | pooled AUROC | mean per-horizon AUROC (repo `h_auroc`) | horizon-only pooled | Δ pooled vs rand (paired) | Δ per-horizon vs rand (paired) | seeds better than rand (per-h) | target cos init → post | target eff. rank (post) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FD001 | pre | 256 | 3 (42,123,456) | 0.9583 ± 0.0021 | 0.9012 ± 0.0068 | 0.8579 | -0.0105 ± 0.0018 (n=3) | -0.0225 ± 0.0063 (n=3) | 0/3 | 0.6526 → 0.9990 | 1.3 |
| FD001 | preE_newP | 256 | 3 (42,123,456) | 0.9594 ± 0.0028 | 0.9014 ± 0.0080 | 0.8579 | -0.0095 ± 0.0027 (n=3) | -0.0223 ± 0.0077 (n=3) | 0/3 | 0.6526 → 0.9990 | 1.3 |
| FD001 | randE_preP | 256 | 3 (42,123,456) | 0.9686 ± 0.0013 | 0.9229 ± 0.0058 | 0.8579 | -0.0002 ± 0.0007 (n=3) | -0.0008 ± 0.0047 (n=3) | 1/3 | 0.6526 → 0.9990 | 1.3 |
| FD001 | rand | 256 | 3 (42,123,456) | 0.9689 ± 0.0012 | 0.9237 ± 0.0020 | 0.8579 | — | — | — | 0.6526 → 0.6526 | 13.2 |
| FD003 | pre | 256 | 3 (42,123,456) | 0.9519 ± 0.0023 | 0.9411 ± 0.0032 | 0.7532 | -0.0179 ± 0.0041 (n=3) | -0.0155 ± 0.0056 (n=3) | 0/3 | 0.6608 → 0.9991 | 1.3 |
| FD003 | preE_newP | 256 | 3 (42,123,456) | 0.9516 ± 0.0014 | 0.9403 ± 0.0011 | 0.7532 | -0.0183 ± 0.0031 (n=3) | -0.0163 ± 0.0036 (n=3) | 0/3 | 0.6608 → 0.9991 | 1.3 |
| FD003 | randE_preP | 256 | 3 (42,123,456) | 0.9698 ± 0.0021 | 0.9574 ± 0.0034 | 0.7532 | -0.0000 ± 0.0007 (n=3) | +0.0007 ± 0.0020 (n=3) | 2/3 | 0.6608 → 0.9991 | 1.3 |
| FD003 | rand | 256 | 3 (42,123,456) | 0.9698 ± 0.0018 | 0.9566 ± 0.0024 | 0.7532 | — | — | — | 0.6608 → 0.6608 | 13.4 |
| MBA | pre | 256 | 3 (42,123,456) | 0.9293 ± 0.0027 | 0.8196 ± 0.0098 | 0.8522 | -0.0045 ± 0.0042 (n=3) | +0.0210 ± 0.0430 (n=3) | 2/3 | 0.4660 → 0.9681 | 4.6 |
| MBA | preE_newP | 256 | 3 (42,123,456) | 0.9304 ± 0.0026 | 0.8258 ± 0.0118 | 0.8522 | -0.0034 ± 0.0048 (n=3) | +0.0272 ± 0.0416 (n=3) | 2/3 | 0.4660 → 0.9681 | 4.6 |
| MBA | randE_preP | 256 | 3 (42,123,456) | 0.9300 ± 0.0062 | 0.7856 ± 0.0477 | 0.8522 | -0.0037 ± 0.0045 (n=3) | -0.0131 ± 0.0177 (n=3) | 1/3 | 0.4660 → 0.9681 | 4.6 |
| MBA | rand | 256 | 3 (42,123,456) | 0.9337 ± 0.0062 | 0.7986 ± 0.0512 | 0.8522 | — | — | — | 0.4660 → 0.4660 | 6.1 |

**Table 3 — Step 3c/3d: target-side regularisers (tr = var/cov penalty also on targets; sig = OUR crude SIGReg-style Epps-Pulley penalty on targets, not the paper's)**

| dataset | condition | d | n (seeds) | pooled AUROC | mean per-horizon AUROC (repo `h_auroc`) | horizon-only pooled | Δ pooled vs rand (paired) | Δ per-horizon vs rand (paired) | seeds better than rand (per-h) | target cos init → post | target eff. rank (post) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FD001 | pre | 256 | 3 (42,123,456) | 0.9583 ± 0.0021 | 0.9012 ± 0.0068 | 0.8579 | -0.0105 ± 0.0018 (n=3) | -0.0225 ± 0.0063 (n=3) | 0/3 | 0.6526 → 0.9990 | 1.3 |
| FD001 | tr | 256 | 3 (42,123,456) | 0.9668 ± 0.0019 | 0.9191 ± 0.0040 | 0.8579 | -0.0020 ± 0.0007 (n=3) | -0.0046 ± 0.0030 (n=3) | 0/3 | 0.6526 → 0.9998 | 54.2 |
| FD001 | rand | 256 | 3 (42,123,456) | 0.9689 ± 0.0012 | 0.9237 ± 0.0020 | 0.8579 | — | — | — | 0.6526 → 0.6526 | 13.2 |
| FD003 | pre | 256 | 3 (42,123,456) | 0.9519 ± 0.0023 | 0.9411 ± 0.0032 | 0.7532 | -0.0179 ± 0.0041 (n=3) | -0.0155 ± 0.0056 (n=3) | 0/3 | 0.6608 → 0.9991 | 1.3 |
| FD003 | tr | 256 | 3 (42,123,456) | 0.9618 ± 0.0064 | 0.9542 ± 0.0070 | 0.7532 | -0.0080 ± 0.0053 (n=3) | -0.0024 ± 0.0046 (n=3) | 1/3 | 0.6608 → 0.9998 | 40.2 |
| FD003 | sig | 256 | 3 (42,123,456) | 0.9702 ± 0.0052 | 0.9638 ± 0.0062 | 0.7532 | +0.0004 ± 0.0056 (n=3) | +0.0072 ± 0.0077 (n=3) | 3/3 | 0.6608 → 0.6666 | 1.2 |
| FD003 | rand | 256 | 3 (42,123,456) | 0.9698 ± 0.0018 | 0.9566 ± 0.0024 | 0.7532 | — | — | — | 0.6608 → 0.6608 | 13.4 |
| MBA | pre | 256 | 3 (42,123,456) | 0.9293 ± 0.0027 | 0.8196 ± 0.0098 | 0.8522 | -0.0045 ± 0.0042 (n=3) | +0.0210 ± 0.0430 (n=3) | 2/3 | 0.4660 → 0.9681 | 4.6 |
| MBA | tr | 256 | 3 (42,123,456) | 0.9301 ± 0.0041 | 0.8196 ± 0.0080 | 0.8522 | -0.0037 ± 0.0070 (n=3) | +0.0210 ± 0.0434 (n=3) | 2/3 | 0.4660 → 0.9757 | 10.1 |
| MBA | sig | 256 | 3 (42,123,456) | 0.9277 ± 0.0051 | 0.8010 ± 0.0221 | 0.8522 | -0.0061 ± 0.0011 (n=3) | +0.0023 ± 0.0312 (n=3) | 2/3 | 0.4660 → 0.4922 | 1.8 |
| MBA | rand | 256 | 3 (42,123,456) | 0.9337 ± 0.0062 | 0.7986 ± 0.0512 | 0.8522 | — | — | — | 0.4660 → 0.4660 | 6.1 |
| MBA | pre_c2400 | 256 | 3 (42,123,456) | 0.8931 ± 0.0112 | 0.6988 ± 0.0779 | 0.8522 | -0.0407 ± 0.0053 (n=3) | -0.0998 ± 0.0470 (n=3) | 0/3 | 0.4660 → 0.9985 | 2.1 |
| MBA | tr_c2400 | 256 | 3 (42,123,456) | 0.9061 ± 0.0094 | 0.7359 ± 0.0368 | 0.8522 | -0.0277 ± 0.0045 (n=3) | -0.0627 ± 0.0364 (n=3) | 0/3 | 0.4660 → 0.9991 | 36.6 |
| MBA | sig_c2400 | 256 | 3 (42,123,456) | 0.9141 ± 0.0070 | 0.7933 ± 0.0156 | 0.8522 | -0.0197 ± 0.0123 (n=3) | -0.0054 ± 0.0598 (n=3) | 2/3 | 0.4660 → 0.0151 | 1.8 |

**Table 4 — Step 4: width sweep, pretrained vs random (separate per-dataset encoders)**

| dataset | condition | d | n (seeds) | pooled AUROC | mean per-horizon AUROC (repo `h_auroc`) | horizon-only pooled | Δ pooled vs rand (paired) | Δ per-horizon vs rand (paired) | seeds better than rand (per-h) | target cos init → post | target eff. rank (post) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FD001 | pre | 64 | 3 (42,123,456) | 0.9449 ± 0.0028 | 0.8845 ± 0.0011 | 0.8579 | -0.0125 ± 0.0060 (n=3) | -0.0087 ± 0.0146 (n=3) | 1/3 | 0.6973 → 0.9967 | 1.1 |
| FD001 | rand | 64 | 3 (42,123,456) | 0.9573 ± 0.0033 | 0.8932 ± 0.0149 | 0.8579 | — | — | — | 0.6973 → 0.6973 | 9.3 |
| FD001 | pre | 128 | 3 (42,123,456) | 0.9578 ± 0.0012 | 0.9050 ± 0.0039 | 0.8579 | -0.0082 ± 0.0010 (n=3) | -0.0109 ± 0.0072 (n=3) | 0/3 | 0.6790 → 0.9970 | 1.1 |
| FD001 | rand | 128 | 3 (42,123,456) | 0.9660 ± 0.0022 | 0.9160 ± 0.0110 | 0.8579 | — | — | — | 0.6790 → 0.6790 | 10.7 |
| FD001 | pre | 256 | 3 (42,123,456) | 0.9583 ± 0.0021 | 0.9012 ± 0.0068 | 0.8579 | -0.0105 ± 0.0018 (n=3) | -0.0225 ± 0.0063 (n=3) | 0/3 | 0.6526 → 0.9990 | 1.3 |
| FD001 | rand | 256 | 3 (42,123,456) | 0.9689 ± 0.0012 | 0.9237 ± 0.0020 | 0.8579 | — | — | — | 0.6526 → 0.6526 | 13.2 |
| FD003 | pre | 64 | 3 (42,123,456) | 0.9162 ± 0.0160 | 0.8949 ± 0.0267 | 0.7532 | -0.0364 ± 0.0172 (n=3) | -0.0428 ± 0.0260 (n=3) | 0/3 | 0.6924 → 0.9973 | 1.1 |
| FD003 | rand | 64 | 3 (42,123,456) | 0.9526 ± 0.0038 | 0.9378 ± 0.0046 | 0.7532 | — | — | — | 0.6924 → 0.6924 | 9.2 |
| FD003 | pre | 128 | 3 (42,123,456) | 0.9428 ± 0.0035 | 0.9297 ± 0.0031 | 0.7532 | -0.0197 ± 0.0032 (n=3) | -0.0197 ± 0.0025 (n=3) | 0/3 | 0.7027 → 0.9971 | 1.1 |
| FD003 | rand | 128 | 3 (42,123,456) | 0.9624 ± 0.0018 | 0.9495 ± 0.0007 | 0.7532 | — | — | — | 0.7027 → 0.7027 | 10.9 |
| FD003 | pre | 256 | 3 (42,123,456) | 0.9519 ± 0.0023 | 0.9411 ± 0.0032 | 0.7532 | -0.0179 ± 0.0041 (n=3) | -0.0155 ± 0.0056 (n=3) | 0/3 | 0.6608 → 0.9991 | 1.3 |
| FD003 | rand | 256 | 3 (42,123,456) | 0.9698 ± 0.0018 | 0.9566 ± 0.0024 | 0.7532 | — | — | — | 0.6608 → 0.6608 | 13.4 |
| MBA | pre | 64 | 3 (42,123,456) | 0.9133 ± 0.0087 | 0.7959 ± 0.0214 | 0.8522 | -0.0068 ± 0.0054 (n=3) | -0.0130 ± 0.0135 (n=3) | 0/3 | 0.5142 → 0.8087 | 5.9 |
| MBA | rand | 64 | 3 (42,123,456) | 0.9201 ± 0.0119 | 0.8090 ± 0.0348 | 0.8522 | — | — | — | 0.5142 → 0.5142 | 6.2 |
| MBA | pre | 128 | 3 (42,123,456) | 0.9270 ± 0.0059 | 0.8025 ± 0.0438 | 0.8522 | -0.0062 ± 0.0012 (n=3) | -0.0084 ± 0.0142 (n=3) | 1/3 | 0.4846 → 0.9118 | 5.6 |
| MBA | rand | 128 | 3 (42,123,456) | 0.9332 ± 0.0049 | 0.8110 ± 0.0303 | 0.8522 | — | — | — | 0.4846 → 0.4846 | 6.1 |
| MBA | pre | 256 | 3 (42,123,456) | 0.9293 ± 0.0027 | 0.8196 ± 0.0098 | 0.8522 | -0.0045 ± 0.0042 (n=3) | +0.0210 ± 0.0430 (n=3) | 2/3 | 0.4660 → 0.9681 | 4.6 |
| MBA | rand | 256 | 3 (42,123,456) | 0.9337 ± 0.0062 | 0.7986 ± 0.0512 | 0.8522 | — | — | — | 0.4660 → 0.4660 | 6.1 |

**Table 5 — per-seed embedding diagnostics after pretraining (512 fixed pretraining cuts)**

| run | target cos | target eff. rank (centred) | context cos | context eff. rank | predictor-output cos |
|---|---|---|---|---|---|
| ETTm1 pre s123 | 0.6131 | 12.3 | 0.6315 | 13.3 | 0.7458 |
| ETTm1 pre s42 | 0.9489 | 5.3 | 0.9069 | 13.3 | 0.9741 |
| ETTm1 pre s456 | 0.9648 | 5.5 | 0.8996 | 13.2 | 0.9723 |
| ETTm1 rand s123 | 0.4163 | 13.8 | 0.5425 | 13.3 | 0.6306 |
| ETTm1 rand s42 | 0.4114 | 12.5 | 0.5637 | 13.1 | 0.6373 |
| ETTm1 rand s456 | 0.4313 | 10.8 | 0.5837 | 12.1 | 0.6091 |
| FD001 pre s123 | 0.9991 | 1.4 | 0.9914 | 20.4 | 0.9991 |
| FD001 pre s42 | 0.9990 | 1.3 | 0.9928 | 20.7 | 0.9991 |
| FD001 pre s456 | 0.9990 | 1.3 | 0.9932 | 22.7 | 0.9990 |
| FD001 rand s123 | 0.6488 | 15.3 | 0.5769 | 20.9 | 0.6188 |
| FD001 rand s42 | 0.6396 | 12.5 | 0.5724 | 21.1 | 0.6599 |
| FD001 rand s456 | 0.6695 | 11.9 | 0.6213 | 20.1 | 0.6479 |
| FD001 tr s123 | 0.9998 | 55.7 | 0.9481 | 10.2 | 0.9997 |
| FD001 tr s42 | 0.9998 | 53.8 | 0.9528 | 7.1 | 0.9997 |
| FD001 tr s456 | 0.9998 | 53.0 | 0.9627 | 7.4 | 0.9997 |
| FD003 pre s123 | 0.9991 | 1.4 | 0.9913 | 20.6 | 0.9991 |
| FD003 pre s42 | 0.9991 | 1.4 | 0.9931 | 23.3 | 0.9992 |
| FD003 pre s456 | 0.9989 | 1.3 | 0.9930 | 21.9 | 0.9989 |
| FD003 rand s123 | 0.6557 | 15.2 | 0.5895 | 20.5 | 0.6363 |
| FD003 rand s42 | 0.6666 | 13.6 | 0.5821 | 20.8 | 0.6501 |
| FD003 rand s456 | 0.6600 | 11.5 | 0.6282 | 20.0 | 0.6425 |
| FD003 sig s123 | 0.9992 | 1.0 | 0.8911 | 9.4 | 0.9994 |
| FD003 sig s42 | 0.9993 | 1.0 | 0.9032 | 9.7 | 0.9995 |
| FD003 sig s456 | 0.0012 | 1.6 | 0.6987 | 1.5 | 0.0012 |
| FD003 tr s123 | 0.9998 | 42.5 | 0.9334 | 9.2 | 0.9996 |
| FD003 tr s42 | 0.9998 | 41.0 | 0.9514 | 8.9 | 0.9997 |
| FD003 tr s456 | 0.9998 | 37.2 | 0.9544 | 7.3 | 0.9997 |
| MBA pre s123 | 0.9729 | 4.4 | 0.9674 | 10.5 | 0.9701 |
| MBA pre_c2400 s123 | 0.9982 | 1.8 | 0.9853 | 7.3 | 0.9982 |
| MBA pre s42 | 0.9606 | 4.4 | 0.9596 | 10.4 | 0.9649 |
| MBA pre_c2400 s42 | 0.9985 | 2.7 | 0.9893 | 6.5 | 0.9985 |
| MBA pre s456 | 0.9709 | 5.1 | 0.9603 | 9.9 | 0.9757 |
| MBA pre_c2400 s456 | 0.9990 | 1.9 | 0.9934 | 9.3 | 0.9990 |
| MBA rand s123 | 0.4898 | 5.8 | 0.8564 | 13.0 | 0.6729 |
| MBA rand s42 | 0.4942 | 6.4 | 0.8378 | 11.4 | 0.6572 |
| MBA rand s456 | 0.4139 | 6.0 | 0.8650 | 11.9 | 0.6625 |
| MBA sig s123 | 0.4801 | 1.8 | 0.9585 | 12.2 | 0.5858 |
| MBA sig_c2400 s123 | 0.0113 | 1.7 | 0.9945 | 8.0 | 0.0115 |
| MBA sig s42 | 0.4283 | 1.7 | 0.9552 | 10.9 | 0.5628 |
| MBA sig_c2400 s42 | 0.0153 | 1.7 | 0.9906 | 7.9 | 0.0154 |
| MBA sig s456 | 0.5684 | 1.9 | 0.9556 | 10.7 | 0.6811 |
| MBA sig_c2400 s456 | 0.0187 | 1.9 | 0.9925 | 8.5 | 0.0166 |
| MBA tr s123 | 0.9772 | 8.3 | 0.9671 | 11.0 | 0.9806 |
| MBA tr_c2400 s123 | 0.9989 | 37.9 | 0.9892 | 20.6 | 0.9995 |
| MBA tr s42 | 0.9738 | 11.5 | 0.9590 | 10.9 | 0.9842 |
| MBA tr_c2400 s42 | 0.9993 | 34.8 | 0.9813 | 6.6 | 0.9996 |
| MBA tr s456 | 0.9762 | 10.5 | 0.9581 | 10.4 | 0.9842 |
| MBA tr_c2400 s456 | 0.9992 | 37.1 | 0.9878 | 5.2 | 0.9997 |
| PSM pre s123 | 0.4353 | 2.5 | 0.8741 | 29.7 | 0.9490 |
| PSM pre s42 | 0.3373 | 2.1 | 0.9278 | 30.1 | 0.9742 |
| PSM pre s456 | 0.4326 | 2.3 | 0.9216 | 22.9 | 0.9392 |
| PSM rand s123 | 0.1547 | 19.6 | 0.8039 | 34.6 | 0.6709 |
| PSM rand s42 | 0.1564 | 20.5 | 0.7775 | 29.5 | 0.6715 |
| PSM rand s456 | 0.1488 | 18.4 | 0.7659 | 29.4 | 0.6555 |

**Table 6 — per-horizon AUROC by horizon band (mean over valid horizons in each third of the grid, then mean ± sd over seeds), d=256**

| dataset | condition | n | short third | middle third | long third |
|---|---|---|---|---|---|
| ETTm1 | pre | 3 | 0.8819 ± 0.0125 | 0.7979 ± 0.0089 | 0.7487 ± 0.0115 |
| ETTm1 | rand | 3 | 0.8864 ± 0.0061 | 0.7986 ± 0.0121 | 0.7521 ± 0.0119 |
| FD001 | pre | 3 | 0.9280 ± 0.0036 | 0.9151 ± 0.0076 | 0.8606 ± 0.0107 |
| FD001 | rand | 3 | 0.9309 ± 0.0052 | 0.9286 ± 0.0039 | 0.9116 ± 0.0054 |
| FD003 | pre | 3 | 0.9375 ± 0.0044 | 0.9462 ± 0.0042 | 0.9396 ± 0.0053 |
| FD003 | rand | 3 | 0.9376 ± 0.0016 | 0.9692 ± 0.0027 | 0.9631 ± 0.0039 |
| MBA | pre | 3 | 0.9001 ± 0.0074 | 0.8052 ± 0.0133 | 0.7548 ± 0.0150 |
| MBA | rand | 3 | 0.9297 ± 0.0089 | 0.7732 ± 0.0595 | 0.6950 ± 0.0980 |
| PSM | pre | 3 | 0.6238 ± 0.0260 | 0.6312 ± 0.0194 | 0.6321 ± 0.0197 |
| PSM | rand | 3 | 0.6290 ± 0.0104 | 0.6320 ± 0.0065 | 0.6326 ± 0.0069 |

**Table 7 — measured wall-clock per run (seconds; 1 torch thread, two runs sharing 2 CPU cores)**

| dataset | condition | d | params | pretrain s | finetune s | eval s | finetune epochs run (last epoch index per seed) |
|---|---|---|---|---|---|---|---|
| ETTm1 | pre | 256 | 2.10M | 7 | 682 | 22 | [17, 16, 11] |
| ETTm1 | rand | 256 | 2.10M | 1 | 606 | 22 | [11, 17, 11] |
| FD001 | pre | 128 | 0.57M | 147 | 38 | 1 | [29, 29, 29] |
| FD001 | pre | 256 | 2.16M | 323 | 93 | 2 | [29, 29, 29] |
| FD001 | pre | 64 | 0.16M | 85 | 19 | 1 | [29, 29, 29] |
| FD001 | rand | 128 | 0.57M | 1 | 38 | 1 | [29, 29, 29] |
| FD001 | rand | 256 | 2.16M | 1 | 91 | 2 | [29, 29, 29] |
| FD001 | rand | 64 | 0.16M | 1 | 18 | 0 | [29, 29, 29] |
| FD003 | pre | 128 | 0.57M | 176 | 66 | 1 | [29, 29, 29] |
| FD003 | pre | 256 | 2.16M | 389 | 158 | 4 | [29, 29, 29] |
| FD003 | pre | 64 | 0.16M | 100 | 33 | 1 | [29, 29, 29] |
| FD003 | rand | 128 | 0.57M | 1 | 65 | 1 | [29, 29, 29] |
| FD003 | rand | 256 | 2.16M | 1 | 155 | 3 | [29, 29, 29] |
| FD003 | rand | 64 | 0.16M | 1 | 32 | 1 | [29, 29, 29] |
| MBA | pre | 128 | 0.52M | 3 | 34 | 1 | [29, 29, 29] |
| MBA | pre | 256 | 2.06M | 5 | 89 | 2 | [29, 29, 29] |
| MBA | pre_c2400 | 256 | 2.06M | 235 | 88 | 2 | [29, 27, 29] |
| MBA | pre | 64 | 0.13M | 2 | 14 | 0 | [29, 29, 29] |
| MBA | rand | 128 | 0.52M | 1 | 34 | 1 | [29, 29, 28] |
| MBA | rand | 256 | 2.06M | 1 | 70 | 2 | [23, 16, 29] |
| MBA | rand | 64 | 0.13M | 1 | 15 | 0 | [29, 29, 29] |
| PSM | pre | 256 | 2.25M | 6 | 547 | 30 | [16, 13, 11] |
| PSM | rand | 256 | 2.25M | 1 | 432 | 30 | [11, 10, 10] |


## 3-step plan: CPU pilot -> single GPU -> foundation scale

All GPU-hour figures are extrapolations, not measurements. Inputs and arithmetic are in
`/home/claude/phase2/gpu_est.py` -> `gpu_est.json` (also under `gpu_estimate` in `numbers.json`).

**Step A — finish the CPU pilot (no GPU; roughly 6-8 more core-hours).**
What is still missing from this report (see "Incomplete"), plus two controls this pilot did not run:
`target_mode='frozen_target'` (REPRODUCIBILITY.md says the run behind the paper's table used a detached
target, i.e. a target encoder left at initialisation; that is the configuration to compare against a random
encoder before anything is said about the paper's numbers), and a pretraining-length dose-response
(0, 5, 15, 50 epochs) on FD001/FD003.
Measured cost per full run (pretrain + finetune + eval, d=256, one thread; Table 7): FD001 418 s,
FD003 551 s, MBA 96 s, PSM 583 s, ETTm1 711 s (3-seed means).

**Step B — single GPU: the full 14-benchmark protocol with the random-encoder control.**
14 datasets x 5 seeds x 5 conditions (published objective, random encoder, target var/cov, the paper's real
SIGReg, frozen target) = 350 runs at d=256. The mean measured cost over our five datasets is 472
core-seconds per run, so 350 runs is about 46 core-hours. The only GPU
calibration we have is the paper's statement that pretraining takes under 1 minute per dataset on one A10G,
against our measured 323 s for FD001 pretraining on one core: a speed-up of at least 5.4x. At that lower
bound the d=256 grid is about 8.5 A10G-hours. Width rungs d=512 and d=1024 would cost about 3.9x and 15.5x
the d=256 CPU time if cost scaled with parameter count. That is an upper bound: in our own sweep the cost
grew more slowly than parameters (FD001 pretraining took 85 s, 147 s and 323 s at 0.16M, 0.57M and 2.16M),
and a GPU is far from saturated by a 2M-parameter model. A realistic total for the whole Step B grid is on
the order of 20-60 GPU-hours on one A10G/A100-class card. We could not measure this.
Kill criterion for Step B: if the published objective beats the random encoder by more than one seed-sd on
most of the 14 datasets at the paper's configuration of record, the hypothesis is dead and we say so.

**Step C — foundation scale (FactoryNet + 14 benchmarks, KITE variate tokens).**
Extrapolation method: measured pretraining throughput is 308 windows/s per core at 2.16M parameters and at
most 32 context tokens (FD003). Cost per window is scaled linearly in parameter count and in tokens per
window. Assumptions, none of them measured: KITE variate tokens give C x 32 tokens with C = 20 variates on
average; one pretraining window per 16-step stride over FactoryNet's 51M datapoints (3.19M windows per
epoch); 30 epochs; an A100-class GPU delivers 500-2000x the throughput of one of our CPU cores once the model
is large enough to fill it. Under those assumptions one pretraining run costs about 0.9-3.4 GPU-hours at 2M
parameters, 4-16 at 10M and 16-64 at 40M. Three pretraining objectives x three seeds x three sizes is about
190-750 GPU-hours; the random-encoder control needs no pretraining, only the cheap frozen-encoder finetunes.
KITE's variate-token encoder is not in the public repo, so the token-count assumption is the weakest link:
an error of 2x in the average variate count moves every Step C figure by 2x.
The question Step C answers: does the pretrained-minus-random gap change sign as corpus size and model size
grow together, and does that depend on keeping the targets from collapsing?

## Why this matters for the 2027 TSFM / world-model papers

- A frozen random-init encoder with a trained head is a strong baseline in time-series classification
  (ROCKET is the known precedent: random convolutional kernels plus a linear classifier). Our C-MAPSS result
  says the same kind of baseline is competitive for event forecasting with a HEPA head. A foundation-model
  paper that reports "pretrained frozen encoder + small head beats baselines" is much stronger if the table
  also has the row "same architecture, encoder never trained".
- The cost is negligible: the random-encoder row needs no pretraining and reuses the finetune pipeline
  (`scripts/train.py --pre-epochs 0` already does it; we verified it leaves the encoder at initialisation
  and runs the identical finetune path).
- On the metric: the repo's `h_auroc` is already the mean per-horizon AUROC, so the paper's headline metric
  is horizon-debiased. Phase 1's "pooled h-AUROC" was the AUROC over all (window, horizon) cells, which is a
  different number. What is worth adding to an evaluation protocol is (i) the horizon-only baseline next to
  any pooled number, because on our datasets it alone reaches 0.54-0.86 pooled AUROC, and (ii) the
  random-encoder row. Together they separate "the head learned the horizon prior", "the architecture's
  inductive bias is enough" and "pretraining added something".
- For a world-model claim the relevant quantity is what pretraining adds over the untrained architecture.
  Showing where (which objective, which scale, how much data) that turns positive is a result either way.

## Caveats

- CPU only, 2 cores, two runs sharing them. Three seeds per cell; the seed changes initialisation, cut
  sampling and shuffling but not the train/test split, so the ± is not a data-resampling interval.
- Compute: the brief allowed about 3.5 hours of queued training. Because of the interruption and the
  coordinator's instruction to rerun what was killed, the total was higher (see
  `/home/claude/scout/queue.log`, tags `p2-*`).
- Five of the paper's fourteen benchmarks. The two where the random encoder clearly wins are both C-MAPSS
  subsets, which use global z-scoring with no per-window RevIN; remaining useful life is carried by the
  absolute sensor level, which any random projection preserves. This may be a property of C-MAPSS rather than
  of the objective.
- On the single-stream datasets (MBA, PSM, ETTm1) the public `scripts/train.py` builds 40 pretraining cuts
  in total (`n_cuts=40` per sequence, one sequence): one batch per epoch and at most 50 optimiser steps.
  "Pretrained vs random" on those datasets at the public pipeline therefore compares a random encoder with
  one that has taken at most 50 steps. We do not know whether the paper's internal runs used more cuts.
- The public code differs from the paper in ways its own REPRODUCIBILITY.md lists: var/cov (VICReg-style)
  regulariser on the predictor output rather than SIGReg, alpha 0.04, jointly trained target encoder versus
  a detached target in the run behind the paper's table, 50/30 epochs here versus 100/50 in the paper
  snapshot. Nothing here is a statement about the paper's reported numbers.
- "sig" is our own short Epps-Pulley penalty on random 1-D projections of the target embeddings, with one
  untuned weight. It is not the paper's SIGReg and it did not produce isotropic targets (Table 5).
- Mean pairwise cosine is a poor collapse diagnostic on its own: it is dominated by a shared mean vector.
  The target var/cov penalty raises the centred effective rank of the targets from about 1.3 to 40-54 on
  C-MAPSS while the cosine goes up, not down. We report both.
- Tone: none of this says the code is broken. The pipeline does what it documents. The finding is about
  what the objective needs in order to add value over the untrained architecture, and at what scale.
