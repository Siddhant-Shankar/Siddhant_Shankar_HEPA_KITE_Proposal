"""GPU-hour extrapolation from measured CPU timings. All assumptions printed."""
import json
G = json.load(open("/home/claude/phase2/agg.json"))["groups"]
out = {}
# measured: pretraining throughput at d=256 (2.16M params), batch 64, 38 batches/epoch, 50 epochs on C-MAPSS
for ds in ("FD001", "FD003"):
    g = G[f"{ds}|pre|256|"]; steps = 38 * 50
    out[f"{ds}_pretrain_s_per_step_cpu"] = g["t_pretrain_s"] / steps
    out[f"{ds}_pretrain_windows_per_s_cpu"] = 2400 * 50 / g["t_pretrain_s"]
# measured core-seconds for one full (pretrain + finetune + eval) run per dataset at d=256
per = {}
for ds in ("FD001", "FD003", "MBA", "PSM", "ETTm1"):
    k = f"{ds}|pre|256|"
    if k in G: per[ds] = G[k]["t_pretrain_s"] + G[k]["t_finetune_s"] + G[k]["t_eval_s"]
out["core_s_per_pre_run_d256"] = per
mean_run = sum(per.values()) / len(per)
out["mean_core_s_per_run_d256"] = mean_run
# authors' statement: pretraining < 1 min per dataset on one A10G (paper) -> lower bound on A10G/CPU-core speedup at d=256
spd = G["FD001|pre|256|"]["t_pretrain_s"] / 60.0
out["a10g_vs_core_speedup_lower_bound_from_FD001"] = spd
# Stage 2 (single GPU): 14 datasets x 5 seeds x 5 conditions (pre, rand, target-reg, real SIGReg, frozen-target) at d=256
runs2 = 14 * 5 * 5
core_h2 = runs2 * mean_run / 3600
out["stage2"] = {"runs_d256": runs2, "core_hours_d256": core_h2, "gpu_hours_d256_at_lower_bound_speedup": core_h2 / spd,
                 "width_rungs": "d=128,256,512,1024 (cost assumed ∝ params: x0.26, x1, x~3.9, x~15.5 of d=256)",
                 "gpu_hours_all_widths_at_lower_bound_speedup": core_h2 / spd * (0.26 + 1 + 3.9 + 15.5)}
# Stage 3 (foundation): FLOP arithmetic calibrated on the measured CPU throughput.
# cost per training window ∝ (encoder params) x (tokens per window). Calibration point: d=256 HEPA, channel-fused,
# <=32 context tokens (512/16) + <=10 target tokens, measured windows/s on one core.
w_per_s = out["FD003_pretrain_windows_per_s_cpu"]
P0, tok0 = 2.16e6, 32
gpu_over_core = (500, 2000)   # ASSUMED A100-class throughput relative to one of these CPU cores for models large enough to fill the GPU
corpus_windows = 51e6 / 16     # FactoryNet 51M datapoints, one window per 16-step patch stride (ASSUMED)
epochs = 30                    # ASSUMED
st3 = {}
for name, P, C in (("2M", 2.16e6, 20), ("10M", 10e6, 20), ("40M", 40e6, 20)):
    tok = 32 * C               # KITE variate tokens: C variates x 32 patches (C=20 ASSUMED average)
    core_s = corpus_windows * epochs / w_per_s * (P / P0) * (tok / tok0)
    st3[name] = {"core_hours": core_s / 3600, "gpu_hours_low": core_s / 3600 / gpu_over_core[1], "gpu_hours_high": core_s / 3600 / gpu_over_core[0]}
out["stage3_per_pretraining_run"] = st3
n_runs3 = 4 * 3  # conditions (published objective, target-reg, SIGReg, random-init control needs no pretraining -> 3 pretrain conds) ...
out["stage3_assumptions"] = {"gpu_over_core": gpu_over_core, "corpus_windows_per_epoch": corpus_windows, "epochs": epochs,
                             "variates": 20, "pretrain_conditions": 3, "seeds": 3}
tot_lo = sum(v["gpu_hours_low"] for v in st3.values()) * 9
tot_hi = sum(v["gpu_hours_high"] for v in st3.values()) * 9
out["stage3_total_gpu_hours_3conds_3seeds_3sizes"] = [tot_lo, tot_hi]
json.dump(out, open("/home/claude/phase2/gpu_est.json", "w"), indent=1)
print(json.dumps(out, indent=1))
