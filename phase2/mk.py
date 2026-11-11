"""Emit a job line unless its result exists. usage: mk.py DATASET COND SEED [extra args...]"""
import sys, os
ds, cond, seed = sys.argv[1:4]; extra = sys.argv[4:]
d = "256"; tag = ""
for i, a in enumerate(extra):
    if a == "--d-model": d = extra[i+1]
    if a == "--tag": tag = extra[i+1]
name = f"{ds}_{cond}_d{d}_s{seed}{tag}"
if not os.path.exists(f"/home/claude/phase2/out/{name}.json"):
    print(f"python3 -W ignore p2.py --dataset {ds} --cond {cond} --seed {seed} {' '.join(extra)} > logs/{name}.log 2>&1")
