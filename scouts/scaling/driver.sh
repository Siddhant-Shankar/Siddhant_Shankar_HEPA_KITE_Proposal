#!/usr/bin/env bash
# Each line is its own queue entry so other agents can interleave.
cd /home/claude/scout/scaling
Q="bash /home/claude/scout/run_queued.sh"
R=/home/claude/scout/scaling/results
full(){ $Q scaling-full-d$1-s$2 python3 -W ignore /home/claude/HEPA/exp_shared_eps.py --seed $2 --d-model $1 --pre-epochs 20 --out $R > logs/full_d$1_s$2.txt 2>&1; }
dom(){ $Q scaling-$1 python3 -W ignore run_domains.py --tag $1 --pretrain $2 --eval $3 --seed $4 --d-model $5 --pre-epochs ${6:-20} > logs/$1.txt 2>&1; }
full 64 123
full 64 456
for s in 42 123 456; do
  dom dom2_d128_s$s FD001,MBA FD001,MBA $s 128
  dom dom3_d128_s$s FD001,MBA,FD003 FD001,MBA $s 128
done
full 128 123
for s in 42 123; do
  dom sep_FD001_d384_s$s FD001 FD001 $s 384
  dom sep_MBA_d384_s$s MBA MBA $s 384
done
full 128 456
dom shared5_d64_e40_s42 FD001,FD003,MBA,PSM,ETTm1 FD001,FD003,MBA,PSM,ETTm1 42 64 40
dom shared5_d64_e40_s123 FD001,FD003,MBA,PSM,ETTm1 FD001,FD003,MBA,PSM,ETTm1 123 64 40
echo ALLDONE > logs/driver_done
