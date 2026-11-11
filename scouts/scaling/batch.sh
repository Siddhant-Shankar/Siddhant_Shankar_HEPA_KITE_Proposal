#!/usr/bin/env bash
# Runs INSIDE one queue slot: a batch of short runs (<=~12 min total) to cut queue round-trips.
cd /home/claude/scout/scaling
R=/home/claude/scout/scaling/results
full(){ python3 -W ignore /home/claude/HEPA/exp_shared_eps.py --seed $2 --d-model $1 --pre-epochs 20 --out $R > logs/full_d$1_s$2.txt 2>&1; }
dom(){ python3 -W ignore run_domains.py --tag $1 --pretrain $2 --eval $3 --seed $4 --d-model $5 --pre-epochs ${6:-20} > logs/$1.txt 2>&1; }
case $1 in
 B1) dom dom3_d128_s42 FD001,MBA,FD003 FD001,MBA 42 128
     for s in 123 456; do dom dom2_d128_s$s FD001,MBA FD001,MBA $s 128; dom dom3_d128_s$s FD001,MBA,FD003 FD001,MBA $s 128; done
     full 128 123;;
 B2) for s in 42 123; do dom sep_FD001_d384_s$s FD001 FD001 $s 384; dom sep_MBA_d384_s$s MBA MBA $s 384; done;;
 B3) full 128 456
     dom shared5_d64_e40_s42 FD001,FD003,MBA,PSM,ETTm1 FD001,FD003,MBA,PSM,ETTm1 42 64 40
     dom shared5_d64_e40_s123 FD001,FD003,MBA,PSM,ETTm1 FD001,FD003,MBA,PSM,ETTm1 123 64 40;;
esac
