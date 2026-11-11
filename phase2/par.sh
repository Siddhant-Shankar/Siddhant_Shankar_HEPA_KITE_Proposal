#!/usr/bin/env bash
# Run the commands in file $1 (one per line), $2 at a time, each single-threaded. Call via run_queued.sh.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/phase2
xargs -a "$1" -d '\n' -P "${2:-2}" -I{} bash -c '{}'
