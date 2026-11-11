#!/usr/bin/env bash
# usage: drive.sh <jobfile> ; splits into chunks of N lines, each chunk is one queued job (2 procs in parallel)
cd /home/claude/phase2
f="$1"; n="${2:-4}"
split -l "$n" -d "$f" "$f.chunk"
for c in "$f".chunk*; do
  bash /home/claude/scout/run_queued.sh "p2-$(basename $c)" bash par.sh "$c" 2
done
echo "DRIVE DONE $f $(date +%H:%M:%S)"
