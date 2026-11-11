#!/usr/bin/env bash
# Serialize CPU-heavy experiments across agents: only one holder of the lock trains at a time.
# Usage: bash /home/claude/scout/run_queued.sh <tag> <command...>
tag="$1"; shift
echo "$(date +%H:%M:%S) WAIT  $tag" >> /home/claude/scout/queue.log
exec 9>/home/claude/scout/.cpu.lock
flock 9
echo "$(date +%H:%M:%S) START $tag" >> /home/claude/scout/queue.log
export HEPA_DATA_DIR=/home/claude/data OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$@"; rc=$?
echo "$(date +%H:%M:%S) END   $tag rc=$rc" >> /home/claude/scout/queue.log
exit $rc
