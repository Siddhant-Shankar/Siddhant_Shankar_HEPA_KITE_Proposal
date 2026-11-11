#!/usr/bin/env bash
cd /home/claude/scout/scaling
for b in B1 B2 B3; do bash /home/claude/scout/run_queued.sh scaling-$b ./batch.sh $b; done
echo ALLDONE > logs/driver_done
