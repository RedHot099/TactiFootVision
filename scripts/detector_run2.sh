#!/usr/bin/env bash
# detector-data run 2 (ticket 04): stage 2a (football_yolo) then, if it ended with a best.pt, stage 2b (SoccerSynth).
# Launch detached from the worktree (see runs/run2a/RUN.md):
#   setsid nohup bash scripts/detector_run2.sh > /mnt/storage/tactifoot_research/detector_data/runs/run2a/logs/chain.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=/mnt/storage/tactifoot_research/detector_data/runs
for stage in run2a run2b; do
  mkdir -p "$RUNS/$stage/logs"
  echo "$(date -Is) $stage start"
  .venv/bin/python scripts/detector_train.py train --run "$stage" > "$RUNS/$stage/logs/train.log" 2>&1 < /dev/null
  test -f "$RUNS/$stage/yolo11l_1280/weights/best.pt"
  echo "$(date -Is) $stage done"
done
