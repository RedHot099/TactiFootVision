#!/usr/bin/env bash
# The `tactifoot` command: the same package calls as examples 03-05, from the shell.
# Run from the repository root: `bash examples/06_cli.sh`
set -euo pipefail

OUT=outputs/examples/06_cli
VIDEO=data/videos/broadcast_60s.mp4

uv run tactifoot info

# Inference: a run file says how to process video; run inputs are flags.
# --set overrides one run file value for this run (the value is YAML).
uv run tactifoot run configs/pipeline.yaml --video "$VIDEO" --output-dir "$OUT/run" \
    --start 250 --max-frames 50 \
    --set teams=null --set render.annotator.style=video_game --set render.radar=null
ls "$OUT/run"   # result.pkl tracks.csv freeze_frames.csv annotated.mp4

# Training: TrainConfig fields are flags (see `tactifoot train --help`);
# backend options go through --set.
uv run tactifoot train yolo --data data/datasets/football_yolo_sample100 \
    --weights yolo11n.pt --epochs 1 --imgsz 320 --batch-size 16 \
    --output-dir "$OUT/runs" --name yolo11n --exist-ok --set mosaic=0.0

# Evaluation with backend-independent metrics; --set goes to model.evaluate.
uv run tactifoot evaluate yolo --weights models/football_yolo11m.pt \
    --data data/datasets/football_yolo --split valid --set max_images=20
