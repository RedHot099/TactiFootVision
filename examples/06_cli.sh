#!/usr/bin/env bash
# The `tactifoot` command: the same package calls as examples 03-05, from the shell.
# Run from the repository root: `bash examples/06_cli.sh`
set -euo pipefail

OUT=outputs/examples/06_cli
VIDEO=data/videos/broadcast_60s.mp4

uv run tactifoot info

# Inference: a run file says how to process video; run inputs are flags.
# `tactifoot init` writes a commented run file to start from (its weights are
# bare names in the working directory; here they live in models/, and the
# pitch model gets its own settings: a square input and its homography ones).
# --set overrides one run file value for this run (the value is YAML).
# In Python: tv.run_file.load(path, overrides).run(video, output_dir, start=..., end=...).
# render.annotator.trail_seconds draws the ball's last second in the video,
# render.radar.trail_seconds each person's last two seconds on the radar;
# with teams=null every tail is in the default colour. render.radar.draw_visible_area
# outlines the part of the pitch the camera shows, and render.radar.velocity_seconds
# draws each person's velocity as an arrow (the run computes the velocities).
mkdir -p "$OUT" && rm -f "$OUT/pipeline.yaml"
uv run tactifoot init "$OUT/pipeline.yaml"
uv run tactifoot run "$OUT/pipeline.yaml" --video "$VIDEO" --output-dir "$OUT/run" \
    --set detector.weights=models/detector_yolo11l.pt \
    --set keypoints.weights=models/pitch_yolo11m_pose.pt \
    --set keypoints.conf=0.1 --set keypoints.resize=square \
    --set 'homography={min_confidence: 0.3, ransac_threshold: 2.0, refine_with_lines: true}' \
    --start 250 --end 300 \
    --set teams=null --set render.annotator.style=video_game \
    --set render.annotator.trail_seconds=1.0 --set render.radar.trail_seconds=2.0 \
    --set render.radar.draw_visible_area=true --set render.radar.velocity_seconds=1.0
ls "$OUT/run"   # result.pkl tracks.csv freeze_frames.csv carriers.csv passes.csv annotated.mp4

# Training: TrainConfig fields are flags (see `tactifoot train --help`);
# backend options go through --set. A small sample of the football dataset
# keeps this quick; there is no CLI command for subsets, so it is one Python call.
uv run python -c "import tactifoot_vision as tv; \
tv.load_dataset('data/datasets/football_yolo').subset({'train': 100, 'valid': 20}).to_yolo('$OUT/football_sample')"
uv run tactifoot train yolo --data "$OUT/football_sample/data.yaml" \
    --weights yolo11n.pt --epochs 1 --imgsz 320 --batch-size 16 \
    --output-dir "$OUT/runs" --name yolo11n --exist-ok --set mosaic=0.0

# Evaluation with backend-independent metrics; --set goes to model.evaluate,
# --set model.KEY=VALUE to load_model (model.device=cuda:1, model.imgsz=1280, ...).
uv run tactifoot evaluate yolo --weights models/detector_yolo11l.pt \
    --data data/datasets/football_yolo --split valid --set max_images=20 \
    --set model.imgsz=640
