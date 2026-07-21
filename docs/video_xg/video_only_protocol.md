# Video-Only xG Protocol

## Goal

The video-only xG path predicts shot xG from features extracted from match video.
StatsBomb data is allowed only as an evaluation reference, never as model input.

## Allowed Input Features

- `shot_id`: stable identifier produced by the video pipeline.
- `frame_index`: detected shot/contact frame.
- `shot_x`, `shot_y`: ball/shot location projected from video to pitch coordinates.
- `goal_x`, `goal_y`: attacking goal center in the same pitch coordinate system.
- `nearest_player_distance`: nearest non-ball track distance from video projection.
- `goalkeeper_distance`: goalkeeper distance from video projection.
- `defender_count_in_cone`: blockers between the shot and goal.
- `ball_speed`: pre-shot ball speed estimated from reconstructed video trajectory.
- `shot_confidence`: confidence of the video shot detector.

## Forbidden Model Inputs

The model input table must not contain StatsBomb or label columns such as
`location`, `shot_freeze_frame`, `shot_statsbomb_xg`, `statsbomb_xg`,
`shot_outcome`, `is_goal`, `shot_body_part`, `shot_type`, `shot_technique`,
`under_pressure`, `shot_first_time`, or `shot_one_on_one`.

The runtime enforces this with `assert_video_only_columns`.

## Reference Data

Reference data must be passed as a separate CSV with:

```text
shot_id,reference_xg,is_goal
```

`reference_xg` can be StatsBomb xG. `is_goal` is optional and is used only for
Brier, log loss, and calibration diagnostics.

## CLI

Run one model:

```bash
uv run tactifoot video-xg from-features \
  --features path/to/video_features.csv \
  --reference path/to/reference.csv \
  --output-dir results/video_only_xg \
  --model video_freeze_context
```

Compare the default video-only methods:

```bash
uv run tactifoot video-xg compare-methods \
  --features path/to/video_features.csv \
  --reference path/to/reference.csv \
  --output-dir results/video_only_xg
```

Outputs:

- `video_only_shots.csv`
- `video_only_summary.json`
- `method_metrics.csv`
- `comparison_report.md`

## Notebook

Use `notebooks/video_only_xg_experiment.ipynb` to run the experiment from a
prepared feature table or to generate `video_features.csv` from the CV pipeline.
