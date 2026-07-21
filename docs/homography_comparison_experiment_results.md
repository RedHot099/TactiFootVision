# Homography Backend Comparison - SoccerNet-GSR Valid

Date: 2026-05-24  
Branch: `codex/homography-comparison`

## Summary

I downloaded the official SoccerNet-GSR `gamestate-2024` valid split from the
official SoccerNet repository workflow and ran the homography comparison
experiment on the full local `valid` split.

The local experiment ranks only the two methods that can be executed from this
repository today:

1. `current_yolopose_7pt`: the current baseline through `PitchProjector` using
   `models/yolov8n-pose.pt`.
2. `oracle_gsr_lines_ransac`: diagnostic oracle control estimated from GSR GT
   image/pitch footpoint correspondences. This is not a production candidate.

No results are reported for `tvcalib`, `sportlight`, `soccersegcal`,
`pnlcalib`, or `auxflow`, because this repository intentionally does not vendor
those external environments and no common-format homography artifacts were
present locally.

## Data

Official source:

- SoccerNet GSR task: <https://www.soccer-net.org/tasks/game-state-reconstruction>
- SoccerNet GSR repository: <https://github.com/SoccerNet/sn-gamestate>

Download command used:

```bash
uv run python - <<'PY'
from SoccerNet.Downloader import SoccerNetDownloader

loader = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
loader.downloadDataTask(task="gamestate-2024", split=["valid"])
PY
```

The downloader wrote:

```text
data/SoccerNetGS/gamestate-2024/valid.zip
```

I unpacked it to:

```text
data/SoccerNetGS/valid/SNGS-*/Labels-GameState.json
data/SoccerNetGS/valid/SNGS-*/img1/*.jpg
```

Dataset sanity check:

| Item | Value |
| --- | ---: |
| Split | `valid` |
| Sequences | 58 |
| Frames | 43,500 |
| Label version | `1.3` |
| Homography records written | 87,000 |
| Projection records written | 893,017 |

## Run

Config:

```text
configs/experiments/homography_comparison_valid_current_oracle.yaml
```

Command:

```bash
uv run tactifoot experiment homography-comparison \
  --config configs/experiments/homography_comparison_valid_current_oracle.yaml
```

Artifacts:

```text
results/experiments/homography_comparison_valid_current_oracle/homographies.parquet
results/experiments/homography_comparison_valid_current_oracle/projections.parquet
results/experiments/homography_comparison_valid_current_oracle/metrics.json
results/experiments/homography_comparison_valid_current_oracle/ranking.csv
results/experiments/homography_comparison_valid_current_oracle/report.md
results/experiments/homography_comparison_valid_current_oracle/failure_cases/
results/experiments/homography_comparison_valid_current_oracle/comparison_videos/
```

## Ranking

Ranking score uses the planned normalized weighted formula over median error,
p90 error, `success@2m`, availability, and temporal jitter.

| Rank | Method | Score | Availability | Median Error m | Mean Error m | P90 Error m | Success@1m | Success@2m | Success@5m | LocSim tau5 | Temporal Jitter |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `oracle_gsr_lines_ransac` | 0.003820 | 0.9810 | 0.0963 | 2.1837 | 0.3003 | 0.9851 | 0.9935 | 0.9969 | 0.9889 | 0.1073 |
| 2 | `current_yolopose_7pt` | 0.965031 | 0.2327 | 93.7011 | 211.3479 | 179.5593 | 0.000039 | 0.000411 | 0.002479 | 0.000815 | 0.2873 |

Bootstrap 95% CI for median error, resampled by sequence:

| Method | Median Error Low | Median Error High |
| --- | ---: | ---: |
| `oracle_gsr_lines_ransac` | 0.0917 | 0.1020 |
| `current_yolopose_7pt` | 84.9495 | 100.8870 |

Availability details:

| Method | Available Frames | Unavailable Frames | Expected Frames |
| --- | ---: | ---: | ---: |
| `oracle_gsr_lines_ransac` | 42,675 | 825 | 43,500 |
| `current_yolopose_7pt` | 10,123 | 33,377 | 43,500 |

Projection counts:

| Method | Projection Rows |
| --- | ---: |
| `oracle_gsr_lines_ransac` | 737,286 |
| `current_yolopose_7pt` | 155,731 |

Failure gallery:

```text
results/experiments/homography_comparison_valid_current_oracle/failure_cases/README.md
results/experiments/homography_comparison_valid_current_oracle/failure_cases/failure_cases.csv
results/experiments/homography_comparison_valid_current_oracle/failure_cases/*.png
```

The gallery contains the top 25 `current_yolopose_7pt` frames by per-frame
`p90_error_m`, with image footpoints, a pitch GT-vs-prediction minimap, and a
compact table of the largest object-level errors.

Video comparison:

```text
results/experiments/homography_comparison_valid_current_oracle/comparison_videos/SNGS-021_current_yolopose_7pt_vs_oracle_gsr_lines_ransac_mirrored_y.mp4
```

This is a 30-second, 25 FPS comparison over all 750 frames of `SNGS-021`.
The pitch minimaps are mirrored along the other pitch axis for readability only:
`pitch_y -> -pitch_y`.

## External Backend Status

| Backend | Status In This Run | Reason |
| --- | --- | --- |
| `tvcalib` | Not evaluated | No `homographies.parquet`/JSONL artifact present |
| `sportlight` | Not evaluated | No `homographies.parquet`/JSONL artifact present |
| `soccersegcal` | Not evaluated | No `homographies.parquet`/JSONL artifact present |
| `pnlcalib` | Not evaluated | No `homographies.parquet`/JSONL artifact present |
| `auxflow` | Not evaluated | No `homographies.parquet`/JSONL artifact present |

This is intentional for the current package design: external repositories should
run in isolated environments and export the common schema documented in
`docs/homography_backends.md`. The production repository should import those
artifacts, not vendor all external calibration stacks into `pyproject.toml`.

## Interpretation

`current_yolopose_7pt` is not viable as a SoccerNet-GSR homography backend. It
finds an available homography on only 23.3% of frames, and even those frames have
median projection error of 93.7 m. The near-zero `success@2m` confirms that the
current baseline is measuring a known modeling error: human YOLO-pose keypoints
are being interpreted as pitch keypoints.

`oracle_gsr_lines_ransac` validates the evaluation harness and data flow. It
achieves 9.6 cm median error and 99.35% `success@2m`, which shows that the GSR
footpoint-to-pitch evaluation path is working. Its higher mean error compared
with p90 indicates a small number of outliers. Because it estimates homography
from GT image/pitch player correspondences and evaluates on the same annotation
family, it must remain a diagnostic upper-bound control, not a ranking candidate.

## Recommendation

Do not integrate `current_yolopose_7pt` into production homography. It should
remain only as a historical baseline.

The next production decision needs imported artifacts for at least `pnlcalib`,
`auxflow`, and one line-segmentation method (`sportlight` or `soccersegcal`) on
the same `valid` split. Prioritize `pnlcalib` and `auxflow` first: they match the
planned evaluation goal most directly, because PnLCalib optimizes points/lines
and AuxFlow targets temporal propagation on SoccerNet-GSR-like data.

Until those artifacts are generated, the defensible recommendation is:

1. Replace the current YOLO-pose baseline as soon as an external calibration
   artifact beats the current baseline with nonzero availability.
2. Use `oracle_gsr_lines_ransac` only as a regression guard for parser, metrics,
   and projection math.
3. Treat any final backend choice as blocked until the five external methods
   export the common homography schema for `valid`.

## Reproduction Notes

The complete local ranking source is:

```text
results/experiments/homography_comparison_valid_current_oracle/ranking.csv
```

The smaller smoke runs used before the full run are available under:

```text
results/experiments/homography_comparison_valid_local
results/experiments/homography_comparison_valid_subset
```
