# Homography Comparison - Hard Numbers

Date: 2026-05-25  
Dataset: SoccerNet-GSR `gamestate-2024` `valid`  
Sequences: 58  
Frames: 43,500

## What Was Run

Command:

```bash
uv run tactifoot experiment homography-comparison \
  --config configs/experiments/homography_comparison_valid_current_oracle.yaml
```

Output:

```text
results/experiments/homography_comparison_valid_current_oracle
```

This run compares the two methods that currently have executable/local
homography sources:

- `current_yolopose_7pt`
- `oracle_gsr_lines_ransac`

The remaining planned methods do not yet have local common-format artifacts, so
they are not assigned numerical quality scores in this report.

## Primary Metrics

| Method | Rankable | Availability | Median Error m | Mean Error m | P90 Error m | Success@1m | Success@2m | Success@5m | LocSim tau5 | Temporal Jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_gsr_lines_ransac` | yes | 98.10% | 0.0963 | 2.1837 | 0.3003 | 98.51% | 99.35% | 99.69% | 0.9889 | 0.1073 |
| `current_yolopose_7pt` | yes | 23.27% | 93.7011 | 211.3479 | 179.5593 | 0.0039% | 0.0411% | 0.2479% | 0.0008 | 0.2873 |

Ranking score:

| Rank | Method | Score |
| ---: | --- | ---: |
| 1 | `oracle_gsr_lines_ransac` | 0.003820 |
| 2 | `current_yolopose_7pt` | 0.965031 |

## Confidence Intervals

Bootstrap 95% CI for median error, resampled by sequence:

| Method | Median Error Low m | Median Error High m |
| --- | ---: | ---: |
| `oracle_gsr_lines_ransac` | 0.0917 | 0.1020 |
| `current_yolopose_7pt` | 84.9495 | 100.8870 |

## Frame And Projection Counts

| Method | Expected Frames | Available Frames | Unavailable Frames | Projection Rows |
| --- | ---: | ---: | ---: | ---: |
| `oracle_gsr_lines_ransac` | 43,500 | 42,675 | 825 | 737,286 |
| `current_yolopose_7pt` | 43,500 | 10,123 | 33,377 | 155,731 |

`current_yolopose_7pt` has zero available homographies on 8 of 58 sequences.
Its best sequence-level availability is `SNGS-085` at 79.07%; its worst
nonzero sequence-level availability is `SNGS-053` at 0.13%.

## Error Distribution

| Method | P95 Error m | P99 Error m | Max Error m |
| --- | ---: | ---: | ---: |
| `oracle_gsr_lines_ransac` | 0.4527 | 1.3366 | 216,971.0450 |
| `current_yolopose_7pt` | 275.6624 | 1,222.9930 | 1,508,131.0268 |

The oracle max error is an extreme outlier; its p99 remains 1.34 m. This is why
median and p90 are better primary indicators than max error for this dataset.

## Direct Delta

| Comparison | Value |
| --- | ---: |
| Current median error / oracle median error | 973.43x |
| Current p90 error / oracle p90 error | 597.95x |
| Current mean error / oracle mean error | 96.78x |
| Oracle availability minus current availability | +74.83 percentage points |
| Oracle Success@2m minus current Success@2m | +99.31 percentage points |
| Oracle Success@5m minus current Success@5m | +99.44 percentage points |

## External Method Data Availability

| Method | Artifact Path | Status | Hard Numbers Available |
| --- | --- | --- | --- |
| `tvcalib` | `results/homography_backends/tvcalib/homographies.parquet` | missing | no |
| `sportlight` | `results/homography_backends/sportlight/homographies.parquet` | missing | no |
| `soccersegcal` | `results/homography_backends/soccersegcal/homographies.parquet` | missing | no |
| `pnlcalib` | `results/homography_backends/pnlcalib/homographies.parquet` | missing | no |
| `auxflow` | `results/homography_backends/auxflow/homographies.parquet` | missing | no |

These methods should not be interpreted as scoring `0`; they were not measured
because they did not produce the required common artifact yet.

## Bottom Line

The hard numeric conclusion from the completed experiment is:

- `current_yolopose_7pt` is not viable: low availability and near-zero
  `success@2m`.
- `oracle_gsr_lines_ransac` confirms the evaluation path can produce sub-meter
  results when correspondences are valid, but it is not a production method.
- The real production comparison is blocked on generating external artifacts for
  `pnlcalib`, `sportlight`, `soccersegcal`, `tvcalib`, and `auxflow`.

