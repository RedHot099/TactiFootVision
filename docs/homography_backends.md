# Homography Backend Artifacts

External calibration repositories stay outside `pyproject.toml`. Run each backend in
its own environment and export one artifact with the common schema consumed by
`HomographyComparisonRunner`.

Required `homographies.parquet` or JSONL columns:

| Column | Description |
| --- | --- |
| `sequence` | SoccerNet-GSR sequence name, for example `SNGS-001` |
| `frame` | Integer frame number matching `Labels-GameState.json` |
| `method` | One of `tvcalib`, `sportlight`, `soccersegcal`, `pnlcalib`, `auxflow`, `current_yolopose_7pt`, `oracle_gsr_lines_ransac` |
| `status` | `available` or `unavailable` |
| `homography_3x3` | Image-to-pitch 3x3 matrix, nested list or JSON string |
| `runtime_ms` | Optional runtime per frame |
| `inliers` | Optional correspondence inlier count |
| `source_artifact` | Optional backend-native output path |
| `failure_reason` | Required when unavailable |

Recommended smoke protocol:

```bash
tactifoot experiment homography-comparison --config configs/experiments/homography_comparison_smoke.yaml
```

For backend smoke tests, export 3 frames first, validate with:

```python
from tactifoot_vision.export.homography import read_homographies

records = read_homographies("path/to/homographies.parquet")
assert records
```

Backend notes:

- `tvcalib`: run TrackLab/SoccerNet calibration in its Python 3.9 environment and convert camera parameters or homographies to image-to-pitch matrices.
- `sportlight`: export the voted calibration result per frame to the common schema.
- `soccersegcal`: export optimizer homographies after line segmentation and rendering refinement.
- `pnlcalib`: use the SoccerNet-GSR PnLCalib integration output; keep refinement metadata in `source_artifact`.
- `auxflow`: export propagated anchor-frame homographies per frame; mark frames without valid propagation as `unavailable`.
