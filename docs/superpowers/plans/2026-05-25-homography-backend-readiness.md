# Homography Backend Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce valid, comparable `homographies.parquet` artifacts for `tvcalib`, `sportlight`, `soccersegcal`, `pnlcalib`, and `auxflow` on SoccerNet-GSR `valid`, then rerun the existing homography comparison against `current_yolopose_7pt`.

**Architecture:** Keep external calibration repositories outside `pyproject.toml` and outside the production package dependency graph. Each backend runs in an isolated environment under `external/homography_backends/`, writes native outputs under `results/homography_backends/<method>/native/`, and then a thin exporter converts those outputs into the common schema consumed by `HomographyComparisonRunner`.

**Tech Stack:** Python 3.10/3.11 where upstream allows it, per-backend venv/conda envs, SoccerNet-GSR v1.3 at `data/SoccerNetGS/valid`, OpenCV, NumPy, pandas/pyarrow for artifact validation, existing `tactifoot_vision.export.homography.read_homographies`.

---

## Current State

Already available:

- Dataset: `data/SoccerNetGS/valid`, 58 sequences, 43,500 frames.
- Baseline result: `results/experiments/homography_comparison_valid_current_oracle`.
- Common artifact docs: `docs/homography_backends.md`.
- Importer/validator path: `tactifoot_vision.export.homography.read_homographies`.
- Current methods in ranking: `current_yolopose_7pt`, `oracle_gsr_lines_ransac`.

Missing:

- Native external backend runs.
- Per-backend converters into the common schema.
- Smoke artifacts for 3 frames.
- Full-valid artifacts for all five external candidates.
- Final comparison config referencing all artifacts.

Non-negotiable fairness rule:

- Candidate methods must not use GSR `bbox_pitch` GT or oracle homographies.
- Only `oracle_gsr_lines_ransac` may use GT pitch annotations, and it stays outside production ranking.

## File Structure

Create:

- `configs/experiments/homography_comparison_valid_all_backends.yaml`
  - Final ranking config referencing all imported external artifacts.
- `docs/homography_backends/tvcalib.md`
  - Environment, native command, output conversion notes.
- `docs/homography_backends/sportlight.md`
  - Environment, pretrained weight requirements, native command, conversion notes.
- `docs/homography_backends/soccersegcal.md`
  - Environment, pretrained line-segmentation/refinement requirements, conversion notes.
- `docs/homography_backends/pnlcalib.md`
  - Environment, SoccerNet-GSR/PnLCalib invocation, conversion notes.
- `docs/homography_backends/auxflow.md`
  - Anchor policy, optical-flow propagation command, conversion notes.
- `tools/homography_backends/README.md`
  - Shared operational instructions for external envs.
- `tools/homography_backends/make_smoke_subset.py`
  - Writes a small CSV with sequence/frame/image paths used by all backend smoke runs.
- `tools/homography_backends/validate_artifact.py`
  - Thin CLI wrapper around `read_homographies`.
- `tools/homography_backends/export_common.py`
  - Shared writer for JSONL/parquet common records, intentionally lightweight.
- `tools/homography_backends/convert_tvcalib.py`
- `tools/homography_backends/convert_sportlight.py`
- `tools/homography_backends/convert_soccersegcal.py`
- `tools/homography_backends/convert_pnlcalib.py`
- `tools/homography_backends/convert_auxflow.py`

Do not modify:

- `pyproject.toml` for external backend dependencies.
- Core production pipeline behavior.
- Existing baseline metrics except by rerunning final comparison into a new output directory.

## Common Artifact Contract

All converters must write:

```text
results/homography_backends/<method>/homographies.parquet
```

Required columns:

```text
sequence, frame, method, status, homography_3x3, runtime_ms, inliers, source_artifact, failure_reason
```

Accepted unavailable row example:

```json
{
  "sequence": "SNGS-021",
  "frame": 1,
  "method": "tvcalib",
  "status": "unavailable",
  "homography_3x3": null,
  "runtime_ms": null,
  "inliers": null,
  "source_artifact": "results/homography_backends/tvcalib/native/SNGS-021/000001.json",
  "failure_reason": "native calibration did not return a valid field-plane transform"
}
```

Accepted available row example:

```json
{
  "sequence": "SNGS-021",
  "frame": 1,
  "method": "pnlcalib",
  "status": "available",
  "homography_3x3": [[0.1, 0.0, -52.5], [0.0, -0.1, 34.0], [0.0, 0.0, 1.0]],
  "runtime_ms": 41.2,
  "inliers": 18,
  "source_artifact": "results/homography_backends/pnlcalib/native/SNGS-021/000001.json",
  "failure_reason": null
}
```

## Task 1: Shared Smoke Subset And Artifact Tooling

**Files:**

- Create: `tools/homography_backends/README.md`
- Create: `tools/homography_backends/make_smoke_subset.py`
- Create: `tools/homography_backends/export_common.py`
- Create: `tools/homography_backends/validate_artifact.py`
- Test manually against existing local data and a synthetic JSONL.

- [ ] **Step 1: Create shared tooling directory**

Run:

```bash
mkdir -p tools/homography_backends results/homography_backends/smoke
```

Expected:

```text
tools/homography_backends
results/homography_backends/smoke
```

- [ ] **Step 2: Add smoke subset generator**

Create `tools/homography_backends/make_smoke_subset.py`:

```python
from pathlib import Path

import pandas as pd


ROOT = Path("data/SoccerNetGS/valid")
OUTPUT = Path("results/homography_backends/smoke/frames.csv")
SEQUENCE = "SNGS-021"
FRAMES = (1, 250, 500)


def main() -> int:
    rows = []
    for frame in FRAMES:
        image_path = ROOT / SEQUENCE / "img1" / f"{frame:06d}.jpg"
        labels_path = ROOT / SEQUENCE / "Labels-GameState.json"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not labels_path.is_file():
            raise FileNotFoundError(labels_path)
        rows.append(
            {
                "sequence": SEQUENCE,
                "frame": frame,
                "image_path": str(image_path),
                "labels_path": str(labels_path),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run smoke subset generator**

Run:

```bash
uv run python tools/homography_backends/make_smoke_subset.py
cat results/homography_backends/smoke/frames.csv
```

Expected:

```text
sequence,frame,image_path,labels_path
SNGS-021,1,data/SoccerNetGS/valid/SNGS-021/img1/000001.jpg,...
SNGS-021,250,data/SoccerNetGS/valid/SNGS-021/img1/000250.jpg,...
SNGS-021,500,data/SoccerNetGS/valid/SNGS-021/img1/000500.jpg,...
```

- [ ] **Step 4: Add common writer**

Create `tools/homography_backends/export_common.py`:

```python
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COLUMNS = [
    "sequence",
    "frame",
    "method",
    "status",
    "homography_3x3",
    "runtime_ms",
    "inliers",
    "source_artifact",
    "failure_reason",
]


def validate_matrix(matrix: object) -> list[list[float]]:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"homography_3x3 must be 3x3, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("homography_3x3 contains NaN or Inf")
    return array.tolist()


def available_record(
    *,
    sequence: str,
    frame: int,
    method: str,
    homography_3x3: object,
    runtime_ms: float | None,
    inliers: int | None,
    source_artifact: str | None,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "frame": int(frame),
        "method": method,
        "status": "available",
        "homography_3x3": json.dumps(validate_matrix(homography_3x3)),
        "runtime_ms": runtime_ms,
        "inliers": inliers,
        "source_artifact": source_artifact,
        "failure_reason": None,
    }


def unavailable_record(
    *,
    sequence: str,
    frame: int,
    method: str,
    failure_reason: str,
    source_artifact: str | None,
    runtime_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "frame": int(frame),
        "method": method,
        "status": "unavailable",
        "homography_3x3": None,
        "runtime_ms": runtime_ms,
        "inliers": None,
        "source_artifact": source_artifact,
        "failure_reason": failure_reason,
    }


def write_parquet(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(output, index=False)
    return output
```

- [ ] **Step 5: Add artifact validator CLI**

Create `tools/homography_backends/validate_artifact.py`:

```python
import argparse
from pathlib import Path

from tactifoot_vision.export.homography import read_homographies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()

    records = read_homographies(Path(args.artifact))
    available = sum(record.is_available for record in records)
    unavailable = len(records) - available
    methods = sorted({record.method for record in records})
    print(
        {
            "records": len(records),
            "available": available,
            "unavailable": unavailable,
            "methods": methods,
        }
    )
    if not records:
        raise SystemExit("artifact has no records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Verify shared tooling with a synthetic artifact**

Run:

```bash
uv run python - <<'PY'
from tools.homography_backends.export_common import available_record, write_parquet

rows = [
    available_record(
        sequence="SNGS-021",
        frame=1,
        method="pnlcalib",
        homography_3x3=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        runtime_ms=1.0,
        inliers=4,
        source_artifact="synthetic",
    )
]
print(write_parquet(rows, "results/homography_backends/smoke/synthetic.parquet"))
PY
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/smoke/synthetic.parquet
```

Expected:

```text
{'records': 1, 'available': 1, 'unavailable': 0, 'methods': ['pnlcalib']}
```

## Task 2: Final Comparison Config

**Files:**

- Create: `configs/experiments/homography_comparison_valid_all_backends.yaml`

- [ ] **Step 1: Create final ranking config**

Create `configs/experiments/homography_comparison_valid_all_backends.yaml`:

```yaml
name: homography_comparison_valid_all_backends
kind: homography_comparison
soccernet_root: data/SoccerNetGS
output_dir: results/experiments/homography_comparison_valid_all_backends
pipeline:
  keypoints:
    backend: yolo_pose
    checkpoint: models/yolov8n-pose.pt
    confidence: 0.5
  projection:
    enabled: true
    min_keypoint_confidence: 0.5
    min_keypoints: 4
    smoothing_window: 1
    project_ball: true
homography_comparison:
  split: valid
  methods:
    - current_yolopose_7pt
    - oracle_gsr_lines_ransac
  external_homographies:
    - results/homography_backends/tvcalib/homographies.parquet
    - results/homography_backends/sportlight/homographies.parquet
    - results/homography_backends/soccersegcal/homographies.parquet
    - results/homography_backends/pnlcalib/homographies.parquet
    - results/homography_backends/auxflow/homographies.parquet
  confidence_iterations: 200
```

- [ ] **Step 2: Validate config loading before artifacts exist**

Run:

```bash
uv run python - <<'PY'
from tactifoot_vision.config import load_experiment_config

config = load_experiment_config(
    "configs/experiments/homography_comparison_valid_all_backends.yaml"
)
print(config.name)
print(config.homography_comparison.external_homographies)
PY
```

Expected:

```text
homography_comparison_valid_all_backends
(PosixPath('results/homography_backends/tvcalib/homographies.parquet'), ...)
```

## Task 3: PnLCalib Backend Artifact

**Files:**

- Create: `docs/homography_backends/pnlcalib.md`
- Create: `tools/homography_backends/convert_pnlcalib.py`
- External checkout: `external/homography_backends/PnLCalib`
- Output: `results/homography_backends/pnlcalib/homographies.parquet`

Rationale: implement first because PnLCalib is closest to the planned points-and-lines calibration target and has SoccerNet-GSR alignment in the dataset ecosystem.

- [ ] **Step 1: Clone and pin upstream**

Run:

```bash
mkdir -p external/homography_backends
git clone https://github.com/mguti97/PnLCalib external/homography_backends/PnLCalib
git -C external/homography_backends/PnLCalib rev-parse HEAD \
  > results/homography_backends/pnlcalib_upstream_sha.txt
```

Expected:

```text
results/homography_backends/pnlcalib_upstream_sha.txt
```

- [ ] **Step 2: Create `docs/homography_backends/pnlcalib.md`**

Write:

```markdown
# PnLCalib Backend

Repository: https://github.com/mguti97/PnLCalib

Pinned SHA:

```bash
cat results/homography_backends/pnlcalib_upstream_sha.txt
```

Inputs:

- SoccerNet-GSR images under `data/SoccerNetGS/valid/SNGS-*/img1`.
- No `bbox_pitch` GT is allowed for candidate calibration.

Outputs:

- Native outputs: `results/homography_backends/pnlcalib/native`.
- Common artifact: `results/homography_backends/pnlcalib/homographies.parquet`.

Smoke command:

```bash
uv run python tools/homography_backends/make_smoke_subset.py
# Run upstream PnLCalib on the three frames listed in results/homography_backends/smoke/frames.csv.
uv run python tools/homography_backends/convert_pnlcalib.py \
  --frames results/homography_backends/smoke/frames.csv \
  --native results/homography_backends/pnlcalib/native \
  --output results/homography_backends/pnlcalib/smoke.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/pnlcalib/smoke.parquet
```

Full command:

```bash
uv run python tools/homography_backends/convert_pnlcalib.py \
  --frames results/homography_backends/pnlcalib/full_frames.csv \
  --native results/homography_backends/pnlcalib/native \
  --output results/homography_backends/pnlcalib/homographies.parquet
```
```

- [ ] **Step 3: Implement converter skeleton**

Create `tools/homography_backends/convert_pnlcalib.py`:

```python
import argparse
import json
from pathlib import Path

import pandas as pd

from tools.homography_backends.export_common import (
    available_record,
    unavailable_record,
    write_parquet,
)


METHOD = "pnlcalib"


def native_output_path(native_root: Path, sequence: str, frame: int) -> Path:
    return native_root / sequence / f"{frame:06d}.json"


def read_native_homography(path: Path) -> tuple[object | None, int | None, float | None, str | None]:
    if not path.is_file():
        return None, None, None, "native output missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("homography_3x3") or payload.get("image_to_pitch_homography")
    if matrix is None:
        return None, None, payload.get("runtime_ms"), "native homography missing"
    return matrix, payload.get("inliers"), payload.get("runtime_ms"), None


def convert(frames_csv: Path, native_root: Path, output: Path) -> Path:
    rows = []
    frames = pd.read_csv(frames_csv)
    for item in frames.itertuples(index=False):
        sequence = str(item.sequence)
        frame = int(item.frame)
        native_path = native_output_path(native_root, sequence, frame)
        matrix, inliers, runtime_ms, failure = read_native_homography(native_path)
        if failure is not None:
            rows.append(
                unavailable_record(
                    sequence=sequence,
                    frame=frame,
                    method=METHOD,
                    failure_reason=failure,
                    source_artifact=str(native_path),
                    runtime_ms=runtime_ms,
                )
            )
            continue
        rows.append(
            available_record(
                sequence=sequence,
                frame=frame,
                method=METHOD,
                homography_3x3=matrix,
                runtime_ms=runtime_ms,
                inliers=None if inliers is None else int(inliers),
                source_artifact=str(native_path),
            )
        )
    return write_parquet(rows, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(convert(Path(args.frames), Path(args.native), Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Test converter missing-native behavior**

Run:

```bash
uv run python tools/homography_backends/make_smoke_subset.py
uv run python tools/homography_backends/convert_pnlcalib.py \
  --frames results/homography_backends/smoke/frames.csv \
  --native results/homography_backends/pnlcalib/native \
  --output results/homography_backends/pnlcalib/smoke_missing_native.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/pnlcalib/smoke_missing_native.parquet
```

Expected:

```text
{'records': 3, 'available': 0, 'unavailable': 3, 'methods': ['pnlcalib']}
```

- [ ] **Step 5: Run upstream PnLCalib smoke**

Use the upstream README to create its environment in `external/homography_backends/PnLCalib`. Run only the three smoke frames. The required deliverable is native JSON per frame:

```text
results/homography_backends/pnlcalib/native/SNGS-021/000001.json
results/homography_backends/pnlcalib/native/SNGS-021/000250.json
results/homography_backends/pnlcalib/native/SNGS-021/000500.json
```

Each JSON must contain either:

```json
{"homography_3x3": [[...], [...], [...]], "runtime_ms": 0.0, "inliers": 0}
```

or:

```json
{"image_to_pitch_homography": [[...], [...], [...]], "runtime_ms": 0.0, "inliers": 0}
```

- [ ] **Step 6: Convert and validate PnLCalib smoke**

Run:

```bash
uv run python tools/homography_backends/convert_pnlcalib.py \
  --frames results/homography_backends/smoke/frames.csv \
  --native results/homography_backends/pnlcalib/native \
  --output results/homography_backends/pnlcalib/smoke.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/pnlcalib/smoke.parquet
```

Expected:

```text
{'records': 3, 'available': 3, 'unavailable': 0, 'methods': ['pnlcalib']}
```

If availability is lower, inspect native logs before proceeding.

- [ ] **Step 7: Run PnLCalib full valid**

Create `results/homography_backends/pnlcalib/full_frames.csv` from all valid frames:

```bash
uv run python - <<'PY'
from pathlib import Path
import pandas as pd

rows = []
for seq in sorted(Path("data/SoccerNetGS/valid").glob("SNGS-*")):
    for image in sorted((seq / "img1").glob("*.jpg")):
        rows.append(
            {
                "sequence": seq.name,
                "frame": int(image.stem),
                "image_path": str(image),
                "labels_path": str(seq / "Labels-GameState.json"),
            }
        )
out = Path("results/homography_backends/pnlcalib/full_frames.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(out, len(rows))
PY
```

Run upstream PnLCalib over the listed frames and write native JSONs.

- [ ] **Step 8: Convert and validate PnLCalib full**

Run:

```bash
uv run python tools/homography_backends/convert_pnlcalib.py \
  --frames results/homography_backends/pnlcalib/full_frames.csv \
  --native results/homography_backends/pnlcalib/native \
  --output results/homography_backends/pnlcalib/homographies.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/pnlcalib/homographies.parquet
```

Expected:

```text
records == 43500
method == pnlcalib
```

## Task 4: TVCalib Backend Artifact

**Files:**

- Create: `docs/homography_backends/tvcalib.md`
- Create: `tools/homography_backends/convert_tvcalib.py`
- External checkout: `external/homography_backends/tvcalib`
- Output: `results/homography_backends/tvcalib/homographies.parquet`

- [ ] **Step 1: Clone and pin upstream**

Run:

```bash
mkdir -p external/homography_backends
git clone https://github.com/MM4SPA/tvcalib external/homography_backends/tvcalib
git -C external/homography_backends/tvcalib rev-parse HEAD \
  > results/homography_backends/tvcalib_upstream_sha.txt
```

- [ ] **Step 2: Document TVCalib constraints**

Create `docs/homography_backends/tvcalib.md`:

```markdown
# TVCalib Backend

Repository: https://github.com/MM4SPA/tvcalib

TVCalib may output camera parameters rather than a direct image-to-pitch
homography. The converter must produce the field-plane image-to-pitch transform
for `z = 0`.

Native outputs:

```text
results/homography_backends/tvcalib/native/<sequence>/<frame>.json
```

Accepted native JSON:

```json
{
  "homography_3x3": [[...], [...], [...]],
  "runtime_ms": 0.0,
  "inliers": 0
}
```

If upstream produces camera intrinsics/extrinsics only, first convert them to a
field-plane homography in the TVCalib environment and write the JSON above.
```

- [ ] **Step 3: Create converter by copying PnLCalib converter and changing method**

Run:

```bash
cp tools/homography_backends/convert_pnlcalib.py \
  tools/homography_backends/convert_tvcalib.py
python - <<'PY'
from pathlib import Path
path = Path("tools/homography_backends/convert_tvcalib.py")
text = path.read_text()
text = text.replace('METHOD = "pnlcalib"', 'METHOD = "tvcalib"')
text = text.replace('pnlcalib', 'tvcalib')
path.write_text(text)
PY
```

- [ ] **Step 4: Validate missing-native behavior**

Run:

```bash
uv run python tools/homography_backends/convert_tvcalib.py \
  --frames results/homography_backends/smoke/frames.csv \
  --native results/homography_backends/tvcalib/native \
  --output results/homography_backends/tvcalib/smoke_missing_native.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/tvcalib/smoke_missing_native.parquet
```

Expected:

```text
{'records': 3, 'available': 0, 'unavailable': 3, 'methods': ['tvcalib']}
```

- [ ] **Step 5: Run TVCalib smoke and full valid**

Use the same smoke/full sequence as Task 3. Native JSONs must match the accepted schema. Then run:

```bash
uv run python tools/homography_backends/convert_tvcalib.py \
  --frames results/homography_backends/pnlcalib/full_frames.csv \
  --native results/homography_backends/tvcalib/native \
  --output results/homography_backends/tvcalib/homographies.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/tvcalib/homographies.parquet
```

Expected:

```text
records == 43500
method == tvcalib
```

## Task 5: Sportlight Backend Artifact

**Files:**

- Create: `docs/homography_backends/sportlight.md`
- Create: `tools/homography_backends/convert_sportlight.py`
- External checkout: `external/homography_backends/soccernet-calibration-sportlight`
- Output: `results/homography_backends/sportlight/homographies.parquet`

- [ ] **Step 1: Clone and pin upstream**

Run:

```bash
mkdir -p external/homography_backends
git clone https://github.com/NikolasEnt/soccernet-calibration-sportlight \
  external/homography_backends/soccernet-calibration-sportlight
git -C external/homography_backends/soccernet-calibration-sportlight rev-parse HEAD \
  > results/homography_backends/sportlight_upstream_sha.txt
```

- [ ] **Step 2: Create documentation**

Create `docs/homography_backends/sportlight.md`:

```markdown
# Sportlight Backend

Repository: https://github.com/NikolasEnt/soccernet-calibration-sportlight

The backend must run without using GSR `bbox_pitch` GT. It may use its own
trained keypoint/line models and voter.

Native JSON target:

```json
{
  "homography_3x3": [[...], [...], [...]],
  "runtime_ms": 0.0,
  "inliers": 0,
  "voter_score": 0.0
}
```
```

- [ ] **Step 3: Create converter**

Run:

```bash
cp tools/homography_backends/convert_pnlcalib.py \
  tools/homography_backends/convert_sportlight.py
python - <<'PY'
from pathlib import Path
path = Path("tools/homography_backends/convert_sportlight.py")
text = path.read_text()
text = text.replace('METHOD = "pnlcalib"', 'METHOD = "sportlight"')
text = text.replace('pnlcalib', 'sportlight')
path.write_text(text)
PY
```

- [ ] **Step 4: Smoke, convert, validate**

Run upstream Sportlight on `results/homography_backends/smoke/frames.csv`, then:

```bash
uv run python tools/homography_backends/convert_sportlight.py \
  --frames results/homography_backends/smoke/frames.csv \
  --native results/homography_backends/sportlight/native \
  --output results/homography_backends/sportlight/smoke.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/sportlight/smoke.parquet
```

Expected:

```text
records == 3
method == sportlight
```

- [ ] **Step 5: Full valid**

Run upstream Sportlight on all frames in `results/homography_backends/pnlcalib/full_frames.csv`, then:

```bash
uv run python tools/homography_backends/convert_sportlight.py \
  --frames results/homography_backends/pnlcalib/full_frames.csv \
  --native results/homography_backends/sportlight/native \
  --output results/homography_backends/sportlight/homographies.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/sportlight/homographies.parquet
```

Expected:

```text
records == 43500
method == sportlight
```

## Task 6: SoccerSegCal Backend Artifact

**Files:**

- Create: `docs/homography_backends/soccersegcal.md`
- Create: `tools/homography_backends/convert_soccersegcal.py`
- External checkout: `external/homography_backends/soccersegcal`
- Output: `results/homography_backends/soccersegcal/homographies.parquet`

- [ ] **Step 1: Clone and pin upstream**

Run:

```bash
mkdir -p external/homography_backends
git clone https://github.com/Spiideo/soccersegcal external/homography_backends/soccersegcal
git -C external/homography_backends/soccersegcal rev-parse HEAD \
  > results/homography_backends/soccersegcal_upstream_sha.txt
```

- [ ] **Step 2: Create converter**

Run:

```bash
cp tools/homography_backends/convert_pnlcalib.py \
  tools/homography_backends/convert_soccersegcal.py
python - <<'PY'
from pathlib import Path
path = Path("tools/homography_backends/convert_soccersegcal.py")
text = path.read_text()
text = text.replace('METHOD = "pnlcalib"', 'METHOD = "soccersegcal"')
text = text.replace('pnlcalib', 'soccersegcal')
path.write_text(text)
PY
```

- [ ] **Step 3: Create documentation**

Create `docs/homography_backends/soccersegcal.md`:

```markdown
# SoccerSegCal Backend

Repository: https://github.com/Spiideo/soccersegcal

The backend should run line segmentation and differentiable-rendering
optimization in its own environment. Candidate ranking may use image pixels and
line detections, but not GSR `bbox_pitch` object GT.

Common artifact:

```text
results/homography_backends/soccersegcal/homographies.parquet
```
```

- [ ] **Step 4: Smoke and full valid**

Run upstream SoccerSegCal on the smoke frames, convert, validate:

```bash
uv run python tools/homography_backends/convert_soccersegcal.py \
  --frames results/homography_backends/smoke/frames.csv \
  --native results/homography_backends/soccersegcal/native \
  --output results/homography_backends/soccersegcal/smoke.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/soccersegcal/smoke.parquet
```

Then run full valid and validate:

```bash
uv run python tools/homography_backends/convert_soccersegcal.py \
  --frames results/homography_backends/pnlcalib/full_frames.csv \
  --native results/homography_backends/soccersegcal/native \
  --output results/homography_backends/soccersegcal/homographies.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/soccersegcal/homographies.parquet
```

Expected:

```text
records == 43500
method == soccersegcal
```

## Task 7: AuxFlow Backend Artifact

**Files:**

- Create: `docs/homography_backends/auxflow.md`
- Create: `tools/homography_backends/convert_auxflow.py`
- Output: `results/homography_backends/auxflow/homographies.parquet`

Rationale: AuxFlow needs anchor-frame homographies. For fair ranking, anchors must come from the best non-oracle backend after Tasks 3-6. Do not use `oracle_gsr_lines_ransac` anchors in the production ranking.

- [ ] **Step 1: Create converter**

Run:

```bash
cp tools/homography_backends/convert_pnlcalib.py \
  tools/homography_backends/convert_auxflow.py
python - <<'PY'
from pathlib import Path
path = Path("tools/homography_backends/convert_auxflow.py")
text = path.read_text()
text = text.replace('METHOD = "pnlcalib"', 'METHOD = "auxflow"')
text = text.replace('pnlcalib', 'auxflow')
path.write_text(text)
PY
```

- [ ] **Step 2: Create documentation**

Create `docs/homography_backends/auxflow.md`:

```markdown
# AuxFlow Backend

AuxFlow propagates homographies temporally from anchor frames.

Fair anchor policy:

- Choose anchors from the best non-oracle single-frame backend on the validation
  smoke run.
- Do not use `oracle_gsr_lines_ransac` anchors in the production comparison.
- Record the anchor source in native JSON under `anchor_method`.

Native JSON:

```json
{
  "homography_3x3": [[...], [...], [...]],
  "runtime_ms": 0.0,
  "inliers": 0,
  "anchor_method": "pnlcalib",
  "anchor_frame": 250
}
```
```

- [ ] **Step 3: Select anchor backend**

After Tasks 3-6 smoke runs, inspect smoke metrics:

```bash
uv run tactifoot experiment homography-comparison \
  --config configs/experiments/homography_comparison_valid_all_backends.yaml
```

If this fails because full artifacts are not ready, create a temporary smoke config with only smoke artifacts. Select the best non-oracle backend using:

1. Higher `availability`.
2. Lower `median_error_m`.
3. Lower `p90_error_m`.

Write chosen method:

```bash
echo pnlcalib > results/homography_backends/auxflow/anchor_method.txt
```

Replace `pnlcalib` with the actual selected non-oracle method.

- [ ] **Step 4: Run AuxFlow smoke and full valid**

Run AuxFlow in its isolated environment using anchor homographies from:

```text
results/homography_backends/<anchor_method>/homographies.parquet
```

Write native JSONs under:

```text
results/homography_backends/auxflow/native/<sequence>/<frame>.json
```

Convert and validate:

```bash
uv run python tools/homography_backends/convert_auxflow.py \
  --frames results/homography_backends/pnlcalib/full_frames.csv \
  --native results/homography_backends/auxflow/native \
  --output results/homography_backends/auxflow/homographies.parquet
uv run python tools/homography_backends/validate_artifact.py \
  results/homography_backends/auxflow/homographies.parquet
```

Expected:

```text
records == 43500
method == auxflow
```

## Task 8: Full Comparison And Report Refresh

**Files:**

- Modify: `docs/homography_comparison_experiment_results.md`
- Read: `results/experiments/homography_comparison_valid_all_backends/metrics.json`
- Read: `results/experiments/homography_comparison_valid_all_backends/ranking.csv`

- [ ] **Step 1: Validate every external artifact exists**

Run:

```bash
for method in tvcalib sportlight soccersegcal pnlcalib auxflow; do
  uv run python tools/homography_backends/validate_artifact.py \
    "results/homography_backends/${method}/homographies.parquet"
done
```

Expected:

```text
records == 43500
```

for each method.

- [ ] **Step 2: Run full comparison**

Run:

```bash
uv run tactifoot experiment homography-comparison \
  --config configs/experiments/homography_comparison_valid_all_backends.yaml
```

Expected files:

```text
results/experiments/homography_comparison_valid_all_backends/homographies.parquet
results/experiments/homography_comparison_valid_all_backends/projections.parquet
results/experiments/homography_comparison_valid_all_backends/metrics.json
results/experiments/homography_comparison_valid_all_backends/ranking.csv
results/experiments/homography_comparison_valid_all_backends/report.md
```

- [ ] **Step 3: Inspect ranking**

Run:

```bash
cat results/experiments/homography_comparison_valid_all_backends/ranking.csv
```

Expected:

```text
method,score,median_error_m,p90_error_m,success@2m,availability,temporal_jitter,rankable
...
```

with rows for:

```text
auxflow
current_yolopose_7pt
oracle_gsr_lines_ransac
pnlcalib
soccersegcal
sportlight
tvcalib
```

- [ ] **Step 4: Refresh Markdown report**

Update `docs/homography_comparison_experiment_results.md` with:

- Dataset summary unchanged.
- Ranking table from `ranking.csv`.
- Per-method metrics from `metrics.json`.
- External backend status changed from "Not evaluated" to available/unavailable percentages.
- Recommendation section naming the best production candidate, excluding `oracle_gsr_lines_ransac`.
- Residual risks: external weights, GPU runtime, per-sequence failure clusters.

- [ ] **Step 5: Generate comparison videos for top methods**

Generate videos for:

```text
current_yolopose_7pt vs best_production_method
best_production_method vs oracle_gsr_lines_ransac
```

Use the existing layout and keep the readability mirror consistent:

```text
pitch_y -> -pitch_y
```

Output:

```text
results/experiments/homography_comparison_valid_all_backends/comparison_videos/
```

## Task 9: Review And Quality Gates

**Files:**

- Review all files created under `tools/homography_backends/`, `configs/experiments/`, `docs/homography_backends/`, and the final report.

- [ ] **Step 1: Run package quality gates**

Run:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/tactifoot_vision
uv run pytest -m "not slow and not model"
```

Expected:

```text
All checks passed
Success: no issues found
tests pass
```

- [ ] **Step 2: Run tool syntax checks**

Run:

```bash
uv run python -m py_compile tools/homography_backends/*.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run artifact consistency check**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
from tactifoot_vision.export.homography import read_homographies

for method in ["tvcalib", "sportlight", "soccersegcal", "pnlcalib", "auxflow"]:
    path = Path(f"results/homography_backends/{method}/homographies.parquet")
    records = read_homographies(path)
    assert len(records) == 43500, (method, len(records))
    assert {record.method for record in records} == {method}
    print(method, "ok")
PY
```

Expected:

```text
tvcalib ok
sportlight ok
soccersegcal ok
pnlcalib ok
auxflow ok
```

- [ ] **Step 4: Code review**

Use the requested review workflow:

```bash
bash ~/.codex/skills/python-review/scripts/collect-review-context.sh --working-tree yes
```

Review focus:

- Converter correctness.
- Homography direction is image-to-pitch, not pitch-to-image.
- No GT `bbox_pitch` leakage in candidate backends.
- `auxflow` anchor method is non-oracle.
- All artifacts have exactly 43,500 rows.

## Acceptance Criteria

The work is complete when:

- `results/homography_backends/tvcalib/homographies.parquet` exists and validates.
- `results/homography_backends/sportlight/homographies.parquet` exists and validates.
- `results/homography_backends/soccersegcal/homographies.parquet` exists and validates.
- `results/homography_backends/pnlcalib/homographies.parquet` exists and validates.
- `results/homography_backends/auxflow/homographies.parquet` exists and validates.
- `results/experiments/homography_comparison_valid_all_backends/ranking.csv` contains all five external methods plus current baseline and oracle.
- `docs/homography_comparison_experiment_results.md` names the best production candidate excluding oracle.
- Quality gates pass.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Upstream repo expects SoccerNet Calibration instead of GSR layout | Build a temporary input symlink/copy layout under `results/homography_backends/<method>/input` without modifying `data/SoccerNetGS`. |
| Upstream outputs camera parameters, not homography | Convert camera model to field-plane image-to-pitch homography inside native environment before writing common JSON. |
| Pretrained weights are missing | Record blocker in the backend doc, do not fabricate results, and continue with the remaining backends. |
| GPU runtime too high for full valid | Run per-backend full valid sequence batches and append native JSONs; convert only after all batches complete. |
| Homography direction mismatch | Validate by projecting GSR image footpoints and checking median error on smoke frames before running full valid. |
| AuxFlow uses oracle anchors by accident | Enforce `anchor_method != oracle_gsr_lines_ransac` in `docs/homography_backends/auxflow.md` and in the native run metadata. |

## Execution Order

1. Task 1: shared tooling.
2. Task 2: final comparison config.
3. Task 3: PnLCalib.
4. Task 4: TVCalib.
5. Task 5: Sportlight.
6. Task 6: SoccerSegCal.
7. Task 7: AuxFlow after at least one non-oracle anchor backend is ready.
8. Task 8: final comparison and report.
9. Task 9: review and quality gates.

