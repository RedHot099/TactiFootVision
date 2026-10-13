"""Pitch keypoints from SoccerNet broadcast frames (pitch-keypoints phase).

- `fit`: frame -> pitch homography fitted to SoccerNet's annotated pitch lines; the 32 `SoccerPitch` vertices.
- `sources`: GSR 2024 clips (through `gsr_eval.read_clip`) and Camera Calibration 2023 images as `Frame`s.
- `dataset`: per-image record, the rejection rule, the YOLO-pose label line, overlays (ticket 01).
- `exclusions`: split hygiene of the dataset: xG held-out games, GSR evaluation games, duplicates (council C3).
- `gsr_eval`: homography evaluation in metres on GSR (ticket 02).
"""
