# TactiFoot Vision Architecture

The production package lives under `src/tactifoot_vision`.

Core Modules:

- `domain`: immutable frame, detection, track, projection, export and experiment data.
- `config`: Pydantic schemas shared by Python and YAML.
- `datasets`: SoccerNet/MOT parsing and COCO conversion.
- `detection`: YOLO/RF-DETR adapters behind detector/trainable interfaces.
- `tracking`: ByteTrack and SAM2 adapters behind the `Tracker` interface.
- `keypoints`: YOLO-pose keypoint detection.
- `projection`: pitch model, homography estimation and pitch-space projection.
- `team_assignment`: crops, embeddings, reducers, clustering and assignment.
- `experiments`: orchestration only; model logic stays in Modules.
- `evaluation`: MOT/SoccerNet metrics and StatsBomb360 projection-table helpers.
- `export`: stateless artifact writers.

Legacy code under `legacy/` is reference material only.

Known limits:

- BoTSORT is disabled until a stable adapter is selected.
- TrackEval is optional; internal MOT metrics are the default.
- Real model checks are opt-in pytest tests marked `model`, `sam2`, or `slow`.
- `team_assignment.crop_method=sam2_mask` is explicitly unsupported in the
  team-classification experiment runner until SAM2 cropper config is wired.
- Team-classification metrics are unsupervised unless tracks carry `team_label`
  or `team_id` metadata in `Track.data`.
- Native StatsBomb360 export is not implemented; current helpers expect
  normalized projection tables.
