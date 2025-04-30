# TactiFoot Vision

**TactiFoot Vision** is a comprehensive Python toolkit for soccer match video analysis. It provides high-performance detection, keypoint estimation, homography mapping, player & ball tracking, visualization, and data export—all driven by a single, Pydantic-validated YAML configuration.

---

## 🌟 Features

- **Object Detection**: YOLO & RF-DETR handlers for player, ball, referee, goalkeeper.
- **Keypoint Estimation**: YOLO-Pose for pitch landmark detection and homography.
- **Homography & Mapping**: Smooth RANSAC homographies and transform frame ↔ pitch coordinates.
- **Tracking**: ByteTrack-based multi-object tracker and raw ball path collection with outlier filtering.
- **Visualization**: OpenCV `PitchVisualizer` with overlay support; Matplotlib utilities for charts.
- **Data Export**: Per-frame CSV of freeze frames, homography matrices, timestamps, and more.
- **Config-Driven**: Single `default_config.yaml` governs all components with sensible defaults.
- **Scripts**: Ready-to-use scripts for detection and StatsBomb merging.

---

## 🚀 Quickstart

1. **Clone & Install**
   ```bash
   git clone https://github.com/yourorg/tactifoot_vision.git
   cd tactifoot_vision
   # Install dependencies and environment
   poetry install
   ```

2. **Prepare Configuration**
   Copy and edit `config/default_config.yaml`:

   ```yaml
   paths:
     input_video:            /path/to/video.mp4
     output_video:           /path/to/output.mp4
     model_dir:              /path/to/models
     statsbomb_input_csv:    data/statsbomb.csv
     pipeline_input_csv:     data/pipeline.csv
     merged_output_csv:      data/merged.csv

   detection:
     model_type: yolo
     # …other sections (keypoints, tracking, geometry, visualization, processing, training)
   ```

3. **Run Detection & Tracking**
   ```bash
   poetry run python scripts/run_detection.py --config config/default_config.yaml
   ```

4. **Merge with StatsBomb**
   ```bash
   poetry run python scripts/merge_pipeline_statsbomb.py --config config/default_config.yaml
   ```

## 🛠 Configuration

All options live in `config/default_config.yaml`.

Pydantic enforces types, ranges, and resolves relative paths.

Change only the settings you need—everything else uses production-grade defaults.

## 📈 Mini Roadmap

- Add support for additional tracking backends (e.g., DeepSORT, OC-SORT)
- Dockerize the entire pipeline for zero-install, reproducible deployments
- Integrate alternative detection models (e.g., Detectron2, MMDetection)
- Add built-in benchmarking & metric reporting (FPS, mAP, tracking metrics)
