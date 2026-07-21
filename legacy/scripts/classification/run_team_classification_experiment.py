#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import logging
import warnings
import os
import sys
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import threading

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import umap
import torch
from tqdm.auto import tqdm
import yaml

warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state",
    module="umap.umap_",
)

_STRUCTURED_PIPELINE_SEEN: Dict[Path, set[str]] = {}
_STRUCTURED_PIPELINE_CLEARED: set[Path] = set()
_STRUCTURED_PREDICTIONS_CLEARED: set[Path] = set()


def log_banner(title: str) -> None:
    logging.info("=" * 72)
    logging.info(title)
    logging.info("=" * 72)


def progress(iterable, desc: str) -> Iterable:
    return tqdm(iterable, desc=desc, leave=False)

# Ensure project root is on sys.path so `tactifoot_vision` imports work when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tactifoot_vision.team import SiglipTeamClassifier, ResnetTeamClassifier


SAM2_REPO_ROOT = PROJECT_ROOT / "external" / "segment-anything-2-real-time"
DEFAULT_SAM2_CONFIG = SAM2_REPO_ROOT / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"

CROP_METHOD_CENTER = "center"
CROP_METHOD_SAM2 = "sam2_mask"
CROP_METHOD_OPENCV = "opencv_mask"

STAGE_RATIO_SWEEP = "ratio_sweep"
STAGE_SAM2_COMPARISON = "sam2_comparison"
STAGE_OPENCV_COMPARISON = "opencv_comparison"
STAGE_COLOR_UMAP = "umap_color"

EMBED_BACKEND_SIGLIP = "siglip"
EMBED_BACKEND_CLIP = "clip"
EMBED_BACKEND_RESNET = "resnet"
EMBED_BACKEND_RESNET16 = "resnet16"


@dataclass
class SequenceMetadata:
    name: str
    frame_rate: float
    image_dir: Path
    image_ext: str
    num_frames: Optional[int] = None


@dataclass
class CropInfo:
    frame_index: int
    player_id: str
    team_label: int
    crop_center_ratio: float
    saved_path: Optional[Path] = None
    crop_method: str = CROP_METHOD_CENTER
    flow_mean: Optional[float] = None
    flow_std: Optional[float] = None


class FlowComputer:
    """Helper that keeps state between consecutive frames for dense optical flow."""

    def __init__(self, args: Optional[argparse.Namespace], enabled: bool) -> None:
        self.args = args
        self.enabled = enabled
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_index: Optional[int] = None

    def compute(self, frame_gray: np.ndarray, frame_index: int) -> Optional[np.ndarray]:
        if not self.enabled:
            self._prev_gray = frame_gray
            self._prev_index = int(frame_index)
            return None
        flow: Optional[np.ndarray] = None
        if (
            self._prev_gray is not None
            and self._prev_index is not None
            and int(frame_index) - int(self._prev_index) == 1
        ):
            flow = compute_dense_flow(self._prev_gray, frame_gray, self.args)
        self._prev_gray = frame_gray
        self._prev_index = int(frame_index)
        return flow


class Cropper:
    """Encapsulates the different cropping strategies and fallbacks."""

    def __init__(
        self,
        crop_center_ratio: float,
        crop_method: str,
        sam2_cropper: Optional[Sam2Cropper],
        cv2_mask_iter: int,
        cv2_mask_margin_ratio: float,
    ) -> None:
        self.crop_center_ratio = float(max(0.05, min(1.0, crop_center_ratio)))
        self.crop_method = crop_method
        self.sam2_cropper = sam2_cropper
        self.cv2_mask_iter = int(cv2_mask_iter)
        self.cv2_mask_margin_ratio = float(cv2_mask_margin_ratio)

    def prepare_frame(self, frame: np.ndarray) -> None:
        if self.crop_method == CROP_METHOD_SAM2 and self.sam2_cropper is not None:
            self.sam2_cropper.prepare_frame(frame)

    def _center_crop(
        self, frame: np.ndarray, box: tuple[int, int, int, int], width: int, height: int
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        if self.crop_center_ratio < 1.0:
            x1, y1, x2, y2 = center_shrink_box(
                x1, y1, x2, y2, width, height, self.crop_center_ratio
            )
        return frame[y1:y2, x1:x2]

    def crop(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[Optional[np.ndarray], str]:
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        crop_method_used = self.crop_method

        if self.crop_method == CROP_METHOD_SAM2 and self.sam2_cropper is not None:
            crop = self.sam2_cropper.extract_crop(frame, (x1, y1, x2, y2))
            if crop is not None:
                return crop, crop_method_used
            crop_method_used = CROP_METHOD_CENTER

        if self.crop_method == CROP_METHOD_OPENCV:
            crop = apply_opencv_mask(
                frame,
                (x1, y1, x2, y2),
                iter_count=self.cv2_mask_iter,
                margin_ratio=self.cv2_mask_margin_ratio,
            )
            if crop is not None:
                return crop, crop_method_used
            crop_method_used = CROP_METHOD_CENTER

        crop = self._center_crop(frame, (x1, y1, x2, y2), width, height)
        if crop.size == 0:
            return None, crop_method_used
        return crop, crop_method_used


class Sam2Cropper:
    """Thin wrapper around SAM2 image predictor for per-frame masks."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        *,
        device: Optional[str] = None,
        multimask_output: bool = False,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.config_path.is_file() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM2 config ({self.config_path}) or checkpoint ({self.checkpoint_path}) not found"
            )
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.multimask_output = multimask_output
        self._predictor = self._init_predictor()

    def _add_repo_to_path(self) -> Optional[Path]:
        repo_root = None
        for parent in self.config_path.parents:
            if parent.name == "segment-anything-2-real-time":
                repo_root = parent
                break
        if repo_root and str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        return repo_root

    def _init_predictor(self):
        repo_root = self._add_repo_to_path()
        try:
            from hydra.core.global_hydra import GlobalHydra  # type: ignore
            from hydra import initialize_config_dir  # type: ignore
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "SAM2 dependencies are missing. Ensure hydra-core and the SAM2 repo are installed."
            ) from exc

        config_dir = self.config_path.parent
        for parent in self.config_path.parents:
            if parent.name == "configs":
                config_dir = parent
                break
        config_name = self.config_path.stem
        if repo_root:
            configs_root = repo_root / "sam2" / "configs"
            if configs_root.is_dir():
                try:
                    rel = self.config_path.relative_to(configs_root)
                    config_name = str(rel.with_suffix("")).replace("\\", "/")
                except ValueError:
                    config_name = self.config_path.stem
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(config_dir), job_name="sam2_cropper")
        model = build_sam2(
            config_file=config_name,
            ckpt_path=str(self.checkpoint_path),
            device=self.device,
        )
        return SAM2ImagePredictor(model)

    def prepare_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

    def extract_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None
        try:
            masks, _, _ = self._predictor.predict(
                box=np.array([[float(x1), float(y1), float(x2), float(y2)]], dtype=np.float32),
                multimask_output=self.multimask_output,
                normalize_coords=False,
            )
        except Exception as exc:
            logging.exception("SAM2 mask prediction failed: %s", exc)
            return None
        if masks.size == 0:
            return None
        mask = masks[0]
        if mask.ndim == 3:
            mask = mask[0]
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return None
        ys, xs = np.where(mask_bool)
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        top = max(0, top)
        left = max(0, left)
        bottom = min(frame.shape[0], bottom)
        right = min(frame.shape[1], right)
        if bottom - top <= 1 or right - left <= 1:
            return None
        crop = frame[top:bottom, left:right].copy()
        mask_roi = mask_bool[top:bottom, left:right]
        crop[~mask_roi] = 0
        return crop

    def extract_crops(self, frame: np.ndarray, bboxes: np.ndarray) -> List[Optional[np.ndarray]]:
        """Batch variant of extract_crop for multiple boxes in the current frame."""
        if bboxes.size == 0:
            return []
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError("bboxes must have shape (N, 4)")
        try:
            masks, _, _ = self._predictor.predict(
                box=bboxes.astype(np.float32),
                multimask_output=self.multimask_output,
                normalize_coords=False,
            )
        except Exception as exc:
            logging.exception("SAM2 batch mask prediction failed: %s", exc)
            return [None for _ in range(int(bboxes.shape[0]))]
        if masks is None or getattr(masks, "size", 0) == 0:
            return [None for _ in range(int(bboxes.shape[0]))]
        # Expected: (N, H, W) or (N, 1, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        outputs: List[Optional[np.ndarray]] = []
        for idx in range(int(bboxes.shape[0])):
            try:
                mask = masks[idx]
            except Exception:
                outputs.append(None)
                continue
            if mask is None:
                outputs.append(None)
                continue
            mask_bool = np.asarray(mask).astype(bool)
            if mask_bool.ndim != 2 or not mask_bool.any():
                outputs.append(None)
                continue
            ys, xs = np.where(mask_bool)
            top, bottom = int(ys.min()), int(ys.max()) + 1
            left, right = int(xs.min()), int(xs.max()) + 1
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            if bottom - top <= 1 or right - left <= 1:
                outputs.append(None)
                continue
            crop = frame[top:bottom, left:right].copy()
            mask_roi = mask_bool[top:bottom, left:right]
            crop[~mask_roi] = 0
            outputs.append(crop)
        return outputs


def apply_opencv_mask(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    iter_count: int = 5,
    margin_ratio: float = 0.15,
) -> Optional[np.ndarray]:
    """Segment a player crop with GrabCut guided by the detection bounding box."""
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return None
    span = max(x2 - x1, y2 - y1)
    margin = int(round(span * max(0.0, margin_ratio)))
    rx1 = max(0, x1 - margin)
    ry1 = max(0, y1 - margin)
    rx2 = min(width, x2 + margin)
    ry2 = min(height, y2 + margin)
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1
    if roi_w <= 1 or roi_h <= 1:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    # Run GrabCut on the local ROI to avoid allocating full-frame masks for every box.
    local_rect = (max(0, x1 - rx1), max(0, y1 - ry1), max(1, x2 - x1), max(1, y2 - y1))
    if local_rect[2] <= 1 or local_rect[3] <= 1:
        return None
    mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, local_rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    lx1 = max(0, x1 - rx1)
    ly1 = max(0, y1 - ry1)
    lx2 = min(roi.shape[1], x2 - rx1)
    ly2 = min(roi.shape[0], y2 - ry1)
    crop_mask = fg_mask[ly1:ly2, lx1:lx2]
    crop = roi[ly1:ly2, lx1:lx2]
    if crop.size == 0 or crop_mask.size == 0:
        return None
    if int(np.count_nonzero(crop_mask)) == 0:
        return None
    return crop * crop_mask[..., None]


def _load_config_defaults(config_path: Optional[Path], parser: argparse.ArgumentParser) -> Dict[str, Any]:
    if not config_path:
        return {}
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a top-level mapping of argument names to values.")
    valid_keys = {action.dest for action in parser._actions if action.dest and action.dest != "help"}
    defaults: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in valid_keys:
            logging.warning("Ignoring unknown config key: %s", key)
            continue
        defaults[key] = value
    # preserve the config path in parsed args
    defaults.setdefault("config", config_path)
    return defaults


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    # First, parse only the config path to allow overriding defaults from YAML.
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML file with argument defaults. CLI flags override values from the file.",
    )
    config_args, _ = prelim.parse_known_args(list(argv))

    parser = argparse.ArgumentParser(
        description="Run team classification experiments over SoccerTrack sequences."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML file with argument defaults. CLI flags override values from the file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=False,
        help="Root directory of the SoccerTrack Wide-View dataset (contains sequence folders).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/team_classification/numeric/team_classification_metrics.csv"),
        help=(
            "Path to write the metrics CSV "
            "(default: results/team_classification/numeric/team_classification_metrics.csv)."
        ),
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=None,
        help=(
            "Path to write aggregated metrics in Parquet format. "
            "If omitted, defaults to the same name as --output-csv with '.parquet' suffix."
        ),
    )
    parser.add_argument(
        "--structured-results-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for structured per-pipeline artefacts, e.g. "
            "'results/team_classification/raw/structured'. When set, the script saves intermediate results into "
            "results/team_classification/raw/structured/<color>/crop_<...>/<embedding>/umap_<k>/<cluster>/..."
        ),
    )
    parser.add_argument(
        "--structured-write-pipeline-metrics",
        action="store_true",
        help="Write per-pipeline metrics.csv inside each structured pipeline directory.",
    )
    parser.add_argument(
        "--structured-write-pipeline-predictions",
        action="store_true",
        help="Write per-pipeline predictions.csv inside each structured pipeline directory (can be large).",
    )
    parser.add_argument(
        "--structured-force",
        action="store_true",
        help="Recompute structured artefacts even when cached files already exist.",
    )
    parser.add_argument(
        "--structured-cache-only",
        action="store_true",
        help=(
            "When --structured-results-root is set, load cached per-sequence artefacts (.npz/.npy) and "
            "skip dataset crop extraction. Useful for resuming clustering/UMAP sweeps without re-reading frames."
        ),
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=2.0,
        help="Temporal stride in seconds between sampled frames (default: 2.0).",
    )
    parser.add_argument(
        "--umap-min",
        type=int,
        default=3,
        help="Minimum number of UMAP components to evaluate (inclusive).",
    )
    parser.add_argument(
        "--umap-max",
        type=int,
        default=128,
        help="Maximum number of UMAP components to evaluate (inclusive, default: 128).",
    )
    parser.add_argument(
        "--umap-step",
        type=int,
        default=1,
        help="Step size for sweeping UMAP components (default: 1).",
    )
    parser.add_argument(
        "--color-spaces",
        type=str,
        default="rgb,h",
        help="Comma-separated list of color spaces to test (supported: rgb,h,hsv).",
    )
    parser.add_argument(
        "--player-class-ids",
        type=str,
        default="1,2",
        help="Comma-separated list of class IDs that correspond to players in the annotations.",
    )
    parser.add_argument(
        "--team-column-name",
        type=str,
        default=None,
        help="Name of the column that stores team IDs (if omitted, falls back to index lookup).",
    )
    parser.add_argument(
        "--team-column-index",
        type=int,
        default=7,
        help="Zero-based column index to use for team labels when no column name is provided.",
    )
    parser.add_argument(
        "--mot-team-source",
        type=str,
        choices=["gt_column", "gameinfo"],
        default="gt_column",
        help="How to derive team labels for MOT sequences (gt column vs SoccerNet gameinfo.ini mapping).",
    )
    parser.add_argument(
        "--class-column-name",
        type=str,
        default="class_id",
        help="Name of the column used for filtering player detections (default: class_id).",
    )
    parser.add_argument(
        "--class-column-index",
        type=int,
        default=7,
        help="Zero-based column index for player class filtering fallback (default: 7).",
    )
    parser.add_argument(
        "--team-label-map",
        type=str,
        default=None,
        help=(
            "Optional mapping from raw team values to canonical IDs, e.g. '1:0,2:1' to map labels."
        ),
    )
    parser.add_argument(
        "--csv-team-ids",
        type=str,
        default="0,1",
        help=(
            "Comma-separated team IDs to include when parsing SoccerTrack CSV files (default: 0,1)."
        ),
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=0,
        help="Maximum number of recordings to process (0 or negative means all, default: 0).",
    )
    parser.add_argument(
        "--save-crops-dir",
        type=Path,
        default=Path("components"),
        help=(
            "Directory where extracted crops are stored as <dir>/<game_id>/<team>/... (default: components)."
        ),
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="Disable saving player crops to disk.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/team_classification/numeric/team_classification_predictions.csv"),
        help=(
            "Path to write detailed per-crop predictions "
            "(default: results/team_classification/numeric/team_classification_predictions.csv)."
        ),
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=None,
        help="Optional cap on sampled frames per sequence (0 or negative means no cap).",
    )
    parser.add_argument(
        "--crop-center-ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help=(
            "Comma-separated list of center crop ratios (0,1] evaluated during the initial sweep."
        ),
    )
    default_sam2_config = DEFAULT_SAM2_CONFIG if DEFAULT_SAM2_CONFIG.is_file() else None
    default_sam2_ckpt = DEFAULT_SAM2_CHECKPOINT if DEFAULT_SAM2_CHECKPOINT.is_file() else None
    parser.add_argument(
        "--sam2-config-path",
        type=Path,
        default=default_sam2_config,
        help="Path to SAM2 config YAML used for SAM2 mask crops (default: sam2.1_hiera_s if available).",
    )
    parser.add_argument(
        "--sam2-checkpoint-path",
        type=Path,
        default=default_sam2_ckpt,
        help="Path to SAM2 checkpoint (default: sam2.1_hiera_small if available).",
    )
    parser.add_argument(
        "--sam2-device",
        type=str,
        default=None,
        help="Override device for SAM2 inference (default: auto CUDA/CPU).",
    )
    parser.add_argument(
        "--skip-sam2",
        action="store_true",
        help="Disable SAM2 cropping stage even if checkpoints are available.",
    )
    parser.add_argument(
        "--skip-opencv-mask",
        action="store_true",
        help="Disable OpenCV GrabCut mask comparison stage.",
    )
    parser.add_argument(
        "--cv2-mask-iter",
        type=int,
        default=5,
        help="GrabCut iterations for OpenCV masking (default: 5).",
    )
    parser.add_argument(
        "--cv2-mask-margin-ratio",
        type=float,
        default=0.15,
        help="Extra margin (fraction of box size) around detections for GrabCut.",
    )
    parser.add_argument(
        "--use-optical-flow-features",
        action="store_true",
        help="Append optical flow magnitude statistics (mean/std) to feature vectors.",
    )
    parser.add_argument(
        "--flow-pyr-scale",
        type=float,
        default=0.5,
        help="Farneback pyr_scale parameter.",
    )
    parser.add_argument(
        "--flow-levels",
        type=int,
        default=3,
        help="Farneback pyramid levels.",
    )
    parser.add_argument(
        "--flow-winsize",
        type=int,
        default=15,
        help="Farneback window size.",
    )
    parser.add_argument(
        "--flow-iterations",
        type=int,
        default=3,
        help="Farneback iterations per level.",
    )
    parser.add_argument(
        "--flow-poly-n",
        type=int,
        default=5,
        help="Farneback poly_n parameter.",
    )
    parser.add_argument(
        "--flow-poly-sigma",
        type=float,
        default=1.2,
        help="Farneback poly_sigma parameter.",
    )
    parser.add_argument(
        "--kmeans-init",
        type=int,
        default=10,
        help="Number of KMeans initializations (n_init).",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        choices=["kmeans", "dbscan", "cmeans"],
        default="kmeans",
        help="Clustering algorithm to use for team assignment.",
    )
    parser.add_argument(
        "--cluster-methods",
        type=str,
        default=None,
        help="Optional comma-separated list of clustering methods to sweep (overrides --cluster-method).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon (distance threshold).",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples per cluster.",
    )
    parser.add_argument(
        "--cmeans-m",
        type=float,
        default=2.0,
        help="Fuzzy c-means fuzziness parameter m (≥1.0).",
    )
    parser.add_argument(
        "--cmeans-max-iter",
        type=int,
        default=50,
        help="Maximum iterations for fuzzy c-means.",
    )
    parser.add_argument(
        "--cmeans-tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for fuzzy c-means.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (passed through SigLIP config).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP embeddings (passed through SigLIP config).",
    )
    parser.add_argument(
        "--umap-metric",
        type=str,
        default="euclidean",
        help="UMAP distance metric (passed through SigLIP config).",
    )
    parser.add_argument(
        "--umap-random-state",
        type=str,
        default="0",
        help="UMAP random_state (int) or 'none' to allow parallel UMAP (default: 0).",
    )
    parser.add_argument(
        "--umap-n-jobs",
        type=int,
        default=-1,
        help="UMAP n_jobs (ignored when random_state is set) (default: -1).",
    )
    parser.add_argument(
        "--umap-components-strategy",
        type=str,
        choices=["fit_each", "fit_max_slice"],
        default="fit_each",
        help="How to evaluate multiple UMAP component counts: fit each separately (exact) or fit once at max and slice prefixes (fast).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for clustering tasks (default: min(8, available CPUs)).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        choices=[EMBED_BACKEND_SIGLIP, EMBED_BACKEND_CLIP, EMBED_BACKEND_RESNET, EMBED_BACKEND_RESNET16],
        default="siglip",
        help="Which embedding backend to use for clustering.",
    )
    parser.add_argument(
        "--embedding-backends",
        type=str,
        default=None,
        help="Optional comma-separated list of embedding backends to sweep (overrides --embedding-backend).",
    )
    parser.add_argument(
        "--include-no-umap",
        action="store_true",
        help="Include a no-UMAP baseline in the UMAP grid stage.",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Run only the unified grid stage (skip legacy stages 1-3).",
    )
    parser.add_argument(
        "--color-hist-bins",
        type=int,
        default=32,
        help="Number of histogram bins to append as color features (0 disables).",
    )
    parser.add_argument(
        "--color-hist-weight",
        type=float,
        default=1.0,
        help="Scale factor applied to color histogram features.",
    )
    parser.add_argument(
        "--resnet-batch-size",
        type=int,
        default=64,
        help="Batch size for ResNet embedding extraction (reduce if you hit CUDA OOM).",
    )
    # Apply defaults from config file, then parse full arguments so CLI overrides YAML.
    defaults = _load_config_defaults(config_args.config, parser)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )
    # Silence overly verbose external libraries (SAM2 logs per-frame at INFO).
    for name in ("sam2", "sam2.sam2_image_predictor", "hydra"):
        logging.getLogger(name).setLevel(logging.WARNING)
    class _DropNoisySam2Logs(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return True
            if (
                "For numpy array image, we assume" in msg
                or "Computing image embeddings for the provided image" in msg
                or "Image embeddings computed." in msg
            ):
                return False
            return True

    root = logging.getLogger()
    for handler in list(getattr(root, "handlers", [])):
        handler.addFilter(_DropNoisySam2Logs())


def read_sequence_metadata(seq_dir: Path) -> SequenceMetadata:
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    if "Sequence" not in parser:
        raise ValueError(f"seqinfo.ini at {seqinfo_path} lacks [Sequence] section")
    section = parser["Sequence"]
    frame_rate = float(section.get("frameRate", 25))
    image_dir_name = section.get("imDir", "img1")
    image_dir = (seq_dir / image_dir_name).resolve()
    image_ext = section.get("imExt", ".jpg")
    seq_length = section.getint("seqLength", fallback=None)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return SequenceMetadata(
        name=seq_dir.name,
        frame_rate=frame_rate,
        image_dir=image_dir,
        image_ext=image_ext,
        num_frames=seq_length,
    )


def parse_soccernet_gameinfo_team_map(seq_dir: Path) -> Dict[int, int]:
    """Return mapping track_id -> team_label (0 left, 1 right) from SoccerNet gameinfo.ini."""
    gameinfo_path = seq_dir / "gameinfo.ini"
    if not gameinfo_path.is_file():
        return {}
    parser = configparser.ConfigParser()
    try:
        parser.read(gameinfo_path)
    except configparser.Error:
        return {}
    if "Sequence" not in parser:
        return {}
    section = parser["Sequence"]
    num = section.getint("num_tracklets", fallback=0)
    mapping: Dict[int, int] = {}
    for idx in range(1, max(0, int(num)) + 1):
        raw = section.get(f"trackletID_{idx}", fallback="")
        if not raw:
            continue
        raw_lower = raw.lower().strip()
        # "player team left;10", "goalkeepers team left;y", etc.
        if not (raw_lower.startswith("player ") or raw_lower.startswith("goalkeeper")):
            continue
        if "team left" in raw_lower:
            mapping[idx] = 0
        elif "team right" in raw_lower:
            mapping[idx] = 1
    return mapping


def load_ground_truth(seq_dir: Path) -> pd.DataFrame:
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    df = pd.read_csv(gt_path, header=None)
    column_names = [
        "frame",
        "track_id",
        "x",
        "y",
        "w",
        "h",
        "confidence",
    ]
    remaining = df.shape[1] - len(column_names)
    extra_names: List[str] = []
    if remaining > 0:
        extra_names.append("class_id")
        remaining -= 1
    if remaining > 0:
        extra_names.append("visibility")
        remaining -= 1
    for idx in range(remaining):
        extra_names.append(f"extra_{idx}")
    df.columns = column_names + extra_names
    return df


def resolve_column_name(
    df: pd.DataFrame, preferred_name: Optional[str], fallback_index: int
) -> str:
    if preferred_name and preferred_name in df.columns:
        return preferred_name
    if fallback_index < 0 or fallback_index >= df.shape[1]:
        raise ValueError(
            f"Fallback index {fallback_index} is out of bounds for dataframe with {df.shape[1]} columns."
        )
    return df.columns[fallback_index]


def parse_id_list(raw: str) -> List[int]:
    if isinstance(raw, (list, tuple)):
        return [int(item) for item in raw]
    raw_items = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw_items]


def parse_float_list(raw: str) -> List[float]:
    if isinstance(raw, (list, tuple)):
        return [float(item) for item in raw if str(item).strip() != ""]
    items: List[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(float(item))
        except ValueError:
            continue
    return items


def parse_mapping(raw: Optional[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not raw:
        return mapping
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                mapping[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return mapping
    for item in str(raw).split(","):
        if ":" not in item:
            continue
        key_str, value_str = item.split(":", 1)
        try:
            key = int(key_str.strip())
            value = int(value_str.strip())
        except ValueError:
            continue
        mapping[key] = value
    return mapping


def parse_color_spaces(raw: object) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    return [space.strip().lower() for space in str(raw).split(",") if space.strip()]


def clamp_box(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> Optional[tuple[int, int, int, int]]:
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width, int(np.ceil(x + w)))
    y2 = min(height, int(np.ceil(y + h)))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return x1, y1, x2, y2


def center_shrink_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    if ratio >= 1.0:
        return x1, y1, x2, y2
    ratio = max(0.05, float(ratio))
    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return x1, y1, x2, y2
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    half_w = new_w / 2.0
    half_h = new_h / 2.0
    new_x1 = max(0, int(round(cx - half_w)))
    new_y1 = max(0, int(round(cy - half_h)))
    new_x2 = min(width, int(round(cx + half_w)))
    new_y2 = min(height, int(round(cy + half_h)))
    if new_x2 - new_x1 <= 1 or new_y2 - new_y1 <= 1:
        return x1, y1, x2, y2
    return new_x1, new_y1, new_x2, new_y2


def format_ratio_identifier(ratio: float) -> str:
    formatted = f"{ratio:.3f}".rstrip("0").rstrip(".")
    formatted = formatted.replace(".", "p")
    if not formatted:
        formatted = "0"
    return formatted


def format_ratio_folder(ratio: float) -> str:
    formatted = f"{float(ratio):.3f}".rstrip("0")
    if formatted.endswith("."):
        formatted += "0"
    return formatted or "0"


def sanitize_identifier(value: str) -> str:
    safe = str(value).strip().lower()
    safe = safe.replace(os.sep, "_").replace("/", "_")
    safe = re.sub(r"[^a-z0-9._-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or "unknown"


def structured_crop_dir(crop_method: str, crop_center_ratio: float) -> str:
    method = str(crop_method).strip().lower()
    if method == CROP_METHOD_CENTER:
        return f"crop_{format_ratio_folder(crop_center_ratio)}"
    if method in {CROP_METHOD_OPENCV, CROP_METHOD_SAM2}:
        return method
    return method or "crop_unknown"


def structured_umap_dir(umap_applied: bool, umap_components: int) -> str:
    if not bool(umap_applied):
        return "umap_none"
    return f"umap_{int(umap_components)}"


def structured_cluster_dir(cluster_method: str) -> str:
    method = str(cluster_method).strip().lower()
    if method in {"kmeans", "k_means"}:
        return "k_means"
    return sanitize_identifier(method)


def resolve_structured_embedding_dir(
    structured_root: Path,
    *,
    color_space: str,
    crop_method: str,
    crop_center_ratio: float,
    embedding_backend: str,
) -> Path:
    return (
        structured_root
        / sanitize_identifier(color_space)
        / structured_crop_dir(crop_method, crop_center_ratio)
        / sanitize_identifier(embedding_backend)
    )


def resolve_structured_umap_dir_path(
    embedding_dir: Path, *, umap_applied: bool, umap_components: int
) -> Path:
    return embedding_dir / structured_umap_dir(bool(umap_applied), int(umap_components))


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.tmp{path.suffix}"
    np.save(str(tmp), array)
    tmp.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.tmp{path.suffix}"
    np.savez_compressed(str(tmp), **arrays)
    tmp.replace(path)


def load_feature_cache(
    feature_path: Path,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not feature_path.is_file():
        return None
    try:
        with np.load(str(feature_path)) as loaded:
            features = np.asarray(loaded["features"], dtype=np.float32)
            labels = np.asarray(loaded["labels"], dtype=int)
            frame_index = np.asarray(
                loaded.get("frame_index", np.array([], dtype=int)),
                dtype=int,
            )
            player_id = np.asarray(
                loaded.get("player_id", np.array([], dtype=str)),
                dtype=str,
            )
            return features, labels, frame_index, player_id
    except Exception:
        return None


def _get_or_load_seen_sequences(metrics_path: Path) -> set[str]:
    seen = _STRUCTURED_PIPELINE_SEEN.get(metrics_path)
    if seen is not None:
        return seen
    existing: set[str] = set()
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        try:
            df = pd.read_csv(metrics_path, usecols=["sequence"])
            existing = set(str(val) for val in df["sequence"].dropna().astype(str).tolist())
        except Exception:
            existing = set()
    _STRUCTURED_PIPELINE_SEEN[metrics_path] = existing
    return existing


def append_rows_dedup_by_sequence(
    path: Path, rows: Sequence[Dict[str, object]], *, force: bool = False
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path not in _STRUCTURED_PIPELINE_CLEARED:
        if path.is_file():
            try:
                path.unlink()
            except Exception:
                pass
        _STRUCTURED_PIPELINE_SEEN.pop(path, None)
        _STRUCTURED_PIPELINE_CLEARED.add(path)
    seen = _get_or_load_seen_sequences(path)
    new_rows = []
    for row in rows:
        seq = str(row.get("sequence", "")).strip()
        if not seq:
            continue
        if seq in seen:
            continue
        new_rows.append(row)
        seen.add(seq)
    if not new_rows:
        return
    write_header = not path.is_file() or path.stat().st_size == 0
    pd.DataFrame(new_rows).to_csv(
        path,
        index=False,
        mode="a" if not write_header else "w",
        header=write_header,
    )


def load_single_row_by_sequence(csv_path: Path, sequence_name: str) -> Optional[Dict[str, object]]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "sequence" not in df.columns or df.empty:
        return None
    mask = df["sequence"].astype(str) == str(sequence_name)
    if not bool(mask.any()):
        return None
    try:
        return df.loc[mask].iloc[-1].to_dict()
    except Exception:
        return None


def append_rows_no_dedup(path: Path, rows: Sequence[Dict[str, object]], *, force: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path not in _STRUCTURED_PREDICTIONS_CLEARED:
        if path.is_file():
            try:
                path.unlink()
            except Exception:
                pass
        _STRUCTURED_PREDICTIONS_CLEARED.add(path)
    write_header = not path.is_file() or path.stat().st_size == 0
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        mode="a" if not write_header else "w",
        header=write_header,
    )


def write_parquet_from_csv(
    csv_path: Path,
    parquet_path: Path,
    *,
    chunksize: int = 200_000,
) -> None:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        logging.warning("Skipping parquet export; CSV missing/empty: %s", csv_path)
        return
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.is_file():
        parquet_path.unlink()
    del chunksize
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False, compression="zstd")


def compute_dense_flow(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    args: Optional[argparse.Namespace],
) -> Optional[np.ndarray]:
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return None
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    if args is not None:
        pyr_scale = float(getattr(args, "flow_pyr_scale", pyr_scale))
        levels = int(getattr(args, "flow_levels", levels))
        winsize = int(getattr(args, "flow_winsize", winsize))
        iterations = int(getattr(args, "flow_iterations", iterations))
        poly_n = int(getattr(args, "flow_poly_n", poly_n))
        poly_sigma = float(getattr(args, "flow_poly_sigma", poly_sigma))
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        return flow
    except Exception:
        return None


def extract_player_crops(
    seq_meta: SequenceMetadata,
    gt_df: pd.DataFrame,
    frame_indices: Sequence[int],
    class_col: str,
    team_col: str,
    player_class_ids: Iterable[int],
    team_map: Dict[int, int],
    args: argparse.Namespace,
    crop_center_ratio: float,
    crop_method: str = CROP_METHOD_CENTER,
    sam2_cropper: Optional[Sam2Cropper] = None,
    track_team_map: Optional[Dict[int, int]] = None,
) -> tuple[List[np.ndarray], List[int], List[CropInfo]]:
    crops: List[np.ndarray] = []
    labels: List[int] = []
    infos: List[CropInfo] = []
    frame_set = set(int(idx) for idx in frame_indices)
    filtered = gt_df[gt_df["frame"].isin(frame_set)]
    class_id_set = set(int(cid) for cid in player_class_ids)
    grouped = filtered.groupby("frame")
    cv2_mask_iter = int(getattr(args, "cv2_mask_iter", 2))
    cv2_mask_margin_ratio = float(getattr(args, "cv2_mask_margin_ratio", 0.05))
    cropper = Cropper(crop_center_ratio, crop_method, sam2_cropper, cv2_mask_iter, cv2_mask_margin_ratio)
    flow_computer = FlowComputer(args, enabled=bool(getattr(args, "use_optical_flow_features", False)))
    flow_enabled = flow_computer.enabled
    sorted_frames = sorted(grouped.groups.keys())

    for frame_idx in sorted_frames:
        detections = grouped.get_group(frame_idx)
        image_path = seq_meta.image_dir / f"{int(frame_idx):06d}{seq_meta.image_ext}"
        if not image_path.is_file():
            logging.warning("Frame %s missing at %s", frame_idx, image_path)
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.warning("Frame %s could not be read from %s", frame_idx, image_path)
            continue
        flow = None
        if flow_enabled:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = flow_computer.compute(frame_gray, int(frame_idx))
        height, width = frame.shape[:2]
        # Special fast-path: batch SAM2 inference per frame instead of per detection.
        if crop_method == CROP_METHOD_SAM2 and sam2_cropper is not None:
            try:
                sam2_cropper.prepare_frame(frame)
            except Exception as exc:
                logging.exception("Failed to prepare frame %s for SAM2: %s", frame_idx, exc)
                continue
            packed: List[tuple[tuple[int, int, int, int], int, str]] = []
            boxes: List[List[float]] = []
            for _, row in detections.iterrows():
                class_value = int(row[class_col]) if class_col in detections.columns else int(row.iloc[0])
                if class_id_set and class_value not in class_id_set:
                    continue
                track_id = None
                if "track_id" in detections.columns:
                    try:
                        track_id = int(row["track_id"])
                    except Exception:
                        track_id = None
                if track_team_map is not None:
                    if track_id is None:
                        continue
                    mapped = track_team_map.get(track_id)
                    if mapped is None:
                        continue
                    team_label = int(mapped)
                else:
                    team_raw = int(row[team_col]) if team_col in detections.columns else class_value
                    team_label = int(team_map.get(team_raw, team_raw))
                clamped = clamp_box(row["x"], row["y"], row["w"], row["h"], width, height)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                player_identifier = str(track_id) if track_id is not None else "unknown"
                packed.append((clamped, team_label, player_identifier))
                boxes.append([float(x1), float(y1), float(x2), float(y2)])

            if not packed:
                continue
            sam2_crops = sam2_cropper.extract_crops(frame, np.asarray(boxes, dtype=np.float32))
            for (clamped, team_label, player_identifier), crop in zip(packed, sam2_crops):
                x1, y1, x2, y2 = clamped
                crop_method_used = CROP_METHOD_SAM2
                if crop is None or getattr(crop, "size", 0) == 0:
                    crop_method_used = CROP_METHOD_CENTER
                    cx1, cy1, cx2, cy2 = center_shrink_box(
                        x1, y1, x2, y2, width, height, float(crop_center_ratio)
                    )
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop is None or crop.size == 0:
                        continue
                    x1, y1, x2, y2 = cx1, cy1, cx2, cy2
                flow_mean = None
                flow_std = None
                if flow is not None and flow.size > 0:
                    flow_roi = flow[y1:y2, x1:x2]
                    if flow_roi.size > 0:
                        mag = np.sqrt(flow_roi[..., 0] ** 2 + flow_roi[..., 1] ** 2)
                        flow_mean = float(np.mean(mag))
                        flow_std = float(np.std(mag))
                crops.append(crop)
                labels.append(int(team_label))
                infos.append(
                    CropInfo(
                        frame_index=int(frame_idx),
                        player_id=str(player_identifier),
                        team_label=int(team_label),
                        crop_center_ratio=float(crop_center_ratio),
                        saved_path=None,
                        crop_method=crop_method_used,
                        flow_mean=flow_mean,
                        flow_std=flow_std,
                    )
                )
            continue

        try:
            cropper.prepare_frame(frame)
        except Exception as exc:
            logging.exception("Failed to prepare frame %s for crop method %s: %s", frame_idx, crop_method, exc)
            continue
        for _, row in detections.iterrows():
            class_value = int(row[class_col]) if class_col in detections.columns else int(row.iloc[0])
            if class_id_set and class_value not in class_id_set:
                continue
            track_id = None
            if "track_id" in detections.columns:
                try:
                    track_id = int(row["track_id"])
                except Exception:
                    track_id = None

            if track_team_map is not None:
                if track_id is None:
                    continue
                mapped = track_team_map.get(track_id)
                if mapped is None:
                    continue
                team_label = int(mapped)
            else:
                team_raw = int(row[team_col]) if team_col in detections.columns else class_value
                team_label = int(team_map.get(team_raw, team_raw))
            clamped = clamp_box(row["x"], row["y"], row["w"], row["h"], width, height)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            crop, crop_method_used = cropper.crop(frame, clamped)
            if crop is None or crop.size == 0:
                continue
            flow_mean = None
            flow_std = None
            if flow is not None and flow.size > 0:
                flow_roi = flow[y1:y2, x1:x2]
                if flow_roi.size > 0:
                    mag = np.sqrt(flow_roi[..., 0] ** 2 + flow_roi[..., 1] ** 2)
                    flow_mean = float(np.mean(mag))
                    flow_std = float(np.std(mag))
            crops.append(crop)
            labels.append(int(team_label))
            player_identifier = "unknown"
            if track_id is not None:
                player_identifier = str(track_id)
            infos.append(
                CropInfo(
                    frame_index=int(frame_idx),
                    player_id=player_identifier,
                    team_label=int(team_label),
                    crop_center_ratio=float(crop_center_ratio),
                    saved_path=None,
                    crop_method=crop_method_used,
                    flow_mean=flow_mean,
                    flow_std=flow_std,
                )
            )
    return crops, labels, infos


def extract_player_crops_center_multi_ratio(
    seq_meta: SequenceMetadata,
    gt_df: pd.DataFrame,
    frame_indices: Sequence[int],
    class_col: str,
    team_col: str,
    player_class_ids: Iterable[int],
    team_map: Dict[int, int],
    args: argparse.Namespace,
    crop_center_ratios: Sequence[float],
    track_team_map: Optional[Dict[int, int]] = None,
) -> Dict[float, tuple[List[np.ndarray], List[int], List[CropInfo]]]:
    ratios = sorted({float(max(0.05, min(1.0, float(r)))) for r in crop_center_ratios})
    out: Dict[float, tuple[List[np.ndarray], List[int], List[CropInfo]]] = {
        ratio: ([], [], []) for ratio in ratios
    }
    if not ratios:
        return out

    frame_set = set(int(idx) for idx in frame_indices)
    filtered = gt_df[gt_df["frame"].isin(frame_set)]
    class_id_set = set(int(cid) for cid in player_class_ids)
    grouped = filtered.groupby("frame")
    flow_computer = FlowComputer(args, enabled=bool(getattr(args, "use_optical_flow_features", False)))
    flow_enabled = flow_computer.enabled
    sorted_frames = sorted(grouped.groups.keys())

    for frame_idx in sorted_frames:
        detections = grouped.get_group(frame_idx)
        image_path = seq_meta.image_dir / f"{int(frame_idx):06d}{seq_meta.image_ext}"
        if not image_path.is_file():
            logging.warning("Frame %s missing at %s", frame_idx, image_path)
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.warning("Frame %s could not be read from %s", frame_idx, image_path)
            continue
        flow = None
        if flow_enabled:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = flow_computer.compute(frame_gray, int(frame_idx))
        height, width = frame.shape[:2]

        for _, row in detections.iterrows():
            class_value = int(row[class_col]) if class_col in detections.columns else int(row.iloc[0])
            if class_id_set and class_value not in class_id_set:
                continue
            track_id = None
            if "track_id" in detections.columns:
                try:
                    track_id = int(row["track_id"])
                except Exception:
                    track_id = None

            if track_team_map is not None:
                if track_id is None:
                    continue
                mapped = track_team_map.get(track_id)
                if mapped is None:
                    continue
                team_label = int(mapped)
            else:
                team_raw = int(row[team_col]) if team_col in detections.columns else class_value
                team_label = int(team_map.get(team_raw, team_raw))
            clamped = clamp_box(row["x"], row["y"], row["w"], row["h"], width, height)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            player_identifier = str(track_id) if track_id is not None else "unknown"
            for ratio in ratios:
                rx1, ry1, rx2, ry2 = center_shrink_box(x1, y1, x2, y2, width, height, ratio)
                crop = frame[ry1:ry2, rx1:rx2]
                if crop is None or crop.size == 0:
                    continue
                flow_mean = None
                flow_std = None
                if flow is not None and flow.size > 0:
                    flow_roi = flow[ry1:ry2, rx1:rx2]
                    if flow_roi.size > 0:
                        mag = np.sqrt(flow_roi[..., 0] ** 2 + flow_roi[..., 1] ** 2)
                        flow_mean = float(np.mean(mag))
                        flow_std = float(np.std(mag))
                crops, labels, infos = out[ratio]
                crops.append(crop)
                labels.append(int(team_label))
                infos.append(
                    CropInfo(
                        frame_index=int(frame_idx),
                        player_id=player_identifier,
                        team_label=int(team_label),
                        crop_center_ratio=float(ratio),
                        saved_path=None,
                        crop_method=CROP_METHOD_CENTER,
                        flow_mean=flow_mean,
                        flow_std=flow_std,
                    )
                )
    return out


def compute_frame_indices(
    seq_meta: SequenceMetadata, stride_seconds: float, max_frames: Optional[int]
) -> List[int]:
    if seq_meta.frame_rate <= 0:
        raise ValueError(f"Invalid frame rate for sequence {seq_meta.name}: {seq_meta.frame_rate}")
    stride_frames = max(1, int(round(seq_meta.frame_rate * stride_seconds)))
    seq_length = seq_meta.num_frames or 0
    if seq_length <= 0:
        image_files = sorted(seq_meta.image_dir.glob(f"*{seq_meta.image_ext}"))
        seq_length = len(image_files)
    frame_indices = list(range(1, seq_length + 1, stride_frames))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    return frame_indices


def dataset_has_mot_sequences(dataset_root: Path) -> bool:
    try:
        for entry in dataset_root.iterdir():
            if entry.is_dir() and (entry / "seqinfo.ini").is_file():
                return True
    except FileNotFoundError:
        return False
    return False


def extract_crops_from_csv_video(
    video_path: Path,
    csv_path: Path,
    stride_seconds: float,
    max_frames: Optional[int],
    team_map: Dict[int, int],
    target_team_ids: Sequence[int],
    save_dir: Optional[Path] = None,
    crop_center_ratio: float = 1.0,
    crop_method: str = CROP_METHOD_CENTER,
    sam2_cropper: Optional[Sam2Cropper] = None,
    args: Optional[argparse.Namespace] = None,
) -> tuple[List[np.ndarray], List[int], List[CropInfo]]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=[0, 1, 2])
    if df.shape[0] <= 1:
        logging.warning("CSV %s does not contain any frame annotations.", csv_path.name)
        return [], [], []

    frame_col = df.columns[0]
    df = df.iloc[1:].copy()
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df = df.dropna(subset=[frame_col])
    df[frame_col] = df[frame_col].astype(int)
    frame_numbers = df[frame_col].tolist()
    frame_to_index = {int(frame): idx for idx, frame in enumerate(frame_numbers)}

    target_set = {int(team_id) for team_id in target_team_ids} if target_team_ids else set()
    team_columns: List[Tuple[str, int]] = []
    for key in df.columns.get_level_values(0).unique():
        if key == frame_col[0]:
            continue
        if isinstance(key, str) and key.startswith("Unnamed"):
            continue
        try:
            team_id = int(key)
        except (TypeError, ValueError):
            continue
        if target_set and team_id not in target_set:
            continue
        team_columns.append((key, team_id))

    if not team_columns:
        logging.warning("No matching team columns found in %s", csv_path.name)
        return [], [], []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    flow_computer = FlowComputer(args, enabled=bool(args and getattr(args, "use_optical_flow_features", False)))
    cv2_mask_iter = int(getattr(args, "cv2_mask_iter", 5))
    cv2_mask_margin_ratio = float(getattr(args, "cv2_mask_margin_ratio", 0.15))
    cropper = Cropper(
        crop_center_ratio,
        crop_method,
        sam2_cropper,
        cv2_mask_iter,
        cv2_mask_margin_ratio,
    )
    flow_enabled = flow_computer.enabled

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count_attr = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_count_attr) if frame_count_attr else len(frame_numbers)
        if frame_count <= 0:
            frame_count = len(frame_numbers)
        stride_frames = max(1, int(round(fps * stride_seconds)))
        frame_indices = list(range(0, frame_count, stride_frames))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

        crops: List[np.ndarray] = []
        labels: List[int] = []
        infos: List[CropInfo] = []
        save_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        base_save_dir = save_dir.resolve() if save_dir else None
        game_id_parts = video_path.stem.split("_")
        game_id = "_".join(game_id_parts[:3]) if len(game_id_parts) >= 3 else video_path.stem

        for frame_idx in frame_indices:
            frame_number = frame_idx + 1
            row_idx = frame_to_index.get(frame_number)
            if row_idx is None:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = capture.read()
            if not success or frame is None:
                logging.warning(
                    "Failed to read frame %s from %s", frame_idx, video_path.name
                )
                continue
            flow = None
            if flow_enabled:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = flow_computer.compute(frame_gray, int(frame_number))
            height, width = frame.shape[:2]
            try:
                cropper.prepare_frame(frame)
            except Exception as exc:
                logging.exception(
                    "Failed to prepare frame %s for %s: %s",
                    frame_idx,
                    video_path.name,
                    exc,
                )
                continue
            row = df.iloc[row_idx]
            for key, team_id in team_columns:
                team_series = row[key]
                if isinstance(team_series, float) and pd.isna(team_series):
                    continue
                if isinstance(team_series, pd.Series) and team_series.isna().all():
                    continue
                if not isinstance(team_series, pd.Series):
                    continue
                player_ids = team_series.index.get_level_values(0).unique()
                for player_id in player_ids:
                    try:
                        left = float(team_series[(player_id, "bb_left")])
                        top = float(team_series[(player_id, "bb_top")])
                        width_bb = float(team_series[(player_id, "bb_width")])
                        height_bb = float(team_series[(player_id, "bb_height")])
                    except KeyError:
                        continue
                    if any(pd.isna(val) for val in (left, top, width_bb, height_bb)):
                        continue
                    if width_bb <= 1.0 or height_bb <= 1.0:
                        continue
                    clamped = clamp_box(left, top, width_bb, height_bb, width, height)
                    if clamped is None:
                        continue
                    x1, y1, x2, y2 = clamped
                    crop, crop_method_used = cropper.crop(frame, clamped)
                    if crop is None or crop.size == 0:
                        continue
                    mapped_label = team_map.get(team_id, team_id)
                    flow_mean = None
                    flow_std = None
                    if flow is not None and flow.size > 0:
                        flow_roi = flow[y1:y2, x1:x2]
                        if flow_roi.size > 0:
                            mag = np.sqrt(flow_roi[..., 0] ** 2 + flow_roi[..., 1] ** 2)
                            flow_mean = float(np.mean(mag))
                            flow_std = float(np.std(mag))
                    crops.append(crop)
                    labels.append(int(mapped_label))
                    if base_save_dir is not None:
                        team_subdir = base_save_dir / game_id / f"{int(mapped_label)}"
                        team_subdir.mkdir(parents=True, exist_ok=True)
                        key = (game_id, int(mapped_label))
                        save_counts[key] += 1
                        sanitized_player = str(player_id).replace("/", "_")
                        filename = (
                            f"frame{frame_number:05d}_player{sanitized_player}_{save_counts[key]:04d}.png"
                        )
                        target_path = team_subdir / filename
                        cv2.imwrite(str(target_path), crop)
                        saved_path = target_path
                    else:
                        saved_path = None
                    infos.append(
                        CropInfo(
                            frame_index=int(frame_number),
                            player_id=str(player_id),
                            team_label=int(mapped_label),
                            crop_center_ratio=float(crop_center_ratio),
                            saved_path=saved_path,
                            crop_method=crop_method_used,
                            flow_mean=flow_mean,
                            flow_std=flow_std,
                        )
                    )
    finally:
        capture.release()

    return crops, labels, infos


def collect_csv_sequences(
    dataset_root: Path,
    args: argparse.Namespace,
    team_map: Dict[int, int],
    target_team_ids: Sequence[int],
    save_crops_dir: Optional[Path],
    sequence_limit: Optional[int],
    crop_center_ratio: float,
    crop_method: str = CROP_METHOD_CENTER,
    sam2_cropper: Optional[Sam2Cropper] = None,
) -> Iterable[Tuple[str, List[np.ndarray], List[int], List[CropInfo]]]:
    video_files = sorted(dataset_root.glob("*.mp4"))
    if not video_files:
        logging.warning("No MP4 files found in %s", dataset_root)
        return

    if sequence_limit is not None:
        video_files = video_files[:sequence_limit]

    csv_lookup = {path.stem: path for path in dataset_root.glob("*.csv")}

    for video_path in video_files:
        csv_path = csv_lookup.get(video_path.stem)
        if not csv_path:
            logging.warning("Missing CSV annotations for %s", video_path.name)
            continue
        try:
            crops, labels, infos = extract_crops_from_csv_video(
                video_path,
                csv_path,
                args.sample_seconds,
                args.max_frames_per_sequence,
                team_map,
                target_team_ids,
                save_crops_dir,
                crop_center_ratio,
                crop_method,
                sam2_cropper,
                args,
            )
        except Exception as exc:
            logging.exception(
                "Skipping %s due to extraction error: %s", video_path.name, exc
            )
            continue
        yield (video_path.stem, crops, labels, infos)


_THREAD_LOCAL = threading.local()


def parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def normalize_embedding_backend(name: str) -> tuple[str, str]:
    """Return (label, internal) name. resnet16 is an alias of resnet18 embeddings."""
    label = (name or "").strip().lower()
    if label == EMBED_BACKEND_RESNET16:
        return EMBED_BACKEND_RESNET16, EMBED_BACKEND_RESNET
    if label == "resnet18":
        return "resnet18", EMBED_BACKEND_RESNET
    if label in {EMBED_BACKEND_RESNET, EMBED_BACKEND_SIGLIP, EMBED_BACKEND_CLIP}:
        return label, label
    return label, label


class ClipVisionEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None) -> None:
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModel
        except ImportError:
            from transformers import AutoImageProcessor as CLIPImageProcessor  # type: ignore
            from transformers import CLIPVisionModel  # type: ignore

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = 32

    def embed(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        from PIL import Image

        pil_images: List[Image.Image] = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
        if not pil_images:
            return np.empty((0, 0), dtype=np.float32)
        batches: List[np.ndarray] = []
        use_cuda = self.device.startswith("cuda")
        effective_bs = int(max(1, self.batch_size))
        start = 0
        while start < len(pil_images):
            remaining = len(pil_images) - start
            current_bs = min(effective_bs, remaining)
            while True:
                try:
                    batch_imgs = pil_images[start : start + current_bs]
                    inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
                    with torch.inference_mode():
                        out = self.model(**inputs)
                        pooled = getattr(out, "pooler_output", None)
                        if pooled is None:
                            pooled = out.last_hidden_state.mean(dim=1)
                    batches.append(pooled.detach().cpu().numpy())
                    del inputs
                    if use_cuda:
                        torch.cuda.empty_cache()
                    start += current_bs
                    break
                except torch.OutOfMemoryError:
                    if not use_cuda:
                        raise
                    torch.cuda.empty_cache()
                    if current_bs <= 1:
                        raise
                    current_bs = max(1, current_bs // 2)
        return np.concatenate(batches, axis=0).astype(np.float32)


def get_embedder(internal_backend: str, *, resnet_batch_size: int = 64):
    cache = getattr(_THREAD_LOCAL, "embedders", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.embedders = cache
    key = internal_backend.lower()
    cache_key = key if key != EMBED_BACKEND_RESNET else f"{key}:{int(resnet_batch_size)}"
    embedder = cache.get(cache_key)
    if embedder is not None:
        return embedder
    if key == EMBED_BACKEND_RESNET:
        embedder = ResnetTeamClassifier(batch_size=int(resnet_batch_size))
    elif key == EMBED_BACKEND_CLIP:
        embedder = ClipVisionEmbedder()
    else:
        config = {
            "use_umap": False,
            "color_space": "rgb",
            "color_hist_bins": 0,
            "color_hist_weight": 0.0,
        }
        embedder = SiglipTeamClassifier(
            model_name="google/siglip-base-patch16-224",
            siglip_config=config,
        )
    cache[cache_key] = embedder
    return embedder


def compute_color_hist_features(
    crops_bgr: List[np.ndarray], color_space: str, bins: int, weight: float
) -> np.ndarray:
    bins = int(bins)
    if bins <= 0:
        return np.empty((len(crops_bgr), 0), dtype=np.float32)
    cs = str(color_space).lower()
    if cs not in {"rgb", "hsv", "h"}:
        cs = "rgb"
    hist_channels = 1 if cs == "h" else 3
    hist_range = (0.0, 180.0) if cs == "h" else (0.0, 256.0)
    out: List[np.ndarray] = []
    for crop in crops_bgr:
        if crop is None or crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            out.append(np.zeros((hist_channels * bins,), dtype=np.float32))
            continue
        if cs == "rgb":
            src = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            channels = cv2.split(src)
        elif cs == "hsv":
            src = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            channels = cv2.split(src)
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            channels = [hsv[:, :, 0]]
        parts: List[np.ndarray] = []
        for ch in channels:
            hist = cv2.calcHist([ch], [0], None, [bins], [float(hist_range[0]), float(hist_range[1])])
            hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
            parts.append(hist.astype(np.float32))
        vec = np.concatenate(parts, axis=0) if parts else np.zeros((hist_channels * bins,), dtype=np.float32)
        out.append(vec)
    feats = np.stack(out).astype(np.float32)
    if weight != 1.0:
        feats *= float(weight)
    return feats


def get_embedding_backend(color_space: str, backend_name: str):
    """Return a cached embedding backend instance for the given color space."""
    key = (backend_name.lower(), color_space.lower())
    cache = getattr(_THREAD_LOCAL, "embedding_backends", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.embedding_backends = cache
    backend = cache.get(key)
    if backend is None:
        if backend_name.lower() == "resnet":
            backend = ResnetTeamClassifier()
        else:
            config = {
                "use_umap": False,
                "color_space": key[1],
                "color_hist_bins": 32,
                "color_hist_weight": 1.0,
            }
            backend = SiglipTeamClassifier(
                model_name="google/siglip-base-patch16-224",
                siglip_config=config,
            )
        cache[key] = backend
    return backend


def align_clusters(
    predictions: np.ndarray, labels: np.ndarray, unique_labels: Sequence[int]
) -> Optional[np.ndarray]:
    if predictions.size == 0:
        return None
    clusters = sorted(set(int(p) for p in predictions))
    if len(clusters) != len(unique_labels):
        logging.warning(
            "Cluster count %s does not match number of unique labels %s", len(clusters), len(unique_labels)
        )
        return None
    best_mapping: Dict[int, int] = {}
    best_accuracy = -1
    for perm in permutations(unique_labels):
        mapping = {cluster: perm[idx] for idx, cluster in enumerate(clusters)}
        mapped = np.array([mapping[int(p)] for p in predictions], dtype=int)
        accuracy = float(np.mean(mapped == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping
    if not best_mapping:
        return None
    return np.array([best_mapping[int(p)] for p in predictions], dtype=int)


def build_metric_row(
    sequence: str,
    color_space: str,
    umap_components: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crop_center_ratio: float,
    crop_method: str,
    experiment_stage: str,
    umap_applied: bool,
    original_feature_dim: int,
    embedding_backend: str,
    cluster_method: str,
) -> Optional[Dict[str, object]]:
    if y_true.size == 0 or y_pred.size == 0:
        return None
    unique_labels = sorted(set(int(val) for val in np.unique(y_true)))
    if len(unique_labels) != 2:
        logging.warning(
            "Sequence %s expected binary labels but found: %s", sequence, unique_labels
        )
        return None
    neg_label, pos_label = unique_labels[0], unique_labels[1]
    y_true_pos = y_true == pos_label
    y_pred_pos = y_pred == pos_label
    y_true_neg = y_true == neg_label
    y_pred_neg = y_pred == neg_label
    tp = int(np.sum(y_true_pos & y_pred_pos))
    tn = int(np.sum(y_true_neg & y_pred_neg))
    fp = int(np.sum(y_true_neg & y_pred_pos))
    fn = int(np.sum(y_true_pos & y_pred_neg))
    total = int(y_true.size)
    accuracy = float((tp + tn) / total) if total else 0.0
    return {
        "sequence": sequence,
        "color_space": color_space,
        "umap_components": umap_components,
        "original_feature_dim": int(original_feature_dim),
        "umap_applied": bool(umap_applied),
        "crop_center_ratio": float(crop_center_ratio),
        "crop_method": crop_method,
        "experiment_stage": experiment_stage,
        "embedding_backend": embedding_backend,
        "cluster_method": str(cluster_method),
        "positive_label": pos_label,
        "negative_label": neg_label,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": accuracy,
        "num_samples": total,
    }


def _build_prediction_rows(
    sequence_name: str,
    color_space: str,
    umap_components: int,
    crop_center_ratio: float,
    y_true_list: List[int],
    aligned_list: List[int],
    infos: Sequence[CropInfo],
    crop_method: str,
    experiment_stage: str,
    embedding_backend: str,
    cluster_method: str,
) -> List[Dict[str, object]]:
    prediction_rows: List[Dict[str, object]] = []
    if infos and len(infos) == len(aligned_list):
        for info, truth, pred in zip(infos, y_true_list, aligned_list):
            prediction_rows.append(
                {
                    "sequence": sequence_name,
                    "color_space": color_space,
                    "umap_components": int(umap_components),
                    "experiment_stage": experiment_stage,
                    "crop_center_ratio": float(info.crop_center_ratio),
                    "crop_method": info.crop_method,
                    "embedding_backend": embedding_backend,
                    "cluster_method": str(cluster_method),
                    "frame_index": info.frame_index,
                    "player_id": info.player_id,
                    "true_label": int(truth),
                    "predicted_label": int(pred),
                    "saved_crop_path": str(info.saved_path) if info.saved_path else "",
                }
            )
    else:
        for idx, (truth, pred) in enumerate(zip(y_true_list, aligned_list)):
            info = infos[idx] if idx < len(infos) else None
            prediction_rows.append(
                {
                    "sequence": sequence_name,
                    "color_space": color_space,
                    "umap_components": int(umap_components),
                    "experiment_stage": experiment_stage,
                    "crop_center_ratio": float(info.crop_center_ratio if info else crop_center_ratio),
                    "crop_method": info.crop_method if info else crop_method,
                    "embedding_backend": embedding_backend,
                    "cluster_method": str(cluster_method),
                    "frame_index": info.frame_index if info else -1,
                    "player_id": info.player_id if info else str(idx),
                    "true_label": int(truth),
                    "predicted_label": int(pred),
                    "saved_crop_path": str(info.saved_path) if info and info.saved_path else "",
                }
            )
    return prediction_rows


def compute_weighted_accuracy(rows: Sequence[Dict[str, object]]) -> float:
    total_samples = sum(int(row.get("num_samples", 0)) for row in rows)
    correct = sum(
        int(row.get("true_positive", 0)) + int(row.get("true_negative", 0)) for row in rows
    )
    if total_samples <= 0:
        return 0.0
    return float(correct / total_samples)


def resolve_stage_save_dir(
    base_dir: Optional[Path],
    stage_name: str,
    crop_method: str,
    ratio: float,
    color_space: Optional[str] = None,
) -> Optional[Path]:
    if base_dir is None:
        return None
    safe_stage = stage_name.replace(" ", "_")
    stage_dir = base_dir / safe_stage
    if crop_method == CROP_METHOD_SAM2:
        identifier = "sam2_mask"
    elif crop_method == CROP_METHOD_OPENCV:
        identifier = "opencv_mask"
    else:
        identifier = f"ratio_{format_ratio_identifier(ratio)}"
    if color_space:
        identifier = f"{identifier}_{color_space.lower()}"
    target_dir = stage_dir / identifier
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def run_stage_experiments(
    dataset_root: Path,
    args: argparse.Namespace,
    color_spaces: Sequence[str],
    umap_values: Sequence[Optional[int]],
    crop_center_ratio: float,
    crop_method: str,
    experiment_stage: str,
    prediction_sink: List[Dict[str, object]],
    team_map: Dict[int, int],
    player_class_ids: Sequence[int],
    csv_team_ids: Sequence[int],
    has_mot_sequences: bool,
    sam2_cropper: Optional[Sam2Cropper],
    save_crops_dir: Optional[Path],
) -> List[Dict[str, object]]:
    stage_results: List[Dict[str, object]] = []
    sequence_limit = args.sequence_limit if args.sequence_limit and args.sequence_limit > 0 else None
    progress_desc = f"{experiment_stage} | {crop_method}"
    if crop_method != CROP_METHOD_SAM2:
        progress_desc += f" | ratio={crop_center_ratio:.2f}"
    if has_mot_sequences:
        sequences = [entry for entry in sorted(dataset_root.iterdir()) if (entry / "seqinfo.ini").is_file()]
        if sequence_limit is not None:
            sequences = sequences[:sequence_limit]
        iterator = tqdm(sequences, desc=progress_desc, unit="seq", leave=False) if sequences else []
        if bool(getattr(args, "structured_cache_only", False)):
            for seq_dir in iterator:
                seq_results = run_experiment_for_sequence_from_structured_cache(
                    seq_dir.name,
                    color_spaces,
                    umap_values,
                    args,
                    prediction_sink,
                    crop_center_ratio,
                    crop_method,
                    experiment_stage,
                )
                stage_results.extend(seq_results)
            return stage_results
        for seq_dir in iterator:
            try:
                seq_meta = read_sequence_metadata(seq_dir)
                gt_df = load_ground_truth(seq_dir)
            except Exception as exc:
                logging.exception("Skipping sequence %s due to error: %s", seq_dir.name, exc)
                continue
            try:
                class_col = resolve_column_name(gt_df, args.class_column_name, args.class_column_index)
                team_col = resolve_column_name(gt_df, args.team_column_name, args.team_column_index)
            except Exception as exc:
                logging.exception("Failed to resolve columns for %s: %s", seq_dir.name, exc)
                continue
            track_team_map = None
            if str(getattr(args, "mot_team_source", "gt_column")).lower() == "gameinfo":
                track_team_map = parse_soccernet_gameinfo_team_map(seq_dir)
                if not track_team_map:
                    logging.warning(
                        "MOT team source is gameinfo but mapping is empty for %s; skipping.",
                        seq_dir.name,
                    )
                    continue
            frame_indices = compute_frame_indices(seq_meta, args.sample_seconds, args.max_frames_per_sequence)
            try:
                crops, labels, infos = extract_player_crops(
                    seq_meta,
                    gt_df,
                    frame_indices,
                    class_col,
                    team_col,
                    player_class_ids,
                    team_map,
                    args,
                    crop_center_ratio,
                    crop_method,
                    sam2_cropper,
                    track_team_map=track_team_map,
                )
            except Exception as exc:
                logging.exception("Extraction failed for %s during stage %s: %s", seq_dir.name, experiment_stage, exc)
                continue
            if not crops:
                logging.warning(
                    "%s: Sequence %s produced no crops (method=%s, ratio=%.3f)",
                    experiment_stage,
                    seq_meta.name,
                    crop_method,
                    crop_center_ratio,
                )
                continue
            seq_results = run_experiment_for_sequence(
                seq_meta.name,
                crops,
                labels,
                infos,
                color_spaces,
                umap_values,
                args,
                prediction_sink,
                crop_center_ratio,
                crop_method,
                experiment_stage,
            )
            stage_results.extend(seq_results)
    else:
        first_color = color_spaces[0] if color_spaces else None
        stage_save_dir = resolve_stage_save_dir(
            save_crops_dir, experiment_stage, crop_method, crop_center_ratio, first_color
        )
        csv_sequences = collect_csv_sequences(
            dataset_root,
            args,
            team_map,
            csv_team_ids,
            stage_save_dir,
            sequence_limit,
            crop_center_ratio,
            crop_method,
            sam2_cropper,
        )
        if csv_sequences is None:
            logging.warning(
                "%s: No CSV/video pairs found under %s (method=%s, ratio=%.3f)",
                experiment_stage,
                dataset_root,
                crop_method,
                crop_center_ratio,
            )
            return stage_results
        any_sequence = False
        iterator = tqdm(csv_sequences, desc=progress_desc, unit="seq", leave=False)
        for seq_name, crops, labels, infos in iterator:
            any_sequence = True
            if not crops:
                logging.warning(
                    "%s: Sequence %s produced no crops (method=%s, ratio=%.3f)",
                    experiment_stage,
                    seq_name,
                    crop_method,
                    crop_center_ratio,
                )
                continue
            seq_results = run_experiment_for_sequence(
                seq_name,
                crops,
                labels,
                infos,
                color_spaces,
                umap_values,
                args,
                prediction_sink,
                crop_center_ratio,
                crop_method,
                experiment_stage,
            )
            stage_results.extend(seq_results)
        if not any_sequence:
            logging.warning(
                "%s: No CSV/video pairs found under %s (method=%s, ratio=%.3f)",
                experiment_stage,
                dataset_root,
                crop_method,
                crop_center_ratio,
            )
    return stage_results


def init_sam2_cropper(args: argparse.Namespace) -> Optional[Sam2Cropper]:
    if args.skip_sam2:
        logging.info("SAM2 stage disabled via --skip-sam2.")
        return None
    if not args.sam2_config_path or not args.sam2_checkpoint_path:
        logging.warning("SAM2 config/checkpoint not provided; skipping SAM2 stage.")
        return None
    try:
        logging.info(
            "Initializing SAM2 cropper (config=%s, checkpoint=%s)",
            args.sam2_config_path,
            args.sam2_checkpoint_path,
        )
        return Sam2Cropper(
            args.sam2_config_path,
            args.sam2_checkpoint_path,
            device=args.sam2_device,
        )
    except Exception as exc:
        logging.exception("Failed to initialize SAM2 cropper: %s", exc)
        return None


def _run_single_configuration(
    sequence_name: str,
    color_space: str,
    umap_components: Optional[int],
    features: np.ndarray,
    y_true: np.ndarray,
    unique_labels: Sequence[int],
    crop_infos: Sequence[CropInfo],
    crop_center_ratio: float,
    crop_method: str,
    experiment_stage: str,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    kmeans_init: int,
    cluster_method: str,
    cluster_params: Dict[str, object],
) -> tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    try:
        original_dim = features.shape[1] if features.ndim == 2 else 0
        umap_applied = umap_components is not None and int(umap_components) > 0
        sample_count = features.shape[0]
        if sample_count <= 0:
            return None, []
        if umap_applied and sample_count <= int(umap_components):
            return None, []
        # Keep UMAP within spectral solver limits: components < n_samples - 1
        effective_components = (
            min(int(umap_components), max(sample_count - 3, 2)) if umap_applied else None
        )
        if umap_applied and (effective_components is None or effective_components < 2):
            return None, []
        if umap_applied and effective_components >= sample_count - 1:
            return None, []
        if umap_applied:
            reducer = umap.UMAP(
                n_components=effective_components,
                n_neighbors=min(umap_neighbors, max(sample_count - 2, 2)),
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=0,
            )
            transformed = reducer.fit_transform(features)
            recorded_components = int(effective_components)
        else:
            transformed = features
            recorded_components = original_dim
        predictions: np.ndarray
        if cluster_method == "dbscan":
            eps = float(cluster_params.get("dbscan_eps", 0.5))
            min_samples = int(cluster_params.get("dbscan_min_samples", 5))
            predictions, mask = run_dbscan_predict(
                transformed, unique_labels, eps=eps, min_samples=min_samples
            )
            if predictions is None or mask is None:
                return None, []
            y_true = y_true[mask]
            crop_infos = tuple(info for idx, info in enumerate(crop_infos) if mask[idx])
            transformed = transformed[mask]
        elif cluster_method == "cmeans":
            predictions = run_cmeans_predict(
                transformed,
                n_clusters=len(unique_labels),
                m=float(cluster_params.get("cmeans_m", 2.0)),
                max_iter=int(cluster_params.get("cmeans_max_iter", 50)),
                tol=float(cluster_params.get("cmeans_tol", 1e-4)),
                random_state=0,
            )
        else:
            kmeans = KMeans(
                n_clusters=len(unique_labels),
                n_init=kmeans_init,
                random_state=0,
            )
            predictions = kmeans.fit_predict(transformed)
    except Exception as exc:
        logging.exception(
            "Failed to run clustering for sequence %s (color=%s, umap=%s): %s",
            sequence_name,
            color_space,
            umap_components,
            exc,
        )
        return None, []

    aligned = align_clusters(predictions, y_true, unique_labels)
    if aligned is None:
        return None, []

    row = build_metric_row(
        sequence_name,
        color_space,
        int(recorded_components),
        y_true,
        aligned,
        crop_center_ratio,
        crop_method,
        experiment_stage,
        umap_applied,
        original_dim,
        cluster_params.get("embedding_backend", "siglip"),
        str(cluster_method),
    )

    if row is None:
        return None, []

    y_true_list = y_true.tolist()
    aligned_list = aligned.tolist()
    prediction_rows = _build_prediction_rows(
        sequence_name,
        color_space,
        recorded_components,
        crop_center_ratio,
        y_true_list,
        aligned_list,
        crop_infos,
        crop_method,
        experiment_stage,
        cluster_params.get("embedding_backend", "siglip"),
        str(cluster_method),
    )
    return row, prediction_rows


def _run_single_configuration_from_transformed(
    sequence_name: str,
    color_space: str,
    recorded_components: int,
    transformed: np.ndarray,
    y_true: np.ndarray,
    unique_labels: Sequence[int],
    crop_infos: Sequence[CropInfo],
    crop_center_ratio: float,
    crop_method: str,
    experiment_stage: str,
    kmeans_init: int,
    cluster_method: str,
    cluster_params: Dict[str, object],
    *,
    umap_applied: bool,
    original_dim: int,
    umap_fit_components: int,
) -> tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    try:
        predictions: np.ndarray
        infos_local = crop_infos
        y_true_local = y_true
        if cluster_method == "dbscan":
            eps = float(cluster_params.get("dbscan_eps", 0.5))
            min_samples = int(cluster_params.get("dbscan_min_samples", 5))
            predictions, mask = run_dbscan_predict(
                transformed, unique_labels, eps=eps, min_samples=min_samples
            )
            if predictions is None or mask is None:
                return None, []
            y_true_local = y_true_local[mask]
            infos_local = tuple(info for idx, info in enumerate(infos_local) if mask[idx])
        elif cluster_method == "cmeans":
            predictions = run_cmeans_predict(
                transformed,
                n_clusters=len(unique_labels),
                m=float(cluster_params.get("cmeans_m", 2.0)),
                max_iter=int(cluster_params.get("cmeans_max_iter", 50)),
                tol=float(cluster_params.get("cmeans_tol", 1e-4)),
                random_state=0,
            )
        else:
            kmeans = KMeans(
                n_clusters=len(unique_labels),
                n_init=kmeans_init,
                random_state=0,
            )
            predictions = kmeans.fit_predict(transformed)
    except Exception as exc:
        logging.exception(
            "Failed to run clustering for sequence %s (color=%s, umap=%s, method=%s): %s",
            sequence_name,
            color_space,
            recorded_components,
            cluster_method,
            exc,
        )
        return None, []

    aligned = align_clusters(predictions, y_true_local, unique_labels)
    if aligned is None:
        return None, []

    embedding_backend = str(cluster_params.get("embedding_backend", "siglip"))
    row = build_metric_row(
        sequence_name,
        color_space,
        int(recorded_components),
        y_true_local,
        aligned,
        crop_center_ratio,
        crop_method,
        experiment_stage,
        bool(umap_applied),
        int(original_dim),
        embedding_backend,
        str(cluster_method),
    )

    if row is None:
        return None, []
    row["umap_fit_components"] = int(umap_fit_components)

    if not bool(cluster_params.get("save_predictions", True)):
        return row, []

    y_true_list = y_true_local.tolist()
    aligned_list = aligned.tolist()
    prediction_rows = _build_prediction_rows(
        sequence_name,
        color_space,
        int(recorded_components),
        crop_center_ratio,
        y_true_list,
        aligned_list,
        infos_local,
        crop_method,
        experiment_stage,
        embedding_backend,
        str(cluster_method),
    )
    return row, prediction_rows


def build_flow_features(crop_infos: Sequence[CropInfo]) -> np.ndarray:
    feats: List[List[float]] = []
    for info in crop_infos:
        mean = info.flow_mean if info.flow_mean is not None else 0.0
        std = info.flow_std if info.flow_std is not None else 0.0
        feats.append([float(mean), float(std)])
    if not feats:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def run_dbscan_predict(
    data: np.ndarray,
    unique_labels: Sequence[int],
    eps: float,
    min_samples: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run DBSCAN and coerce the output into exactly ``len(unique_labels)`` clusters.

    Notes:
    - DBSCAN can return noise points (-1) and an arbitrary number of clusters.
    - Downstream we require exactly 2 clusters to align labels (home/away), so we:
      * keep the largest clusters,
      * assign all remaining points (noise + small clusters) to the nearest kept centroid.
    - If DBSCAN degenerates to 0 clusters / a single cluster without noise, we fall back to k-means.
    """
    if data.size == 0:
        return None, None
    target_clusters = int(len(unique_labels))
    if target_clusters <= 0:
        return None, None

    def _coerce_labels(db_labels: np.ndarray) -> Optional[np.ndarray]:
        labels = np.asarray(db_labels, dtype=int)
        cluster_ids = [int(val) for val in np.unique(labels) if int(val) != -1]
        has_noise = bool(np.any(labels == -1))

        # Case 1: at least K clusters -> keep K largest and merge the rest.
        if len(cluster_ids) >= target_clusters:
            counts = {cid: int(np.sum(labels == cid)) for cid in cluster_ids}
            kept = sorted(cluster_ids, key=lambda cid: (counts.get(cid, 0), cid), reverse=True)[
                :target_clusters
            ]
            if len(kept) != target_clusters:
                return None
            centroids = []
            for cid in kept:
                pts = data[labels == cid]
                if pts.size == 0:
                    return None
                centroids.append(np.mean(pts, axis=0))
            centroids_arr = np.stack(centroids, axis=0)
            mapping = {cid: idx for idx, cid in enumerate(kept)}
            out = np.full(labels.shape[0], -1, dtype=int)
            for cid, idx in mapping.items():
                out[labels == cid] = int(idx)
            other_mask = out == -1
            if np.any(other_mask):
                pts = data[other_mask]
                diffs = pts[:, None, :] - centroids_arr[None, :, :]
                dists = np.einsum("ijk,ijk->ij", diffs, diffs)
                out[other_mask] = np.argmin(dists, axis=1).astype(int)
            if len(np.unique(out)) != target_clusters:
                return None
            return out

        # Case 2 (K=2): single cluster + noise -> treat noise as the second cluster.
        if target_clusters == 2 and len(cluster_ids) == 1 and has_noise:
            out = np.zeros(labels.shape[0], dtype=int)
            out[labels == -1] = 1
            if len(np.unique(out)) == 2:
                return out
            return None

        return None

    eps_base = max(float(eps), 1e-6)
    # Keep attempts minimal; DBSCAN is the slow part and we run this many times in a grid.
    for eps_try in (eps_base, eps_base * 2.0, eps_base * 0.5):
        db = DBSCAN(eps=float(eps_try), min_samples=int(min_samples))
        coerced = _coerce_labels(db.fit_predict(data))
        if coerced is not None:
            mask = np.ones(coerced.shape[0], dtype=bool)
            return coerced, mask

    try:
        kmeans = KMeans(n_clusters=target_clusters, n_init=10, random_state=0)
        preds = kmeans.fit_predict(data)
        mask = np.ones(preds.shape[0], dtype=bool)
        return preds.astype(int), mask
    except Exception:
        return None, None


def run_cmeans_predict(
    data: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 50,
    tol: float = 1e-4,
    random_state: int = 0,
) -> np.ndarray:
    if data.size == 0:
        return np.array([], dtype=int)
    m = max(float(m), 1.0001)
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    # Initialize membership matrix randomly
    u = rng.random((n_clusters, n_samples))
    u = u / u.sum(axis=0, keepdims=True)
    data = np.asarray(data, dtype=float)
    for _ in range(max_iter):
        u_prev = u.copy()
        um = u ** m
        centers = (um @ data) / np.clip(um.sum(axis=1, keepdims=True), 1e-8, None)
        dist = np.linalg.norm(data[None, :, :] - centers[:, None, :], axis=2) + 1e-8
        inv_dist = dist ** (-2 / (m - 1))
        u = inv_dist / inv_dist.sum(axis=0, keepdims=True)
        if np.linalg.norm(u - u_prev) < tol:
            break
    labels = np.argmax(u, axis=0).astype(int)
    return labels


def run_experiment_for_sequence_from_structured_cache(
    sequence_name: str,
    color_spaces: Sequence[str],
    umap_range: Sequence[Optional[int]],
    args: argparse.Namespace,
    prediction_sink: List[Dict[str, object]],
    crop_center_ratio: float,
    crop_method: str,
    experiment_stage: str,
) -> List[Dict[str, object]]:
    structured_root_raw = getattr(args, "structured_results_root", None)
    if not structured_root_raw:
        logging.warning(
            "structured_cache_only is enabled but --structured-results-root is not set; skipping %s",
            sequence_name,
        )
        return []
    structured_root = Path(structured_root_raw).expanduser().resolve()
    structured_root.mkdir(parents=True, exist_ok=True)

    structured_force = bool(getattr(args, "structured_force", False))
    structured_write_pipeline_metrics = bool(getattr(args, "structured_write_pipeline_metrics", False))
    structured_write_pipeline_predictions = bool(getattr(args, "structured_write_pipeline_predictions", False))

    emit_cached_rows = True
    try:
        output_csv_path = Path(getattr(args, "output_csv")).expanduser().resolve()
        if output_csv_path.is_file() and output_csv_path.stat().st_size > 0:
            emit_cached_rows = False
    except Exception:
        emit_cached_rows = True

    sequence_stem = sanitize_identifier(sequence_name)

    embedding_backends_raw = parse_csv_list(args.embedding_backends) if args.embedding_backends else []
    embedding_backends = embedding_backends_raw or [str(args.embedding_backend)]
    resolved_backends = [normalize_embedding_backend(name) for name in embedding_backends]

    cluster_methods = parse_csv_list(args.cluster_methods) if args.cluster_methods else []
    cluster_methods = cluster_methods or [str(args.cluster_method)]

    feature_map: Dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    info_source: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    for label_name, _internal_name in resolved_backends:
        for color_space in color_spaces:
            color_key = str(color_space).strip().lower()
            embedding_dir = resolve_structured_embedding_dir(
                structured_root,
                color_space=color_key,
                crop_method=crop_method,
                crop_center_ratio=float(crop_center_ratio),
                embedding_backend=label_name,
            )
            feature_path = embedding_dir / f"{sequence_stem}.npz"
            cached = load_feature_cache(feature_path)
            if cached is None:
                continue
            cached_features, cached_labels, frame_index, player_id = cached
            if cached_features.ndim != 2 or cached_labels.ndim != 1:
                continue
            if cached_features.shape[0] != cached_labels.shape[0] or cached_features.shape[0] <= 0:
                continue
            feature_map[(label_name, color_key)] = (cached_features, cached_labels.astype(int))
            if info_source is None and cached_labels.shape[0] > 0:
                info_source = (cached_labels.astype(int), frame_index, player_id)

    if not feature_map:
        return []

    first_labels = next(iter(feature_map.values()))[1]
    unique_labels = sorted(set(int(val) for val in np.unique(first_labels)))
    if len(unique_labels) < 2:
        logging.warning("Sequence %s has fewer than two teams in cached labels; skipping", sequence_name)
        return []

    save_predictions = bool(getattr(args, "predictions_csv", None)) or bool(
        getattr(args, "structured_write_pipeline_predictions", False)
    )
    infos_tuple: tuple[CropInfo, ...] = ()
    if save_predictions and info_source is not None:
        y_true_info, frame_index, player_id = info_source
        infos: List[CropInfo] = []
        for idx, truth in enumerate(y_true_info.tolist()):
            frame = int(frame_index[idx]) if idx < int(frame_index.shape[0]) else -1
            pid = str(player_id[idx]) if idx < int(player_id.shape[0]) else str(idx)
            infos.append(
                CropInfo(
                    frame_index=frame,
                    player_id=pid,
                    team_label=int(truth),
                    crop_center_ratio=float(crop_center_ratio),
                    saved_path=None,
                    crop_method=crop_method,
                )
            )
        infos_tuple = tuple(infos)

    umap_random_state_raw = str(getattr(args, "umap_random_state", "0")).strip().lower()
    umap_random_state: Optional[int]
    if umap_random_state_raw in {"none", "null", ""}:
        umap_random_state = None
    else:
        umap_random_state = int(umap_random_state_raw)
    umap_n_jobs = int(getattr(args, "umap_n_jobs", -1))

    transformed_entries: List[
        tuple[str, str, int, bool, int, int, np.ndarray, np.ndarray]
    ] = []
    include_baseline = any(val is None for val in umap_range)
    positive_requested = [int(val) for val in umap_range if val is not None and int(val) > 0]
    strategy = str(getattr(args, "umap_components_strategy", "fit_each")).strip().lower()

    for (backend_label, color_space), (features, label_array) in feature_map.items():
        original_dim = int(features.shape[1] if features.ndim == 2 else 0)
        sample_count = int(features.shape[0] if features.ndim == 2 else 0)
        if sample_count <= 0:
            continue
        embedding_dir = resolve_structured_embedding_dir(
            structured_root,
            color_space=str(color_space),
            crop_method=crop_method,
            crop_center_ratio=float(crop_center_ratio),
            embedding_backend=backend_label,
        )

        if include_baseline:
            transformed_entries.append(
                (backend_label, str(color_space), original_dim, False, original_dim, 0, features, label_array)
            )

        if not positive_requested:
            continue

        if strategy == "fit_max_slice":
            max_req = max(positive_requested)
            effective_max = min(int(max_req), max(sample_count - 3, 2))
            if effective_max < 2 or effective_max >= sample_count - 1:
                continue
            valid_requested = sorted({int(req) for req in positive_requested if 2 <= int(req) <= effective_max})
            if not valid_requested:
                continue

            transformed_max: Optional[np.ndarray] = None
            max_file = embedding_dir / f"umap_{int(effective_max)}" / f"{sequence_stem}.npy"
            if not structured_force and max_file.is_file():
                try:
                    transformed_max = np.asarray(np.load(str(max_file)), dtype=np.float32)
                except Exception:
                    transformed_max = None

            missing_any = False
            if not structured_force:
                for requested in valid_requested:
                    out_file = embedding_dir / f"umap_{int(requested)}" / f"{sequence_stem}.npy"
                    if not out_file.is_file():
                        missing_any = True
                        break
            else:
                missing_any = True

            if transformed_max is None and missing_any:
                reducer = umap.UMAP(
                    n_components=int(effective_max),
                    n_neighbors=min(int(args.umap_neighbors), max(sample_count - 2, 2)),
                    min_dist=float(args.umap_min_dist),
                    metric=str(args.umap_metric),
                    random_state=umap_random_state,
                    n_jobs=umap_n_jobs,
                )
                transformed_max = reducer.fit_transform(features).astype(np.float32)
                atomic_save_npy(max_file, np.ascontiguousarray(transformed_max))

            for requested in valid_requested:
                effective_components = int(requested)
                out_file = embedding_dir / f"umap_{int(effective_components)}" / f"{sequence_stem}.npy"
                transformed_k: Optional[np.ndarray] = None
                if not structured_force and out_file.is_file():
                    try:
                        transformed_k = np.asarray(np.load(str(out_file)), dtype=np.float32)
                    except Exception:
                        transformed_k = None
                if transformed_k is None and transformed_max is not None:
                    transformed_k = np.ascontiguousarray(transformed_max[:, :effective_components]).astype(np.float32)
                    atomic_save_npy(out_file, transformed_k)
                if transformed_k is None:
                    if transformed_max is None:
                        reducer = umap.UMAP(
                            n_components=int(effective_components),
                            n_neighbors=min(int(args.umap_neighbors), max(sample_count - 2, 2)),
                            min_dist=float(args.umap_min_dist),
                            metric=str(args.umap_metric),
                            random_state=umap_random_state,
                            n_jobs=umap_n_jobs,
                        )
                        transformed_k = reducer.fit_transform(features).astype(np.float32)
                    else:
                        transformed_k = np.ascontiguousarray(transformed_max[:, :effective_components]).astype(np.float32)
                transformed_entries.append(
                    (
                        backend_label,
                        str(color_space),
                        int(effective_components),
                        True,
                        original_dim,
                        int(effective_max),
                        transformed_k,
                        label_array,
                    )
                )
        else:
            for requested in positive_requested:
                if sample_count <= requested:
                    continue
                effective_components = min(int(requested), max(sample_count - 3, 2))
                if effective_components < 2 or effective_components >= sample_count - 1:
                    continue
                out_file = embedding_dir / f"umap_{int(effective_components)}" / f"{sequence_stem}.npy"
                transformed: Optional[np.ndarray] = None
                if not structured_force and out_file.is_file():
                    try:
                        transformed = np.asarray(np.load(str(out_file)), dtype=np.float32)
                    except Exception:
                        transformed = None
                if transformed is None:
                    reducer = umap.UMAP(
                        n_components=int(effective_components),
                        n_neighbors=min(int(args.umap_neighbors), max(sample_count - 2, 2)),
                        min_dist=float(args.umap_min_dist),
                        metric=str(args.umap_metric),
                        random_state=umap_random_state,
                        n_jobs=umap_n_jobs,
                    )
                    transformed = reducer.fit_transform(features).astype(np.float32)
                    atomic_save_npy(out_file, np.ascontiguousarray(transformed))
                transformed_entries.append(
                    (
                        backend_label,
                        str(color_space),
                        int(effective_components),
                        True,
                        original_dim,
                        int(effective_components),
                        transformed,
                        label_array,
                    )
                )

    if not transformed_entries:
        return []

    structured_metric_buffers: Dict[Path, List[Dict[str, object]]] = {}
    structured_prediction_buffers: Dict[Path, List[Dict[str, object]]] = {}
    results: List[Dict[str, object]] = []

    futures = []
    max_workers = max(1, int(args.num_workers or 1))
    task_count = len(transformed_entries) * len(cluster_methods)
    max_workers = min(max_workers, max(1, task_count))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for (
            backend_label,
            color_space,
            recorded_components,
            umap_applied,
            original_dim,
            umap_fit_components,
            transformed,
            label_array,
        ) in transformed_entries:
            for cluster_method in cluster_methods:
                if structured_write_pipeline_metrics and not structured_force:
                    embedding_dir = resolve_structured_embedding_dir(
                        structured_root,
                        color_space=str(color_space),
                        crop_method=crop_method,
                        crop_center_ratio=float(crop_center_ratio),
                        embedding_backend=backend_label,
                    )
                    umap_dir = resolve_structured_umap_dir_path(
                        embedding_dir,
                        umap_applied=bool(umap_applied),
                        umap_components=int(recorded_components),
                    )
                    cluster_dir = umap_dir / structured_cluster_dir(str(cluster_method))
                    metrics_path = cluster_dir / "metrics.csv"
                    if sequence_name in _get_or_load_seen_sequences(metrics_path):
                        if emit_cached_rows:
                            cached_row = load_single_row_by_sequence(metrics_path, sequence_name)
                            if cached_row is not None:
                                results.append(cached_row)
                        continue

                futures.append(
                    executor.submit(
                        _run_single_configuration_from_transformed,
                        sequence_name,
                        str(color_space),
                        int(recorded_components),
                        transformed,
                        label_array,
                        unique_labels,
                        infos_tuple,
                        crop_center_ratio,
                        crop_method,
                        experiment_stage,
                        int(args.kmeans_init),
                        str(cluster_method),
                        {
                            "dbscan_eps": args.dbscan_eps,
                            "dbscan_min_samples": args.dbscan_min_samples,
                            "cmeans_m": args.cmeans_m,
                            "cmeans_max_iter": args.cmeans_max_iter,
                            "cmeans_tol": args.cmeans_tol,
                            "embedding_backend": backend_label,
                            "save_predictions": save_predictions,
                        },
                        umap_applied=bool(umap_applied),
                        original_dim=int(original_dim),
                        umap_fit_components=int(umap_fit_components),
                    )
                )

        for future in as_completed(futures):
            row, prediction_rows = future.result()
            if row is not None:
                results.append(row)
                if structured_write_pipeline_metrics:
                    embedding_dir = resolve_structured_embedding_dir(
                        structured_root,
                        color_space=str(row.get("color_space", "")),
                        crop_method=str(row.get("crop_method", "")),
                        crop_center_ratio=float(row.get("crop_center_ratio", crop_center_ratio)),
                        embedding_backend=str(row.get("embedding_backend", "")),
                    )
                    umap_dir = resolve_structured_umap_dir_path(
                        embedding_dir,
                        umap_applied=bool(row.get("umap_applied", False)),
                        umap_components=int(row.get("umap_components", 0)),
                    )
                    cluster_dir = umap_dir / structured_cluster_dir(str(row.get("cluster_method", "")))
                    metrics_path = cluster_dir / "metrics.csv"
                    structured_metric_buffers.setdefault(metrics_path, []).append(row)
            if prediction_rows:
                prediction_sink.extend(prediction_rows)
                if structured_write_pipeline_predictions and row is not None:
                    embedding_dir = resolve_structured_embedding_dir(
                        structured_root,
                        color_space=str(row.get("color_space", "")),
                        crop_method=str(row.get("crop_method", "")),
                        crop_center_ratio=float(row.get("crop_center_ratio", crop_center_ratio)),
                        embedding_backend=str(row.get("embedding_backend", "")),
                    )
                    umap_dir = resolve_structured_umap_dir_path(
                        embedding_dir,
                        umap_applied=bool(row.get("umap_applied", False)),
                        umap_components=int(row.get("umap_components", 0)),
                    )
                    cluster_dir = umap_dir / structured_cluster_dir(str(row.get("cluster_method", "")))
                    predictions_path = cluster_dir / "predictions.csv"
                    structured_prediction_buffers.setdefault(predictions_path, []).extend(prediction_rows)

    if structured_write_pipeline_metrics:
        for metrics_path, rows in structured_metric_buffers.items():
            append_rows_dedup_by_sequence(metrics_path, rows, force=structured_force)
    if structured_write_pipeline_predictions:
        for predictions_path, rows in structured_prediction_buffers.items():
            append_rows_no_dedup(predictions_path, rows, force=structured_force)

    return results


def run_experiment_for_sequence(
    sequence_name: str,
    crops: List[np.ndarray],
    labels: List[int],
    crop_infos: Sequence[CropInfo],
    color_spaces: Sequence[str],
    umap_range: Sequence[Optional[int]],
    args: argparse.Namespace,
    prediction_sink: List[Dict[str, object]],
    crop_center_ratio: float,
    crop_method: str,
    experiment_stage: str,
) -> List[Dict[str, object]]:
    if not crops or not labels:
        logging.warning("Sequence %s has no player crops; skipping", sequence_name)
        return []
    if len(crop_infos) != len(crops):
        logging.warning(
            "Sequence %s metadata length (%s) does not match crops (%s); predictions will omit metadata.",
            sequence_name,
            len(crop_infos),
            len(crops),
        )
    infos_list = list(crop_infos)
    max_len = min(len(crops), len(labels))
    valid_crops: List[np.ndarray] = []
    valid_labels: List[int] = []
    valid_infos: List[CropInfo] = []
    for idx in range(max_len):
        crop = crops[idx]
        if crop is None or crop.size == 0:
            continue
        if crop.ndim != 3 or crop.shape[2] != 3:
            continue
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            continue
        valid_crops.append(crop)
        valid_labels.append(int(labels[idx]))
        if idx < len(infos_list):
            valid_infos.append(infos_list[idx])

    crops = valid_crops
    labels = valid_labels
    infos_list = valid_infos
    if not crops or not labels:
        logging.warning("Sequence %s has no valid player crops after filtering; skipping", sequence_name)
        return []
    max_common = min(len(crops), len(labels), len(infos_list))
    crops = crops[:max_common]
    labels = labels[:max_common]
    infos_list = infos_list[:max_common]

    unique_labels = sorted(set(int(label) for label in labels))
    if len(unique_labels) < 2:
        logging.warning("Sequence %s has fewer than two teams; skipping", sequence_name)
        return []

    results: List[Dict[str, object]] = []
    sequence_stem = sanitize_identifier(sequence_name)
    structured_root_raw = getattr(args, "structured_results_root", None)
    structured_root = Path(structured_root_raw).expanduser().resolve() if structured_root_raw else None
    structured_force = bool(getattr(args, "structured_force", False))
    structured_write_pipeline_metrics = bool(getattr(args, "structured_write_pipeline_metrics", False))
    structured_write_pipeline_predictions = bool(getattr(args, "structured_write_pipeline_predictions", False))
    if structured_root is not None:
        structured_root.mkdir(parents=True, exist_ok=True)
    emit_cached_rows = True
    try:
        output_csv_path = Path(getattr(args, "output_csv")).expanduser().resolve()
        if output_csv_path.is_file() and output_csv_path.stat().st_size > 0:
            emit_cached_rows = False
    except Exception:
        emit_cached_rows = True

    embedding_backends_raw = parse_csv_list(args.embedding_backends) if args.embedding_backends else []
    embedding_backends = embedding_backends_raw or [str(args.embedding_backend)]
    resolved_backends = [normalize_embedding_backend(name) for name in embedding_backends]

    cluster_methods = parse_csv_list(args.cluster_methods) if args.cluster_methods else []
    cluster_methods = cluster_methods or [str(args.cluster_method)]

    feature_map: Dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for label_name, internal_name in resolved_backends:
        missing_colors: List[str] = []
        for color_space in color_spaces:
            color_key = str(color_space).strip().lower()
            if structured_root is None:
                missing_colors.append(color_key)
                continue
            embedding_dir = resolve_structured_embedding_dir(
                structured_root,
                color_space=color_key,
                crop_method=crop_method,
                crop_center_ratio=float(crop_center_ratio),
                embedding_backend=label_name,
            )
            feature_path = embedding_dir / f"{sequence_stem}.npz"
            if structured_force:
                missing_colors.append(color_key)
                continue
            cached = load_feature_cache(feature_path)
            if cached is None:
                missing_colors.append(color_key)
                continue
            cached_features, cached_labels, _, _ = cached
            if cached_features.ndim != 2 or cached_labels.ndim != 1:
                missing_colors.append(color_key)
                continue
            if cached_features.shape[0] != cached_labels.shape[0] or cached_features.shape[0] <= 0:
                missing_colors.append(color_key)
                continue
            feature_map[(label_name, color_key)] = (cached_features, cached_labels.astype(int))

        if not missing_colors:
            continue

        try:
            embedder = get_embedder(
                internal_name, resnet_batch_size=int(getattr(args, "resnet_batch_size", 64))
            )
            if hasattr(embedder, "embed"):
                embeddings = embedder.embed(crops)  # type: ignore[attr-defined]
            else:
                embeddings = embedder._extract_features(crops)  # type: ignore[attr-defined]
        except Exception as exc:
            logging.exception(
                "Failed to extract embeddings for sequence %s (backend=%s): %s",
                sequence_name,
                label_name,
                exc,
            )
            continue
        if embeddings.size == 0:
            logging.warning(
                "Sequence %s backend %s produced no embeddings; skipping",
                sequence_name,
                label_name,
            )
            continue

        for color_key in missing_colors:
            try:
                hist = compute_color_hist_features(
                    crops,
                    color_space=color_key,
                    bins=int(args.color_hist_bins),
                    weight=float(args.color_hist_weight),
                )
            except Exception as exc:
                logging.exception(
                    "Failed to compute hist for sequence %s (backend=%s, color=%s): %s",
                    sequence_name,
                    label_name,
                    color_key,
                    exc,
                )
                hist = np.empty((len(crops), 0), dtype=np.float32)

            feature_len = embeddings.shape[0]
            min_len = min(
                feature_len,
                len(labels),
                hist.shape[0] if hist.ndim == 2 else len(labels),
            )
            if min_len <= 0:
                continue
            if feature_len != min_len:
                logging.warning(
                    "Sequence %s backend %s produced %s embeddings for %s labels; truncating to %s",
                    sequence_name,
                    label_name,
                    feature_len,
                    len(labels),
                    min_len,
                )
            emb = np.ascontiguousarray(embeddings[:min_len]).astype(np.float32)
            hfeat = (
                np.ascontiguousarray(hist[:min_len]).astype(np.float32)
                if hist.size
                else np.empty((min_len, 0), dtype=np.float32)
            )
            features = np.concatenate([emb, hfeat], axis=1) if hfeat.size else emb
            label_array = np.array(labels[:min_len], dtype=int)
            if args.use_optical_flow_features:
                flow_feats = build_flow_features(infos_list)
                if flow_feats.shape[0] >= min_len:
                    flow_feats = flow_feats[:min_len]
                else:
                    flow_feats = np.pad(
                        flow_feats,
                        ((0, max(0, min_len - flow_feats.shape[0])), (0, 0)),
                        constant_values=0.0,
                    )
                features = np.concatenate([features, flow_feats.astype(np.float32)], axis=1)

            feature_map[(label_name, color_key)] = (features, label_array)
            if structured_root is not None:
                embedding_dir = resolve_structured_embedding_dir(
                    structured_root,
                    color_space=color_key,
                    crop_method=crop_method,
                    crop_center_ratio=float(crop_center_ratio),
                    embedding_backend=label_name,
                )
                feature_path = embedding_dir / f"{sequence_stem}.npz"
                frame_index = np.array(
                    [info.frame_index for info in infos_list[:min_len]],
                    dtype=np.int32,
                )
                player_id = np.array(
                    [info.player_id for info in infos_list[:min_len]],
                    dtype=str,
                )
                atomic_save_npz(
                    feature_path,
                    features=features.astype(np.float32),
                    labels=label_array.astype(np.int64),
                    frame_index=frame_index,
                    player_id=player_id,
                )

    if not feature_map:
        return results

    infos_tuple = tuple(infos_list)
    save_predictions = bool(getattr(args, "predictions_csv", None)) or bool(
        getattr(args, "structured_write_pipeline_predictions", False)
    )

    umap_random_state_raw = str(getattr(args, "umap_random_state", "0")).strip().lower()
    umap_random_state: Optional[int]
    if umap_random_state_raw in {"none", "null", ""}:
        umap_random_state = None
    else:
        umap_random_state = int(umap_random_state_raw)
    umap_n_jobs = int(getattr(args, "umap_n_jobs", -1))

    transformed_entries: List[
        tuple[str, str, int, bool, int, int, np.ndarray, np.ndarray]
    ] = []
    include_baseline = any(val is None for val in umap_range)
    positive_requested = [int(val) for val in umap_range if val is not None and int(val) > 0]
    strategy = str(getattr(args, "umap_components_strategy", "fit_each")).strip().lower()
    for (backend_label, color_space), (features, label_array) in feature_map.items():
        original_dim = int(features.shape[1] if features.ndim == 2 else 0)
        sample_count = int(features.shape[0] if features.ndim == 2 else 0)
        if sample_count <= 0:
            continue
        embedding_dir = None
        if structured_root is not None:
            embedding_dir = resolve_structured_embedding_dir(
                structured_root,
                color_space=str(color_space),
                crop_method=crop_method,
                crop_center_ratio=float(crop_center_ratio),
                embedding_backend=backend_label,
            )
        if include_baseline:
            transformed_entries.append(
                (backend_label, str(color_space), original_dim, False, original_dim, 0, features, label_array)
            )

        if not positive_requested:
            continue

        if strategy == "fit_max_slice":
            max_req = max(positive_requested)
            effective_max = min(int(max_req), max(sample_count - 3, 2))
            if effective_max < 2 or effective_max >= sample_count - 1:
                continue
            valid_requested = sorted({int(req) for req in positive_requested if 2 <= int(req) <= effective_max})
            if not valid_requested:
                continue

            transformed_max: Optional[np.ndarray] = None
            if embedding_dir is not None:
                max_file = embedding_dir / f"umap_{int(effective_max)}" / f"{sequence_stem}.npy"
                if not structured_force and max_file.is_file():
                    try:
                        transformed_max = np.asarray(np.load(str(max_file)), dtype=np.float32)
                    except Exception:
                        transformed_max = None

            missing_any = False
            if embedding_dir is not None and not structured_force:
                for requested in valid_requested:
                    out_file = embedding_dir / f"umap_{int(requested)}" / f"{sequence_stem}.npy"
                    if not out_file.is_file():
                        missing_any = True
                        break
            else:
                missing_any = True

            if transformed_max is None and missing_any:
                reducer = umap.UMAP(
                    n_components=int(effective_max),
                    n_neighbors=min(int(args.umap_neighbors), max(sample_count - 2, 2)),
                    min_dist=float(args.umap_min_dist),
                    metric=str(args.umap_metric),
                    random_state=umap_random_state,
                    n_jobs=umap_n_jobs,
                )
                transformed_max = reducer.fit_transform(features).astype(np.float32)
                if embedding_dir is not None:
                    atomic_save_npy(max_file, np.ascontiguousarray(transformed_max))

            for requested in valid_requested:
                effective_components = int(requested)
                transformed_k: Optional[np.ndarray] = None
                if embedding_dir is not None:
                    out_file = embedding_dir / f"umap_{int(effective_components)}" / f"{sequence_stem}.npy"
                    if not structured_force and out_file.is_file():
                        try:
                            transformed_k = np.asarray(np.load(str(out_file)), dtype=np.float32)
                        except Exception:
                            transformed_k = None
                    if transformed_k is None and transformed_max is not None:
                        transformed_k = np.ascontiguousarray(transformed_max[:, :effective_components]).astype(np.float32)
                        atomic_save_npy(out_file, transformed_k)
                if transformed_k is None:
                    if transformed_max is None:
                        reducer = umap.UMAP(
                            n_components=int(effective_components),
                            n_neighbors=min(int(args.umap_neighbors), max(sample_count - 2, 2)),
                            min_dist=float(args.umap_min_dist),
                            metric=str(args.umap_metric),
                            random_state=umap_random_state,
                            n_jobs=umap_n_jobs,
                        )
                        transformed_k = reducer.fit_transform(features).astype(np.float32)
                    else:
                        transformed_k = np.ascontiguousarray(transformed_max[:, :effective_components]).astype(np.float32)
                transformed_entries.append(
                    (
                        backend_label,
                        str(color_space),
                        int(effective_components),
                        True,
                        original_dim,
                        int(effective_max),
                        transformed_k,
                        label_array,
                    )
                )
        else:
            for requested in positive_requested:
                if sample_count <= requested:
                    continue
                effective_components = min(int(requested), max(sample_count - 3, 2))
                if effective_components < 2 or effective_components >= sample_count - 1:
                    continue
                transformed: Optional[np.ndarray] = None
                out_file = None
                if embedding_dir is not None:
                    out_file = embedding_dir / f"umap_{int(effective_components)}" / f"{sequence_stem}.npy"
                    if not structured_force and out_file.is_file():
                        try:
                            transformed = np.asarray(np.load(str(out_file)), dtype=np.float32)
                        except Exception:
                            transformed = None
                if transformed is None:
                    reducer = umap.UMAP(
                        n_components=int(effective_components),
                        n_neighbors=min(int(args.umap_neighbors), max(sample_count - 2, 2)),
                        min_dist=float(args.umap_min_dist),
                        metric=str(args.umap_metric),
                        random_state=umap_random_state,
                        n_jobs=umap_n_jobs,
                    )
                    transformed = reducer.fit_transform(features).astype(np.float32)
                    if out_file is not None:
                        atomic_save_npy(out_file, np.ascontiguousarray(transformed))
                transformed_entries.append(
                    (
                        backend_label,
                        str(color_space),
                        int(effective_components),
                        True,
                        original_dim,
                        int(effective_components),
                        transformed,
                        label_array,
                    )
                )

    structured_metric_buffers: Dict[Path, List[Dict[str, object]]] = {}
    structured_prediction_buffers: Dict[Path, List[Dict[str, object]]] = {}

    futures = []
    max_workers = max(1, int(args.num_workers or 1))
    task_count = len(transformed_entries) * len(cluster_methods)
    max_workers = min(max_workers, max(1, task_count))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for (
            backend_label,
            color_space,
            recorded_components,
            umap_applied,
            original_dim,
            umap_fit_components,
            transformed,
            label_array,
        ) in transformed_entries:
            for cluster_method in cluster_methods:
                if structured_root is not None and structured_write_pipeline_metrics and not structured_force:
                    embedding_dir = resolve_structured_embedding_dir(
                        structured_root,
                        color_space=str(color_space),
                        crop_method=crop_method,
                        crop_center_ratio=float(crop_center_ratio),
                        embedding_backend=backend_label,
                    )
                    umap_dir = resolve_structured_umap_dir_path(
                        embedding_dir,
                        umap_applied=bool(umap_applied),
                        umap_components=int(recorded_components),
                    )
                    cluster_dir = umap_dir / structured_cluster_dir(str(cluster_method))
                    metrics_path = cluster_dir / "metrics.csv"
                    if sequence_name in _get_or_load_seen_sequences(metrics_path):
                        if emit_cached_rows:
                            cached_row = load_single_row_by_sequence(metrics_path, sequence_name)
                            if cached_row is not None:
                                results.append(cached_row)
                        continue

                futures.append(
                    executor.submit(
                        _run_single_configuration_from_transformed,
                        sequence_name,
                        str(color_space),
                        int(recorded_components),
                        transformed,
                        label_array,
                        unique_labels,
                        infos_tuple,
                        crop_center_ratio,
                        crop_method,
                        experiment_stage,
                        int(args.kmeans_init),
                        str(cluster_method),
                        {
                            "dbscan_eps": args.dbscan_eps,
                            "dbscan_min_samples": args.dbscan_min_samples,
                            "cmeans_m": args.cmeans_m,
                            "cmeans_max_iter": args.cmeans_max_iter,
                            "cmeans_tol": args.cmeans_tol,
                            "embedding_backend": backend_label,
                            "save_predictions": save_predictions,
                        },
                        umap_applied=bool(umap_applied),
                        original_dim=int(original_dim),
                        umap_fit_components=int(umap_fit_components),
                    )
                )

        for future in as_completed(futures):
            row, prediction_rows = future.result()
            if row is not None:
                results.append(row)
                if structured_root is not None and structured_write_pipeline_metrics:
                    embedding_dir = resolve_structured_embedding_dir(
                        structured_root,
                        color_space=str(row.get("color_space", "")),
                        crop_method=str(row.get("crop_method", "")),
                        crop_center_ratio=float(row.get("crop_center_ratio", crop_center_ratio)),
                        embedding_backend=str(row.get("embedding_backend", "")),
                    )
                    umap_dir = resolve_structured_umap_dir_path(
                        embedding_dir,
                        umap_applied=bool(row.get("umap_applied", False)),
                        umap_components=int(row.get("umap_components", 0)),
                    )
                    cluster_dir = umap_dir / structured_cluster_dir(str(row.get("cluster_method", "")))
                    metrics_path = cluster_dir / "metrics.csv"
                    structured_metric_buffers.setdefault(metrics_path, []).append(row)
            if prediction_rows:
                prediction_sink.extend(prediction_rows)
                if structured_root is not None and structured_write_pipeline_predictions and row is not None:
                    embedding_dir = resolve_structured_embedding_dir(
                        structured_root,
                        color_space=str(row.get("color_space", "")),
                        crop_method=str(row.get("crop_method", "")),
                        crop_center_ratio=float(row.get("crop_center_ratio", crop_center_ratio)),
                        embedding_backend=str(row.get("embedding_backend", "")),
                    )
                    umap_dir = resolve_structured_umap_dir_path(
                        embedding_dir,
                        umap_applied=bool(row.get("umap_applied", False)),
                        umap_components=int(row.get("umap_components", 0)),
                    )
                    cluster_dir = umap_dir / structured_cluster_dir(str(row.get("cluster_method", "")))
                    predictions_path = cluster_dir / "predictions.csv"
                    structured_prediction_buffers.setdefault(predictions_path, []).extend(prediction_rows)

    if structured_root is not None and structured_write_pipeline_metrics:
        for metrics_path, rows in structured_metric_buffers.items():
            append_rows_dedup_by_sequence(metrics_path, rows, force=structured_force)
    if structured_root is not None and structured_write_pipeline_predictions:
        for predictions_path, rows in structured_prediction_buffers.items():
            append_rows_no_dedup(predictions_path, rows, force=structured_force)

    return results


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.dataset_root is None:
        raise ValueError("Dataset root must be provided via --dataset-root or config YAML.")
    if args.num_workers is None or args.num_workers <= 0:
        cpu_count = os.cpu_count() or 1
        args.num_workers = max(1, min(cpu_count, 4))
    else:
        args.num_workers = max(1, int(args.num_workers))
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence <= 0:
        args.max_frames_per_sequence = None
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    color_spaces = sorted(set(parse_color_spaces(args.color_spaces)))
    player_class_ids = parse_id_list(args.player_class_ids)
    team_map = parse_mapping(args.team_label_map)
    csv_team_ids = parse_id_list(args.csv_team_ids)
    save_crops_dir = None
    if not args.no_save_crops and args.save_crops_dir is not None:
        save_crops_dir = args.save_crops_dir.resolve()
        save_crops_dir.mkdir(parents=True, exist_ok=True)
    ratio_values = sorted({max(0.05, min(1.0, float(val))) for val in parse_float_list(args.crop_center_ratios) if val > 0.0})
    if not ratio_values:
        ratio_values = [1.0]
    if args.umap_max < args.umap_min:
        raise ValueError("--umap-max must be greater or equal to --umap-min")
    umap_step = max(1, int(getattr(args, "umap_step", 1)))
    umap_values_stage3 = list(range(args.umap_min, args.umap_max + 1, umap_step))
    if not umap_values_stage3:
        umap_values_stage3 = [args.umap_min]
    if args.include_no_umap:
        umap_values_stage3 = [None] + umap_values_stage3

    has_mot_sequences = dataset_has_mot_sequences(dataset_root)
    all_results: List[Dict[str, object]] = []
    all_predictions: List[Dict[str, object]] = []

    if args.grid_only:
        final_color_spaces = color_spaces or ["rgb"]
        sam2_cropper = init_sam2_cropper(args)
        methods_for_grid = [CROP_METHOD_CENTER]
        if not args.skip_opencv_mask:
            methods_for_grid.append(CROP_METHOD_OPENCV)
        if sam2_cropper is not None:
            methods_for_grid.append(CROP_METHOD_SAM2)

        log_banner(
            f"Grid stage: methods={','.join(methods_for_grid)} | ratios={ratio_values} | colors={','.join(final_color_spaces)}"
        )
        output_path = args.output_csv.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_parquet = (
            args.output_parquet.resolve()
            if getattr(args, "output_parquet", None) is not None
            else output_path.with_suffix(".parquet")
        )
        predictions_path = args.predictions_csv.resolve() if args.predictions_csv else None
        if predictions_path is not None:
            predictions_path.parent.mkdir(parents=True, exist_ok=True)

        wrote_any_metrics = output_path.exists() and output_path.stat().st_size > 0
        wrote_any_predictions = False

        completed_methods: set[str] = set()
        use_coarse_resume = (
            has_mot_sequences
            and wrote_any_metrics
            and not bool(getattr(args, "structured_results_root", None))
            and not bool(getattr(args, "structured_cache_only", False))
        )
        if use_coarse_resume:
            try:
                existing = pd.read_csv(output_path, usecols=["sequence", "crop_method"])
                expected_sequences = [entry.name for entry in sorted(dataset_root.iterdir()) if (entry / "seqinfo.ini").is_file()]
                if args.sequence_limit and args.sequence_limit > 0:
                    expected_sequences = expected_sequences[: int(args.sequence_limit)]
                expected_count = len(set(expected_sequences))
                for method in methods_for_grid:
                    seen = int(existing[existing["crop_method"] == method]["sequence"].nunique())
                    if expected_count > 0 and seen >= expected_count:
                        completed_methods.add(method)
                if completed_methods:
                    logging.info("Resume mode: detected completed methods=%s", ",".join(sorted(completed_methods)))
            except Exception:
                completed_methods = set()

        center_handled = False
        # Center crops: for MOT sequences process all ratios in a single pass over frames to avoid re-reading the sequence 5x.
        if (
            CROP_METHOD_CENTER in methods_for_grid
            and has_mot_sequences
            and CROP_METHOD_CENTER not in completed_methods
            and not bool(getattr(args, "structured_cache_only", False))
        ):
            sequences = [entry for entry in sorted(dataset_root.iterdir()) if (entry / "seqinfo.ini").is_file()]
            if args.sequence_limit and args.sequence_limit > 0:
                sequences = sequences[: int(args.sequence_limit)]
            iterator = tqdm(sequences, desc=f"{STAGE_COLOR_UMAP} | center | multi-ratio", unit="seq", leave=False) if sequences else []
            for seq_dir in iterator:
                try:
                    seq_meta = read_sequence_metadata(seq_dir)
                    gt_df = load_ground_truth(seq_dir)
                    class_col = resolve_column_name(gt_df, args.class_column_name, args.class_column_index)
                    team_col = resolve_column_name(gt_df, args.team_column_name, args.team_column_index)
                except Exception as exc:
                    logging.exception("Skipping sequence %s due to error: %s", seq_dir.name, exc)
                    continue

                track_team_map = None
                if str(getattr(args, "mot_team_source", "gt_column")).lower() == "gameinfo":
                    track_team_map = parse_soccernet_gameinfo_team_map(seq_dir)
                    if not track_team_map:
                        logging.warning(
                            "MOT team source is gameinfo but mapping is empty for %s; skipping.",
                            seq_dir.name,
                        )
                        continue
                frame_indices = compute_frame_indices(seq_meta, args.sample_seconds, args.max_frames_per_sequence)
                try:
                    ratio_to_data = extract_player_crops_center_multi_ratio(
                        seq_meta,
                        gt_df,
                        frame_indices,
                        class_col,
                        team_col,
                        player_class_ids,
                        team_map,
                        args,
                        ratio_values,
                        track_team_map=track_team_map,
                    )
                except Exception as exc:
                    logging.exception("Extraction failed for %s during grid center multi-ratio: %s", seq_dir.name, exc)
                    continue

                for ratio in ratio_values:
                    crops, labels, infos = ratio_to_data.get(float(ratio), ([], [], []))
                    if not crops:
                        continue
                    task_predictions: List[Dict[str, object]] = []
                    seq_results = run_experiment_for_sequence(
                        seq_meta.name,
                        crops,
                        labels,
                        infos,
                        final_color_spaces,
                        umap_values_stage3,
                        args,
                        task_predictions,
                        float(ratio),
                        CROP_METHOD_CENTER,
                        STAGE_COLOR_UMAP,
                    )
                    if seq_results:
                        pd.DataFrame(seq_results).to_csv(
                            output_path,
                            index=False,
                            mode="a" if wrote_any_metrics else "w",
                            header=not wrote_any_metrics,
                        )
                        wrote_any_metrics = True
                    if predictions_path is not None and task_predictions:
                        pd.DataFrame(task_predictions).to_csv(
                            predictions_path,
                            index=False,
                            mode="a" if wrote_any_predictions else "w",
                            header=not wrote_any_predictions,
                        )
                        wrote_any_predictions = True
            center_handled = True

        # Other methods (OpenCV / SAM2): single-ratio runs.
        tasks: List[Tuple[str, float]] = []
        if (
            CROP_METHOD_CENTER in methods_for_grid
            and not center_handled
            and CROP_METHOD_CENTER not in completed_methods
        ):
            tasks.extend((CROP_METHOD_CENTER, ratio) for ratio in ratio_values)
        other_methods = [m for m in methods_for_grid if m != CROP_METHOD_CENTER and m not in completed_methods]
        tasks.extend((method, 1.0) for method in other_methods)
        for method, ratio in progress(tasks, desc="Grid method/ratio"):
            task_predictions: List[Dict[str, object]] = []
            stage_results = run_stage_experiments(
                dataset_root,
                args,
                color_spaces=final_color_spaces,
                umap_values=umap_values_stage3,
                crop_center_ratio=ratio,
                crop_method=method,
                experiment_stage=STAGE_COLOR_UMAP,
                prediction_sink=task_predictions,
                team_map=team_map,
                player_class_ids=player_class_ids,
                csv_team_ids=csv_team_ids,
                has_mot_sequences=has_mot_sequences,
                sam2_cropper=sam2_cropper if method == CROP_METHOD_SAM2 else None,
                save_crops_dir=save_crops_dir,
            )
            if stage_results:
                pd.DataFrame(stage_results).to_csv(
                    output_path,
                    index=False,
                    mode="a" if wrote_any_metrics else "w",
                    header=not wrote_any_metrics,
                )
                wrote_any_metrics = True

            if predictions_path is not None and task_predictions:
                pd.DataFrame(task_predictions).to_csv(
                    predictions_path,
                    index=False,
                    mode="a" if wrote_any_predictions else "w",
                    header=not wrote_any_predictions,
                )
                wrote_any_predictions = True

        if not wrote_any_metrics:
            logging.error("No results generated; please check dataset paths and parameters.")
            return
        logging.info("Saved metrics to %s", output_path)
        try:
            write_parquet_from_csv(output_path, output_parquet)
            logging.info("Saved metrics parquet to %s", output_parquet)
        except Exception as exc:
            logging.exception("Failed to write parquet metrics to %s: %s", output_parquet, exc)
        if predictions_path is not None and wrote_any_predictions:
            logging.info("Saved per-crop predictions to %s", predictions_path)
        return

    log_banner(
        f"Stage 1: Center crops | ratios={ratio_values} | colors={','.join(color_spaces)}"
    )
    for ratio in progress(ratio_values, desc="Stage1 ratios"):
        stage_results = run_stage_experiments(
            dataset_root,
            args,
            color_spaces=color_spaces,
            umap_values=[None],
            crop_center_ratio=ratio,
            crop_method=CROP_METHOD_CENTER,
            experiment_stage=STAGE_RATIO_SWEEP,
            prediction_sink=all_predictions,
            team_map=team_map,
            player_class_ids=player_class_ids,
            csv_team_ids=csv_team_ids,
            has_mot_sequences=has_mot_sequences,
            sam2_cropper=None,
            save_crops_dir=save_crops_dir,
        )
        all_results.extend(stage_results)
        for color in color_spaces:
            color_rows = [row for row in stage_results if row.get("color_space") == color]
            stage_accuracy = compute_weighted_accuracy(color_rows)
            logging.info(
                "Center | ratio=%.3f accuracy=%.4f (%s rows) | color=%s",
                ratio,
                stage_accuracy,
                len(color_rows),
                color,
            )

    if not args.skip_opencv_mask:
        log_banner(
            f"Stage 2: OpenCV GrabCut (ratio=1.0) | colors={','.join(color_spaces)}"
        )
        ocv_ratio_values = [1.0]
        for ratio in ocv_ratio_values:
            ocv_results = run_stage_experiments(
                dataset_root,
                args,
                color_spaces=color_spaces,
                umap_values=[None],
                crop_center_ratio=ratio,
                crop_method=CROP_METHOD_OPENCV,
                experiment_stage=STAGE_OPENCV_COMPARISON,
                prediction_sink=all_predictions,
                team_map=team_map,
                player_class_ids=player_class_ids,
                csv_team_ids=csv_team_ids,
                has_mot_sequences=has_mot_sequences,
                sam2_cropper=None,
                save_crops_dir=save_crops_dir,
            )
            all_results.extend(ocv_results)
            for color in color_spaces:
                color_rows = [row for row in ocv_results if row.get("color_space") == color]
                stage_accuracy = compute_weighted_accuracy(color_rows)
                logging.info(
                    "OpenCV | ratio=%.3f accuracy=%.4f (%s rows) | color=%s",
                    ratio,
                    stage_accuracy,
                    len(color_rows),
                    color,
                )
    else:
        logging.info("OpenCV mask stage disabled via --skip-opencv-mask.")

    sam2_cropper = init_sam2_cropper(args)
    if sam2_cropper is not None:
        log_banner(
            f"Stage 3: SAM2 masks (ratio=1.0) | colors={','.join(color_spaces)}"
        )
        sam2_ratio_values = [1.0]
        for ratio in sam2_ratio_values:
            sam2_results = run_stage_experiments(
                dataset_root,
                args,
                color_spaces=color_spaces,
                umap_values=[None],
                crop_center_ratio=ratio,
                crop_method=CROP_METHOD_SAM2,
                experiment_stage=STAGE_SAM2_COMPARISON,
                prediction_sink=all_predictions,
                team_map=team_map,
                player_class_ids=player_class_ids,
                csv_team_ids=csv_team_ids,
                has_mot_sequences=has_mot_sequences,
                sam2_cropper=sam2_cropper,
                save_crops_dir=save_crops_dir,
            )
            all_results.extend(sam2_results)
            for color in color_spaces:
                color_rows = [row for row in sam2_results if row.get("color_space") == color]
                stage_accuracy = compute_weighted_accuracy(color_rows)
                logging.info(
                    "SAM2 | ratio=%.3f accuracy=%.4f (%s rows) | color=%s",
                    ratio,
                    stage_accuracy,
                    len(color_rows),
                    color,
                )
    else:
        logging.info("SAM2 cropper unavailable; skipping SAM2 comparison stage.")

    final_color_spaces = color_spaces or ["rgb"]
    methods_for_umap = [CROP_METHOD_CENTER]
    if not args.skip_opencv_mask:
        methods_for_umap.append(CROP_METHOD_OPENCV)
    if sam2_cropper is not None:
        methods_for_umap.append(CROP_METHOD_SAM2)

    logging.info("Stage 4: colour/UMAP grid per colour space and crop method (independent)")
    tasks: List[Tuple[str, float]] = []
    for method in methods_for_umap:
        ratio_list = ratio_values if method == CROP_METHOD_CENTER else [1.0]
        for ratio in ratio_list:
            tasks.append((method, ratio))

    log_banner(
        f"Stage 4: UMAP grid | colors={','.join(final_color_spaces)} | umap={args.umap_min}-{args.umap_max}:{umap_step}"
    )
    for method, ratio in progress(tasks, desc="Stage4 method/ratio"):
        logging.info(
            "Stage 4 config | method=%s ratio=%.3f umap=%s-%s",
            method,
            ratio,
            args.umap_min,
            args.umap_max,
        )
        stage3_results = run_stage_experiments(
            dataset_root,
            args,
            color_spaces=final_color_spaces,
            umap_values=umap_values_stage3,
            crop_center_ratio=ratio,
            crop_method=method,
            experiment_stage=STAGE_COLOR_UMAP,
            prediction_sink=all_predictions,
            team_map=team_map,
            player_class_ids=player_class_ids,
            csv_team_ids=csv_team_ids,
            has_mot_sequences=has_mot_sequences,
            sam2_cropper=sam2_cropper if method == CROP_METHOD_SAM2 else None,
            save_crops_dir=save_crops_dir,
        )
        all_results.extend(stage3_results)

    if not all_results:
        logging.error("No results generated; please check dataset paths and parameters.")
        return
    output_path = args.output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    logging.info("Saved metrics to %s", output_path)
    output_parquet = (
        args.output_parquet.resolve()
        if getattr(args, "output_parquet", None) is not None
        else output_path.with_suffix(".parquet")
    )
    try:
        write_parquet_from_csv(output_path, output_parquet)
        logging.info("Saved metrics parquet to %s", output_parquet)
    except Exception as exc:
        logging.exception("Failed to write parquet metrics to %s: %s", output_parquet, exc)

    if all_predictions:
        predictions_path = args.predictions_csv.resolve()
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_predictions).to_csv(predictions_path, index=False)
        logging.info("Saved per-crop predictions to %s", predictions_path)


if __name__ == "__main__":
    main()
