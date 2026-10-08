"""SAM2 mask tracking (segment-anything-2-real-time camera predictor) with detector re-seeding.

SAM2 follows the objects it was prompted with but never discovers new ones, so
the detector keeps feeding it: detections that overlap no tracked mask become
*pending candidates*; a candidate seen ``candidate_min_hits`` times is promoted,
which re-prompts SAM2 on the current frame with all tracked boxes plus the new
ones (at most once every ``reseed_interval`` frames).

The SAM2 code is not pip-installable; it is imported from a local checkout of
https://github.com/Gy920/segment-anything-2-real-time (``external/`` in this
repo), which needs ``hydra-core`` and ``iopath`` installed.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.tracking.base import TRACKERS, Tracker
from tactifoot_vision.utils import resolve_device

logger = logging.getLogger(__name__)

# Moving average weight of a new detection box in a pending candidate's box.
_CANDIDATE_MOMENTUM = 0.4


@dataclass
class _Candidate:
    """An untracked detection waiting for enough hits to be added to SAM2."""

    box: np.ndarray
    class_id: int
    class_name: str
    last_seen: int
    hits: int = 1

    def update(self, box: np.ndarray, frame: int) -> None:
        self.box = (
            (1 - _CANDIDATE_MOMENTUM) * self.box + _CANDIDATE_MOMENTUM * box
        ).astype(np.float32)
        self.last_seen = frame
        self.hits += 1


@TRACKERS.register("sam2")
class SAM2Tracker(Tracker):
    """Segmentation-mask tracking with SAM2, prompted and re-seeded from detections.

    The first :meth:`update` prompts SAM2 with every detection. Output
    detections carry masks in ``mask``, boxes derived from the masks, and the
    ``class_id`` / ``data["class_name"]`` of the detection that seeded each
    track (other ``data`` keys are not kept). Objects whose mask is empty in
    a frame are left out of that frame's output. A re-seed prompts SAM2 with
    the tracks visible on that frame only, so a track whose mask is empty on
    the re-seed frame loses its id; when the detector sees it again it comes
    back as a new track.

    Args:
        checkpoint: SAM2 checkpoint, e.g. ``external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt``.
        config: model config file inside the checkout's ``sam2`` package, e.g.
            ``external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_t.yaml``,
            or its name relative to that package (``configs/sam2.1/sam2.1_hiera_t.yaml``)
            together with ``repo_dir``.
        repo_dir: the segment-anything-2-real-time checkout; found from ``config`` when omitted.
        mask_filter_distance: drop mask fragments whose centroid is further than
            this many pixels from the largest fragment (0 disables).
        reseed_interval: minimum number of frames between two re-seeds.
        reseed_iou_threshold: a detection overlapping a tracked box by at least
            this IoU is considered tracked.
        candidate_min_hits: detections needed before a pending candidate is added.
        device: torch device; ``None`` picks CUDA when available.
    """

    name = "sam2"

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path,
        repo_dir: str | Path | None = None,
        mask_filter_distance: float = 300.0,
        reseed_interval: int = 30,
        reseed_iou_threshold: float = 0.3,
        candidate_min_hits: int = 3,
        device: str | None = None,
    ) -> None:
        if reseed_interval < 0:
            raise ValueError("reseed_interval must be >= 0")
        if candidate_min_hits < 1:
            raise ValueError("candidate_min_hits must be >= 1")
        self.mask_filter_distance = mask_filter_distance
        self.reseed_interval = reseed_interval
        self.reseed_iou_threshold = reseed_iou_threshold
        self.candidate_min_hits = candidate_min_hits
        # Pending candidates merge with a new detection at a lower IoU than tracks do,
        # and expire when not seen for a while.
        self._candidate_merge_iou = min(0.5, 0.75 * reseed_iou_threshold)
        self._candidate_timeout = max(reseed_interval, candidate_min_hits + 2)
        self.device = resolve_device(device)
        self._predictor = _build_predictor(checkpoint, config, repo_dir, self.device)
        self.reset()

    def reset(self, fps: float | None = None) -> None:
        self._frame = 0  # frames seen since the last reset
        self._initialized = False
        self._tracks: dict[int, tuple[int, str]] = {}  # track id -> (class id, name)
        self._next_id = 1
        self._candidates: list[_Candidate] = []
        self._last_reseed: int | None = None

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # SAM2 expects RGB
        if not self._initialized:
            self._initialized = True
            class_ids, names = _classes(detections)
            ids = self._allocate_ids(len(detections))
            tracked = self._prompt(rgb, detections.xyxy, ids, class_ids, names)
        else:
            tracked = self._reseed(rgb, detections, self._track(rgb))
        self._frame += 1
        return tracked

    # ------------------------------------------------------------- SAM2 calls
    def _prompt(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        ids: np.ndarray,
        class_ids: np.ndarray,
        names: np.ndarray,
    ) -> sv.Detections:
        """Restart SAM2 on the RGB ``frame`` with one box prompt per object."""
        import torch

        # load_first_frame rebuilds the predictor state; its frame counter must restart too,
        # otherwise memory frames get temporal offsets relative to the old session.
        self._predictor.load_first_frame(frame)
        self._predictor.frame_idx = 0
        self._tracks = {}
        with torch.inference_mode(), self._autocast():
            for box, track_id, class_id, name in zip(
                boxes, ids, class_ids, names, strict=True
            ):
                self._predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(track_id),
                    bbox=np.asarray(box, dtype=np.float32)[None],
                )
                self._tracks[int(track_id)] = (int(class_id), str(name))
        return self._track(frame)

    def _track(self, frame: np.ndarray) -> sv.Detections:
        if not self._tracks:
            return _empty()
        import torch

        with torch.inference_mode(), self._autocast():
            obj_ids, mask_logits = self._predictor.track(frame)
        ids = np.asarray(obj_ids, dtype=int)
        masks = (mask_logits[:, 0] > 0.0).cpu().numpy()
        xyxy, visible = _mask_boxes(masks, self.mask_filter_distance)
        if not visible.any():
            return _empty()
        ids = ids[visible]
        return sv.Detections(
            xyxy=xyxy[visible],
            mask=masks[visible],
            confidence=np.ones(len(ids), dtype=np.float32),
            class_id=np.array([self._tracks[t][0] for t in ids], dtype=int),
            tracker_id=ids,
            data={"class_name": np.array([self._tracks[t][1] for t in ids])},
        )

    def _autocast(self) -> Any:
        import contextlib

        import torch

        if self.device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------- re-seeding
    def _reseed(
        self, frame: np.ndarray, detections: sv.Detections, tracked: sv.Detections
    ) -> sv.Detections:
        """Collect untracked detections; re-prompt SAM2 (on RGB ``frame``) when candidates are confirmed."""
        self._collect_candidates(detections, tracked)
        cooling = (
            self._last_reseed is not None
            and self._frame - self._last_reseed < self.reseed_interval
        )
        ready = [
            c
            for c in self._candidates
            if c.hits >= self.candidate_min_hits and not self._overlaps(c.box, tracked)
        ]
        if cooling or not ready:
            return tracked
        self._candidates = [
            c for c in self._candidates if not any(c is r for r in ready)
        ]
        self._last_reseed = self._frame
        logger.debug(
            "Frame %d: adding %d objects to %d SAM2 tracks",
            self._frame,
            len(ready),
            len(tracked),
        )
        tracked_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        ids = np.concatenate(
            [np.asarray(tracked_ids, dtype=int), self._allocate_ids(len(ready))]
        )
        known = [self._tracks[int(t)] for t in tracked_ids]
        boxes = np.concatenate(
            [tracked.xyxy.reshape(-1, 4), np.stack([c.box for c in ready])]
        )
        class_ids = np.array([k[0] for k in known] + [c.class_id for c in ready])
        names = np.array([k[1] for k in known] + [c.class_name for c in ready])
        return self._prompt(frame, boxes, ids, class_ids, names)

    def _collect_candidates(
        self, detections: sv.Detections, tracked: sv.Detections
    ) -> None:
        untracked = np.ones(len(detections), dtype=bool)
        if len(detections) and len(tracked):
            overlap = sv.box_iou_batch(detections.xyxy, tracked.xyxy).max(axis=1)
            untracked = overlap < self.reseed_iou_threshold
        class_ids, names = _classes(detections)
        for i in np.flatnonzero(untracked):
            box = detections.xyxy[i].astype(np.float32)
            best, best_iou = None, 0.0
            if self._candidates:
                ious = sv.box_iou_batch(
                    box[None], np.stack([c.box for c in self._candidates])
                )[0]
                best, best_iou = int(ious.argmax()), float(ious.max())
            if best is not None and best_iou >= self._candidate_merge_iou:
                self._candidates[best].update(box, self._frame)
            else:
                self._candidates.append(
                    _Candidate(box, int(class_ids[i]), str(names[i]), self._frame)
                )
        # A candidate on top of a tracked box is that track seen again (e.g. after
        # its mask was empty for a few frames), not a new object.
        self._candidates = [
            c
            for c in self._candidates
            if self._frame - c.last_seen < self._candidate_timeout
            and not self._overlaps(c.box, tracked)
        ]

    def _overlaps(self, box: np.ndarray, tracked: sv.Detections) -> bool:
        """Whether ``box`` overlaps a tracked box by at least ``reseed_iou_threshold``."""
        if not len(tracked):
            return False
        iou = sv.box_iou_batch(np.asarray(box, dtype=np.float32)[None], tracked.xyxy)
        return bool(iou.max() >= self.reseed_iou_threshold)

    def _allocate_ids(self, count: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + count, dtype=int)
        self._next_id += count
        return ids


def _build_predictor(
    checkpoint: str | Path, config: str | Path, repo_dir: str | Path | None, device: str
) -> Any:
    """Import SAM2 from the local checkout and build its camera predictor."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    repo, config_name = _resolve_config(Path(config), repo_dir)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from sam2.build_sam import build_sam2_camera_predictor
    except ImportError as exc:
        raise ImportError(
            f"Cannot import SAM2 from {repo} ({exc}). The SAM2 tracker needs the "
            "segment-anything-2-real-time checkout plus `pip install hydra-core iopath`."
        ) from exc
    logger.info("Loading SAM2 %s from %s on %s", config_name, checkpoint, device)
    return build_sam2_camera_predictor(
        config_file=config_name, ckpt_path=str(checkpoint), device=device
    )


def _resolve_config(config: Path, repo_dir: str | Path | None) -> tuple[Path, str]:
    """Return the checkout root and the hydra config name (relative to its ``sam2`` package)."""
    if repo_dir is None:
        repo = next(
            (
                p
                for p in config.resolve().parents
                if (p / "sam2" / "build_sam.py").is_file()
            ),
            None,
        )
        if repo is None:
            raise ValueError(
                f"No segment-anything-2-real-time checkout found above {config}; pass repo_dir"
            )
    else:
        repo = Path(repo_dir).resolve()
        if not (repo / "sam2" / "build_sam.py").is_file():
            raise FileNotFoundError(
                f"{repo} is not a segment-anything-2-real-time checkout (no sam2/build_sam.py)"
            )
    package = repo / "sam2"
    path = config.resolve() if config.is_file() else package / config
    if not path.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {config}")
    try:
        name = path.relative_to(package).as_posix()
    except ValueError:
        raise ValueError(
            f"SAM2 configs must live inside {package} (hydra loads them from there), got {path}"
        ) from None
    return repo, name


def _mask_boxes(
    masks: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (``sv.mask_to_xyxy`` convention) of ``(N, H, W)`` masks and which are non-empty.

    Fragments further than ``max_distance`` pixels from a mask's largest
    fragment are erased from ``masks`` in place first (0 disables). Work is
    limited to each mask's bounding region, as full-frame masks are mostly empty.
    """
    xyxy = np.zeros((len(masks), 4), dtype=np.float32)
    rows, cols = masks.any(axis=2), masks.any(axis=1)
    visible = rows.any(axis=1)
    for i in np.flatnonzero(visible):
        y, x = np.flatnonzero(rows[i]), np.flatnonzero(cols[i])
        y1, y2, x1, x2 = y[0], y[-1] + 1, x[0], x[-1] + 1
        if max_distance > 0:
            region = masks[i, y1:y2, x1:x2]
            region[:] = _filter_segments(region, max_distance)
            y, x = (
                np.flatnonzero(region.any(axis=1)),
                np.flatnonzero(region.any(axis=0)),
            )
            y1, y2, x1, x2 = y1 + y[0], y1 + y[-1] + 1, x1 + x[0], x1 + x[-1] + 1
        xyxy[i] = x1, y1, x2 - 1, y2 - 1
    return xyxy, visible


def _filter_segments(mask: np.ndarray, max_distance: float) -> np.ndarray:
    """Keep the largest connected fragment and fragments whose centroid is near it."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 2:  # background plus at most one fragment
        return mask
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    distances = np.linalg.norm(centroids - centroids[main], axis=1)
    keep = distances <= max_distance
    keep[0] = False  # background
    return keep[labels]


def _classes(detections: sv.Detections) -> tuple[np.ndarray, np.ndarray]:
    n = len(detections)
    class_ids = (
        detections.class_id
        if detections.class_id is not None
        else np.full(n, -1, dtype=int)
    )
    names = detections.data.get("class_name", np.full(n, "", dtype=object))
    return np.asarray(class_ids), np.asarray(names)


def _empty() -> sv.Detections:
    empty = sv.Detections.empty()
    empty.tracker_id = np.zeros(0, dtype=int)
    empty.data = {"class_name": np.zeros(0, dtype=str)}
    return empty
