"""What the GSR evaluation runs: a keypoint model, homography methods on its keypoints, one clip at a time
(pitch-keypoints 02; ticket 03 scores its candidates with it).

A `KeypointModel` is a weights file, its input (`stretch`: the frame resized to the training size, ignoring the aspect
ratio; `letterbox`: Ultralytics' default, padded to a multiple of the stride; `square`: padded to a square canvas as in
Ultralytics training, `SquareInput`), an optional inference size and the instance threshold. A `Method` turns one
frame's keypoints into a frame -> pitch homography with `HomographyEstimator.fit`: its keypoint threshold, RANSAC limit,
line refinement on or off, and where the people boxes removed from the refinement's line mask come from (`detector`:
the package detector at xg-detector's C4/D3 settings, the reference; `gt`: GSR's boxes, a diagnostic; `none`). The
method with `source="label"` is the oracle: ticket 01's label homography (a line fit to the annotated lines) where the
label was kept.

`baseline_methods` are the current model's: D1-D4 of `research/xg_detector_audit2_configs.json` (D3 and D4 with the
detector mask, as declared), `D3_gtmask` / `D4_gtmask`, and `label`. `load_methods` reads any other set from JSON:
`{"name": {"min_confidence": 0.5, "refine_with_lines": true, "mask": "detector", "ransac_threshold": 10.0}, ...}`.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from xgpoc.detector import homography_dev as dev
from xgpoc.detector import lines
from xgpoc.pitchkp import gsr_eval as ge

INPUTS = ("stretch", "letterbox", "square")
PAD_VALUE = 114                             # Ultralytics' letterbox grey
BACKENDS = ("yolo_pose", "heatmap")
MASKS = ("detector", "gt", "none")
SOURCES = ("keypoints", "label")


@dataclass(frozen=True)
class KeypointModel:
    """A pitch keypoint model as the evaluation runs it: weights, input mode, inference size, instance threshold, and
    its backend: the package's `yolo_pose`, or ticket 03's research `heatmap` model (`xgpoc.pitchkp.heatmap`; its input
    is fixed, so `input` and `imgsz` do not apply and must stay at "letterbox" / None)."""

    weights: Path
    input: str = "stretch"
    imgsz: int | None = None                # None: the training size
    instance_conf: float = dev.INSTANCE_CONF
    backend: str = "yolo_pose"

    def __post_init__(self) -> None:
        if self.input not in INPUTS:
            raise ValueError(f"input must be one of {INPUTS}, got {self.input!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.backend == "heatmap" and (self.input != "letterbox" or self.imgsz is not None):
            raise ValueError("the heatmap model has a fixed 960 x 544 input: input='letterbox', imgsz=None")

    def load(self, device: str | None = None):
        if self.backend == "heatmap":
            from xgpoc.pitchkp.heatmap import HeatmapKeypointModel

            return HeatmapKeypointModel(self.weights, device)
        import tactifoot_vision as tv

        model = tv.load_model("yolo_pose", self.weights, conf=self.instance_conf, imgsz=self.imgsz,
                              stretch=self.input == "stretch", device=device)
        return SquareInput(model) if self.input == "square" else model

    def settings(self) -> dict:
        settings = {"weights": str(self.weights), "input": self.input, "imgsz": self.imgsz,
                    "instance_conf": self.instance_conf}
        # the backend is named only when it is not yolo_pose, so ticket 02's caches keep their identity
        return settings if self.backend == "yolo_pose" else settings | {"backend": self.backend}


class SquareInput:
    """A keypoint model fed the frame padded to a square (centred, grey `PAD_VALUE`), as Ultralytics' training
    letterboxes every image; its outputs are moved back to frame pixels. The package's `stretch=False` input pads only
    to a multiple of the stride (960 x 544 for a 16:9 frame), which a model trained on square canvases never saw:
    ticket 03's y11m_960 found the instance on 20% of GSR valid images that way and on 100% with the square canvas."""

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, image: np.ndarray) -> sv.KeyPoints:
        height, width = image.shape[:2]
        side = max(height, width)
        top, left = (side - height) // 2, (side - width) // 2
        canvas = np.full((side, side, 3), PAD_VALUE, image.dtype)
        canvas[top:top + height, left:left + width] = image
        found = self.model(canvas)
        if len(found):
            found.xy = found.xy - np.float32([left, top])
            found.data["xyxy"] = found.data["xyxy"] - np.float32([left, top, left, top])
        return found


@dataclass(frozen=True)
class Method:
    """How one frame's keypoints become a homography (`source="keypoints"`), or the label oracle (`"label"`)."""

    name: str
    source: str = "keypoints"
    min_confidence: float = 0.5
    ransac_threshold: float = 10.0
    refine_with_lines: bool = False
    mask: str = "none"

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"{self.name}: source must be one of {SOURCES}, got {self.source!r}")
        if self.mask not in MASKS:
            raise ValueError(f"{self.name}: mask must be one of {MASKS}, got {self.mask!r}")
        if self.source == "label" and self.refine_with_lines:
            raise ValueError(f"{self.name}: the label homography is not refined")

    def estimator(self):
        from tactifoot_vision.pitch import HomographyEstimator

        return HomographyEstimator(min_confidence=self.min_confidence, ransac_threshold=self.ransac_threshold,
                                   refine_with_lines=self.refine_with_lines)

    def settings(self) -> dict:
        return dataclasses.asdict(self)


def baseline_methods(configs_path: Path) -> dict[str, Method]:
    """D1-D4 of the configs file (D3/D4 with the detector mask), D3_gtmask, D4_gtmask and the label oracle."""
    configs = json.loads(Path(configs_path).read_text())
    ransac = configs["common"]["homography"]["ransac_threshold_m"]
    methods = {}
    for name, c in configs["configs"].items():
        if c["keypoint_input"] != "stretch":
            raise ValueError(f"{name}: the baseline evaluates the stretched keypoint input, got {c['keypoint_input']}")
        methods[name] = Method(name, min_confidence=c["keypoint_conf"], ransac_threshold=ransac,
                               refine_with_lines=c["refine_with_lines"],
                               mask="detector" if c["refine_with_lines"] else "none")
    for name in [n for n, m in methods.items() if m.refine_with_lines]:
        methods[f"{name}_gtmask"] = dataclasses.replace(methods[name], name=f"{name}_gtmask", mask="gt")
    methods["label"] = Method("label", source="label")
    return methods


def load_methods(path: Path) -> dict[str, Method]:
    """Methods from a JSON object {name: Method fields}."""
    return {name: Method(name, **fields) for name, fields in json.loads(Path(path).read_text()).items()}


def keypoints_from_row(row, instance_conf: float) -> sv.KeyPoints | None:
    """A cached first instance (`dev.keypoint_row`) as `sv.KeyPoints`, or None without an instance or below
    `instance_conf`."""
    xy = row["kp_xy"]
    if xy is None or (isinstance(xy, float) and np.isnan(xy)) or not row["kp_instance_conf"] >= instance_conf:
        return None
    return sv.KeyPoints(xy=np.asarray(xy, np.float32).reshape(1, -1, 2),
                        confidence=np.asarray(row["kp_conf"], np.float32).reshape(1, -1))


def eval_frame(path: str, size: tuple[int, int] = ge.EVAL_SIZE) -> np.ndarray:
    """The frame at `path` (BGR) resized to `size` (w, h) with area interpolation."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image if image.shape[1::-1] == size else cv2.resize(image, size, interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class ClipTask:
    """One clip's inputs: keypoint rows indexed by frame (`dev.keypoint_row` columns, 720p), the detector's people
    boxes per frame (720p xyxy; needed by `mask="detector"`) and the kept label homographies per frame (720p frame ->
    corner metres; needed by `source="label"`)."""

    split: str
    folder: Path
    keypoints: pd.DataFrame
    methods: dict[str, Method]
    instance_conf: float = dev.INSTANCE_CONF
    detections: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    labels: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)


def evaluate_clip(task: ClipTask) -> tuple[list[dict], pd.DataFrame]:
    """Per-frame rows of every method on one clip's scored frames (and a row per unscored frame), and its convention
    check."""
    cv2.setNumThreads(1)
    clip = ge.read_clip(task.folder)
    estimators = {name: m.estimator() for name, m in task.methods.items() if m.source == "keypoints"}
    unrefined = {name: dataclasses.replace(m, refine_with_lines=False).estimator()
              for name, m in task.methods.items() if m.refine_with_lines}
    width, height = ge.EVAL_SIZE
    rows = []
    for frame in clip.frames.itertuples(index=False):
        scale = (width / frame.width, height / frame.height)
        persons = clip.persons[clip.persons.frame == frame.frame]
        truth_rows = persons[ge.on_pitch(persons)]
        base = {"split": task.split, "clip": clip.name, "game": clip.game, "frame": frame.frame,
                "persons": len(truth_rows), "scored": len(truth_rows) > 0}
        if not base["scored"]:
            rows += [base | {"method": name} for name in task.methods]
            continue
        image = eval_frame(frame.path)
        view = lines.LineView.of(image)
        feet, truth = ge.feet(truth_rows, scale), truth_rows[["pitch_x", "pitch_y"]].to_numpy(float)
        base |= {"gt_median_x": float(np.median(truth[:, 0])), "near_goal": bool(ge.near_goal(np.median(truth[:, 0])))}
        boxes = {"gt": persons[["x1", "y1", "x2", "y2"]].to_numpy(float) * np.tile(scale, 2), "none": None}
        if any(m.mask == "detector" for m in task.methods.values()):
            if frame.frame not in task.detections:
                raise ValueError(f"{clip.name} frame {frame.frame}: a method uses the detector mask but the task has no "
                                 "detections for the frame; pass the detector's people boxes (ClipTask.detections)")
            boxes["detector"] = task.detections[frame.frame]
        found = keypoints_from_row(task.keypoints.loc[frame.frame], task.instance_conf)
        for name, method in task.methods.items():
            if method.source == "label":
                matrix = task.labels.get(frame.frame)
            else:
                matrix = estimators[name].fit(found, image, boxes[method.mask])
            errors = ge.person_errors(matrix, feet, truth, (height, width))
            confident = 0 if found is None else int((found.confidence[0] >= method.min_confidence).sum())
            row = base | {"method": name, "has_homography": matrix is not None,
                          "keypoints": confident if method.source == "keypoints" else np.nan,
                          "error_m": float(np.median(errors)) if errors is not None else np.nan,
                          "error_mean_m": float(errors.mean()) if errors is not None else np.nan,
                          "capped": int((errors >= ge.CAP_M).sum()) if errors is not None else 0,
                          "agreement": lines.score(view, matrix)["agreement"],
                          "homography": None if matrix is None else [float(v) for v in np.ravel(matrix)]}
            if name in unrefined:
                start = unrefined[name].fit(found)
                row["refined"] = float(matrix is not None and start is not None and not np.allclose(matrix, start))
            rows.append(row)
    return rows, ge.convention_check(clip)
