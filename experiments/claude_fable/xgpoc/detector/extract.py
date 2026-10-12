"""The four declared extractor configurations, run with the package's models on shot frames (xg-detector spec, Extraction).

    config  detector input  ball threshold  ball fallback (-400/-800 ms)
    C1      640             0.30            no
    C2      640             0.30            yes
    C3      1920            0.15            no
    C4      1920            0.15            yes

Common to all four: YOLO11m detector at conf 0.3 and NMS IoU 0.5 for the people (the package's run-file values,
`configs/pipeline.yaml`), pitch keypoints (YOLOv8n-pose with the package's letterbox input at its training size 640), the
stateless homography of each frame (`HomographyEstimator.fit`: keypoints with confidence >= 0.5, RANSAC 10 m), SigLIP
single-frame teams (k-means k = 2 on the frame's player crops, no UMAP). The package's `FrameAnalyzer` does all of it;
this module only chooses the settings, reads the frames and turns results into cache rows.

The one departure from the run file: the pitch instance threshold is 0.1, not 0.5 (the research extractor's value).
On 300 random paired t0 frames the homography fitted on 53% at 0.5 and 61% at 0.1; the per-keypoint threshold of 0.5,
which decides what the fit uses, is unchanged. Chosen on coverage alone (no position error, no label), before the
audit, the same in all four configurations; the instance confidence is cached (`kp_instance_conf`) so the audit can
also report the 0.5 subset.

The ball fallback is the package's `fill_ball`: a t0 frame without a ball on the pitch takes the ball of the -400 ms
frame, else of the -800 ms frame, each placed by its own homography. Those frames are analysed only when needed, without
teams, and they never contribute anything but the ball. No frame after t0 is read.
"""

import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.pipeline import FrameAnalyzer, FrameResult, fill_ball
from xgpoc.detector.features import missing_features, shot_features

# The trained weights live in the main checkout's results/ (not in git); models/football_yolo11m.pt, the run file's
# name, does not exist locally. The completion record pins them by sha256.
MAIN_CHECKOUT = Path(os.environ.get("TACTIFOOT_REPO", "/home/kuba/projects/tactifoot_vision"))
DETECTOR_WEIGHTS = MAIN_CHECKOUT / "results/project/raw/yolo11m/training/detect_yolo11m/weights/best.pt"
KEYPOINT_WEIGHTS = MAIN_CHECKOUT / "results/project/raw/yolo_pose/training/kp_yolo_pose/weights/best.pt"
SIGLIP = "google/siglip-base-patch16-224"

FRAME_COLUMNS = {0: "frame_path", -400: "frame_path_-400", -800: "frame_path_-800"}
FALLBACK_OFFSETS = (-400, -800)


@dataclass(frozen=True)
class ExtractorConfig:
    name: str
    imgsz: int
    ball_conf: float
    ball_fallback: bool

    @property
    def detector(self) -> tuple[int, float]:
        """Configurations with the same detector settings share their t0 detections."""
        return self.imgsz, self.ball_conf


CONFIGS = {c.name: c for c in (ExtractorConfig("C1", 640, 0.30, False), ExtractorConfig("C2", 640, 0.30, True),
                               ExtractorConfig("C3", 1920, 0.15, False), ExtractorConfig("C4", 1920, 0.15, True))}
COMMON = {"detector_conf": 0.3, "detector_iou": 0.5, "keypoint_conf": 0.1, "homography_min_confidence": 0.5,
          "homography_ransac_threshold": 10.0, "team_embedder": SIGLIP, "team_reducer": None, "n_teams": 2,
          "crop_scale": 0.8, "fallback_offsets_ms": list(FALLBACK_OFFSETS)}


def settings(config: ExtractorConfig) -> dict:
    """Everything that defines a configuration's output, for its completion record."""
    return asdict(config) | COMMON


def build_analyzers(imgsz: int, ball_conf: float, embedder, device: str | None = None) -> tuple[FrameAnalyzer, FrameAnalyzer]:
    """(t0 analyzer with teams, analyzer of the fallback frames without teams), sharing the models."""
    import tactifoot_vision as tv

    detector = tv.load_model("yolo", DETECTOR_WEIGHTS, conf=COMMON["detector_conf"], iou=COMMON["detector_iou"],
                             imgsz=imgsz, class_conf={"ball": ball_conf}, device=device)
    keypoints = tv.load_model("yolo_pose", KEYPOINT_WEIGHTS, conf=COMMON["keypoint_conf"], device=device)
    homography = tv.pitch.HomographyEstimator(min_confidence=COMMON["homography_min_confidence"],
                                              ransac_threshold=COMMON["homography_ransac_threshold"])
    teams = tv.teams.TeamClassifier(embedder, n_teams=COMMON["n_teams"], reducer=COMMON["team_reducer"])
    with_teams = FrameAnalyzer(detector, keypoints, teams, homography=homography, crop_scale=COMMON["crop_scale"])
    ball_only = FrameAnalyzer(detector, keypoints, None, homography=homography)
    return with_teams, ball_only


# ------------------------------------------------------------------ cache rows
def _floats(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def frame_row(shot_id: str, offset_ms: int, path: str, frame: FrameResult | None) -> dict:
    """One row of `frames.parquet`: detections (people and the ball), first keypoint instance, homography.

    `frame=None` records an unreadable image. Lists hold one entry per person; the ball columns hold 0 or 1 entries.
    """
    row = {"shot_id": shot_id, "offset_ms": offset_ms, "frame_path": path, "readable": frame is not None}
    if frame is None:
        return row
    people, ball, kp = frame.detections, frame.ball, frame.keypoints
    has_kp = kp is not None and len(kp) > 0
    row |= {"homography": _floats(frame.homography) if frame.homography is not None else None,
            "kp_xy": _floats(kp.xy[0]) if has_kp else None,
            "kp_conf": _floats(kp.confidence[0]) if has_kp and kp.confidence is not None else None,
            "kp_instance_conf": float(kp.data["box_confidence"][0]) if has_kp and "box_confidence" in kp.data else np.nan,
            "det_class": [str(n) for n in people.data["class_name"]], "det_conf": _floats(people.confidence),
            "det_xyxy": _floats(people.xyxy), "det_pitch_xy": _floats(frame.pitch_xy), "det_team": [int(t) for t in frame.team_ids],
            "ball_conf": _floats(ball.confidence), "ball_xyxy": _floats(ball.xyxy), "ball_pitch_xy": _floats(frame.ball_xy)}
    return row


def frame_from_row(row) -> FrameResult | None:
    """The `FrameResult` a cache row was written from (people, ball, first keypoint instance, homography)."""
    if not row["readable"]:
        return None

    def array(name, width):
        return np.asarray(row[name], dtype=float).reshape(-1, width)
    names = np.asarray(row["det_class"], dtype=str)
    people = sv.Detections(xyxy=array("det_xyxy", 4).astype(np.float32), confidence=np.asarray(row["det_conf"], np.float32),
                           data={"class_name": names, "pitch_xy": array("det_pitch_xy", 2),
                                 "team_id": np.asarray(row["det_team"], dtype=int)})
    ball = sv.Detections(xyxy=array("ball_xyxy", 4).astype(np.float32), confidence=np.asarray(row["ball_conf"], np.float32),
                         data={"class_name": np.full(len(row["ball_conf"]), "ball", dtype=str), "pitch_xy": array("ball_pitch_xy", 2)})
    keypoints = None
    if row["kp_xy"] is not None:
        conf = np.asarray(row["kp_conf"], np.float32)[None] if row["kp_conf"] is not None else None
        keypoints = sv.KeyPoints(xy=array("kp_xy", 2).astype(np.float32)[None], confidence=conf)
    homography = array("homography", 3) if row["homography"] is not None else None
    return FrameResult(index=0, timestamp=0.0, detections=people, ball=ball, keypoints=keypoints, homography=homography)


# ------------------------------------------------------------------ extraction
@dataclass
class Extraction:
    """Output of one configuration: every frame row it used and one feature row per shot."""

    config: ExtractorConfig
    frames: list[dict]
    features: list[dict]


def extract_detector(shots: pd.DataFrame, configs: Sequence[ExtractorConfig], analyzer: Callable, ball_only: Callable,
                     read_image: Callable, progress: Callable[[Iterator], Iterator] = iter) -> dict[str, Extraction]:
    """Run configurations that share detector settings: one t0 analysis per shot, fallback frames only when needed.

    `shots`: `shot_id` and the `FRAME_COLUMNS`. `analyzer` / `ball_only`: frame -> `FrameResult` (with and without
    teams). `read_image`: path -> BGR image or None.
    """
    if len({c.detector for c in configs}) != 1:
        raise ValueError("configurations of one pass must share the detector settings")
    out = {c.name: Extraction(c, [], []) for c in configs}
    for shot in progress(shots.to_dict("records")):
        shot_id, path = shot["shot_id"], shot[FRAME_COLUMNS[0]]
        image = read_image(path)
        t0 = analyzer(image) if image is not None else None
        t0_row = frame_row(shot_id, 0, path, t0)
        earlier: dict[int, tuple[dict, FrameResult | None]] = {}
        for config in configs:
            extraction = out[config.name]
            extraction.frames.append(t0_row)
            if t0 is None:
                features = missing_features()
            else:
                lazily = _earlier_frames(shot, read_image, ball_only, earlier) if config.ball_fallback else []
                filled = fill_ball(t0, lazily)
                features = shot_features(filled, COMMON["homography_min_confidence"])
                if config.ball_fallback:
                    extraction.frames += [row for row, _ in earlier.values()]
            source = features["ball_source"]
            offset = np.nan if np.isnan(source) else ([0, *FALLBACK_OFFSETS])[int(source)]
            extraction.features.append({"shot_id": shot_id, "frame_readable": float(t0 is not None), "ball_offset_ms": offset}
                                       | features)
    return out


def _earlier_frames(shot: dict, read_image: Callable, ball_only: Callable,
                    analysed: dict[int, tuple[dict, FrameResult | None]]) -> Iterator[FrameResult]:
    """The shot's -400 and -800 ms frames, analysed on first use; `analysed` keeps (cache row, result) per offset.

    One result per offset, in `FALLBACK_OFFSETS` order, so `fill_ball`'s k-th frame is always offset k: a missing or
    unreadable frame is a frame without a ball.
    """
    for offset in FALLBACK_OFFSETS:
        if offset not in analysed:
            path = shot[FRAME_COLUMNS[offset]]
            image = read_image(path) if isinstance(path, str) else None
            result = ball_only(image) if image is not None else None
            analysed[offset] = (frame_row(shot["shot_id"], offset, path, result), result)
        yield analysed[offset][1] or _NOTHING


_NOTHING = FrameResult(index=0, timestamp=0.0, detections=sv.Detections.empty(), ball=sv.Detections.empty())


def frames_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["shot_id", "offset_ms"], ascending=[True, False], kind="stable").reset_index(drop=True)


def json_text(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
