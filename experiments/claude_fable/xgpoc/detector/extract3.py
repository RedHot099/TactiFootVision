"""Audit 3's extractor configurations (pitch-keypoints 04): the SoccerNet pitch model and a declared shot-location source.

Every configuration shares C4/D3's detector (YOLO11m at 1920, conf 0.3, NMS IoU 0.5, ball 0.15) and teams (SigLIP,
k-means k = 2 on the frame's player crops), and the pitch keypoint model of pitch-keypoints 03 as it was evaluated:
`models/pitch_yolo11m_pose_sn960.pt` through `tv.load_model("yolo_pose", W, conf=0.1, imgsz=960, resize="square")`,
with the homography settings chosen on GSR valid (keypoint threshold 0.3, RANSAC 2 m, line refinement whose line mask
leaves out the detector's people boxes, `FrameAnalyzer` passes them). They differ only in the **shot-location source**:
the ordered frame offsets whose ball may place the shot (`source`). A frame's feature row takes the ball of the first
offset in that order whose frame has a ball on the pitch, placed by that frame's own homography; offsets after the
row's own frame are left out of its order (the -400 ms row of the full run never reads t0). Without any ball the shot
location is the shooter flagged `shooter_by_goal` (xg-detector's rule, `features.situation`). The players, teams and
goalkeeper always come from the row's own frame.

The configurations are declared, with the evidence for each, in `research/pitchkp_audit3_configs.json` before audit 3
reads any StatsBomb data (`AUDIT_3_DECLARATION_SHA256`); `CONFIGS` must equal the file (a test ties them).

One pass serves every configuration: a shot's frames are analysed once with the shared models (t0, and the -400 ms
frame with teams in the full run; any other frame without teams, and only when a configuration's order reaches it),
and each configuration takes its features from them. The frames table of a configuration holds every frame the pass
analysed for the shot; `replay_features` recomputes the features from it without a GPU.
"""

import dataclasses
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline import FrameAnalyzer, FrameResult
from xgpoc.detector import extract as ex
from xgpoc.detector.features import missing_features, shot_features
from xgpoc.detector.records import CHECKOUT

KEYPOINT_WEIGHTS = CHECKOUT / "models/pitch_yolo11m_pose_sn960.pt"
KEYPOINT_SHA256 = "5be9954a0682bfd6827a4b293cf33e2b348c7a3ef7e086415de4c7d9199f005d"
COMMON = {"detector_conf": 0.3, "detector_iou": 0.5, "detector_imgsz": 1920, "ball_conf": 0.15,
          "keypoint_conf": 0.1, "keypoint_imgsz": 960, "keypoint_resize": "square",
          "homography_min_confidence": 0.3, "homography_ransac_threshold": 2.0, "refine_with_lines": True,
          "team_embedder": ex.SIGLIP, "team_reducer": None, "n_teams": 2, "crop_scale": 0.8}
STUDY_OFFSETS = (0, -400)
FRAME_COLUMNS = ex.FRAME_COLUMNS          # offset -> index column of the frame path (0, -400, -800 ms)


@dataclass(frozen=True)
class SourceConfig:
    """An audit-3 configuration: its name and the ordered offsets (ms) whose ball may place the shot."""

    name: str
    source: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source or len(set(self.source)) != len(self.source) or not set(self.source) <= set(FRAME_COLUMNS):
            raise ValueError(f"{self.name}: source must be distinct offsets of {sorted(FRAME_COLUMNS)}, got {self.source}")

    def order(self, offset_ms: int) -> tuple[int, ...]:
        """The source order of the feature row of the frame at `offset_ms`: no frame after it."""
        return tuple(o for o in self.source if o <= offset_ms)


# Declared in research/pitchkp_audit3_configs.json (see there for the evidence of each).
CONFIGS = {c.name: c for c in (
    SourceConfig("E1", (0, -400, -800)),      # audit 2's (D3's) rule, only the pitch model and its settings swapped
    SourceConfig("E2", (-400, -800)),         # the fixed order over -400 / -800 ms: never the t0 ball
    SourceConfig("E3", (-400, -800, 0)))}     # the same, with the t0 ball as the last fallback
AUDIT_3 = tuple(CONFIGS)
AUDIT_3_DECLARATION = Path(__file__).resolve().parents[2] / "research/pitchkp_audit3_configs.json"
AUDIT_3_DECLARATION_SHA256 = None         # set when the declaration is committed, before audit 3


def settings(config: SourceConfig) -> dict:
    """Everything that defines a configuration's output, for its completion record."""
    return {"name": config.name, "source": list(config.source)} | COMMON | {
        "keypoint_weights": str(KEYPOINT_WEIGHTS), "keypoint_sha256": KEYPOINT_SHA256}


def ball_analyzer(device: str | None = None) -> FrameAnalyzer:
    """The analyzer without teams: detector, pitch keypoints and homography at the settings of `COMMON`."""
    import tactifoot_vision as tv

    detector = tv.load_model("yolo", ex.DETECTOR_WEIGHTS, conf=COMMON["detector_conf"], iou=COMMON["detector_iou"],
                             imgsz=COMMON["detector_imgsz"], class_conf={"ball": COMMON["ball_conf"]}, device=device)
    keypoints = tv.load_model("yolo_pose", KEYPOINT_WEIGHTS, conf=COMMON["keypoint_conf"], imgsz=COMMON["keypoint_imgsz"],
                              resize=COMMON["keypoint_resize"], device=device)
    homography = tv.pitch.HomographyEstimator(min_confidence=COMMON["homography_min_confidence"],
                                              ransac_threshold=COMMON["homography_ransac_threshold"],
                                              refine_with_lines=COMMON["refine_with_lines"])
    return FrameAnalyzer(detector, keypoints, None, homography=homography)


def build_analyzers(embedder, device: str | None = None) -> tuple[FrameAnalyzer, FrameAnalyzer]:
    """(analyzer with teams, `ball_analyzer`), sharing the models."""
    import tactifoot_vision as tv

    ball_only = ball_analyzer(device)
    teams = tv.teams.TeamClassifier(embedder, n_teams=COMMON["n_teams"], reducer=COMMON["team_reducer"])
    with_teams = FrameAnalyzer(ball_only.detector, ball_only.keypoint_model, teams, homography=ball_only.homography,
                               crop_scale=COMMON["crop_scale"])
    return with_teams, ball_only


# ------------------------------------------------------------------ the source rule
def has_pitch_ball(result: FrameResult | None) -> bool:
    return result is not None and bool(np.isfinite(result.ball_xy).all(axis=1).any())


def chosen_ball(frames: dict[int, FrameResult | None], order: Sequence[int]) -> tuple[int, int] | None:
    """(position in `order`, offset) of the first frame of `order` with a ball on the pitch, or None. Every offset of
    `order` must be a key of `frames` (None: unreadable or not available) up to the chosen one."""
    for position, offset in enumerate(order):
        if offset not in frames:
            raise ValueError(f"the source order needs the {offset} ms frame, which was not analysed")
        if has_pitch_ball(frames[offset]):
            return position, offset
    return None


def with_ball(result: FrameResult, source: FrameResult, marker: int) -> FrameResult:
    """`result` (the row's own frame) with the on-pitch balls of `source` (another frame, or itself), each placed by
    its own frame's homography; `data["fallback"]` = `marker` (0: the row's own ball)."""
    ball = source.ball[np.isfinite(source.ball_xy).all(axis=1)]
    ball.data["fallback"] = np.full(len(ball), marker, dtype=int)
    return dataclasses.replace(result, ball=ball)


def without_ball(result: FrameResult) -> FrameResult:
    """`result` without any ball (its own may lie outside the order, as t0's does for E2)."""
    ball = result.ball[np.zeros(len(result.ball), dtype=bool)]
    ball.data["pitch_xy"] = np.zeros((0, 2))
    return dataclasses.replace(result, ball=ball)


def row_features(offset_ms: int, frames: dict[int, FrameResult | None], config: SourceConfig) -> dict:
    """One feature row: the frame at `offset_ms` with the ball of the configuration's source order, and the offset of
    the frame the ball came from (`ball_offset_ms`, NaN without a ball). `ball_source` is 0 for the row's own ball,
    else k for the k-th other frame of the order (as `fill_ball` numbers the fallback frames of audits 1-2)."""
    result = frames.get(offset_ms)
    if result is None:
        return {"offset_ms": offset_ms, "frame_readable": 0.0, "ball_offset_ms": np.nan} | missing_features()
    picked = chosen_ball(frames, config.order(offset_ms))
    if picked is None:
        filled, ball_offset = without_ball(result), np.nan     # no ball of the order on the pitch: none is used
    else:
        ball_offset = picked[1]
        others = [o for o in config.order(offset_ms) if o != offset_ms]
        filled = with_ball(result, frames[ball_offset], 0 if ball_offset == offset_ms else others.index(ball_offset) + 1)
    features = shot_features(filled, COMMON["homography_min_confidence"])
    return {"offset_ms": offset_ms, "frame_readable": 1.0, "ball_offset_ms": ball_offset} | features


def needed(frames: dict[int, FrameResult | None], order: Sequence[int]) -> Iterator[int]:
    """Offsets of `order` not yet in `frames` that the rule would read, in order (a generator: the caller analyses each
    before asking for the next)."""
    for offset in order:
        if offset not in frames:
            yield offset
        if has_pitch_ball(frames[offset]):
            return


# ------------------------------------------------------------------ extraction
@dataclass
class Extraction:
    configs: tuple[SourceConfig, ...]
    frames: list[dict]
    features: dict[str, list[dict]]


def extract(shots: pd.DataFrame, configs: Sequence[SourceConfig], analyzer: Callable, ball_only: Callable,
            read_image: Callable, offsets: Sequence[int] = (0,),
            progress: Callable[[Iterator], Iterator] = iter) -> Extraction:
    """The feature rows at `offsets` (t0 for the audit, `STUDY_OFFSETS` for the full run) of every configuration.

    `offsets` are analysed with teams (`analyzer`); any other frame a source order reaches without teams
    (`ball_only`), once per shot, shared by the configurations. `shots`: shot_id and the `FRAME_COLUMNS`.
    """
    out = Extraction(tuple(configs), [], {c.name: [] for c in configs})
    for shot in progress(shots.to_dict("records")):
        rows: dict[int, dict] = {}
        frames: dict[int, FrameResult | None] = {}

        def analyse(offset, model, shot=shot, rows=rows, frames=frames):
            path = shot[FRAME_COLUMNS[offset]]
            image = read_image(path) if isinstance(path, str) else None
            frames[offset] = model(image) if image is not None else None
            rows[offset] = ex.frame_row(shot["shot_id"], offset, path, frames[offset])

        for offset in offsets:
            analyse(offset, analyzer)
        for config in configs:
            for offset in offsets:
                for missing in needed(frames, config.order(offset)):
                    analyse(missing, ball_only)
                out.features[config.name].append({"shot_id": shot["shot_id"]} | row_features(offset, frames, config))
        out.frames += [rows[o] for o in sorted(rows, reverse=True)]
    return out


def replay_features(frames: pd.DataFrame, config: SourceConfig, offsets: Sequence[int] = (0,)) -> pd.DataFrame:
    """Feature rows recomputed from a configuration's cached frames, without a GPU (one row per shot and offset with a
    cached frame), in shot order."""
    rows = []
    for shot_id, group in frames.groupby("shot_id", sort=True):
        cached = {int(r["offset_ms"]): ex.frame_from_row(r) for r in group.to_dict("records")}
        for offset in offsets:
            if offset in cached:
                rows.append({"shot_id": shot_id} | row_features(offset, cached, config))
    return pd.DataFrame(rows)


def declaration() -> dict:
    return json.loads(AUDIT_3_DECLARATION.read_text())
