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
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.pipeline import FrameAnalyzer, FrameResult, fill_ball
from xgpoc.detector.features import FLAGS, missing_features, shot_features

# The trained weights live in the main checkout's results/ (not in git); models/football_yolo11m.pt, the run file's
# name, does not exist locally. The completion record pins them by sha256.
MAIN_CHECKOUT = Path(os.environ.get("TACTIFOOT_REPO", "/home/kuba/projects/tactifoot_vision"))
DETECTOR_WEIGHTS = MAIN_CHECKOUT / "results/project/raw/yolo11m/training/detect_yolo11m/weights/best.pt"
KEYPOINT_WEIGHTS = MAIN_CHECKOUT / "results/project/raw/yolo_pose/training/kp_yolo_pose/weights/best.pt"
SIGLIP = "google/siglip-base-patch16-224"

FRAME_COLUMNS = {0: "frame_path", -400: "frame_path_-400", -800: "frame_path_-800"}
FALLBACK_OFFSETS = (-400, -800)
STUDY_OFFSETS = (0, -400)        # the offsets the models use (spec, story 11); -800 is only ever a ball fallback


def fallback_offsets(offset_ms: int) -> tuple[int, ...]:
    """The earlier frames whose ball a frame at `offset_ms` may take, most recent first: (-400, -800) for t0, (-800,)
    for -400. Never a frame after the shot's own frame."""
    return tuple(o for o in FALLBACK_OFFSETS if o < offset_ms)


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
    if _listed(row["kp_xy"]):
        conf = np.asarray(row["kp_conf"], np.float32)[None] if _listed(row["kp_conf"]) else None
        keypoints = sv.KeyPoints(xy=array("kp_xy", 2).astype(np.float32)[None], confidence=conf)
    homography = array("homography", 3) if _listed(row["homography"]) else None
    return FrameResult(index=0, timestamp=0.0, detections=people, ball=ball, keypoints=keypoints, homography=homography)


def _listed(value) -> bool:
    """A cached list value is present: not None, and not the NaN a parquet column of nulls only reads back as."""
    return value is not None and not (isinstance(value, float) and np.isnan(value))


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
            lazily = _earlier_frames(shot, read_image, ball_only, earlier, FALLBACK_OFFSETS) if config.ball_fallback else []
            features = frame_features(t0, lazily)
            if config.ball_fallback:
                extraction.frames += [row for row, _ in earlier.values()]
            extraction.features.append({"shot_id": shot_id} | feature_record(0, t0 is not None, features))
    return out


def extract_study(shots: pd.DataFrame, config: ExtractorConfig, analyzer: Callable, ball_only: Callable,
                  read_image: Callable, progress: Callable[[Iterator], Iterator] = iter) -> Extraction:
    """One configuration on the study offsets of every shot (spec, story 11): t0 and -400 ms, each analysed with teams
    and each a feature row (`offset_ms`); -800 ms only as a ball fallback, analysed without teams and only when needed.

    With the fallback, t0 takes the ball of -400 (the same analysed frame) and then -800; -400 takes the ball of -800.
    A shot's t0 features are therefore exactly what `extract_detector` gives for the same configuration.
    """
    out = Extraction(config, [], [])
    for shot in progress(shots.to_dict("records")):
        analysed: dict[int, tuple[dict, FrameResult | None]] = {}
        for offset in STUDY_OFFSETS:
            path = shot[FRAME_COLUMNS[offset]]
            image = read_image(path) if isinstance(path, str) else None
            result = analyzer(image) if image is not None else None
            analysed[offset] = (frame_row(shot["shot_id"], offset, path, result), result)
        for offset in STUDY_OFFSETS:
            result = analysed[offset][1]
            earlier = fallback_offsets(offset)
            lazily = _earlier_frames(shot, read_image, ball_only, analysed, earlier) if config.ball_fallback else []
            features = frame_features(result, lazily)
            out.features.append({"shot_id": shot["shot_id"]} | feature_record(offset, result is not None, features))
        out.frames += [row for row, _ in analysed.values()]
    return out


def frame_features(result: FrameResult | None, earlier: Iterable[FrameResult]) -> dict[str, float]:
    """`shot_features` of an analysed frame after the ball fallback from `earlier` (empty: no fallback); a missing frame
    gives `missing_features`."""
    if result is None:
        return missing_features()
    return shot_features(fill_ball(result, earlier), COMMON["homography_min_confidence"])


def feature_record(offset_ms: int, readable: bool, features: dict[str, float]) -> dict:
    """A feature row's bookkeeping columns and features; `ball_offset_ms` is the offset of the frame the ball came from."""
    source = features["ball_source"]
    ball_offset = np.nan if np.isnan(source) else (offset_ms, *fallback_offsets(offset_ms))[int(source)]
    return {"offset_ms": offset_ms, "frame_readable": float(readable), "ball_offset_ms": ball_offset} | features


def _earlier_frames(shot: dict, read_image: Callable, ball_only: Callable,
                    analysed: dict[int, tuple[dict, FrameResult | None]], offsets: Sequence[int]) -> Iterator[FrameResult]:
    """The shot's frames at `offsets`, analysed (without teams) on first use unless `analysed` already holds them;
    `analysed` keeps (cache row, result) per offset.

    One result per offset, in order, so `fill_ball`'s k-th frame is always `offsets[k - 1]`: a missing or unreadable
    frame is a frame without a ball.
    """
    for offset in offsets:
        if offset not in analysed:
            path = shot[FRAME_COLUMNS[offset]]
            image = read_image(path) if isinstance(path, str) else None
            result = ball_only(image) if image is not None else None
            analysed[offset] = (frame_row(shot["shot_id"], offset, path, result), result)
        yield analysed[offset][1] or _NOTHING


def replay_features(frames: pd.DataFrame, config: ExtractorConfig, offsets: Sequence[int] = (0,)) -> pd.DataFrame:
    """Feature rows recomputed from a configuration's cached frames, without a GPU (one row per shot and offset).

    The same steps as the extraction (`frame_features`, `feature_record`) on the `frame_from_row` results. A fallback
    frame that the ball fallback asks for must be in the cache; a frame the fallback never reached may be absent.
    """
    rows = []
    for shot_id, group in frames.groupby("shot_id", sort=True):
        cached = {int(r["offset_ms"]): r for r in group.to_dict("records")}

        def earlier(chain, cached=cached, shot_id=shot_id):
            for offset in chain:
                if offset not in cached:
                    raise ValueError(f"{shot_id}: the ball fallback needs the {offset} ms frame, which is not cached")
                yield frame_from_row(cached[offset]) or _NOTHING
        for offset in offsets:
            if offset not in cached:
                continue
            result = frame_from_row(cached[offset])
            chain = earlier(fallback_offsets(offset)) if config.ball_fallback else []
            rows.append({"shot_id": shot_id} | feature_record(offset, result is not None, frame_features(result, chain)))
    return pd.DataFrame(rows)


_NOTHING = FrameResult(index=0, timestamp=0.0, detections=sv.Detections.empty(), ball=sv.Detections.empty())


def frame_table(index: Path, shot_ids: list[str]) -> pd.DataFrame:
    """shot_id and the t0 / -400 / -800 ms frame paths of the given shots, in their order (no other index column)."""
    frames = pd.read_parquet(index, columns=["shot_id", *FRAME_COLUMNS.values()]).set_index("shot_id")
    missing = sorted(set(shot_ids) - set(frames.index))
    if missing:
        raise ValueError(f"{len(missing)} shots are not in {index}, e.g. {missing[0]}")
    return frames.loc[shot_ids].reset_index()


def feature_differences(a: pd.DataFrame, b: pd.DataFrame, columns, tolerance: float) -> pd.DataFrame:
    """Where two feature tables with the same index differ by more than `tolerance` (NaN equals NaN): a boolean table
    over the rows and the `columns` present in both."""
    shared = a.columns.intersection(b.columns).intersection(list(columns))
    x, y = a[shared].to_numpy(float), b[shared].to_numpy(float)
    return pd.DataFrame(~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tolerance)), index=a.index, columns=shared)


def coverage(features: pd.DataFrame) -> dict:
    """Share of feature rows with each flag, and counts."""
    return {"shots": len(features), "frames_readable": int(features.frame_readable.sum())} | {
        flag: round(float(features[flag].mean()), 4) for flag in FLAGS}


COVERAGE_COLUMNS = ["frame_readable", "homography_ok", "ball_ok", "ball_fallback", "shot_ok", "shot_from_ball",
                    "shooter_by_goal", "teams_ok", "gk_ok", "gk_detected", "gk_proxy", "def_cone"]


def coverage_by_role(features: pd.DataFrame, shot_games: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Coverage of the study feature rows by split role and offset, plus "all": shots, matches and the share of rows
    with each of `COVERAGE_COLUMNS` (def_cone: a defenders-in-cone count exists).

    `shot_games`: shot_id, game (index); `plan`: game, role (xg-vendor split plan). Every shot must have a role.
    """
    roles = shot_games.merge(plan[["game", "role"]], on="game", how="left", validate="many_to_one")
    if roles.role.isna().any():
        raise ValueError(f"{int(roles.role.isna().sum())} shots have no role in the split plan")
    table = features.merge(roles, on="shot_id", how="left", validate="many_to_one")
    table["def_cone"] = table.ff_def_cone.notna().astype(float)
    rows = []
    for role, part in [*table.groupby("role", sort=True), ("all", table)]:
        for offset, rows_at in part.groupby("offset_ms", sort=False):
            rows.append({"role": role, "offset_ms": int(offset), "shots": len(rows_at), "matches": int(rows_at.game.nunique())}
                        | {c: float(rows_at[c].mean()) for c in COVERAGE_COLUMNS})
    return pd.DataFrame(rows).sort_values(["role", "offset_ms"], ascending=[True, False], kind="stable").reset_index(drop=True)


def frames_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["shot_id", "offset_ms"], ascending=[True, False], kind="stable").reset_index(drop=True)


def json_text(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
