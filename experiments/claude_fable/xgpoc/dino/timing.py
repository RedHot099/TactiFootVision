"""Inference-time benchmark of A (image: G-combo), B (automatic geometry) and C (both) (xg-dino ticket 03, council C1).

The deployment path of each arm for one new shot, built from the frozen pieces and timed per stage:

    A  decode the t0 frame -> G input (resize, overlay mask, normalise, host-to-device) -> the shared DINOv2-L trunk
       (the 20 frozen blocks, once) -> n fold-model tails (1 = one member, 10 = the deployed G-combo) -> the deployed
       calibration (mean temperature-scaled logit z, Platt, prior shift) on the CPU.
    B  the real uncached feature path of the frozen feet_t0 E3 configuration (`cache/xgd_full4_feet_t0/E3`): decode the
       t0 and -400 ms frames and analyse both with teams (dd2c YOLO11l at 1280, the SoccerNet pitch keypoint model,
       homography with line refinement, SigLIP team embeddings and per-frame k-means); the -800 ms frame is decoded and
       analysed without teams only when the feet_t0 ball order reaches it (`extract3.needed`); then the feet rule,
       the sanitation and the stack row of both offsets (`geometry_row`, the computation of `stack_location.e3_table`
       for one shot) and the B logistic on the arm's design (`Logistic`).
    C  one decode of the shot's frames shared by A and B, both paths, then the C logistic on z and the geometry.

`StageClock` times named stages with a CUDA synchronisation on both sides, so a stage's time includes its GPU work.
Stages may nest (the analyser call holds the detector, keypoint, homography and team stages); `analyse_rest` is what
is left of the analyser outside them. `instrument` puts the stage timers on a `FrameAnalyzer` without changing what it
computes.
"""

import contextlib
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch

from tactifoot_vision.pipeline import FrameAnalyzer, FrameResult
from tactifoot_vision.pipeline import frame as frame_module
from xgpoc.data.datasets import FILL, overlay_mask, to_tensor
from xgpoc.detector import extract3 as x3
from xgpoc.detector import location as loc
from xgpoc.detector import sanitation as sn
from xgpoc.eval.calibration import PlattScaler, prior_shift
from xgpoc.probe.scoring import trunk_features
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si

RULE = loc.FEET_T0
TEAM_OFFSETS = si.OFFSETS                     # analysed with teams: t0 and -400 ms (the two rows of the stack table)


# ------------------------------------------------------------------ clock and summaries
class StageClock:
    """Milliseconds per named stage, accumulated until `take`; `sync` runs before and after every stage."""

    def __init__(self, sync: Callable[[], None] = lambda: None, now: Callable[[], float] = time.perf_counter):
        self.sync, self.now = sync, now
        self.ms: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.sync()
        start = self.now()
        try:
            yield
        finally:
            self.sync()
            self.ms[name] += (self.now() - start) * 1000

    def take(self) -> dict[str, float]:
        """The accumulated stages, cleared."""
        out, self.ms = dict(self.ms), defaultdict(float)
        return out


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summary(values) -> dict[str, float]:
    """Median, quartiles (linear interpolation) and IQR of `values`, and their count."""
    v = np.asarray(values, float)
    q25, median, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "n": int(v.size)}


# ------------------------------------------------------------------ instrumented analyser
class _Timed:
    """`target` with its method `method` timed as stage `name`; every other attribute is the target's."""

    def __init__(self, target, method: str, name: str, clock: StageClock):
        self._target, self._method, self._name, self._clock = target, method, name, clock

    def __getattr__(self, attr):
        value = getattr(self._target, attr)
        if attr != self._method:
            return value

        def timed(*args, **kwargs):
            with self._clock.stage(self._name):
                return value(*args, **kwargs)
        return timed


def instrument(analyzer: FrameAnalyzer, clock: StageClock) -> FrameAnalyzer:
    """A shallow copy of `analyzer` (same models) whose detector, keypoint model and homography fit are timed as
    `detect`, `keypoints`, `homography`; the team step is timed by `timed_teams`."""
    timed = object.__new__(FrameAnalyzer)
    timed.__dict__ = dict(analyzer.__dict__)
    timed.detector = _Timed(analyzer.detector, "predict", "detect", clock)
    if analyzer.keypoint_model is not None:
        timed.keypoint_model = _Timed(analyzer.keypoint_model, "predict", "keypoints", clock)
    timed.homography = _Timed(analyzer.homography, "fit", "homography", clock)
    return timed


@contextlib.contextmanager
def timed_teams(clock: StageClock) -> Iterator[None]:
    """`FrameAnalyzer`'s team step (crops, SigLIP embeddings, k-means, goalkeepers) timed as `teams` while open."""
    original = frame_module.assign_frame_teams

    def timed(*args, **kwargs):
        with clock.stage("teams"):
            return original(*args, **kwargs)
    frame_module.assign_frame_teams = timed
    try:
        yield
    finally:
        frame_module.assign_frame_teams = original


# ------------------------------------------------------------------ B: features and logistic
def kp_instance_conf(result: FrameResult | None) -> float:
    """The cached `kp_instance_conf` of `extract.frame_row`: the first keypoint instance's box confidence."""
    if result is None:
        return np.nan
    kp = result.keypoints
    if kp is None or len(kp) == 0 or "box_confidence" not in kp.data:
        return np.nan
    return float(kp.data["box_confidence"][0])


def geometry_row(frames: dict[int, FrameResult | None], rule: loc.LocationRule = RULE) -> dict[str, float]:
    """`inputs.E3_COLUMNS` of one shot from its analysed frames (offset -> result, None = unreadable), as
    `stack_location.e3_table` computes them under `rule`, angle wrap included. The E3 ball-source columns that
    `e3_flags_at` would take from E3's order are left as the rule's own: they are outside the C1 geometry allowlist."""
    row: dict[str, float] = {}
    for offset in si.OFFSETS:
        filled = loc.filled_frame(offset, frames, rule.source)
        raw = loc.rule_features(filled, rule)
        clean = (sn.missing_row() if filled is None
                 else sn.sanitised_features(rule.situation(filled), filled.homography is not None))
        row |= si.offset_row(offset, raw, clean, kp_instance_conf(frames.get(offset)))
    for offset in si.OFFSETS:
        name = si.col("angle", offset)
        row[name] = float(si.wrap_angle(row[name]))
    return row


@dataclass(frozen=True)
class Logistic:
    """A fitted arm at deployment: its preprocessor (imputation, spline, standardisation) and coefficients."""

    pre: sf.Preprocessor
    beta: np.ndarray

    def __post_init__(self):
        if len(self.beta) != 1 + self.width:
            raise ValueError(f"{self.pre.arm.name}: {len(self.beta)} coefficients for {self.width} design columns")

    @property
    def width(self) -> int:
        return len(self.pre.centre)

    def __call__(self, rows: pd.DataFrame) -> np.ndarray:
        return sf.predict(self.beta, self.pre(rows))


def surrogate(pre: sf.Preprocessor, seed: int = 0) -> Logistic:
    """A logistic of the arm's width with fixed random coefficients: the fitted coefficients of ticket 02 were not
    stored, and the cost of a prediction does not depend on their values."""
    return Logistic(pre, np.random.default_rng(seed).normal(0.0, 0.1, 1 + len(pre.centre)))


class BPath:
    """B's frame analysis and features for one shot (module docstring); stages `decode`, `analyse` (with the nested
    stages of `instrument`), `features`."""

    def __init__(self, analyzer: FrameAnalyzer, ball_only: FrameAnalyzer, clock: StageClock,
                 read: Callable[[str], np.ndarray | None] = cv2.imread, rule: loc.LocationRule = RULE):
        self.analyzer, self.ball_only = instrument(analyzer, clock), instrument(ball_only, clock)
        self.clock, self.read, self.rule = clock, read, rule

    def decode(self, shot: dict, offsets=TEAM_OFFSETS) -> dict[int, np.ndarray | None]:
        with self.clock.stage("decode"):
            return {o: self.read(shot[x3.FRAME_COLUMNS[o]]) if isinstance(shot[x3.FRAME_COLUMNS[o]], str) else None
                    for o in offsets}

    def analyse(self, shot: dict, images: dict[int, np.ndarray | None]) -> dict[int, FrameResult | None]:
        """Every frame the rule reads: the team offsets from `images`, then any fallback the ball order reaches
        (decoded here). Returns offset -> result."""
        frames: dict[int, FrameResult | None] = {}
        with timed_teams(self.clock):
            for offset in TEAM_OFFSETS:
                with self.clock.stage("analyse"):
                    frames[offset] = self.analyzer(images[offset]) if images[offset] is not None else None
            for offset in TEAM_OFFSETS:
                for missing in x3.needed(frames, self.rule.source.order(offset)):
                    image = self.decode(shot, (missing,))[missing]
                    with self.clock.stage("analyse"):
                        frames[missing] = self.ball_only(image) if image is not None else None
        return frames

    def features(self, frames: dict[int, FrameResult | None]) -> dict[str, float]:
        with self.clock.stage("features"):
            return geometry_row(frames, self.rule)


def analyse_rest(stages: dict[str, float]) -> float:
    """The analyser's time outside its timed parts (projection, ball and people split, copies)."""
    return stages.get("analyse", 0.0) - sum(stages.get(k, 0.0) for k in ("detect", "keypoints", "homography", "teams"))


# ------------------------------------------------------------------ A: the shared trunk and the tails
class GInput:
    """`datasets.load_image` + `to_tensor` (no crop) from an already decoded BGR frame."""

    def __init__(self, size_wh: tuple[int, int], mask: str):
        self.size = tuple(size_wh)
        self.blank = overlay_mask(self.size, mask) if mask != "none" else None

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        if self.blank is not None:
            img[self.blank] = FILL
        return to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class APath:
    """G-combo on decoded frames: shared trunk once, then the tails of `members` (`SharedScorer` members, each with
    `.tail` and `.temperature`); stages `g_input`, `g_trunk`, `g_tails`, `g_calibrate`."""

    def __init__(self, vit: torch.nn.Module, n_trained: int, members, g_input: GInput, calibration: dict,
                 clock: StageClock, device: str):
        self.vit, self.n_trained, self.members, self.g_input = vit, n_trained, list(members), g_input
        self.calibration, self.clock, self.device = calibration, clock, device

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray], n_tails: int, amp: bool) -> tuple[np.ndarray, np.ndarray]:
        """(z, deployed probability) per image: z = mean over the first `n_tails` members of logit / T."""
        members = self.members[:n_tails]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            with self.clock.stage("g_input"):
                x = torch.stack([self.g_input(i) for i in images]).to(self.device, non_blocking=True)
            with self.clock.stage("g_trunk"):
                features = trunk_features(self.vit, x, self.n_trained)
            with self.clock.stage("g_tails"):
                logits = torch.stack([m.tail(features)[:, 0].float() for m in members]).cpu().numpy().astype(np.float64)
        with self.clock.stage("g_calibrate"):
            z = (logits / np.array([m.temperature for m in members])[:, None]).mean(0)
            c = self.calibration
            p = prior_shift(PlattScaler(**c["platt"])(z), c["oof_rate"], c["pooled_rate"])
        return z, p
