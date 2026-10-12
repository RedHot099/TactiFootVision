"""One frame on its own: the single-frame counterpart of the video pipeline."""

import dataclasses
from collections.abc import Iterable, Sequence

import numpy as np
import supervision as sv

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline.pipeline import (
    GOALKEEPER,
    _detect_ball_and_people,
    _project,
)
from tactifoot_vision.pipeline.result import NO_TEAM, FrameResult
from tactifoot_vision.pitch.homography import HomographyEstimator
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.teams.classifier import TeamClassifier
from tactifoot_vision.teams.frame import (
    assign_frame_teams,
    check_single_frame_classifier,
)


class FrameAnalyzer:
    """Detections, homography, pitch positions and teams of one frame, from that frame alone.

    For frames that are not a video sequence (one frame per shot, frames of
    different matches). Unlike :class:`~tactifoot_vision.pipeline.Pipeline`
    there is no tracker, no state between calls and nothing averaged over
    frames: the homography is the stateless
    :meth:`~tactifoot_vision.pitch.HomographyEstimator.fit` of this frame's
    keypoints, and teams come from this frame's crops
    (:func:`~tactifoot_vision.teams.assign_frame_teams`, goalkeepers by pitch
    position). As in the pipeline, the most confident ``ball_class``
    detection is the ball, people are placed at the bottom centre of their
    box and the ball at its centre.

    Args:
        detector: a detection :class:`Model` (``data["class_name"]`` filled).
        keypoint_model: a pose :class:`Model` predicting the pitch landmarks;
            without it ``homography`` is ``None`` and ``pitch_xy`` NaN.
        team_classifier: ``TeamClassifier(reducer=None)``; ``None`` skips teams.
        pitch: pitch model; defaults to ``homography.pitch`` or ``SoccerPitch()``.
        homography: estimator whose settings (``min_confidence``,
            ``ransac_threshold``) the per-frame fit uses; its state is never used.
        ball_class: detector class name of the ball.
        include_classes: people classes to keep (the ball is always kept); ``None`` keeps all.
        player_class, goalkeeper_class: classes clustered by kit, and placed by position.
        crop_scale, crop_center_ratio: crop geometry for the team embeddings.

    Example::

        analyzer = tv.pipeline.FrameAnalyzer(
            detector, keypoint_model, tv.teams.TeamClassifier("siglip", reducer=None)
        )
        result = analyzer(cv2.imread("shot.jpg"))
        result.pitch_xy, result.team_ids, result.ball_xy
    """

    def __init__(
        self,
        detector: Model,
        keypoint_model: Model | None = None,
        team_classifier: TeamClassifier | None = None,
        pitch: SoccerPitch | None = None,
        homography: HomographyEstimator | None = None,
        ball_class: str = "ball",
        include_classes: Sequence[str] | None = None,
        player_class: str = "player",
        goalkeeper_class: str = GOALKEEPER,
        crop_scale: float = 0.8,
        crop_center_ratio: float = 1.0,
    ) -> None:
        if detector.task != Task.DETECT:
            raise ValueError(
                f"detector must be a detection model, got a {detector.task} model"
            )
        if keypoint_model is not None and keypoint_model.task != Task.POSE:
            raise ValueError(
                f"keypoint_model must be a pose model, got a {keypoint_model.task} model"
            )
        if pitch is not None and homography is not None and homography.pitch != pitch:
            raise ValueError(
                f"pitch {pitch} differs from the homography estimator's {homography.pitch}"
            )
        if team_classifier is not None:
            check_single_frame_classifier(team_classifier)
        self.detector = detector
        self.keypoint_model = keypoint_model
        self.team_classifier = team_classifier
        self.pitch = pitch or (homography.pitch if homography else SoccerPitch())
        self.homography = homography or HomographyEstimator(self.pitch)
        self.ball_class = ball_class
        self.include_classes = (
            None if include_classes is None else tuple(include_classes)
        )
        self.player_class = player_class
        self.goalkeeper_class = goalkeeper_class
        self.crop_scale = crop_scale
        self.crop_center_ratio = crop_center_ratio

    def __call__(
        self, frame: np.ndarray, index: int = 0, timestamp: float = 0.0
    ) -> FrameResult:
        """Analyse one BGR frame; ``index`` and ``timestamp`` only label the result."""
        ball, people = _detect_ball_and_people(
            self.detector, frame, self.ball_class, self.include_classes
        )
        people.mask = None  # full-frame masks (~2 MB per object at 1080p) are not kept
        keypoints = None
        matrix = None
        if self.keypoint_model is not None:
            keypoints = self.keypoint_model.predict(frame)
            matrix = self.homography.fit(keypoints)
        people.data["pitch_xy"] = _project(people, sv.Position.BOTTOM_CENTER, matrix)
        ball.data["pitch_xy"] = _project(ball, sv.Position.CENTER, matrix)
        if self.team_classifier is not None and len(people):
            people.data["team_id"] = assign_frame_teams(
                frame,
                people,
                self.team_classifier,
                positions=people.data["pitch_xy"],
                player_class=self.player_class,
                goalkeeper_class=self.goalkeeper_class,
                crop_scale=self.crop_scale,
                crop_center_ratio=self.crop_center_ratio,
            )
        else:
            people.data["team_id"] = np.full(len(people), NO_TEAM, dtype=int)
        return FrameResult(
            index=index,
            timestamp=timestamp,
            detections=people,
            ball=ball,
            keypoints=keypoints,
            homography=matrix,
        )

    def __repr__(self) -> str:
        return (
            f"FrameAnalyzer(detector={self.detector!r}, keypoint_model={self.keypoint_model!r}, "
            f"team_classifier={self.team_classifier!r})"
        )


def fill_ball(result: FrameResult, earlier: Iterable[FrameResult]) -> FrameResult:
    """Ball fallback: ``result`` with the ball of an earlier frame when it has none on the pitch.

    When ``result`` has no ball with a pitch position, the ball of the first
    ``earlier`` frame (most recent first) that has one is used: its
    ``pitch_xy`` comes from that frame's own homography, its box is in that
    frame's pixels. ``earlier`` is consumed lazily, so a generator analyses
    only the frames it needs. The returned ball (a copy; ``result`` is not
    changed) has ``data["fallback"]``: 0 for the frame's own ball, ``k`` for
    the ``k``-th earlier frame. Without any ball the result keeps its own
    (possibly empty, or off the pitch) ball.
    """
    if _has_pitch_ball(result):
        return dataclasses.replace(result, ball=_with_fallback(result.ball, 0))
    for k, other in enumerate(earlier, start=1):
        if _has_pitch_ball(other):
            ball = other.ball[np.isfinite(other.ball_xy).all(axis=1)]
            return dataclasses.replace(result, ball=_with_fallback(ball, k))
    return dataclasses.replace(result, ball=_with_fallback(result.ball, 0))


def _has_pitch_ball(result: FrameResult) -> bool:
    return bool(np.isfinite(result.ball_xy).all(axis=1).any())


def _with_fallback(ball: sv.Detections, source: int) -> sv.Detections:
    copy = ball[np.ones(len(ball), dtype=bool)]
    copy.data["fallback"] = np.full(len(copy), source, dtype=int)
    return copy
