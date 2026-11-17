"""Score a ``lines`` model through the homography it gives: the estimator over labelled clips, then ``evaluate_homography``."""

from collections.abc import Iterable, Mapping
from typing import Any

import supervision as sv

from tactifoot_vision.data.labelled_clip import LabelledClip
from tactifoot_vision.data.video import VideoReader
from tactifoot_vision.enums import Task
from tactifoot_vision.evaluation.homography import (
    evaluate_homography,
    evaluate_homography_clips,
)
from tactifoot_vision.evaluation.metrics import HomographyMetrics
from tactifoot_vision.models.base import Model


def evaluate_lines(
    model: Model,
    clips: LabelledClip | Iterable[LabelledClip],
    *,
    smooth: bool = True,
    estimator: Mapping[str, Any] | None = None,
    draws: int = 2000,
    seed: int = 0,
) -> HomographyMetrics:
    """The pitch position error of ``model``'s homographies on labelled clips.

    Each clip's video runs frame by frame through ``model`` and a
    :class:`~tactifoot_vision.pitch.LineHomographyEstimator` (``estimator``: its
    keyword arguments), through the pipeline's own
    :func:`~tactifoot_vision.pipeline.homographies.frame_homography`. With ``smooth``, the
    whole-video pass :func:`~tactifoot_vision.pipeline.smooth_line_homographies`
    follows (the ``final`` homographies), with the estimator's own filter
    settings (its ``temporal_filter``). An estimator without a filter
    (``temporal_filter=None``) has no pass to run: its per-frame matrices are
    scored, ``smooth`` or not. The result is scored with
    :func:`evaluate_homography` for one clip, or pooled with
    :func:`evaluate_homography_clips` (``draws``, ``seed``) for several.
    The clips need truth to score against: a calibration, or people's
    pitch positions.
    """
    from tactifoot_vision.pipeline import (
        FrameResult,
        PipelineResult,
        smooth_line_homographies,
    )
    from tactifoot_vision.pipeline.homographies import frame_homography
    from tactifoot_vision.pitch import LineHomographyEstimator

    if model.task != Task.LINES:
        raise ValueError(
            f"evaluate_lines needs a lines model, got a {model.task.value} model"
        )
    clips = [clips] if isinstance(clips, LabelledClip) else list(clips)
    if not clips:
        raise ValueError("evaluate_lines needs at least one labelled clip")
    pairs = []
    for clip in clips:
        reader = VideoReader(clip.video_path)
        homography = LineHomographyEstimator(**dict(estimator or {}))
        homography.reset(frame_rate=reader.fps)
        empty = sv.Detections.empty()
        frames = []
        for index, frame in reader.frames():
            _, matrix = frame_homography(model, homography, frame)
            frames.append(FrameResult(index, index / reader.fps, empty, empty, ball_candidates=empty,
                                      homography=matrix, homography_fit=homography.fit))  # fmt: skip
        result = PipelineResult(frames, reader.fps, reader.size, [], homography.pitch,
                                temporal_filter=homography.temporal_filter, temporal_filter_recorded=True)  # fmt: skip
        if (
            smooth and homography.temporal_filter is not None
        ):  # no filter: no pass to run
            result = smooth_line_homographies(
                result
            )  # with the settings the result records
        pairs.append((result, clip))
    if len(pairs) == 1:
        return evaluate_homography(*pairs[0])
    return evaluate_homography_clips(pairs, draws=draws, seed=seed)
