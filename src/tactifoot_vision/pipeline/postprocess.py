"""Post-processes: whole-video steps on a finished :class:`PipelineResult` (ADR 0002).

A post-process is a function ``(result, **options) -> PipelineResult`` that
returns a new result and never changes its input (the new result may share
the parts it leaves unchanged). Register it by name::

    @POSTPROCESSES.register("join_tracks")
    def join_tracks(result: PipelineResult, max_gap: float = 1.6) -> PipelineResult: ...

    joined = join_tracks(result, max_gap=2.0)          # notebook mode
    joined = POSTPROCESSES.get("join_tracks")(result)  # by name, as run files do

Registering wraps the function, keeping its signature, so that every call
appends an :class:`AppliedPostProcess` (the name and every option, defaults
included) to the returned result's ``postprocesses``. When a registered
post-process calls another, only the outer call is recorded. A post-process that
moves positions or changes identities keeps the result consistent with
:func:`reproject` and :func:`remap_track_ids`, and measures in metres with
:func:`metres_per_unit`.
"""

import copy
import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from tactifoot_vision.enums import DetectionKey
from tactifoot_vision.pipeline.result import (
    BALL_ANCHOR,
    PERSON_ANCHOR,
    AppliedPostProcess,
    PipelineResult,
    project_detections,
)
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.registry import Registry

_STD_SIZE = np.array([105.0, 68.0])  # metres of the pitch every pitch unit maps to


class _PostProcessRegistry(Registry[PipelineResult]):
    def register[F: Callable[..., Any]](self, name: str) -> Callable[[F], F]:
        name = str(name)
        add = super().register(name)

        def decorator(function: F) -> F:
            signature = inspect.signature(function)

            @functools.wraps(function)
            def recorded(*args: Any, **kwargs: Any) -> PipelineResult:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                result, *_ = bound.arguments.values()
                # copied before the call, so neither the post-process nor the
                # caller changing an option afterwards rewrites the record
                record = AppliedPostProcess(
                    name, copy.deepcopy(_options(signature, bound))
                )
                output = function(*args, **kwargs)
                if not isinstance(output, PipelineResult):
                    raise TypeError(
                        f"Post-process {name!r} returned {type(output).__name__}; "
                        "it must return a new PipelineResult"
                    )
                history = copy.deepcopy(result.postprocesses)
                return dataclasses.replace(output, postprocesses=[*history, record])

            add(recorded)
            return cast(F, recorded)

        return decorator


POSTPROCESSES = _PostProcessRegistry("post-process")


def _options(
    signature: inspect.Signature, bound: inspect.BoundArguments
) -> dict[str, Any]:
    """Every argument but the result, with ``**kwargs`` spread out."""
    options: dict[str, Any] = {}
    for key, value in list(bound.arguments.items())[1:]:
        if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
            options |= value
        else:
            options[key] = value
    return options


def metres_per_unit(pitch: SoccerPitch) -> np.ndarray:
    """Metres per pitch unit along the length and the width.

    A pitch of ``length x width`` units is taken as a standard 105 x 68 m
    pitch, as :class:`~tactifoot_vision.pitch.SoccerPitch` scales its markings.
    """
    return _STD_SIZE / [pitch.length, pitch.width]


def reproject(
    result: PipelineResult, homographies: Sequence[np.ndarray | None]
) -> PipelineResult:
    """Copy of ``result`` with new frame->pitch homographies and the positions they give.

    ``homographies[i]`` replaces ``result.frames[i].homography``; ``None``
    means no homography and NaN positions. ``pitch_xy`` is projected from the
    boxes as :meth:`Pipeline.run` does it: people at the bottom centre, the
    ball and its candidates at their centre. It does not choose the ball
    again: run :func:`~tactifoot_vision.pipeline.track_ball` after it.
    """
    if len(homographies) != len(result.frames):
        raise ValueError(
            f"The result has {len(result.frames)} frames but got "
            f"{len(homographies)} homographies; pass one per frame (None for none)"
        )
    frames = []
    for frame, homography in zip(result.frames, homographies, strict=True):
        matrix = None if homography is None else np.array(homography, dtype=float)
        people = copy.deepcopy(frame.detections)
        people.data[DetectionKey.PITCH_XY] = project_detections(
            people, PERSON_ANCHOR, matrix
        )
        ball = copy.deepcopy(frame.ball)
        candidates = copy.deepcopy(frame.ball_candidates)
        for balls in (ball, candidates):
            balls.data[DetectionKey.PITCH_XY] = project_detections(
                balls, BALL_ANCHOR, matrix
            )
        frames.append(
            dataclasses.replace(
                frame,
                detections=people,
                ball=ball,
                ball_candidates=candidates,
                homography=matrix,
            )
        )
    return dataclasses.replace(result, frames=frames)


def remap_track_ids(
    result: PipelineResult, mapping: Mapping[int, int]
) -> PipelineResult:
    """Copy of ``result`` whose people's ``tracker_id`` go through ``mapping``.

    Ids missing from ``mapping`` stay as they are. The tracker's own id is
    kept in ``detections.data["original_tracker_id"]``, also over several
    remaps. Frames without a tracker are left alone.
    """
    frames = []
    for frame in result.frames:
        people = frame.detections
        if people.tracker_id is not None:
            people = copy.deepcopy(people)
            people.data.setdefault(
                DetectionKey.ORIGINAL_TRACKER_ID, people.tracker_id.copy()
            )
            people.tracker_id = np.array(
                [mapping.get(int(i), int(i)) for i in people.tracker_id],
                dtype=people.tracker_id.dtype,
            )
        frames.append(dataclasses.replace(frame, detections=people))
    return dataclasses.replace(result, frames=frames)
