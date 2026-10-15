"""Team accuracy of a pipeline result against a labelled clip."""

import logging
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment

from tactifoot_vision.data.labelled_clip import UNKNOWN_TEAM, LabelledClip, Truth
from tactifoot_vision.evaluation.metrics import TeamMetrics
from tactifoot_vision.pipeline.result import NO_TEAM, ROLES, PipelineResult

logger = logging.getLogger(__name__)

# The mapped team of a run team id with no true team: scored true team ids
# are NO_TEAM or above (UNKNOWN_TEAM is not scored), so it is never right.
_UNMAPPED = NO_TEAM - 1


class _Matches(NamedTuple):
    """The true people matched to a detection over the scored frames, one entry each."""

    track_ids: np.ndarray  # the detection's tracker_id
    predicted: np.ndarray  # the detection's team_id
    true: np.ndarray  # the true person's team_id
    predicted_roles: np.ndarray  # the detection's class_name
    true_roles: np.ndarray  # the true person's class_name
    instances: int  # true people in the scored frames, matched or not


class _Scored(NamedTuple):
    """One clip's matched people, scored after its own team mapping."""

    matches: _Matches
    right: np.ndarray  # whether each matched person's mapped team is right
    mapping: dict[int, int]
    num_frames: int


def evaluate_teams(
    result: PipelineResult, clip: LabelledClip, iou_threshold: float = 0.5
) -> TeamMetrics:
    """Team accuracy per detection and per track, detection recall and role confusion of ``result`` on ``clip``.

    Only the frames ``result`` processed are scored. In each, true people
    are matched one to one to detected people by Hungarian matching on IoU,
    a pair needing at least ``iou_threshold``; classes are not compared, so a
    player detected as a referee is still scored on their team, and counted
    in the role confusion. The run's team ids are then mapped onto the true
    teams by the one-to-one assignment that agrees on the most matched people
    (Hungarian), so a swapped labelling is not counted as wrong. ``NO_TEAM``
    is right for someone without a true team (a referee) and wrong for anyone
    else; a team given to a referee is wrong. A true person of
    ``UNKNOWN_TEAM`` is matched, so their detection is not matched to anyone
    else, but left out of every number. Teams are read from ``team_id``
    and roles from ``class_name`` after any post-process that changed them.

    Args:
        result: a tracked run on the clip's video, with its frame indices.
        clip: the truth; it must carry boxes and teams.
        iou_threshold: the least IoU at which a detection matches a true person.
    """
    metrics = _metrics([_score(result, clip, iou_threshold)])
    _log(clip.name, metrics)
    return metrics


def evaluate_teams_clips(
    pairs: Iterable[tuple[PipelineResult, LabelledClip]], iou_threshold: float = 0.5
) -> TeamMetrics:
    """The metrics of :func:`evaluate_teams` over several ``(result, clip)`` pairs combined.

    Each clip's run team ids are mapped onto its own true teams, since both
    are arbitrary per run and per clip; the matched people of every clip are
    then pooled, their tracks kept apart, so the numbers are those of one
    clip holding them all with its teams kept apart. ``team_mapping`` is
    ``None`` over several clips, which have one mapping each; ``num_frames``
    is the scored frames of all of them.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("evaluate_teams_clips needs at least one (result, clip) pair")
    metrics = _metrics([_score(result, clip, iou_threshold) for result, clip in pairs])
    _log(f"{len(pairs)} clips", metrics)
    return metrics


def _score(result: PipelineResult, clip: LabelledClip, iou_threshold: float) -> _Scored:
    """``result``'s matched people on ``clip``, right or wrong under its team mapping."""
    matches = _matched_people(result, clip, iou_threshold)
    mapping = _team_mapping(matches.predicted, matches.true)
    mapped = np.where(matches.predicted == NO_TEAM, NO_TEAM, _UNMAPPED)
    for predicted, true in mapping.items():
        mapped[matches.predicted == predicted] = true
    return _Scored(matches, mapped == matches.true, mapping, len(result.frames))


def _metrics(scored: list[_Scored]) -> TeamMetrics:
    """The metrics of every clip's scored people pooled, each clip's tracks apart."""
    right = np.concatenate([s.right for s in scored])
    clips = np.concatenate([np.full(len(s.right), i) for i, s in enumerate(scored)])
    track_ids = np.concatenate([s.matches.track_ids for s in scored])
    matched = len(right)
    instances = sum(s.matches.instances for s in scored)
    return TeamMetrics(
        accuracy=float(right.mean()) if matched else float("nan"),
        track_accuracy=(
            float(pd.Series(right).groupby([clips, track_ids]).mean().mean())
            if matched
            else float("nan")
        ),
        recall=matched / instances if instances else float("nan"),
        team_mapping=scored[0].mapping if len(scored) == 1 else None,
        role_confusion=_role_confusion(
            np.concatenate([s.matches.true_roles for s in scored]),
            np.concatenate([s.matches.predicted_roles for s in scored]),
        ),
        correct=int(right.sum()),
        matched=matched,
        instances=instances,
        num_frames=sum(s.num_frames for s in scored),
    )


def _role_confusion(true: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Matched people counted by true role (rows) and predicted role (columns)."""
    counts = np.zeros((len(ROLES), len(ROLES)), int)
    np.add.at(counts, (_role_index(true), _role_index(predicted)), 1)
    return pd.DataFrame(
        counts,
        index=pd.Index(ROLES, name="true_role"),
        columns=pd.Index(ROLES, name="predicted_role"),
    )


def _role_index(class_names: np.ndarray) -> np.ndarray:
    """Each class name's position in ``ROLES``: its own role, else ``other``."""
    index = np.full(len(class_names), ROLES.index("other"))
    for i, role in enumerate(ROLES):
        index[class_names == role] = i
    return index


def _log(label: str, metrics: TeamMetrics) -> None:
    logger.info(
        "Teams on %s (%d frames): accuracy %.4f, per track %.4f, recall %.4f",
        label,
        metrics.num_frames,
        metrics.accuracy,
        metrics.track_accuracy,
        metrics.recall,
    )


def _matched_people(
    result: PipelineResult, clip: LabelledClip, iou_threshold: float
) -> _Matches:
    """Each processed frame's true people matched one to one to its detected people."""
    missing = {Truth.BOXES, Truth.TEAMS} - clip.truth
    if missing:
        raise ValueError(
            f"Labelled clip {clip.name} has no {' or '.join(sorted(missing))}; "
            "evaluate_teams needs a clip with boxes and teams"
        )
    track_ids, predicted, true, instances = [], [], [], 0
    predicted_roles, true_roles = [], []
    for frame in result.frames:
        try:
            labelled = clip.frame(frame.index)
        except IndexError:
            raise ValueError(
                f"Processed frame {frame.index} is not in labelled clip {clip.name}; "
                "run the pipeline on the clip's video, within its frames"
            ) from None
        detected, truth = frame.detections, labelled.detections
        if len(detected) and detected.tracker_id is None:
            raise ValueError(
                "evaluate_teams needs the people's track ids; the result was "
                "made without a tracker"
            )
        if len(detected) and (missing := _missing_keys(detected)):
            raise ValueError(
                f"The result has no {missing} in frame {frame.index}; "
                "evaluate_teams needs a result with classes and teams"
            )
        if len(truth) and (missing := _missing_keys(truth)):
            raise ValueError(
                f"Labelled clip {clip.name} has no {missing} in frame "
                f"{frame.index}; a clip with boxes and teams gives every person both"
            )
        true_teams = np.asarray(truth.data.get("team_id", []), int)
        if len(truth) and np.min(true_teams) < UNKNOWN_TEAM:
            raise ValueError(
                f"Labelled clip {clip.name} has a team_id below UNKNOWN_TEAM "
                f"({UNKNOWN_TEAM}) in frame {frame.index}; true teams are ids "
                "from 0, NO_TEAM for people without a team or UNKNOWN_TEAM for "
                "people whose team is unknown"
            )
        instances += int(np.sum(true_teams != UNKNOWN_TEAM))
        rows, cols = _match(truth, detected, iou_threshold)
        scored = true_teams[rows] != UNKNOWN_TEAM
        rows, cols = rows[scored], cols[scored]
        if len(cols):
            track_ids.append(np.asarray(detected.tracker_id, int)[cols])
            predicted.append(frame.team_ids[cols])
            true.append(true_teams[rows])
            predicted_roles.append(np.asarray(detected.data["class_name"], str)[cols])
            true_roles.append(np.asarray(truth.data["class_name"], str)[rows])
    return _Matches(
        track_ids=_joined(track_ids, int),
        predicted=_joined(predicted, int),
        true=_joined(true, int),
        predicted_roles=_joined(predicted_roles, str),
        true_roles=_joined(true_roles, str),
        instances=instances,
    )


def _joined(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    """``parts`` end to end, an empty array of ``dtype`` without any."""
    return np.concatenate([np.zeros(0, dtype), *parts])


def _missing_keys(detections: sv.Detections) -> str:
    """The keys scoring needs that ``detections.data`` lacks, joined by "or"."""
    return " or ".join(k for k in ("class_name", "team_id") if k not in detections.data)


def _match(
    truth: sv.Detections, detected: sv.Detections, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``truth`` and columns of ``detected`` paired by Hungarian matching on IoU."""
    if not len(truth) or not len(detected):
        return np.zeros(0, int), np.zeros(0, int)
    iou = sv.box_iou_batch(truth.xyxy, detected.xyxy)
    score = np.where(iou >= iou_threshold, iou, 0.0)
    rows, cols = linear_sum_assignment(-score)
    kept = score[rows, cols] > 0
    return rows[kept], cols[kept]


def _team_mapping(predicted: np.ndarray, true: np.ndarray) -> dict[int, int]:
    """Run team id -> true team id: the one-to-one assignment that agrees most.

    Only people with a team on both sides vote; a pair that never agrees is
    left out, so its run team id is wrong wherever it is given. Between
    assignments that agree equally often, scipy's order decides.
    """
    both = (predicted != NO_TEAM) & (true != NO_TEAM)
    predicted_ids, rows = np.unique(predicted[both], return_inverse=True)
    true_ids, cols = np.unique(true[both], return_inverse=True)
    agree = np.zeros((len(predicted_ids), len(true_ids)))
    np.add.at(agree, (rows, cols), 1)
    rows, cols = linear_sum_assignment(-agree)
    return {
        int(predicted_ids[r]): int(true_ids[c])
        for r, c in zip(rows, cols, strict=True)
        if agree[r, c] > 0
    }
