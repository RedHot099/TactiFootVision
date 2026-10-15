"""Team accuracy of a pipeline result against a labelled clip."""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment

from tactifoot_vision.data.labelled_clip import LabelledClip, Truth
from tactifoot_vision.evaluation.metrics import TeamMetrics
from tactifoot_vision.pipeline.result import NO_TEAM, PipelineResult

logger = logging.getLogger(__name__)

# The mapped team of a run team id with no true team: true team ids are
# NO_TEAM or above, so it is never right.
_UNMAPPED = NO_TEAM - 1


class _Matches(NamedTuple):
    """The true people matched to a detection over the scored frames, one entry each."""

    track_ids: np.ndarray  # the detection's tracker_id
    predicted: np.ndarray  # the detection's team_id
    true: np.ndarray  # the true person's team_id
    instances: int  # true people in the scored frames, matched or not


def evaluate_teams(
    result: PipelineResult, clip: LabelledClip, iou_threshold: float = 0.5
) -> TeamMetrics:
    """Team accuracy per detection and per track, and detection recall, of ``result`` on ``clip``.

    Only the frames ``result`` processed are scored. In each, true people
    are matched one to one to detected people by Hungarian matching on IoU,
    a pair needing at least ``iou_threshold``; classes are not compared, so a
    player detected as a referee is still scored on their team. The run's
    team ids are then mapped onto the true teams by the one-to-one
    assignment that agrees on the most matched people (Hungarian), so a
    swapped labelling is not counted as wrong. ``NO_TEAM`` is right for
    someone without a true team (a referee) and wrong for anyone else; a
    team given to a referee is wrong. Teams are read from ``team_id`` after
    any post-process that changed them.

    Args:
        result: a tracked run on the clip's video, with its frame indices.
        clip: the truth; it must carry boxes and teams.
        iou_threshold: the least IoU at which a detection matches a true person.
    """
    matches = _matched_people(result, clip, iou_threshold)
    mapping = _team_mapping(matches.predicted, matches.true)
    mapped = np.where(matches.predicted == NO_TEAM, NO_TEAM, _UNMAPPED)
    for predicted, true in mapping.items():
        mapped[matches.predicted == predicted] = true
    right = mapped == matches.true
    matched = len(right)
    metrics = TeamMetrics(
        accuracy=float(right.mean()) if matched else float("nan"),
        track_accuracy=(
            float(pd.Series(right).groupby(matches.track_ids).mean().mean())
            if matched
            else float("nan")
        ),
        recall=matched / matches.instances if matches.instances else float("nan"),
        team_mapping=mapping,
        correct=int(right.sum()),
        matched=matched,
        instances=matches.instances,
        num_frames=len(result.frames),
    )
    logger.info(
        "Teams on %s (%d frames): accuracy %.4f, per track %.4f, recall %.4f",
        clip.name,
        metrics.num_frames,
        metrics.accuracy,
        metrics.track_accuracy,
        metrics.recall,
    )
    return metrics


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
        if len(detected) and "team_id" not in detected.data:
            raise ValueError(
                f"The result has no team_id in frame {frame.index}; "
                "evaluate_teams needs a result with teams"
            )
        if len(truth) and "team_id" not in truth.data:
            raise ValueError(
                f"Labelled clip {clip.name} has no team_id in frame "
                f"{frame.index}; a clip with teams gives every person one"
            )
        if len(truth) and np.min(truth.data["team_id"]) < NO_TEAM:
            raise ValueError(
                f"Labelled clip {clip.name} has a team_id below NO_TEAM "
                f"({NO_TEAM}) in frame {frame.index}; true teams are ids from 0, "
                "or NO_TEAM for people without a team"
            )
        instances += len(truth)
        rows, cols = _match(truth, detected, iou_threshold)
        if len(cols):
            track_ids.append(np.asarray(detected.tracker_id, int)[cols])
            predicted.append(frame.team_ids[cols])
            true.append(np.asarray(truth.data["team_id"], int)[rows])
    return _Matches(
        *(
            np.concatenate([np.zeros(0, int), *parts])
            for parts in (track_ids, predicted, true)
        ),
        instances,
    )


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
