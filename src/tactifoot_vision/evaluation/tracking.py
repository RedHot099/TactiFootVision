"""Multi-object tracking metrics of a pipeline result against a labelled clip."""

import logging
from collections import Counter
from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment

from tactifoot_vision.data.labelled_clip import LabelledClip, Truth
from tactifoot_vision.evaluation.metrics import TrackingMetrics
from tactifoot_vision.pipeline.result import FrameResult, PipelineResult

logger = logging.getLogger(__name__)

_BALL_ID = 0  # track id of a ball the pipeline did not track: there is one ball
_CONTINUING = 1000.0  # TrackEval's bonus that keeps last frame's matches
_EPS = np.finfo(float).eps

_COLUMNS = [
    "idf1", "mota", "id_switches", "false_positives", "false_negatives", "instances",
]  # fmt: skip


class _Objects(NamedTuple):
    """The objects of one class in one frame."""

    ids: np.ndarray  # track ids
    xyxy: np.ndarray


_EMPTY = _Objects(np.zeros(0, int), np.zeros((0, 4)))


@dataclass
class _Counts:
    """What the metrics of one class, or of several added up, are computed from."""

    instances: int = 0  # true boxes
    predicted: int = 0  # predicted boxes
    matched: int = 0  # boxes matched frame by frame (CLEAR)
    id_switches: int = 0
    id_matched: int = 0  # boxes matched to their track's true identity (IDTP)

    def __add__(self, other: "_Counts") -> "_Counts":
        return _Counts(
            *(getattr(self, f.name) + getattr(other, f.name) for f in fields(self))
        )

    @property
    def false_negatives(self) -> int:
        return self.instances - self.matched

    @property
    def false_positives(self) -> int:
        return self.predicted - self.matched

    @property
    def mota(self) -> float:
        # As TrackEval: without true boxes it is -false_positives, not NaN.
        correct = self.matched - self.false_positives - self.id_switches
        return correct / max(1, self.instances)

    @property
    def idf1(self) -> float:
        return 2 * self.id_matched / max(2, self.instances + self.predicted)


def evaluate_tracking(
    result: PipelineResult, clip: LabelledClip, iou_threshold: float = 0.5
) -> TrackingMetrics:
    """IDF1, MOTA, ID switches, false positives and false negatives of ``result`` on ``clip``.

    Only the frames ``result`` processed are scored, so a run with ``stride``
    or a frame range is not charged with misses in frames it skipped. Each
    class of the clip (people and the ball) is scored on its own, the
    predictions matched to it by class name; predicted classes the clip does
    not have are left out. Identities are the people's ``tracker_id``, as a
    tracker or a post-process that remaps them left it; a ball without one is
    a single identity. The definitions follow TrackEval: a box matches a true
    box when their IoU is at least ``iou_threshold``, frame by frame with
    Hungarian matching on IoU that keeps last frame's matches (CLEAR MOT),
    and over the whole run with a one-to-one pairing of identities (IDF1).

    Args:
        result: a tracked run on the clip's video, with its frame indices.
        clip: the truth; it must carry identities and boxes.
        iou_threshold: the least IoU at which a box matches a true box.
    """
    missing = {Truth.IDENTITIES, Truth.BOXES} - clip.truth
    if missing:
        raise ValueError(
            f"Labelled clip {clip.name} has no {' or '.join(sorted(missing))}; "
            "evaluate_tracking needs a clip with identities and boxes"
        )
    truth, predicted = [], []
    for frame in result.frames:
        try:
            labelled = clip.frame(frame.index)
        except IndexError:
            raise ValueError(
                f"Processed frame {frame.index} is not in labelled clip {clip.name}; "
                "run the pipeline on the clip's video, within its frames"
            ) from None
        objects = _by_class(labelled.detections, labelled.ball)
        for name, (ids, _) in objects.items():
            duplicate = _first_duplicate(ids)
            if duplicate is not None:
                raise ValueError(
                    f"Labelled clip {clip.name} has track id {duplicate} twice in "
                    f"frame {frame.index} ({name}); give each object its own id"
                )
        truth.append(objects)
        predicted.append(_predicted(frame))

    class_names = sorted(
        {name for f in clip for name in _by_class(f.detections, f.ball)}
    )
    ignored = sorted({name for p in predicted for name in p} - set(class_names))
    if ignored:
        logger.warning(
            "Ignoring predicted classes that clip %s does not have: %s",
            clip.name,
            ", ".join(ignored),
        )
    counts = {
        name: _count(
            [t.get(name, _EMPTY) for t in truth],
            [p.get(name, _EMPTY) for p in predicted],
            iou_threshold,
        )
        for name in class_names
    }
    total = sum(counts.values(), _Counts())
    per_class = pd.DataFrame(
        [[getattr(c, column) for column in _COLUMNS] for c in counts.values()],
        index=pd.Index(list(counts), name="class_name"),
        columns=_COLUMNS,
    )
    metrics = TrackingMetrics(
        idf1=total.idf1,
        mota=total.mota,
        id_switches=total.id_switches,
        false_positives=total.false_positives,
        false_negatives=total.false_negatives,
        per_class=per_class,
        num_frames=len(result.frames),
    )
    logger.info(
        "Tracking on %s (%d frames): IDF1 %.4f, MOTA %.4f, %d ID switches",
        clip.name,
        metrics.num_frames,
        metrics.idf1,
        metrics.mota,
        metrics.id_switches,
    )
    return metrics


def _predicted(frame: FrameResult) -> dict[str, _Objects]:
    """The frame's people and ball by class, an untracked ball given ``_BALL_ID``."""
    people = frame.detections
    if len(people) and people.tracker_id is None:
        raise ValueError(
            "evaluate_tracking needs the people's track ids; the result "
            "was made without a tracker"
        )
    duplicate = _first_duplicate(people.tracker_id)
    if duplicate is not None:
        raise ValueError(
            f"The result has track id {duplicate} twice in frame "
            f"{frame.index}; a remap must not join tracks that overlap in time"
        )
    ball = frame.ball
    if ball.tracker_id is None:
        ball = _with_ids(ball, np.full(len(ball), _BALL_ID))
    return _by_class(people, ball)


def _first_duplicate(ids: np.ndarray | None) -> int | None:
    """A track id that occurs more than once in ``ids``, if any (TrackEval rejects them)."""
    repeated = [i for i, n in Counter(ids).items() if n > 1]
    return int(repeated[0]) if repeated else None


def _with_ids(detections: sv.Detections, ids: np.ndarray) -> sv.Detections:
    return sv.Detections(
        xyxy=detections.xyxy, tracker_id=ids, data=dict(detections.data)
    )


def _by_class(*groups: sv.Detections) -> dict[str, _Objects]:
    """Class name -> objects of the detections in ``groups``."""
    objects: dict[str, _Objects] = {}
    for detections in groups:
        if len(detections) == 0:
            continue
        names = np.asarray(detections.data["class_name"]).astype(str)
        for name in np.unique(names):
            here = names == name
            ids, xyxy = objects.get(name, _EMPTY)
            objects[name] = _Objects(
                np.concatenate([ids, np.asarray(detections.tracker_id)[here]]),
                np.concatenate([xyxy, np.asarray(detections.xyxy)[here]]),
            )
    return objects


def _count(
    truth: list[_Objects], predicted: list[_Objects], iou_threshold: float
) -> _Counts:
    """Counts of one class over the scored frames, ``truth[i]`` against ``predicted[i]``."""
    counts = _Counts()
    last_frame_match: dict[int, int] = {}  # true id -> predicted id, last frame
    last_match: dict[int, int] = {}  # true id -> predicted id it last matched
    overlaps: Counter[tuple[int, int]] = Counter()  # (true, predicted) id -> frames
    for (true_ids, true_xyxy), (ids, xyxy) in zip(truth, predicted, strict=True):
        true_ids, ids = true_ids.astype(int).tolist(), ids.astype(int).tolist()
        counts.instances += len(true_ids)
        counts.predicted += len(ids)
        if not true_ids or not ids:
            continue  # as TrackEval: last frame's matches carry over
        iou = sv.box_iou_batch(true_xyxy, xyxy)
        for row, col in zip(*np.nonzero(iou >= iou_threshold), strict=True):
            overlaps[true_ids[row], ids[col]] += 1
        close = iou >= iou_threshold - _EPS  # TrackEval's CLEAR tolerance
        continuing = np.array(
            [[last_frame_match.get(t) == p for p in ids] for t in true_ids]
        )
        score = np.where(close, _CONTINUING * continuing + iou, 0.0)
        rows, cols = linear_sum_assignment(-score)
        matches = [
            (true_ids[r], ids[c])
            for r, c in zip(rows, cols, strict=True)
            if score[r, c] > _EPS
        ]
        counts.matched += len(matches)
        counts.id_switches += sum(
            t in last_match and last_match[t] != p for t, p in matches
        )
        last_match.update(matches)
        last_frame_match = dict(matches)
    counts.id_matched = _best_identity_overlap(overlaps)
    return counts


def _best_identity_overlap(overlaps: Counter[tuple[int, int]]) -> int:
    """Matched boxes of the one-to-one pairing of identities that matches the most.

    ``overlaps`` counts the frames in which a true and a predicted identity
    match; identities that never match add nothing to the pairing.
    """
    if not overlaps:
        return 0
    true_index = {t: i for i, t in enumerate(dict.fromkeys(t for t, _ in overlaps))}
    predicted_index = {
        p: i for i, p in enumerate(dict.fromkeys(p for _, p in overlaps))
    }
    frames = np.zeros((len(true_index), len(predicted_index)))
    for (t, p), n in overlaps.items():
        frames[true_index[t], predicted_index[p]] = n
    rows, cols = linear_sum_assignment(-frames)
    return int(frames[rows, cols].sum())
