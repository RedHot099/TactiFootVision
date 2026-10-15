"""Tracking metrics of a PipelineResult against a labelled clip, on hand-computed fixtures."""

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.data import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.evaluation import TrackingMetrics, evaluate_tracking
from tactifoot_vision.pipeline import FrameResult, PipelineResult, remap_track_ids

# Boxes far enough apart that only the intended pairs overlap.
LEFT = (0, 0, 10, 10)
RIGHT = (100, 0, 110, 10)
FAR = (200, 0, 210, 10)
BALL = (50, 50, 54, 54)
BALL_ELSEWHERE = (300, 50, 304, 54)


def _detections(rows, tracked=True):
    """``sv.Detections`` of ``(track id, class name, box)`` rows."""
    return sv.Detections(
        xyxy=np.array([box for _, _, box in rows], np.float32).reshape(-1, 4),
        tracker_id=np.array([i for i, _, _ in rows], int) if tracked else None,
        data={"class_name": np.array([name for _, name, _ in rows], dtype=str)},
    )


def _clip(frames, truth=frozenset({Truth.IDENTITIES, Truth.BOXES, Truth.BALL})):
    """A labelled clip from ``{index: (people rows, ball rows)}``, every index labelled."""
    return LabelledClip(
        name="fixture",
        video_path=Path("img1/%06d.jpg"),
        fps=25.0,
        width=400,
        height=100,
        frames=tuple(
            LabelledFrame(index, _detections(people), _detections(ball))
            for index, (people, ball) in sorted(frames.items())
        ),
        truth=truth,
    )


def _result(frames):
    """A PipelineResult from ``{index: (people rows, ball boxes)}``; the ball is untracked."""
    results = []
    for index, (people, ball) in sorted(frames.items()):
        ball = _detections([(0, "ball", box) for box in ball], tracked=False)
        results.append(
            FrameResult(
                index=index,
                timestamp=index / 25,
                detections=_detections(people),
                ball=ball,
                ball_candidates=ball,
            )
        )
    return PipelineResult(results, 25.0, (400, 100), ["ball", "player", "referee"])


# Truth: players 1 (left) and 2 (right) and ball 7 in frames 0-3.
TRUTH = {
    i: ([(1, "player", LEFT), (2, "player", RIGHT)], [(7, "ball", BALL)])
    for i in range(4)
}
# The tracker follows player 1 as 11, then as 12 from frame 2 (an ID switch),
# misses player 2 in frame 2 (a miss), sees a player 31 where there is
# nobody in frames 1 and 2 (two false positives), and puts the ball elsewhere
# in frame 3 (a false positive and a miss). The referee is not a class of the
# clip, so it is not scored.
PREDICTED = {
    0: ([(11, "player", LEFT), (21, "player", RIGHT)], [BALL]),
    1: (
        [(11, "player", LEFT), (21, "player", RIGHT), (31, "player", FAR)],
        [BALL],
    ),
    2: ([(12, "player", LEFT), (31, "player", FAR), (41, "referee", FAR)], [BALL]),
    3: ([(12, "player", LEFT), (21, "player", RIGHT)], [BALL_ELSEWHERE]),
}


def test_hand_computed_fixture_gives_exact_metrics():
    metrics = evaluate_tracking(_result(PREDICTED), _clip(TRUTH))

    assert isinstance(metrics, TrackingMetrics)
    assert metrics.num_frames == 4
    players = metrics.per_class.loc["player"]
    # 8 true boxes, 9 predicted: 7 matched, 1 miss, 2 false positives, 1 switch.
    assert players["false_negatives"] == 1
    assert players["false_positives"] == 2
    assert players["id_switches"] == 1
    assert players["instances"] == 8
    assert players["mota"] == pytest.approx(1 - 4 / 8)
    # Best identity pairs 1-11 (or 1-12, 2 frames) and 2-21 (3 frames).
    assert players["idf1"] == pytest.approx(2 * 5 / (8 + 9))
    ball = metrics.per_class.loc["ball"]
    assert (ball["false_negatives"], ball["false_positives"]) == (1, 1)
    assert ball["id_switches"] == 0
    assert ball["mota"] == pytest.approx(1 - 2 / 4)
    assert ball["idf1"] == pytest.approx(2 * 3 / (4 + 4))
    assert list(metrics.per_class.index) == ["ball", "player"]
    # Overall: the counts of every class added up.
    assert (metrics.false_negatives, metrics.false_positives) == (2, 3)
    assert metrics.id_switches == 1
    assert metrics.mota == pytest.approx(1 - 6 / 12)
    assert metrics.idf1 == pytest.approx(2 * 8 / (12 + 13))
    assert metrics.as_dict() == {
        "idf1": metrics.idf1,
        "mota": metrics.mota,
        "id_switches": 1,
        "false_positives": 3,
        "false_negatives": 2,
    }


def test_a_perfect_run_scores_one():
    perfect = {
        i: ([(5, "player", LEFT), (6, "player", RIGHT)], [BALL]) for i in range(4)
    }
    metrics = evaluate_tracking(_result(perfect), _clip(TRUTH))
    assert (metrics.idf1, metrics.mota) == (1.0, 1.0)
    assert (metrics.id_switches, metrics.false_positives) == (0, 0)
    assert metrics.false_negatives == 0


def test_a_track_lost_and_found_again_with_its_id_is_no_switch():
    truth = {i: ([(1, "player", LEFT)], []) for i in range(3)}
    found = ([(11, "player", LEFT)], [])
    predicted = {0: found, 1: ([], []), 2: found}
    metrics = evaluate_tracking(_result(predicted), _clip(truth))
    assert metrics.id_switches == 0
    assert metrics.false_negatives == 1


def test_a_match_is_kept_over_a_frame_without_predictions():
    # Track 11 matches player 1, loses it for a frame, then a better-fitting
    # track 12 appears beside it: as in TrackEval, 11 keeps the match.
    truth = {i: ([(1, "player", LEFT)], []) for i in range(3)}
    predicted = {
        0: ([(11, "player", LEFT)], []),
        1: ([], []),
        2: ([(11, "player", (0, 0, 10, 8)), (12, "player", LEFT)], []),
    }
    metrics = evaluate_tracking(_result(predicted), _clip(truth))
    assert metrics.id_switches == 0
    assert (metrics.false_negatives, metrics.false_positives) == (1, 1)


def test_a_class_without_truth_in_the_scored_frames_has_mota_minus_false_positives():
    truth = dict(TRUTH)
    truth[3] = (TRUTH[3][0] + [(9, "referee", FAR)], TRUTH[3][1])
    predicted = {0: ([(41, "referee", FAR)], [])}
    referee = evaluate_tracking(_result(predicted), _clip(truth)).per_class.loc[
        "referee"
    ]
    assert (referee["instances"], referee["false_positives"]) == (0, 1)
    assert referee["mota"] == -1.0  # TrackEval's (TP - FP - IDSW) / max(1, GT)


def test_a_clip_without_identities_is_an_error():
    clip = _clip(TRUTH, truth=frozenset({Truth.BOXES}))
    with pytest.raises(ValueError, match="fixture.*identities"):
        evaluate_tracking(_result(PREDICTED), clip)


def test_a_strided_run_is_scored_on_its_processed_frames_only():
    perfect = {i: ([(5, "player", LEFT), (6, "player", RIGHT)], [BALL]) for i in (0, 2)}
    metrics = evaluate_tracking(_result(perfect), _clip(TRUTH))
    assert metrics.num_frames == 2
    # Frames 1 and 3 hold truth but were not processed: no misses there.
    assert metrics.false_negatives == 0
    assert metrics.per_class.loc["player", "instances"] == 4
    assert (metrics.idf1, metrics.mota) == (1.0, 1.0)


def test_identity_metrics_follow_a_remapped_tracker_id():
    joined = remap_track_ids(_result(PREDICTED), {12: 11})
    metrics = evaluate_tracking(joined, _clip(TRUTH))
    players = metrics.per_class.loc["player"]
    assert players["id_switches"] == 0
    assert players["mota"] == pytest.approx(1 - 3 / 8)
    assert players["idf1"] == pytest.approx(2 * 7 / (8 + 9))


def test_a_processed_frame_missing_from_the_clip_is_an_error():
    with pytest.raises(ValueError, match="frame 9.*fixture"):
        evaluate_tracking(_result({9: ([], [])}), _clip(TRUTH))


def test_untracked_people_are_an_error():
    result = _result({0: ([(1, "player", LEFT)], [])})
    result.frames[0].detections.tracker_id = None
    with pytest.raises(ValueError, match="track ids"):
        evaluate_tracking(result, _clip(TRUTH))


def test_a_track_id_twice_in_one_frame_is_an_error():
    result = _result({0: ([(1, "player", LEFT), (1, "player", RIGHT)], [])})
    with pytest.raises(ValueError, match="track id 1 twice in frame 0"):
        evaluate_tracking(result, _clip(TRUTH))


def test_a_true_track_id_twice_in_one_frame_is_an_error():
    truth = dict(TRUTH)
    truth[2] = ([(1, "player", LEFT), (1, "player", RIGHT)], TRUTH[2][1])
    with pytest.raises(
        ValueError, match="clip fixture has track id 1 twice in frame 2"
    ):
        evaluate_tracking(_result(PREDICTED), _clip(truth))
