"""Team accuracy of a PipelineResult against a labelled clip, on hand-computed fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import supervision as sv

from tactifoot_vision.data import UNKNOWN_TEAM, LabelledClip, LabelledFrame, Truth
from tactifoot_vision.evaluation import (
    TeamMetrics,
    evaluate_teams,
    evaluate_teams_clips,
)
from tactifoot_vision.pipeline import (
    NO_TEAM,
    FrameResult,
    PipelineResult,
    remap_track_ids,
)

# Boxes far enough apart that only the intended pairs overlap.
A = (0, 0, 10, 10)
B = (100, 0, 110, 10)
C = (200, 0, 210, 10)
G = (250, 0, 260, 10)
R = (300, 0, 310, 10)
EMPTY = (380, 80, 390, 90)  # nobody stands here

TEAMS = frozenset({Truth.IDENTITIES, Truth.BOXES, Truth.TEAMS})


def _detections(rows, tracked=True):
    """``sv.Detections`` of ``(track id, class name, team id, box)`` rows."""
    return sv.Detections(
        xyxy=np.array([box for *_, box in rows], np.float32).reshape(-1, 4),
        tracker_id=np.array([row[0] for row in rows], int) if tracked else None,
        data={
            "class_name": np.array([row[1] for row in rows], dtype=str),
            "team_id": np.array([row[2] for row in rows], int),
        },
    )


def _clip(frames, truth=TEAMS):
    """A labelled clip from ``{index: people rows}``, every index labelled."""
    return LabelledClip(
        name="fixture",
        video_path=Path("img1/%06d.jpg"),
        fps=25.0,
        width=400,
        height=100,
        frames=tuple(
            LabelledFrame(index, _detections(people), _detections([]))
            for index, people in sorted(frames.items())
        ),
        truth=truth,
    )


def _result(frames, tracked=True):
    """A PipelineResult from ``{index: people rows}``, without a ball."""
    no_ball = _detections([])
    return PipelineResult(
        [
            FrameResult(
                index=index,
                timestamp=index / 25,
                detections=_detections(people, tracked),
                ball=no_ball,
                ball_candidates=no_ball,
            )
            for index, people in sorted(frames.items())
        ],
        25.0,
        (400, 100),
        ["ball", "goalkeeper", "player", "referee"],
    )


# Truth: players 1 and 2 of team 0, player 3 of team 1 and referee 4 without a
# team, in frames 0 and 1.
TRUTH = {
    i: [
        (1, "player", 0, A),
        (2, "player", 0, B),
        (3, "player", 1, C),
        (4, "referee", NO_TEAM, R),
    ]
    for i in range(2)
}


def test_swapped_team_ids_score_one():
    swapped = {
        i: [
            (11, "player", 1, A),
            (12, "player", 1, B),
            (13, "player", 0, C),
            (14, "referee", NO_TEAM, R),
        ]
        for i in range(2)
    }

    metrics = evaluate_teams(_result(swapped), _clip(TRUTH))

    assert isinstance(metrics, TeamMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.track_accuracy == 1.0
    assert metrics.recall == 1.0
    assert metrics.team_mapping == {1: 0, 0: 1}


def test_a_referee_given_a_team_and_a_player_given_no_team_are_counted_exactly():
    # Teams 5 and 7 map to true teams 0 and 1 (3 and 1 agreeing detections).
    # Referee track 40 is given team 7 in both frames, player track 20 is
    # NO_TEAM in frame 1, player 3 is missed in frame 1 and track 50 is a
    # detection where nobody stands, which is not scored.
    predicted = {
        0: [
            (10, "player", 5, A),
            (20, "player", 5, B),
            (30, "player", 7, C),
            (40, "referee", 7, R),
        ],
        1: [
            (10, "player", 5, A),
            (20, "player", NO_TEAM, B),
            (40, "referee", 7, R),
            (50, "player", 5, EMPTY),
        ],
    }

    metrics = evaluate_teams(_result(predicted), _clip(TRUTH))

    assert metrics.team_mapping == {5: 0, 7: 1}
    assert metrics.instances == 8
    assert metrics.matched == 7
    assert metrics.correct == 4  # A twice, B and C in frame 0
    assert metrics.accuracy == pytest.approx(4 / 7)
    # Each track counts once: 10 is 2/2 right, 20 1/2, 30 1/1 and 40 0/2.
    assert metrics.track_accuracy == pytest.approx((1 + 0.5 + 1 + 0) / 4)
    assert metrics.recall == pytest.approx(7 / 8)
    assert metrics.num_frames == 2


def test_recall_is_reported_next_to_the_accuracies():
    predicted = {0: [(11, "player", 0, A)]}

    metrics = evaluate_teams(_result(predicted), _clip(TRUTH))

    assert metrics.as_dict() == {
        "accuracy": 1.0,
        "track_accuracy": 1.0,
        "recall": 0.25,
    }


def test_no_team_is_right_for_a_referee_and_wrong_for_a_player():
    predicted = {
        0: [(11, "player", NO_TEAM, A), (14, "referee", NO_TEAM, R)],
    }

    metrics = evaluate_teams(_result(predicted), _clip(TRUTH))

    assert metrics.correct == 1
    assert metrics.accuracy == 0.5


def test_a_team_id_beyond_the_true_teams_is_wrong():
    # Three predicted teams for two true ones: 9, which agrees with true team
    # 0 less often than 0 does, maps to none of them.
    predicted = {
        0: [(11, "player", 0, A), (12, "player", 9, B), (13, "player", 1, C)],
        1: [(11, "player", 0, A)],
    }

    metrics = evaluate_teams(_result(predicted), _clip(TRUTH))

    assert metrics.team_mapping == {0: 0, 1: 1}
    assert metrics.accuracy == pytest.approx(3 / 4)


def test_people_are_matched_by_iou_whatever_their_class():
    # A player detected as a referee is still scored on its team; a box that
    # overlaps player 1 by IoU 1/3 is below the threshold and misses it.
    predicted = {
        0: [(11, "referee", 0, (5, 0, 15, 10)), (12, "referee", 0, B)],
    }

    metrics = evaluate_teams(_result(predicted), _clip(TRUTH))
    loose = evaluate_teams(_result(predicted), _clip(TRUTH), iou_threshold=0.3)

    assert metrics.matched == 1
    assert metrics.accuracy == 1.0
    assert loose.matched == 2


def test_track_accuracy_follows_a_remapped_tracker_id():
    # Player 1 is tracked as 11 with the right team, then as 12 with the wrong
    # one; joining 12 into 11 (a player-track post-process) makes one track.
    predicted = {
        0: [(11, "player", 0, A)],
        1: [(11, "player", 0, A)],
        2: [(12, "player", 1, A)],
    }
    truth = {i: [(1, "player", 0, A), (2, "player", 1, C)] for i in range(3)}
    joined = remap_track_ids(_result(predicted), {12: 11})

    before = evaluate_teams(_result(predicted), _clip(truth))
    after = evaluate_teams(joined, _clip(truth))

    assert before.track_accuracy == 0.5  # 11 right, 12 wrong
    assert after.track_accuracy == pytest.approx(2 / 3)
    assert after.accuracy == pytest.approx(2 / 3)


def test_only_the_processed_frames_are_scored():
    metrics = evaluate_teams(_result({1: TRUTH[1]}), _clip(TRUTH))

    assert metrics.instances == 4
    assert metrics.recall == 1.0
    assert metrics.num_frames == 1


def test_nothing_matched_gives_nan_accuracies():
    metrics = evaluate_teams(_result({0: []}), _clip(TRUTH))

    assert np.isnan(metrics.accuracy)
    assert np.isnan(metrics.track_accuracy)
    assert metrics.recall == 0.0
    assert metrics.team_mapping == {}


def test_a_clip_without_teams_is_an_error():
    clip = _clip(TRUTH, truth=frozenset({Truth.IDENTITIES, Truth.BOXES}))

    with pytest.raises(ValueError, match="fixture has no teams"):
        evaluate_teams(_result(TRUTH), clip)


def test_untracked_people_are_an_error():
    with pytest.raises(ValueError, match="track ids"):
        evaluate_teams(_result(TRUTH, tracked=False), _clip(TRUTH))


def test_a_processed_frame_missing_from_the_clip_is_an_error():
    with pytest.raises(ValueError, match="frame 9 is not in labelled clip fixture"):
        evaluate_teams(_result({9: []}), _clip(TRUTH))


def test_a_result_without_teams_is_an_error():
    result = _result(TRUTH)
    for frame in result.frames:
        del frame.detections.data["team_id"]

    with pytest.raises(ValueError, match="result has no team_id in frame 0"):
        evaluate_teams(result, _clip(TRUTH))


def test_a_clip_frame_without_team_ids_is_an_error():
    clip = _clip(TRUTH)
    del clip.frame(1).detections.data["team_id"]

    with pytest.raises(ValueError, match="fixture has no team_id in frame 1"):
        evaluate_teams(_result(TRUTH), clip)


def test_people_without_a_class_name_are_an_error():
    result, clip = _result(TRUTH), _clip(TRUTH)
    del result.frames[0].detections.data["class_name"]
    del clip.frame(1).detections.data["class_name"]

    with pytest.raises(ValueError, match="result has no class_name in frame 0"):
        evaluate_teams(result, _clip(TRUTH))
    with pytest.raises(ValueError, match="fixture has no class_name in frame 1"):
        evaluate_teams(_result(TRUTH), clip)


def test_people_of_an_unknown_true_team_are_matched_but_not_scored():
    # Player 3's team is unknown in frame 1: whatever the run gives them is
    # not scored, and their detection is not matched to anyone else.
    truth = {0: TRUTH[0], 1: [*TRUTH[1][:2], (3, "player", UNKNOWN_TEAM, C)]}
    predicted = {
        i: [(11, "player", 0, A), (12, "player", 0, B), (13, "player", 1, C)]
        for i in range(2)
    }
    predicted[1][2] = (13, "referee", 0, C)  # wrong team and role, unscored

    metrics = evaluate_teams(_result(predicted), _clip(truth))

    assert (metrics.correct, metrics.matched, metrics.instances) == (5, 5, 6)
    assert metrics.accuracy == 1.0
    assert metrics.team_mapping == {0: 0, 1: 1}
    assert metrics.role_confusion.loc["player", "referee"] == 0


def test_a_true_team_id_below_unknown_team_is_an_error():
    clip = _clip(TRUTH)
    clip.frame(1).detections.data["team_id"][0] = UNKNOWN_TEAM - 1

    with pytest.raises(
        ValueError, match="fixture has a team_id below UNKNOWN_TEAM .* frame 1"
    ):
        evaluate_teams(_result(TRUTH), clip)


def _roles(counts):
    """A role confusion matrix with these ``{(true, predicted): count}``, zero elsewhere."""
    roles = ["player", "goalkeeper", "referee", "other"]
    matrix = pd.DataFrame(
        0,
        index=pd.Index(roles, name="true_role"),
        columns=pd.Index(roles, name="predicted_role"),
    )
    for (true, predicted), count in counts.items():
        matrix.loc[true, predicted] = count
    return matrix


def test_a_goalkeeper_absorbed_into_a_team_appears_in_the_role_matrix_exactly():
    # Goalkeeper 5 of team 0 is detected as a player of run team 3 (true team
    # 0) in both frames, and as a goalkeeper in neither; referee 4 is detected
    # as a player of run team 3 in frame 1, which is the wrong team; a
    # linesman (class "linesman", role other) is detected as a referee.
    truth = {
        i: [*TRUTH[i], (5, "goalkeeper", 0, G), (6, "linesman", NO_TEAM, EMPTY)]
        for i in range(2)
    }
    predicted = {
        0: [
            (10, "player", 3, A),
            (20, "player", 3, B),
            (30, "player", 8, C),
            (40, "referee", NO_TEAM, R),
            (50, "player", 3, G),
            (60, "referee", NO_TEAM, EMPTY),
        ],
        1: [
            (10, "player", 3, A),
            (40, "player", 3, R),
            (50, "player", 3, G),
        ],
    }

    metrics = evaluate_teams(_result(predicted), _clip(truth))

    pd.testing.assert_frame_equal(
        metrics.role_confusion,
        _roles(
            {
                ("player", "player"): 4,
                ("goalkeeper", "player"): 2,
                ("referee", "referee"): 1,
                ("referee", "player"): 1,
                ("other", "referee"): 1,
            }
        ),
    )
    assert metrics.team_mapping == {3: 0, 8: 1}
    assert metrics.matched == 9
    assert metrics.recall == pytest.approx(9 / 12)
    assert metrics.correct == 8  # all but the referee given team 3
    assert metrics.accuracy == pytest.approx(8 / 9)
    # Each track counts once: 40 is 1/2 right, the five others all right.
    assert metrics.track_accuracy == pytest.approx((5 + 0.5) / 6)


def test_the_role_matrix_counts_only_matched_people():
    metrics = evaluate_teams(_result({0: [(11, "player", 0, A)]}), _clip(TRUTH))

    pd.testing.assert_frame_equal(
        metrics.role_confusion, _roles({("player", "player"): 1})
    )


def _moved(frames, first_index, id_offset):
    """``{index: people rows}`` from ``first_index`` on, track and team ids moved by ``id_offset``."""
    return {
        first_index + i: [
            (t + id_offset, name, team if team == NO_TEAM else team + id_offset, box)
            for t, name, team, box in frames[k]
        ]
        for i, k in enumerate(sorted(frames))
    }


def test_the_aggregate_equals_evaluating_the_concatenated_clips():
    # Each clip has its own run and its own teams: the second run's team ids
    # are swapped against the first's, a goalkeeper is absorbed into a team
    # and a referee is given one.
    first = {
        0: [(10, "player", 5, A), (20, "player", 5, B), (30, "player", 7, C)],
        1: [(10, "player", 5, A), (40, "referee", 7, R)],
    }
    second_truth = {
        i: [(1, "player", 0, A), (2, "player", 1, B), (5, "goalkeeper", 0, G)]
        for i in range(3)
    }
    second = {
        0: [(10, "player", 1, A), (20, "player", 0, B), (50, "player", 1, G)],
        1: [(10, "player", 1, A), (50, "player", 0, G)],
        2: [(20, "player", 0, B)],
    }
    pairs = [
        (_result(first), _clip(TRUTH)),
        (_result(second), _clip(second_truth)),
    ]
    # One clip holding both, its people and teams kept apart: the second
    # clip's frames follow the first's and its track and team ids, true and
    # run, are moved by 100.
    concatenated = evaluate_teams(
        _result(first | _moved(second, 2, 100)),
        _clip(TRUTH | _moved(second_truth, 2, 100)),
    )

    combined = evaluate_teams_clips(pairs)

    assert concatenated.team_mapping == {5: 0, 7: 1, 101: 100, 100: 101}
    assert combined.team_mapping is None
    assert combined.num_frames == concatenated.num_frames == 5
    assert (combined.correct, combined.matched, combined.instances) == (
        concatenated.correct,
        concatenated.matched,
        concatenated.instances,
    )
    assert combined.correct < combined.matched < combined.instances
    assert combined.as_dict() == pytest.approx(concatenated.as_dict())
    pd.testing.assert_frame_equal(combined.role_confusion, concatenated.role_confusion)
    assert combined.role_confusion.loc["goalkeeper", "player"] == 2


def test_the_aggregate_of_one_clip_is_that_clip():
    single = evaluate_teams(_result(TRUTH), _clip(TRUTH))

    combined = evaluate_teams_clips([(_result(TRUTH), _clip(TRUTH))])

    assert combined.as_dict() == single.as_dict()
    assert combined.team_mapping == single.team_mapping == {0: 0, 1: 1}
    assert (combined.correct, combined.matched, combined.instances) == (8, 8, 8)
    pd.testing.assert_frame_equal(combined.role_confusion, single.role_confusion)


def test_the_aggregate_of_no_clips_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_teams_clips([])
