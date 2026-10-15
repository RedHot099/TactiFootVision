"""Tracking metrics of a PipelineResult against a labelled clip, on hand-computed fixtures."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import supervision as sv

from tactifoot_vision.data import (
    LabelledClip,
    LabelledFrame,
    Truth,
    load_labelled_clip,
)
from tactifoot_vision.evaluation import (
    TrackingMetrics,
    evaluate_tracking,
    evaluate_tracking_clips,
)
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
    metrics = evaluate_tracking(_result(PREDICTED), _clip(TRUTH), hota=False)

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
    as_dict = metrics.as_dict()
    assert all(np.isnan(as_dict.pop(name)) for name in ("hota", "deta", "assa"))
    assert as_dict == {
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
    metrics = evaluate_tracking(_result(perfect), _clip(TRUTH), hota=False)
    assert (metrics.idf1, metrics.mota) == (1.0, 1.0)
    assert (metrics.id_switches, metrics.false_positives) == (0, 0)
    assert metrics.false_negatives == 0


def test_a_track_lost_and_found_again_with_its_id_is_no_switch():
    truth = {i: ([(1, "player", LEFT)], []) for i in range(3)}
    found = ([(11, "player", LEFT)], [])
    predicted = {0: found, 1: ([], []), 2: found}
    metrics = evaluate_tracking(_result(predicted), _clip(truth), hota=False)
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
    metrics = evaluate_tracking(_result(predicted), _clip(truth), hota=False)
    assert metrics.id_switches == 0
    assert (metrics.false_negatives, metrics.false_positives) == (1, 1)


def test_a_class_without_truth_in_the_scored_frames_has_mota_minus_false_positives():
    truth = dict(TRUTH)
    truth[3] = (TRUTH[3][0] + [(9, "referee", FAR)], TRUTH[3][1])
    predicted = {0: ([(41, "referee", FAR)], [])}
    referee = evaluate_tracking(
        _result(predicted), _clip(truth), hota=False
    ).per_class.loc["referee"]
    assert (referee["instances"], referee["false_positives"]) == (0, 1)
    assert referee["mota"] == -1.0  # TrackEval's (TP - FP - IDSW) / max(1, GT)


def test_a_clip_without_identities_is_an_error():
    clip = _clip(TRUTH, truth=frozenset({Truth.BOXES}))
    with pytest.raises(ValueError, match="fixture.*identities"):
        evaluate_tracking(_result(PREDICTED), clip, hota=False)


def test_a_strided_run_is_scored_on_its_processed_frames_only():
    perfect = {i: ([(5, "player", LEFT), (6, "player", RIGHT)], [BALL]) for i in (0, 2)}
    metrics = evaluate_tracking(_result(perfect), _clip(TRUTH), hota=False)
    assert metrics.num_frames == 2
    # Frames 1 and 3 hold truth but were not processed: no misses there.
    assert metrics.false_negatives == 0
    assert metrics.per_class.loc["player", "instances"] == 4
    assert (metrics.idf1, metrics.mota) == (1.0, 1.0)


def test_identity_metrics_follow_a_remapped_tracker_id():
    joined = remap_track_ids(_result(PREDICTED), {12: 11})
    metrics = evaluate_tracking(joined, _clip(TRUTH), hota=False)
    players = metrics.per_class.loc["player"]
    assert players["id_switches"] == 0
    assert players["mota"] == pytest.approx(1 - 3 / 8)
    assert players["idf1"] == pytest.approx(2 * 7 / (8 + 9))


def test_a_processed_frame_missing_from_the_clip_is_an_error():
    with pytest.raises(ValueError, match="frame 9.*fixture"):
        evaluate_tracking(_result({9: ([], [])}), _clip(TRUTH), hota=False)


def test_untracked_people_are_an_error():
    result = _result({0: ([(1, "player", LEFT)], [])})
    result.frames[0].detections.tracker_id = None
    with pytest.raises(ValueError, match="track ids"):
        evaluate_tracking(result, _clip(TRUTH), hota=False)


def test_a_track_id_twice_in_one_frame_is_an_error():
    result = _result({0: ([(1, "player", LEFT), (1, "player", RIGHT)], [])})
    with pytest.raises(ValueError, match="track id 1 twice in frame 0"):
        evaluate_tracking(result, _clip(TRUTH), hota=False)


def test_a_true_track_id_twice_in_one_frame_is_an_error():
    truth = dict(TRUTH)
    truth[2] = ([(1, "player", LEFT), (1, "player", RIGHT)], TRUTH[2][1])
    with pytest.raises(
        ValueError, match="clip fixture has track id 1 twice in frame 2"
    ):
        evaluate_tracking(_result(PREDICTED), _clip(truth), hota=False)


def test_hota_without_trackeval_is_an_error_that_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "trackeval", None)
    with pytest.raises(ImportError, match=r"uv sync --extra eval.*hota=False"):
        evaluate_tracking(_result(PREDICTED), _clip(TRUTH))
    metrics = evaluate_tracking(_result(PREDICTED), _clip(TRUTH), hota=False)
    assert np.isnan([metrics.hota, metrics.deta, metrics.assa]).all()
    assert metrics.per_class[["hota", "deta", "assa"]].isna().all(axis=None)


def test_a_broken_trackeval_is_not_reported_as_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "trackeval", ModuleType("trackeval"))
    monkeypatch.setitem(sys.modules, "trackeval.metrics", ModuleType("metrics"))
    with pytest.raises(ImportError, match="cannot import name 'HOTA'"):
        evaluate_tracking(_result(PREDICTED), _clip(TRUTH))


needs_trackeval = pytest.mark.skipif(
    importlib.util.find_spec("trackeval") is None,
    reason="needs TrackEval: uv sync --extra eval",
)


def _noisy_fixture():
    """Three players over ten frames, seen with noisy boxes and these mistakes:
    player 1 changes track at frame 5, player 2 is missed in frame 3, players
    2 and 3 swap tracks at frame 6, and a track 41 follows nobody in frames 2-4.
    """
    rng = np.random.default_rng(0)

    def box(x):
        return (x, 20, x + 20, 60)

    def noisy(box):
        return tuple(float(v) for v in np.round(box + rng.uniform(-3, 3, 4), 2))

    truth, predicted = {}, {}
    for i in range(10):
        one, two, three = box(10 + 5 * i), box(100 + 3 * i), box(200 - 4 * i)
        truth[i] = ([(1, "player", one), (2, "player", two), (3, "player", three)], [])
        rows = [(11 if i < 5 else 12, "player", noisy(one))]
        if i != 3:
            rows.append((21 if i < 6 else 31, "player", noisy(two)))
        rows.append((31 if i < 6 else 21, "player", noisy(three)))
        if 2 <= i <= 4:
            rows.append((41, "player", noisy(box(300))))
        predicted[i] = (rows, [])
    return truth, predicted


def _write_mot_truth(path, frames):
    """The players of ``{index: (people rows, ball)}`` as MOTChallenge ground truth."""
    tail = "1,1,1"  # not ignored, pedestrian, visible
    lines = [
        f"{index + 1},{track_id},{x1},{y1},{x2 - x1},{y2 - y1},{tail}"
        for index, (people, _) in sorted(frames.items())
        for track_id, name, (x1, y1, x2, y2) in people
        if name == "player"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _trackeval_on_mot(tmp_path, monkeypatch, truth, predicted):
    """TrackEval's own HOTA, CLEAR and Identity results for the players, read from MOT files."""
    import trackeval

    _write_mot_truth(tmp_path / "gt" / "clip" / "gt" / "gt.txt", truth)
    # The tracker's file is the export of its players, so TrackEval reads it.
    players = {
        i: ([row for row in people if row[1] == "player"], [])
        for i, (people, _) in predicted.items()
    }
    _result(players).to_mot(tmp_path / "trackers" / "ours" / "data" / "clip.txt")
    quiet = {"PRINT_CONFIG": False}
    dataset = trackeval.datasets.MotChallenge2DBox(
        {
            **quiet,
            "GT_FOLDER": str(tmp_path / "gt"),
            "TRACKERS_FOLDER": str(tmp_path / "trackers"),
            "OUTPUT_FOLDER": str(tmp_path / "out"),
            "TRACKERS_TO_EVAL": ["ours"],
            "SEQ_INFO": {"clip": len(truth)},
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,  # no MOT17 distractor classes here
        }
    )
    evaluator = trackeval.Evaluator(
        {
            **quiet,
            "PRINT_RESULTS": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "LOG_ON_ERROR": None,
        }
    )
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(quiet),
        trackeval.metrics.Identity(quiet),
    ]
    with monkeypatch.context() as patch:
        # TrackEval still uses np.float and np.int, which numpy 1.24 removed.
        # Lent here, not by hota._numpy_aliases, so the reference run does not
        # depend on the code under test.
        patch.setattr(np, "float", float, raising=False)
        patch.setattr(np, "int", int, raising=False)
        results, _ = evaluator.evaluate([dataset], metrics)
    return results["MotChallenge2DBox"]["ours"]["clip"]["pedestrian"]


@needs_trackeval
@pytest.mark.parametrize(
    ("truth", "predicted"),
    [(TRUTH, PREDICTED), _noisy_fixture()],
    ids=["hand", "noisy"],
)
def test_native_idf1_and_mota_agree_with_trackeval(
    tmp_path, monkeypatch, truth, predicted
):
    reference = _trackeval_on_mot(tmp_path, monkeypatch, truth, predicted)
    metrics = evaluate_tracking(_result(predicted), _clip(truth), hota=False)
    players = metrics.per_class.loc["player"]
    clear = reference["CLEAR"]
    assert clear["IDSW"] > 0 and clear["CLR_FP"] > 0 and clear["CLR_FN"] > 0
    assert players["idf1"] == pytest.approx(reference["Identity"]["IDF1"])
    assert players["mota"] == pytest.approx(clear["MOTA"])
    assert players["id_switches"] == clear["IDSW"]
    assert players["false_positives"] == clear["CLR_FP"]
    assert players["false_negatives"] == clear["CLR_FN"]


@needs_trackeval
def test_hota_matches_trackeval_on_a_fixture(tmp_path, monkeypatch):
    truth, predicted = _noisy_fixture()
    metrics = evaluate_tracking(_result(predicted), _clip(truth))
    reference = _trackeval_on_mot(tmp_path, monkeypatch, truth, predicted)["HOTA"]
    expected = [np.mean(reference[name]) for name in ("HOTA", "DetA", "AssA")]
    assert 0 < expected[0] < 1
    players = metrics.per_class.loc["player"]
    assert list(players[["hota", "deta", "assa"]]) == pytest.approx(expected)
    # Players are the only class, so they are the overall numbers too.
    assert [metrics.hota, metrics.deta, metrics.assa] == pytest.approx(expected)
    assert metrics.as_dict()["hota"] == metrics.hota


@needs_trackeval
def test_overall_hota_adds_up_the_detections_of_every_class():
    # The ball never overlaps a player, so adding up the classes as TrackEval
    # does gives the HOTA of scoring them all as one class.
    truth, predicted = _noisy_fixture()
    ball = [(150, 70, 160, 80), (152, 70, 162, 80), (300, 0, 310, 10)]
    truth = {
        i: (people, [(7, "ball", ball[i % 2])]) for i, (people, _) in truth.items()
    }
    predicted = {
        i: (people, [ball[i % 3]] if i else []) for i, (people, _) in predicted.items()
    }
    by_class = evaluate_tracking(_result(predicted), _clip(truth))

    one_class_truth = {
        i: (people + [(t, "player", box) for t, _, box in ball], [])
        for i, (people, ball) in truth.items()
    }
    one_class = {
        i: (people + [(99, "player", box) for box in ball], [])
        for i, (people, ball) in predicted.items()
    }
    together = evaluate_tracking(_result(one_class), _clip(one_class_truth))
    hota = by_class.per_class["hota"]
    assert hota["ball"] != pytest.approx(hota["player"])
    assert [by_class.hota, by_class.deta, by_class.assa] == pytest.approx(
        [together.hota, together.deta, together.assa]
    )


@needs_trackeval
def test_a_clip_without_objects_has_hota_zero():
    metrics = evaluate_tracking(_result({0: ([], [])}), _clip({0: ([], [])}))
    assert (metrics.hota, metrics.deta, metrics.assa) == (0.0, 0.0, 0.0)
    assert metrics.per_class.empty


def test_to_mot_writes_one_row_per_object_with_1_based_frames(tmp_path):
    result = _result({0: ([(11, "player", (0, 0, 10, 20))], [BALL]), 2: ([], [])})
    result.frames[0].detections.confidence = np.array([0.5])

    path = result.to_mot(tmp_path / "run" / "mot.txt")

    # The untracked ball is one track, after the people's largest id.
    assert path.read_text().splitlines() == [
        "1,11,0.00,0.00,10.00,20.00,0.5,-1,-1,-1",
        "1,12,50.00,50.00,4.00,4.00,-1,-1,-1,-1",
    ]


def test_to_mot_needs_track_ids(tmp_path):
    result = _result({0: ([(1, "player", LEFT)], [])})
    result.frames[0].detections.tracker_id = None
    with pytest.raises(ValueError, match="track ids"):
        result.to_mot(tmp_path / "mot.txt")


@pytest.mark.parametrize("hota", [False, pytest.param(True, marks=needs_trackeval)])
def test_mot_export_round_trips_into_the_evaluator(tmp_path, hota):
    # The export, read back as a SoccerNet Tracking sequence, is the truth the
    # same result scores perfectly on: same frames, boxes and identities.
    result = _result({i: PREDICTED[i] for i in (0, 1, 3)})  # frame 2 skipped
    folder = tmp_path / "SNMOT-001"
    result.to_mot(folder / "gt" / "gt.txt")
    (folder / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-001\nframeRate=25\nseqLength=4\n"
        "imWidth=400\nimHeight=100\n"
    )
    # The referee 41 is only in frame 2, so the ball's id follows player 31.
    classes = {11: "player", 12: "player", 21: "player", 31: "player", 32: "ball"}
    (folder / "gameinfo.ini").write_text(
        "[Sequence]\n"
        + "".join(f"trackletID_{i}= {name};1\n" for i, name in classes.items())
    )

    clip = load_labelled_clip(folder)
    metrics = evaluate_tracking(result, clip, hota=hota)

    assert metrics.num_frames == 3
    assert list(metrics.per_class.index) == ["ball", "player"]
    assert list(metrics.per_class["instances"]) == [3, 7]
    assert (metrics.idf1, metrics.mota) == (1.0, 1.0)
    assert (metrics.false_positives, metrics.false_negatives) == (0, 0)
    assert metrics.id_switches == 0
    if hota:
        assert metrics.hota == pytest.approx(1.0)


def _shifted(frames, first_index, id_offset):
    """``{index: (people rows, ball)}`` from ``first_index`` on, people ids moved by ``id_offset``."""
    return {
        first_index + i: ([(t + id_offset, n, box) for t, n, box in people], ball)
        for i, (people, ball) in enumerate(frames[k] for k in sorted(frames))
    }


@pytest.mark.parametrize("hota", [False, pytest.param(True, marks=needs_trackeval)])
def test_the_aggregate_equals_evaluating_the_concatenated_clips(hota):
    noisy_truth, noisy_predicted = _noisy_fixture()
    pairs = [
        (_result(PREDICTED), _clip(TRUTH)),
        (_result(noisy_predicted), _clip(noisy_truth)),
    ]
    # One clip holding both, its identities kept apart: the second clip's
    # frames follow the first's and its track ids are moved by 100.
    after = len(TRUTH)
    concatenated = evaluate_tracking(
        _result(PREDICTED | _shifted(noisy_predicted, after, 100)),
        _clip(TRUTH | _shifted(noisy_truth, after, 100)),
        hota=hota,
    )

    combined = evaluate_tracking_clips(pairs, hota=hota)

    assert combined.num_frames == concatenated.num_frames == 14
    assert combined.id_switches == concatenated.id_switches > 1
    assert combined.as_dict() == pytest.approx(concatenated.as_dict(), nan_ok=True)
    pd.testing.assert_frame_equal(combined.per_class, concatenated.per_class)


def test_the_aggregate_of_no_clips_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_tracking_clips([], hota=False)
