"""The post-process seam (ADR 0002): contract, registry, record and helpers."""

import copy
import inspect

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.pipeline import (
    POSTPROCESSES,
    AppliedPostProcess,
    FrameResult,
    PipelineResult,
    remap_track_ids,
    reproject,
)

# frame -> pitch: halve the pixels, then shift by (10, 5)
HALF_AND_SHIFT = np.array([[0.5, 0, 10], [0, 0.5, 5], [0, 0, 1]], float)


def _frame(index: int, tracker_ids: tuple[int, ...] | None = (1, 2)) -> FrameResult:
    n = 2 if tracker_ids is None else len(tracker_ids)
    people = sv.Detections(
        xyxy=np.array([[0, 0, 20, 40], [100, 50, 140, 90]], np.float32)[:n],
        confidence=np.full(n, 0.9, np.float32),
        tracker_id=None if tracker_ids is None else np.array(tracker_ids),
        data={
            "class_name": np.array(["player"] * n),
            "pitch_xy": np.full((n, 2), np.nan),
            "team_id": np.zeros(n, int),
        },
    )
    ball = sv.Detections(
        xyxy=np.array([[60, 20, 64, 24]], np.float32),
        confidence=np.array([0.8], np.float32),
        data={"class_name": np.array(["ball"]), "pitch_xy": np.full((1, 2), np.nan)},
    )
    return FrameResult(index=index, timestamp=index / 25, detections=people, ball=ball)


def _result(*frames: FrameResult) -> PipelineResult:
    return PipelineResult(list(frames), 25.0, (200, 100), ["ball", "player"])


@POSTPROCESSES.register("test_offset_ids")
def offset_ids(result: PipelineResult, offset: int = 100, scale: int = 1):
    """Toy post-process: every track id becomes ``id * scale + offset``."""
    mapping = {i: i * scale + offset for i in result.track_ids}
    return remap_track_ids(result, mapping)


def _ids(result: PipelineResult) -> list[list[int]]:
    return [
        f.detections.tracker_id.tolist()
        for f in result
        if f.detections.tracker_id is not None
    ]


# ------------------------------------------------------------------ contract
def test_a_post_process_returns_a_changed_copy_and_leaves_the_input_alone():
    before = _result(_frame(0), _frame(1, (2,)))
    snapshot = copy.deepcopy(before)

    after = POSTPROCESSES.get("test_offset_ids")(before, offset=10)

    assert _ids(after) == [[11, 12], [12]]
    assert _ids(before) == _ids(snapshot) == [[1, 2], [2]]
    assert "original_tracker_id" not in before[0].detections.data
    assert before.postprocesses == []
    assert after is not before


def test_the_registry_hands_out_the_decorated_function_with_its_signature():
    assert POSTPROCESSES.get("test_offset_ids") is offset_ids
    assert "test_offset_ids" in POSTPROCESSES.names()
    assert list(inspect.signature(offset_ids).parameters) == [
        "result",
        "offset",
        "scale",
    ]


def test_the_result_records_each_post_process_with_its_options_and_defaults():
    result = _result(_frame(0))

    once = offset_ids(result, offset=10)
    twice = POSTPROCESSES.get("test_offset_ids")(once, scale=2)

    assert once.postprocesses == [
        AppliedPostProcess("test_offset_ids", {"offset": 10, "scale": 1})
    ]
    assert twice.postprocesses == [
        AppliedPostProcess("test_offset_ids", {"offset": 10, "scale": 1}),
        AppliedPostProcess("test_offset_ids", {"offset": 100, "scale": 2}),
    ]
    assert result.postprocesses == []
    assert _ids(twice) == [[122, 124]]


def test_a_post_process_refuses_an_unknown_option():
    with pytest.raises(TypeError, match="max_gap"):
        offset_ids(_result(_frame(0)), max_gap=3)


@POSTPROCESSES.register("test_returns_frames")
def returns_frames(result: PipelineResult):
    return result.frames


def test_a_post_process_must_return_a_result():
    with pytest.raises(TypeError, match="'test_returns_frames' returned list"):
        returns_frames(_result(_frame(0)))


def test_the_record_survives_save_load_and_export(tmp_path):
    result = offset_ids(_result(_frame(0)), offset=5)
    expected = [AppliedPostProcess("test_offset_ids", {"offset": 5, "scale": 1})]

    loaded = PipelineResult.load(result.save(tmp_path / "saved.pkl"))
    exported = PipelineResult.load(result.export(tmp_path / "run") / "result.pkl")

    assert loaded.postprocesses == exported.postprocesses == expected


def test_a_fresh_result_has_an_empty_record():
    assert _result().postprocesses == []


# ------------------------------------------------------------------- helpers
def test_reproject_moves_pitch_positions_to_the_new_homographies():
    result = _result(_frame(0), _frame(1))
    snapshot = copy.deepcopy(result)

    moved = reproject(result, [HALF_AND_SHIFT, None])

    # people at the bottom centre of their box, the ball at its centre
    np.testing.assert_allclose(moved[0].pitch_xy, [[15, 25], [70, 50]])
    np.testing.assert_allclose(moved[0].ball_xy, [[41, 16]])
    np.testing.assert_array_equal(moved[0].homography, HALF_AND_SHIFT)
    assert np.isnan(moved[1].pitch_xy).all() and np.isnan(moved[1].ball_xy).all()
    assert moved[1].homography is None
    # the rest of the frame and the input are untouched
    assert moved[0].detections.data["team_id"].tolist() == [0, 0]
    assert np.isnan(result[0].pitch_xy).all() and result[0].homography is None
    np.testing.assert_array_equal(result[0].pitch_xy, snapshot[0].pitch_xy)


def test_reproject_needs_one_homography_per_frame():
    with pytest.raises(ValueError, match="2 frames.*1 homograph"):
        reproject(_result(_frame(0), _frame(1)), [HALF_AND_SHIFT])


def test_remap_track_ids_keeps_the_original_ids():
    result = _result(_frame(0, (1, 2)), _frame(1, (2, 3)), _frame(2, None))

    remapped = remap_track_ids(result, {1: 7, 2: 1})
    again = remap_track_ids(remapped, {7: 9})

    assert _ids(remapped) == [[7, 1], [1, 3]]  # 3 is not in the map: kept
    assert again[0].detections.tracker_id.tolist() == [9, 1]
    for remap in (remapped, again):
        originals = [f.detections.data["original_tracker_id"] for f in remap.frames[:2]]
        assert [o.tolist() for o in originals] == [[1, 2], [2, 3]]
    # an untracked frame stays untracked, and the input keeps its ids
    assert remapped[2].detections.tracker_id is None
    assert _ids(result) == [[1, 2], [2, 3]]
    assert "original_tracker_id" not in result[0].detections.data
