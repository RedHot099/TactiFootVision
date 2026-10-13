"""Audit 3 of pitch-keypoints 04: the kick rule, the declared shot-location sources and the GSR ball truth."""

import json

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as e3
from xgpoc.detector import kick
from xgpoc.detector.features import COLUMNS
from xgpoc.pitchkp import gsr_ball as gb

from tactifoot_vision.pipeline import NO_TEAM

PEOPLE = [(93.6, 34.2, 0, "player"), (100.0, 34.0, 1, "player"), (104.0, 34.5, 1, "goalkeeper"),
          (50.0, 30.0, NO_TEAM, "referee")]


# ------------------------------------------------------------------ kick rule
def _track(speeds, start=(80.0, 34.0), direction=(1.0, 0.0)):
    """Positions of a ball moving at `speeds[t]` m/s between frame t and t + 1 (25 fps)."""
    steps = np.asarray(speeds, float)[:, None] * 0.04 * np.asarray(direction, float)
    return np.vstack([start, start + np.cumsum(steps, axis=0)])


def test_gaps_up_to_two_frames_are_filled_and_longer_ones_kept():
    xy = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [3, 3], [np.nan] * 2, [np.nan] * 2, [np.nan] * 2, [7, 7.0]])
    filled = kick.fill_gaps(xy)
    np.testing.assert_allclose(filled[:4], [[0, 0], [1, 1], [2, 2], [3, 3]])
    assert np.isnan(filled[4:7]).all()


def test_the_running_median_removes_one_wrong_position_and_keeps_a_monotone_run():
    xy = np.array([[0, 0], [1, 0], [30, 30], [3, 0], [4, 0.0]])
    np.testing.assert_allclose(kick.running_median(xy)[1:4], [[1, 0], [3, 0], [4, 0]])
    run = np.column_stack([np.arange(6.0), np.zeros(6)])
    np.testing.assert_allclose(kick.running_median(run)[1:-1], run[1:-1])


def test_a_struck_ball_gives_the_kick_frame_near_its_onset():
    xy = _track([2.0] * 15 + [25.0] * 15)                         # rolling, then struck at frame 15
    index, location = kick.kick(xy)
    assert index is not None and 13 <= index <= 15
    assert np.hypot(*(location - xy[15])) < 1.0


def test_a_slow_ball_and_one_wrong_detection_give_no_kick():
    assert kick.kick(_track([3.0] * 30))[0] is None
    xy = _track([3.0] * 30)
    xy[12] += (20.0, 0.0)                                          # one detection on another object
    assert kick.kick(xy)[0] is None


def test_a_first_time_shot_changes_the_motion():
    incoming = _track([12.0] * 12, direction=(0.0, 1.0))
    outgoing = _track([25.0] * 15, start=incoming[-1], direction=(1.0, 0.0))[1:]
    index, _ = kick.kick(np.vstack([incoming, outgoing]))
    assert index is not None and 9 <= index <= 12


def test_without_both_neighbours_a_frame_cannot_be_the_kick():
    xy = _track([2.0] * 5 + [25.0] * 25)
    xy[:5] = np.nan                                                # nothing before the strike
    assert kick.kick(xy)[0] is None or kick.kick(xy)[0] >= 5 + kick.SPAN


# ------------------------------------------------------------------ sources
class Analyser:
    def __init__(self, results):
        self.results, self.calls = results, []

    def __call__(self, image):
        self.calls.append(image)
        return self.results[image]


def _shots():
    return pd.DataFrame({"shot_id": ["a", "b", "c", "d"], "frame_path": ["a0", "b0", "c0", "d0"],
                         "frame_path_-400": ["a4", "b4", "c4", "d4"], "frame_path_-800": ["a8", "b8", None, "d8"]})


def _results():
    return {"a0": frame_with(PEOPLE, (94.0, 34.0)), "a4": frame_with(PEOPLE, (92.0, 33.0)),     # balls at t0 and -400
            "b0": frame_with(PEOPLE, None), "b4": frame_with(PEOPLE, None), "b8": frame_with(PEOPLE, (90.0, 30.0)),
            "c0": frame_with(PEOPLE, None), "c4": frame_with(PEOPLE, (np.nan, np.nan)),          # c: no ball anywhere
            "d0": frame_with(PEOPLE, (95.0, 35.0)), "d4": frame_with(PEOPLE, None), "d8": frame_with(PEOPLE, None)}


def _run(configs, offsets=(0,)):
    results = _results()
    analyser, ball_only, read = Analyser(results), Analyser(results), []

    def read_image(path):
        read.append(path)
        return path if path in results else None
    out = e3.extract(_shots(), configs, analyser, ball_only, read_image, offsets=offsets)
    return out, analyser, ball_only, read


def test_the_declared_orders():
    assert {n: c.source for n, c in e3.CONFIGS.items()} == {"E1": (0, -400, -800), "E2": (-400, -800), "E3": (-400, -800, 0)}
    assert e3.CONFIGS["E3"].order(-400) == (-400, -800)           # a row never reads a later frame
    with pytest.raises(ValueError, match="distinct offsets"):
        e3.SourceConfig("X", (0, 0))


def test_e1_is_audit_2s_rule_on_the_same_frames():
    out, *_ = _run([e3.CONFIGS["E1"]])
    results = _results()
    d = ex.extract_detector(_shots(), [ex.CONFIGS["C4"]], Analyser(results), Analyser(results),
                            lambda p: p if p in results else None)["C4"]
    for a, b in zip(out.features["E1"], d.features, strict=True):
        assert all(a[k] == pytest.approx(b[k], nan_ok=True) for k in [*COLUMNS, "ball_offset_ms", "frame_readable"])


def test_one_pass_reads_each_frame_once_and_only_when_an_order_reaches_it():
    out, analyser, ball_only, read = _run(list(e3.CONFIGS.values()))
    assert analyser.calls == ["a0", "b0", "c0", "d0"]
    assert ball_only.calls == ["a4", "b4", "b8", "c4", "d4", "d8"]   # E2 always reads -400, -800 only without a ball
    assert sorted(read) == sorted(analyser.calls + ball_only.calls)
    f = {n: pd.DataFrame(rows).set_index("shot_id") for n, rows in out.features.items()}
    assert f["E1"].ball_offset_ms.tolist()[:2] == [0, -800] and f["E1"].loc["d", "ball_offset_ms"] == 0
    assert f["E2"].loc["a", "ball_offset_ms"] == -400 and f["E2"].loc["a", "shot_x"] == 92.0
    assert np.isnan(f["E2"].loc["d", "ball_offset_ms"]) and f["E2"].loc["d", "shooter_by_goal"] == 1.0   # never t0
    assert f["E3"].loc["d", "ball_offset_ms"] == 0 and f["E3"].loc["d", "ball_source"] == 0          # t0 last
    assert f["E2"].loc["b", "ball_source"] == 2 and f["E2"].loc["a", "ball_fallback"] == 1.0
    assert np.isnan(f["E3"].loc["c", "ball_offset_ms"])
    assert [(r["shot_id"], r["offset_ms"]) for r in out.frames][:3] == [("a", 0), ("a", -400), ("b", 0)]


def test_the_400_row_of_the_study_never_takes_the_t0_ball():
    out, analyser, *_ = _run([e3.CONFIGS["E3"]], offsets=e3.STUDY_OFFSETS)
    assert analyser.calls[:2] == ["a0", "a4"]                      # both study frames with teams
    f = pd.DataFrame(out.features["E3"]).set_index(["shot_id", "offset_ms"])
    assert f.loc[("d", -400), "ball_ok"] == 0.0 and f.loc[("d", 0), "ball_offset_ms"] == 0
    assert f.loc[("a", -400), "ball_offset_ms"] == -400 and f.loc[("a", -400), "ball_source"] == 0


@pytest.mark.parametrize("offsets", [(0,), e3.STUDY_OFFSETS])
def test_replay_from_the_cached_frames_gives_the_extracted_features(tmp_path, offsets):
    out, *_ = _run(list(e3.CONFIGS.values()), offsets=offsets)
    ex.frames_table(out.frames).to_parquet(tmp_path / "frames.parquet")
    frames = pd.read_parquet(tmp_path / "frames.parquet")
    for name, config in e3.CONFIGS.items():
        replayed = e3.replay_features(frames, config, offsets)
        extracted = pd.DataFrame(out.features[name])
        pd.testing.assert_frame_equal(replayed.sort_values(["shot_id", "offset_ms"]).reset_index(drop=True),
                                      extracted.sort_values(["shot_id", "offset_ms"]).reset_index(drop=True),
                                      check_dtype=False)


def test_replay_refuses_a_frame_the_order_needs_but_the_cache_lacks():
    out, *_ = _run([e3.CONFIGS["E2"]])
    frames = ex.frames_table([r for r in out.frames if not (r["shot_id"] == "b" and r["offset_ms"] == -800)])
    with pytest.raises(ValueError, match="-800 ms frame"):
        e3.replay_features(frames, e3.CONFIGS["E2"])


# ------------------------------------------------------------------ GSR ball truth
def _clip(tmp_path, ball_frames, action_ms=10_000, clip_start=0):
    images = [{"image_id": str(f), "file_name": f"{f:06d}.jpg"} for f in range(1, 400)]
    annotations = [{"image_id": str(f), "category_id": 4, "bbox_image": {"x": 1, "y": 2, "w": 3, "h": 4},
                    "bbox_pitch": {"x_bottom_middle": x - 52.5, "y_bottom_middle": y - 34.0}} for f, x, y in ball_frames]
    labels = {"info": {"name": "SNGS-X", "action_class": "Shots on target", "action_position": str(action_ms),
                       "clip_start": str(clip_start)},
              "images": images, "annotations": annotations, "categories": [{"id": 4, "name": "ball"}]}
    (tmp_path / "Labels-GameState.json").write_text(json.dumps(labels))
    return tmp_path


def test_t0_is_the_latest_frame_at_or_before_the_action():
    assert gb.anchor_frame({"action_position": "10000", "clip_start": "0"}) == 251   # 10 s = frame 251 exactly
    assert gb.anchor_frame({"action_position": "10039", "clip_start": "0"}) == 251
    assert gb.anchor_frame({"action_position": "10040", "clip_start": "0"}) == 252


def test_read_ball_converts_to_corner_metres_and_drops_frames_with_two_balls(tmp_path):
    folder = _clip(tmp_path, [(1, 60.0, 30.0), (2, 61.0, 31.0), (2, 10.0, 10.0)])
    _, ball = gb.read_ball(folder)
    assert ball.frame.tolist() == [1] and (ball.x[0], ball.y[0]) == (60.0, 30.0)
    assert (ball.x2[0], ball.y2[0]) == (4, 6)


def test_the_truth_timing_of_a_ball_struck_before_t0(tmp_path):
    t0 = 251
    xy = _track([2.0] * 15 + [25.0] * 30, start=(80.0, 34.0))     # struck at frame t0 - 30 + 15 = t0 - 15 (-600 ms)
    frames = [(t0 - 30 + i, x, y) for i, (x, y) in enumerate(xy)]
    _, ball = gb.read_ball(_clip(tmp_path, frames))
    row = gb.timing_row({"name": "SNGS-X", "action_class": "Shots on target", "action_position": "10000",
                         "clip_start": "0"}, ball)
    assert -680 <= row["kick_ms"] <= -600
    assert row["to_kick_800"] < 1.0 and row["to_kick_400"] == pytest.approx(5 * 0.04 * 25, abs=1.2)
    assert row["move_400_0"] == pytest.approx(10 * 0.04 * 25) and row["to_kick_0"] > row["to_kick_400"]


# ------------------------------------------------------------------ declaration
def test_the_configurations_are_the_committed_declaration():
    from xgpoc.bundle import sha256_file

    assert sha256_file(e3.AUDIT_3_DECLARATION) == e3.AUDIT_3_DECLARATION_SHA256
    declared = e3.declaration()
    assert {n: tuple(c["source_ms"]) for n, c in declared["configs"].items()} == {n: c.source for n, c in e3.CONFIGS.items()}
    common = declared["common"]
    assert common["keypoints"]["weights_sha256"] == e3.KEYPOINT_SHA256
    assert common["keypoints"]["load"] == "tv.load_model('yolo_pose', weights, conf=0.1, imgsz=960, resize='square')"
    assert (e3.COMMON["keypoint_conf"], e3.COMMON["keypoint_imgsz"], e3.COMMON["keypoint_resize"]) == (0.1, 960, "square")
    assert common["homography"]["estimator"] == ("tv.pitch.HomographyEstimator(min_confidence=0.3, ransac_threshold=2.0, "
                                                 "refine_with_lines=True)")
    assert (e3.COMMON["homography_min_confidence"], e3.COMMON["homography_ransac_threshold"]) == (0.3, 2.0)
    detector = common["detector"]
    assert (detector["imgsz"], detector["conf"], detector["iou"], detector["class_conf"]["ball"]) == (
        e3.COMMON["detector_imgsz"], e3.COMMON["detector_conf"], e3.COMMON["detector_iou"], e3.COMMON["ball_conf"])
    assert "audit 2's StatsBomb diagnostics" in declared["evidence"]["provenance"]


def test_real_analysers_load_the_declared_models(monkeypatch):
    import tactifoot_vision as tv
    from tactifoot_vision.models.ultralytics import YOLODetector, YOLOPoseModel

    class Names:
        names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    monkeypatch.setattr(YOLODetector, "_load", lambda self, weights: setattr(self, "_yolo", Names()))
    monkeypatch.setattr(YOLOPoseModel, "_load", lambda self, weights: setattr(self, "_yolo", Names()))

    class Embedder(tv.teams.Embedder):
        name = "fake"

        def embed(self, crops):
            return np.zeros((len(crops), 2), np.float32)
    analyser, ball_only = e3.build_analyzers(Embedder(), device="cpu")
    assert analyser.detector is ball_only.detector and analyser.keypoint_model is ball_only.keypoint_model
    assert analyser.keypoint_model.resize == "square" and analyser.keypoint_model.imgsz == 960
    assert analyser.keypoint_model.weights == e3.KEYPOINT_WEIGHTS and ball_only.team_classifier is None
    h = analyser.homography
    assert (h.min_confidence, h.ransac_threshold, h.refine_with_lines) == (0.3, 2.0, True)
    assert analyser.detector.class_conf == {"ball": 0.15} and analyser.detector.imgsz == 1920
