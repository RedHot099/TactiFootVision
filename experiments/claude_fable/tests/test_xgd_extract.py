"""The declared configurations, cache rows and the extraction loop with fake analysers (xg-detector 01)."""

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector import extract as ex
from xgpoc.detector.features import COLUMNS, shot_features

from tactifoot_vision.pipeline import NO_TEAM

PEOPLE = [(93.6, 34.2, 0, "player"), (100.0, 34.0, 1, "player"), (104.0, 34.5, 1, "goalkeeper"),
          (50.0, 30.0, NO_TEAM, "referee")]


def test_the_four_configurations_are_the_declared_ones():
    table = {name: (c.imgsz, c.ball_conf, c.ball_fallback) for name, c in ex.CONFIGS.items()}
    assert table == {"C1": (640, 0.30, False), "C2": (640, 0.30, True), "C3": (1920, 0.15, False), "C4": (1920, 0.15, True)}
    assert ex.CONFIGS["C1"].detector == ex.CONFIGS["C2"].detector != ex.CONFIGS["C3"].detector
    assert ex.FALLBACK_OFFSETS == (-400, -800) and set(ex.FRAME_COLUMNS) == {0, -400, -800}
    assert ex.settings(ex.CONFIGS["C4"])["team_reducer"] is None


def _detailed_frame():
    frame = frame_with(PEOPLE, (94.0, 34.0), homography=np.diag([0.1, 0.1, 1.0]), keypoint_conf=np.linspace(0, 1, 32))
    frame.keypoints.xy[:] = np.arange(64, dtype=np.float32).reshape(1, 32, 2)
    return frame


def test_cache_rows_round_trip_to_the_same_features():
    frame = _detailed_frame()
    frame.keypoints.data["box_confidence"] = np.float32([0.42])
    row = ex.frame_row("s1", 0, "/f/s1.jpg", frame)
    assert row["kp_instance_conf"] == pytest.approx(0.42)
    back = ex.frame_from_row(pd.DataFrame([row]).iloc[0].to_dict())
    np.testing.assert_array_equal(back.homography, frame.homography)
    np.testing.assert_array_equal(back.keypoints.xy, frame.keypoints.xy)
    np.testing.assert_allclose(back.keypoints.confidence, frame.keypoints.confidence)
    np.testing.assert_array_equal(back.pitch_xy, frame.pitch_xy)
    np.testing.assert_array_equal(back.team_ids, frame.team_ids)
    assert back.detections.data["class_name"].tolist() == [c for *_, c in PEOPLE]
    np.testing.assert_array_equal(back.ball_xy, frame.ball_xy)
    a, b = shot_features(frame), shot_features(back)
    assert all(a[k] == pytest.approx(b[k], nan_ok=True) for k in COLUMNS)


def test_rows_of_frames_without_a_fit_or_a_ball_or_an_image(tmp_path):
    bare = frame_with([], None)
    bare.homography = None
    row = ex.frame_row("s2", -400, "/f/m400.jpg", bare)
    assert row["homography"] is None and row["kp_xy"] is None and row["ball_conf"] == [] and row["det_class"] == []
    rows = [row, ex.frame_row("s3", 0, "/missing.jpg", None), ex.frame_row("s1", 0, "/f.jpg", _detailed_frame())]
    table = ex.frames_table(rows)
    table.to_parquet(tmp_path / "frames.parquet")                    # the mixed rows are one parquet table
    back = pd.read_parquet(tmp_path / "frames.parquet")
    assert back.shot_id.tolist() == ["s1", "s2", "s3"]
    rebuilt = [ex.frame_from_row(r) for r in back.to_dict("records")]
    assert rebuilt[2] is None and rebuilt[1].homography is None and len(rebuilt[1].ball) == 0
    np.testing.assert_array_equal(rebuilt[0].pitch_xy, _detailed_frame().pitch_xy)


class Analyser:
    """Fake FrameAnalyzer: the 'image' is the frame path; returns the planted FrameResult of that path."""

    def __init__(self, results):
        self.results, self.calls = results, []

    def __call__(self, image):
        self.calls.append(image)
        return self.results[image]


def _shots():
    return pd.DataFrame({"shot_id": ["a", "b", "c"],
                         "frame_path": ["a0", "b0", "c0"], "frame_path_-400": ["a4", "b4", "c4"],
                         "frame_path_-800": ["a8", "b8", None]})


def _results():
    no_ball = frame_with(PEOPLE, None)
    return {"a0": frame_with(PEOPLE, (94.0, 34.0)),                  # own ball
            "b0": no_ball, "b4": frame_with(PEOPLE, None), "b8": frame_with(PEOPLE, (90.0, 30.0)),  # ball at -800
            "c0": no_ball, "c4": frame_with(PEOPLE, (np.nan, np.nan))}  # c: no ball on the pitch anywhere


def test_one_pass_serves_both_configurations_and_reads_earlier_frames_only_when_needed():
    analyser, ball_only = Analyser(_results()), Analyser(_results())
    read = []

    def read_image(path):
        read.append(path)
        return path if path in _results() else None
    out = ex.extract_detector(_shots(), [ex.CONFIGS["C1"], ex.CONFIGS["C2"]], analyser, ball_only, read_image)
    assert analyser.calls == ["a0", "b0", "c0"]                      # t0 once per shot, shared by C1 and C2
    assert ball_only.calls == ["b4", "b8", "c4"]                     # fallback frames once, only without a t0 ball
    assert read == ["a0", "b0", "b4", "b8", "c0", "c4"]              # never a frame after t0, never -800 needlessly
    c1, c2 = (pd.DataFrame(out[n].features).set_index("shot_id") for n in ("C1", "C2"))
    assert c1.ball_ok.tolist() == [1.0, 0.0, 0.0] and c1.ball_offset_ms.tolist()[0] == 0
    assert c2.ball_ok.tolist() == [1.0, 1.0, 0.0] and c2.ball_fallback.tolist() == [0.0, 1.0, 0.0]
    assert c2.loc["b", "ball_offset_ms"] == -800 and c2.loc["b", "shot_x"] == 90.0
    assert np.isnan(c2.loc["c", "ball_offset_ms"]) and c2.loc["c", "shooter_by_goal"] == 1.0
    # The frames each configuration used: t0 for both, the analysed earlier frames for C2 (c's -800 has no path).
    assert [(r["shot_id"], r["offset_ms"]) for r in out["C1"].frames] == [("a", 0), ("b", 0), ("c", 0)]
    c2_frames = [(r["shot_id"], r["offset_ms"], r["readable"]) for r in out["C2"].frames]
    assert c2_frames == [("a", 0, True), ("b", 0, True), ("b", -400, True), ("b", -800, True), ("c", 0, True),
                         ("c", -400, True), ("c", -800, False)]
    # Teams come from t0 only: the fallback analyser has none, and the t0 people are kept.
    assert c2.loc["b", "teams_ok"] == 1.0


def test_an_unreadable_400_frame_keeps_the_800_label():
    results = _results()
    shots = _shots().iloc[[1]]
    out = ex.extract_detector(shots, [ex.CONFIGS["C2"]], Analyser(results), Analyser(results),
                              lambda p: None if p == "b4" else p)
    row = out["C2"].features[0]
    assert (row["ball_source"], row["ball_offset_ms"], row["shot_x"]) == (2.0, -800, 90.0)
    assert [(r["offset_ms"], r["readable"]) for r in out["C2"].frames] == [(0, True), (-400, False), (-800, True)]


def test_an_unreadable_t0_frame_is_a_missing_row_and_mixed_detectors_are_refused():
    shots = _shots().iloc[:1]
    out = ex.extract_detector(shots, [ex.CONFIGS["C3"], ex.CONFIGS["C4"]], Analyser({}), Analyser({}), lambda p: None)
    row = out["C4"].features[0]
    assert row["frame_readable"] == 0.0 and row["shot_ok"] == 0.0 and np.isnan(row["dist"])
    assert out["C4"].frames == [ex.frame_row("a", 0, "a0", None)]
    with pytest.raises(ValueError, match="share the detector settings"):
        ex.extract_detector(shots, [ex.CONFIGS["C1"], ex.CONFIGS["C3"]], Analyser({}), Analyser({}), lambda p: None)


def test_real_analysers_share_one_detector_and_only_the_t0_one_has_teams(monkeypatch):
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
    analyser, ball_only = ex.build_analyzers(1920, 0.15, Embedder(), device="cpu")
    assert analyser.detector is ball_only.detector and analyser.detector.imgsz == 1920
    assert analyser.detector.class_conf == {"ball": 0.15} and analyser.detector.conf == 0.3 and analyser.detector.iou == 0.5
    assert analyser.team_classifier.reducer is None and ball_only.team_classifier is None
    assert analyser.homography.min_confidence == 0.5 and analyser.homography.ransac_threshold == 10.0
    assert analyser.keypoint_model.conf == 0.1 and analyser.keypoint_model.imgsz is None   # letterbox at 640


# ------------------------------------------------------------ study offsets (ticket 02)
def _study_results():
    results = _results()
    results["a4"] = frame_with(PEOPLE, (93.0, 33.0))                 # a's -400 has its own ball
    return results


def test_fallback_chains_never_reach_after_the_frame():
    assert ex.fallback_offsets(0) == (-400, -800) and ex.fallback_offsets(-400) == (-800,)
    assert ex.STUDY_OFFSETS == (0, -400)


def test_the_study_extraction_analyses_t0_and_400_with_teams_and_800_only_for_the_ball():
    analyser, ball_only, read = Analyser(_study_results()), Analyser(_study_results()), []

    def read_image(path):
        read.append(path)
        return path if path in _study_results() else None
    out = ex.extract_study(_shots(), ex.CONFIGS["C4"], analyser, ball_only, read_image)
    assert analyser.calls == ["a0", "a4", "b0", "b4", "c0", "c4"]   # both study offsets, with teams
    assert ball_only.calls == ["b8"]                                 # -800 once, only when no ball is nearer
    assert read == ["a0", "a4", "b0", "b4", "b8", "c0", "c4"]        # never a frame after t0; c's -800 has no path
    f = pd.DataFrame(out.features).set_index(["shot_id", "offset_ms"])
    assert len(f) == 6 and f.index.get_level_values(1).tolist() == [0, -400] * 3
    assert f.loc[("a", 0), "ball_offset_ms"] == 0 and f.loc[("a", -400), "ball_offset_ms"] == -400
    assert f.loc[("b", 0), "ball_offset_ms"] == -800 and f.loc[("b", -400), "ball_offset_ms"] == -800
    assert (f.loc[("b", -400), "ball_source"], f.loc[("b", -400), "shot_x"]) == (1.0, 90.0)
    assert f.loc[("b", -400), "teams_ok"] == 1.0                     # the -400 study frame has teams
    assert np.isnan(f.loc[("c", 0), "ball_offset_ms"]) and np.isnan(f.loc[("c", -400), "ball_offset_ms"])
    assert [(r["shot_id"], r["offset_ms"]) for r in out.frames] == [("a", 0), ("a", -400), ("b", 0), ("b", -400), ("b", -800),
                                                                    ("c", 0), ("c", -400), ("c", -800)]


def test_study_t0_rows_equal_the_audit_extraction_of_the_same_configuration():
    results = _study_results()
    study = ex.extract_study(_shots(), ex.CONFIGS["C4"], Analyser(results), Analyser(results), lambda p: p if p in results else None)
    audit = ex.extract_detector(_shots(), [ex.CONFIGS["C4"]], Analyser(results), Analyser(results),
                                lambda p: p if p in results else None)["C4"]
    t0 = [row for row in study.features if row["offset_ms"] == 0]
    assert len(t0) == len(audit.features)
    for a, b in zip(t0, audit.features, strict=True):
        assert all(a[k] == pytest.approx(b[k], nan_ok=True) for k in b)


def test_without_fallback_the_study_reads_no_800_frame():
    results = _study_results()
    analyser, ball_only = Analyser(results), Analyser(results)
    out = ex.extract_study(_shots(), ex.CONFIGS["C3"], analyser, ball_only, lambda p: p if p in results else None)
    assert ball_only.calls == [] and all(r["offset_ms"] != -800 for r in out.frames)
    f = pd.DataFrame(out.features).set_index(["shot_id", "offset_ms"])
    assert f.loc[("b", 0), "ball_ok"] == 0.0 and f.loc[("a", -400), "ball_offset_ms"] == -400


@pytest.mark.parametrize("name", ["C3", "C4"])
def test_replay_from_the_cached_frames_gives_the_extracted_features(tmp_path, name):
    results = _study_results()
    out = ex.extract_study(_shots(), ex.CONFIGS[name], Analyser(results), Analyser(results), lambda p: p if p in results else None)
    ex.frames_table(out.frames).to_parquet(tmp_path / "frames.parquet")
    replayed = ex.replay_features(pd.read_parquet(tmp_path / "frames.parquet"), ex.CONFIGS[name], ex.STUDY_OFFSETS)
    extracted = pd.DataFrame(out.features)
    pd.testing.assert_frame_equal(replayed.sort_values(["shot_id", "offset_ms"]).reset_index(drop=True),
                                  extracted.sort_values(["shot_id", "offset_ms"]).reset_index(drop=True), check_dtype=False)


def test_replay_refuses_a_missing_fallback_frame(tmp_path):
    results = _study_results()
    out = ex.extract_study(_shots().iloc[[1]], ex.CONFIGS["C4"], Analyser(results), Analyser(results), lambda p: p)
    frames = ex.frames_table([r for r in out.frames if r["offset_ms"] != -800])
    with pytest.raises(ValueError, match="-800 ms frame"):
        ex.replay_features(frames, ex.CONFIGS["C4"])


def test_coverage_by_role_counts_rows_per_role_and_offset():
    features = pd.DataFrame({"shot_id": ["a", "a", "b", "b", "c", "c"], "offset_ms": [0, -400] * 3,
                             **{c: [1.0, 0.0, 0.0, 0.0, 1.0, 1.0] for c in ex.COVERAGE_COLUMNS if c != "def_cone"},
                             "ff_def_cone": [2.0, np.nan, np.nan, np.nan, 0.0, 1.0]})
    games = pd.DataFrame({"shot_id": ["a", "b", "c"], "game": ["g1", "g1", "g2"]})
    plan = pd.DataFrame({"game": ["g1", "g2", "g3"], "role": ["training", "gate", "evaluation"]})
    table = ex.coverage_by_role(features, games, plan).set_index(["role", "offset_ms"])
    assert table.loc[("training", 0), ["shots", "matches", "shot_ok", "def_cone"]].tolist() == [2, 1, 0.5, 0.5]
    assert table.loc[("gate", -400), "shot_ok"] == 1.0 and table.loc[("all", 0), "shots"] == 3
    assert table.loc[("all", -400), "def_cone"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="no role"):
        ex.coverage_by_role(features, games, plan.iloc[:1])
