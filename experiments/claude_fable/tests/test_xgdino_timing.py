"""Inference-time benchmark helpers (xg-dino 03): the stage clock, the summaries, the stage timers on a frame analyser,
the deployment logistic, and B's per-shot feature row against the frozen feet_t0 table. CPU only."""

import numpy as np
import pandas as pd
import pytest
import supervision as sv
from xgpoc.detector import extract as ex
from xgpoc.detector import location as loc
from xgpoc.dino import ablation as ab
from xgpoc.dino import timing as tm
from xgpoc.stack import fit as sf
from xgpoc.stack_location import inputs as sl

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.pipeline import FrameAnalyzer
from tactifoot_vision.pitch.pitch import SoccerPitch


class FakeClock:
    """Time advances by 1 s on every read; counts the synchronisations."""

    def __init__(self):
        self.t, self.syncs = 0.0, 0

    def now(self) -> float:
        self.t += 1.0
        return self.t

    def sync(self) -> None:
        self.syncs += 1


def test_stage_clock_accumulates_nests_and_clears():
    fake = FakeClock()
    clock = tm.StageClock(fake.sync, fake.now)
    with clock.stage("outer"):
        with clock.stage("inner"):
            pass
    with clock.stage("inner"):
        pass
    assert clock.take() == {"outer": 3000.0, "inner": 2000.0}
    assert fake.syncs == 6
    assert clock.take() == {}


def test_stage_clock_records_a_failing_stage():
    fake = FakeClock()
    clock = tm.StageClock(fake.sync, fake.now)
    with pytest.raises(RuntimeError), clock.stage("x"):
        raise RuntimeError
    assert clock.take() == {"x": 1000.0}


def test_summary_quartiles():
    s = tm.summary([1, 2, 3, 4, 5])
    assert s == {"median": 3.0, "q25": 2.0, "q75": 4.0, "iqr": 2.0, "n": 5}


class FakeDetector:
    task = Task.DETECT

    def predict(self, image):
        d = sv.Detections(xyxy=np.array([[10, 10, 20, 40], [50, 50, 52, 52]], float), confidence=np.array([0.9, 0.8]),
                          class_id=np.array([0, 1]))
        d.data["class_name"] = np.array(["player", "ball"])
        return d


class FakeKeypoints:
    task = Task.POSE

    def predict(self, image):
        return sv.KeyPoints.empty()


class FakeHomography:
    pitch = SoccerPitch()

    def __init__(self):
        self.calls = 0

    def fit(self, keypoints, frame, boxes):
        self.calls += 1
        return None


def test_instrument_times_each_model_and_keeps_the_result():
    homography = FakeHomography()
    analyzer = FrameAnalyzer(FakeDetector(), FakeKeypoints(), None, homography=homography)
    clock = tm.StageClock()
    timed = tm.instrument(analyzer, clock)
    image = np.zeros((64, 64, 3), np.uint8)
    with clock.stage("analyse"):
        result = timed(image)
    stages = clock.take()
    assert set(stages) == {"analyse", "detect", "keypoints", "homography"}
    assert stages["analyse"] >= stages["detect"] + stages["keypoints"] + stages["homography"]
    assert tm.analyse_rest(stages) >= 0
    reference = analyzer(image)
    assert np.array_equal(result.detections.xyxy, reference.detections.xyxy)
    assert np.array_equal(result.ball.xyxy, reference.ball.xyxy)
    assert homography.calls == 2
    assert analyzer.detector.__class__ is FakeDetector            # the original analyser is untouched


def test_logistic_width_and_prediction():
    table = pd.DataFrame({"a": [0.0, 1.0, 2.0, np.nan], "b": [1.0, 0.0, 1.0, 0.0]})
    arm = sf.Arm("T", "descriptive", ("a", "b"), (), 3)
    pre = sf.Preprocessor.fit(arm, table)
    model = tm.surrogate(pre, seed=0)
    assert model.width == 2 and len(model.beta) == 3
    assert np.allclose(model(table), sf.predict(model.beta, pre(table)))
    with pytest.raises(ValueError):
        tm.Logistic(pre, np.zeros(2))


@pytest.mark.skipif(not (ab.FROZEN.feet / "frames.parquet").exists(), reason="the frozen feet_t0 cache is not here")
def test_geometry_row_replays_the_frozen_table():
    """B's per-shot computation on cached frames equals `stack_location.e3_table` under feet_t0 on the C1 allowlist."""
    feet = ab.FROZEN.feet
    ids = sorted(pd.read_parquet(feet / "features.parquet", columns=["shot_id"]).shot_id.unique())[:: 997][:12]
    frames = pd.read_parquet(feet / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(feet / "features.parquet", filters=[("shot_id", "in", ids)])
    frozen, _ = sl.e3_table(frames, features, loc.FEET_T0)
    rows = {}
    for shot_id, group in frames.groupby("shot_id", sort=True):
        cached = {int(r["offset_ms"]): ex.frame_from_row(r) for r in group.to_dict("records")}
        rows[shot_id] = tm.geometry_row(cached)
    live = pd.DataFrame.from_dict(rows, orient="index")
    columns = list(ab.GEOMETRY)
    pd.testing.assert_frame_equal(live.loc[frozen.index, columns], frozen[columns], check_names=False)
