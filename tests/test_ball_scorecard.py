"""ball-detection 02: the scorecard's scoring, paired bootstrap, dump reading and splits (scripts/ball_scorecard.py),
on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import supervision as sv
from supervision.metrics import MeanAveragePrecision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_scorecard as bs  # noqa: E402
import detector_bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

CLASSES = ["ball", "goalkeeper", "player", "referee"]
CLIPS = {"SNGS-001": "2", "SNGS-002": "3", "SNGS-003": "5"}


def synthetic(seed: int = 0, frames: int = 4):
    """A GSR-like dataset (3 clips of games 2 / 3 / 5) and noisy predictions with distinct confidences."""
    rng = np.random.default_rng(seed)
    samples, found = [], {}
    for clip, game in CLIPS.items():
        for f in range(frames):
            frame = 1 + 5 * f
            n = int(rng.integers(2, 6))
            xy = rng.uniform(0, 400, (n, 2))
            boxes = np.hstack([xy, xy + rng.uniform(8, 40, (n, 2))])
            class_ids = rng.integers(0, 4, n)
            annotations = tv.data.Annotations(boxes, class_ids)
            meta = {"sequence": clip, "game": game, "frame": str(frame)}
            samples.append(
                tv.data.Sample(Path(f"{clip}/{frame}.jpg"), 640, 480, annotations, meta)
            )
            keep = rng.uniform(size=n) < 0.8
            noisy = boxes[keep] + rng.normal(0, 3, (keep.sum(), 4))
            extra = rng.uniform(0, 400, (2, 2))
            noisy = np.vstack([noisy, np.hstack([extra, extra + 12])])
            noisy[:, 2:] = np.maximum(noisy[:, 2:], noisy[:, :2] + 2)
            ids = np.concatenate([class_ids[keep], rng.integers(0, 4, 2)])
            found[f"{clip}/{frame:06d}"] = sv.Detections(
                xyxy=noisy.astype(np.float32),
                confidence=rng.uniform(0.001, 1, len(ids)).astype(np.float32),
                class_id=ids,
                data={"class_name": np.array(CLASSES)[ids]},
            )
    dataset = tv.data.Dataset("detect", CLASSES, {"valid": samples})
    return dataset, found


def supervision_ap(dataset, found, weights: dict[str, int] | None = None):
    """supervision's AP per class on the images, each repeated ``weights[clip]`` times."""
    metric = MeanAveragePrecision()
    for sample in dataset["valid"]:
        clip, frame = sample.meta["sequence"], int(sample.meta["frame"])
        for _ in range((weights or {}).get(clip, 1)):
            metric.update(
                found[f"{clip}/{frame:06d}"], sample.annotations.to_detections()
            )
    result = metric.compute()
    return dict(
        zip(result.matched_classes.tolist(), result.ap_per_class, strict=True)
    ), result


def test_score_dump_is_supervisions_map_and_any_subset_scores_the_same_way():
    dataset, found = synthetic()
    ap, result = supervision_ap(dataset, found)
    scored = detector_bench.score_dump(dataset, "valid", found, "gsr")
    assert scored["map50_95"] == pytest.approx(result.map50_95, abs=1e-12)
    assert scored["map50"] == pytest.approx(result.map50, abs=1e-12)
    for class_id, name in enumerate(CLASSES):
        assert scored["per_class"][name]["map50"] == pytest.approx(
            ap[class_id][0], abs=1e-12
        )
    scores = detector_bench.image_scores(dataset, "valid", found, "gsr")
    keep = scores.keys["game"].isin(["2", "3"]).to_numpy()
    selection = dataset.with_split(
        "valid", [s for s in dataset["valid"] if s.meta["game"] != "5"]
    )
    assert detector_bench.summarise_scores(scores, keep) == detector_bench.score_dump(
        selection, "valid", found, "gsr"
    )


def test_resampled_ap_is_supervision_on_the_drawn_images():
    dataset, found = synthetic(seed=1)
    scores = detector_bench.image_scores(dataset, "valid", found, "gsr")
    keep = np.ones(len(scores.keys), bool)
    draws = bs.Draws(scores.keys["clip"])
    aps = bs.resampled_ap(scores, keep, draws)
    ap, _ = supervision_ap(dataset, found)
    for class_id, (point, per_draw, has) in aps.items():
        assert point == pytest.approx(ap[class_id], abs=1e-12)
        assert per_draw.shape == (bs.BOOTSTRAP["draws"], 10) and has.all()
    for d in (
        0,
        1,
        7,
    ):  # a draw repeats clips: supervision on the repeated images gives the same AP
        weights = dict(zip(draws.clips, draws.weights[d].tolist(), strict=True))
        expected, _ = supervision_ap(dataset, found, weights)
        for class_id, (_, per_draw, _) in aps.items():
            assert per_draw[d] == pytest.approx(
                expected.get(class_id, np.zeros(10)), abs=1e-12
            )


def test_draws_are_seeded_clip_counts():
    draws = bs.Draws(["b", "a", "c", "a"])
    assert draws.clips.tolist() == ["a", "b", "c"]
    assert draws.weights.shape == (2000, 3) and (draws.weights.sum(axis=1) == 3).all()
    first = np.random.default_rng(0).integers(0, 3, (2000, 3))[0]
    assert draws.weights[0].tolist() == np.bincount(first, minlength=3).tolist()
    assert draws.per_clip(["a", "c", "c"], [1, 2, 3]).tolist() == [1, 0, 5]
    with pytest.raises(ValueError, match="outside"):
        draws.codes(["d"])


def test_paired_differences_share_the_draws():
    draws = bs.Draws(["a", "b", "c", "d"])
    ones = np.ones(4)
    better = bs.ratio_draws(draws, np.array([1.0, 1, 1, 0]), ones)
    worse = bs.ratio_draws(draws, np.array([1.0, 0, 1, 0]), ones)
    assert better[0] == 0.75 and worse[0] == 0.5
    same = bs.paired({"x": better}, {"x": better})["x"]
    assert same["difference"] == 0 and same["ci"] == [0.0, 0.0]
    gain = bs.paired({"x": better}, {"x": worse})["x"]
    assert gain["difference"] == 0.25
    # clip b is the only frame-pair that differs: every draw's difference is 0 or positive
    assert gain["ci"][0] >= 0 and gain["ci"][1] > 0
    assert bs.paired({"x": better, "y": better}, {"x": worse}).keys() == {"x"}


def test_placement_numbers_and_denominator():
    rows = pd.DataFrame(
        {
            "clip": ["a", "a", "a", "b", "b"],
            "near_goal": [True, False, False, True, False],
            "homography": [True, True, False, True, True],
            "error_m": [1.0, 7.0, np.nan, np.nan, 5.0],
        }
    )
    found = bs.placement_draws(rows, bs.Draws(rows["clip"]))
    assert found["well_placed"][0] == pytest.approx(
        2 / 5
    )  # every frame counts, no homography is a miss
    assert found["well_placed_with_homography"][0] == pytest.approx(2 / 4)
    assert found["ceiling"][0] == pytest.approx(4 / 5)
    assert found["placed"][0] == pytest.approx(3 / 5)
    assert found["far_of_placed"][0] == pytest.approx(
        1 / 3
    )  # 5.0 m is well placed, 7 m is far
    assert found["median_m"][0] == 5.0
    assert found["near_goal_well_placed"][0] == pytest.approx(1 / 2)
    assert bs.placement_counts(rows) == {
        "frames": 5,
        "without_homography": 1,
        "near_goal_frames": 2,
        "clips": 2,
    }


def test_splits_by_game_and_placement_reads_only_the_splits_clips(monkeypatch):
    assert bs.SPLITS["selection"] == ("2", "3") and bs.SPLITS["confirm"] == ("5",)
    assert set(bs.SPLITS["full"]) == set(bs.SPLITS["selection"]) | set(
        bs.SPLITS["confirm"]
    )
    dataset, _ = synthetic()
    assert {s.meta["game"] for s in bs.subset(dataset, ("5",))["valid"]} == {"5"}

    asked = []

    def placement_rows(top_ball, homographies, split):
        rows = []
        for clip, game in CLIPS.items():
            centre = top_ball(clip, 1, Path("unused"))
            asked.append((clip, centre is not None))
            rows.append({"clip": clip, "game": game, "frame": 1, "error_m": 0.0})
        return pd.DataFrame(rows)

    monkeypatch.setattr(detector_bench, "placement_rows", placement_rows)
    found = {
        f"{clip}/000001": sv.Detections(
            xyxy=np.array([[30.0, 30.0, 36.0, 36.0]], np.float32),
            confidence=np.array([0.9], np.float32),
            class_id=np.array([0]),
            data={"class_name": np.array(["ball"])},
        )
        for clip in CLIPS
    }
    images = pd.DataFrame(
        {
            "set": "gsr",
            "clip": list(CLIPS),
            "game": list(CLIPS.values()),
            "width": 1920,
            "height": 1080,
        }
    )
    rows = bs.placement_table(found, images, {}, bs.SPLITS["selection"])
    assert rows["clip"].tolist() == ["SNGS-001", "SNGS-002"]
    assert asked == [("SNGS-001", True), ("SNGS-002", True), ("SNGS-003", False)]
    top = bs.split_top_ball(found, (1.5, 1.5), {"SNGS-001"})
    assert top("SNGS-001", 1, Path("unused")).tolist() == pytest.approx([22.0, 22.0])


def record(**changes) -> dict:
    base = {
        "conf": 0.001,
        "max_det": 300,
        "classes": dict(enumerate(CLASSES)),
        "rows": {k: detector_bench.ROWS[k] for k in ("1280", "e3")},
        "sets": {"gsr": {"images": 2}},
        "seconds": {"gsr/1280": 2.0},
    }
    return base | changes


def test_check_dump_needs_the_row_settings_the_card_depends_on():
    bs.check_dump(record())
    bs.check_dump(
        record(rows={"1920": {"imgsz": 1920, "iou": 0.7, "input_size": None}})
    )
    with pytest.raises(SystemExit, match="AP needs the tail"):
        bs.check_dump(record(conf=0.01))
    with pytest.raises(SystemExit, match="max_det"):
        bs.check_dump({k: v for k, v in record().items() if k != "max_det"})
    wrong = {"1280": {"imgsz": 1280, "iou": 0.5, "input_size": None}}
    with pytest.raises(SystemExit, match="not the bench's"):
        bs.check_dump(record(rows=wrong))


def test_class_map_and_cost_from_the_dump_record():
    assert detector_bench.class_map(["ball", "person"], CLASSES) == {
        "ball": "ball",
        "person": None,
    }
    assert bs.dump_class_map(record(), "gsr", CLASSES) == {c: c for c in CLASSES}
    recorded = record(sets={"gsr": {"class_map": {"ball": "ball"}}})
    assert bs.dump_class_map(recorded, "gsr", CLASSES) == {"ball": "ball"}
    images = pd.DataFrame({"set": ["gsr", "gsr"]})
    assert bs.cost(record(), images)["ms_per_frame"] == {"gsr/1280": 1000.0}
    assert "wall" in bs.cost(record(), images)["kind"]
    timed = record(
        ms_per_frame={"gsr/1280": 21.5}, gpu="NVIDIA GeForce RTX 4070 Ti SUPER"
    )
    assert bs.cost(timed, images)["ms_per_frame"] == {"gsr/1280": 21.5}


def test_hits_and_false_balls():
    truth = pd.DataFrame(
        {
            "clip": ["a", "b"],
            "image": ["a/000001", "b/000001"],
            "x1": [100.0, 100.0],
            "y1": [100.0, 100.0],
            "x2": [110.0, 110.0],
            "y2": [110.0, 110.0],
        }
    )
    found = {
        "a/000001": sv.Detections(
            xyxy=np.array(
                [[104.0, 100.0, 114.0, 110.0], [300, 300, 310, 310]], np.float32
            ),
            confidence=np.array([0.5, 0.2], np.float32),
            class_id=np.array([0, 0]),
            data={"class_name": np.array(["ball", "ball"])},
        )
    }
    import ball_errors as be

    table = be.candidate_columns(truth, found)
    draws = bs.Draws(truth["clip"])
    hits = bs.hit_draws(table, draws)
    assert hits["recall_iou@0.15"][0] == 0.0  # 4 px off a 10 px ball: IoU 0.43
    assert hits["recall_centre@0.15"][0] == 0.5
    keys = pd.DataFrame({"image": ["a/000001", "b/000001"], "clip": ["a", "b"]})
    falses = bs.false_ball_draws(found, keys, truth, draws)
    assert falses["false_balls_per_frame"][0] == 0.5
    assert falses["false_share_of_kept_balls"][0] == 0.5


def test_model_top_ball_scales_a_native_frame_to_the_homography_pixels(tmp_path):
    import cv2

    path = tmp_path / "000001.jpg"
    cv2.imwrite(str(path), np.zeros((1080, 1920, 3), np.uint8))

    class Ball:
        def predict(self, frame):
            assert frame.shape == (1080, 1920, 3)
            return sv.Detections(
                xyxy=np.array([[297.0, 147.0, 303.0, 153.0]]),
                confidence=np.array([0.9]),
                data={"class_name": np.array(["ball"])},
            )

    top = detector_bench.model_top_ball(Ball(), None)
    assert top("c", 1, path).tolist() == pytest.approx([200.0, 100.0])


def test_agree_compares_every_number_but_cost_and_paths():
    a = {
        "cost": {"ms": 1.0},
        "cards": {
            "x": {"v": 0.5, "bins": [{"feature": "size", "level": "<8", "r": 0.1}]}
        },
    }
    b = {
        "cost": {"ms": 9.0},
        "cards": {
            "x": {
                "v": 0.5004,
                "bins": [{"feature": "size", "level": "<8", "r": np.nan}],
            }
        },
    }
    report = bs.agree(a, b)
    assert report["numbers"] == 2 and report["over_1e-3"] == ["/cards/x/bins/size=<8/r"]
    b["cards"]["x"]["bins"][0]["r"] = 0.1
    report = bs.agree(a, b)
    assert report["over_1e-3"] == [] and report["max_abs_difference"] == pytest.approx(
        4e-4
    )


def test_pair_row_same_name_else_a_single_rows_reference_else_1280():
    from types import SimpleNamespace

    two = SimpleNamespace(rows=["1280", "e3"])
    assert bs.pair_row("e3", two) == "e3"
    assert bs.pair_row("tiles2x2+pf45", two) == "1280"
    single = SimpleNamespace(rows=["1280+pf45"])
    assert bs.pair_row("tiles2x2+pf45", single) == "1280+pf45"
    assert bs.pair_row("1280+pf45", single) == "1280+pf45"
