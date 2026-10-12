"""The label-free homography check and its development loop (xg-detector 08)."""

import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv
from xgpoc.detector import homography_dev as dev
from xgpoc.detector import lines

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# frame <- pitch: a broadcast-like view of the right half (pitch metres -> 1280 x 720 pixels), far touchline on top
CAMERA = cv2.getPerspectiveTransform(np.float32([[52.5, 0], [105, 0], [105, 68], [52.5, 68]]),
                                     np.float32([[200, 80], [1180, 90], [1270, 700], [10, 690]])).astype(float)
TRUTH = np.linalg.inv(CAMERA) / np.linalg.inv(CAMERA)[2, 2]      # frame -> pitch


def shifted(dx: float, dy: float = 0.0) -> np.ndarray:
    """The true frame -> pitch homography with every position moved (dx, dy) metres."""
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) @ TRUTH


@pytest.fixture(scope="module")
def view() -> lines.LineView:
    return lines.LineView.of(lines.render(TRUTH))


# ------------------------------------------------------------------ the metric
def test_the_true_homography_agrees_perfectly(view):
    score = lines.score(view, TRUTH)
    assert score["points"] > 1000 and score["off_grass"] == 0
    assert score["agreement"] == 1.0 and score["recall"] > 0.99


def test_a_known_shift_lowers_the_agreement(view):
    agreement = [lines.score(view, shifted(dx))["agreement"] for dx in (0.0, 0.5, 1.0, 2.0, 5.0)]
    assert agreement[0] == 1.0 and agreement[1] < 0.6
    assert all(a > b for a, b in zip(agreement[1:], agreement[2:], strict=False))
    # Lines along the shift (touchlines, box sides) still agree: a shift never drops it to 0.
    assert agreement[-1] > 0.2
    assert lines.score(view, shifted(0.0, 1.0))["agreement"] < 0.6


def test_frames_without_a_scorable_homography(view):
    missing = lines.score(view, None)
    assert missing["points"] == 0 and np.isnan([missing["agreement"], missing["recall"]]).all()
    singular = TRUTH.copy()
    singular[2] = singular[0]                                    # maps the frame onto a line
    assert np.isnan(lines.score(view, singular)["agreement"])
    stands = lines.LineView.of(np.full((720, 1280, 3), 90, np.uint8))  # no grass: nothing to score
    score = lines.score(stands, TRUTH)
    assert np.isnan(score["agreement"]) and score["off_grass"] == 1.0


def test_projected_samples_stay_in_front_of_the_camera():
    # The horizon of this view is above the frame: every sample of the far half maps behind the camera or outside.
    xy = lines.projected_samples(TRUTH, (720, 1280))
    assert len(xy) and (xy[:, 0] >= 0).all() and (xy[:, 1] >= 0).all()
    assert (xy[:, 0] <= 1279).all() and (xy[:, 1] <= 719).all()


# ---------------------------------------------------------- checks without lines
def test_players_on_the_pitch_and_their_height():
    feet = cv2.perspectiveTransform(np.float64([[[80, 30]], [[95, 50]]]), CAMERA).reshape(-1, 2)
    # Boxes 1.8 m tall at the local horizontal scale: ratio 1 under the true homography.
    scale = [abs(cv2.perspectiveTransform(np.float64([[[x - 0.5, y]], [[x + 0.5, y]]]), CAMERA)[:, 0, 0] @ [-1, 1])
             for x, y in ((80, 30), (95, 50))]
    boxes = np.array([[fx - 5, fy - 1.8 * s, fx + 5, fy] for (fx, fy), s in zip(feet, scale, strict=True)])
    assert lines.players_on_pitch(TRUTH, boxes) == 1.0
    assert lines.height_ratio(TRUTH, boxes) == pytest.approx(1.0, abs=0.02)
    # A homography twice too large (zoomed out on the pitch) halves the pixels per metre: players look twice as tall.
    zoomed = np.diag([2.0, 2.0, 1.0]) @ TRUTH
    assert lines.height_ratio(zoomed, boxes) == pytest.approx(2.0, rel=0.05)
    assert lines.players_on_pitch(zoomed, boxes) == 0.0           # (160, 60) and (190, 100) are off the pitch
    assert np.isnan(lines.players_on_pitch(None, boxes)) and np.isnan(lines.height_ratio(TRUTH, np.zeros((0, 4))))


# ------------------------------------------------------------- the dev loop
def _plan_and_index():
    plan = pd.DataFrame({"game": [f"g{i}" for i in range(9)],
                         "role": ["training"] * 5 + ["gate"] * 2 + ["evaluation"] * 2,
                         "fold": [0, 1, 2, 3, 4, -1, -1, -1, -1]})
    index = pd.DataFrame({"shot_id": [f"s{i}_{k}" for i in range(9) for k in range(4)],
                          "game": [f"g{i}" for i in range(9) for _ in range(4)],
                          "frame_path": [f"/frames/{i}_{k}.png" for i in range(9) for k in range(4)]})
    return plan, index


def test_only_training_matches_are_sampled_and_split_by_fold():
    plan, index = _plan_and_index()
    frames = dev.sample_training_frames(plan, index, 20, seed=0)
    assert len(frames) == 20 and frames.shot_id.is_unique
    assert set(frames.game) <= {"g0", "g1", "g2", "g3", "g4"}
    assert (frames.subset == np.where(frames.fold <= 2, "dev", "confirm")).all()
    with pytest.raises(ValueError):                               # more shots than the training matches have
        dev.sample_training_frames(plan, index, 21, seed=0)
    again = dev.sample_training_frames(plan.sample(frac=1, random_state=1), index.sample(frac=1, random_state=2), 20, 0)
    assert again.shot_id.tolist() == frames.shot_id.tolist()     # the input order does not change the sample


def test_keypoint_rows_round_trip_and_respect_the_instance_threshold():
    xy = np.arange(64, dtype=np.float32).reshape(1, 32, 2)
    keypoints = sv.KeyPoints(xy=xy, confidence=np.full((1, 32), 0.7, np.float32), data={"box_confidence": np.float32([0.4])})
    back = dev.keypoints_from_row(dev.keypoint_row(keypoints))
    np.testing.assert_array_equal(back.xy, xy)
    np.testing.assert_array_equal(back.confidence, keypoints.confidence)
    weak = sv.KeyPoints(xy=xy, confidence=np.full((1, 32), 0.7, np.float32), data={"box_confidence": np.float32([0.05])})
    assert dev.keypoints_from_row(dev.keypoint_row(weak)) is None
    assert dev.keypoints_from_row(dev.keypoint_row(sv.KeyPoints.empty())) is None


def test_the_summary_counts_good_frames_over_all_frames():
    per_frame = pd.DataFrame({
        "variant": ["a"] * 4, "subset": ["dev", "dev", "confirm", "confirm"],
        "has_homography": [True, True, True, False], "agreement": [0.9, 0.2, np.nan, np.nan],
        "recall": [0.5, 0.1, np.nan, np.nan], "players_on_pitch": [1.0, 0.5, 1.0, np.nan],
        "height_ratio": [1.0, np.e, np.nan, np.nan]})
    summary = dev.summarise(per_frame).set_index("subset")
    assert summary.loc["all", "homography"] == 0.75 and summary.loc["all", "good"] == 0.25
    assert summary.loc["all", "unconfirmed"] == 0.5 and summary.loc["dev", "good"] == 0.5
    assert summary.loc["dev", "height_log_median"] == pytest.approx(0.5)


def test_the_dev_script_reads_everything_inside_the_guard():
    tree = ast.parse((ROOT / "scripts/xgd_homography_dev.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    blocks = [n for n in main.body if isinstance(n, ast.With)
              and any(ast.unparse(i.context_expr) == "no_statsbomb_reads()" for i in n.items)]
    assert len(blocks) == 1
    inside = {id(n) for n in ast.walk(blocks[0])}
    reads = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and ast.unparse(n.func).split(".")[-1] in ("read_csv", "read_parquet", "imread", "keypoint_cache", "c4_frames")]
    assert reads and all(id(n) in inside for n in reads)
    import xgd_homography_dev

    assert {v.keypoint_input for v in xgd_homography_dev.VARIANTS} == {"letterbox", "stretch"}
