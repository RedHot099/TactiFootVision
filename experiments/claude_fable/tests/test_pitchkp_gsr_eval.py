"""The homography evaluation in metres on SoccerNet GSR (pitch-keypoints 02)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from xgpoc.detector import lines
from xgpoc.pitchkp import gsr_eval as ge

from tactifoot_vision.pitch import SoccerPitch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

W, H = 1920, 1080
# frame <- pitch (corner metres -> 1920 x 1080 pixels): a broadcast-like view of the right half, far touchline on top
CAMERA = cv2.getPerspectiveTransform(np.float32([[52.5, 0], [105, 0], [105, 68], [52.5, 68]]),
                                     np.float32([[300, 120], [1770, 135], [1905, 1050], [15, 1035]])).astype(float)
TRUTH = np.linalg.inv(CAMERA) / np.linalg.inv(CAMERA)[2, 2]      # frame -> pitch
TO_720 = np.diag([1280 / W, 720 / H, 1.0])                         # 1080p pixels -> 720p pixels


def to_frame(xy) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(xy, float).reshape(-1, 1, 2), CAMERA).reshape(-1, 2)


# Persons on the right half (corner metres), one beyond the margin, one without a pitch position, and the ball.
PERSONS = np.array([[60.0, 10.0], [70.0, 30.0], [80.0, 50.0], [90.0, 20.0], [95.0, 60.0], [100.0, 34.0],
                    [65.0, 45.0], [75.0, 15.0], [85.0, 40.0]])


def gsr_labels(frames=(1, 2, 6), shift=(0.0, 0.0)) -> dict:
    """A GSR-like Labels-GameState.json of the synthetic view; `shift` moves every annotated pitch position."""
    images, annotations = [], []
    for frame in frames:
        image_id = f"9{frame:06d}"
        images.append({"image_id": image_id, "file_name": f"{frame:06d}.jpg", "width": W, "height": H})
        for i, (x, y) in enumerate(PERSONS):
            fx, fy = to_frame([[x, y]])[0]
            pitch = {"x_bottom_middle": x - 52.5 + shift[0], "y_bottom_middle": y - 34 + shift[1]}
            annotations.append({"image_id": image_id, "supercategory": "object", "category_id": 1 + (i == 0),
                                "bbox_image": {"x": fx - 10, "y": fy - 60, "w": 20, "h": 60}, "bbox_pitch": pitch})
        annotations += [
            {"image_id": image_id, "supercategory": "object", "category_id": 3,      # off the pitch by > 3 m
             "bbox_image": {"x": 900, "y": 40, "w": 20, "h": 60},
             "bbox_pitch": {"x_bottom_middle": 20.0, "y_bottom_middle": -40.0}},
            {"image_id": image_id, "supercategory": "object", "category_id": 1,      # no camera: no pitch position
             "bbox_image": {"x": 950, "y": 500, "w": 20, "h": 60}, "bbox_pitch": None},
            {"image_id": image_id, "supercategory": "object", "category_id": 4,      # the ball is not a person
             "bbox_image": {"x": 960, "y": 540, "w": 5, "h": 5},
             "bbox_pitch": {"x_bottom_middle": 10.0, "y_bottom_middle": 0.0}}]
        line = {}
        for name, ends in {"Side line top": [[55, 0], [100, 0]], "Middle line": [[52.5, 5], [52.5, 60]],
                           "Side line right": [[105, 10], [105, 60]], "Side line bottom": [[60, 68], [100, 68]]}.items():
            points = to_frame(np.linspace(*np.asarray(ends, float), 5)) / [W, H]
            line[name] = [{"x": float(px), "y": float(py)} for px, py in points]
        annotations.append({"image_id": image_id, "supercategory": "pitch", "category_id": 5, "lines": line})
    categories = [{"supercategory": "object", "id": 1, "name": "player"},
                  {"supercategory": "object", "id": 2, "name": "goalkeeper"},
                  {"supercategory": "object", "id": 3, "name": "referee"},
                  {"supercategory": "object", "id": 4, "name": "ball"},
                  {"supercategory": "pitch", "id": 5, "name": "pitch"}]
    return {"info": {"im_dir": "img1"}, "images": images, "annotations": annotations, "categories": categories}


def write_clip(folder: Path, labels: dict, render: bool = False) -> Path:
    (folder / "img1").mkdir(parents=True)
    (folder / "Labels-GameState.json").write_text(json.dumps(labels))
    if render:
        image = lines.render(TRUTH, shape=(H, W), width_px=4)
        for item in labels["images"]:
            cv2.imwrite(str(folder / "img1" / item["file_name"]), image)
    return folder


# ------------------------------------------------------------------------------------------------- the reader
def test_read_clip_converts_to_the_corner_origin_and_keeps_persons_only(tmp_path):
    clip = ge.read_clip(write_clip(tmp_path / "SNGS-900", gsr_labels()))  # default: the scored frames 1, 6, ...
    assert list(clip.frames.frame) == [1, 6] and clip.name == "SNGS-900"
    assert clip.frames.path.iloc[0].endswith("SNGS-900/img1/000001.jpg")
    persons = clip.persons[clip.persons.frame == 1]
    assert len(persons) == len(PERSONS) + 2                       # the ball is dropped
    assert "ball" not in set(persons.category) and {"player", "goalkeeper", "referee"} <= set(persons.category)
    np.testing.assert_allclose(persons[["pitch_x", "pitch_y"]].to_numpy()[:len(PERSONS)], PERSONS)
    assert persons.pitch_x.isna().sum() == 1                      # the person without a camera annotation
    assert ge.on_pitch(persons).sum() == len(PERSONS)             # the off-pitch referee and the NaN person do not count
    assert set(clip.lines[6]) == {"Side line top", "Middle line", "Side line right", "Side line bottom"}


def test_the_scored_frames_are_the_frames_ticket_01_keeps():
    import pk_fetch_soccernet as fetch  # ticket 01 downloads exactly these frames

    assert [f for f in range(1, 751) if ge.keeps_frame(f)] == list(range(1, 751, 5))
    assert all(ge.keeps_frame(f) == fetch.keep_member("gamestate-2024", f"SNGS-1/img1/{f:06d}.jpg")
               for f in range(1, 751))


def test_on_pitch_margin_is_three_metres():
    persons = pd.DataFrame({"pitch_x": [-3.0, -3.01, 108.0, 52.5, 52.5], "pitch_y": [34, 34, 34, -3.0, 71.01]})
    assert list(ge.on_pitch(persons)) == [True, False, True, True, False]


# ------------------------------------------------------------------------------------------------- the error
def test_the_true_homography_scores_zero_and_a_known_shift_scores_that_shift():
    feet = to_frame(PERSONS)
    assert ge.person_errors(TRUTH, feet, PERSONS, (H, W)) == pytest.approx(np.zeros(len(PERSONS)), abs=1e-6)
    for dx, dy in ((1.0, 0.0), (0.0, 2.0), (3.0, 4.0)):
        shifted = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]) @ TRUTH
        errors = ge.person_errors(shifted, feet, PERSONS, (H, W))
        assert errors == pytest.approx(np.full(len(PERSONS), np.hypot(dx, dy)), abs=1e-6)
    assert ge.person_errors(None, feet, PERSONS, (H, W)) is None


def test_feet_beyond_the_horizon_and_absurd_errors_are_capped():
    # A homography whose horizon crosses the frame at y = 540: feet above it have no pitch position.
    tilted = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, -1 / 540, 1.0]])
    errors = ge.person_errors(tilted, np.array([[960.0, 1000.0], [960.0, 1079.0], [960.0, 100.0]]),
                              np.zeros((3, 2)), (H, W))
    assert errors[2] == ge.CAP_M and (errors <= ge.CAP_M).all()
    scale = np.diag([1000.0, 1000.0, 1.0]) @ TRUTH
    assert (ge.person_errors(scale, to_frame(PERSONS), PERSONS, (H, W)) == ge.CAP_M).all()


# ------------------------------------------------------------------------------------------------- the summary
def per_frame_table() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for c in range(6):
        for f in range(20):
            has = (c + f) % 5 != 0
            rows.append({"split": "valid", "method": "M", "clip": f"SNGS-{c:03d}", "frame": f, "scored": f != 19,
                         "has_homography": has, "error_m": rng.gamma(2.0, 1.0 + c) if has else np.nan,
                         "agreement": rng.uniform() if has else np.nan, "refined": float(f % 2) if has else 0.0})
    return pd.DataFrame(rows)


def test_statistics_count_frames_without_a_homography_as_failures():
    frames = pd.DataFrame({"has_homography": [True, True, True, False], "error_m": [0.5, 1.5, 6.0, np.nan],
                           "agreement": [0.9, 0.4, np.nan, np.nan], "refined": [1.0, 0.0, 1.0, 0.0]})
    s = ge.frame_statistics(frames)
    assert s["availability"] == 0.75 and s["median_m"] == 1.5 and s["mean_m"] == pytest.approx(8 / 3)
    assert (s["under_1m"], s["under_2m"], s["under_5m"]) == (0.25, 0.5, 0.5)
    assert s["lines_good"] == 0.25 and s["agreement_median"] == pytest.approx(0.65) and s["refined"] == 0.5
    assert "refined" not in ge.frame_statistics(frames.drop(columns="refined"))


def test_summary_is_deterministic_under_the_seed_and_resamples_clips():
    table = per_frame_table()
    first, second = ge.summarise(table, draws=200), ge.summarise(table, draws=200)
    pd.testing.assert_frame_equal(first, second)
    other = ge.summarise(table, draws=200, seed=1)
    assert not np.allclose(first.ci_low.dropna(), other.ci_low.dropna())
    values = first.set_index("metric")
    assert values.loc["frames", "value"] == 6 * 19 and values.loc["clips", "value"] == 6   # unscored frames dropped
    point = values.drop(["frames", "clips"])
    assert ((point.ci_low <= point.value + 1e-12) & (point.value <= point.ci_high + 1e-12)).all()
    # A draw resamples whole clips: with one clip there is nothing to resample and the interval collapses.
    one = ge.summarise(table[table["clip"] == "SNGS-000"], draws=50).set_index("metric")
    assert one.loc["median_m", "ci_low"] == one.loc["median_m", "ci_high"] == one.loc["median_m", "value"]


# ------------------------------------------------------------------------------------------------- conventions
def test_convention_check_finds_the_lines_where_soccerpitch_has_them(tmp_path):
    clip = ge.read_clip(write_clip(tmp_path / "SNGS-900", gsr_labels()), lambda f: f == 1)
    check = ge.convention_check(clip)
    assert len(check) == 1 and check.feet_loo_median_m.iloc[0] < 1e-3
    for name in ("Side line top", "Middle line", "Side line right", "Side line bottom"):
        assert check[name].iloc[0] < 1e-3


def test_convention_check_flags_a_mirrored_convention(tmp_path):
    labels = gsr_labels(frames=(1,))
    for item in labels["annotations"]:                            # y growing away from the camera instead
        if item.get("bbox_pitch"):
            item["bbox_pitch"]["y_bottom_middle"] *= -1
    check = ge.convention_check(ge.read_clip(write_clip(tmp_path / "SNGS-901", labels)))
    assert check["Side line top"].iloc[0] > 60 and check["Side line bottom"].iloc[0] > 60


# ------------------------------------------------------------------------------------------------- the run script
def test_methods_are_d1_to_d4_of_the_configs_file():
    import pitchkp_gsr_eval as run

    methods = run.methods_from_configs(run.CONFIGS, ["D1", "D2", "D3", "D4"])
    assert {n: (m.min_confidence, m.refine_with_lines, m.keypoint_input, m.ransac_threshold)
            for n, m in methods.items()} == {"D1": (0.5, False, "stretch", 10.0), "D2": (0.3, False, "stretch", 10.0),
                                             "D3": (0.5, True, "stretch", 10.0), "D4": (0.3, True, "stretch", 10.0)}


def test_the_refinement_runs_with_d3s_declared_settings():
    """The package's refinement defaults are what the configs file declares for D1-D4 (Variant passes none)."""
    import inspect

    import pitchkp_gsr_eval as run

    from tactifoot_vision.pitch.lines import refine_homography

    declared = json.loads(run.CONFIGS.read_text())["common"]["homography"]["refinement_defaults"]
    defaults = {k: p.default for k, p in inspect.signature(refine_homography).parameters.items()
                if p.default is not inspect.Parameter.empty and k in declared}
    assert defaults == declared


def oracle_keypoints(frames) -> pd.DataFrame:
    """Keypoint rows at 720p placed exactly on the landmarks the true view sees."""
    vertices = SoccerPitch().vertices.astype(float)
    xy = cv2.perspectiveTransform(vertices.reshape(-1, 1, 2), TO_720 @ CAMERA).reshape(-1, 2)
    seen = (xy[:, 0] >= 0) & (xy[:, 0] < 1280) & (xy[:, 1] >= 0) & (xy[:, 1] < 720)
    row = {"kp_xy": [float(v) for v in xy.ravel()], "kp_conf": [float(s) for s in seen], "kp_instance_conf": 0.9}
    return pd.DataFrame([{"frame": f} | row for f in frames]).set_index("frame")


@pytest.mark.parametrize("shift", [(0.0, 0.0), (1.5, -2.0)])
def test_end_to_end_oracle_keypoints_score_zero_or_the_known_shift(tmp_path, shift):
    import pitchkp_gsr_eval as run

    folder = write_clip(tmp_path / "SNGS-902", gsr_labels(frames=(1, 6), shift=shift), render=True)
    methods = run.methods_from_configs(run.CONFIGS, ["D1", "D3"])
    rows, convention = run.evaluate_clip(("valid", folder, oracle_keypoints([1, 6]), methods))
    table = pd.DataFrame(rows)
    assert len(table) == 4 and table.scored.all() and table.has_homography.all()
    assert (table.persons == len(PERSONS)).all()
    assert table.error_m.to_numpy() == pytest.approx(np.full(4, np.hypot(*shift)), abs=0.05)
    assert (table.agreement > 0.95).all()                          # the rendered lines agree with the true view
    assert table.loc[table.method == "D3", "refined"].notna().all()
    assert len(convention) == 2
