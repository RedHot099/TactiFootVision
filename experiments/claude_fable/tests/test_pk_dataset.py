"""SoccerNet line annotations -> 32-vertex pitch keypoint labels (pitch-keypoints 01)."""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from xgpoc.pitchkp import dataset as D
from xgpoc.pitchkp import fit as F
from xgpoc.pitchkp import sources as S

from tactifoot_vision.pitch import SoccerPitch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def camera(position, target, focal: float) -> np.ndarray:
    """Pitch -> 720p frame homography of a pinhole camera at `position` (x, y, height) looking at `target`."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    R = np.stack([right, down, forward])
    K = np.array([[focal, 0, F.FRAME_W / 2], [0, focal, F.FRAME_H / 2], [0, 0, 1.0]])
    return (K @ np.hstack([R, -R @ position[:, None]]))[:, [0, 1, 3]]


def annotate(to_frame: np.ndarray, step: float = 1.0, noise: float = 0.0, seed: int = 0) -> dict:
    """SoccerNet-style annotations (fractions of the image) of every marking the camera sees."""
    rng = np.random.default_rng(seed)
    markings = {}
    for name, (a, b) in F.STRAIGHT.items():
        A, B = F.VERTICES[a], F.VERTICES[b]
        markings[name] = A + np.linspace(0, 1, int(np.linalg.norm(B - A) / step) + 2)[:, None] * (B - A)
    for name, (cx, cy) in F.CIRCLES.items():
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        q = np.column_stack([cx + F.RADIUS * np.cos(t), cy + F.RADIUS * np.sin(t)])
        if name == "Circle left":
            q = q[q[:, 0] > F.VERTICES[9, 0]]
        elif name == "Circle right":
            q = q[q[:, 0] < F.VERTICES[17, 0]]
        markings[name] = q
    out = {}
    for name, q in markings.items():
        p = np.column_stack([q, np.ones(len(q))]) @ to_frame.T
        p = p[p[:, 2] > 0]
        xy = p[:, :2] / p[:, 2:] + rng.normal(0, noise, (len(p), 2))
        xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < F.FRAME_W) & (xy[:, 1] >= 0) & (xy[:, 1] < F.FRAME_H)]
        if len(xy) >= 2:
            out[name] = [{"x": x / F.FRAME_W, "y": y / F.FRAME_H} for x, y in xy]
    return out


VIEWS = {
    "wide": camera((52.5, 103, 22), (52.5, 30, 0), 900),
    "right box": camera((52.5, 103, 22), (90, 34, 0), 1400),
    "left box": camera((52.5, 103, 22), (20, 30, 0), 1200),
    "centre circle, two lines": camera((52.5, 103, 22), (52.5, 34, 0), 2500),
}


def truth_labels(to_frame: np.ndarray) -> F.Labels:
    H = np.linalg.inv(to_frame)
    return F.labels(H / H[2, 2])


# --------------------------------------------------------------------------- (a) known homography -> exact vertices
@pytest.mark.parametrize("view", sorted(VIEWS))
def test_lines_rendered_through_a_known_homography_give_its_vertices(view):
    lines = annotate(VIEWS[view])
    if view.startswith("centre"):
        assert set(lines) == {"Side line top", "Middle line", "Circle central"}   # the hard case: 2 lines + circle
    fit = F.fit_lines(F.annotation_points(lines))
    want, got = truth_labels(VIEWS[view]), F.labels(fit.H)
    assert fit.rms < 1e-6
    assert (got.visible == want.visible).all() and want.visible.sum() >= 5
    np.testing.assert_allclose(got.xy, want.xy, atol=1e-6)
    assert np.allclose(got.box, want.box)


def test_noisy_annotations_give_vertices_within_their_uncertainty():
    to_frame = VIEWS["right box"]
    fit = F.fit_lines(F.annotation_points(annotate(to_frame, noise=1.0)))
    want = truth_labels(to_frame)
    err = np.linalg.norm(F.labels(fit.H).xy - want.xy, axis=1)[want.visible]
    sd = F.vertex_sd(fit)[want.visible]
    assert 0.8 < fit.rms < 1.2
    assert err.max() < 2.0 and (err < 3 * sd + 0.1).all()


def test_an_underdetermined_fit_is_rejected():
    lines = {k: v for k, v in annotate(VIEWS["centre circle, two lines"]).items() if k != "Side line top"}
    record = D.process(S.Frame("calib", "train", "g", Path("x.jpg"), lines))
    assert record["reject"] in ("no fit", "vertex sd") and record["sd_max"] == np.inf


def test_mirrored_line_names_find_no_camera_view():
    """Left and right names swapped without flipping the image: only a mirror image fits, which is never kept."""
    lines = annotate(VIEWS["right box"])
    swapped = {F.FLIP_NAMES[name]: points for name, points in lines.items()}
    record = D.process(S.Frame("calib", "train", "g", Path("x.jpg"), swapped))
    assert record["reject"] != ""
    assert not record["fitted"] or record["rms"] > D.RMS_MAX_PX


# --------------------------------------------------------------------------- (c) orientation and flip_idx
def test_flip_idx_matches_the_data_yaml_and_the_pitch():
    text = Path("/home/kuba/projects/tactifoot_vision/data/keypoints/data.yaml")
    if text.exists():
        assert f"flip_idx: {list(D.FLIP_IDX)}" in text.read_text()
    v = SoccerPitch().vertices
    mirrored = np.column_stack([SoccerPitch().length - v[:, 0], v[:, 1]])
    np.testing.assert_allclose(mirrored[list(D.FLIP_IDX)], v, atol=1e-4)


@pytest.mark.parametrize("view", ["wide", "right box", "left box"])
def test_a_flipped_image_gets_the_flip_idx_labels(view):
    lines = annotate(VIEWS[view])
    lab = F.labels(F.fit_lines(F.annotation_points(lines)).H)
    flipped = F.labels(F.fit_lines(F.annotation_points(D.flipped_lines(lines))).H)
    want_xy, want_visible = D.flip_labels(lab.xy, lab.visible)
    assert (flipped.visible == want_visible).all()
    np.testing.assert_allclose(flipped.xy, want_xy, atol=1e-5)


def test_far_touchline_is_on_top_and_left_goal_on_the_left():
    lab = truth_labels(VIEWS["wide"])
    xy = lab.xy
    assert lab.visible[[0, 13, 16, 24]].all()
    assert xy[13, 1] < xy[14, 1] < xy[15, 1] < xy[16, 1]          # far (y = 0) above near (y = 68)
    assert xy[0, 0] < xy[13, 0] < xy[24, 0]                       # left goal (x = 0) left of the right goal
    H = F.fit_lines(F.annotation_points(annotate(VIEWS["wide"]))).H
    assert F.views_pitch(H[None], np.array([[640.0, 700.0]]))[0]  # det(H) w > 0
    assert F.image_up_direction(H)[1] < -0.99


def test_the_view_class_follows_the_camera_side():
    for position, target, view in (((52.5, 103, 22), (60, 30, 0), "main side"),
                                   ((52.5, 103, 22), (3, 30, 0), "main side"),        # main camera at a corner
                                   ((52.5, -30, 18), (60, 40, 0), "opposite side"),
                                   ((-30, 34, 15), (30, 30, 0), "along the pitch")):
        H = np.linalg.inv(camera(position, target, 1200))
        assert D.view_class(F.image_up_direction(H)[1]) == view
    assert D.view_class(np.nan) == "unresolved"


# --------------------------------------------------------------------------- (b) GSR positions
def test_feet_error_is_zero_for_the_true_homography_and_a_known_shift_otherwise():
    to_frame = VIEWS["wide"]
    H = np.linalg.inv(to_frame)
    pitch = np.array([[30.0, 20.0], [52.5, 34.0], [70.0, 50.0]])
    p = np.column_stack([pitch, np.ones(3)]) @ to_frame.T
    feet = p[:, :2] / p[:, 2:]
    np.testing.assert_allclose(D.feet_errors(H, feet, pitch), 0, atol=1e-9)
    shift = np.array([[1, 0, 1.5], [0, 1, 2.0], [0, 0, 1.0]]) @ H
    np.testing.assert_allclose(D.feet_errors(shift, feet, pitch), 2.5, atol=1e-9)


def test_gsr_clip_reader(tmp_path):
    clip = tmp_path / "valid" / "SNGS-999"
    (clip / "img1").mkdir(parents=True)
    images = [{"image_id": f"9{n:06d}", "file_name": f"{n:06d}.jpg", "width": 1920, "height": 1080,
               "is_labeled": True} for n in range(1, 12)]
    annotations = [{"image_id": im["image_id"], "supercategory": "pitch", "category_id": 5,
                    "lines": {"Middle line": [{"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.9}]}} for im in images]
    box = {"x": 960 - 20, "y": 500, "w": 40, "h": 100}
    for x, y in ((10.0, -5.0), (70.0, 0.0)):                       # the second is off the pitch: dropped
        annotations.append({"image_id": "9000006", "supercategory": "object", "category_id": 1, "bbox_image": box,
                            "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y}})
    (clip / "Labels-GameState.json").write_text(json.dumps(
        {"info": {"game_id": "7", "im_dir": "img1"}, "images": images, "annotations": annotations,
         "categories": [{"id": 1, "name": "player", "supercategory": "object"}]}))
    frames = list(S.gsr_clip(clip, "valid"))
    assert [f.frame for f in frames] == [1, 6, 11]
    six = frames[1]
    assert six.key == "gsr/valid/SNGS-999/000006" and six.game == "7" and six.image == clip / "img1/000006.jpg"
    np.testing.assert_allclose(six.feet_image, [[640.0, 400.0]])
    np.testing.assert_allclose(six.feet_pitch, [[62.5, 29.0]])                  # centre origin + (52.5, 34)
    np.testing.assert_allclose(F.annotation_points(six.lines)["Middle line"], [[640, 144], [640, 648]])
    assert len(frames[0].feet_image) == 0


# --------------------------------------------------------------------------- export
def test_yolo_line_and_the_rule():
    lines = annotate(VIEWS["right box"])
    record = D.process(S.Frame("calib", "train", "g", Path("x.jpg"), lines))
    assert record["reject"] == "" and record["view"] == "main side"
    assert record["flip_visible_equal"] and record["flip_max_px"] < 1e-5
    values = D.yolo_line(record).split()
    assert len(values) == 5 + 32 * 3 and values[0] == "0"
    kp = np.array(values[5:], dtype=float).reshape(32, 3)
    assert set(kp[:, 2]) <= {0.0, 2.0}
    assert (kp[kp[:, 2] == 0, :2] == 0).all()
    assert ((kp[:, :2] >= 0) & (kp[:, :2] <= 1)).all()
    np.testing.assert_allclose(kp[kp[:, 2] > 0, :2] * (F.FRAME_W, F.FRAME_H),
                               truth_labels(VIEWS["right box"]).xy[kp[:, 2] > 0], atol=1e-3)
    for change, reason in (({"rms": 4.5}, "rms"), ({"max_residual": 17.0}, "max residual"),
                           ({"sd_max": 11.0}, "vertex sd"), ({"box": None}, "no pitch area"),
                           ({"view": "along the pitch"}, "view: along the pitch"), ({"fitted": False}, "no fit"),
                           ({"flip_max_px": 1.5}, "mirror-inconsistent"),
                           ({"flip_visible_equal": False}, "mirror-inconsistent")):
        assert D.rejection({**record, **change}) == reason


# --------------------------------------------------------------------------- download helpers
def test_fetch_keeps_every_fifth_gsr_frame_and_refuses_paths_outside(tmp_path):
    import pk_fetch_soccernet as fetch

    assert [n for n in range(1, 17) if fetch.keep_member("gamestate-2024", f"S/img1/{n:06d}.jpg")] == [1, 6, 11, 16]
    assert fetch.keep_member("gamestate-2024", "SNGS-1/img1/000006.jpg")
    assert not fetch.keep_member("gamestate-2024", "SNGS-1/img1/000007.jpg")
    assert fetch.keep_member("gamestate-2024", "SNGS-1/Labels-GameState.json")
    assert fetch.keep_member("calibration-2023", "train/00007.jpg")
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("SNGS-1/img1/000001.jpg", b"a")
        zf.writestr("SNGS-1/img1/000002.jpg", b"b")
        zf.writestr("SNGS-1/Labels-GameState.json", b"{}")
    assert fetch.safe_extract(good, tmp_path / "out", "gamestate-2024") == (2, 1)
    assert sorted(p.name for p in (tmp_path / "out/SNGS-1").rglob("*.*")) == ["000001.jpg", "Labels-GameState.json"]
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.jpg", b"x")
    (tmp_path / "bad.zip").write_bytes(bad.getvalue())
    with pytest.raises(RuntimeError, match="leaves"):
        fetch.safe_extract(tmp_path / "bad.zip", tmp_path / "out2", "calibration-2023")
    assert not (tmp_path / "evil.jpg").exists()


# --------------------------------------------------------------------------- split hygiene (council C3, Q2)
def test_near_duplicate_pairs_match_brute_force():
    from xgpoc.pitchkp import exclusions as X

    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, 40, dtype=np.int64).astype(np.uint64)
    flips = [np.uint64(1) << np.uint64(b) for b in rng.integers(0, 64, 200)]
    hashes = list(base) + [base[i % 40] ^ flips[i] ^ flips[i + 1] ^ (flips[i + 2] if i % 3 else 0)
                           for i in range(0, 150, 3)]
    hashes = np.array(hashes, dtype=np.uint64)
    brute = [(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
             if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= X.DUPLICATE_BITS]
    assert brute and X.near_duplicate_pairs(hashes) == brute


def test_dhash_sees_a_recompressed_frame_as_a_duplicate():
    import cv2
    from xgpoc.pitchkp import exclusions as X

    rng = np.random.default_rng(1)
    image = cv2.GaussianBlur(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8), (31, 31), 0)
    same = cv2.imdecode(cv2.imencode(".jpg", cv2.resize(image, (1920, 1080)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1], 1)
    other = cv2.GaussianBlur(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8), (31, 31), 0)
    assert bin(X.dhash(image) ^ X.dhash(same)).count("1") <= X.DUPLICATE_BITS
    assert bin(X.dhash(image) ^ X.dhash(other)).count("1") > 16


def test_exclusions_drop_held_out_games_shared_games_and_duplicates():
    import pandas as pd
    from xgpoc.pitchkp import exclusions as X

    plan = pd.DataFrame({"game": ["england_epl/2015-2016/2016-01-02 - 18-00 Chelsea 2 - 2 West Brom",
                                  "england_epl/2015-2016/2016-01-03 - 18-00 Arsenal 1 - 0 Newcastle Utd"],
                         "role": ["evaluation", "training"]})
    held, train_game = "england_epl/2015-2016/2016-01-02 Chelsea - West Brom", \
        "england_epl/2015-2016/2016-01-03 Arsenal - Newcastle Utd"
    rows = [  # key, source, export_split, game, dhash; the image name says which pictures are the same
        ("c1", "calib", "train", held, 1), ("c2", "calib", "val", held, 2), ("c3", "calib", "test_calib", held, 4),
        ("c4", "calib", "train", train_game, 0xFF00), ("g1", "gsr", "train", "2", 0xFF0000),
        ("g2", "gsr", "test_gsr", "2", 0xF0F0F0F0), ("g3", "gsr", "train", "7", 0xF0F0F0F1),
        ("g4", "gsr", "train", "7", 0x0F0F0F0F00)]
    frame = pd.DataFrame(rows, columns=["key", "source", "export_split", "game", "dhash"])
    frame["image"] = ["a", "b", "c", "d", "e", "same", "same", "f"]
    frame.loc[len(frame)] = ["c5", "calib", "val", train_game, 0xFF01, "g"]   # hash close to c4, another picture
    out, evidence = X.apply(frame, plan, correlation=lambda a, b: 1.0 if a == b else 0.3)
    reason = dict(zip(out["key"], out["exclusion"], strict=True))
    assert reason == {"c1": "xg held-out game", "c2": "xg held-out game", "c3": "", "c4": "",
                      "g1": "game in GSR evaluation split", "g2": "", "g3": "duplicate of a higher-priority split",
                      "g4": "", "c5": ""}
    assert [sorted(p[:2]) for p in evidence["cross_split_duplicates"]] == [["g2", "g3"]]
    assert ["c4", "c5"] in [sorted(p[:2]) for p in evidence["cross_split_candidates"]]
    assert evidence["gsr_games"]["test_gsr"] == ["2"]
