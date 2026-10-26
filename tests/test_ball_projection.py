"""ball-detection 11: the projection point chosen in the ball-selection dump (scripts/ball_projection.py), read by the
unchanged placement read, the variant rule, and the camera recovered from a homography for the diagnostics."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_projection as bp  # noqa: E402

COLUMNS = ["x1", "y1", "x2", "y2"]


def gsr_labels(folder: Path, game: str, frames: list[int], truth_xy) -> None:
    """A minimal GSR ``Labels-GameState.json``: one ball per frame at ``truth_xy`` (centre-origin metres)."""
    folder.mkdir(parents=True)
    labels = {
        "info": {"game_id": game},
        "categories": [{"id": 4, "name": "ball"}],
        "images": [{"image_id": f, "file_name": f"{f:06d}.jpg"} for f in frames],
        "annotations": [
            {
                "category_id": 4,
                "image_id": f,
                "bbox_pitch": {
                    "x_bottom_middle": truth_xy[0],
                    "y_bottom_middle": truth_xy[1],
                },
            }
            for f in frames
        ],
    }
    (folder / "Labels-GameState.json").write_text(json.dumps(labels))


def dump(clip: str, rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"clip": clip, "frame": f, "status": s}
            | dict(zip(COLUMNS, xyxy, strict=True))
            for f, s, xyxy in rows
        ]
    )


def test_with_point_moves_the_point_down_the_box_and_leaves_frames_without_a_ball_empty():
    table = dump(
        "SNGS-001",
        [
            (1, "measured", (100, 200, 110, 240)),
            (6, "interpolated", (0, 0, 20, 10)),
            (11, "none", (np.nan,) * 4),
        ],
    )
    centre, half, bottom = (bp.with_point(table, k) for k in (0.0, 0.5, 1.0))
    assert centre.point_x.tolist()[:2] == [105, 10]
    assert centre.point_y.tolist()[:2] == [220, 5]
    assert half.point_y.tolist()[:2] == [230, 7.5]
    assert bottom.point_y.tolist()[:2] == [240, 10]
    assert bottom.iloc[2][["point_x", "point_y"]].isna().all()
    assert "point_x" not in table  # the dump itself is not changed


@pytest.fixture
def placement_case(tmp_path, monkeypatch):
    """Truth at corner metres (50, 30) = 720p pixel (500, 300) under H = 0.1 m per px = 1080p (750, 450), on frames
    1 and 6 of a game 2 clip. The tracked box (1080p, 240 px high) has its bottom on the truth: the centre lies 120 px
    = 80 px at 720p = 8 m off, k = 0.25 6 m, k = 0.5 4 m, k = 1 exact. Top-1 has a ball on frame 1 only. A game 5
    clip with the same balls has no homography and is left out."""
    for clip, game in (("SNGS-001", "2"), ("SNGS-002", "5")):
        gsr_labels(tmp_path / "valid" / clip, game, [1, 2, 6], (50 - 52.5, 30 - 34))
    monkeypatch.setattr(bp.bench, "GSR", tmp_path)
    homographies = {("SNGS-001", f): np.diag([0.1, 0.1, 1.0]) for f in (1, 6)}
    box = (740, 450 - 240, 760, 450)
    tracks = pd.concat(
        [
            dump(c, [(1, "measured", box), (6, "interpolated", box)])
            for c in ("SNGS-001", "SNGS-002")
        ],
        ignore_index=True,
    )
    top1 = pd.concat(
        [
            dump(c, [(1, "measured", box), (6, "none", (np.nan,) * 4)])
            for c in ("SNGS-001", "SNGS-002")
        ],
        ignore_index=True,
    )
    return tracks, top1, homographies


def test_the_read_places_the_dumps_point_on_the_selection_games_only(placement_case):
    tracks, top1, homographies = placement_case
    centre = bp.placement(tracks, 0.0, homographies)
    bottom = bp.placement(tracks, 1.0, homographies)
    assert centre["clip"].unique().tolist() == ["SNGS-001"]
    assert centre.frame.tolist() == [1, 6]
    assert centre.error_m.tolist() == pytest.approx([8.0, 8.0])
    assert bottom.error_m.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert bp.placement(tracks, 0.25, homographies).error_m.tolist() == pytest.approx(
        [6.0, 6.0]
    )


def test_variants_pair_with_the_centre_and_b_takes_the_smallest_best_k(placement_case):
    tracks, top1, homographies = placement_case
    found = bp.read_variants(tracks, top1, homographies, ["SNGS-001"])
    assert set(found) == {0.0, *bp.GRID}
    gain = found[1.0]["track_vs_centre"]
    assert gain["well_placed"]["difference"] == pytest.approx(1.0)
    assert gain["far_of_all"]["difference"] == pytest.approx(-1.0)
    assert found[0.0]["track"]["far_of_all"][0] == pytest.approx(1.0)
    assert found[0.25]["track_vs_centre"]["well_placed"]["difference"] == 0
    # top-1 has one ball of the two frames: the single-frame row moves by half
    assert found[1.0]["top1_vs_centre"]["well_placed"]["difference"] == pytest.approx(
        0.5
    )
    assert found[1.0]["track_vs_top1_centre"]["well_placed"][
        "difference"
    ] == pytest.approx(1.0)
    assert bp.choose_k(found) == 0.5  # 0.5, 0.75 and 1 tie: the smaller k
    assert bp.verdict(gain)["wins"]
    assert not bp.verdict(found[0.25]["track_vs_centre"])["wins"]


def test_verdict_needs_a_positive_lower_bound_and_the_far_barrier():
    def entry(low, far):
        return {
            "well_placed": {"difference": low + 0.01, "ci": [low, low + 0.02]},
            "far_of_all": {"difference": far, "ci": [far, far]},
        }

    assert bp.verdict(entry(0.001, 0.01))["wins"]  # +1 pp is allowed
    assert not bp.verdict(entry(0.001, 0.0101))["wins"]
    assert not bp.verdict(entry(0.0, -0.05))["wins"]


def test_camera_recovers_the_ground_point_and_height_of_a_broadcast_camera():
    """A camera 12 m high at corner metres (52.5, 90), 20 m behind the far touchline, looking at (40, 30)."""
    centre, target, focal = (
        np.array([52.5, 90.0, 12.0]),
        np.array([40.0, 30.0, 0.0]),
        1500.0,
    )
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])  # world -> camera rows
    k = np.array([[focal, 0, 640], [0, focal, 360], [0, 0, 1.0]])
    t = -rotation @ centre
    pitch_to_image = k @ np.stack([rotation[:, 0], rotation[:, 1], t], 1)
    found = bp.camera(np.linalg.inv(pitch_to_image))
    assert found == pytest.approx((focal, 52.5, 90.0, 12.0), rel=1e-6)


def test_error_components_split_the_error_along_and_across_the_camera_ray():
    rows = pd.DataFrame(
        {
            "clip": "c",
            "truth_x": [52.5, 52.5],
            "truth_y": [30.0, 30.0],
            "x": [52.5, 53.5],
            "y": [28.0, 30.0],  # 2 m further from a camera at y = 90; 1 m sideways
            "error_m": [2.0, 1.0],
        }
    )
    cameras = pd.DataFrame(
        {"clip": ["c"], "cam_x": [52.5], "cam_y": [90.0], "cam_h": [12.11]}
    )
    table = bp.error_components(rows, cameras)
    assert table.along_m.tolist() == pytest.approx([2.0, 0.0])
    assert np.abs(table.across_m).tolist() == pytest.approx([0.0, 1.0])
    assert table.distance_m.tolist() == pytest.approx([60.0, 60.0])
    assert table.expected_centre_offset_m.tolist() == pytest.approx([0.55, 0.55])
