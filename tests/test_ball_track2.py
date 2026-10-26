"""ball-detection 12: the tracking fixes of scripts/ball_track2.py on synthetic clips: every fix off is the package's
track_ball, a gap under a moving camera filled in the image, a restart between two objects, the corridor and a ball
at rest on the penalty spot."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_temporal as bt  # noqa: E402
import ball_track2 as tr  # noqa: E402

SETTING = {  # ticket 04's chosen setting
    "min_confidence": 0.05,
    "restart_confidence": 0.15,
    "restart_after": 0.25,
    "gate": 3.0,
    "gate_speed": 65.0,
    "max_gap": 2.0,
    "static_window": 1000.0,
}
SPOT = (94.0, 34.0)  # the right penalty spot of the default pitch, corner metres


def camera(t: int) -> np.ndarray:
    """pitch metres -> frame pixels at frame ``t``: 16 px per metre, panning 12 px per frame and zooming."""
    zoom = 1 + 0.004 * t
    return np.array([[16 * zoom, 0, 100 - 12 * t], [0, 12 * zoom, 80], [0, 0, 1.0]])


def to_frame(t: int, xy) -> np.ndarray:
    p = camera(t) @ np.array([*xy, 1.0])
    return p[:2] / p[2]


def motion(frames: int) -> np.ndarray:
    return np.stack(
        [camera(t + 1) @ np.linalg.inv(camera(t)) for t in range(frames - 1)]
    )


def homographies(frames: int) -> list[np.ndarray]:
    return [np.linalg.inv(camera(t)) for t in range(frames)]


def dump(balls: dict[int, list[tuple[tuple[float, float], float]]]) -> pd.DataFrame:
    """Dump rows: per 0-based frame the ball candidates as (pitch metres, confidence)."""
    rows = []
    for t, found in balls.items():
        for xy, conf in found:
            x, y = to_frame(t, xy)
            rows.append(
                {"frame": t + 1, "class_id": 0, "conf": conf}
                | dict(zip(bt.COLUMNS, [x - 4, y - 4, x + 4, y + 4], strict=True))
            )
    table = pd.DataFrame(rows, columns=["frame", "class_id", "conf", *bt.COLUMNS])
    table["conf"] = table.conf.astype(np.float32)
    return table


def tracked(balls, frames, fixes=tr.OFF, matrices=None, moves=None, **setting):
    result = bt.clip_result(dump(balls), matrices or homographies(frames))
    found, stats = tr.track_ball2(
        result,
        motion(frames) if moves is None else moves,
        fixes,
        **(SETTING | setting),
    )
    return bt.chosen_balls(found), stats


def placed(table: pd.DataFrame, frames: int) -> np.ndarray:
    """Each chosen box centre on the pitch through the true camera."""
    centres = bt.be.centres(table[bt.COLUMNS].to_numpy(float))
    out = np.full((frames, 2), np.nan)
    for t in range(frames):
        if np.isfinite(centres[t]).all():
            p = np.linalg.inv(camera(t)) @ np.array([*centres[t], 1.0])
            out[t] = p[:2] / p[2]
    return out


def live(t: int) -> tuple[float, float]:
    return 60 + 0.4 * t, 20 + 0.1 * t


def test_every_fix_off_is_the_package_tracker():
    frames = 40
    balls = {
        t: [(SPOT, 0.7)]
        + ([] if t in range(15, 22) else [(live(t), 0.6 if t < 3 else 0.08)])
        for t in range(frames)
    }
    balls[30].append(((20, 60), 0.5))  # a stray restart candidate far away
    result = bt.clip_result(dump(balls), homographies(frames))
    package = bt.chosen_balls(bt.track_ball(result, **SETTING))
    mine, stats = tr.track_ball2(result, motion(frames), tr.OFF, **SETTING)
    pd.testing.assert_frame_equal(bt.chosen_balls(mine), package)
    assert (package.status == "interpolated").sum() == 7
    assert stats["corridor_measured"] == stats["gaps_across_restart"] == 0


def test_a_gap_under_a_moving_camera_is_filled_in_the_image():
    """The ball rests on the pitch; the homographies of the two ends are 2 m off, the gap's are right. On the pitch
    the fill inherits the ends' error, in the image it is placed by the gap frame's own homography."""
    frames = 20
    balls = {t: [((50.0, 30.0), 0.6)] for t in range(frames) if t not in range(5, 15)}
    shift = np.array([[1, 0, 2.0], [0, 1, 0], [0, 0, 1]])  # pitch metres + 2 m along x
    matrices = homographies(frames)
    for t in (4, 15):
        matrices[t] = shift @ matrices[t]
    pitch, _ = tracked(balls, frames, matrices=matrices)
    image, stats = tracked(balls, frames, tr.Fixes(image_fill=True), matrices=matrices)
    gap = list(range(5, 15))
    assert (image.status.iloc[gap] == "interpolated").all()
    np.testing.assert_allclose(
        placed(pitch, frames)[gap], [[52.0, 30.0]] * 10, atol=1e-3
    )
    np.testing.assert_allclose(
        placed(image, frames)[gap], [[50.0, 30.0]] * 10, atol=1e-3
    )
    assert stats["image_filled"] == 10 and stats["pitch_fallback_filled"] == 0
    broken = motion(frames)
    broken[8] = np.nan  # a cut inside the gap: back to the pitch
    fallback, stats = tracked(
        balls, frames, tr.Fixes(image_fill=True), matrices=matrices, moves=broken
    )
    pd.testing.assert_frame_equal(fallback, pitch)
    assert stats["pitch_fallback_filled"] == 10


def test_a_restart_on_another_object_starts_a_segment_that_is_not_joined():
    frames = 40
    balls = {t: [(live(t), 0.6)] for t in range(10)}
    balls |= {t: [((10.0 + 0.2 * t, 60.0), 0.5)] for t in range(25, frames)}
    joined, _ = tracked(balls, frames)
    apart, stats = tracked(balls, frames, tr.Fixes(restart_bounds=True))
    gap = list(range(10, 25))
    assert (
        joined.status.iloc[gap] == "interpolated"
    ).all()  # the package draws a line between the two
    assert (apart.status.iloc[gap] == "none").all()
    assert (apart.status.iloc[25:] == "measured").all()
    assert stats["restarts"] == 2 and stats["gaps_across_restart"] == 1


def test_the_corridor_takes_a_weak_ball_on_the_path_over_the_interpolation():
    """A weak lone detection in the middle of a gap is removed as isolated; the corridor brings it back as measured,
    a candidate off the path stays out."""
    frames = 30
    balls = {t: [(live(t), 0.6)] for t in range(frames) if t not in range(8, 22)}
    balls[15] = [(live(15), 0.06), ((live(15)[0], live(15)[1] + 4), 0.06)]
    without, _ = tracked(balls, frames)
    with_, stats = tracked(balls, frames, tr.Fixes(corridor=2.0))
    assert without.status.iloc[15] == "interpolated"
    assert with_.status.iloc[15] == "measured"
    assert with_.conf.iloc[15] == pytest.approx(0.06)
    np.testing.assert_allclose(placed(with_, frames)[15], live(15), atol=1e-3)
    assert (with_.status.iloc[8:15] == "interpolated").all()
    assert (with_.status.iloc[16:22] == "interpolated").all()
    assert stats["corridor_measured"] == 1


def test_a_ball_followed_onto_the_penalty_spot_is_kept_and_the_painted_spot_alone_is_not():
    frames = 50
    path = {t: (SPOT[0] - 5 + 0.5 * t, SPOT[1]) for t in range(10)}
    balls = {t: [(path[t], 0.6)] for t in range(10)}
    balls |= {t: [(SPOT, 0.6)] for t in range(10, 40)}  # comes to rest on the spot
    package, _ = tracked(balls, frames)
    spots, stats = tracked(balls, frames, tr.Fixes(spot_gate=2.0))
    assert (package.status.iloc[10:40] != "measured").all()
    assert (spots.status.iloc[:40] == "measured").all()
    assert stats["spot_chosen"] == 31  # frame 9 is within 0.6 m of the spot already
    painted = {
        t: [(SPOT, 0.7)] for t in range(frames)
    }  # the painted spot, every frame, no ball
    for fixes in (tr.Fixes(spot_gate=2.0), tr.Fixes(spot_gate=2.0, spot_window=0.4)):
        alone, _ = tracked(painted, frames, fixes)
        assert (alone.status == "none").all()


def test_a_moving_ball_seen_first_on_the_spot_may_restart_the_track():
    frames = 20
    balls = {t: [((SPOT[0] - 0.5 + 0.6 * t, SPOT[1]), 0.6)] for t in range(frames)}
    gate_only, _ = tracked(balls, frames, tr.Fixes(spot_gate=2.0))
    moving, _ = tracked(balls, frames, tr.Fixes(spot_gate=2.0, spot_window=0.4))
    assert gate_only.status.iloc[0] == "none"  # on the spot and no track yet
    assert moving.status.iloc[0] == "measured"
    assert (moving.status == "measured").all()


def test_fixes_reject_bad_values():
    with pytest.raises(ValueError):
        tr.Fixes(corridor=0)
    with pytest.raises(ValueError):
        tr.Fixes(spot_window=1.0)
    assert len(tr.grid("all")) == 36 and len(tr.grid("spots")) == 9
    assert tr.fixes_name(tr.grid("all")[0]) == "image+corridor1+bounds+spot1w0.4"
