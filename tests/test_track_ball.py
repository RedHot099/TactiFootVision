"""The ``track_ball`` post-process: one ball per frame chosen among the ball candidates."""

import copy

import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.pipeline import (
    POSTPROCESSES,
    FrameResult,
    PipelineResult,
    reproject,
    track_ball,
)
from tactifoot_vision.pitch import SoccerPitch

FPS = 25.0
GAP = range(40, 44)  # frames in which the moving ball is not detected


def _homography(pitch: SoccerPitch) -> np.ndarray:
    """frame -> pitch: the pitch fills 1000 x 600 pixels."""
    return np.diag([pitch.length / 1000, pitch.width / 600, 1.0])


def _pixels(pitch: SoccerPitch, metres: tuple[float, float]) -> np.ndarray:
    """Frame pixel of a point given in metres of a standard 105 x 68 m pitch."""
    units = np.array(metres) * [pitch.length / 105, pitch.width / 68]
    return units * [1000 / pitch.length, 600 / pitch.width]


def _box(centre: np.ndarray, width: float = 6, height: float = 6) -> list[float]:
    x, y = centre
    return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]


def _frame(
    index: int,
    balls: list[tuple[list[float], float]],
    people: tuple[list[float], ...] = (),
) -> FrameResult:
    """A frame whose ball candidates are ``(xyxy, confidence)``, best first."""
    balls = sorted(balls, key=lambda ball: -ball[1])
    candidates = sv.Detections(
        xyxy=np.array([box for box, _ in balls], np.float32).reshape(-1, 4),
        confidence=np.array([conf for _, conf in balls], np.float32),
        class_id=np.zeros(len(balls), int),
        data={"class_name": np.array(["ball"] * len(balls))},
    )
    detections = sv.Detections(
        xyxy=np.array(people, np.float32).reshape(-1, 4),
        confidence=np.full(len(people), 0.9, np.float32),
        class_id=np.ones(len(people), int),
        data={"class_name": np.array(["player"] * len(people))},
    )
    return FrameResult(
        index,
        index / FPS,
        detections,
        copy.deepcopy(candidates[:1]),
        ball_candidates=candidates,
    )


def _result(pitch: SoccerPitch, frames: list[FrameResult]) -> PipelineResult:
    """The frames with pitch positions projected as the pipeline does."""
    result = PipelineResult(frames, FPS, (1100, 700), ["ball", "player"], pitch=pitch)
    return reproject(result, [_homography(pitch)] * len(frames))


def _ball_metres(t: float) -> tuple[float, float]:
    """The live ball: 12.5 m/s along the pitch, 5 m/s across."""
    return 30 + 0.5 * t, 20 + 0.2 * t


def _scene(pitch: SoccerPitch, frames: int = 75) -> PipelineResult:
    """The live ball, a spare ball behind the goal, a spot and a player read as a ball.

    The spare ball, the right penalty spot (seen on every other frame, too
    rarely to count as static) and a ball box drawn around a player all
    score higher than the live ball, which is missing during ``GAP``.
    """
    spare = _box(_pixels(pitch, (106.5, 30)))
    spot = _box(_pixels(pitch, (94, 34)))
    result = []
    for t in range(frames):
        player = _box(_pixels(pitch, (60 - 0.1 * t, 50)), width=20, height=40)
        balls = [(spare, 0.9), (player, 0.95)]
        if t % 2 == 0:
            balls.append((spot, 0.8))
        if t not in GAP:
            balls.append((_box(_pixels(pitch, _ball_metres(t))), 0.6))
        result.append(_frame(t, balls, people=[player]))
    return _result(pitch, result)


def _units(pitch: SoccerPitch, metres: tuple[float, float]) -> np.ndarray:
    return np.array(metres) * [pitch.length / 105, pitch.width / 68]


def _interpolated(frame: FrameResult) -> list[bool]:
    return frame.ball.data["interpolated"].tolist()


@pytest.mark.parametrize(
    "pitch", [SoccerPitch(), SoccerPitch(120, 80)], ids=["105x68", "120x80"]
)
def test_track_ball_follows_the_moving_ball_past_a_spare_ball_and_a_spot(pitch):
    result = _scene(pitch)

    tracked = track_ball(result)

    for t, frame in enumerate(tracked):
        assert len(frame.ball) == 1, t
        np.testing.assert_allclose(
            frame.ball_xy[0], _units(pitch, _ball_metres(t)), atol=1e-3
        )
        assert _interpolated(frame) == [t in GAP]
    # before tracking the most confident candidate, the player box, is the ball
    assert result[0].ball.xyxy.tolist() == result[0].detections.xyxy.tolist()


def test_track_ball_fills_a_short_gap_and_flags_it():
    pitch = SoccerPitch()
    result = _scene(pitch)

    tracked = track_ball(result)

    for t in GAP:
        ball = tracked.frame(t).ball
        assert ball.data["interpolated"].tolist() == [True]
        assert np.isnan(ball.confidence).all()  # no detection, no confidence
        assert ball.data["class_name"].tolist() == ["ball"]
        # the box sits on the filled position, the size of its neighbours'
        np.testing.assert_allclose(
            ball.xyxy[0], _box(_pixels(pitch, _ball_metres(t))), atol=1e-3
        )
        assert len(tracked.frame(t).ball_candidates) == len(
            result.frame(t).ball_candidates
        )
    table = tracked.to_dataframe()
    balls = table[table["object"] == "ball"]
    assert balls["interpolated"].tolist() == [t in GAP for t in range(len(result))]
    assert not table.loc[table["object"] == "person", "interpolated"].any()
    frames = tracked.to_freeze_frames()
    assert frames.loc[frames["type"] == "ball", "interpolated"].sum() == len(GAP)
    # the input is left alone and keeps its unflagged ball
    assert "interpolated" not in result.frame(GAP[0]).ball.data
    assert result.to_dataframe()["interpolated"].eq(False).all()


def test_track_ball_leaves_a_gap_longer_than_max_gap_empty():
    pitch = SoccerPitch()

    tracked = track_ball(_scene(pitch), max_gap=len(GAP) / FPS)  # 4 frames: 0.16 s

    for t in GAP:
        assert len(tracked.frame(t).ball) == 0
    assert _interpolated(tracked.frame(GAP[-1] + 1)) == [False]


def test_track_ball_rejects_a_player_sized_ball_candidate():
    # RF-DETR scores every query in every class: a player box can come back as
    # a confident ball candidate, while a ball at a player's feet sits inside
    # the player's box without matching it.
    pitch = SoccerPitch()
    frames = []
    for t in range(10):
        feet = _pixels(pitch, _ball_metres(t))
        player = _box(feet - [0, 20], width=20, height=40)
        frames.append(_frame(t, [(player, 0.95), (_box(feet), 0.6)], people=[player]))
    result = _result(pitch, frames)

    tracked = track_ball(result)

    for t, frame in enumerate(tracked):
        np.testing.assert_allclose(
            frame.ball_xy[0], _units(pitch, _ball_metres(t)), atol=1e-3
        )
    # with the overlap check off, the player box starts the track
    assert track_ball(result, max_person_iou=1.0)[0].ball.confidence[
        0
    ] == pytest.approx(0.95)


def test_track_ball_restarts_only_on_a_confident_candidate():
    pitch = SoccerPitch()
    far = _box(_pixels(pitch, (80, 50)))
    frames = []
    for t in range(40):
        if t < 10:
            balls = [(_box(_pixels(pitch, _ball_metres(t))), 0.6)]
        elif t < 25:  # weak far candidates only: no restart
            balls = [(far, 0.3)]
        else:  # a confident one restarts the track
            balls = [(far, 0.3), (_box(_pixels(pitch, (70, 10))), 0.5)]
        frames.append(_frame(t, balls))

    tracked = track_ball(_result(pitch, frames))

    assert all(len(tracked[t].ball) == 1 for t in range(10))
    assert all(len(tracked[t].ball) == 0 for t in range(10, 25))
    for t in range(25, 40):
        np.testing.assert_allclose(tracked[t].ball_xy[0], (70, 10), atol=1e-3)
        assert _interpolated(tracked[t]) == [False]
    # a lower restart confidence lets the weak candidates restart it
    assert len(track_ball(_result(pitch, frames), restart_confidence=0.2)[12].ball) == 1


def test_track_ball_removes_isolated_outliers():
    pitch = SoccerPitch()
    frames = []
    for t in range(60):
        balls = []
        if t < 10 or 28 <= t < 45:
            balls.append((_box(_pixels(pitch, (40, 30))), 0.6))  # the ball, at rest
        if t == 20:  # a confident false detection after the ball was lost
            balls.append((_box(_pixels(pitch, (90, 60))), 0.9))
        if t == 50:  # a weak lone detection inside the gate, 0.24 s after the ball
            balls.append((_box(_pixels(pitch, (40.5, 30))), 0.3))
        frames.append(_frame(t, balls))

    tracked = track_ball(_result(pitch, frames), static_window=10.0)

    # the false detection restarts the track, but differs from the ball
    # before and after it; the weak one has no neighbour within `isolation`
    assert len(tracked[20].ball) == 0
    assert len(tracked[50].ball) == 0
    assert len(tracked[9].ball) == len(tracked[30].ball) == 1
    # the ball 0.24 s before it is within a 0.3 s isolation
    wider = track_ball(_result(pitch, frames), static_window=10.0, isolation=0.3)
    assert len(wider[50].ball) == 1


def test_track_ball_stops_extrapolating_the_lost_ball_after_max_gap():
    # Lost at 10 m/s, the ball comes back weak (no restart) where it was
    # lost, 2 s later: a prediction carried on for 2 s would sit 20 m away,
    # beyond the gate, which stops widening after max_gap.
    pitch = SoccerPitch()
    frames = []
    for t in range(80):
        if t < 10:
            balls = [(_box(_pixels(pitch, (30 + 0.4 * t, 20))), 0.6)]
        elif t >= 60:
            balls = [(_box(_pixels(pitch, (35, 20))), 0.4)]
        else:
            balls = []
        frames.append(_frame(t, balls))

    tracked = track_ball(_result(pitch, frames))

    assert all(len(tracked[t].ball) == 1 for t in range(60, 80))
    assert all(len(tracked[t].ball) == 0 for t in range(10, 60))


def test_track_ball_needs_increasing_timestamps():
    pitch = SoccerPitch()
    ball = [(_box(_pixels(pitch, (30, 20))), 0.6)]
    with pytest.raises(ValueError, match="timestamps"):
        track_ball(_result(pitch, [_frame(1, ball), _frame(1, ball)]))


def test_track_ball_rejects_static_candidates_only_over_a_long_enough_window():
    pitch = SoccerPitch()
    spare = _box(_pixels(pitch, (106.5, 30)))
    frames = [_frame(t, [(spare, 0.9)]) for t in range(75)]  # 3 s

    assert all(len(f.ball) == 0 for f in track_ball(_result(pitch, frames)))
    # a window longer than twice the clip cannot tell a still ball from a static one
    assert all(
        len(f.ball) == 1 for f in track_ball(_result(pitch, frames), static_window=7.0)
    )


def test_track_ball_is_a_registered_post_process_that_leaves_its_input_alone():
    result = _scene(SoccerPitch(), frames=50)
    snapshot = copy.deepcopy(result)

    tracked = POSTPROCESSES.get("track_ball")(result, max_gap=0.3)

    assert tracked.postprocesses[-1].name == "track_ball"
    assert tracked.postprocesses[-1].options["max_gap"] == 0.3
    assert "gate" in tracked.postprocesses[-1].options  # defaults are recorded
    for before, after in zip(result, snapshot, strict=True):
        for name in ("ball", "ball_candidates"):
            old, new = getattr(before, name), getattr(after, name)
            np.testing.assert_array_equal(old.xyxy, new.xyxy)
            np.testing.assert_array_equal(old.data["pitch_xy"], new.data["pitch_xy"])


@pytest.mark.parametrize(
    "options",
    [
        {"gate": 0},
        {"max_gap": -1},
        {"static_share": 1.5},
        {"max_person_iou": 0},
        {"restart_confidence": 2},
        {"restart_after": -0.1},
        {"spot_radius": -1},
        {"isolation": -1},
    ],
)
def test_track_ball_validates_its_options(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        track_ball(_scene(SoccerPitch(), frames=3), **options)
