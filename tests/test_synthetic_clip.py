"""Synthetic labelled clip: pitch markings rendered through known homographies."""

import cv2
import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.data import (
    LabelledClip,
    Truth,
    VideoReader,
    broadcast_homographies,
    synthetic_pitch_clip,
)
from tactifoot_vision.evaluation import evaluate_homography
from tactifoot_vision.pipeline import FrameResult, PipelineResult
from tactifoot_vision.pitch import (
    HomographyEstimator,
    SoccerPitch,
    frame_to_pitch,
    pitch_to_frame,
)

FRAME = (320, 180)


def _camera(pan=0.0, pitch=None):
    """Frame -> pitch homography of a broadcast-like view of the centre, panned ``pan`` metres along the length."""
    pitch = pitch or SoccerPitch()
    sx, sy = pitch.length / 105, pitch.width / 68
    centre = 52.5 + pan
    src = np.float32(
        [[centre - 25, 68], [centre + 25, 68], [centre + 35, 0], [centre - 35, 0]]
    ) * np.float32([sx, sy])
    dst = np.float32([[0, 180], [320, 180], [290, 15], [30, 15]])
    return cv2.getPerspectiveTransform(dst, src)


def _frames(clip):
    return [frame for _, frame in VideoReader(clip.video_path).frames()]


def _result(clip, homographies, pitch=None):
    empty = sv.Detections.empty()
    frames = [
        FrameResult(i, i / clip.fps, empty, empty, ball_candidates=empty, homography=h)
        for i, h in enumerate(homographies)
    ]
    return PipelineResult(
        frames,
        clip.fps,
        (clip.width, clip.height),
        ["ball", "player"],
        pitch or SoccerPitch(),
    )


def test_the_clip_carries_its_calibration_and_its_video(tmp_path):
    cameras = [_camera(pan) for pan in (0, 1, 2)]

    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", cameras, FRAME, fps=10)

    assert isinstance(clip, LabelledClip)
    assert clip.truth == {Truth.CALIBRATION}
    assert (clip.name, clip.video_path) == ("pan", tmp_path / "pan.mp4")
    assert (clip.width, clip.height, clip.fps) == (*FRAME, 10)
    assert [f.index for f in clip] == [0, 1, 2]
    for frame, camera in zip(clip, cameras, strict=True):
        np.testing.assert_allclose(frame.homography, camera)
        assert len(frame.detections) == 0 and len(frame.ball) == 0
    video = VideoReader(clip.video_path)
    assert (video.size, video.frame_count, video.fps) == (FRAME, 3, 10)


@pytest.mark.parametrize("pitch", [SoccerPitch(), SoccerPitch(120, 80)])
def test_markings_are_drawn_where_the_calibration_puts_them(tmp_path, pitch):
    camera = _camera(pitch=pitch)

    clip = synthetic_pitch_clip(
        tmp_path / "static.mp4", [camera], FRAME, pitch=pitch, line_width=0.4
    )

    (image,) = _frames(clip)
    # The halfway line's ends and the centre circle's crossings with it.
    # Frame pixel i covers [i, i + 1).
    on_lines = np.floor(pitch_to_frame(pitch.vertices[13:17], camera)).astype(int)
    for x, y in on_lines:
        x, y = min(x, FRAME[0] - 1), min(y, FRAME[1] - 1)
        assert image[y, x].min() > 150, (x, y, image[y, x])
    # Grass between the circle and the penalty box: green, not line.
    grass = pitch_to_frame([[pitch.length * 0.33, pitch.width * 0.5]], camera)
    x, y = np.floor(grass[0]).astype(int)
    blue, green, red = image[y, x].astype(int)
    assert green > red + 30 and green > blue + 30 and green < 200


def test_a_panned_frame_draws_its_markings_where_its_calibration_puts_them(tmp_path):
    cameras = [_camera(0), _camera(10)]  # the second looks 10 m to the right

    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", cameras, FRAME, line_width=0.4)

    before, after = _frames(clip)
    # The right penalty box's front corners: in view only after the pan.
    corners = pitch_to_frame(SoccerPitch().vertices[[17, 20]], cameras[1])
    for x, y in np.floor(corners).astype(int):
        assert after[y, x].min() > 150, (x, y, after[y, x])
        assert before[y, x].min() < 150, (x, y, before[y, x])


def test_the_far_side_of_the_horizon_is_not_pitch(tmp_path):
    # Pitch -> frame of a camera standing at x = 50 on the pitch, looking
    # towards x = 105, with the horizon at frame row 40: without care, the
    # half behind it would mirror into the sky.
    to_frame = np.array([[160, 100, -11400], [40, 0, -1700], [1, 0, -50.0]])
    clip = synthetic_pitch_clip(
        tmp_path / "horizon.mp4", [np.linalg.inv(to_frame)], FRAME
    )

    (image,) = _frames(clip)
    sky = image[:35].reshape(-1, 3).astype(int)
    assert np.ptp(sky, axis=0).max() < 20  # one flat colour
    blue, green, red = sky.mean(axis=0)
    assert not (green > red + 30 and green > blue + 30)
    blue, green, red = image[120, 60].astype(int)  # pitch at about (53.8, 30.3)
    assert green > red + 30 and green > blue + 30


def test_evaluating_the_true_homographies_scores_zero(tmp_path):
    clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", [_camera(pan) for pan in range(4)], FRAME
    )

    metrics = evaluate_homography(_result(clip, [f.homography for f in clip]), clip)

    assert metrics.median_error_m == pytest.approx(0, abs=1e-6)
    assert metrics.p90_error_m == pytest.approx(0, abs=1e-6)
    assert metrics.availability == 1
    assert metrics.num_frames == 4


def test_homography_estimator_on_its_keypoints_has_a_small_error(tmp_path):
    clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", [_camera(pan) for pan in range(5)], FRAME
    )
    pitch, rng = SoccerPitch(), np.random.default_rng(0)
    estimator = HomographyEstimator(pitch, smoothing_window=1)

    homographies = []
    for frame in clip:
        # No keypoint model runs here: the landmarks the frame shows, at the
        # pixels the calibration puts them, with 1 px of noise.
        xy = pitch_to_frame(pitch.vertices, frame.homography)
        shown = ((xy >= 0) & (xy <= FRAME)).all(axis=1)
        xy = xy + rng.normal(0, 1, xy.shape)
        keypoints = sv.KeyPoints(
            xy=xy[None].astype(np.float32),
            confidence=shown[None].astype(np.float32),
        )
        homographies.append(estimator.update(keypoints))
    metrics = evaluate_homography(_result(clip, homographies), clip)

    assert metrics.availability == 1
    assert metrics.median_error_m < 0.5


def test_output_is_deterministic_for_a_seed(tmp_path):
    cameras = [_camera(pan) for pan in (0, 1)]

    def render(name, **options):
        clip = synthetic_pitch_clip(tmp_path / name, cameras, FRAME, **options)
        return np.stack(_frames(clip))

    plain = render("plain.mp4")
    first = render("a.mp4", noise=10, occluders=4, seed=1)
    again = render("b.mp4", noise=10, occluders=4, seed=1)
    other = render("c.mp4", noise=10, occluders=4, seed=2)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert not np.array_equal(first, plain)


def test_occluders_hide_part_of_the_pitch(tmp_path):
    camera = _camera()
    plain = synthetic_pitch_clip(tmp_path / "plain.mp4", [camera], FRAME)
    hidden = synthetic_pitch_clip(tmp_path / "hidden.mp4", [camera], FRAME, occluders=5)

    (a,), (b,) = _frames(plain), _frames(hidden)
    changed = np.abs(a.astype(int) - b.astype(int)).max(axis=2) > 40
    assert 0.01 < changed.mean() < 0.5


@pytest.mark.parametrize(
    ("homographies", "options", "message"),
    [
        ([], {}, "at least one homography"),
        ([np.eye(2)], {}, "3x3"),
        ([np.zeros((3, 3))], {}, "invertible"),
        ([[[1, 0, 0], [0, 1, 0], [1, 0, -160]]], {}, "horizon"),
        ([np.eye(3)], {"frame_size": (321, 180)}, "even"),
        ([np.eye(3)], {"noise": -1}, "noise"),
        ([np.eye(3)], {"occluders": -1}, "occluders"),
        ([np.eye(3)], {"line_width": 0}, "line_width"),
        ([np.eye(3)], {"fps": 0}, "fps"),
    ],
)
def test_invalid_arguments_are_refused(tmp_path, homographies, options, message):
    options = {"frame_size": FRAME, **options}
    with pytest.raises(ValueError, match=message):
        synthetic_pitch_clip(tmp_path / "x.mp4", homographies, **options)
    assert not (tmp_path / "x.mp4").exists()


def test_an_image_sequence_path_writes_one_image_per_frame_from_0(tmp_path):
    # Frames with a file of their own, so a dataset can reference them in place.
    cameras = [_camera(pan) for pan in (0, 1, 2)]
    pattern = tmp_path / "pan" / "%06d.png"

    clip = synthetic_pitch_clip(pattern, cameras, FRAME, line_width=0.4)

    assert (clip.name, clip.video_path) == ("pan", pattern)
    assert sorted(p.name for p in pattern.parent.iterdir()) == [
        f"{n:06d}.png" for n in range(3)
    ]
    video = VideoReader(pattern)
    assert (video.size, video.frame_count) == (FRAME, 3)
    # Lossless: the same pixels as the video's first frame, but for its codec.
    video_clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", cameras, FRAME, line_width=0.4
    )
    image, encoded = (
        cv2.imread(str(pattern.parent / "000000.png")),
        _frames(video_clip)[0],
    )
    assert np.abs(image.astype(int) - encoded).mean() < 3


def test_an_image_sequence_with_images_beyond_the_new_frames_is_refused(tmp_path):
    # A longer earlier render would leave frames the new clip does not label.
    pattern = tmp_path / "pan" / "%06d.png"
    synthetic_pitch_clip(pattern, [_camera(pan) for pan in (0, 1, 2)], FRAME)

    with pytest.raises(ValueError, match="000002.png"):
        synthetic_pitch_clip(pattern, [_camera(pan) for pan in (0, 1)], FRAME)
    # The same length again overwrites its images.
    synthetic_pitch_clip(pattern, [_camera(pan) for pan in (3, 4, 5)], FRAME)


# ----------------------------------------------------- broadcast camera views
def _centres(homographies, frame=FRAME):
    centre = np.array([frame]) / 2
    return np.array([frame_to_pitch(centre, h)[0] for h in homographies])


def test_broadcast_homographies_follow_the_seed():
    first = broadcast_homographies(5, FRAME, seed=1)
    again = broadcast_homographies(5, FRAME, seed=1)
    other = broadcast_homographies(5, FRAME, seed=2)

    assert len(first) == 5 and all(h.shape == (3, 3) for h in first)
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert not np.allclose(np.stack(first), np.stack(other))


def test_every_broadcast_view_has_the_frame_centre_on_the_pitch(tmp_path):
    homographies = broadcast_homographies(200, FRAME, seed=0)

    centres = _centres(homographies)
    assert ((centres >= 0) & (centres <= [105, 68])).all()
    # The renderer's precondition: it renders them all without complaint.
    synthetic_pitch_clip(tmp_path / "views.mp4", homographies[:3], FRAME)


def test_broadcast_views_look_across_from_one_touchline_and_pan_tilt_and_zoom():
    homographies = broadcast_homographies(
        200, FRAME, field_of_view=(15.0, 60.0), seed=0
    )
    width, height = FRAME

    centres = _centres(homographies)
    # Pan along the whole length; tilt from the near touchline into the far
    # half (the far touchline itself stays above the centre, in view).
    assert centres[:, 0].min() < 15 and centres[:, 0].max() > 90
    assert centres[:, 1].min() < 25 and centres[:, 1].max() > 60
    for h in homographies:
        # As a main broadcast camera sees it: the touchline y = 68 is nearest
        # the bottom of the frame, and x grows to the right.
        middle = frame_to_pitch(
            [[width / 2, height / 2], [width / 2, height], [width * 0.6, height / 2]],
            h,
        )
        assert middle[1, 1] > middle[0, 1]
        assert middle[2, 0] > middle[0, 0]
    # Zoom: from wide views to close ones.
    spans = [
        np.linalg.norm(
            np.subtract(*frame_to_pitch([[0, height / 2], [width, height / 2]], h))
        )
        for h in homographies
    ]
    assert min(spans) < 20 and max(spans) > 50


@pytest.mark.parametrize(
    ("count", "frame_size", "message"),
    [(0, FRAME, "count"), (1, (0, 180), "frame_size")],
)
def test_broadcast_homographies_refuse_invalid_arguments(count, frame_size, message):
    with pytest.raises(ValueError, match=message):
        broadcast_homographies(count, frame_size)


def _landmarks_in_view(homography, size):
    width, height = size
    points = pitch_to_frame(SoccerPitch().vertices, homography)
    inside = (points[:, 0] >= 0) & (points[:, 0] < width)
    return int((inside & (points[:, 1] >= 0) & (points[:, 1] < height)).sum())


def test_a_wider_field_of_view_shows_more_pitch_landmarks():
    size = (640, 360)
    counts = {
        fov: np.median(
            [
                _landmarks_in_view(h, size)
                for h in broadcast_homographies(60, size, field_of_view=fov, seed=1)
            ]
        )
        for fov in ((15.0, 30.0), (35.0, 75.0))
    }
    assert counts[(35.0, 75.0)] > counts[(15.0, 30.0)] + 3


@pytest.mark.parametrize("field_of_view", [(0, 40), (50, 40), (40, 180)])
def test_broadcast_homographies_refuse_an_invalid_field_of_view(field_of_view):
    with pytest.raises(ValueError, match="field_of_view"):
        broadcast_homographies(2, field_of_view=field_of_view)
