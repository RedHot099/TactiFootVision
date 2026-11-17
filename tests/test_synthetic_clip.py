"""Synthetic labelled clip: pitch markings rendered through known homographies."""

import cv2
import numpy as np
import pytest
import supervision as sv

import tactifoot_vision as tv
from line_points import line_points
from tactifoot_vision.data import (
    LabelledClip,
    Truth,
    VideoReader,
    synthetic_pitch_clip,
)
from tactifoot_vision.enums import PitchMarking, Task
from tactifoot_vision.evaluation import evaluate_homography
from tactifoot_vision.models import Model
from tactifoot_vision.pipeline import FrameResult, PipelineResult
from tactifoot_vision.pitch import HomographyEstimator, SoccerPitch, pitch_to_frame

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


def test_homography_estimator_on_its_line_points_has_a_small_error(tmp_path):
    clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", [_camera(pan) for pan in range(5)], FRAME
    )
    estimator, rng = HomographyEstimator(SoccerPitch()), np.random.default_rng(0)

    homographies = []
    for frame in clip:
        # No line model runs here: the points of the markings the frame
        # shows, at the pixels the calibration puts them, with 1 px of noise.
        points = line_points(frame.homography, FRAME, noise_px=1.0, rng=rng)
        homographies.append(estimator.update(points, FRAME))
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


class _PerfectLines(Model):
    """Returns, frame by frame, the line points a perfect model finds under the clip's calibration."""

    name = "perfect_lines"
    task = Task.LINES

    def __init__(self, homographies, noise_px=0.0):
        self.homographies, self.noise_px, self.seen = list(homographies), noise_px, 0
        super().__init__(None, device="cpu")

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self):
        return [m.value for m in PitchMarking]

    def predict(self, image):
        matrix = self.homographies[
            self.seen % len(self.homographies)
        ]  # clip after clip
        self.seen += 1
        rng = np.random.default_rng(self.seen)
        return line_points(
            matrix, (image.shape[1], image.shape[0]), noise_px=self.noise_px, rng=rng
        )


def test_evaluate_lines_scores_a_line_model_through_its_homography(tmp_path):
    from tactifoot_vision.evaluation import evaluate, evaluate_lines

    clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", [_camera(pan) for pan in range(6)], FRAME
    )
    truth = [f.homography for f in clip]
    metrics = evaluate_lines(_PerfectLines(truth), clip)
    assert metrics.availability == 1 and metrics.median_error_m < 0.05
    noisy = evaluate_lines(
        _PerfectLines(truth, noise_px=1.0), [clip, clip], smooth=False, draws=50
    )
    assert noisy.availability == 1 and noisy.intervals is not None
    with pytest.raises(ValueError, match="evaluate_lines"):
        evaluate(
            _PerfectLines(truth), tv.data.Dataset(Task.LINES, ["x"], {"valid": []})
        )


def test_evaluate_lines_smooths_with_the_estimator_s_own_filter(tmp_path):
    """The final track evaluate_lines scores is the RTS pass of the filter its estimator ran, not of the default one."""
    import supervision as sv

    from tactifoot_vision.data import VideoReader
    from tactifoot_vision.evaluation import evaluate_homography, evaluate_lines
    from tactifoot_vision.pipeline import (
        FrameResult,
        PipelineResult,
        smooth_homographies,
    )
    from tactifoot_vision.pipeline.homographies import frame_homography
    from tactifoot_vision.pitch import HomographyEstimator

    clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", [_camera(pan) for pan in range(8)], FRAME
    )
    truth = [f.homography for f in clip]
    settings = {
        "accel_m": 0.6,
        "meas_m": 0.3,
        "gate_chi2": 25.0,
    }  # a different noise ratio
    got = evaluate_lines(
        _PerfectLines(truth, noise_px=2.0),
        clip,
        estimator={"temporal_filter": settings},
    )
    # the same by hand: the estimator, then the post-process with those settings
    model, estimator = (
        _PerfectLines(truth, noise_px=2.0),
        HomographyEstimator(temporal_filter=settings),
    )
    reader = VideoReader(clip.video_path)
    estimator.reset(frame_rate=reader.fps)
    empty, frames = sv.Detections.empty(), []
    for index, frame in reader.frames():
        _, matrix = frame_homography(model, estimator, frame)
        frames.append(FrameResult(index, index / reader.fps, empty, empty, ball_candidates=empty,
                                  homography=matrix, homography_fit=estimator.fit))  # fmt: skip
    result = PipelineResult(frames, reader.fps, reader.size, [], estimator.pitch)
    want = evaluate_homography(smooth_homographies(result, **settings), clip)
    default = evaluate_homography(smooth_homographies(result), clip)
    assert got.as_dict() == want.as_dict()
    assert got.as_dict() != default.as_dict()  # the settings steer the score
    # without a filter there is no pass: the per-frame matrices are scored
    off = evaluate_lines(
        _PerfectLines(truth, noise_px=2.0), clip, estimator={"temporal_filter": None}
    )
    raw = evaluate_lines(
        _PerfectLines(truth, noise_px=2.0),
        clip,
        smooth=False,
        estimator={"temporal_filter": None},
    )
    assert off.as_dict() == raw.as_dict()
