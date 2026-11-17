"""The synthetic labelled clip with the line estimator and a line model: ``evaluate_lines`` (ADR 0005)."""

import numpy as np
import pytest

import tactifoot_vision as tv
from line_points import line_points
from tactifoot_vision.data import synthetic_pitch_clip
from tactifoot_vision.enums import PitchMarking, Task
from tactifoot_vision.evaluation import evaluate_homography
from tactifoot_vision.models import Model
from tactifoot_vision.pitch import LineHomographyEstimator, SoccerPitch
from test_synthetic_clip import FRAME, _camera, _result


def test_the_line_estimator_on_its_line_points_has_a_small_error(tmp_path):
    clip = synthetic_pitch_clip(
        tmp_path / "pan.mp4", [_camera(pan) for pan in range(5)], FRAME
    )
    estimator, rng = LineHomographyEstimator(SoccerPitch()), np.random.default_rng(0)

    homographies = []
    for frame in clip:
        # No line model runs here: the points of the markings the frame
        # shows, at the pixels the calibration puts them, with 1 px of noise.
        points = line_points(frame.homography, FRAME, noise_px=1.0, rng=rng)
        homographies.append(estimator.update(points, FRAME))
    metrics = evaluate_homography(_result(clip, homographies), clip)

    assert metrics.availability == 1
    assert metrics.median_error_m < 0.5


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
        smooth_line_homographies,
    )
    from tactifoot_vision.pipeline.homographies import frame_homography
    from tactifoot_vision.pitch import LineHomographyEstimator

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
        LineHomographyEstimator(temporal_filter=settings),
    )
    reader = VideoReader(clip.video_path)
    estimator.reset(frame_rate=reader.fps)
    empty, frames = sv.Detections.empty(), []
    for index, frame in reader.frames():
        _, matrix = frame_homography(model, estimator, frame)
        frames.append(FrameResult(index, index / reader.fps, empty, empty, ball_candidates=empty,
                                  homography=matrix, homography_fit=estimator.fit))  # fmt: skip
    result = PipelineResult(frames, reader.fps, reader.size, [], estimator.pitch)
    want = evaluate_homography(smooth_line_homographies(result, **settings), clip)
    default = evaluate_homography(smooth_line_homographies(result), clip)
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
