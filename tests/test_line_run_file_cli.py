"""Run files and the ``tactifoot`` command with a ``lines`` section, beside ``keypoints``."""

import logging
from dataclasses import asdict

import cv2
import numpy as np
import pytest

import tactifoot_vision as tv
from line_points import line_points
from tactifoot_vision import run_file
from tactifoot_vision.cli import main
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.models import MODELS, Model
from tactifoot_vision.pipeline import PipelineResult, smooth_line_homographies
from tactifoot_vision.pitch import (
    FitConfig,
    HomographyEstimator,
    LineHomographyEstimator,
    SoccerPitch,
    fit,
)
from tactifoot_vision.pitch.temporal import FilterConfig
from test_run_file_cli import (  # noqa: F401  (_restore_logging: an autouse fixture)
    DETECTOR,
    _load,
    _restore_logging,
    _write,
)


@MODELS.register("test_pan_lines")
class PanLines(Model):
    """The line points a perfect line model finds while the camera pans, with a little noise."""

    task = Task.LINES
    name = "test_pan_lines"

    def __init__(self, weights=None, *, conf=0.2, device="cpu"):
        super().__init__(weights, conf=conf, device=device)

    def _load(self, weights):
        self.seen = 0

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self):
        return [m.value for m in tv.enums.PitchMarking]

    def predict(self, image):
        h, w = image.shape[:2]
        k, self.seen = self.seen, self.seen + 1
        camera = fit._camera(
            target=(30.0 + 0.4 * k, 34.0, 0.0), focal_px=1600
        )  # 1920 x 1080
        camera = camera @ np.diag(
            [1920 / w, 1080 / h, 1.0]
        )  # the same view in this frame
        return line_points(camera, (w, h), noise_px=0.6, rng=np.random.default_rng(k))


LINES = {"type": "test_pan_lines"}


def _pan_video(path, frames=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for _ in range(frames):
        writer.write(np.zeros((180, 320, 3), np.uint8))
    writer.release()
    return path


def _homographies(result):
    return [f.homography for f in result]


# --------------------------------------------------------------- run files
def test_a_lines_section_builds_the_line_model_and_the_line_estimator(tmp_path):
    pipeline = _load(
        tmp_path,
        {
            "detector": DETECTOR,
            "lines": LINES,
            "pitch": {"length": 120, "width": 80},
            "homography": {"min_prob": 0.4, "max_age_s": 0.2, "fit": {"gate_px": 6.0}},
        },
    ).build_pipeline()
    assert isinstance(pipeline.line_model, PanLines) and pipeline.keypoint_model is None
    estimator = pipeline.homography
    assert isinstance(estimator, LineHomographyEstimator)
    assert estimator.pitch == SoccerPitch(120, 80) == pipeline.pitch
    assert estimator.min_prob == 0.4 and estimator.max_age_s == 0.2
    assert estimator.fit_config == FitConfig(gate_px=6.0)


def test_without_lines_the_homography_section_builds_the_keypoint_estimator(tmp_path):
    for lines in ({}, {"lines": None}):
        pipeline = _load(
            tmp_path, {"detector": DETECTOR, "homography": {"max_age": 5}} | lines
        ).build_pipeline()
        assert type(pipeline.homography) is HomographyEstimator
        assert pipeline.homography.max_age == 5 and pipeline.line_model is None


def test_an_empty_homography_section_builds_the_bare_line_estimator(tmp_path):
    built = _load(
        tmp_path, {"detector": DETECTOR, "lines": LINES, "homography": {}}
    ).build_pipeline()
    bare = LineHomographyEstimator()
    assert repr(built.homography) == repr(bare)


@pytest.mark.parametrize(
    ("homography", "bad", "valid"),
    [({"max_age": 5}, "max_age", "max_age_s"), ({"bogus": 1}, "bogus", "temporal_filter")],
)  # fmt: skip
def test_with_lines_the_homography_keys_are_the_line_estimator_s(
    tmp_path, homography, bad, valid
):
    with pytest.raises(ValueError, match="valid keys") as error:
        _load(
            tmp_path, {"detector": DETECTOR, "lines": LINES, "homography": homography}
        )
    message = str(error.value)
    assert f"Unknown key {bad!r} in section 'homography'" in message
    assert valid in message
    # and the keypoint estimator's keys without lines
    with pytest.raises(ValueError, match="Unknown key 'max_age_s'"):
        _load(tmp_path, {"detector": DETECTOR, "homography": {"max_age_s": 1.0}})


def test_unknown_keys_of_the_lines_section_fail_at_load(tmp_path):
    with pytest.raises(ValueError, match="Unknown key 'bogus' in section 'lines'"):
        _load(tmp_path, {"detector": DETECTOR, "lines": DETECTOR | {"bogus": 1}})
    with pytest.raises(ValueError, match="Section 'lines' needs a 'type' key"):
        _load(tmp_path, {"detector": DETECTOR, "lines": None}, ["lines.conf=0.3"])


def test_keypoints_and_lines_together_fail_at_load(tmp_path):
    with pytest.raises(ValueError, match="'keypoints' and 'lines' are two sources"):
        _load(tmp_path, {"detector": DETECTOR, "keypoints": DETECTOR, "lines": LINES})
    pipeline = _load(
        tmp_path, {"detector": DETECTOR, "keypoints": None, "lines": LINES}
    ).build_pipeline()
    assert isinstance(pipeline.line_model, PanLines)


def test_the_lines_weights_resolve_against_the_run_file(tmp_path):
    path = _write(
        tmp_path / "configs" / "run.yaml",
        {"detector": DETECTOR, "lines": LINES | {"weights": "../models/lines.pt"}},
    )
    sections = run_file.load(path).sections
    assert sections["lines"]["weights"] == str(tmp_path / "models" / "lines.pt")


# ----------------------------------------- the filter's settings reach the post-process
def test_the_run_file_s_filter_settings_steer_smooth_line_homographies(tmp_path):
    settings = {"accel_m": 0.6, "meas_m": 0.3, "gate_chi2": 25.0}
    sections = {
        "detector": DETECTOR,
        "lines": LINES,
        "homography": {"temporal_filter": settings},
    }
    video = _pan_video(tmp_path / "pan.mp4")
    raw = _load(tmp_path, sections).run(
        video, tmp_path / "raw", render=False, progress=False
    )
    assert (
        raw.temporal_filter == FilterConfig(**settings) and raw.temporal_filter_recorded
    )
    smoothed = _load(
        tmp_path, sections | {"postprocess": [{"type": "smooth_line_homographies"}]}
    ).run(video, tmp_path / "smoothed", render=False, progress=False)
    explicit = smooth_line_homographies(raw, **settings)
    for got, want in zip(_homographies(smoothed), _homographies(explicit), strict=True):
        np.testing.assert_array_equal(got, want)
    default = smooth_line_homographies(
        raw, **asdict(FilterConfig())
    )  # the package defaults instead
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(_homographies(smoothed), _homographies(default), strict=True)
    )


def test_a_run_without_a_filter_keeps_its_per_frame_homographies(tmp_path, caplog):
    sections = {
        "detector": DETECTOR,
        "lines": LINES,
        "homography": {"temporal_filter": None},
    }
    video = _pan_video(tmp_path / "pan.mp4")
    raw = _load(tmp_path, sections).run(
        video, tmp_path / "raw", render=False, progress=False
    )
    assert raw.temporal_filter is None and raw.temporal_filter_recorded
    with caplog.at_level(logging.WARNING):
        smoothed = _load(
            tmp_path, sections | {"postprocess": [{"type": "smooth_line_homographies"}]}
        ).run(video, tmp_path / "smoothed", render=False, progress=False)
    assert "without a temporal filter" in caplog.text
    for got, want in zip(_homographies(smoothed), _homographies(raw), strict=True):
        np.testing.assert_array_equal(got, want)


# ------------------------------------------------------------------ the CLI
def test_cli_run_with_a_lines_section_writes_the_run_folder(tmp_path):
    config = _write(
        tmp_path / "run.yaml",
        {
            "detector": DETECTOR,
            "lines": LINES,
            "postprocess": [{"type": "smooth_line_homographies"}],
        },
    )
    video = _pan_video(tmp_path / "pan.mp4", frames=6)
    out = tmp_path / "out"
    args = ["run", str(config), "--video", str(video), "--output-dir", str(out)]
    assert main([*args, "--set", "render.radar=null"]) == 0
    result = PipelineResult.load(out / "result.pkl")
    assert len(result) == 6 and result.temporal_filter == FilterConfig()
    assert [p.name for p in result.postprocesses] == ["smooth_line_homographies"]
    assert all(f.homography is not None and f.homography_fit.fresh for f in result)
    assert (out / "annotated.mp4").is_file()


def test_cli_info_lists_the_lines_backend(capsys):
    assert main(["info"]) == 0
    [models] = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("models:")
    ]
    assert "lines" in models.split(":")[1].replace(",", " ").split()
