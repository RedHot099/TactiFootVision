"""The pipeline with a line model (ADR 0005), beside the keypoint model it does not change."""

import logging
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

import tactifoot_vision as tv
from line_points import line_points
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.enums import PitchMarking
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline import (
    NO_TEAM,
    Pipeline,
    PipelineResult,
    smooth_homographies,
    smooth_line_homographies,
)
from tactifoot_vision.pitch import (
    HomographyEstimator,
    LineHomographyEstimator,
    LinePoints,
    frame_to_pitch,
)
from tactifoot_vision.pitch.temporal import FilterConfig
from tactifoot_vision.teams import TeamClassifier
from test_pipeline import (
    FRAME_TO_PITCH,
    HEIGHT,
    WIDTH,
    ColorDetector,
    KnownHomographyKeypoints,
    MeanColorEmbedder,
    _write_video,
)

ROOT = Path(__file__).resolve().parents[1]


class KnownHomographyLines(Model):
    """Finds the points of every marking the fixed homography shows, as a perfect line model would."""

    name = "known_lines"
    task = Task.LINES

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        return [m.value for m in PitchMarking]

    def predict(self, image: np.ndarray) -> LinePoints:
        return line_points(FRAME_TO_PITCH, (image.shape[1], image.shape[0]))


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    return _write_video(tmp_path_factory.mktemp("video") / "synthetic.avi")


def _pipeline(**kwargs) -> Pipeline:
    options = {
        "line_model": KnownHomographyLines(device="cpu"),
        "team_classifier": TeamClassifier(MeanColorEmbedder(), reducer=None),
    } | kwargs
    return Pipeline(ColorDetector(device="cpu"), **options)


@pytest.fixture(scope="module")
def result(video) -> PipelineResult:
    return _pipeline().run(video, progress=False)


CORNERS = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]


def test_pitch_positions_follow_the_line_homography(result):
    frame = result[2]
    assert frame.homography is not None
    np.testing.assert_allclose(
        frame_to_pitch(CORNERS, frame.homography),
        frame_to_pitch(CORNERS, FRAME_TO_PITCH),
        atol=1e-3,
    )
    anchors = frame.detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    np.testing.assert_allclose(
        frame.pitch_xy, frame_to_pitch(anchors, FRAME_TO_PITCH), atol=1e-3
    )
    ball_centre = frame.ball.get_anchors_coordinates(sv.Position.CENTER)
    np.testing.assert_allclose(
        frame.ball_xy, frame_to_pitch(ball_centre, FRAME_TO_PITCH), atol=1e-3
    )
    assert frame.line_points is not None and len(frame.line_points) > 100
    assert frame.homography_fit.fresh and frame.homography_fit.uncertainty_m < 0.5
    # the keypoint fields stay empty
    assert frame.keypoints is None and frame.line_fit is None


def test_the_result_records_the_filter_the_pass_ran(result, video):
    assert result.temporal_filter == FilterConfig() and result.temporal_filter_recorded
    unfiltered = _pipeline(
        homography=LineHomographyEstimator(temporal_filter=None), team_classifier=None
    ).run(video, end=3, progress=False)
    assert unfiltered.temporal_filter is None and unfiltered.temporal_filter_recorded


def test_teams_still_come_from_the_kit_vote(result):
    players = [f.team_ids for f in result]
    assert {tuple(t) for t in players} != {(NO_TEAM,) * len(players[0])}


def test_a_strided_run_steps_the_filter_in_physical_time(video):
    pipeline = _pipeline(team_classifier=None)
    result = pipeline.run(video, start=1, end=9, stride=2, progress=False)
    assert [f.index for f in result] == [1, 3, 5, 7]
    assert pipeline.homography._dt == 2.0  # the filter steps in physical time: 2 / 25 s
    assert pipeline.homography._max_age == 25  # 2 s at 12.5 processed frames per second


def test_smooth_line_homographies_runs_on_the_line_result(result):
    smoothed = smooth_line_homographies(result)
    assert [p.name for p in smoothed.postprocesses] == ["smooth_line_homographies"]
    for frame in smoothed:
        np.testing.assert_allclose(
            frame_to_pitch(CORNERS, frame.homography),
            frame_to_pitch(CORNERS, FRAME_TO_PITCH),
            atol=1e-3,
        )


def test_the_line_model_comes_with_the_line_estimator():
    detector, lines = ColorDetector(device="cpu"), KnownHomographyLines(device="cpu")
    assert isinstance(
        Pipeline(detector, line_model=lines).homography, LineHomographyEstimator
    )
    pitch = tv.pitch.SoccerPitch(120, 80)
    assert Pipeline(detector, line_model=lines, pitch=pitch).homography.pitch == pitch
    # without a line model the default estimator is today's
    assert type(Pipeline(detector).homography) is HomographyEstimator
    assert "line_model=" in repr(Pipeline(detector, line_model=lines))
    assert "line_model=" not in repr(Pipeline(detector))


def test_validation_of_the_second_source():
    detector = ColorDetector(device="cpu")
    lines = KnownHomographyLines(device="cpu")
    keypoints = KnownHomographyKeypoints(device="cpu")
    with pytest.raises(ValueError, match="line_model must be a lines model"):
        Pipeline(detector, line_model=keypoints)
    with pytest.raises(
        ValueError, match="keypoint_model and line_model are two sources"
    ):
        Pipeline(detector, keypoint_model=keypoints, line_model=lines)
    with pytest.raises(
        ValueError, match="fixed_homography and line_model are two sources"
    ):
        Pipeline(detector, line_model=lines, fixed_homography=np.eye(3))
    with pytest.raises(ValueError, match="needs a LineHomographyEstimator"):
        Pipeline(detector, line_model=lines, homography=HomographyEstimator())
    with pytest.raises(ValueError, match="needs a HomographyEstimator"):
        Pipeline(
            detector, keypoint_model=keypoints, homography=LineHomographyEstimator()
        )


# ---------------------------------------------- the keypoint path is untouched
@pytest.fixture(scope="module")
def keypoint_result(video) -> PipelineResult:
    return Pipeline(
        ColorDetector(device="cpu"),
        keypoint_model=KnownHomographyKeypoints(device="cpu"),
        team_classifier=TeamClassifier(MeanColorEmbedder(), reducer=None),
    ).run(video, progress=False)


def test_a_keypoint_run_sets_no_line_field(keypoint_result):
    assert not keypoint_result.temporal_filter_recorded
    assert keypoint_result.temporal_filter is None
    for frame in keypoint_result:
        assert frame.keypoints is not None and frame.keypoints.xy.shape == (1, 32, 2)
        assert frame.line_points is None and frame.homography_fit is None
        np.testing.assert_allclose(
            frame.homography, FRAME_TO_PITCH / FRAME_TO_PITCH[2, 2], atol=1e-5
        )


def test_smooth_homographies_on_a_keypoint_run_is_savitzky_golay(keypoint_result):
    from scipy.signal import savgol_filter

    smoothed = smooth_homographies(keypoint_result, window=0.2, order=1)
    assert [p.name for p in smoothed.postprocesses] == ["smooth_homographies"]
    # by hand: each matrix scaled to map the frame centre with w = 1, then a
    # centred Savitzky-Golay filter over 5 frames (0.2 s at 25 fps), entry by entry
    centre = np.array([WIDTH / 2, HEIGHT / 2, 1.0])
    scaled = np.stack(
        [f.homography / (f.homography @ centre)[2] for f in keypoint_result]
    )
    want = savgol_filter(scaled, 5, 1, axis=0, mode="interp")
    for frame, matrix in zip(smoothed, want, strict=True):
        np.testing.assert_allclose(frame.homography, matrix, atol=1e-12)


def test_smooth_line_homographies_leaves_a_keypoint_run_unchanged(
    keypoint_result, caplog
):
    with caplog.at_level(logging.WARNING):
        smoothed = smooth_line_homographies(keypoint_result)
    assert "no per-frame fits" in caplog.text
    assert [p.name for p in smoothed.postprocesses] == ["smooth_line_homographies"]
    for before, after in zip(keypoint_result, smoothed, strict=True):
        np.testing.assert_array_equal(after.homography, before.homography)
        np.testing.assert_array_equal(after.pitch_xy, before.pitch_xy)


# ----------------------------------------------------------------- real models
@pytest.mark.model
def test_real_video_pipeline_with_the_line_model(
    tmp_path, detector_weights, line_weights
):
    video = ROOT / "data" / "videos" / "broadcast_60s.mp4"
    if not video.is_file():
        pytest.skip("real video not available")
    if "yolo" not in tv.models.available_models():
        pytest.skip("yolo backend not available")
    pipeline = Pipeline(
        tv.load_model("yolo", detector_weights),
        line_model=tv.load_model("lines", line_weights),
        team_classifier=TeamClassifier("siglip"),
    )
    result = pipeline.run(video, end=50, progress=False)
    assert len(result) == 50
    table = result.to_dataframe()
    players = table[(table.object == "person") & (table.class_name == "player")]
    assert players.track_id.nunique() >= 15
    # both teams are present and most players have one
    assert set(players.team_id) >= {0, 1}
    assert (players.team_id != NO_TEAM).mean() > 0.9
    # the homography puts players on the pitch
    pitch = result.pitch
    on_pitch = players.pitch_x.between(-5, pitch.length + 5) & players.pitch_y.between(
        -5, pitch.width + 5
    )
    assert on_pitch.mean() > 0.9
    # each track keeps one team
    assert (players.groupby("track_id").team_id.nunique() == 1).all()
    assert result.save(tmp_path / "result.pkl").is_file()
