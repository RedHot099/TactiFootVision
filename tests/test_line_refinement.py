"""Line refinement in HomographyEstimator, scored in metres on synthetic clips of exact calibration."""

import cv2
import numpy as np
import pytest
import supervision as sv

from tactifoot_vision.data import VideoReader, synthetic_pitch_clip
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.evaluation import evaluate_homography
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline import FrameResult, Pipeline, PipelineResult
from tactifoot_vision.pitch import (
    HomographyEstimator,
    LineFit,
    LineRefinerConfig,
    SoccerPitch,
    pitch_to_frame,
)

FRAME = (640, 360)


CORNERS = np.float32([[0, 360], [640, 360], [640, 0], [0, 0]])  # of the frame


def _view(corners_on_pitch, pitch):
    """Frame -> pitch homography putting the frame's ``CORNERS`` at these 105 x 68 m points."""
    scale = np.float32([pitch.length / 105, pitch.width / 68])
    return cv2.getPerspectiveTransform(CORNERS, np.float32(corners_on_pitch) * scale)


def _camera(pitch=None, pan=0.0):
    """A broadcast view of the right penalty area, panned ``pan`` metres."""
    corners = np.float32([[56, 64], [100, 64], [112, 2], [50, 2]]) + [pan, 0]
    return _view(corners, pitch or SoccerPitch())


def _close_up(pitch=None):
    """A frame showing only grass between the penalty area and the centre circle."""
    return _view([[24, 34], [34, 34], [33, 27], [25, 27]], pitch or SoccerPitch())


def _perturbed(homography, shift_px, seed):
    """``homography`` with the frame's corners moved by up to ``shift_px`` pixels."""
    rng = np.random.default_rng(seed)
    moved = CORNERS + rng.uniform(-shift_px, shift_px, CORNERS.shape).astype(np.float32)
    pitch_xy = cv2.perspectiveTransform(CORNERS[:, None], homography)
    return cv2.getPerspectiveTransform(moved, pitch_xy.reshape(-1, 2))


def _keypoints(pitch, homography):
    """The landmarks ``homography`` puts inside the frame, as a keypoint model would give them."""
    xy = pitch_to_frame(pitch.vertices, homography)
    shown = ((xy >= 0) & (xy <= FRAME)).all(axis=1)
    return sv.KeyPoints(
        xy=xy[None].astype(np.float32), confidence=shown[None].astype(np.float32)
    )


def _frames(clip):
    return [frame for _, frame in VideoReader(clip.video_path).frames()]


def _error_m(clip, homographies, pitch=None):
    empty = sv.Detections.empty()
    frames = [
        FrameResult(i, i / clip.fps, empty, empty, ball_candidates=empty, homography=h)
        for i, h in enumerate(homographies)
    ]
    result = PipelineResult(
        frames, clip.fps, (clip.width, clip.height), ["player"], pitch or SoccerPitch()
    )
    return evaluate_homography(result, clip).median_error_m


def _refine(pitch, starts, images, boxes=None, **options):
    estimator = HomographyEstimator(
        pitch, smoothing_window=1, refine_with_lines=True, **options
    )
    refined, fits = [], []
    for start, image in zip(starts, images, strict=True):
        refined.append(estimator.update(_keypoints(pitch, start), image, boxes))
        fits.append(estimator.line_fit)
    return refined, fits


@pytest.mark.parametrize("pitch", [SoccerPitch(), SoccerPitch(120, 80)])
def test_refinement_recovers_the_true_homography_from_perturbed_starts(tmp_path, pitch):
    cameras = [_camera(pitch, pan) for pan in (0, 0.5, 1, 1.5)]
    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", cameras, FRAME, pitch=pitch)
    starts = [_perturbed(h, 15, seed) for seed, h in enumerate(cameras)]

    refined, fits = _refine(pitch, starts, _frames(clip))

    assert _error_m(clip, starts, pitch) > 0.5
    assert _error_m(clip, refined, pitch) < 0.1
    for fit in fits:
        assert isinstance(fit, LineFit)
        assert fit.inlier_share > 0.9
        assert fit.median_px < 1
        assert fit.points >= 80


def test_refinement_ignores_white_kits_inside_the_given_boxes(tmp_path):
    pitch, camera = SoccerPitch(), _camera()
    clip = synthetic_pitch_clip(tmp_path / "kits.mp4", [camera], FRAME)
    (image,) = _frames(clip)
    # Two players hide the penalty area's side line, their white kits drawn
    # as a stripe 6 px beside it.
    boxes = np.array([[395, 60, 445, 150], [415, 170, 470, 285]], float)
    side = pitch_to_frame(pitch.vertices[[17, 20]], camera) + [6, 0]
    kits = np.full_like(image, (40, 130, 50))
    cv2.line(kits, *np.round(side).astype(int).tolist(), (235, 235, 235), 2)
    for x1, y1, x2, y2 in boxes.astype(int):
        image[y1:y2, x1:x2] = kits[y1:y2, x1:x2]
    start = _perturbed(camera, 10, seed=0)

    (with_boxes,), (fit,) = _refine(pitch, [start], [image], boxes)
    (without_boxes,), _ = _refine(pitch, [start], [image])

    assert fit is not None
    assert _error_m(clip, [with_boxes]) < 0.1
    assert _error_m(clip, [without_boxes]) > 0.15


def test_too_few_visible_lines_fall_back_to_the_keypoint_homography(tmp_path):
    pitch, camera = SoccerPitch(), _close_up()
    clip = synthetic_pitch_clip(tmp_path / "close.mp4", [camera], FRAME)
    start = _perturbed(camera, 5, seed=0)
    raw = HomographyEstimator(pitch, smoothing_window=1).update(
        _keypoints(pitch, start)
    )

    (refined,), (fit,) = _refine(pitch, [start], _frames(clip))

    assert fit is None
    np.testing.assert_array_equal(refined, raw)


def test_lines_hidden_from_view_fall_back_too(tmp_path):
    # The markings project into the frame, but the frame shows plain grass.
    pitch, camera = SoccerPitch(), _camera()
    grass = np.full((FRAME[1], FRAME[0], 3), (45, 135, 50), np.uint8)
    raw = HomographyEstimator(pitch, smoothing_window=1).update(
        _keypoints(pitch, camera)
    )

    (refined,), (fit,) = _refine(pitch, [camera], [grass])

    assert fit is None
    np.testing.assert_array_equal(refined, raw)


@pytest.mark.parametrize("off_m", [10, 30])
def test_a_start_too_far_off_falls_back_instead_of_degenerating(tmp_path, off_m):
    # From these starts the fit used to fold the frame beyond the horizon while
    # reporting most model points on a line (they had left the frame).
    pitch, camera = SoccerPitch(), _camera()
    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", [camera], FRAME)
    start = _camera(pitch, -off_m)
    raw = HomographyEstimator(pitch, smoothing_window=1).update(
        _keypoints(pitch, start)
    )

    (refined,), (fit,) = _refine(pitch, [start], _frames(clip))

    assert fit is None
    np.testing.assert_array_equal(refined, raw)
    assert np.isfinite(_error_m(clip, [refined]))


def test_min_line_inlier_share_sets_the_fallback(tmp_path):
    pitch, camera = SoccerPitch(), _camera()
    clip = synthetic_pitch_clip(
        tmp_path / "busy.mp4", [camera], FRAME, occluders=15, noise=6, seed=2
    )
    start = _perturbed(camera, 10, seed=3)
    _, (fit,) = _refine(pitch, [start], _frames(clip))
    assert 0.5 < fit.inlier_share < 1

    _, (strict,) = _refine(
        pitch, [start], _frames(clip), min_line_inlier_share=fit.inlier_share + 1e-3
    )
    _, (lenient,) = _refine(
        pitch, [start], _frames(clip), min_line_inlier_share=fit.inlier_share
    )

    assert strict is None
    assert lenient == fit


def test_line_options_reach_the_line_mask(tmp_path):
    pitch, camera = SoccerPitch(), _camera()
    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", [camera], FRAME)
    start = _perturbed(camera, 10, seed=0)

    # No pixel is unsaturated enough to be a line: nothing to fit to.
    _, (fit,) = _refine(
        pitch, [start], _frames(clip), line_options={"max_line_saturation": 1}
    )
    (refined,), (tuned,) = _refine(
        pitch, [start], _frames(clip), line_options=LineRefinerConfig(tophat_px=11)
    )

    assert fit is None
    assert tuned is not None and _error_m(clip, [refined]) < 0.1


def test_line_settings_are_read_only_and_follow_the_pitch(tmp_path):
    estimator = HomographyEstimator(refine_with_lines=True)
    for name in ("refine_with_lines", "min_line_points", "line_options"):
        with pytest.raises(AttributeError):
            setattr(estimator, name, None)
    assert estimator.line_options == LineRefinerConfig()
    estimator.update(None, np.zeros((FRAME[1], FRAME[0], 3), np.uint8))

    pitch = SoccerPitch(120, 80)
    estimator.pitch = pitch
    camera = _camera(pitch)
    clip = synthetic_pitch_clip(tmp_path / "big.mp4", [camera], FRAME, pitch=pitch)
    start = _perturbed(camera, 15, seed=0)
    refined = estimator.update(_keypoints(pitch, start), *_frames(clip))

    assert estimator.line_fit is not None
    assert _error_m(clip, [refined], pitch) < 0.1


def test_min_line_points_sets_the_fallback(tmp_path):
    pitch, camera = SoccerPitch(), _camera()
    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", [camera], FRAME)
    start = _perturbed(camera, 10, seed=1)

    _, (fit,) = _refine(pitch, [start], _frames(clip), min_line_points=100_000)

    assert fit is None


def test_refinement_starts_from_the_previous_refined_matrix(tmp_path):
    # After the first frame the keypoints put the camera 20 m off, too far for
    # a fit from them alone; the fit from the frame before holds the pan.
    pitch = SoccerPitch()
    pans = (0, 0.3, 0.6, 0.9)
    cameras = [_camera(pitch, pan) for pan in pans]
    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", cameras, FRAME)
    frames = _frames(clip)
    starts = [_perturbed(cameras[0], 15, seed=0)]
    starts += [_camera(pitch, pan - 20) for pan in pans[1:]]
    alone = [
        _refine(pitch, [s], [f])[0][0] for s, f in zip(starts, frames, strict=True)
    ]

    refined, fits = _refine(pitch, starts, frames)

    assert _error_m(clip, alone) > 5
    assert _error_m(clip, refined) < 0.1
    assert all(fit.inlier_share > 0.9 for fit in fits)


def test_refinement_is_off_by_default_and_needs_the_frame():
    pitch, camera = SoccerPitch(), _camera()
    keypoints = _keypoints(pitch, camera)
    image = np.zeros((FRAME[1], FRAME[0], 3), np.uint8)
    plain = HomographyEstimator(pitch)
    assert plain.refine_with_lines is False

    with_frame = plain.update(keypoints, image, np.zeros((0, 4)))

    np.testing.assert_array_equal(
        with_frame, HomographyEstimator(pitch).update(keypoints)
    )
    assert plain.line_fit is None
    with pytest.raises(ValueError, match="frame"):
        HomographyEstimator(pitch, refine_with_lines=True).update(keypoints)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_line_points": 3}, "min_line_points"),
        ({"line_inlier_px": 0}, "line_inlier_px"),
        ({"line_inlier_px": 6}, "line_inlier_px"),
        ({"min_line_inlier_share": 1.5}, "min_line_inlier_share"),
        ({"line_options": {"stages": []}}, "stages"),
        ({"line_options": {"grass_hsv_low": (30, 60)}}, "grass_hsv_low"),
        ({"line_options": {"tophat_px": 0}}, "tophat_px"),
        ({"line_options": {"good_inlier_share": 0}}, "good_inlier_share"),
    ],
)
def test_refinement_options_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HomographyEstimator(refine_with_lines=True, **kwargs)


class _OnePlayer(Model):
    """Detects one player standing on the penalty area's side line."""

    name = "one_player"
    task = Task.DETECT

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        return ["ball", "goalkeeper", "player"]

    def predict(self, image: np.ndarray) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([[420, 150, 450, 230]], np.float32),
            confidence=np.array([0.9], np.float32),
            class_id=np.array([2]),
            data={"class_name": np.array(["player"])},
        )


class _OffKeypoints(Model):
    """The landmarks of a homography 15 px off the clip's calibration."""

    name = "off_keypoints"
    task = Task.POSE

    def __init__(self, homography: np.ndarray) -> None:
        super().__init__(device="cpu")
        self.keypoints = _keypoints(SoccerPitch(), homography)

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        return ["pitch"]

    def predict(self, image: np.ndarray) -> sv.KeyPoints:
        return self.keypoints


class _BoxSpy(HomographyEstimator):
    def update(self, keypoints, frame=None, boxes=None):
        self.boxes = boxes
        return super().update(keypoints, frame, boxes)


def test_the_pipeline_refines_on_its_frames_and_keeps_each_fit(tmp_path):
    camera = _camera()
    clip = synthetic_pitch_clip(tmp_path / "static.mp4", [camera] * 3, FRAME)
    keypoints = _OffKeypoints(_perturbed(camera, 15, seed=0))

    def run(estimator):
        pipeline = Pipeline(_OnePlayer(), keypoints, tracker=None, homography=estimator)
        return pipeline, pipeline.run(clip.video_path, progress=False)

    _, raw = run(HomographyEstimator())
    pipeline, refined = run(_BoxSpy(refine_with_lines=True))

    assert evaluate_homography(raw, clip).median_error_m > 0.5
    assert evaluate_homography(refined, clip).median_error_m < 0.1
    assert all(frame.line_fit is None for frame in raw)
    assert all(frame.line_fit.inlier_share > 0.9 for frame in refined)
    np.testing.assert_array_equal(pipeline.homography.boxes, [[420, 150, 450, 230]])


def test_boxes_beyond_the_frame_hide_nothing_inside_it(tmp_path):
    pitch, camera = SoccerPitch(), _camera()
    clip = synthetic_pitch_clip(tmp_path / "pan.mp4", [camera], FRAME)
    start = _perturbed(camera, 15, seed=0)
    # A tracker can predict boxes that have left the frame.
    boxes = np.array([[-200, -300, -150, -100], [-90, 40, -60, 120]], float)

    (refined,), (fit,) = _refine(pitch, [start], _frames(clip), boxes)

    assert fit is not None
    assert _error_m(clip, [refined]) < 0.1
