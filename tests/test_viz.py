import ast
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pytest
import supervision as sv
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

import tactifoot_vision.viz as viz
from tactifoot_vision.data.annotations import Annotations, Task
from tactifoot_vision.data.dataset import Dataset, Sample
from tactifoot_vision.evaluation.metrics import DetectionMetrics, KeypointMetrics
from tactifoot_vision.models.base import TrainResult
from tactifoot_vision.pipeline.result import FrameResult, ObjectMasks, PipelineResult
from tactifoot_vision.pitch import SoccerPitch, frame_to_pitch, pitch_to_frame

matplotlib.use("Agg")

W, H = 320, 240
FPS = 10.0
PITCH = SoccerPitch()
# frame -> pitch: a camera looking at the middle of the pitch
HOMOGRAPHY = cv2.getPerspectiveTransform(
    np.float32([[40, 60], [280, 60], [310, 220], [10, 220]]),
    np.float32([[30, 10], [75, 10], [75, 58], [30, 58]]),
)


def _frame_result(index: int) -> FrameResult:
    shift = index * 2.0
    boxes = np.array(
        [
            [60 + shift, 100, 75 + shift, 140],
            [150, 90 + shift, 165, 130 + shift],
            [220 - shift, 120, 235 - shift, 160],
            [100, 150, 115, 190],
        ],
        dtype=np.float32,
    )
    people = sv.Detections(
        xyxy=boxes,
        confidence=np.full(4, 0.9, dtype=np.float32),
        class_id=np.array([2, 2, 1, 3]),
        tracker_id=np.array([1, 2, 3, 4]),
    )
    people.data["class_name"] = np.array(["player", "player", "goalkeeper", "referee"])
    people.data["team_id"] = np.array([0, 1, 1, -1])
    people.data["pitch_xy"] = frame_to_pitch(
        people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), HOMOGRAPHY
    )
    ball = sv.Detections(
        xyxy=np.array([[180 + shift, 170, 186 + shift, 176]], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        class_id=np.array([0]),
    )
    ball.data["class_name"] = np.array(["ball"])
    ball.data["pitch_xy"] = frame_to_pitch(
        ball.get_anchors_coordinates(sv.Position.CENTER), HOMOGRAPHY
    )
    confidence = np.where(np.arange(32) % 3 == 0, 0.9, 0.1).astype(np.float32)
    keypoints = sv.KeyPoints(
        xy=pitch_to_frame(PITCH.vertices, HOMOGRAPHY)[None].astype(np.float32),
        confidence=confidence[None],
    )
    return FrameResult(index, index / FPS, people, ball, keypoints, HOMOGRAPHY)


@pytest.fixture
def result() -> PipelineResult:
    frames = [_frame_result(i) for i in range(0, 10, 2)]  # stride 2
    return PipelineResult(
        frames, FPS, (W, H), ["ball", "goalkeeper", "player", "referee"]
    )


@pytest.fixture
def empty_result() -> PipelineResult:
    return PipelineResult([], FPS, (W, H), [])


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(10):
        frame = np.full((H, W, 3), (40, 120 + i, 40), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _frame() -> np.ndarray:
    return np.full((H, W, 3), (40, 120, 40), dtype=np.uint8)


def _read_all(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


# ------------------------------------------------------------------ annotate
@pytest.mark.parametrize("style", ["standard", "video_game"])
def test_annotate_draws_on_a_copy(result: PipelineResult, style: str) -> None:
    frame = _frame()
    image = viz.FrameAnnotator(style=style).annotate(frame, result[0])
    assert image.shape == frame.shape and image.dtype == np.uint8
    assert (image != frame).any()
    assert (frame == _frame()).all()  # input untouched


def test_annotate_uses_team_colours(result: PipelineResult) -> None:
    annotator = viz.FrameAnnotator(
        team_colors=("#0000FF", "#FF0000"), draw_labels=False, draw_keypoints=False,
        draw_pitch_lines=False,
    )  # fmt: skip
    image = annotator.annotate(_frame(), result[0])
    x1, y1, x2, _ = result[0].detections.xyxy[0].astype(int)  # team 0 box edge
    assert tuple(image[y1, (x1 + x2) // 2]) == (255, 0, 0)  # BGR blue


def test_annotate_toggles(result: PipelineResult) -> None:
    frame = _frame()
    nothing = viz.FrameAnnotator(
        draw_boxes=False,
        draw_labels=False,
        draw_keypoints=False,
        draw_pitch_lines=False,
    ).annotate(frame, result[0])
    assert (nothing == frame).all()
    lines_only = viz.FrameAnnotator(
        draw_boxes=False, draw_labels=False, draw_keypoints=False
    ).annotate(frame, result[0])
    assert (lines_only != frame).any()


def _with_masks(frame_result: FrameResult) -> FrameResult:
    """The frame with a mask over the top half of every person's box."""
    dense = np.zeros((len(frame_result.detections), H, W), dtype=bool)
    for mask, (x1, y1, x2, y2) in zip(
        dense, frame_result.detections.xyxy.astype(int), strict=True
    ):
        mask[y1 : (y1 + y2) // 2, x1:x2] = True
    frame_result.masks = ObjectMasks.from_dense(dense)
    return frame_result


def test_annotate_draws_masks_at_their_frame_position() -> None:
    frame_result = _with_masks(_frame_result(0))
    annotator = viz.FrameAnnotator(
        draw_masks=True, draw_boxes=False, draw_labels=False,
        draw_keypoints=False, draw_pitch_lines=False,
    )  # fmt: skip
    image = annotator.annotate(_frame(), frame_result)
    x1, y1, x2, y2 = frame_result.detections.xyxy[0].astype(int)
    assert (image[y1 + 2, x1 + 2] != _frame()[y1 + 2, x1 + 2]).any()  # masked
    assert (image[y2 - 2, x1 + 2] == _frame()[y2 - 2, x1 + 2]).all()  # box, no mask
    team_0 = np.array(viz.radar.as_color(viz.radar.TEAM_COLORS[0]).as_bgr())
    drawn = image[y1 + 2, x1 + 2].astype(int) - _frame()[y1 + 2, x1 + 2]
    assert np.sign(drawn).tolist() == np.sign(team_0 - _frame()[0, 0]).tolist()


def test_annotate_warns_once_when_masks_are_missing(caplog) -> None:
    annotator = viz.FrameAnnotator(draw_masks=True)
    with caplog.at_level("WARNING", logger="tactifoot_vision"):
        for i in range(3):
            annotator.annotate(_frame(), _frame_result(i))
    assert sum("no masks" in r.message for r in caplog.records) == 1


def test_render_video_draws_masks(
    result: PipelineResult, video: Path, tmp_path: Path
) -> None:
    for frame_result in result:
        _with_masks(frame_result)
    plain = dict(radar=False, progress=False)
    without, _ = _read_all(viz.render_video(result, video, tmp_path / "a.mp4", **plain))
    with_masks, _ = _read_all(
        viz.render_video(
            result, video, tmp_path / "b.mp4",
            annotator=viz.FrameAnnotator(draw_masks=True), **plain,
        )
    )  # fmt: skip
    x1, y1, x2, _ = result[0].detections.xyxy[0].astype(int)
    patch = np.s_[y1 + 3 : y1 + 8, x1 + 3 : x2 - 3]
    difference = np.abs(with_masks[0][patch].astype(int) - without[0][patch]).mean()
    assert difference > 20


def test_annotate_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="style"):
        viz.FrameAnnotator(style="cartoon")


def test_draw_annotations_hides_unlabelled_keypoints() -> None:
    keypoints = np.zeros((1, 3, 3), dtype=np.float32)
    keypoints[0, 0] = (50, 50, 2)
    keypoints[0, 1] = (200, 200, 0)  # not labelled -> not drawn
    annotations = Annotations(
        boxes=[[20, 20, 100, 100]], class_ids=[0], keypoints=keypoints
    )
    frame = _frame()
    image = viz.draw_annotations(frame, annotations, ["pitch"])
    assert image.shape == frame.shape and (image != frame).any()
    assert (image[200, 200] == frame[200, 200]).all()
    assert (image[50, 50] != frame[50, 50]).any()


# --------------------------------------------------------------------- radar
def test_radar_draws_players_and_ball(result: PipelineResult) -> None:
    radar = viz.PitchRadar(width_px=400, padding_px=20)
    empty = radar.draw()
    assert empty.shape == (radar.height_px, 400, 3) and empty.dtype == np.uint8
    image = radar.draw(result[0])
    assert (image != empty).any()
    x, y = radar.to_pixels(result[0].pitch_xy[0])[0].round().astype(int)
    assert tuple(image[y, x]) == sv.Color.from_hex("#00BFFF").as_bgr()


def test_radar_draw_points_and_nan() -> None:
    radar = viz.PitchRadar(draw_ids=True)
    xy = np.array([[10.0, 10.0], [np.nan, np.nan]])
    image = radar.draw_points(xy, colors=["#FF0000", "#00FF00"])
    x, y = radar.to_pixels(xy[0])[0].round().astype(int)
    assert tuple(image[y, x]) == (0, 0, 255)
    assert (radar.draw_points(xy, colors="#FF0000") == image).all()
    with pytest.raises(ValueError, match="colours"):
        radar.draw_points(xy, colors=["#FF0000"])


@pytest.mark.parametrize(
    ("position", "corner"),
    [("bottom-right", (-1, -1)), ("top-left", (0, 0)), ("top-right", (0, -1))],
)
def test_overlay_placement(position: str, corner: tuple[int, int]) -> None:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    patch = np.full((50, 100, 3), 255, dtype=np.uint8)
    out = viz.overlay(
        frame, patch, position=position, width_fraction=0.25, alpha=1.0, padding=0
    )
    assert out.shape == frame.shape and (frame == 0).all()
    changed = np.argwhere(out.any(axis=2))
    rows, cols = changed[:, 0], changed[:, 1]
    assert cols.max() - cols.min() + 1 == 100 and rows.max() - rows.min() + 1 == 50
    assert (rows.min() == 0) if corner[0] == 0 else (rows.max() == 199)
    assert (cols.min() == 0) if corner[1] == 0 else (cols.max() == 399)


def test_overlay_blends_and_validates() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    patch = np.full((10, 10, 3), 200, dtype=np.uint8)
    out = viz.overlay(frame, patch, position="center", width_fraction=0.5, alpha=0.5)
    assert out[50, 50, 0] == 100
    with pytest.raises(ValueError, match="position"):
        viz.overlay(frame, patch, position="middle-left")
    with pytest.raises(ValueError, match="alpha"):
        viz.overlay(frame, patch, alpha=2)


# --------------------------------------------------------------------- video
def test_render_video_writes_processed_frames(
    result: PipelineResult, video: Path, tmp_path: Path
) -> None:
    output = viz.render_video(
        result, video, tmp_path / "out" / "annotated.mp4", progress=False
    )
    assert output == tmp_path / "out" / "annotated.mp4"
    frames, fps = _read_all(output)
    assert len(frames) == len(result) == 5
    assert frames[0].shape == (H, W, 3)
    assert fps == pytest.approx(FPS / 2)


def test_render_video_without_radar_and_custom_fps(
    result: PipelineResult, video: Path, tmp_path: Path
) -> None:
    output = viz.render_video(
        result, video, tmp_path / "plain.mp4", radar=False, fps=3, progress=False,
        annotator=viz.FrameAnnotator(style="video_game"),
    )  # fmt: skip
    frames, fps = _read_all(output)
    assert len(frames) == 5 and fps == pytest.approx(3)


def test_render_video_rejects_empty_result(
    empty_result: PipelineResult, video: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no frames"):
        viz.render_video(empty_result, video, tmp_path / "x.mp4")


def test_render_video_seeks_to_the_first_frame_of_the_result(tmp_path: Path) -> None:
    source = tmp_path / "steps.mp4"  # frame i has green level 20 * i
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(10):
        writer.write(np.full((H, W, 3), (0, 20 * i, 0), dtype=np.uint8))
    writer.release()
    frames = [
        FrameResult(i, i / FPS, sv.Detections.empty(), sv.Detections.empty())
        for i in (3, 5, 7)
    ]
    partial = PipelineResult(frames, FPS, (W, H), [])
    output = viz.render_video(
        partial, source, tmp_path / "part.mp4", radar=False, progress=False
    )
    written, fps = _read_all(output)
    assert fps == pytest.approx(FPS / 2)
    assert [round(frame[H // 2, W // 2, 1] / 20) for frame in written] == [3, 5, 7]


def test_render_video_checks_the_source_and_the_pitch(
    result: PipelineResult, video: Path, tmp_path: Path
) -> None:
    wrong_size = PipelineResult(result.frames, FPS, (W * 2, H), result.class_names)
    with pytest.raises(ValueError, match="640x240"):
        viz.render_video(wrong_size, video, tmp_path / "a.mp4", progress=False)
    with pytest.raises(ValueError, match="pitch=result.pitch"):
        viz.render_video(
            result, video, tmp_path / "b.mp4", progress=False,
            radar=viz.PitchRadar(pitch=SoccerPitch(120, 80)),
        )  # fmt: skip
    result.video_path = video
    assert viz.render_video(result, None, tmp_path / "c.mp4", progress=False).is_file()


# --------------------------------------------------------------------- plots
def _dataset(tmp_path: Path) -> Dataset:
    samples = []
    for i in range(3):
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), _frame())
        annotations = Annotations(
            boxes=[[10, 10, 60, 90], [100, 50, 140, 120]], class_ids=[0, 1]
        )
        samples.append(Sample(path, W, H, annotations))
    return Dataset(Task.DETECT, ["player", "ball"], {"train": samples})


def _detection_metrics(scale: float = 1.0) -> DetectionMetrics:
    per_class = pd.DataFrame(
        {
            "map50_95": [0.3 * scale, 0.6],
            "map50": [0.5, 0.9],
            "map75": [0.3, 0.7],
            "instances": [10, 50],
        },
        index=["ball", "player"],
    )
    return DetectionMetrics(0.45 * scale, 0.7, 0.5, per_class, 20)


def _keypoint_metrics() -> KeypointMetrics:
    per_keypoint = pd.DataFrame(
        {"mean_error_px": np.linspace(2, 9, 32), "pck": 0.9, "count": [0] + [5] * 31}
    )
    return KeypointMetrics(5.5, 0.9, 0.05, 1.0, per_keypoint, 10)


def test_show_single_and_grid() -> None:
    assert isinstance(viz.show(_frame()), Figure)
    fig = viz.show(
        [_frame(), _frame()[..., 0], _frame()], titles=["a", "b", "c"], cols=2
    )
    assert isinstance(fig, Figure) and len([a for a in fig.axes if a.images]) == 3
    with pytest.raises(ValueError, match="titles"):
        viz.show([_frame()], titles=["a", "b"])


def test_show_samples_and_augmentations(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    assert isinstance(viz.show_samples(dataset, n=2), Figure)

    def flip(image, annotations, rng=None):
        assert isinstance(rng, np.random.Generator)
        return cv2.flip(image, 1), annotations

    fig = viz.show_augmentations(dataset, flip, n=2)
    assert isinstance(fig, Figure) and len([a for a in fig.axes if a.images]) == 4
    with pytest.raises(ValueError, match="empty"):
        viz.show_samples(dataset, split="valid")


def test_plot_training_ultralytics_and_rfdetr(tmp_path: Path) -> None:
    epochs = np.arange(1, 6)
    ultralytics = pd.DataFrame({
        "epoch": epochs, "train/box_loss": 1 / epochs, "val/box_loss": 1.2 / epochs,
        "train/cls_loss": 2 / epochs, "metrics/mAP50(B)": epochs / 10,
        "metrics/mAP50-95(B)": epochs / 20, "lr/pg0": 0.01,
    })  # fmt: skip
    run = TrainResult("yolo", tmp_path / "best.pt", tmp_path, ultralytics, {})
    fig = viz.plot_training(run)
    assert isinstance(fig, Figure)
    assert {a.get_title(loc="left") for a in fig.axes if a.get_visible()} == {
        "box loss", "cls loss", "box metrics",
    }  # fmt: skip
    rfdetr = pd.DataFrame({
        "epoch": epochs - 1, "train_loss": 5 / epochs, "test_loss": 6 / epochs,
        "train_loss_ce": 1 / epochs, "train_loss_ce_0": 1.0, "train_class_error": 3.0,
        "test_coco_eval_bbox": [[e / 10, e / 8, e / 12] + [0.0] * 9 for e in epochs],
    })  # fmt: skip
    fig = viz.plot_training(rfdetr)
    titles = {a.get_title(loc="left") for a in fig.axes if a.get_visible()}
    assert titles == {"total loss", "ce loss", "box metrics"}
    assert isinstance(viz.plot_training(pd.DataFrame({"epoch": [1]})), Figure)


def test_pitch_plots(result: PipelineResult, empty_result: PipelineResult) -> None:
    assert isinstance(viz.draw_pitch(), Figure)
    for res in (result, empty_result):
        assert isinstance(viz.plot_heatmap(res), Figure)
        assert isinstance(viz.plot_heatmap(res, team=1), Figure)
        assert isinstance(viz.plot_tracks(res), Figure)
    assert isinstance(viz.plot_tracks(result, track_ids=[1, 99], max_tracks=1), Figure)
    assert isinstance(viz.draw_pitch(pitch=SoccerPitch(120, 80)), Figure)


def test_plot_metrics_variants() -> None:
    assert isinstance(viz.plot_metrics(_detection_metrics()), Figure)
    compare = {"YOLO": _detection_metrics(), "RF-DETR": _detection_metrics(1.1)}
    assert isinstance(viz.plot_metrics(compare, metric="map50"), Figure)
    assert isinstance(viz.plot_metrics(_keypoint_metrics()), Figure)
    assert isinstance(
        viz.plot_metrics({"a": _keypoint_metrics(), "b": _keypoint_metrics()}), Figure
    )
    with pytest.raises(ValueError, match="metric"):
        viz.plot_metrics(_detection_metrics(), metric="f1")
    with pytest.raises(TypeError):
        viz.plot_metrics({"a": _detection_metrics(), "b": _keypoint_metrics()})


def test_plot_distance_histogram() -> None:
    assert isinstance(
        viz.plot_distance_histogram([0.5, 1.0, np.nan, 2.5, np.inf]), Figure
    )
    assert isinstance(viz.plot_distance_histogram(pd.Series([], dtype=float)), Figure)


def test_viz_modules_do_not_import_matplotlib_at_top_level() -> None:
    package = Path(viz.__file__).parent
    for path in package.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.startswith("matplotlib") for n in names), path.name


def test_notebook_figures_survive_a_backend_switch():
    # Ultralytics training switches matplotlib to Agg and back, which drops the
    # inline formatters; figures created afterwards must still render as images.
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    cells = [
        "import matplotlib.pyplot as plt\nimport tactifoot_vision as tv\n"
        "backend = plt.get_backend()\n"
        "plt.switch_backend('Agg')\nplt.switch_backend(backend)",
        "tv.viz.draw_pitch()",
    ]
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(c) for c in cells]
    )
    nbclient.NotebookClient(notebook, kernel_name="python3", timeout=120).execute()
    outputs = notebook.cells[1].outputs
    assert any("image/png" in output.get("data", {}) for output in outputs)


def test_render_video_passes_overlay_padding(
    result: PipelineResult, video: Path, tmp_path: Path, monkeypatch
):
    seen = []
    real_overlay = viz.video.overlay

    def spy(*args, **kwargs):
        seen.append(kwargs["padding"])
        return real_overlay(*args, **kwargs)

    monkeypatch.setattr(viz.video, "overlay", spy)
    viz.render_video(
        result, video, tmp_path / "a.mp4", overlay_padding=3, progress=False
    )
    assert seen and set(seen) == {3}


def _empty_frames(*indices: int) -> PipelineResult:
    frames = [
        FrameResult(i, i / FPS, sv.Detections.empty(), sv.Detections.empty())
        for i in indices
    ]
    return PipelineResult(frames, FPS, (W, H), [])


def test_render_video_fails_when_no_frame_was_written(video: Path, tmp_path: Path):
    beyond = _empty_frames(50)  # the source has 10 frames
    with pytest.raises(RuntimeError, match="no frame"):
        viz.render_video(beyond, video, tmp_path / "a.mp4", progress=False)


def test_render_video_fails_on_missing_frames(video: Path, tmp_path: Path):
    sparse = _empty_frames(0, 4, 20)
    with pytest.raises(RuntimeError, match=r"clip\.mp4.*frame 20"):
        viz.render_video(sparse, video, tmp_path / "a.mp4", progress=False)


class _RotatedReader:
    """A reader whose header says W x H but whose frames come out H x W."""

    def __init__(self, path) -> None:
        self.path, self.fps, self.size = Path(path), FPS, (W, H)

    def frames(self, start=0, end=None, stride=1):
        for index in range(start, end, stride):
            yield index, np.zeros((W, H, 3), np.uint8)


def test_render_video_checks_the_decoded_frame_size(
    result: PipelineResult, video: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(viz.video, "VideoReader", _RotatedReader)
    with pytest.raises(ValueError, match="rotation"):
        viz.render_video(result, video, tmp_path / "a.mp4", progress=False)


def test_heatmap_without_smoothing(result: PipelineResult) -> None:
    assert isinstance(viz.plot_heatmap(result, smoothing=0), Figure)
    with pytest.raises(ValueError, match="smoothing"):
        viz.plot_heatmap(result, smoothing=-1)


def test_colours_given_as_lists(result: PipelineResult) -> None:
    # YAML gives BGR colours as lists.
    radar = viz.PitchRadar(
        team_colors=[[255, 0, 0], [0, 0, 255]], ball_color=[0, 255, 255]
    )
    xy = np.array([[10.0, 10.0], [50.0, 30.0]])
    image = radar.draw_points(xy, colors=[0, 0, 255])
    x, y = radar.to_pixels(xy[1])[0].round().astype(int)
    assert tuple(image[y, x]) == (0, 0, 255)
    per_point = radar.draw_points(xy, colors=[[0, 0, 255], [0, 255, 0]])
    assert tuple(per_point[y, x]) == (0, 255, 0)
    x, y = radar.to_pixels(result[0].pitch_xy[0])[0].round().astype(int)
    assert tuple(radar.draw(result[0])[y, x]) == (255, 0, 0)
    annotator = viz.FrameAnnotator(team_colors=[[255, 0, 0], [0, 0, 255]])
    assert annotator.annotate(_frame(), result[0]).shape == (H, W, 3)


def test_annotate_skips_non_finite_keypoints(result: PipelineResult) -> None:
    frame_result = result[0]
    xy = frame_result.keypoints.xy.copy()
    xy[0, 0] = np.nan
    xy[0, 3] = np.inf
    frame_result.keypoints = sv.KeyPoints(
        xy=xy, confidence=np.ones((1, 32), np.float32)
    )
    image = viz.FrameAnnotator().annotate(_frame(), frame_result)
    assert image.shape == (H, W, 3)


def test_plot_training_rejects_a_non_numeric_epoch_column() -> None:
    history = pd.DataFrame({"epoch": ["a", "b"], "train/box_loss": [1.0, 0.5]})
    with pytest.raises(ValueError, match="epoch"):
        viz.plot_training(history)


def test_masks_reaching_past_the_frame_are_clipped():
    masks = ObjectMasks(origins=np.array([[75, 55]]), crops=[np.ones((10, 10), bool)])
    dense = masks.to_dense(80, 60)
    assert dense.shape == (1, 60, 80) and dense.sum() == 5 * 5
    frame_result = _frame_result(0)
    frame_result.masks = ObjectMasks(
        origins=np.array([[75, 55]] * len(frame_result.detections)),
        crops=[np.ones((10, 10), bool)] * len(frame_result.detections),
    )
    image = np.zeros((60, 80, 3), np.uint8)
    out = viz.FrameAnnotator(draw_masks=True).annotate(image, frame_result)
    assert out.shape == image.shape


def test_a_singular_homography_draws_no_pitch_lines():
    frame_result = _frame_result(0)
    frame_result.homography = np.zeros((3, 3))
    image = _frame()
    out = viz.FrameAnnotator(draw_boxes=False, draw_labels=False).annotate(
        image, frame_result
    )
    assert out.shape == image.shape


# ------------------------------------------------------- review round 2
def test_render_video_refuses_to_overwrite_its_source(
    result: PipelineResult, video: Path, tmp_path: Path, monkeypatch
):
    before = video.read_bytes()
    with pytest.raises(ValueError, match="source"):
        viz.render_video(result, video, video, progress=False)
    monkeypatch.chdir(video.parent)
    (tmp_path / "link.mp4").symlink_to(video)
    with pytest.raises(ValueError, match="source"):
        viz.render_video(
            result, Path("clip.mp4"), tmp_path / "link.mp4", progress=False
        )
    assert video.read_bytes() == before


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"position": "bottom-middle"}, "position"),
        ({"width_fraction": 0}, "width_fraction"),
        ({"alpha": -0.1}, "alpha"),
        ({"padding": -1}, "padding"),
    ],
)
def test_overlay_options_have_a_validator(options, match):
    with pytest.raises(ValueError, match=match):
        viz.check_overlay(**options)
    viz.check_overlay()  # nothing given: nothing to check


def test_render_video_checks_overlay_options_before_writing(
    result: PipelineResult, video: Path, tmp_path: Path
):
    with pytest.raises(ValueError, match="position"):
        viz.render_video(
            result, video, tmp_path / "a.mp4", overlay_position="bottom-middle",
            radar=False, progress=False,
        )  # fmt: skip
    assert not (tmp_path / "a.mp4").exists()
    with pytest.raises(ValueError, match="fps"):
        viz.render_video(result, video, tmp_path / "a.mp4", fps=0, progress=False)
    assert not (tmp_path / "a.mp4").exists()


class _FailingAnnotator(viz.FrameAnnotator):
    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.calls, self.fail_at = 0, fail_at

    def annotate(self, frame, frame_result):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("boom")
        return super().annotate(frame, frame_result)


def test_render_video_deletes_a_partial_output(
    result: PipelineResult, video: Path, tmp_path: Path
):
    output = tmp_path / "annotated.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        viz.render_video(
            result, video, output, annotator=_FailingAnnotator(3), progress=False
        )
    assert not output.exists()
    with pytest.raises(RuntimeError, match="frame 20"):
        viz.render_video(_empty_frames(0, 4, 20), video, output, progress=False)
    assert not output.exists()


def test_annotator_draws_masks_whose_origin_lies_outside_the_frame():
    frame_result = _frame_result(0)
    n = len(frame_result.detections)
    frame_result.masks = ObjectMasks(
        origins=np.array([[-5, -4]] * n), crops=[np.ones((10, 10), bool)] * n
    )
    annotator = viz.FrameAnnotator(
        draw_masks=True, draw_boxes=False, draw_labels=False,
        draw_keypoints=False, draw_pitch_lines=False,
    )  # fmt: skip
    out = annotator.annotate(np.zeros((60, 80, 3), np.uint8), frame_result)
    tinted = np.argwhere(out.any(axis=2))
    assert tinted[:, 0].max() == 5 and tinted[:, 1].max() == 4
    assert len(tinted) == 6 * 5


# ------------------------------------------------------- review round 3
def test_render_video_refuses_a_hard_link_to_its_source(
    result: PipelineResult, video: Path, tmp_path: Path
):
    alias = tmp_path / "alias.mp4"
    os.link(video, alias)  # another name for the same file: resolved paths differ
    before = video.read_bytes()
    with pytest.raises(ValueError, match="source"):
        viz.render_video(result, video, alias, progress=False)
    assert video.read_bytes() == before
    assert alias.samefile(video)


@pytest.mark.parametrize("size", ["player_radius", "ball_radius", "line_thickness"])
@pytest.mark.parametrize("value", [0, -1])
def test_radar_rejects_sizes_below_one_pixel(size: str, value: int) -> None:
    with pytest.raises(ValueError, match=size):
        viz.PitchRadar(**{size: value})


def test_radar_draw_points_checks_its_radius() -> None:
    radar = viz.PitchRadar(player_radius=5)
    xy = np.array([[10.0, 10.0]])
    for radius in (0, -1):
        with pytest.raises(ValueError, match="radius"):
            radar.draw_points(xy, radius=radius)
    assert (radar.draw_points(xy) == radar.draw_points(xy, radius=5)).all()
    assert (radar.draw_points(xy, radius=2) != radar.draw_points(xy)).any()


def _people(index: int, rows: list[tuple[int, str, int, list[float]]]) -> FrameResult:
    """A frame whose people are ``(track id, class name, team id, pitch xy)`` rows."""
    people = sv.Detections(
        xyxy=np.zeros((len(rows), 4), np.float32),
        confidence=np.full(len(rows), 0.9, np.float32),
        class_id=np.zeros(len(rows), int),
        tracker_id=np.array([track for track, *_ in rows]),
        data={
            "class_name": np.array([name for _, name, *_ in rows]),
            "team_id": np.array([team for _, _, team, _ in rows]),
            "pitch_xy": np.array([xy for *_, xy in rows], dtype=float),
        },
    )
    return FrameResult(index, index / FPS, people, sv.Detections.empty())


@pytest.fixture
def positions() -> PipelineResult:
    """Track 1 (team 0) at three positions, track 2 (team 1) at two, a referee."""
    nan = [np.nan, np.nan]
    frames = [
        _people(0, [(1, "player", 0, [10, 20]), (2, "player", 1, [50, 30]),
                    (3, "referee", -1, [80, 60])]),
        _people(1, [(1, "player", 0, [12, 21]), (2, "player", 1, nan),
                    (3, "referee", -1, [80, 60])]),
        _people(2, [(1, "player", 0, [14, 22]), (2, "player", 1, [49, 31])]),
    ]  # fmt: skip
    return PipelineResult(frames, FPS, (W, H), ["player", "referee"])


def test_plot_tracks_draws_each_track_at_its_pitch_positions(positions):
    lines = viz.plot_tracks(positions).axes[0].lines
    assert [line.get_xydata().tolist() for line in lines] == [
        [[10, 20], [12, 21], [14, 22]],
        [[50, 30], [49, 31]],
    ]
    colors = [to_rgb(line.get_color()) for line in lines]
    assert colors == [to_rgb("#00BFFF"), to_rgb("#FF1493")]  # teams 0 and 1
    (only,) = viz.plot_tracks(positions, track_ids=[2]).axes[0].lines
    assert only.get_xydata().tolist() == [[50, 30], [49, 31]]
    (longest,) = viz.plot_tracks(positions, max_tracks=1).axes[0].lines
    assert len(longest.get_xydata()) == 3


@pytest.mark.parametrize(
    ("team", "cells"),
    [
        (0, [[13, 6], [13, 8], [14, 9]]),
        (1, [[19, 33], [20, 32]]),
        (None, [[13, 6], [13, 8], [14, 9], [19, 33], [20, 32]]),  # no referee
    ],
)
def test_plot_heatmap_counts_only_the_selected_team(positions, team, cells):
    # 1.5 m cells on 105 x 68 m: 70 columns of 1.5 m, 45 rows of 68/45 m.
    ax = viz.plot_heatmap(positions, team=team, cell_m=1.5, smoothing=0).axes[0]
    hist = np.asarray(ax.images[0].get_array())
    assert hist.shape == (45, 70)
    assert np.argwhere(hist > 0).tolist() == cells
    assert (hist[hist > 0] == 1).all()
    assert ax.get_title(loc="left") == f"{len(cells)} positions in 3 frames"
