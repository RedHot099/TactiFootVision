import ast
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pytest
import supervision as sv
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
