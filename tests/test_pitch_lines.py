"""Pitch line data and the ``lines`` backend on synthetic images: labels, targets, decoding, training, the CLI."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import tactifoot_vision as tv
from tactifoot_vision.cli import main
from tactifoot_vision.data.pitch_lines import (
    calibration_lines,
    gamestate_lines,
    load_pitch_lines,
    render_line_targets,
)
from tactifoot_vision.enums import PitchMarking, Task
from tactifoot_vision.pitch.markings import INDEX

W, H = 192, 108


def _image(path, segments):
    """A green image with white lines through ``segments`` (fractions)."""
    image = np.full((H, W, 3), (40, 140, 40), np.uint8)
    for (x0, y0), (x1, y1) in segments:
        cv2.line(
            image,
            (round(x0 * W), round(y0 * H)),
            (round(x1 * W), round(y1 * H)),
            (255, 255, 255),
            2,
        )
    cv2.imwrite(str(path), image)


def _calibration(root, split, count=3):
    folder = root / split
    folder.mkdir(parents=True)
    for i in range(count):
        y = 0.3 + 0.1 * i
        labels = {
            "Side line top": [{"x": 0.05, "y": y}, {"x": 0.95, "y": y}],
            "Middle line": [{"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}],
            "Goal left post left ": [{"x": 0.1, "y": 0.1}, {"x": 0.1, "y": 0.2}],
            "Line unknown": [{"x": 0.2, "y": 0.2}, {"x": 0.3, "y": 0.3}],
            "Circle central": [],
        }
        (folder / f"{i:05d}.json").write_text(json.dumps(labels))
        _image(
            folder / f"{i:05d}.jpg", [((0.05, y), (0.95, y)), ((0.5, 0.1), (0.5, 0.9))]
        )
    return root


def _gamestate(root, split="train", frames=5, on_disk=(1, 2, 3, 5)):
    """One GSR clip: a side line drawn on key frames 1 and 5, interpolated in between, moving at frame 3."""
    clip = root / split / "SNGS-001"
    (clip / "img1").mkdir(parents=True)
    ys = {
        1: 0.30,
        2: 0.32,
        3: 0.34,
        4: 0.40,
        5: 0.46,
    }  # linear 1-3 and 3-5, a kink at 3
    images = [
        {"image_id": f"1{t:06d}", "file_name": f"{t:06d}.jpg", "is_labeled": True}
        for t in range(1, frames + 1)
    ]
    annotations = [
        {"image_id": f"1{t:06d}", "supercategory": "pitch",
         "lines": {"Side line top": [{"x": 0.1, "y": ys[t]}, {"x": 0.9, "y": ys[t]}], "Goal right crossbar": []}}
        for t in range(1, frames + 1)
    ]  # fmt: skip
    labels = {
        "info": {"seq_length": frames},
        "images": images,
        "annotations": annotations,
        "categories": [],
    }
    (clip / "Labels-GameState.json").write_text(json.dumps(labels))
    for t in on_disk:
        _image(clip / "img1" / f"{t:06d}.jpg", [((0.1, ys[t]), (0.9, ys[t]))])
    return root


def test_calibration_lines_keep_the_planar_markings_in_pixels(tmp_path):
    samples = calibration_lines(_calibration(tmp_path, "train") / "train")
    assert len(samples) == 3
    lines = samples[0].annotations.lines
    assert set(lines) == {
        "Side line top",
        "Middle line",
    }  # posts, unknown and empty lines left out
    np.testing.assert_allclose(
        lines["Middle line"], [[0.5 * W, 0.1 * H], [0.5 * W, 0.9 * H]], rtol=1e-6
    )
    assert (
        samples[0].annotations.view == "main" and samples[0].annotations.weight == 1.0
    )
    assert len(samples[0].annotations) == 0  # no objects


def test_calibration_lines_follow_the_rule_table_order_and_views(tmp_path):
    folder = _calibration(tmp_path, "train") / "train"
    samples = calibration_lines(
        folder, keep={"00002": "along", "00000": "main", "99999": "main"}
    )
    assert [s.image_path.stem for s in samples] == ["00002", "00000"]
    assert [s.annotations.view for s in samples] == ["along", "main"]
    with pytest.raises(FileNotFoundError, match="label rule"):
        calibration_lines(folder, keep={"99999": "main"})


def test_gamestate_lines_weigh_interpolated_frames_less(tmp_path):
    samples = gamestate_lines(_gamestate(tmp_path) / "train", interpolated_weight=0.25)
    assert [int(s.image_path.stem) for s in samples] == [
        1,
        2,
        3,
        5,
    ]  # frame 4 is not on disk
    # 1 and 5 are the first and last labelled frames, 3 is where the line bends
    assert [s.annotations.weight for s in samples] == [1.0, 0.25, 1.0, 1.0]
    assert set(samples[0].annotations.lines) == {"Side line top"}


def test_load_pitch_lines_builds_a_lines_dataset(tmp_path):
    calib = _calibration(tmp_path / "calib", "valid", count=2)
    _calibration(calib, "train")
    gsr = _gamestate(tmp_path / "gsr")
    ds = load_pitch_lines(calib, gsr, rule="none")
    assert ds.task == Task.LINES and ds.class_names == [m.value for m in PitchMarking]
    assert len(ds["train"]) == 3 + 4 and len(ds["valid"]) == 2
    summary = ds.summary()
    assert (
        summary.loc["train", "images"] == 7
        and summary.loc["train", "Side line top"] == 7
    )
    assert summary.loc["train", "Middle line"] == 3
    with pytest.raises(ValueError, match="no YOLO layout"):
        ds.to_yolo(tmp_path / "yolo")
    with pytest.raises(ValueError, match="not pitch lines"):
        tv.augment.augment_dataset(ds, tv.augment.HorizontalFlip(), tmp_path / "aug")
    ruled = load_pitch_lines(
        calib, rule="soccernet-v2"
    )  # 00000-00002 are rule images; valid is not filtered
    assert [s.image_path.stem for s in ruled["train"]] == ["00000", "00001", "00002"]
    assert len(ruled["valid"]) == 2
    with pytest.raises(ValueError, match="rule"):
        load_pitch_lines(calib, rule="other")
    with pytest.raises(ValueError, match="Give"):
        load_pitch_lines()


def test_shipped_rule_table_holds_the_v2_training_images():
    from tactifoot_vision.data.pitch_lines import _rule_table

    table = _rule_table()
    assert len(table) == 8316
    assert sum(v == "along" for v in table.values()) == 731
    assert set(table.values()) == {"main", "along"}


def test_targets_peak_on_the_line_at_the_cell_convention():
    stride = 2
    lines = {
        "Middle line": np.array([[41.0, 0.0], [41.0, 60.0]]),
        "Line unknown": np.array([[0, 0], [9, 9.0]]),
    }
    target = render_line_targets(lines, (32, 32), stride)
    k = INDEX[PitchMarking.MIDDLE_LINE]
    assert target.shape == (20, 32, 32) and target.dtype == np.float32
    assert np.count_nonzero(target) == np.count_nonzero(
        target[k]
    )  # unknown names draw nothing
    row = target[k, 10]
    assert row.argmax() == 20 and row[20] == pytest.approx(
        1.0
    )  # input pixel 41 -> cell 41 / 2 - 0.5 = 20
    # one cell off: exp(-1/2), with cv2's 3 x 3 distance approximation
    assert row[19] == pytest.approx(row[21]) == pytest.approx(np.exp(-0.5), abs=0.03)
    assert row[15] == 0.0  # cut beyond 4 sigma


# ------------------------------------------------------------------ backend
torch = pytest.importorskip("torch")


def _tiny(**options):
    return tv.load_model("lines", width=64, pretrained=False, device="cpu", **options)


def test_decode_finds_ridge_points_in_frame_pixels():
    from tactifoot_vision.models import _line_net as L

    spec = L.NetSpec(
        width=64
    )  # image 64 x 36, input 64 x 48, output 32 x 24 at stride 2
    logits = torch.full((1, 20, 24, 32), -10.0)
    k = INDEX[PitchMarking.SIDE_LINE_TOP]
    logits[0, k, 5, :] = 3.0  # a horizontal ridge on output row 5
    logits[0, k, 4, :] = logits[
        0, k, 6, :
    ] = -2.0  # below the threshold, symmetric: no sub-cell shift
    logits[0, k, 20, :] = 3.0  # below the image rows (the padding): ignored
    (points,) = L.decode(logits, [(640, 360)], spec, threshold=0.2, bin=spec.bin)
    assert set(points.marking) == {k}
    assert len(points) == 32 // 3 + 1  # one per 3 x 3 block along the row
    # cell 5 -> input pixel 2 * 5.5 = 11 of 36 rows -> 110 of 360 frame rows
    np.testing.assert_allclose(points.xy[:, 1], 110.0, atol=1e-4)
    assert np.all(points.prob > 0.9)


def test_backend_predicts_line_points_and_checks_its_options(tmp_path):
    model = _tiny()
    assert (
        model.task == Task.LINES
        and model.spec.width == 64
        and len(model.class_names) == 20
    )
    points = model.predict(np.zeros((90, 160, 3), np.uint8))
    assert isinstance(points, tv.pitch.LinePoints)
    with pytest.raises(ValueError, match="checkpoint carries"):
        tv.load_model("lines", tmp_path / "x.pt", width=64)
    with pytest.raises(ValueError, match="stride"):
        _tiny(stride=8)


def _dataset(tmp_path):
    calib = _calibration(tmp_path / "calib", "train", count=4)
    _calibration(calib, "valid", count=2)
    return load_pitch_lines(calib, rule="none")


def test_training_writes_weights_and_a_manifest_and_resumes(tmp_path):
    ds = _dataset(tmp_path)
    settings = {
        "epochs": 2,
        "batch_size": 2,
        "imgsz": 64,
        "workers": 0,
        "device": "cpu",
        "output_dir": tmp_path / "runs",
    }
    model = _tiny()
    with pytest.raises(tv.models.TrainingStopped, match="after epoch 1"):
        model.train(ds, **settings, stop=lambda epoch: True)
    run = tmp_path / "runs" / "lines"
    assert (run / "weights" / "last.pt").is_file() and not (
        run / "weights" / "final.pt"
    ).exists()
    # the manifest and the sample index exist before the run ends
    stopped = json.loads((run / "manifest.json").read_text())
    assert stopped["weights"] is None and len(stopped["sessions"]) == 1
    index = [json.loads(x) for x in (run / "samples.jsonl").read_text().splitlines()]
    assert len(index) == 6 and {x["split"] for x in index} == {"train", "valid"}
    assert index[0]["view"] == "main" and len(index[0]["lines_sha256"]) == 64
    result = _tiny().train(ds, **settings, exist_ok=True)  # resumes at epoch 2
    assert result.weights == run / "weights" / "final.pt" and result.run_dir == run
    assert list(result.history["epoch"]) == [1, 2]
    assert set(result.metrics) == {"loss", "valid_loss"}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["recipe"]["epochs"] == 2 and manifest["recipe"]["lr"] == 6e-4
    assert manifest["data"]["train"] == 4 and len(manifest["sessions"]) == 2
    assert manifest["weights_sha256"] and manifest["decoder_version"] == 1
    loaded = tv.load_model("lines", result.weights, device="cpu")
    assert loaded.spec.width == 64
    # the same seed gives the same weights: the new network is drawn from it too
    again = _tiny().train(ds, **(settings | {"output_dir": tmp_path / "again"}))
    first, second = (
        torch.load(w, weights_only=False)["model"]
        for w in (result.weights, again.weights)
    )
    assert all(torch.equal(first[k], second[k]) for k in first)
    with pytest.raises(ValueError, match="another recipe"):
        _tiny().train(ds, **(settings | {"epochs": 3}), exist_ok=True)
    # another split of the data is refused too, and a checkpoint without a manifest
    with pytest.raises(ValueError, match="another recipe or data"):
        _tiny().train(ds.subset({"train": 3}), **settings, exist_ok=True)
    (run / "manifest.json").unlink()
    with pytest.raises(ValueError, match="without a manifest"):
        _tiny().train(ds, **settings, exist_ok=True)


def test_training_with_loader_workers_after_opencv_threads_does_not_hang(tmp_path):
    """A forked loader worker could deadlock in OpenCV once the parent had used its threads."""
    import subprocess
    import sys

    _dataset(tmp_path)
    script = f"""
import cv2, numpy as np
import tactifoot_vision as tv
x = np.random.rand(1500, 1500, 3).astype(np.float32)
for _ in range(3):
    cv2.resize(x, (999, 777)); cv2.GaussianBlur(x, (15, 15), 0)
ds = tv.load_pitch_lines({str(tmp_path / "calib")!r}, rule="none")
model = tv.load_model("lines", width=64, pretrained=False, device="cpu")
model.train(ds, epochs=1, batch_size=2, imgsz=64, workers=2, device="cpu", output_dir={str(tmp_path / "runs")!r})
print("trained")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert "trained" in out.stdout, out.stderr[-2000:]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"imgsz": 640}, "input width"), ({"patience": 3}, "patience"), ({"mosaic": 0.0}, "Unknown lines option"),
     ({"stop": 3}, "function")],
)  # fmt: skip
def test_training_refuses_settings_it_cannot_use(tmp_path, overrides, message):
    ds = _dataset(tmp_path)
    with pytest.raises(ValueError, match=message):
        _tiny().train(
            ds, **({"imgsz": 64, "output_dir": tmp_path / "runs"} | overrides)
        )
    assert not (tmp_path / "runs").exists()


def test_training_needs_a_lines_dataset():
    ds = tv.data.Dataset(Task.DETECT, ["ball"], {"train": [], "valid": []})
    with pytest.raises(ValueError, match="lines model but the dataset is detect"):
        _tiny().train(ds)


def test_cli_trains_lines_from_soccernet_folders(tmp_path, monkeypatch, capsys):
    calls = []

    class Trained:
        weights, metrics = tmp_path / "w.pt", {"loss": 1.0}

    monkeypatch.setattr(
        tv, "train", lambda *a, **kw: calls.append((a, kw)) or Trained()
    )
    calib = _calibration(tmp_path / "calib", "train")
    args = [
        "train",
        "lines",
        "--calibration",
        str(calib),
        "--rule",
        "none",
        "--splits",
        "[train]",
        "--epochs",
        "3",
    ]
    assert main(args) == 0
    ((name, dataset), kwargs), = calls  # fmt: skip
    assert name == "lines" and dataset.task == Task.LINES and len(dataset["train"]) == 3
    assert kwargs == {"epochs": 3}
    with pytest.raises(SystemExit):
        main(["train", "lines", "--data", "d", "--calibration", str(calib)])
    with pytest.raises(SystemExit):
        main(["train", "lines"])


# --------------------------------------------------------------- real weights
@pytest.mark.model
def test_real_line_model_fits_a_broadcast_frame(line_weights):
    video = (
        Path(__file__).resolve().parents[1] / "data" / "videos" / "broadcast_60s.mp4"
    )
    if not video.is_file():
        pytest.skip("real video not available")
    model = tv.load_model("lines", line_weights)
    assert model.spec == tv.models._line_net.NetSpec("resnet18", 2, 640)
    reader = tv.VideoReader(video)
    estimator = tv.pitch.LineHomographyEstimator()
    fresh = 0
    for _, frame in reader.frames(start=250, end=275):
        points = model.predict(frame)
        assert len(points) > 50 and len(set(points.marking.tolist())) >= 3
        estimator.update(points, reader.size)
        fresh += estimator.fit.fresh
    assert fresh >= 20
    assert estimator.fit.uncertainty_m < 1.0
    # a broadcast view: the frame outline is finite on the pitch, or beyond the horizon (None)
    corners = tv.pitch.visible_area(estimator.matrix, reader.size)
    assert corners is None or np.isfinite(corners).all()
