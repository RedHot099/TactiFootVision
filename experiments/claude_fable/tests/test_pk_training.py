"""Training inputs of the pitch keypoint candidates (pitch-keypoints 03): run lists and the heatmap model's pieces."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from xgpoc.pitchkp import dataset as D
from xgpoc.pitchkp import evaluator as ev
from xgpoc.pitchkp import heatmap as H
from xgpoc.pitchkp import training as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def fake_dataset(root: Path, n_train: int = 5, n_val: int = 9) -> Path:
    for split, n in (("train", n_train), ("val", n_val), ("test_gsr", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            (root / "images" / split / f"{split}_{i:03d}.jpg").write_bytes(b"")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                                    "kpt_shape": [32, 3], "flip_idx": list(D.FLIP_IDX), "nc": 1,
                                                    "names": ["pitch"]}))
    return root


def test_training_yaml_lists_train_and_a_val_subset_and_never_the_test_splits(tmp_path):
    root = fake_dataset(tmp_path / "ds")
    data_yaml = T.training_yaml(root, tmp_path / "run", val_step=4)
    config = yaml.safe_load(data_yaml.read_text())
    assert config["flip_idx"] == list(D.FLIP_IDX) and config["kpt_shape"] == [32, 3]
    train = (tmp_path / "run" / config["train"]).read_text().split()
    val = (tmp_path / "run" / config["val"]).read_text().split()
    assert [Path(p).name for p in train] == [f"train_{i:03d}.jpg" for i in range(5)]
    assert [Path(p).name for p in val] == ["val_000.jpg", "val_004.jpg", "val_008.jpg"]
    assert {Path(p).parent.name for p in train + val} == {"train", "val"}
    limited = T.training_yaml(root, tmp_path / "run2", train_limit=2, seed=0)
    assert len((limited.parent / "train.txt").read_text().split()) == 2
    with pytest.raises(ValueError, match="val_step"):
        T.training_yaml(root, tmp_path / "run3", val_step=0)


def test_augmentation_keeps_one_pitch_per_image():
    assert T.YOLO_AUGMENT["mosaic"] == 0 and T.YOLO_AUGMENT["mixup"] == 0 and T.YOLO_AUGMENT["cutmix"] == 0
    assert T.YOLO_AUGMENT["copy_paste"] == 0 and T.YOLO_AUGMENT["flipud"] == 0 and T.YOLO_AUGMENT["fliplr"] > 0


def test_option_parses_values():
    import pk_train_yolo as run

    assert run.option("scale=0") == ("scale", 0)
    assert run.option("optimizer=AdamW") == ("optimizer", "AdamW")
    with pytest.raises(Exception, match="key=value"):
        run.option("scale")


def test_targets_and_decoding_recover_sub_pixel_keypoints():
    xy = np.zeros((32, 2), np.float32)
    visible = np.zeros(32, bool)
    xy[[0, 5, 9]] = [[101.3, 57.9], [480.0, 270.0], [958.7, 2.2]]
    visible[[0, 5, 9]] = True
    xy[3], visible[3] = (2000.0, 10.0), True                      # outside the map: no target
    heatmaps, index, offset = H.targets(xy, visible)
    assert heatmaps.shape == (32, *H.HEATMAP_SHAPE) and heatmaps[visible & (index >= 0)].max() == 1.0
    assert index[3] == -1 and not heatmaps[3].any() and not heatmaps[1].any()
    # Logits that reproduce the targets decode to the keypoints.
    logits = torch.from_numpy(np.log(np.clip(heatmaps, 1e-6, 1 - 1e-6) / (1 - np.clip(heatmaps, 1e-6, 1 - 1e-6))))
    offsets = torch.zeros(1, 32, 2, *H.HEATMAP_SHAPE)
    w = H.HEATMAP_SHAPE[1]
    for k in np.flatnonzero(index >= 0):
        offsets[0, k, :, index[k] // w, index[k] % w] = torch.from_numpy(offset[k])
    decoded, conf = H.decode(logits[None], offsets)
    np.testing.assert_allclose(decoded[0][[0, 5, 9]], xy[[0, 5, 9]], atol=1e-3)
    assert (conf[0][[0, 5, 9]] > 0.99).all() and conf[0][1] < 0.01
    assert float(H.offset_loss(offsets, torch.from_numpy(index)[None], torch.from_numpy(offset)[None])) < 1e-6


def test_focal_loss_prefers_the_target():
    heatmaps, _, _ = H.targets(np.array([[480.0, 270.0]] * 32, np.float32), np.ones(32, bool))
    target = torch.from_numpy(heatmaps)[None]
    good = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    assert float(H.focal_loss(good, target)) < float(H.focal_loss(torch.zeros_like(good), target))


def test_flip_follows_flip_idx_and_mirrors_x():
    image = np.zeros((540, 960, 3), np.uint8)
    xy = np.arange(64, dtype=np.float32).reshape(32, 2)
    visible = np.arange(32) % 2 == 0
    flipped, fxy, fvis = H.flip(image, xy, visible, D.FLIP_IDX)
    for i, j in enumerate(D.FLIP_IDX):
        assert fvis[i] == visible[j] and fxy[i, 0] == 960 - xy[j, 0] and fxy[i, 1] == xy[j, 1]
    assert flipped.shape == image.shape


def test_affine_moves_image_and_keypoints_together_and_drops_the_ones_that_leave():
    image = np.zeros((540, 960, 3), np.uint8)
    cv2.circle(image, (300, 200), 3, (255, 255, 255), -1)        # a dot centred on continuous (300.5, 200.5)
    xy = np.float32([[300.5, 200.5], [950.0, 530.0]] + [[0, 0]] * 30)
    visible = np.array([True, True] + [False] * 30)
    out, oxy, ovis = H.affine(image, xy, visible, scale=1.2, shift=(10.0, -5.0))
    ys, xs = np.nonzero(out[:, :, 0] > 127)
    assert abs(xs.mean() + 0.5 - oxy[0, 0]) < 0.3 and abs(ys.mean() + 0.5 - oxy[0, 1]) < 0.3
    assert ovis[0] and not ovis[1] and (oxy[1] == 0).all()


def test_dataset_item_reads_labels_and_shapes(tmp_path):
    images = tmp_path / "ds/images/train"
    labels = tmp_path / "ds/labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    cv2.imwrite(str(images / "a.jpg"), np.full((540, 960, 3), 90, np.uint8))
    values = [0, 0.5, 0.5, 1.0, 1.0] + [0.0] * 96
    values[5:8] = [0.25, 0.5, 2]
    (labels / "a.txt").write_text(" ".join(map(str, values)))
    assert H.label_path(images / "a.jpg") == labels / "a.txt"
    item = H.KeypointImages([images / "a.jpg"], D.FLIP_IDX)[0]
    x, heatmaps, index, offset, xy, visible = item
    assert x.shape == (3, H.INPUT_H, H.INPUT_W) and heatmaps.shape == (32, *H.HEATMAP_SHAPE)
    assert visible.sum() == 1 and xy[0].tolist() == [240.0, 270.0] and index[0] >= 0
    augmented = H.KeypointImages([images / "a.jpg"], D.FLIP_IDX, H.Augment(), seed=1)
    assert augmented[0][0].shape == x.shape


def test_heatmap_model_runs_on_a_frame_and_returns_keypoints(tmp_path):
    net = H.HeatmapNet(H.HeatmapConfig(backbone="resnet18", decoder_channels=16), pretrained=False)
    H.save_checkpoint(tmp_path / "m.pt", net)
    model = ev.KeypointModel(tmp_path / "m.pt", "letterbox", None, 0.1, "heatmap").load("cpu")
    found = model(np.zeros((720, 1280, 3), np.uint8))
    assert found.xy.shape == (1, 32, 2) and found.confidence.shape == (1, 32)
    assert float(found.data["box_confidence"][0]) == pytest.approx(float(found.confidence.max()))
    with pytest.raises(ValueError, match="fixed"):
        ev.KeypointModel(tmp_path / "m.pt", "stretch", None, 0.1, "heatmap")
    assert "backend" not in ev.KeypointModel(tmp_path / "m.pt").settings()     # ticket 02's cache identities hold


def test_square_input_pads_the_frame_to_a_centred_square_and_maps_keypoints_back():
    seen = {}

    def model(canvas):
        seen["canvas"] = canvas
        ys, xs = np.nonzero(canvas[:, :, 0] == 255)                # the frame's marked pixel, in canvas pixels
        return sv_keypoints(np.float32([[[xs[0], ys[0]]] * 32]))

    image = np.zeros((720, 1280, 3), np.uint8)
    image[100, 200] = 255
    found = ev.SquareInput(model)(image)
    canvas = seen["canvas"]
    assert canvas.shape == (1280, 1280, 3) and (canvas[:280] == ev.PAD_VALUE).all() and (canvas[1000:] == 114).all()
    np.testing.assert_allclose(found.xy[0, 0], [200, 100])
    np.testing.assert_allclose(found.data["xyxy"][0], [0, -280, 1280, 1000])


def sv_keypoints(xy):
    import supervision as sv

    return sv.KeyPoints(xy=xy, confidence=np.ones(xy.shape[:2], np.float32), class_id=np.zeros(1, int),
                        data={"xyxy": np.float32([[0, 0, 1280, 1280]]), "box_confidence": np.float32([0.9])})
