"""Class-masked YOLO training (``annotated_classes``): loss, mosaic rule, trainer. CPU, synthetic data."""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

import tactifoot_vision as tv
from tactifoot_vision.models._yolo_class_mask import (
    ClassMaskedDataset,
    ClassMaskedLoss,
    ClassMaskedTrainer,
    read_class_masks,
)

NAMES = ["ball", "goalkeeper", "player", "referee"]


@pytest.fixture(scope="module")
def model() -> DetectionModel:
    torch.manual_seed(0)
    net = DetectionModel("yolo11n.yaml", nc=len(NAMES), verbose=False)
    net.args = get_cfg()
    return net.train()


def _preds(model: DetectionModel) -> list[torch.Tensor]:
    torch.manual_seed(1)
    with torch.no_grad():
        feats = model(torch.rand(2, 3, 64, 64))
    return [f.clone().requires_grad_() for f in feats]


def _batch(class_mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
    batch = {
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0], [2.0]]),
        "bboxes": torch.tensor(
            [[0.3, 0.3, 0.2, 0.3], [0.7, 0.6, 0.2, 0.3], [0.5, 0.5, 0.3, 0.4]]
        ),
    }
    if class_mask is not None:
        batch["class_mask"] = class_mask
    return batch


def test_all_ones_mask_gives_the_stock_loss(model):
    preds = _preds(model)
    stock, stock_items = v8DetectionLoss(model)(preds, _batch(None))
    ones = torch.ones(2, len(NAMES), dtype=torch.bool)
    masked, items = ClassMaskedLoss(model)(preds, _batch(ones))
    assert torch.allclose(masked, stock) and torch.allclose(items, stock_items)


def test_masked_class_gets_no_gradient_and_box_losses_stay(model):
    preds = _preds(model)
    mask = torch.ones(2, len(NAMES), dtype=torch.bool)
    mask[0, 1] = mask[0, 3] = (
        False  # image 0: its goalkeeper box and referees unlabelled
    )
    loss, items = ClassMaskedLoss(model)(preds, _batch(mask))
    loss.sum().backward()
    _, stock_items = v8DetectionLoss(model)(preds, _batch(None))

    reg = model.model[-1].reg_max * 4  # cls channels follow the box channels
    for feat in preds:
        for c in (1, 3):
            assert torch.count_nonzero(feat.grad[0, reg + c]) == 0
            assert torch.count_nonzero(feat.grad[1, reg + c]) > 0
        assert torch.count_nonzero(feat.grad[0, reg + 2]) > 0
    assert items[1] < stock_items[1]  # cls loss lost terms
    assert torch.allclose(items[[0, 2]], stock_items[[0, 2]])  # box and DFL unchanged


# ------------------------------------------------------------ data and mosaic
def _tiny_set(root: Path) -> tuple[Path, Path]:
    """Eight 64x64 images: 0-3 fully annotated, 4-7 annotate only player."""
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rows = []
    for i in range(8):
        image = images / f"img{i}.jpg"
        cv2.imwrite(str(image), np.full((64, 64, 3), 40 + 20 * i, np.uint8))
        (labels / f"img{i}.txt").write_text("2 0.5 0.5 0.3 0.4\n0 0.2 0.2 0.1 0.1\n")
        link = root / "links" / image.name  # the table may name a link to the image
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(image)
        rows.append((str(link), ",".join(NAMES) if i < 4 else "player"))
    table = root / "annotated.csv"
    pd.DataFrame(rows, columns=["image", "annotated_classes"]).to_csv(
        table, index=False
    )
    return images, table


def _dataset(images: Path, table: Path, **hyp) -> ClassMaskedDataset:
    return ClassMaskedDataset(
        img_path=str(images),
        imgsz=64,
        batch_size=4,
        augment=True,
        hyp=get_cfg(overrides={"mosaic": 1.0, "mixup": 0.0} | hyp),
        data={"names": dict(enumerate(NAMES)), "nc": len(NAMES), "channels": 3},
        class_masks=read_class_masks(table, NAMES),
    )


def test_table_keys_are_real_paths_and_unknown_classes_fail(tmp_path):
    images, table = _tiny_set(tmp_path)
    masks = read_class_masks(table, NAMES)
    assert masks[os.path.realpath(images / "img5.jpg")].tolist() == [0, 0, 1, 0]
    with pytest.raises(ValueError, match="unknown class"):
        read_class_masks(table, ["ball", "player"])


def test_mosaic_partners_share_the_mask(tmp_path):
    images, table = _tiny_set(tmp_path)
    data = _dataset(images, table)
    for first in range(len(data)):
        for _ in range(20):
            partner = data._partner(first)
            assert data.mask_group[partner] == data.mask_group[first]
    for i in range(len(data)):
        assert data[i]["class_mask"].tolist() == data.class_masks[i].tolist()


def test_mosaic_mask_is_the_and_of_its_tiles(tmp_path, monkeypatch):
    images, table = _tiny_set(tmp_path)
    data = _dataset(images, table)
    full = next(i for i in range(8) if data.class_masks[i].all())
    partial = next(i for i in range(8) if not data.class_masks[i].all())
    monkeypatch.setattr(data, "_partner", lambda first: partial)
    assert data[full]["class_mask"].tolist() == [False, False, True, False]
    monkeypatch.setattr(data, "_partner", lambda first: full)
    assert data[full]["class_mask"].all()
    batch = data.collate_fn([data[full], data[partial]])
    assert batch["class_mask"].shape == (2, len(NAMES))


def test_dataset_refuses_a_table_that_names_none_of_its_images(tmp_path):
    images, _ = _tiny_set(tmp_path)
    other = tmp_path / "other.csv"
    other.write_text("image,annotated_classes\n/nowhere/x.jpg,player\n")
    with pytest.raises(ValueError, match="No training image"):
        _dataset(images, other)


def test_trainer_masks_the_training_model_only(tmp_path):
    images, table = _tiny_set(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {tmp_path}\ntrain: images\nval: images\nnames: {NAMES}\n"
    )
    trainer = ClassMaskedTrainer(
        overrides={
            "model": "yolo11n.yaml",
            "data": str(data_yaml),
            "epochs": 1,
            "imgsz": 64,
            "batch": 4,
            "device": "cpu",
            "workers": 0,
            "amp": False,
            "plots": False,
            "val": False,
            "project": str(tmp_path / "runs"),
            "name": "masked",
        },
        class_masks=read_class_masks(table, NAMES),
    )
    trainer.train()
    assert isinstance(trainer.model.criterion, ClassMaskedLoss)
    assert isinstance(trainer.train_loader.dataset, ClassMaskedDataset)
    assert not isinstance(getattr(trainer.ema.ema, "criterion", None), ClassMaskedLoss)
    assert np.isfinite(trainer.tloss.numpy()).all()
    checkpoint = torch.load(trainer.last, weights_only=False)
    assert "ClassMasked" not in repr(checkpoint["ema"])


def test_repeated_images_in_a_list_get_their_mask_on_every_copy(tmp_path):
    """Oversampling lists an image several times; the export links each copy to the same file."""
    images, table = _tiny_set(tmp_path)
    listed = [images / "img0.jpg"] * 3 + [images / "img5.jpg"] * 2
    (tmp_path / "train.txt").write_text("".join(f"{p}\n" for p in listed))
    data_yaml = tmp_path / "list.yaml"
    data_yaml.write_text(
        f"path: {tmp_path}\ntrain: train.txt\nval: train.txt\nnames: {NAMES}\n"
    )
    dataset = tv.load_dataset(data_yaml)
    assert len(dataset["train"]) == 5
    exported = dataset.to_yolo(tmp_path / "export")
    data = _dataset(exported.parent / "train/images", table)
    assert len(data) == 5
    assert sorted(data.class_masks.sum(axis=1).tolist()) == [1, 1, 4, 4, 4]
