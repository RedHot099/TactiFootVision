import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tactifoot_vision.datasets import (
    SoccerNetTrackingDataset,
    iter_sequence_dirs,
    read_seqinfo,
)


def make_sequence(
    root: Path, name: str = "SNMOT-001", *, gameinfo: bool = True
) -> Path:
    sequence = root / name
    (sequence / "gt").mkdir(parents=True)
    (sequence / "img1").mkdir()
    (sequence / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={name}",
                "frameRate=25",
                "seqLength=1",
                "imWidth=20",
                "imHeight=10",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )
    if gameinfo:
        (sequence / "gameinfo.ini").write_text(
            "\n".join(["[Sequence]", "num_tracklets=1", "trackletID_1=Player 1"]),
            encoding="utf-8",
        )
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,2,3,5,4,1,-1,-1,-1\n", encoding="utf-8"
    )
    cv2.imwrite(
        str(sequence / "img1" / "000001.jpg"), np.zeros((10, 20, 3), dtype=np.uint8)
    )
    return sequence


def test_soccernet_to_coco_converts_synthetic_sequence(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    make_sequence(dataset_root)
    output = tmp_path / "coco"

    report = SoccerNetTrackingDataset(dataset_root).to_coco(
        output,
        valid_fraction=0.0,
        symlink_images=False,
    )

    train_json = json.loads((output / "train" / "_annotations.coco.json").read_text())
    valid_json = json.loads((output / "valid" / "_annotations.coco.json").read_text())
    test_json = json.loads((output / "test" / "_annotations.coco.json").read_text())
    assert report.sequences_total == 1
    assert report.sequences_train == 1
    assert len(train_json["images"]) == 1
    assert train_json["annotations"][0]["category_id"] == 2
    assert train_json["annotations"][0]["bbox"] == [2.0, 3.0, 5.0, 4.0]
    assert len(valid_json["images"]) == 1
    assert len(test_json["images"]) == 1


def test_missing_dataset_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        iter_sequence_dirs(tmp_path / "missing")


def test_invalid_seqinfo_raises(tmp_path: Path) -> None:
    sequence = tmp_path / "SNMOT-001"
    sequence.mkdir()
    (sequence / "seqinfo.ini").write_text("[Other]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Sequence"):
        read_seqinfo(sequence)


def test_empty_class_map_skips_annotations(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    make_sequence(dataset_root, gameinfo=False)
    output = tmp_path / "coco"

    SoccerNetTrackingDataset(dataset_root).to_coco(output, valid_fraction=0.0)

    train_json = json.loads((output / "train" / "_annotations.coco.json").read_text())
    assert train_json["annotations"] == []


def test_single_sequence_default_split_keeps_train_non_empty(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    make_sequence(dataset_root)
    output = tmp_path / "coco"

    SoccerNetTrackingDataset(dataset_root).to_coco(output, symlink_images=False)

    train_json = json.loads((output / "train" / "_annotations.coco.json").read_text())
    valid_json = json.loads((output / "valid" / "_annotations.coco.json").read_text())
    assert len(train_json["images"]) == 1
    assert len(valid_json["images"]) == 1
