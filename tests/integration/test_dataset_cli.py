import json
from pathlib import Path

import cv2
import numpy as np

from tactifoot_vision.cli import main


def make_sequence(root: Path, name: str = "SNMOT-001") -> None:
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


def test_cli_dataset_convert_soccernet_tracking(tmp_path: Path, capsys) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    make_sequence(dataset_root)
    output = tmp_path / "coco"

    assert (
        main(
            [
                "dataset",
                "convert",
                "soccernet-tracking",
                "--input",
                str(dataset_root),
                "--output",
                str(output),
                "--valid-fraction",
                "0",
                "--max-sequences",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["sequences_total"] == 1
    assert payload["sequences_train"] == 1
    assert (output / "train" / "_annotations.coco.json").is_file()
