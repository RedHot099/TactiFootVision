import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tactifoot_vision.datasets.coco import CocoConversionReport

SOCCERNET_CLASS_TO_ID: dict[str, int] = {
    "ball": 0,
    "goalkeeper": 1,
    "player": 2,
    "referee": 3,
}


@dataclass(frozen=True, slots=True)
class SequenceInfo:
    name: str
    frame_rate: int
    seq_length: int
    width: int
    height: int
    image_ext: str


class SoccerNetTrackingDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def to_coco(
        self,
        output_dir: str | Path,
        *,
        valid_fraction: float = 0.2,
        seed: int = 42,
        every_nth_frame: int = 1,
        max_sequences: int | None = None,
        symlink_images: bool = True,
    ) -> CocoConversionReport:
        return export_mot_to_coco(
            self.root,
            Path(output_dir),
            valid_fraction=valid_fraction,
            seed=seed,
            every_nth_frame=every_nth_frame,
            max_sequences=max_sequences,
            symlink_images=symlink_images,
        )


def iter_sequence_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path / "seqinfo.ini").is_file()
            and (path / "gt" / "gt.txt").is_file()
        ],
        key=lambda path: path.name,
    )


def read_seqinfo(seq_dir: Path) -> SequenceInfo:
    seqinfo_path = seq_dir / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    if "Sequence" not in parser:
        raise ValueError(f"Invalid seqinfo.ini (missing [Sequence]): {seqinfo_path}")
    section = parser["Sequence"]
    image_ext = section.get("imExt", fallback=".jpg").strip() or ".jpg"
    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"
    width = section.getint("imWidth", fallback=0)
    height = section.getint("imHeight", fallback=0)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image size in {seqinfo_path}: imWidth={width} imHeight={height}"
        )
    return SequenceInfo(
        name=section.get("name", fallback=seq_dir.name).strip() or seq_dir.name,
        frame_rate=section.getint("frameRate", fallback=25),
        seq_length=section.getint("seqLength", fallback=0),
        width=width,
        height=height,
        image_ext=image_ext,
    )


def load_mot_gt(gt_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(gt_path, header=None)
    dataframe = dataframe.iloc[:, :7]
    dataframe.columns = ["frame", "track_id", "x", "y", "w", "h", "confidence"]
    dataframe["frame"] = dataframe["frame"].astype(int)
    dataframe["track_id"] = dataframe["track_id"].astype(int)
    return dataframe


def parse_tracklet_class_map(gameinfo_path: Path) -> dict[int, str]:
    parser = configparser.ConfigParser()
    parser.read(gameinfo_path)
    if "Sequence" not in parser:
        return {}
    section = parser["Sequence"]
    count = section.getint("num_tracklets", fallback=0)
    mapping: dict[int, str] = {}
    for index in range(1, max(0, count) + 1):
        raw = section.get(f"trackletID_{index}", fallback="").lower().strip()
        label = _tracklet_label(raw)
        if label is not None:
            mapping[index] = label
    return mapping


def clamp_xywh(
    x: float, y: float, w: float, h: float, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 1.0 or new_h <= 1.0:
        return None
    return x1, y1, new_w, new_h


def export_mot_to_coco(
    dataset_root: Path,
    output_root: Path,
    *,
    valid_fraction: float = 0.2,
    seed: int = 42,
    every_nth_frame: int = 1,
    max_sequences: int | None = None,
    symlink_images: bool = True,
) -> CocoConversionReport:
    if not 0.0 <= float(valid_fraction) < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    sequence_dirs = iter_sequence_dirs(dataset_root)
    if max_sequences is not None:
        sequence_dirs = sequence_dirs[:max_sequences]
    split_by_sequence = _split_sequences(
        sequence_dirs, valid_fraction=valid_fraction, seed=seed
    )
    coco_by_split = {split: _empty_coco() for split in ("train", "valid")}
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        (output_root / split).mkdir(parents=True, exist_ok=True)
    image_id = 1
    annotation_id = 1
    every = max(1, int(every_nth_frame))
    for sequence_dir in sequence_dirs:
        split = split_by_sequence[sequence_dir.name]
        seqinfo = read_seqinfo(sequence_dir)
        class_map = parse_tracklet_class_map(sequence_dir / "gameinfo.ini")
        if not class_map:
            continue
        ground_truth = load_mot_gt(sequence_dir / "gt" / "gt.txt")
        for frame_number, rows in ground_truth.groupby("frame", sort=True):
            if int(frame_number) % every != 0:
                continue
            source_image = (
                sequence_dir / "img1" / f"{int(frame_number):06d}{seqinfo.image_ext}"
            )
            if not source_image.is_file():
                continue
            destination_name = (
                f"{sequence_dir.name}_{int(frame_number):06d}{seqinfo.image_ext}"
            )
            destination_image = output_root / split / destination_name
            _copy_or_link(
                source_image, destination_image, symlink_images=symlink_images
            )
            coco_by_split[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": destination_name,
                    "width": seqinfo.width,
                    "height": seqinfo.height,
                }
            )
            for row in rows.itertuples(index=False):
                class_name = class_map.get(int(row.track_id))
                if class_name is None:
                    continue
                bbox = clamp_xywh(
                    float(row.x),
                    float(row.y),
                    float(row.w),
                    float(row.h),
                    width=seqinfo.width,
                    height=seqinfo.height,
                )
                if bbox is None:
                    continue
                x, y, width, height = bbox
                coco_by_split[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": SOCCERNET_CLASS_TO_ID[class_name],
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "segmentation": [
                            [x, y, x, y + height, x + width, y + height, x + width, y]
                        ],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
            image_id += 1
    _write_coco_splits(output_root, coco_by_split)
    source_for_test = "valid" if coco_by_split["valid"]["images"] else "train"
    if source_for_test == "train" and not coco_by_split["valid"]["images"]:
        _mirror_split(output_root, "train", "valid", coco_by_split)
    _mirror_split(output_root, source_for_test, "test", coco_by_split)
    (output_root / "sequence_splits.json").write_text(
        json.dumps(split_by_sequence, indent=2), encoding="utf-8"
    )
    return CocoConversionReport(
        dataset_root=dataset_root,
        output_root=output_root,
        valid_fraction=float(valid_fraction),
        seed=int(seed),
        every_nth_frame=every,
        sequences_total=len(sequence_dirs),
        sequences_train=sum(
            1 for value in split_by_sequence.values() if value == "train"
        ),
        sequences_valid=sum(
            1 for value in split_by_sequence.values() if value == "valid"
        ),
        test_split=f"copy_of_{source_for_test}",
    )


def _tracklet_label(raw: str) -> str | None:
    if raw.startswith("player "):
        return "player"
    if raw.startswith("referee"):
        return "referee"
    if raw.startswith("ball"):
        return "ball"
    if raw.startswith("goalkeeper") or raw.startswith("goalkeepers"):
        return "goalkeeper"
    return None


def _split_sequences(
    sequence_dirs: list[Path], *, valid_fraction: float, seed: int
) -> dict[str, str]:
    if valid_fraction == 0.0 or len(sequence_dirs) <= 1:
        return {sequence_dir.name: "train" for sequence_dir in sequence_dirs}
    rng = np.random.default_rng(seed)
    valid_count = min(
        len(sequence_dirs) - 1,
        max(1, int(round(len(sequence_dirs) * valid_fraction))),
    )
    valid_indexes = set(
        rng.choice(len(sequence_dirs), size=valid_count, replace=False).tolist()
    )
    return {
        sequence_dir.name: "valid" if index in valid_indexes else "train"
        for index, sequence_dir in enumerate(sequence_dirs)
    }


def _empty_coco() -> dict[str, object]:
    return {
        "info": {
            "description": "SoccerNet Tracking converted to COCO.",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": class_name, "supercategory": "common-objects"}
            for class_name, class_id in SOCCERNET_CLASS_TO_ID.items()
        ],
    }


def _copy_or_link(source: Path, destination: Path, *, symlink_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if symlink_images:
        try:
            destination.symlink_to(os.path.relpath(source, start=destination.parent))
            return
        except OSError:
            pass
    destination.write_bytes(source.read_bytes())


def _write_coco_splits(output_root: Path, coco_by_split: dict[str, dict]) -> None:
    for split, coco in coco_by_split.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco, indent=2), encoding="utf-8"
        )


def _mirror_split(
    output_root: Path,
    source_split: str,
    destination_split: str,
    coco_by_split: dict[str, dict],
) -> None:
    source_dir = output_root / source_split
    destination_dir = output_root / destination_split
    destination_dir.mkdir(parents=True, exist_ok=True)
    coco_by_split[destination_split] = coco_by_split[source_split]
    (destination_dir / "_annotations.coco.json").write_text(
        json.dumps(coco_by_split[source_split], indent=2), encoding="utf-8"
    )
    for item in source_dir.iterdir():
        if not item.is_file() or item.name == "_annotations.coco.json":
            continue
        _copy_or_link(item.resolve(), destination_dir / item.name, symlink_images=True)
