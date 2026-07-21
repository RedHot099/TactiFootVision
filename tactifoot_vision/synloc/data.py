from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import zipfile

import cv2
from SoccerNet.Downloader import SoccerNetDownloader
import yaml

from config.synloc_models import SynLocAuxiliaryTask, SynLocDatasetConfig
from tactifoot_vision.synloc.camera import camera_from_image_record


@dataclass(frozen=True)
class SynLocImageRecord:
    image_id: int
    file_name: str
    file_path: Path
    width: int
    height: int
    camera_matrix: list[list[float]]
    dist_poly: list[float]
    undist_poly: list[float]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (3, int(self.height), int(self.width))


@dataclass(frozen=True)
class SynLocAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    position_on_pitch_xyz: list[float]
    bbox_xywh: list[float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynLocSplitData:
    dataset_root: Path
    split: str
    annotation_path: Path
    images: list[SynLocImageRecord]
    images_by_id: dict[int, SynLocImageRecord]
    annotations: list[SynLocAnnotation]
    annotations_by_image: dict[int, list[SynLocAnnotation]]
    categories: dict[int, str]


@dataclass(frozen=True)
class GameStateImageRecord:
    image_key: str
    video_id: str
    image_id: str
    file_path: Path
    width: int
    height: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameStateAnnotation:
    annotation_id: str
    image_key: str
    video_id: str
    image_id: str
    role: str
    position_on_pitch_xyz: list[float]
    bbox_xywh: list[float]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameStateSplitData:
    dataset_root: Path
    split: str
    task: str
    annotation_paths: list[Path]
    images: list[GameStateImageRecord]
    images_by_key: dict[str, GameStateImageRecord]
    annotations: list[GameStateAnnotation]
    annotations_by_image: dict[str, list[GameStateAnnotation]]


def download_synloc_dataset(
    root: Path,
    *,
    splits: list[str],
    image_version: str = "4K",
) -> Path:
    root = Path(root).resolve()
    local_directory = root.parent if root.name == "SpiideoSynLoc" else root
    downloader = SoccerNetDownloader(LocalDirectory=str(local_directory))
    kwargs = {}
    if image_version == "fullhd":
        kwargs["version"] = "fullhd"
    download_splits = ["valid" if split == "val" else split for split in splits]
    downloader.downloadDataTask(task="SpiideoSynLoc", split=download_splits, **kwargs)
    return root if root.name == "SpiideoSynLoc" else root / "SpiideoSynLoc"


def download_gamestate_dataset(
    root: Path,
    *,
    task: SynLocAuxiliaryTask,
    splits: list[str],
) -> Path:
    root = Path(root).resolve()
    if root.name == task:
        local_directory = root.parent
    else:
        local_directory = root
    downloader = SoccerNetDownloader(LocalDirectory=str(local_directory))
    download_splits = ["valid" if split == "val" else split for split in splits]
    downloader.downloadDataTask(task=task, split=download_splits)
    return root if root.name == task else root / task


def annotation_filename_for_split(dataset_root: Path, split: str) -> Path:
    canonical = "valid" if split == "valid" else split
    if canonical == "challenge":
        public_path = dataset_root / "annotations" / "challenge_public.json"
        if public_path.is_file():
            return public_path
    if canonical == "val":
        aliases = ["val.json", "valid.json"]
    elif canonical == "valid":
        aliases = ["valid.json", "val.json"]
    else:
        aliases = [f"{canonical}.json"]
    for name in aliases:
        candidate = dataset_root / "annotations" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find annotation file for split '{split}' under {dataset_root / 'annotations'}."
    )


def load_synloc_split(config: SynLocDatasetConfig) -> SynLocSplitData:
    dataset_root = config.root.resolve()
    annotation_path = annotation_filename_for_split(dataset_root, config.split)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images: list[SynLocImageRecord] = []
    images_by_id: dict[int, SynLocImageRecord] = {}
    split_dir = _resolve_split_dir(dataset_root, config.split)
    for image in payload.get("images", []):
        file_name = str(image["file_name"])
        file_path = _resolve_image_path(split_dir, dataset_root, file_name)
        calibration = camera_from_image_record(image, image_path=file_path)
        record = SynLocImageRecord(
            image_id=int(image["id"]),
            file_name=file_name,
            file_path=file_path,
            width=int(image["width"]),
            height=int(image["height"]),
            camera_matrix=calibration.camera_matrix,
            dist_poly=calibration.dist_poly,
            undist_poly=calibration.undist_poly,
            raw=dict(image),
        )
        images.append(record)
        images_by_id[record.image_id] = record

    annotations: list[SynLocAnnotation] = []
    annotations_by_image: dict[int, list[SynLocAnnotation]] = {}
    for ann in payload.get("annotations", []):
        position = ann.get("position_on_pitch")
        if position is None:
            continue
        parsed = SynLocAnnotation(
            annotation_id=int(ann["id"]),
            image_id=int(ann["image_id"]),
            category_id=int(ann.get("category_id", 1)),
            position_on_pitch_xyz=[float(v) for v in position],
            bbox_xywh=[float(v) for v in ann["bbox"]] if "bbox" in ann else None,
            raw=dict(ann),
        )
        annotations.append(parsed)
        annotations_by_image.setdefault(parsed.image_id, []).append(parsed)

    categories = {
        int(category["id"]): str(category["name"])
        for category in payload.get("categories", [])
    }
    return SynLocSplitData(
        dataset_root=dataset_root,
        split=config.split,
        annotation_path=annotation_path,
        images=images,
        images_by_id=images_by_id,
        annotations=annotations,
        annotations_by_image=annotations_by_image,
        categories=categories,
    )


def smoke_check_synloc_root(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    annotations_dir = root / "annotations"
    status: dict[str, object] = {"root": str(root), "splits": {}}
    if not annotations_dir.is_dir():
        status["annotations_present"] = False
        return status
    status["annotations_present"] = True
    for split in ("train", "val", "valid", "test", "challenge"):
        try:
            ann_path = annotation_filename_for_split(root, split)
        except FileNotFoundError:
            continue
        with ann_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        status["splits"][split] = {
            "annotation_path": str(ann_path),
            "images": len(payload.get("images", [])),
            "annotations": len(payload.get("annotations", [])),
        }
    return status


def smoke_check_gamestate_root(root: Path, *, split_names: tuple[str, ...] = ("train", "valid", "test", "challenge")) -> dict[str, object]:
    root = Path(root).resolve()
    status: dict[str, object] = {"root": str(root), "splits": {}}
    for split in split_names:
        split_dir = root / split
        if not split_dir.exists():
            continue
        label_files = sorted(split_dir.rglob("Labels-GameState.json"))
        detections = 0
        videos = 0
        for label_path in label_files:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            detections += len(_extract_gamestate_detections(payload))
            videos += 1
        status["splits"][split] = {
            "videos": videos,
            "label_files": len(label_files),
            "detections": detections,
        }
    return status


def load_gamestate_split(
    root: Path,
    *,
    split: str,
    task: SynLocAuxiliaryTask | None = None,
    player_roles: tuple[str, ...] = ("player",),
) -> GameStateSplitData:
    dataset_root = _resolve_gamestate_dataset_root(Path(root).resolve(), task)
    split_dir = _resolve_gamestate_split_dir(dataset_root, split)
    annotation_paths = sorted(split_dir.rglob("Labels-GameState.json"))

    images_by_key: dict[str, GameStateImageRecord] = {}
    annotations: list[GameStateAnnotation] = []
    annotations_by_image: dict[str, list[GameStateAnnotation]] = {}
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for detection in _extract_gamestate_detections(payload):
            role = str(detection.get("attributes", {}).get("role", "player")).lower()
            if role not in player_roles:
                continue
            bbox_xywh = _extract_gamestate_bbox(detection)
            if bbox_xywh is None:
                continue
            position = _extract_gamestate_pitch_position(detection)
            if position is None:
                continue

            video_id = str(detection.get("video_id") or annotation_path.parent.name)
            image_id = str(detection.get("image_id"))
            image_key = f"{video_id}:{image_id}"
            image_path = _resolve_gamestate_image_path(annotation_path.parent, image_id)
            width, height = _read_image_size(image_path)
            images_by_key.setdefault(
                image_key,
                GameStateImageRecord(
                    image_key=image_key,
                    video_id=video_id,
                    image_id=image_id,
                    file_path=image_path,
                    width=width,
                    height=height,
                    raw={"annotation_path": str(annotation_path)},
                ),
            )
            parsed = GameStateAnnotation(
                annotation_id=str(detection.get("id", f"{image_key}:{len(annotations)}")),
                image_key=image_key,
                video_id=video_id,
                image_id=image_id,
                role=role,
                position_on_pitch_xyz=position,
                bbox_xywh=bbox_xywh,
                raw=dict(detection),
            )
            annotations.append(parsed)
            annotations_by_image.setdefault(image_key, []).append(parsed)

    return GameStateSplitData(
        dataset_root=dataset_root,
        split=split,
        task=task or dataset_root.name,
        annotation_paths=annotation_paths,
        images=sorted(images_by_key.values(), key=lambda item: item.image_key),
        images_by_key=images_by_key,
        annotations=annotations,
        annotations_by_image=annotations_by_image,
    )


def export_synloc_detection_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    symlink_images: bool = True,
    auxiliary_roots: tuple[Path, ...] = (),
    auxiliary_tasks: tuple[SynLocAuxiliaryTask, ...] = (),
    max_aux_images_per_split: int | None = None,
) -> dict[str, Path]:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    coco_root = output_dir / "coco"
    yolo_root = output_dir / "yolo"
    coco_root.mkdir(parents=True, exist_ok=True)
    yolo_root.mkdir(parents=True, exist_ok=True)

    if auxiliary_tasks and len(auxiliary_tasks) != len(auxiliary_roots):
        raise ValueError("auxiliary_tasks and auxiliary_roots must have the same length.")

    yolo_names = ["person"]
    split_map = {"val": "valid", "valid": "valid"}
    for split in splits:
        split_data = load_synloc_split(SynLocDatasetConfig(root=dataset_root, split=split))
        export_name = split_map.get(split, split)

        coco_split_dir = coco_root / export_name
        yolo_images_dir = yolo_root / export_name / "images"
        yolo_labels_dir = yolo_root / export_name / "labels"
        coco_split_dir.mkdir(parents=True, exist_ok=True)
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)

        coco_payload = {
            "info": {"description": "SynLoc detection export"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "person", "supercategory": "person"}],
        }
        ann_id = 1
        export_images: dict[str, dict[str, object]] = {}
        export_annotations: list[dict[str, object]] = []

        for image_record in split_data.images:
            export_key = f"synloc:{split}:{image_record.image_id}"
            export_images[export_key] = {
                "source_key": export_key,
                "file_path": image_record.file_path,
                "width": image_record.width,
                "height": image_record.height,
            }
            for annotation in split_data.annotations_by_image.get(image_record.image_id, []):
                if annotation.bbox_xywh is None:
                    continue
                export_annotations.append({"image_key": export_key, "bbox_xywh": annotation.bbox_xywh})

        for auxiliary_root, auxiliary_task in zip(auxiliary_roots, auxiliary_tasks):
            auxiliary_split = load_gamestate_split(auxiliary_root, split=export_name, task=auxiliary_task)
            allowed_keys = {record.image_key for record in auxiliary_split.images}
            if max_aux_images_per_split is not None:
                allowed_keys = set(sorted(allowed_keys)[:max_aux_images_per_split])
            for image_record in auxiliary_split.images:
                if image_record.image_key not in allowed_keys:
                    continue
                export_key = f"{auxiliary_task}:{split}:{image_record.image_key}"
                export_images[export_key] = {
                    "source_key": export_key,
                    "file_path": image_record.file_path,
                    "width": image_record.width,
                    "height": image_record.height,
                }
            for annotation in auxiliary_split.annotations:
                if annotation.image_key not in allowed_keys:
                    continue
                export_annotations.append(
                    {
                        "image_key": f"{auxiliary_task}:{split}:{annotation.image_key}",
                        "bbox_xywh": annotation.bbox_xywh,
                    }
                )

        coco_image_id_by_key: dict[str, int] = {}
        for image_index, export_key in enumerate(sorted(export_images), start=1):
            image_info = export_images[export_key]
            file_path = Path(image_info["file_path"])
            file_name = _make_export_image_name(export_key, file_path)
            coco_image_id_by_key[export_key] = image_index
            _link_or_copy(file_path, coco_split_dir / file_name, symlink_images)
            _link_or_copy(file_path, yolo_images_dir / file_name, symlink_images)
            coco_payload["images"].append(
                {
                    "id": image_index,
                    "file_name": file_name,
                    "width": int(image_info["width"]),
                    "height": int(image_info["height"]),
                }
            )

            label_lines: list[str] = []
            for annotation in export_annotations:
                if annotation["image_key"] != export_key:
                    continue
                x, y, w, h = [float(v) for v in annotation["bbox_xywh"]]
                width = float(image_info["width"])
                height = float(image_info["height"])
                coco_payload["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_index,
                        "category_id": 0,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                label_lines.append(
                    f"0 {(x + w / 2.0) / width:.8f} {(y + h / 2.0) / height:.8f} {w / width:.8f} {h / height:.8f}"
                )
            (yolo_labels_dir / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )

        (coco_split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco_payload, indent=2),
            encoding="utf-8",
        )

    _ensure_test_alias(coco_root=coco_root, yolo_root=yolo_root)

    data_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 1,
        "names": yolo_names,
    }
    (yolo_root / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")
    return {"coco_root": coco_root, "yolo_yaml": yolo_root / "data.yaml"}


def _ensure_test_alias(*, coco_root: Path, yolo_root: Path) -> None:
    coco_valid_dir = coco_root / "valid"
    coco_test_dir = coco_root / "test"
    if coco_valid_dir.is_dir() and not coco_test_dir.exists():
        _symlink_tree(coco_valid_dir, coco_test_dir)

    yolo_valid_dir = yolo_root / "valid"
    yolo_test_dir = yolo_root / "test"
    if yolo_valid_dir.is_dir() and not yolo_test_dir.exists():
        _symlink_tree(yolo_valid_dir, yolo_test_dir)


def _symlink_tree(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_dir.iterdir()):
        target_path = target_dir / source_path.name
        if target_path.exists():
            continue
        target_path.symlink_to(source_path.resolve())


def _resolve_split_dir(dataset_root: Path, split: str) -> Path:
    candidates = []
    if split == "val":
        candidates = [dataset_root / "val", dataset_root / "valid"]
    elif split == "valid":
        candidates = [dataset_root / "valid", dataset_root / "val"]
    elif split == "challenge":
        candidates = [dataset_root / "challenge"]
    else:
        candidates = [dataset_root / split]
    for candidate in candidates:
        if not candidate.exists():
            _extract_split_archive(dataset_root, candidate.name)
        if candidate.exists():
            return candidate
    return candidates[0]


def _extract_split_archive(dataset_root: Path, split_name: str) -> None:
    archive_path = dataset_root / f"{split_name}.zip"
    if not archive_path.is_file():
        return
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(dataset_root)


def _resolve_image_path(split_dir: Path, dataset_root: Path, file_name: str) -> Path:
    candidate = split_dir / file_name
    if candidate.exists():
        return candidate.resolve()
    candidate = dataset_root / file_name
    if candidate.exists():
        return candidate.resolve()
    return (split_dir / file_name).resolve()


def _link_or_copy(source: Path, destination: Path, use_symlink: bool) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        destination.symlink_to(source)
    else:
        destination.write_bytes(source.read_bytes())


def _extract_gamestate_detections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("predictions", "annotations", "detections"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_gamestate_bbox(detection: dict[str, Any]) -> list[float] | None:
    bbox = detection.get("bbox_image")
    if isinstance(bbox, dict):
        return [float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(v) for v in bbox]
    return None


def _extract_gamestate_pitch_position(detection: dict[str, Any]) -> list[float] | None:
    bbox_pitch = detection.get("bbox_pitch")
    if not isinstance(bbox_pitch, dict):
        return None
    return [
        float(bbox_pitch["x_bottom_middle"]),
        float(bbox_pitch["y_bottom_middle"]),
        0.0,
    ]


def _resolve_gamestate_dataset_root(root: Path, task: SynLocAuxiliaryTask | None) -> Path:
    if task is not None and root.name != task and (root / task).exists():
        return (root / task).resolve()
    return root.resolve()


def _resolve_gamestate_split_dir(dataset_root: Path, split: str) -> Path:
    candidates = [dataset_root / split]
    if split == "val":
        candidates = [dataset_root / "val", dataset_root / "valid"]
    elif split == "valid":
        candidates = [dataset_root / "valid", dataset_root / "val"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_gamestate_image_path(video_dir: Path, image_id: str) -> Path:
    stem_candidates = [image_id]
    if image_id.isdigit():
        stem_candidates.extend([image_id.zfill(6), image_id.zfill(8), image_id[-6:], image_id[-8:]])
    unique_stems = []
    for stem in stem_candidates:
        if stem and stem not in unique_stems:
            unique_stems.append(stem)
    directory_candidates = [video_dir / "img1", video_dir / "images", video_dir / "imgs", video_dir]
    extensions = (".jpg", ".jpeg", ".png")
    for directory in directory_candidates:
        for stem in unique_stems:
            for extension in extensions:
                candidate = directory / f"{stem}{extension}"
                if candidate.is_file():
                    return candidate.resolve()
    fallback = directory_candidates[0] / f"{unique_stems[0]}.jpg"
    return fallback.resolve()


def _read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        return 1920, 1080
    height, width = image.shape[:2]
    return int(width), int(height)


def _make_export_image_name(export_key: str, source_path: Path) -> str:
    safe_key = export_key.replace(":", "_").replace("/", "_")
    suffix = source_path.suffix or ".jpg"
    return f"{safe_key}{suffix}"
