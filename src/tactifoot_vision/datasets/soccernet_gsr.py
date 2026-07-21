import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS_FILE = "Labels-GameState.json"


@dataclass(frozen=True, slots=True)
class GsrImageBBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def bottom_middle(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h


@dataclass(frozen=True, slots=True)
class GsrPitchBBox:
    x_bottom_middle: float
    y_bottom_middle: float
    x_bottom_left: float | None = None
    y_bottom_left: float | None = None
    x_bottom_right: float | None = None
    y_bottom_right: float | None = None

    @property
    def bottom_middle(self) -> tuple[float, float]:
        return self.x_bottom_middle, self.y_bottom_middle


@dataclass(frozen=True, slots=True)
class GsrFrame:
    image_id: str
    frame: int
    file_name: str | None = None
    width: int | None = None
    height: int | None = None
    has_labeled_pitch: bool = False
    has_labeled_camera: bool = False
    has_labeled_person: bool = False


@dataclass(frozen=True, slots=True)
class GsrAthleteAnnotation:
    annotation_id: str | None
    image_id: str
    frame: int
    track_id: int
    role: str | None
    jersey: str | None
    team: str | None
    bbox_image: GsrImageBBox | None
    bbox_pitch: GsrPitchBBox | None

    @property
    def image_bottom_middle(self) -> tuple[float, float] | None:
        if self.bbox_image is None:
            return None
        return self.bbox_image.bottom_middle

    @property
    def pitch_bottom_middle(self) -> tuple[float, float] | None:
        if self.bbox_pitch is None:
            return None
        return self.bbox_pitch.bottom_middle


@dataclass(frozen=True, slots=True)
class GsrLinePoint:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class GsrLineAnnotation:
    image_id: str
    frame: int
    line_name: str
    points: tuple[GsrLinePoint, ...]


@dataclass(frozen=True, slots=True)
class SoccerNetGsrLabels:
    sequence: str
    version: str
    frames: tuple[GsrFrame, ...]
    athletes: tuple[GsrAthleteAnnotation, ...]
    lines: tuple[GsrLineAnnotation, ...]

    def athletes_for_frame(self, frame: int) -> tuple[GsrAthleteAnnotation, ...]:
        return tuple(
            annotation for annotation in self.athletes if annotation.frame == frame
        )

    def lines_for_frame(self, frame: int) -> tuple[GsrLineAnnotation, ...]:
        return tuple(
            annotation for annotation in self.lines if annotation.frame == frame
        )


class SoccerNetGsrDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def iter_sequences(self, split: str | None = None) -> list[Path]:
        return iter_gsr_sequence_dirs(self.root, split=split)

    def load_sequence(self, sequence_dir: str | Path) -> SoccerNetGsrLabels:
        return read_gsr_labels(Path(sequence_dir))


def iter_gsr_sequence_dirs(root: str | Path, split: str | None = None) -> list[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")
    if split is not None:
        root_path = root_path / split
        if not root_path.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {root_path}")
    if (root_path / LABELS_FILE).is_file():
        return [root_path]
    sequence_dirs: list[Path] = []
    for child in sorted(root_path.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if (child / LABELS_FILE).is_file():
            sequence_dirs.append(child)
            continue
        if split is None and child.name in {"train", "valid", "test", "challenge"}:
            sequence_dirs.extend(iter_gsr_sequence_dirs(child))
    return sequence_dirs


def read_gsr_labels(
    path: str | Path, *, min_version: str | None = "1.3"
) -> SoccerNetGsrLabels:
    labels_path = _labels_path(Path(path))
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    version = str(payload.get("info", {}).get("version", "0"))
    if min_version is not None and _version_tuple(version) < _version_tuple(
        min_version
    ):
        raise ValueError(
            f"SoccerNet-GSR labels must be >= {min_version}; found {version}"
        )
    frames = _parse_frames(payload)
    frames_by_image_id = {frame.image_id: frame for frame in frames}
    categories = _categories_by_id(payload)
    athletes: list[GsrAthleteAnnotation] = []
    lines: list[GsrLineAnnotation] = []
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, Mapping):
            continue
        if _is_object_annotation(annotation):
            athletes.append(_parse_athlete(annotation, frames_by_image_id))
        else:
            lines.extend(
                _parse_lines_from_annotation(annotation, frames_by_image_id, categories)
            )
    lines.extend(_parse_top_level_lines(payload, frames_by_image_id))
    return SoccerNetGsrLabels(
        sequence=labels_path.parent.name,
        version=version,
        frames=frames,
        athletes=tuple(athletes),
        lines=tuple(lines),
    )


def _labels_path(path: Path) -> Path:
    labels_path = path / LABELS_FILE if path.is_dir() else path
    if not labels_path.is_file():
        raise FileNotFoundError(f"SoccerNet-GSR labels not found: {labels_path}")
    return labels_path


def _parse_frames(payload: Mapping[str, Any]) -> tuple[GsrFrame, ...]:
    frames: list[GsrFrame] = []
    for image in payload.get("images", []):
        if not isinstance(image, Mapping):
            continue
        image_id = str(image.get("image_id", image.get("id", "")))
        if not image_id:
            continue
        frames.append(
            GsrFrame(
                image_id=image_id,
                frame=_frame_number(image),
                file_name=_optional_str(image.get("file_name")),
                width=_optional_int(image.get("width")),
                height=_optional_int(image.get("height")),
                has_labeled_pitch=bool(image.get("has_labeled_pitch", False)),
                has_labeled_camera=bool(image.get("has_labeled_camera", False)),
                has_labeled_person=bool(image.get("has_labeled_person", False)),
            )
        )
    return tuple(sorted(frames, key=lambda frame: frame.frame))


def _parse_athlete(
    annotation: Mapping[str, Any], frames_by_image_id: Mapping[str, GsrFrame]
) -> GsrAthleteAnnotation:
    image_id = str(annotation["image_id"])
    attributes = annotation.get("attributes", {})
    if not isinstance(attributes, Mapping):
        attributes = {}
    return GsrAthleteAnnotation(
        annotation_id=_optional_str(annotation.get("id")),
        image_id=image_id,
        frame=_frame_for_image_id(image_id, frames_by_image_id),
        track_id=int(annotation.get("track_id", -1)),
        role=_optional_str(attributes.get("role")),
        jersey=_optional_str(attributes.get("jersey")),
        team=_optional_str(attributes.get("team")),
        bbox_image=_parse_image_bbox(annotation.get("bbox_image")),
        bbox_pitch=_parse_pitch_bbox(annotation.get("bbox_pitch")),
    )


def _parse_image_bbox(value: object) -> GsrImageBBox | None:
    if not isinstance(value, Mapping):
        return None
    try:
        if "x_center" in value and "y_center" in value:
            width = float(value["w"])
            height = float(value["h"])
            return GsrImageBBox(
                x=float(value["x_center"]) - width / 2.0,
                y=float(value["y_center"]) - height / 2.0,
                w=width,
                h=height,
            )
        return GsrImageBBox(
            x=float(value["x"]),
            y=float(value["y"]),
            w=float(value["w"]),
            h=float(value["h"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_lines_from_annotation(
    annotation: Mapping[str, Any],
    frames_by_image_id: Mapping[str, GsrFrame],
    categories: Mapping[int, str],
) -> tuple[GsrLineAnnotation, ...]:
    lines_map = annotation.get("lines")
    if isinstance(lines_map, Mapping):
        image_id = _optional_str(annotation.get("image_id"))
        if image_id is None:
            return ()
        return tuple(
            _parse_line_mapping_for_image(image_id, lines_map, frames_by_image_id)
        )
    line = _parse_line_annotation(annotation, frames_by_image_id, categories)
    return () if line is None else (line,)


def _parse_pitch_bbox(value: object) -> GsrPitchBBox | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return GsrPitchBBox(
            x_bottom_middle=float(value["x_bottom_middle"]),
            y_bottom_middle=float(value["y_bottom_middle"]),
            x_bottom_left=_optional_float(value.get("x_bottom_left")),
            y_bottom_left=_optional_float(value.get("y_bottom_left")),
            x_bottom_right=_optional_float(value.get("x_bottom_right")),
            y_bottom_right=_optional_float(value.get("y_bottom_right")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_line_annotation(
    annotation: Mapping[str, Any],
    frames_by_image_id: Mapping[str, GsrFrame],
    categories: Mapping[int, str],
) -> GsrLineAnnotation | None:
    image_id = _optional_str(annotation.get("image_id"))
    if image_id is None:
        return None
    points = _parse_line_points(
        annotation.get("points")
        or annotation.get("polyline")
        or annotation.get("line")
        or annotation.get("keypoints")
    )
    if len(points) < 2:
        return None
    line_name = _line_name(annotation, categories)
    if line_name is None:
        return None
    return GsrLineAnnotation(
        image_id=image_id,
        frame=_frame_for_image_id(image_id, frames_by_image_id),
        line_name=line_name,
        points=points,
    )


def _parse_top_level_lines(
    payload: Mapping[str, Any], frames_by_image_id: Mapping[str, GsrFrame]
) -> list[GsrLineAnnotation]:
    output: list[GsrLineAnnotation] = []
    for key in ("line_annotations", "lines", "pitch_lines", "camera_lines"):
        container = payload.get(key)
        if isinstance(container, Sequence) and not isinstance(container, str):
            for item in container:
                if isinstance(item, Mapping):
                    line = _parse_line_annotation(item, frames_by_image_id, {})
                    if line is not None:
                        output.append(line)
        elif isinstance(container, Mapping):
            output.extend(_parse_line_mapping(container, frames_by_image_id))
    return output


def _parse_line_mapping(
    container: Mapping[str, Any], frames_by_image_id: Mapping[str, GsrFrame]
) -> list[GsrLineAnnotation]:
    output: list[GsrLineAnnotation] = []
    for image_id, lines_by_name in container.items():
        if not isinstance(lines_by_name, Mapping):
            continue
        output.extend(
            _parse_line_mapping_for_image(
                str(image_id), lines_by_name, frames_by_image_id
            )
        )
    return output


def _parse_line_mapping_for_image(
    image_id: str,
    lines_by_name: Mapping[str, Any],
    frames_by_image_id: Mapping[str, GsrFrame],
) -> list[GsrLineAnnotation]:
    output: list[GsrLineAnnotation] = []
    for line_name, raw_points in lines_by_name.items():
        points = _parse_line_points(raw_points)
        if len(points) < 2:
            continue
        output.append(
            GsrLineAnnotation(
                image_id=image_id,
                frame=_frame_for_image_id(image_id, frames_by_image_id),
                line_name=str(line_name),
                points=points,
            )
        )
    return output


def _parse_line_points(value: object) -> tuple[GsrLinePoint, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    points: list[GsrLinePoint] = []
    for point in value:
        parsed = _parse_line_point(point)
        if parsed is not None:
            points.append(parsed)
    return tuple(points)


def _parse_line_point(value: object) -> GsrLinePoint | None:
    try:
        if isinstance(value, Mapping):
            return GsrLinePoint(float(value["x"]), float(value["y"]))
        if isinstance(value, Sequence) and not isinstance(value, str):
            return GsrLinePoint(float(value[0]), float(value[1]))
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return None


def _is_object_annotation(annotation: Mapping[str, Any]) -> bool:
    return (
        annotation.get("supercategory") == "object"
        or "bbox_pitch" in annotation
        or "bbox_image" in annotation
    )


def _categories_by_id(payload: Mapping[str, Any]) -> dict[int, str]:
    categories: dict[int, str] = {}
    for category in payload.get("categories", []):
        if not isinstance(category, Mapping):
            continue
        try:
            categories[int(category["id"])] = str(category["name"])
        except (KeyError, TypeError, ValueError):
            continue
    return categories


def _line_name(
    annotation: Mapping[str, Any], categories: Mapping[int, str]
) -> str | None:
    explicit_name = (
        _optional_str(annotation.get("line_name"))
        or _optional_str(annotation.get("name"))
        or _optional_str(annotation.get("label"))
    )
    if explicit_name is not None:
        return explicit_name
    try:
        return categories.get(int(annotation.get("category_id", -1)))
    except (TypeError, ValueError):
        return None


def _frame_for_image_id(
    image_id: str, frames_by_image_id: Mapping[str, GsrFrame]
) -> int:
    frame = frames_by_image_id.get(image_id)
    if frame is not None:
        return frame.frame
    return _frame_number({"image_id": image_id})


def _frame_number(image: Mapping[str, Any]) -> int:
    for key in ("frame", "frame_id", "frame_index"):
        value = _optional_int(image.get(key))
        if value is not None:
            return value
    file_name = _optional_str(image.get("file_name"))
    if file_name is not None:
        stem = Path(file_name).stem
        if stem.isdigit():
            return int(stem)
    image_id = str(image.get("image_id", image.get("id", "")))
    digits = "".join(char for char in image_id if char.isdigit())
    if len(digits) >= 6:
        return int(digits[-6:])
    return int(digits) if digits else 0


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        digits = "".join(char for char in part if char.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
