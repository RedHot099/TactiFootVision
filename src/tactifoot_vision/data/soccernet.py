"""SoccerNet readers: a Tracking or a Game State Reconstruction sequence folder as a labelled clip."""

import configparser
import json
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from tactifoot_vision.data.labelled_clip import (
    UNKNOWN_TEAM,
    LabelledClip,
    LabelledFrame,
)
from tactifoot_vision.enums import DetectionKey, ObjectKind, Role, Truth
from tactifoot_vision.pipeline.result import NO_TEAM
from tactifoot_vision.pitch.pitch import SoccerPitch

# First word of a gameinfo.ini tracklet label -> package class name. "other"
# objects (none of the package classes) are left out of the clip.
_TRACKLET_CLASSES = {
    "player": Role.PLAYER,
    "goalkeeper": Role.GOALKEEPER,
    "goalkeepers": Role.GOALKEEPER,
    "referee": Role.REFEREE,
    "ball": ObjectKind.BALL,
    "other": None,
}
# GSR attributes.role -> package class name; "other" is left out likewise.
_GSR_ROLES = {
    "player": Role.PLAYER,
    "goalkeeper": Role.GOALKEEPER,
    "referee": Role.REFEREE,
    "ball": ObjectKind.BALL,
    "other": None,
}
# SoccerNet's team label (the team on the left or right of the main camera's
# picture) -> team id. A label names a team within one clip only;
# evaluate_teams maps team ids per clip anyway.
_SIDES = {"left": 0, "right": 1}
_PITCH = SoccerPitch()  # GSR pitches are 105 x 68 m, as SoccerPitch's default
# GSR pitch metres have their origin at the centre spot; SoccerPitch's is the
# corner where x and y are least. Both run x goal to goal and y towards the
# main camera's touchline, so a shift converts them.
_CENTRE_TO_CORNER = np.array(
    [[1.0, 0.0, _PITCH.length / 2], [0.0, 1.0, _PITCH.width / 2], [0.0, 0.0, 1.0]]
)


def read_tracking(folder: Path) -> LabelledClip:
    """Read a SoccerNet Tracking sequence folder (MOT format).

    ``gt/gt.txt`` rows are ``frame, track id, x, y, w, h, ...`` with 1-based
    frames: MOT frame ``n`` is image ``img1/<n:06d>.jpg``, which the pipeline
    reads as frame ``n - 1``. ``gameinfo.ini`` names each track's class and,
    for players and goalkeepers, their team (``player team left;7``):
    ``left`` is team 0 and ``right`` team 1, a label without a team is
    ``UNKNOWN_TEAM`` and referees have ``NO_TEAM``. The clip has teams when
    some label names one.
    """
    info = _read_ini(folder / "seqinfo.ini")
    try:
        fps = float(info["frameRate"])
        length = int(info["seqLength"])
        width, height = int(info["imWidth"]), int(info["imHeight"])
    except (KeyError, ValueError):
        raise ValueError(
            f"{folder / 'seqinfo.ini'}: [Sequence] needs frameRate, seqLength, "
            "imWidth and imHeight"
        ) from None
    tracklets = _tracklets(folder / "gameinfo.ini")
    rows = _read_gt(folder / "gt" / "gt.txt")
    frame_numbers, track_ids = rows[:, 0].astype(int), rows[:, 1].astype(int)
    unknown = sorted(set(track_ids) - set(tracklets))
    if unknown:
        raise ValueError(
            f"{folder / 'gt' / 'gt.txt'} has boxes of track {unknown[0]}, which "
            f"{folder / 'gameinfo.ini'} does not list"
        )
    outside = frame_numbers[(frame_numbers < 1) | (frame_numbers > length)]
    if len(outside):
        raise ValueError(
            f"{folder / 'gt' / 'gt.txt'} has a box in frame {outside[0]}, outside "
            f"the sequence's frames 1..{length} (seqLength={length})"
        )

    names = np.array([tracklets[t][0] or "" for t in track_ids], dtype=object)
    teams = np.array([tracklets[t][1] for t in track_ids], dtype=int)
    xyxy = rows[:, 2:6].copy()
    xyxy[:, 2:] += xyxy[:, :2]
    kept = names != ""
    frames = []
    for index in range(length):
        here = kept & (frame_numbers == index + 1)
        ball = here & (names == ObjectKind.BALL)
        people = here & (names != ObjectKind.BALL)
        frames.append(
            LabelledFrame(
                index=index,
                detections=_detections(
                    xyxy[people],
                    track_ids[people],
                    names[people],
                    {DetectionKey.TEAM_ID: teams[people]},
                ),
                ball=_detections(xyxy[ball], track_ids[ball], names[ball]),
            )
        )
    image_dir = info.get("imDir", "img1")
    extension = info.get("imExt", ".jpg")
    # OpenCV (FFmpeg) reads an image sequence at 25 fps whatever frameRate
    # says; SoccerNet sequences are 25 fps, so the two agree.
    return LabelledClip(
        name=info.get("name", folder.name),
        video_path=folder / image_dir / f"%06d{extension}",
        fps=fps,
        width=width,
        height=height,
        frames=tuple(frames),
        truth=frozenset(
            {Truth.IDENTITIES, Truth.BOXES, Truth.BALL}
            # Teams when some tracklet label names one.
            | ({Truth.TEAMS} if any(t >= 0 for _, t in tracklets.values()) else set())
        ),
    )


def read_gamestate(path: Path) -> LabelledClip:
    """Read a SoccerNet Game State Reconstruction sequence's ``Labels-GameState.json``.

    Each image ``<n:06d>.jpg`` of ``info.im_dir`` is the pipeline's frame
    ``n - 1``. Object annotations give the boxes (``bbox_image`` centre and
    size, frame pixels), track ids and ``attributes``: the role is the class
    name, the team ``left`` is team 0 and ``right`` team 1, a player or
    goalkeeper with a null team has ``UNKNOWN_TEAM``, referees ``NO_TEAM``,
    and the jersey number is ``data["jersey"]`` (``""`` when null). A
    person's ``bbox_pitch`` bottom middle, in GSR's centre-origin metres, is
    shifted into ``SoccerPitch`` coordinates in ``data["pitch_xy"]`` (NaN when
    null); the ball keeps its box only. A camera annotation becomes the
    frame's homography: the pinhole part of its parameters (distortion is
    left out, as in SoccerNet's own ``Camera.to_homography``), inverted and
    shifted likewise. Frames without one have none. The annotations of an
    image with ``is_labeled`` false are left out, so its frame is empty and
    uncalibrated (evaluators still score it as a frame with nobody in it).
    The clip has teams,
    pitch positions or calibration when some person or frame has them.
    """
    labels = _read_json(path)
    try:
        info, images = labels["info"], labels["images"]
        fps, length = float(info["frame_rate"]), int(info["seq_length"])
        width, height = int(images[0]["width"]), int(images[0]["height"])
        extension = Path(images[0]["file_name"]).suffix
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError(
            f"{path} needs info.frame_rate, info.seq_length and images with "
            "file_name, width and height"
        ) from None
    index_of = _gsr_frame_indices(path, images, length)
    unlabelled = {
        str(i.get("image_id")) for i in images if i.get("is_labeled") is False
    }
    people: list[list[dict]] = [[] for _ in range(length)]
    balls: list[list[dict]] = [[] for _ in range(length)]
    homographies: dict[int, np.ndarray] = {}
    for annotation in labels.get("annotations", []):
        image_id = str(annotation.get("image_id"))
        if image_id not in index_of:
            raise ValueError(
                f"{path}: annotation {annotation.get('id')} is of image {image_id}, "
                "which is not in its images"
            )
        if image_id in unlabelled:
            continue
        index = index_of[image_id]
        try:
            if annotation.get("supercategory") == "camera":
                homographies[index] = _camera_homography(annotation["parameters"])
            elif annotation.get("supercategory") == "object":
                row = _gsr_object(annotation)
                if row is not None:
                    group = balls if row["class_name"] == ObjectKind.BALL else people
                    group[index].append(row)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(
                f"{path}: annotation {annotation.get('id')} of image {image_id} "
                f"is not a GSR {annotation.get('supercategory')} annotation: {error}"
            ) from None

    frames = tuple(
        LabelledFrame(
            index=index,
            detections=_gsr_people(people[index]),
            ball=_detections(
                np.array([row["xyxy"] for row in balls[index]], float),
                np.array([row["track_id"] for row in balls[index]], int),
                np.array([ObjectKind.BALL] * len(balls[index]), dtype=object),
            ),
            homography=homographies.get(index),
        )
        for index in range(length)
    )
    everyone = [row for rows in people for row in rows]
    truth = {Truth.IDENTITIES, Truth.BOXES, Truth.BALL}
    if any(row["team_id"] >= 0 for row in everyone):  # some person has a team
        truth.add(Truth.TEAMS)
    if any(np.isfinite(row["pitch_xy"]).all() for row in everyone):
        truth.add(Truth.PITCH_POSITIONS)
    if homographies:
        truth.add(Truth.CALIBRATION)
    return LabelledClip(
        name=info.get("name", path.parent.name),
        video_path=path.parent / info.get("im_dir", "img1") / f"%06d{extension}",
        fps=fps,
        width=width,
        height=height,
        frames=frames,
        truth=frozenset(truth),
    )


def _read_json(path: Path) -> dict:
    try:
        labels = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not a JSON file: {error}") from None
    if not isinstance(labels, dict):
        raise ValueError(f"{path} is not a GSR labels file (no top-level object)")
    return labels


def _gsr_frame_indices(path: Path, images: list[dict], length: int) -> dict[str, int]:
    """Image id -> frame index: image ``<n:06d>.jpg`` is frame ``n - 1``."""
    index_of = {}
    for image in images:
        name = str(image.get("file_name"))
        stem = Path(name).stem
        if not stem.isdigit() or not 1 <= int(stem) <= length:
            raise ValueError(
                f"{path}: image {name} is not one of the sequence's images "
                f"000001..{length:06d} (seq_length={length})"
            )
        index_of[str(image.get("image_id"))] = int(stem) - 1
    return index_of


def _gsr_object(annotation: dict[str, Any]) -> dict[str, Any] | None:
    """One object annotation in the package conventions; ``None`` for an ``other``."""
    attributes = annotation["attributes"]
    role = attributes["role"]
    if role not in _GSR_ROLES:
        raise ValueError(f"unknown role {role!r}; known: {', '.join(_GSR_ROLES)}")
    class_name = _GSR_ROLES[role]
    if class_name is None:
        return None
    box = annotation["bbox_image"]
    x, y = float(box["x_center"]), float(box["y_center"])
    w, h = float(box["w"]), float(box["h"])
    row = {
        "xyxy": (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
        "track_id": int(annotation["track_id"]),
        "class_name": class_name,
    }
    if class_name == ObjectKind.BALL:
        return row
    team = attributes.get("team")
    if team is not None and team not in _SIDES:
        raise ValueError(f"unknown team {team!r}; known: {', '.join(_SIDES)}, null")
    foot = annotation.get("bbox_pitch")
    pitch_xy = (np.nan, np.nan)
    if foot is not None:
        centred = (float(foot["x_bottom_middle"]), float(foot["y_bottom_middle"]), 1.0)
        pitch_xy = tuple(_CENTRE_TO_CORNER[:2] @ centred)
    return {
        **row,
        "team_id": _team_id(class_name, team),
        "jersey": "" if attributes.get("jersey") is None else str(attributes["jersey"]),
        "pitch_xy": pitch_xy,
    }


def _team_id(class_name: str | None, team: str | None) -> int:
    """A person's team id from their team label; referees have none."""
    if class_name == Role.REFEREE:
        return NO_TEAM
    return _SIDES[team] if team is not None else UNKNOWN_TEAM


def _gsr_people(rows: list[dict]) -> sv.Detections:
    return _detections(
        np.array([row["xyxy"] for row in rows], float),
        np.array([row["track_id"] for row in rows], int),
        np.array([row["class_name"] for row in rows], dtype=object),
        {
            DetectionKey.TEAM_ID: np.array([row["team_id"] for row in rows], int),
            DetectionKey.JERSEY: np.array(
                [row["jersey"] for row in rows], dtype=object
            ).astype(str),
            DetectionKey.PITCH_XY: np.array(
                [row["pitch_xy"] for row in rows], float
            ).reshape(-1, 2),
        },
    )


def _camera_homography(parameters: dict[str, Any]) -> np.ndarray:
    """The frame -> ``SoccerPitch`` homography of a GSR (SoccerNet calibration) camera.

    As SoccerNet's ``Camera``: the camera at ``position_meters`` (z is
    negative above the pitch) turned by pan, tilt and roll in degrees, with
    focal lengths and principal point in pixels.
    """
    fx, fy = float(parameters["x_focal_length"]), float(parameters["y_focal_length"])
    px, py = (float(v) for v in parameters["principal_point"])
    pan, tilt, roll = (
        np.radians(float(parameters[f"{angle}_degrees"]))
        for angle in ("pan", "tilt", "roll")
    )
    position = np.array(parameters["position_meters"], float).reshape(3)
    calibration = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]])
    rotation = (_rz(pan) @ _rx(tilt) @ _rz(roll)).T  # world -> camera
    pitch_to_frame = (
        calibration @ rotation @ np.column_stack([np.eye(3)[:, :2], -position])
    )
    homography = _CENTRE_TO_CORNER @ np.linalg.inv(pitch_to_frame)
    # Not divided by its last entry, which is 0 when pixel (0, 0) is on the horizon.
    return homography / np.linalg.norm(homography)


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _read_ini(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(path.read_text(), source=str(path))
    except configparser.Error as error:
        raise ValueError(f"{path} is not a valid ini file: {error}") from None
    if not parser.has_section("Sequence"):
        raise ValueError(f"{path} has no [Sequence] section")
    return parser["Sequence"]


def _tracklets(path: Path) -> dict[int, tuple[str | None, int]]:
    """Track id -> package class name (``None``: left out) and team id from ``trackletID_<id>`` keys."""
    tracklets = {}
    for key, value in _read_ini(path).items():
        if not key.startswith("trackletid_"):
            continue
        number = key.removeprefix("trackletid_")
        if not number.isdigit():
            raise ValueError(f"{path}: {key} does not end in a track id")
        label = value.split(";")[0].strip()
        words = label.lower().split()
        word = words[0] if words else ""
        if word not in _TRACKLET_CLASSES:
            raise ValueError(
                f"{path}: {key} has the unknown class {label!r}; known: "
                f"{', '.join(_TRACKLET_CLASSES)}"
            )
        # "player team left": the team follows "team", when the label has one.
        team = words[2] if words[1:2] == ["team"] and len(words) > 2 else None
        if team is not None and team not in _SIDES:
            raise ValueError(
                f"{path}: {key} has the unknown team {team!r}; known: "
                f"{', '.join(_SIDES)}"
            )
        name = _TRACKLET_CLASSES[word]
        tracklets[int(number)] = (name, _team_id(name, team))
    return tracklets


def _read_gt(path: Path) -> np.ndarray:
    """``(N, 6)`` float rows ``frame, track id, x, y, w, h``."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return np.zeros((0, 6))
    try:
        return np.loadtxt(lines, delimiter=",", usecols=range(6), ndmin=2)
    except ValueError as error:
        raise ValueError(f"{path} is not a MOT ground-truth file: {error}") from None


def _detections(
    xyxy: np.ndarray,
    track_ids: np.ndarray,
    names: np.ndarray,
    data: dict[str, np.ndarray] | None = None,
) -> sv.Detections:
    return sv.Detections(
        xyxy=xyxy.reshape(-1, 4),
        tracker_id=track_ids.astype(int),
        data={DetectionKey.CLASS_NAME: names.astype(str), **(data or {})},
    )
