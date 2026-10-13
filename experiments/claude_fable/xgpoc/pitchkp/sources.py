"""SoccerNet sources of line-annotated frames: GSR 2024 clips and Camera Calibration 2023 images.

GSR pitch positions (`bbox_pitch`) are metres with the origin at the centre spot, x along the length (left goal at
x = -52.5) and y across with the far touchline ("Side line top") at y = -34; the package's corner origin is
(x + 52.5, y + 34) with the same axis directions (checked in `scripts/pk_build_dataset.py`).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xgpoc.pitchkp.fit import FRAME_W

FRAME_STEP = 5  # GSR: frames 1, 6, 11, ... of each 750-frame clip are kept and used
GSR_HALF = (52.5, 34.0)
ON_PITCH_MARGIN = 3.0         # persons within 3 m of the pitch count (|x| <= 55.5, |y| <= 37 in GSR coordinates)
PERSONS = (1, 2, 3)           # GSR categories player, goalkeeper, referee


def gsr_keeps_frame(frame: int) -> bool:
    """Whether 1-based GSR frame number `frame` is in the every-5th-frame subsample (1, 6, 11, ...)."""
    return (frame - 1) % FRAME_STEP == 0


def gsr_to_pitch(xy) -> np.ndarray:
    """GSR centre-origin metres -> the package's corner-origin metres (axes kept)."""
    return np.asarray(xy, dtype=float) + GSR_HALF


@dataclass
class Frame:
    """One line-annotated image."""

    source: str                 # "gsr" or "calib"
    split: str                  # the source's own split: train / valid / test
    group: str                  # GSR clip (SNGS-xxx) or Calibration game: frames of a group stay in one split
    image: Path
    lines: dict                 # SoccerNet {name: [{"x": fx, "y": fy}, ...]}
    frame: int = 0              # GSR 1-based frame number
    game: str = ""              # GSR info game_id (an index, not a named match); Calibration league/season/date teams
    # GSR persons on the pitch: bottom-middle of bbox_image (720p px) and bbox_pitch / bbox_pitch_raw (corner metres)
    feet_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    feet_pitch: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    feet_pitch_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def key(self) -> str:
        if self.source == "gsr":
            return f"gsr/{self.split}/{self.group}/{self.frame:06d}"
        return f"calib/{self.split}/{self.image.stem}"


def _bottom_middle(box: dict) -> tuple[float, float]:
    return box["x_bottom_middle"], box["y_bottom_middle"]


def gsr_clip(clip_dir: Path, split: str, every: int = FRAME_STEP) -> Iterator[Frame]:
    """The labelled frames of one GSR clip kept by the subsample, in frame order (needs the clip's label file)."""
    data = json.loads((clip_dir / "Labels-GameState.json").read_text())
    info = data["info"]
    pitch = {a["image_id"]: a["lines"] for a in data["annotations"] if a["supercategory"] == "pitch"}
    persons: dict[str, list] = {}
    for a in data["annotations"]:
        if a["category_id"] in PERSONS and a.get("bbox_pitch"):
            persons.setdefault(a["image_id"], []).append(a)
    for image in data["images"]:
        number = int(Path(image["file_name"]).stem)
        if (number - 1) % every or not image.get("is_labeled") or image["image_id"] not in pitch:
            continue
        scale = FRAME_W / image["width"]
        feet, positions, raw = [], [], []
        for a in persons.get(image["image_id"], []):
            x, y = _bottom_middle(a["bbox_pitch"])
            if abs(x) > GSR_HALF[0] + ON_PITCH_MARGIN or abs(y) > GSR_HALF[1] + ON_PITCH_MARGIN:
                continue
            box = a["bbox_image"]
            feet.append(((box["x"] + box["w"] / 2) * scale, (box["y"] + box["h"]) * scale))
            positions.append((x, y))
            raw.append(_bottom_middle(a["bbox_pitch_raw"]) if a.get("bbox_pitch_raw") else (np.nan, np.nan))
        yield Frame(
            source="gsr", split=split, group=clip_dir.name, image=clip_dir / info.get("im_dir", "img1") /
            image["file_name"], lines=pitch[image["image_id"]], frame=number, game=str(info.get("game_id", "")),
            feet_image=np.array(feet, dtype=float).reshape(-1, 2),
            feet_pitch=gsr_to_pitch(np.array(positions, dtype=float).reshape(-1, 2)),
            feet_pitch_raw=gsr_to_pitch(np.array(raw, dtype=float).reshape(-1, 2)),
        )


def gsr_clips(root: Path, split: str) -> list[Path]:
    """Clip folders of a GSR split that have a label file."""
    return sorted(p.parent for p in (root / split).glob("SNGS-*/Labels-GameState.json"))


def calibration_games(root: Path, split: str) -> dict[str, list[str]]:
    """Calibration-2023 images of a split grouped by game: {game: [image stem, ...]}, from the split's `match_info.json`.

    The game is `league/season/20yy-mm-dd home - away` (team names as SoccerNet writes them).
    """
    info = json.loads((root / split / "match_info.json").read_text())
    games: dict[str, list[str]] = {}
    for image, m in sorted(info.items()):
        game = f"{m['league']}/{m['season']}/20{m['date']} {m['match'].strip()}"
        games.setdefault(game, []).append(Path(image).stem)
    return games


def calibration_images(root: Path, split: str, game: str, stems: list[str]) -> Iterator[Frame]:
    """Calibration-2023 images `<stem>.jpg` with `<stem>.json` ({line name: [{"x", "y"}, ...]}) of one game."""
    for stem in stems:
        label = root / split / f"{stem}.json"
        image = label.with_suffix(".jpg")
        if label.exists() and image.exists():
            yield Frame(source="calib", split=split, group=game, image=image, lines=json.loads(label.read_text()),
                        game=game)
