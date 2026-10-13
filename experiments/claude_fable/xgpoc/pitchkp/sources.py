"""SoccerNet sources of line-annotated frames: GSR 2024 clips and Camera Calibration 2023 images.

GSR clips are read with ticket 02's reader (`xgpoc.pitchkp.gsr_eval.read_clip`), the single GSR reader: its frame rule
(frames 1, 6, 11, ...), its persons (every object but the ball) and its centre -> corner conversion (x + 52.5, y + 34).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xgpoc.pitchkp import gsr_eval as ge
from xgpoc.pitchkp.fit import FRAME_W


@dataclass
class Frame:
    """One line-annotated image."""

    source: str                 # "gsr" or "calib"
    split: str                  # the source's own split: train / valid / test
    group: str                  # GSR clip (SNGS-xxx) or Calibration game: frames of a group stay in one split
    image: Path
    lines: dict                 # SoccerNet {name: [{"x": fx, "y": fy}, ...]} or {name: (N, 2) fractions of the image}
    frame: int = 0              # GSR 1-based frame number
    game: str = ""              # GSR info game_id (an index, not a named match); Calibration league/season/date teams
    # GSR persons on the pitch: bottom-middle of bbox_image (720p px) and of bbox_pitch (corner metres)
    feet_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    feet_pitch: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def key(self) -> str:
        if self.source == "gsr":
            return f"gsr/{self.split}/{self.group}/{self.frame:06d}"
        return f"calib/{self.split}/{self.image.stem}"


def gsr_clip(clip_dir: Path, split: str) -> Iterator[Frame]:
    """The kept frames (1, 6, 11, ...) of one GSR clip, in frame order, with their lines and on-pitch persons."""
    clip = ge.read_clip(clip_dir)
    game = str(json.loads((clip_dir / "Labels-GameState.json").read_text())["info"].get("game_id", ""))
    persons = clip.persons[ge.on_pitch(clip.persons)]
    for row in clip.frames.itertuples(index=False):
        mine = persons[persons.frame == row.frame]
        yield Frame(source="gsr", split=split, group=clip.name, image=Path(row.path), lines=clip.lines.get(row.frame, {}),
                    frame=int(row.frame), game=game, feet_image=ge.feet(mine, (FRAME_W / row.width,) * 2),
                    feet_pitch=mine[["pitch_x", "pitch_y"]].to_numpy(dtype=float).reshape(-1, 2))


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
