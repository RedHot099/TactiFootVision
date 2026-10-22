"""SoccerNet-v3 ``Labels-v3.json`` (one game's action and replay frames) mapped to the 4 detection classes.

A label file holds ``actions`` (main-camera frames, ``"<n>.png"``) and ``replays`` (``"<n>_<k>.png"``). Each image
has ``imageMetadata`` (``localpath`` = SoccerNet game path, ``imageType``, ``width``, ``height``, ...) and
``bboxes`` with a ``class`` name and pixel ``points`` ``{x1, x2, y1, y2}``.

Mapping (``CLASS_MAP``): team players -> ``player``, team goalkeepers -> ``goalkeeper``, main and side referees ->
``referee``, ``Ball`` -> ``ball``. People without one of our roles (``IGNORE_CLASSES``: staff, a wall of players)
are returned as ignore regions, because dropping them would teach "person = background". Cards and the referee
flag are dropped. The ball is boxed only while in play, so a dead ball may be visible but unlabelled.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASS_NAMES = ("ball", "goalkeeper", "player", "referee")
CLASS_MAP = {
    "Ball": "ball",
    "Goalkeeper team left": "goalkeeper",
    "Goalkeeper team right": "goalkeeper",
    "Player team left": "player",
    "Player team right": "player",
    "Main referee": "referee",
    "Side referee": "referee",
}
IGNORE_CLASSES = frozenset({"Staff members", "Wall of players"})
DROPPED_CLASSES = frozenset({"Yellow card", "Red card", "Referee flag"})


@dataclass
class V3Image:
    """One labelled v3 frame: boxes in pixel ``xyxy`` with class names from :data:`CLASS_NAMES`."""

    game: str  # SoccerNet game path, "league/season/match"
    name: str  # file name inside the game's Frames-v3.zip, e.g. "3.png" or "3_1.png"
    image_type: str  # "action" (main camera) or "replay"
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) float32 xyxy
    classes: list[str]
    ignore_boxes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), np.float32)
    )
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.game}/{self.name}"


def read_labels(path: str | Path) -> list[V3Image]:
    """Every action and replay image of one ``Labels-v3.json``, sorted by name.

    Raises ``ValueError`` on a class name that is neither mapped, ignored nor dropped, so a new label never
    disappears silently.
    """
    labels = json.loads(Path(path).read_text())
    images = []
    for part in ("actions", "replays"):
        for name, entry in labels.get(part, {}).items():
            meta = entry["imageMetadata"]
            boxes, classes, ignore = [], [], []
            for box in entry.get("bboxes", []):
                label = box["class"]
                p = box["points"]
                xyxy = [
                    min(p["x1"], p["x2"]),
                    min(p["y1"], p["y2"]),
                    max(p["x1"], p["x2"]),
                    max(p["y1"], p["y2"]),
                ]
                if label in CLASS_MAP:
                    boxes.append(xyxy)
                    classes.append(CLASS_MAP[label])
                elif label in IGNORE_CLASSES:
                    ignore.append(xyxy)
                elif label not in DROPPED_CLASSES:
                    raise ValueError(f"{path}: {name} has unknown class {label!r}")
            images.append(
                V3Image(
                    game=meta["localpath"],
                    name=name,
                    image_type=meta.get("imageType", part[:-1]),
                    width=int(meta["width"]),
                    height=int(meta["height"]),
                    boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
                    classes=classes,
                    ignore_boxes=np.asarray(ignore, np.float32).reshape(-1, 4),
                    metadata=meta,
                )
            )
    return sorted(images, key=lambda image: image.name)
