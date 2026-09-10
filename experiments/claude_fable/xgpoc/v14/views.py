"""Local views (shooter-ball, goalkeeper) cut from the t0 frame for the multi-view G.

Boxes are the t0 detections cached in cache/full_v1/positional/tokens.npz:
[x0, y0, x1, y1, ball, goalkeeper, player, referee, valid], normalised to the frame.
Crops are squares in normalised coordinates, so in pixels they keep the 16:9 aspect of
the frame and are resized to LOCAL_SIZE without distortion.

The shooter is a heuristic: the outfield player nearest the ball, accepted only when
that distance is small. The keeper view is the detected goalkeeper alone; no source in
this repo gives the goal position in the image, so a goalkeeper-goal view cannot be
built honestly. A missing ball, an unresolved shooter or a missing goalkeeper yield a
masked view, never a silent fallback to the whole frame. Control views have the same
size and resolution but sit in the opposite corner of the frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from xgpoc.data.datasets import FILL, check_frame_col, overlay_mask, random_crop_box, to_tensor
from xgpoc.v13.inputs import _bounded_square, _control_box, normalized_crop_pixels

FULL_SIZE = (448, 252)
LOCAL_SIZE = (224, 126)      # 16:9, multiples of the 14 px patch: 16 x 9 tokens
VIEWS = ("shooter", "keeper")
MIN_SIDE = 0.25              # normalised; 320 x 180 px in a 720p frame
MARGIN = 0.05
MAX_SHOOTER_DISTANCE = 0.15  # normalised ball-to-player centre distance beyond which nobody is the shooter
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewBoxes:
    boxes: tuple[Box | None, ...]
    controls: tuple[Box | None, ...]
    shooter_distance: float | None

    @property
    def valid(self) -> tuple[float, ...]:
        return tuple(float(b is not None) for b in self.boxes)


def _centre(a: np.ndarray) -> np.ndarray:
    return np.column_stack(((a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2))


def view_boxes(boxes: np.ndarray | None) -> ViewBoxes:
    if boxes is None:
        return ViewBoxes((None,) * len(VIEWS), (None,) * len(VIEWS), None)
    valid = np.asarray(boxes, np.float32)
    valid = valid[valid[:, 8] > 0]
    ball = valid[valid[:, 4] > 0][:1]
    players = valid[valid[:, 6] > 0]
    keepers = valid[valid[:, 5] > 0]
    shooter, distance = None, None
    if len(ball) and len(players):
        d = np.linalg.norm(_centre(players) - _centre(ball), axis=1)
        distance = float(d.min())
        if distance <= MAX_SHOOTER_DISTANCE:
            shooter = _bounded_square(np.vstack([ball, players[d.argmin()][None]]), MIN_SIDE, MARGIN)
    keeper = None
    if len(keepers):
        k = keepers[np.linalg.norm(_centre(keepers) - _centre(ball), axis=1).argmin()] if len(ball) else keepers[0]
        keeper = _bounded_square(k[None], MIN_SIDE, MARGIN)
    chosen = (shooter, keeper)
    return ViewBoxes(chosen, tuple(None if b is None else _control_box(b) for b in chosen), distance)


class MultiViewDataset(Dataset):
    """Full frame prepared exactly as ImageDataset does for G, plus the local views.

    Views are cut only when the input frame is the frame the boxes were detected on
    (`box_frame_col`); an admitted earlier frame keeps the full view and masks the
    local ones. `control=True` swaps every view for its equal-size control crop with the
    same validity flags, so the two variants differ in pixel content only."""

    def __init__(self, df: pd.DataFrame, boxes: dict[str, np.ndarray], train: bool = False, control: bool = False,
                 path_col: str = "frame_path", box_frame_col: str = "frame_path", mask: str = "overlay",
                 rrc_scale: tuple[float, float] | None = None, aux_col: str | None = None):
        check_frame_col(path_col)
        self.df = df.reset_index(drop=True)
        self.boxes, self.train, self.control = boxes, train, control
        self.path_col, self.box_frame_col, self.mask = path_col, box_frame_col, mask
        self.rrc_scale, self.aux_col = rrc_scale, aux_col

    def __len__(self) -> int:
        return len(self.df)

    def view_boxes(self, row) -> ViewBoxes:
        if str(row[self.path_col]) != str(row[self.box_frame_col]):
            return view_boxes(None)
        return view_boxes(self.boxes.get(str(row.shot_id)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        path = str(row[self.path_col])
        raw = cv2.imread(path)
        if raw is None:
            raise FileNotFoundError(path)
        h, w = raw.shape[:2]
        crop = random_crop_box(w, h, self.rrc_scale) if self.train and self.rrc_scale is not None else None
        if crop is None:
            full = cv2.resize(raw, FULL_SIZE, interpolation=cv2.INTER_AREA)
            full[overlay_mask(FULL_SIZE, self.mask)] = FILL
        else:
            native = raw.copy()
            native[overlay_mask((w, h), self.mask)] = FILL
            full = cv2.resize(native[crop[1]:crop[3], crop[0]:crop[2]], FULL_SIZE, interpolation=cv2.INTER_AREA)
        views = np.zeros((len(VIEWS), LOCAL_SIZE[1], LOCAL_SIZE[0], 3), np.uint8)
        vb = self.view_boxes(row)
        valid = np.asarray(vb.valid, np.float32)
        native = None
        for k, box in enumerate(vb.controls if self.control else vb.boxes):
            if box is None:
                continue
            if native is None:
                native = raw.copy()
                native[overlay_mask((w, h), self.mask)] = FILL
            x0, y0, x1, y1 = normalized_crop_pixels(box, w, h)
            views[k] = cv2.resize(native[y0:y1, x0:x1], LOCAL_SIZE, interpolation=cv2.INTER_AREA)
        images = [cv2.cvtColor(full, cv2.COLOR_BGR2RGB)] + [cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in views]
        if self.train:
            if np.random.rand() < 0.5:
                images = [img[:, ::-1] for img in images]
            gain, bias = np.random.uniform(0.8, 1.2), np.random.uniform(-20, 20)
            images = [np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8) for img in images]
        out = {"image": to_tensor(images[0]), "views": torch.stack([to_tensor(img) for img in images[1:]]),
               "view_valid": torch.from_numpy(valid), "y": torch.tensor(float(row.goal))}
        if self.aux_col is not None:
            out["y_aux"] = torch.tensor([float(row[self.aux_col])])
        return out
