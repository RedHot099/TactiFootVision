"""A second stage for the ball candidates of single frames: crops rescored and centred.

The detector's ball confidence tells a ball from a ball-like spot poorly. A
ResNet18 looks at a square crop around each candidate, scores whether it
holds the ball and regresses the ball's centre in it. On SoccerNet GSR valid
games 2+3 (ball-detection 13 on ``research/ball-detection``) the most
confident candidate after this stage was within 5 m of the ball in 62.4% of
frames against 51.6% for the YOLO11l dd2c detector's most confident one
(+10.8 pp [+8.0, +14.0]), but after
:func:`~tactifoot_vision.pipeline.track_ball` only 72.7% against 71.3%
(+1.4 pp [-0.5, +3.3]): the tracker already recovers most of it. So it is
an option for single frames (a shot's moment, a freeze frame), off by
default.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.utils import resolve_device

logger = logging.getLogger(__name__)

_REFERENCE_HEIGHT = 1080  # px: frame height the crop's smallest side is given at
_CROP_SIZE = 96  # px: the network's input side
_CROP_MIN_SIDE = 32.0  # px at 1080p
_CROP_BOX_FACTOR = 3.0  # crop side per longer box side
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, RGB
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_P_CLIP = 1e-6
# sigmoid(a logit(p_crop) + b logit(p_detector) + c), fitted on SoccerNet GSR
# train game 9 for YOLO11l dd2c's confidences (ball-detection 13)
_DD2C_FUSION = (0.6201880409840753, 0.5778215273239043, 3.0840836881257476)


class BallCropScorer:
    """Rescores a frame's ball candidates on crops and moves their boxes onto the ball.

    Each candidate's crop, a square of ``max(32 px at 1080p, 3 x its longer
    side)`` around its centre resized to 96 px, gets a probability of
    holding the ball, which is fused with the detector's confidence into the
    new confidence. With ``move_boxes`` the box is shifted by the regressed
    offset of the ball's centre. Candidates whose new confidence is below
    ``min_confidence`` are dropped.

    Run the detector with weak ball candidates kept (``class_conf={"ball":
    0.01}`` was measured): the crop rescues balls the detector scored low.
    The fusion was fitted for one detector's confidences; another detector
    needs its own ``fusion``. Rescored confidences no longer match
    :func:`~tactifoot_vision.pipeline.track_ball`'s defaults or its
    uncertainty, which were tuned on the detector's own.

    Args:
        weights: the crop network's checkpoint (ball-detection 13's
            ``best.pt``: a state dict under ``"model"``).
        min_confidence: candidates whose new confidence is lower are dropped.
        fusion: ``(a, b, c)`` of ``sigmoid(a logit(p_crop) + b
            logit(confidence) + c)``.
        move_boxes: shift each box onto the regressed ball centre.
        device: torch device; ``None`` picks CUDA when available.
    """

    def __init__(
        self,
        weights: str | Path,
        min_confidence: float = 0.15,
        fusion: Sequence[float] = _DD2C_FUSION,
        move_boxes: bool = True,
        device: str | None = None,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if len(fusion) != 3:
            raise ValueError(f"fusion must be three numbers (a, b, c), got {fusion!r}")
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"No ball crop scorer weights at {self.weights}")
        self.min_confidence = min_confidence
        self.fusion = tuple(float(v) for v in fusion)
        self.move_boxes = move_boxes
        self.device = resolve_device(device)
        self._net = _load_net(self.weights, self.device)

    def __call__(self, image: np.ndarray, candidates: sv.Detections) -> sv.Detections:
        """``candidates`` of ``image`` (BGR) rescored, moved and filtered, in their order."""
        if not len(candidates):
            return candidates
        boxes = candidates.xyxy.astype(float)
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        sides = np.maximum(
            _CROP_MIN_SIDE * image.shape[0] / _REFERENCE_HEIGHT,
            _CROP_BOX_FACTOR * (boxes[:, 2:] - boxes[:, :2]).max(axis=1),
        )
        p_crop, offsets = self._predict(_crops(image, centres, sides))
        detector = (
            np.ones(len(candidates))
            if candidates.confidence is None
            else candidates.confidence.astype(float)
        )
        a, b, c = self.fusion
        confidence = 1 / (1 + np.exp(-(a * _logit(p_crop) + b * _logit(detector) + c)))
        keep = confidence >= self.min_confidence
        scored = candidates[keep]
        scored.confidence = confidence[keep].astype(np.float32)
        if self.move_boxes:
            shift = np.tile(offsets[keep] * sides[keep, None], 2)
            scored.xyxy = (boxes[keep] + shift).astype(candidates.xyxy.dtype)
        return scored

    def _predict(self, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(p_crop, offsets in crop sides)`` of uint8 BGR crops."""
        import torch

        x = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255)
        mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        cuda = str(self.device).startswith("cuda")
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=cuda),
        ):
            logit, offset = self._net((x - mean) / std)
        return (
            torch.sigmoid(logit.float()).cpu().numpy().astype(float),
            offset.float().cpu().numpy().astype(float),
        )

    def __repr__(self) -> str:
        return (
            f"BallCropScorer(weights={str(self.weights)!r}, "
            f"min_confidence={self.min_confidence}, device={self.device!r})"
        )


def _crops(image: np.ndarray, centres: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """``(n, 96, 96, 3)`` squares of ``sides`` around ``centres``, outside the image black."""
    out = np.zeros((len(centres), _CROP_SIZE, _CROP_SIZE, 3), np.uint8)
    for i, ((cx, cy), side) in enumerate(zip(centres, sides, strict=True)):
        k = _CROP_SIZE / side
        matrix = np.array(
            [[k, 0, _CROP_SIZE / 2 - k * cx], [0, k, _CROP_SIZE / 2 - k * cy]]
        )
        out[i] = cv2.warpAffine(
            image,
            matrix,
            (_CROP_SIZE, _CROP_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _P_CLIP, 1 - _P_CLIP)
    return np.log(p / (1 - p))


def _load_net(weights: Path, device: str):
    """The crop network (ResNet18, a ball logit and a centre offset) in eval mode."""
    import torch
    import torchvision

    class CropNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torchvision.models.resnet18(weights=None)
            self.backbone.fc = torch.nn.Identity()
            self.cls = torch.nn.Linear(512, 1)
            self.offset = torch.nn.Linear(512, 2)

        def forward(self, x):
            features = self.backbone(x)
            return self.cls(features).squeeze(1), self.offset(features)

    net = CropNet()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])
    logger.info("Loaded the ball crop scorer %s", weights)
    return net.to(device).eval()
