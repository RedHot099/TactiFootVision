"""A heatmap pitch keypoint model (pitch-keypoints 03, candidate family 2), in research code.

One heatmap per `SoccerPitch` vertex at stride `STRIDE` of a 960 x 544 input (the 16:9 frame resized to 960 x 540,
padded with 4 black rows at the bottom), as in the SoccerNet camera calibration literature (HRNet keypoint heatmaps,
"No Bells, Just Whistles"). A timm backbone (`features_only`) feeds a small top-down decoder that fuses every scale at
stride 4, as CenterNet does: one heatmap per vertex and a sub-pixel offset (x, y) per vertex. Targets are Gaussians
(`SIGMA` heatmap pixels) centred on the heatmap pixel nearest to each visible vertex, peak 1 there; a vertex outside the
frame has an all-zero map. The loss is CenterNet's penalty-reduced focal loss on the sigmoid heatmaps, so a map's peak
is the vertex's confidence, plus an L1 loss on the offset at the peak pixel of each visible vertex. Decoding: the peak
pixel of each map plus its predicted offset.

`HeatmapKeypointModel` runs a checkpoint on a BGR frame of any 16:9 size and returns `sv.KeyPoints` like the package's
`yolo_pose` model (one instance; `data["box_confidence"]` is the largest vertex confidence), so the evaluator and
`HomographyEstimator` take it unchanged.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from torch import nn

INPUT_W, INPUT_H = 960, 544                  # model input; the frame is resized to 960 x 540 and padded at the bottom
FRAME_W, FRAME_H = 960, 540
STRIDE = 4
SIGMA = 2.0                                  # heatmap pixels (8 input px)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class HeatmapConfig:
    backbone: str = "hrnet_w32.ms_in1k"
    num_keypoints: int = 32
    decoder_channels: int = 128
    sigma: float = SIGMA


class HeatmapNet(nn.Module):
    """timm backbone -> top-down fusion of every scale at stride 4 -> one logit map per keypoint."""

    def __init__(self, config: HeatmapConfig, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.config = config
        self.backbone = timm.create_model(config.backbone, pretrained=pretrained, features_only=True)
        info = self.backbone.feature_info
        keep = [i for i, r in enumerate(info.reduction()) if r >= STRIDE]
        self.keep = keep
        self.reductions = [info.reduction()[i] for i in keep]
        c = config.decoder_channels
        self.lateral = nn.ModuleList(nn.Conv2d(info.channels()[i], c, 1) for i in keep)
        self.smooth = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(c, config.num_keypoints, 1)
        nn.init.constant_(self.head.bias, -4.6)              # sigmoid ~0.01: the focal loss's usual prior
        self.offset = nn.Conv2d(c, 2 * config.num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Heatmap logits (N, K, h, w) and offsets (N, K, 2, h, w) in heatmap pixels, at stride 4."""
        features = self.backbone(x)
        size = (x.shape[2] // STRIDE, x.shape[3] // STRIDE)
        fused = 0
        for lateral, i in zip(self.lateral, self.keep, strict=True):
            fused = fused + F.interpolate(lateral(features[i]), size=size, mode="bilinear", align_corners=False)
        fused = self.smooth(fused)
        offsets = self.offset(fused)
        return self.head(fused), offsets.view(offsets.shape[0], -1, 2, *offsets.shape[2:])


HEATMAP_SHAPE = (INPUT_H // STRIDE, INPUT_W // STRIDE)


def targets(xy: np.ndarray, visible: np.ndarray, sigma: float = SIGMA,
            shape: tuple[int, int] = HEATMAP_SHAPE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training targets of one image (`xy` in input pixels): heatmaps (K, h, w) float32 with a Gaussian of peak 1 at
    the map pixel nearest to each visible keypoint, that pixel's flat index (K,) int64 and the sub-pixel offset of the
    keypoint from it (K, 2) float32; keypoints that are not visible or fall outside the map get index -1."""
    h, w = shape
    heatmaps = np.zeros((len(xy), h, w), np.float32)
    index = np.full(len(xy), -1, np.int64)
    offset = np.zeros((len(xy), 2), np.float32)
    ys, xs = np.arange(h, dtype=np.float32)[:, None], np.arange(w, dtype=np.float32)[None, :]
    for k in np.flatnonzero(visible):
        c = to_heatmap(xy[k])
        px, py = int(round(float(c[0]))), int(round(float(c[1])))
        if not (0 <= px < w and 0 <= py < h):
            continue
        heatmaps[k] = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma**2))
        index[k] = py * w + px
        offset[k] = c - (px, py)
    return heatmaps, index, offset


def to_heatmap(xy: np.ndarray) -> np.ndarray:
    """Input pixels -> heatmap pixels. Pixel coordinates are continuous, as in the labels (a fraction of the image
    times its size): pixel i covers [i, i + 1); heatmap pixel j covers input pixels [4j, 4j + 4) with its centre j."""
    return np.asarray(xy, np.float32) / STRIDE - 0.5


def from_heatmap(xy: np.ndarray) -> np.ndarray:
    return (np.asarray(xy, np.float32) + 0.5) * STRIDE


def offset_loss(offsets: torch.Tensor, index: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between the predicted offsets (N, K, 2, h, w) at the target pixels `index` (N, K; -1 = none) and the target
    offsets (N, K, 2), averaged over the visible keypoints."""
    n, k = index.shape
    flat = offsets.float().flatten(3)                                 # (N, K, 2, h*w)
    picked = flat.gather(3, index.clamp(min=0)[:, :, None, None].expand(n, k, 2, 1)).squeeze(3)
    mask = (index >= 0).float()[..., None]
    return (torch.abs(picked - target) * mask).sum() / (2 * mask.sum()).clamp(min=1)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss, normalised by the number of peaks (at least 1)."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    target = target.float()
    peak = target.eq(1).float()
    positive = -torch.log(p) * (1 - p) ** alpha * peak
    negative = -torch.log(1 - p) * p**alpha * (1 - target) ** beta * (1 - peak)
    return (positive.sum() + negative.sum()) / peak.sum().clamp(min=1)


def decode(logits: torch.Tensor, offsets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Heatmap logits (N, K, h, w) and offsets (N, K, 2, h, w) -> xy (N, K, 2) in input pixels and confidence (N, K):
    each map's peak pixel plus its predicted offset."""
    p = torch.sigmoid(logits.float())
    n, k, h, w = p.shape
    conf, index = p.flatten(2).max(dim=2)
    picked = offsets.float().flatten(3).gather(3, index[:, :, None, None].expand(n, k, 2, 1)).squeeze(3)
    xy = torch.stack([index % w, index // w], dim=-1).float() + picked
    return from_heatmap(xy.cpu().numpy()), conf.cpu().numpy()


def to_input(image: np.ndarray) -> np.ndarray:
    """A BGR 16:9 frame -> the model's (3, 544, 960) float32 input: 960 x 540 (area resize), RGB, normalised, padded."""
    if image.shape[:2] != (FRAME_H, FRAME_W):
        image = cv2.resize(image, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    rgb = (image[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    out = np.zeros((INPUT_H, INPUT_W, 3), np.float32)
    out[:FRAME_H] = rgb
    return out.transpose(2, 0, 1)


def save_checkpoint(path: Path, model: HeatmapNet, **extra) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()} | extra, path)


def load_net(path: Path, device: str = "cpu") -> HeatmapNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = HeatmapNet(HeatmapConfig(**checkpoint["config"]), pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    return net.to(device).eval()


class HeatmapKeypointModel:
    """A heatmap checkpoint as a keypoint model: `model(bgr_frame) -> sv.KeyPoints` (one instance, frame pixels)."""

    def __init__(self, weights: Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_net(Path(weights), self.device)

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> sv.KeyPoints:
        height, width = image.shape[:2]
        x = torch.from_numpy(to_input(image))[None].to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            logits, offsets = self.net(x)
        xy, conf = decode(logits, offsets)
        xy = xy[0] * np.float32([width / FRAME_W, height / FRAME_H])
        return sv.KeyPoints(xy=xy[None].astype(np.float32), confidence=conf[0][None].astype(np.float32),
                            class_id=np.zeros(1, int),
                            data={"class_name": np.array(["pitch"]),
                                  "xyxy": np.array([[0, 0, width, height]], np.float32),
                                  "box_confidence": np.array([conf[0].max()], np.float32)})


# ------------------------------------------------------------------------------------------------ training data
def read_label(path: Path, num_keypoints: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """A YOLO-pose label file of one pitch -> xy (K, 2) in 960 x 540 pixels and visibility (K,) bool."""
    values = np.array(Path(path).read_text().split(), np.float32)
    if len(values) != 5 + 3 * num_keypoints:
        raise ValueError(f"{path}: expected one pitch with {num_keypoints} keypoints, got {len(values)} values")
    kp = values[5:].reshape(num_keypoints, 3)
    return kp[:, :2] * (FRAME_W, FRAME_H), kp[:, 2] > 0


def flip(image: np.ndarray, xy: np.ndarray, visible: np.ndarray, flip_idx) -> tuple:
    """Horizontal flip: keypoint i of the flipped image is keypoint flip_idx[i] mirrored (x -> W - x)."""
    idx = np.asarray(flip_idx)
    xy = xy[idx].copy()
    xy[:, 0] = image.shape[1] - xy[:, 0]
    return image[:, ::-1].copy(), xy, visible[idx].copy()


def affine(image: np.ndarray, xy: np.ndarray, visible: np.ndarray, scale: float, shift: tuple[float, float]) -> tuple:
    """Zoom by `scale` about the image centre and shift by `shift` pixels; keypoints leaving the image become invisible.
    Keypoints stay in continuous coordinates (cv2.warpAffine's are pixel centres, hence the half-pixel terms)."""
    h, w = image.shape[:2]
    b = np.array([w / 2 * (1 - scale) + shift[0], h / 2 * (1 - scale) + shift[1]], np.float32)
    matrix = np.float32([[scale, 0, b[0] + 0.5 * scale - 0.5], [0, scale, b[1] + 0.5 * scale - 0.5]])
    out = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    xy = xy * scale + b
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return out, np.where((visible & inside)[:, None], xy, 0.0).astype(np.float32), visible & inside


def hsv_jitter(image: np.ndarray, rng: np.random.Generator, gains=(0.015, 0.7, 0.4)) -> np.ndarray:
    """Ultralytics-style random hue, saturation and value gains."""
    r = rng.uniform(-1, 1, 3) * gains + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s, lut_v = np.clip(x * r[1], 0, 255).astype(np.uint8), np.clip(x * r[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


@dataclass(frozen=True)
class Augment:
    """The training augmentation; every part keeps one whole pitch per image (see `training.YOLO_AUGMENT`)."""

    fliplr: float = 0.5
    scale: float = 0.25                       # zoom in [1 - scale, 1 + scale]
    translate: float = 0.1                    # shift up to this fraction of the width / height
    hsv: bool = True


class KeypointImages(torch.utils.data.Dataset):
    """Images of ticket 01's dataset with their labels, as (input (3, 544, 960), heatmaps, index, offset, xy, visible)."""

    def __init__(self, images: list[Path], flip_idx, augment: Augment | None = None, seed: int = 0) -> None:
        self.images = [Path(p) for p in images]
        self.flip_idx = list(flip_idx)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        path = self.images[i]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        if image.shape[:2] != (FRAME_H, FRAME_W):
            raise ValueError(f"{path}: expected {FRAME_W} x {FRAME_H}, got {image.shape[1]} x {image.shape[0]}")
        xy, visible = read_label(label_path(path), len(self.flip_idx))
        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            a = self.augment
            if rng.random() < a.fliplr:
                image, xy, visible = flip(image, xy, visible, self.flip_idx)
            if a.scale or a.translate:
                shift = rng.uniform(-a.translate, a.translate, 2) * (FRAME_W, FRAME_H)
                image, xy, visible = affine(image, xy, visible, float(rng.uniform(1 - a.scale, 1 + a.scale)),
                                            (float(shift[0]), float(shift[1])))
            if a.hsv:
                image = hsv_jitter(image, rng)
        heatmaps, index, offset = targets(xy, visible)
        return (torch.from_numpy(to_input(image)), torch.from_numpy(heatmaps), torch.from_numpy(index),
                torch.from_numpy(offset), torch.from_numpy(xy.astype(np.float32)), torch.from_numpy(visible))


def label_path(image: Path) -> Path:
    """`.../images/<split>/x.jpg` -> `.../labels/<split>/x.txt` (YOLO layout)."""
    parts = list(Path(image).parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
