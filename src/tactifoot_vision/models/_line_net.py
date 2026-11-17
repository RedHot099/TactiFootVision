"""The line network itself (torch): the architecture, the decoder and the training loop of the ``lines`` backend.

Imported only when a ``lines`` model is created, so ``import
tactifoot_vision.models`` stays free of torch.

* :class:`NetSpec`, :class:`LineNet`: a ResNet encoder (ImageNet weights)
  and an FPN decoder to stride 4, or on to stride 2 with the stem's
  features; input the frame resized to ``width`` x ``width * 9 / 16`` and
  padded at the bottom to a multiple of 16 rows; output 20 logit maps, one
  per :class:`~tactifoot_vision.enums.PitchMarking`.
* :func:`decode`: on the device, a pixel is a point of its marking when its
  probability is above ``threshold`` and it is a ridge (not below both
  neighbours across the line, horizontally or vertically); the most probable
  ridge pixel per ``bin`` x ``bin`` block of output cells is kept, refined
  to sub-cell precision by a parabola across the ridge.
* :func:`fit`: the training loop (focal loss on Gaussian line heatmaps,
  AdamW, cosine schedule with warm-up, bf16 autocast, one checkpoint per
  epoch, resumable).
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tactifoot_vision.enums import LineBackbone, PitchView
from tactifoot_vision.pitch.fit import LinePoints
from tactifoot_vision.pitch.markings import (
    FLIP_ACROSS,
    FLIP_ALONG,
    INDEX,
    MARKINGS,
    NUM_MARKINGS,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# Bumped whenever ``decode`` changes what it returns for the same logits.
DECODER_VERSION = 1


@dataclass(frozen=True)
class NetSpec:
    """A line network's shape: encoder, output stride (2 or 4) and input width (the frame keeps 16:9)."""

    backbone: LineBackbone | str = LineBackbone.RESNET18
    stride: int = 2
    width: int = 640

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backbone", LineBackbone.parse(self.backbone, "backbone")
        )
        if self.stride not in (2, 4):
            raise ValueError(f"stride must be 2 or 4, got {self.stride}")
        if self.width < 64 or self.width % 16:
            raise ValueError(
                f"width must be a multiple of 16 from 64, got {self.width}"
            )

    @property
    def image_size(self) -> tuple[int, int]:
        return self.width, round(self.width * 9 / 16)

    @property
    def input_size(self) -> tuple[int, int]:
        w, h = self.image_size
        return w, -(-h // 16) * 16

    @property
    def out_size(self) -> tuple[int, int]:
        w, h = self.input_size
        return w // self.stride, h // self.stride

    @property
    def bin(self) -> int:
        """One point per bin x bin output cells: 6 input pixels at stride 2, 8 at stride 4."""
        return 3 if self.stride == 2 else 2


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class LineNet(nn.Module):
    def __init__(
        self, spec: NetSpec, pretrained: bool = True, width: int = 128
    ) -> None:
        super().__init__()
        import torchvision

        self.spec = spec
        if spec.backbone is LineBackbone.RESNET34:
            weights = (
                torchvision.models.ResNet34_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            r = torchvision.models.resnet34(weights=weights)
        else:
            weights = (
                torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # stride 2, 64
        self.pool = r.maxpool
        self.layers = nn.ModuleList(
            [r.layer1, r.layer2, r.layer3, r.layer4]
        )  # 4, 8, 16, 32
        chans = (64, 128, 256, 512)
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chans])
        self.smooth = nn.ModuleList([_block(width, width) for _ in chans])
        self.up2 = _block(width + 64, 64) if spec.stride == 2 else None
        self.head = nn.Sequential(
            _block(64 if spec.stride == 2 else width, 64),
            nn.Conv2d(64, NUM_MARKINGS, 1),
        )
        nn.init.constant_(self.head[-1].bias, -4.0)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, 3, H, W) RGB 0-255 float at ``spec.input_size``; returns (B, 20, H / stride, W / stride) logits."""
        x = (x - self.mean) / self.std
        s2 = self.stem(x)
        feats = []
        y = self.pool(s2)
        for layer in self.layers:
            y = layer(y)
            feats.append(y)
        top = self.smooth[3](self.lateral[3](feats[3]))
        for i in (2, 1, 0):
            lat = self.lateral[i](feats[i])
            top = F.interpolate(
                top, size=lat.shape[-2:], mode="bilinear", align_corners=False
            )
            top = self.smooth[i](lat + top)
        if self.up2 is not None:
            top = F.interpolate(
                top, size=s2.shape[-2:], mode="bilinear", align_corners=False
            )
            top = self.up2(torch.cat([top, s2], 1))
        return self.head(top)


def load_net(
    weights: str | Path | None,
    spec: NetSpec,
    pretrained: bool,
    device: str,
    seed: int | None = None,
) -> LineNet:
    """A network from a checkpoint (its saved spec), or a new one of ``spec`` (its new layers drawn from ``seed``)."""
    if weights is None:
        if seed is not None:
            torch.manual_seed(seed)
        net = LineNet(spec, pretrained=pretrained)
    else:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model" not in state:
            raise ValueError(f"{path} is not a lines checkpoint (no 'model' state)")
        net = LineNet(NetSpec(**state.get("spec", {})), pretrained=False)
        net.load_state_dict(state["model"])
    net.eval().to(device)
    if device != "cpu":
        net = net.to(memory_format=torch.channels_last)
    return net


def to_input(frames: list[np.ndarray], spec: NetSpec, device: str) -> torch.Tensor:
    """BGR uint8 frames of any size -> (B, 3, H, W) RGB float at ``spec.input_size`` (area-resized on ``device``)."""
    w, h = spec.input_size
    iw, ih = spec.image_size
    out = torch.full((len(frames), 3, h, w), 114.0, device=device)
    for i, frame in enumerate(frames):
        t = torch.from_numpy(np.ascontiguousarray(frame)).to(device, non_blocking=True)
        t = t.permute(2, 0, 1).flip(0).float()[None]
        if t.shape[-1] != iw or t.shape[-2] != ih:
            t = F.interpolate(t, size=(ih, iw), mode="area")
        out[i, :, :ih, :iw] = t[0]
    return out


@torch.no_grad()
def predict(
    net: LineNet, frames: list[np.ndarray], threshold: float, device: str
) -> list[LinePoints]:
    """The decoded points of ``frames`` (fp16 autocast on CUDA, float32 on the CPU)."""
    spec = net.spec
    x = to_input(frames, spec, device)
    if device != "cpu":
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(x)
    else:
        logits = net(x)
    return decode(
        logits, [(f.shape[1], f.shape[0]) for f in frames], spec, threshold, spec.bin
    )


@torch.no_grad()
def decode(
    logits: torch.Tensor,
    frame_sizes: list[tuple[int, int]],
    spec: NetSpec,
    threshold: float,
    bin: int,
) -> list[LinePoints]:
    """Ridge points of each marking's map (see the module), in each frame's pixels."""
    p = torch.sigmoid(logits.float())
    rows = -(-spec.image_size[1] // spec.stride)
    p = p[..., :rows, :]
    pad = F.pad(p, (1, 1, 1, 1))
    left, right = pad[..., 1:-1, :-2], pad[..., 1:-1, 2:]
    up, down = pad[..., :-2, 1:-1], pad[..., 2:, 1:-1]
    ridge = ((p >= left) & (p >= right)) | ((p >= up) & (p >= down))
    score = torch.where((p > threshold) & ridge, p, torch.zeros_like(p))
    ww = score.shape[-1]
    pooled, idx = F.max_pool2d(
        score, bin, stride=bin, return_indices=True, ceil_mode=True
    )
    keep = pooled > threshold
    out = []
    for i in range(len(score)):
        k_cls, k_y, k_x = torch.nonzero(keep[i], as_tuple=True)
        flat = idx[i][k_cls, k_y, k_x]
        y, x = flat // ww, flat % ww
        val = p[i, k_cls, y, x]
        # sub-cell refinement across the ridge: the sharper of the two directions
        lx, rx = left[i, k_cls, y, x], right[i, k_cls, y, x]
        uy, dy = up[i, k_cls, y, x], down[i, k_cls, y, x]
        sx, sy = lx - 2 * val + rx, uy - 2 * val + dy
        offx = torch.where(
            sx < -1e-6, 0.5 * (lx - rx) / sx.clamp(max=-1e-6), torch.zeros_like(sx)
        )
        offy = torch.where(
            sy < -1e-6, 0.5 * (uy - dy) / sy.clamp(max=-1e-6), torch.zeros_like(sy)
        )
        use_y = sy < sx
        fx = x.float() + torch.where(
            use_y, torch.zeros_like(offx), offx.clamp(-0.5, 0.5)
        )
        fy = y.float() + torch.where(
            use_y, offy.clamp(-0.5, 0.5), torch.zeros_like(offy)
        )
        # output cell centre i -> input pixel s (i + 0.5) of the resized image -> frame pixels
        fw, fh = frame_sizes[i]
        iw, ih = spec.image_size
        u = (fx + 0.5) * spec.stride * fw / iw
        v = (fy + 0.5) * spec.stride * fh / ih
        out.append(
            LinePoints(
                k_cls.cpu().numpy().astype(np.int64),
                torch.stack([u, v], 1).cpu().numpy().astype(np.float32),
                val.cpu().numpy().astype(np.float32),
            )
        )
    return out


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss on soft line heatmaps, per image weighted, normalised by the line pixels."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p**2 * (1 - target) ** 4 * (1 - pos)
    per_image = (pos_loss + neg_loss).flatten(1).sum(1)
    n_pos = pos.flatten(1).sum(1).clamp(min=1.0)
    return (per_image / n_pos * weight).sum() / weight.sum().clamp(min=1e-6)


# ------------------------------------------------------------------ training
class _Samples(torch.utils.data.Dataset):
    """A ``lines`` dataset split as (input, targets, weight) tensors, with augmentation when training."""

    def __init__(self, samples: list, spec: NetSpec, augment: bool, seed: int) -> None:
        self.samples, self.spec, self.augment, self.seed = samples, spec, augment, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import cv2

        from tactifoot_vision.data.pitch_lines import render_line_targets

        sample = self.samples[i]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        image = sample.read_image()
        size = self.spec.image_size
        scale = np.array(size, float) / (image.shape[1], image.shape[0])
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        ann = sample.annotations
        lines = {k: np.asarray(v, float) * scale for k, v in (ann.lines or {}).items()}
        if self.augment:
            image, lines = _augment(
                image,
                lines,
                rng,
                FLIP_ACROSS if ann.view is PitchView.ALONG else FLIP_ALONG,
            )
        target = render_line_targets(lines, self.spec.out_size, self.spec.stride)
        w, h = self.spec.input_size
        canvas = np.full((h, w, 3), 114, np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))
        return (
            x,
            torch.from_numpy(target),
            torch.tensor(float(ann.weight), dtype=torch.float32),
        )


def _augment(
    image: np.ndarray, lines: dict, rng, flip: tuple[int, ...]
) -> tuple[np.ndarray, dict]:
    """Zoom into a random window (half the time), mirror (half the time, names swapped by ``flip``), colour, blur."""
    import cv2

    h, w = image.shape[:2]
    if rng.random() < 0.5:
        s = rng.uniform(1.0, 1.6)
        cw, ch = w / s, h / s
        x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
        m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
        image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR)
        lines = {k: v * s - [x0 * s, y0 * s] for k, v in lines.items()}
    if rng.random() < 0.5:
        image = image[:, ::-1]
        names = {m.value: i for m, i in INDEX.items()}
        lines = {
            (MARKINGS[flip[names[k]]].value if k in names else k): np.column_stack(
                [w - v[:, 0], v[:, 1]]
            )
            for k, v in lines.items()
        }
    hsv = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2HSV).astype(
        np.float32
    )
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.2:
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (k, k), 0)
    return image, lines


# Loader workers start fresh ("spawn"): a forked worker can deadlock inside
# OpenCV when the parent had used OpenCV's threads before (a notebook that
# trained a detector first, say).
_WORKERS = "spawn"


def _single_thread(_worker: int) -> None:
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def fit(
    net: LineNet,
    train: list,
    valid: list,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    workers: int,
    seed: int,
    device: str,
    run_dir: Path,
    weight_decay: float = 1e-4,
    stop: Callable[[int], bool] | None = None,
) -> tuple[Path | None, list[dict]]:
    """Train ``net`` in place; one checkpoint per epoch in ``run_dir/weights/last.pt`` (resumed when present).

    Returns the final weights (``run_dir/weights/final.pt``) and the per-epoch
    log (train loss, valid loss, wall seconds). ``stop(epoch)``, asked after
    each epoch but the last, ends the run at that checkpoint and returns no
    weights; calling again with the same ``run_dir`` resumes it.
    """
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    spec = net.spec
    net.train().to(device)
    if device != "cpu":
        net = net.to(memory_format=torch.channels_last)
    data = _Samples(train, spec, augment=True, seed=seed)
    check = _Samples(valid, spec, augment=False, seed=seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(data) // batch_size)
    total = steps_per_epoch * epochs
    warmup = min(500, max(1, total // 10))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log_path = run_dir / "log.jsonl"
    last = weights_dir / "last.pt"
    start = 0
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["epoch"] + 1
    use_amp = device != "cpu"
    for epoch in range(start, epochs):
        data.epoch = epoch
        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=len(data) >= batch_size,
            pin_memory=use_amp, generator=torch.Generator().manual_seed(seed * 1000 + epoch),
            worker_init_fn=_single_thread if workers else None,
            multiprocessing_context=_WORKERS if workers else None,
        )  # fmt: skip
        net.train()
        started = time.time()
        losses = []
        for x, target, weight in loader:
            x = x.to(device, non_blocking=True).float()
            if use_amp:
                x = x.contiguous(memory_format=torch.channels_last)
            target, weight = (
                target.to(device, non_blocking=True),
                weight.to(device, non_blocking=True),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = net(x)
            loss = focal_loss(logits, target, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        valid_loss = _valid_loss(net, check, batch_size, workers, device, use_amp)
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else float("nan"),
                  "valid_loss": valid_loss, "wall_s": round(time.time() - started, 1)}  # fmt: skip
        state = {"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": epoch, "spec": asdict(spec)}  # fmt: skip
        torch.save(state, weights_dir / "last.tmp")
        (weights_dir / "last.tmp").rename(last)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if epoch + 1 < epochs and stop is not None and stop(epoch + 1):
            return None, _read_log(log_path)
    final = weights_dir / "final.pt"
    torch.save({"model": net.state_dict(), "spec": asdict(spec)}, final)
    return final, _read_log(log_path)


def _read_log(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in open(path) if line.strip()]
        if path.exists()
        else []
    )


@torch.no_grad()
def _valid_loss(
    net: LineNet,
    data: _Samples,
    batch_size: int,
    workers: int,
    device: str,
    use_amp: bool,
) -> float:
    if not len(data):
        return float("nan")
    net.eval()
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=workers,
                                         worker_init_fn=_single_thread if workers else None,
                                         multiprocessing_context=_WORKERS if workers else None)  # fmt: skip
    total, count = 0.0, 0
    for x, target, weight in loader:
        x = x.to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = net(x)
        total += float(focal_loss(logits, target.to(device), weight.to(device))) * len(
            x
        )
        count += len(x)
    return total / count
