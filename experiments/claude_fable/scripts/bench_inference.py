"""Inference cost per shot of the xG configurations in report_complexity.py, on this GPU.

    ./run.sh python scripts/bench_inference.py --out results/v16_complexity/timing.json

What the numbers are:
  * an estimate of compute cost, not a measurement of the deployed G-combo: the fold
    weights were not saved, so every network has random weights (pretrained=False).
    Weights do not change the arithmetic. "fp32" runs with TF32 matmuls on; "fp16" is
    autocast.
  * one measurement boundary on every path: decoded BGR frames in host memory -> per-shot
    output. Every frame is decoded before any timer starts, the positional fallback frames
    included; resizing, overlay masking, normalising and the host-to-device copy are timed.
  * throughput: a batch of `--batch` shots, the time divided by the batch size. Latency:
    one shot per call (batch 1), synchronised per call, median over calls. Both are timed
    end to end per configuration, not multiplied up from the pieces.

Two ways to run n fine-tuned G models:
  naive   every fold model runs the full ViT-L (24 blocks): one network run n times,
          which costs what n networks cost;
  shared  the 20 frozen blocks are identical in every fold model (only the last 4 blocks,
          the final norm and the head were trained), so they run once per frame and n
          distinct trained tails (each with its own weights) run on the shared features.
          The split is checked numerically against the full forward before timing.
The positional extractor of the GBM is the real one (detector + pitch keypoints) on real
test frames, with the -400/-800 ms ball fallback the GBM cache was built with. It runs one
frame at a time, so its per-shot time enters throughput and latency alike.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from xgpoc.data.datasets import FILL, overlay_mask, to_tensor
from xgpoc.features.freeze_frame import FreezeFrameExtractor
from xgpoc.models.c_cnn_e2e import CnnXG
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.paths import INDEX

DEV = "cuda"
N_TRAINED_BLOCKS = 4
MASK = "overlay"
C_SIZE_WH = (448, 256)
G_PASSES = {"g1": 1, "g2": 2, "g5": 5, "g10": 10}
ENS_G, ENS_C = 15, 5  # final stack: 10 G-combo + 5 G-combo-mix (ViT-L), 5 C-combo (ConvNeXt-T), GBM
FALLBACK_COLS = ("frame_path_-400", "frame_path_-800")


class Preprocess:
    """xgpoc.data.datasets.load_image + to_tensor (no crop), from an already decoded frame."""

    def __init__(self, size_wh: tuple[int, int], mask: str):
        self.size, self.blank = size_wh, overlay_mask(size_wh, mask)

    def __call__(self, frames: list[np.ndarray]) -> torch.Tensor:
        out = []
        for f in frames:
            img = cv2.resize(f, self.size, interpolation=cv2.INTER_AREA)
            img[self.blank] = FILL
            out.append(to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return torch.stack(out).to(DEV)


def trunk(model: FinetuneViTXG, x: torch.Tensor) -> torch.Tensor:
    """timm VisionTransformer.forward_features up to the first trained block."""
    vit = model.backbone
    x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
    for blk in vit.blocks[:-N_TRAINED_BLOCKS]:
        x = blk(x)
    return x


class Tail(nn.Module):
    """The trained part of one fold model: last blocks, final norm, head. `seed` perturbs
    a copy of `model`'s tail so that every fold model has its own weights; None keeps it."""

    def __init__(self, model: FinetuneViTXG, seed: int | None):
        super().__init__()
        vit = model.backbone
        self.blocks = copy.deepcopy(vit.blocks[-N_TRAINED_BLOCKS:])
        self.norm, self.head = copy.deepcopy(vit.norm), copy.deepcopy(model.head)
        if seed is not None:
            gen = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)
            with torch.no_grad():
                for p in self.parameters():
                    p.add_(1e-3 * torch.randn(p.shape, generator=gen, device=p.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def pipelines(n_tails: int):
    """name -> {"naive": fn, "shared": fn, "vit_passes": n}, fn(frames) -> per-shot output. Built once: the networks
    and the n distinct tails, with the trunk/tail split checked on a random input."""
    g = FinetuneViTXG("dinov2_l", n_blocks=N_TRAINED_BLOCKS, n_out=2, grad_checkpointing=False, pretrained=False).to(DEV).eval()
    c = CnnXG(pretrained=False, n_out=2).to(DEV).eval()
    tails = [Tail(g, seed=i).eval() for i in range(n_tails)]
    prep_g, prep_c = Preprocess(g.size_wh, MASK), Preprocess(C_SIZE_WH, MASK)
    with torch.no_grad():
        x = torch.randn(2, 3, g.size_wh[1], g.size_wh[0], device=DEV)
        full, split = g(x), Tail(g, seed=None)(trunk(g, x))
        assert torch.allclose(full, split, atol=1e-4), (full - split).abs().max()

    def naive(n: int):
        def run(frames):
            x = prep_g(frames)
            return torch.stack([g(x)[:, 0] for _ in range(n)]).mean(0)
        return run

    def shared(n: int):
        def run(frames):
            feats = trunk(g, prep_g(frames))
            return torch.stack([t(feats)[:, 0] for t in tails[:n]]).mean(0)
        return run

    def with_cnn(vit_part):
        def run(frames):
            xc = prep_c(frames)
            return vit_part(frames), torch.stack([c(xc)[:, 0] for _ in range(ENS_C)]).mean(0)
        return run

    out = {}
    for name, n in G_PASSES.items():
        out[name] = {"naive": naive(n), "shared": shared(n), "vit_passes": n}
    out["ens"] = {"naive": with_cnn(naive(ENS_G)), "shared": with_cnn(shared(ENS_G)), "vit_passes": ENS_G}
    return out


def throughput_ms(fn, frames: list[np.ndarray], reps: int, amp: bool, warmup: int = 3) -> float:
    """Milliseconds per shot over `reps` calls on the whole batch."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            fn(frames)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(frames)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / len(frames) * 1000


def latency_ms(fn, frames: list[np.ndarray], amp: bool, warmup: int = 3) -> float:
    """Median milliseconds of one single-shot call, one call per frame."""
    times = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for f in frames[:warmup]:
            fn([f])
        torch.cuda.synchronize()
        for f in frames:
            t0 = time.perf_counter()
            fn([f])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def decode_shot(row: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """The anchor frame and whichever fallback frames exist, decoded."""
    alts = [cv2.imread(row[col]) for col in FALLBACK_COLS if row.get(col)]
    return cv2.imread(row["frame_path"]), [img for img in alts if img is not None]


def bench_positional(rows: pd.DataFrame) -> dict[str, float]:
    """Milliseconds per shot of the GBM's feature extraction with the ball fallback; every
    frame (fallbacks too) is decoded before the timer."""
    ext = FreezeFrameExtractor(device=DEV)
    shots = [decode_shot(r) for r in rows.to_dict("records")]  # itertuples mangles the "-400" column names
    ext(shots[0][0])  # model load and CUDA warm-up
    times, fallbacks = [], 0
    for primary, alts in shots:
        t0 = time.perf_counter()
        if ext(primary).ball_xy is None:
            for alt in alts:
                fallbacks += 1
                if ext(alt).ball_xy is not None:
                    break
        times.append(time.perf_counter() - t0)
    return {"positional_ms": float(np.mean(times) * 1000), "positional_ms_median": float(np.median(times) * 1000),
            "positional_frames": len(shots), "fallback_calls_per_shot": fallbacks / len(shots)}


def configuration_times(pipes: dict, frames: list[np.ndarray], batch: int, reps: int, amp: bool,
                        pos: dict[str, float]) -> dict[str, dict]:
    """{"configurations": throughput, "latency_batch1": latency}, each pipeline -> ms per
    shot. The ensemble adds the positional path: its mean time to throughput, its median
    to latency (which is itself a median)."""
    timers = {"configurations": (lambda fn: throughput_ms(fn, frames[:batch], reps, amp), pos["positional_ms"]),
              "latency_batch1": (lambda fn: latency_ms(fn, frames, amp), pos["positional_ms_median"])}
    return {kind: {name: {"vit_passes": p["vit_passes"]}
                   | {f"{k}_ms": timer(p[k]) + (pos_ms if name == "ens" else 0.0) for k in ("naive", "shared")}
                   for name, p in pipes.items()}
            for kind, (timer, pos_ms) in timers.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/v16_complexity/timing.json"))
    ap.add_argument("--batch", type=int, default=16, help="shots per call for throughput")
    ap.add_argument("--reps", type=int, default=20, help="timed calls per throughput figure")
    ap.add_argument("--frames", type=int, default=100, help="test frames for latency and the positional path")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    idx = pd.read_parquet(INDEX).drop_duplicates("shot_id")
    rows = idx[idx.split == "test"].sample(a.frames, random_state=0)
    frames = [cv2.imread(p) for p in rows.frame_path]
    pos = bench_positional(rows)
    pipes = pipelines(ENS_G)
    res = {"gpu": torch.cuda.get_device_name(0), "tf32": True, "weights": "random (fold weights not saved)",
           "boundary": "decoded BGR frame in host memory -> per-shot output; decoding excluded on every path",
           "batch": a.batch, **pos}
    for prec, amp in (("fp32", False), ("fp16", True)):
        for kind, table in configuration_times(pipes, frames, a.batch, a.reps, amp, pos).items():
            res[f"{kind}_{prec}"] = table
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
