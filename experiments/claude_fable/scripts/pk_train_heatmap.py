"""Train the heatmap pitch keypoint model (`xgpoc.pitchkp.heatmap`, candidate family 2) on ticket 01's dataset
(pitch-keypoints 03).

    ./run.sh python scripts/pk_train_heatmap.py --name NAME [--backbone hrnet_w32.ms_in1k] [--epochs 30] [--batch 8]
        [--lr 5e-4] [--val-step 4] [--workers 8] [--train-limit N --val-on-train]       smoke / overfit runs
    ./run.sh python scripts/pk_train_heatmap.py --name NAME --resume                     continue from last.pt

Training images: ticket 01's `train` split; validation: every `--val-step`-th `val` image (`training.training_yaml`'s
lists, written to `<run>/data/`); the test splits are never listed. AdamW, cosine schedule with a linear warm-up,
bfloat16 autocast. After every epoch: the validation keypoint error at 960 x 540 (`px_all_median` over every visible
vertex, `px_median` over those predicted with confidence >= 0.5, `recall`, `spurious`) in `results.csv`, `last.pt`
(resumable) and `best.pt` (lowest `px_all_median`). `sessions.jsonl` records each session's wall time, as
`pk_train_yolo.py` does. Reads no StatsBomb data.
"""

import argparse
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pk_train_yolo as yolo_script
import torch
import yaml
from torch.utils.data import DataLoader
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.pitchkp import heatmap as H
from xgpoc.pitchkp import training as T

CONFIDENT = 0.5


def image_list(path: Path) -> list[Path]:
    return [Path(line) for line in path.read_text().splitlines() if line.strip()]


@torch.no_grad()
def validate(net: H.HeatmapNet, loader: DataLoader, device: str) -> dict:
    """Keypoint error at 960 x 540 against the labels of the validation images."""
    net.eval()
    errors, confident_errors, seen, hits, unseen, spurious = [], [], 0, 0, 0, 0
    for batch in loader:
        x, _, _, _, xy, visible = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, offsets = net(x.to(device, non_blocking=True))
        predicted, conf = H.decode(logits, offsets)
        xy, visible = xy.numpy(), visible.numpy()
        distance = np.linalg.norm(predicted - xy, axis=2)
        errors += distance[visible].tolist()
        confident = conf >= CONFIDENT
        confident_errors += distance[visible & confident].tolist()
        seen, hits = seen + int(visible.sum()), hits + int((visible & confident).sum())
        unseen, spurious = unseen + int((~visible).sum()), spurious + int((~visible & confident).sum())
    net.train()
    return {"px_all_median": float(np.median(errors)), "px_all_p90": float(np.quantile(errors, 0.9)),
            "px_median": float(np.median(confident_errors)) if confident_errors else math.nan,
            "recall": hits / max(seen, 1), "spurious": spurious / max(unseen, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backbone", default="hrnet_w32.ms_in1k")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000, help="iterations")
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-step", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-on-train", action="store_true")
    parser.add_argument("--dataset", type=Path, default=T.DATASET)
    parser.add_argument("--runs", type=Path, default=T.RUNS)
    args = parser.parse_args()
    run_dir = args.runs / args.name
    if args.resume:
        saved = json.loads((run_dir / "args.json").read_text())
        args = argparse.Namespace(**(saved | {"resume": True}))
        args.dataset, args.runs = Path(args.dataset), Path(args.runs)
    elif (run_dir / "last.pt").exists():
        raise SystemExit(f"{run_dir} has a checkpoint; pass --resume or use a new --name")
    device = "cuda"
    started, clock = datetime.now(UTC), time.perf_counter()
    yolo_script.log_session(run_dir, {"event": "start", "at": started.isoformat(timespec="seconds"),
                                      "mode": "resume" if args.resume else "train",
                                      "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                      "git": records.git_state(), "libraries": records.library_versions()})
    status = "failed"
    try:
        with no_statsbomb_reads():
            train(args, run_dir, device)
        status = "finished"
    finally:
        yolo_script.log_session(run_dir, {"event": "end", "status": status, "mode": "resume" if args.resume else "train",
                                          "at": datetime.now(UTC).isoformat(timespec="seconds"),
                                          "seconds": round(time.perf_counter() - clock, 1)})


def train(args: argparse.Namespace, run_dir: Path, device: str) -> None:
    torch.manual_seed(args.seed)
    if not args.resume:
        data_yaml = T.training_yaml(args.dataset, run_dir / "data", val_step=args.val_step,
                                    train_limit=args.train_limit, seed=args.seed)
        (run_dir / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items() if k != "resume"}, indent=2))
    data = yaml.safe_load((run_dir / "data/data.yaml").read_text()) if args.resume else yaml.safe_load(data_yaml.read_text())
    train_images = image_list(run_dir / "data/train.txt")
    val_images = train_images if args.val_on_train else image_list(run_dir / "data/val.txt")
    train_set = H.KeypointImages(train_images, data["flip_idx"], H.Augment(), seed=args.seed)
    val_set = H.KeypointImages(val_images, data["flip_idx"], None)
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                        pin_memory=True, persistent_workers=False, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    net = H.HeatmapNet(H.HeatmapConfig(backbone=args.backbone, decoder_channels=args.decoder_channels)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(loader)

    def factor(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start, best = 0, math.inf
    if args.resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]
    results = run_dir / "results.csv"
    for epoch in range(start, args.epochs):
        train_set.epoch = epoch
        epoch_clock, sums, count = time.perf_counter(), {"focal": 0.0, "offset": 0.0}, 0
        for x, heatmaps, index, offset, _, _ in loader:
            x, heatmaps = x.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True)
            index, offset = index.to(device, non_blocking=True), offset.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, offsets = net(x)
            focal = H.focal_loss(logits, heatmaps)
            off = H.offset_loss(offsets, index, offset)
            loss = focal + args.offset_weight * off
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            sums["focal"] += float(focal)
            sums["offset"] += float(off)
            count += 1
            if count % 200 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} iter {count}/{len(loader)} focal {sums['focal'] / count:.4f} "
                      f"offset {sums['offset'] / count:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{time.perf_counter() - epoch_clock:.0f} s", flush=True)
        metrics = validate(net, val_loader, device)
        row = {"epoch": epoch + 1, "seconds": round(time.perf_counter() - epoch_clock, 1),
               "train_focal": sums["focal"] / count, "train_offset": sums["offset"] / count} | metrics
        print(json.dumps(row), flush=True)
        new = not results.exists()
        with results.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        if metrics["px_all_median"] < best:
            best = metrics["px_all_median"]
            H.save_checkpoint(run_dir / "best.pt", net, epoch=epoch, metrics=metrics)
        H.save_checkpoint(run_dir / "last.pt.tmp", net, epoch=epoch, best=best, optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), metrics=metrics)
        (run_dir / "last.pt.tmp").replace(run_dir / "last.pt")


if __name__ == "__main__":
    main()
