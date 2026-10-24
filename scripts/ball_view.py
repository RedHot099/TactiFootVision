"""ball-detection 05: the training views of the ball fine-tune and its control arm.

    uv run python scripts/ball_view.py build          # runs/bd05_{A,B}: lists, data.yaml, class masks, bins
    uv run python scripts/detector_train.py train --run bd05_A   # then bd05_B (see runs/bd05_*/RUN.md)

Both arms fine-tune dd2c on run 2c's recipe (`RUNS["bd05_*"]` in scripts/detector_train.py); only the list differs.

* **A, control:** run 2c's training list unchanged (run 1's sets + football_yolo x20; benchmark games, their
  near-duplicates and the Ekstraklasa dev clips out). Its list hash must equal run 2c's.
* **B, ball:** the same images, with every image in ticket 01's failure bins listed ``FACTOR`` times as often as in A
  (football_yolo's x20 becomes x40). An image is in the bins when one of its truth balls is
  - small: diameter (mean of the box sides) < ``SMALL_PX`` at 1080p-equivalent scale, i.e. the box in the
    1280 network input times 1920 / 1280 (``diameter * 1920 / max(width, height)``; for the 16:9 GSR frames this is
    ticket 01's "< 12 px at 1080p"), or
  - blurred: image speed >= ``ball_errors.BLUR_SPEED`` ball diameters per frame, from the GSR labels' ``track_id``
    at 25 fps (ticket 01's ``track_motion``). Only GSR train has tracks; every other source (SoccerNet-v3 stills,
    ISSIA, TeamTrack, Soccana, Roboflow, football_yolo) is binned by size alone.
  To keep A's epoch length, odd GSR train frames are dropped (seeded order, blind to the bins) until the extra lines
  are gone. Even frames are never dropped, so a dropped frame's neighbours f - 1 and f + 1 (40 ms apart, nearly the
  same picture) stay in the view.

Outputs per arm under `runs/bd05_<arm>/`: `train.txt`, `val.txt` (run 2c's), `data.yaml`, `annotated_classes.parquet`,
`smoke_data/`, and for B `bins.parquet` (per image: size, blur, bin, times listed in A and B). The record goes to
`research/detector_bd05_<arm>_view.json` (per source and per bin: images and lines per epoch).
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import detector_bench as bench  # noqa: E402
import detector_train as dt  # noqa: E402

FACTOR = 2  # ticket 01's suggestion (x3 was an optional, descriptive sensitivity arm: not run)
SMALL_PX = 12.0
INPUT_LONG_SIDE = 1920  # the 1080p-equivalent scale: GSR frames are 1920 x 1080
THIN_SEED = 0
ARMS = {"bd05_A": 1, "bd05_B": FACTOR}  # run -> factor of the in-bin images
BIN_KEYS = ("small", "blur", "small_or_blur", "none")

logger = logging.getLogger("ball_view")


# ------------------------------------------------------------------ bins
def label_path(image: str) -> Path:
    """Ultralytics' images/ -> labels/ rule."""
    return Path(image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def ball_diameters(label: Path, width: int, height: int) -> np.ndarray:
    """1080p-equivalent diameters of the ball boxes (class 0) of a YOLO label file."""
    rows = (
        np.loadtxt(label, ndmin=2) if label.stat().st_size else np.zeros((0, 5), float)
    )
    rows = rows[rows[:, 0] == 0]
    scale = INPUT_LONG_SIDE / max(width, height)
    return (rows[:, 3] * width + rows[:, 4] * height) / 2 * scale


def smallest_ball(view: pd.DataFrame, workers: int = 32) -> pd.Series:
    """Per image the smallest truth ball's 1080p-equivalent diameter (NaN without a ball)."""

    def one(row) -> float:
        d = ball_diameters(label_path(row.image), row.width, row.height)
        return float(d.min()) if len(d) else np.nan

    with ThreadPoolExecutor(workers) as pool:
        values = list(pool.map(one, view.itertuples()))
    return pd.Series(values, index=view.index, name="diameter")


def frame_blur(balls: pd.DataFrame) -> pd.DataFrame:
    """Per (clip, frame) of a GSR ball table (``ball_errors.clip_labels``): ``blur`` (a ball moves >= BLUR_SPEED
    diameters per frame in the image) and ``blur_known`` (some ball of the frame has a speed)."""
    moving = be.track_motion(balls)
    moving["speed_d"] = moving.speed_px / be.diameter(np.array(moving.box.tolist()))
    frames = moving.groupby(["clip", "frame"]).speed_d.agg(
        blur=lambda s: bool((s >= be.BLUR_SPEED).any()),
        blur_known=lambda s: bool(s.notna().any()),
    )
    return frames.reset_index()


def gsr_blur(view: pd.DataFrame) -> pd.DataFrame:
    """``frame_blur`` of every GSR train clip of the view, from its Labels-GameState.json."""
    clips = sorted(view.loc[is_gsr_train(view), "clip"].unique())
    tables = []
    for clip in clips:
        balls, _ = be.clip_labels(bench.GSR / "train" / clip / "Labels-GameState.json")
        tables.append(frame_blur(balls))
    return pd.concat(tables, ignore_index=True)


def is_gsr_train(view: pd.DataFrame) -> pd.Series:
    return (view.source == "soccernet_gsr") & (view.gsr_split == "train").fillna(False)


def ball_bins(
    view: pd.DataFrame, diameter: pd.Series, blur: pd.DataFrame
) -> pd.DataFrame:
    """Per image: ``small``, ``blur`` (False where unknown), ``blur_known`` and ``in_bin`` = small or blur."""
    table = view[["image", "source", "clip", "frame"]].copy()
    table["diameter"] = diameter
    table["small"] = (diameter < SMALL_PX).to_numpy()
    keys = blur.assign(frame=blur.frame.astype("Int64"))
    merged = table[["clip", "frame"]].merge(keys, on=["clip", "frame"], how="left")
    gsr = is_gsr_train(view).to_numpy()
    table["blur"] = gsr & merged.blur.eq(True).to_numpy()
    table["blur_known"] = gsr & merged.blur_known.eq(True).to_numpy()
    table["in_bin"] = table.small | table.blur
    return table


# ------------------------------------------------------------------ listing
def base_times(view: pd.DataFrame, oversample: dict[str, int]) -> np.ndarray:
    """How often A lists each image (run 2c: football_yolo x20)."""
    return view.set.map(lambda s: oversample.get(s, 1)).to_numpy(int)


def oversample_bins(times: np.ndarray, in_bin: np.ndarray, factor: int) -> np.ndarray:
    return np.where(in_bin, times * factor, times)


def thin(
    view: pd.DataFrame, times: np.ndarray, excess: int, seed: int = THIN_SEED
) -> np.ndarray:
    """Drop odd GSR train frames, in a seeded order blind to the bins, until ``excess`` lines are gone; a frame
    listed more than the lines still to remove is skipped."""
    times = times.copy()
    candidates = np.flatnonzero(is_gsr_train(view).to_numpy() & (view.frame % 2 == 1))
    order = np.random.default_rng(seed).permutation(candidates)
    left = excess
    for i in order:
        if left == 0:
            break
        if 0 < times[i] <= left:
            left -= times[i]
            times[i] = 0
    if left:
        raise ValueError(
            f"thinning odd GSR train frames removes at most {excess - left} of {excess} lines"
        )
    return times


def arm_times(
    view: pd.DataFrame, in_bin: np.ndarray, oversample: dict[str, int], factor: int
) -> np.ndarray:
    """Per image how often the arm lists it: A's times, in-bin images x ``factor``, thinned to A's epoch length."""
    base = base_times(view, oversample)
    if factor == 1:
        return base
    raised = oversample_bins(base, in_bin, factor)
    return thin(view, raised, int(raised.sum() - base.sum()))


def listing(view: pd.DataFrame, times: np.ndarray) -> list[str]:
    return sorted(np.repeat(view.image.to_numpy(), times).tolist())


def bin_flags(bins: pd.DataFrame) -> dict[str, np.ndarray]:
    small, blur = bins.small.to_numpy(), bins.blur.to_numpy()
    return {
        "small": small,
        "blur": blur,
        "small_or_blur": small | blur,
        "none": ~(small | blur),
    }


def bin_counts(bins: pd.DataFrame, times: np.ndarray) -> dict:
    """Per bin and per source: distinct images listed and lines per epoch, and the share of the epoch's lines."""
    total = int(times.sum())
    listed = times > 0

    def block(mask: np.ndarray) -> dict:
        out = {}
        for key, flag in bin_flags(bins).items():
            keep = mask & flag
            lines = int(times[keep].sum())
            out[key] = {
                "images": int((keep & listed).sum()),
                "lines": lines,
                "epoch_share": round(lines / total, 4) if total else 0.0,
            }
        return out

    per_source = {
        source: block((bins.source == source).to_numpy())
        for source in sorted(bins.source.unique())
    }
    gsr = (bins.source == "soccernet_gsr").to_numpy()
    return {
        "epoch_lines": total,
        "all": block(np.ones(len(bins), bool)),
        "per_source": per_source,
        "gsr_train": {
            "frames": int(gsr.sum()),
            "frames_listed": int((gsr & listed).sum()),
            "frames_dropped": int((gsr & ~listed).sum()),
            "frames_with_blur_known": int(bins.blur_known[gsr].sum()),
        },
    }


# ------------------------------------------------------------------ build
def write_arm(
    run: str,
    view: pd.DataFrame,
    times: np.ndarray,
    bins: pd.DataFrame,
    pool_images: int,
) -> dict:
    spec, out = dt.RUNS[run], dt.run_dir(run)
    (out / "smoke_data").mkdir(parents=True, exist_ok=True)
    images = listing(view, times)
    kept = view[times > 0]
    val = dt.val_images(run)
    label_check = {
        "train": dt.check_label_paths(sorted(set(images))),
        "val": dt.check_label_paths(val),
    }
    (out / "train.txt").write_text("\n".join(images) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    dt.write_yaml(out / "data.yaml", "train.txt", "val.txt")
    kept[["image", "source", "annotated_classes"]].reset_index(drop=True).to_parquet(
        out / "annotated_classes.parquet", index=False
    )
    rng = np.random.default_rng(dt.SMOKE["seed"])
    pick = sorted(
        rng.choice(images, min(dt.SMOKE["train_images"], len(images)), replace=False)
    )
    (out / "smoke_data/train.txt").write_text("\n".join(pick) + "\n")
    (out / "smoke_data/val.txt").write_text(
        "\n".join(val[:: dt.SMOKE["val_every"] // dt.VAL_EVERY]) + "\n"
    )
    dt.write_yaml(out / "smoke_data/data.yaml", "train.txt", "val.txt")

    factor = ARMS[run]
    train = dt.counts(kept, spec["oversample"])
    for source, rows in bins.groupby("source"):
        train["per_source"][source]["images_per_epoch"] = int(times[rows.index].sum())
    train["images_per_epoch"] = int(times.sum())
    record = {
        "title": spec["title"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": "scripts/ball_view.py",
        "pool_index": str(dt.POOL),
        "pool_images": pool_images,
        "rule": {
            "base_view": "run 2c's view (scripts/detector_train.py build_view + train_list)",
            "sets": list(spec["sets"]),
            "excluded_games": sorted(dt.BENCH_GAMES),
            "excluded_near_dup_sets": sorted(dt.BENCH_DUP_SETS),
            "excluded_gsr_splits": ["valid", "test"],
            "held_out_splits": spec["holdout"],
            "oversample_by_set": spec["oversample"],
            "bin_factor": factor,
            "bins": {
                "small": f"some truth ball has diameter * {INPUT_LONG_SIDE} / max(width, height) < {SMALL_PX} px",
                "blur": f"GSR train only: some ball moves >= {be.BLUR_SPEED} diameters per frame (track_id, 25 fps)",
                "sources_without_tracks": "binned by size alone",
            },
            "thinning": (
                "odd GSR train frames dropped in a seeded order (seed "
                f"{THIN_SEED}), blind to the bins, until the list has A's length; even frames always kept"
                if factor > 1
                else "none"
            ),
            "class_mask": "per image, the source's annotated_classes; classes outside it get no cls loss",
        },
        "init": str(spec["init"]),
        "after": spec["after"],
        "settings": spec["settings"],
        "train": train | {"list_sha256": dt.sha256_lines(images)},
        "bins": bin_counts(bins, times),
        "val": {
            "rule": f"run 2c's: every {dt.VAL_EVERY}th GSR valid bench frame, then every Ekstraklasa dev image",
            "images": len(val),
            "list_sha256": dt.sha256_lines(val),
        },
        "label_paths": label_check,
        "files": {
            "data_yaml": str(out / "data.yaml"),
            "annotated_classes": str(out / "annotated_classes.parquet"),
            "smoke_data_yaml": str(out / "smoke_data/data.yaml"),
        },
    }
    if factor > 1:
        record["files"]["bins"] = str(out / "bins.parquet")
    dt.view_json(run).write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


def build(args: argparse.Namespace) -> None:
    pool = pd.read_parquet(dt.POOL)
    view = dt.build_view(pool, "bd05_A").sort_values("image").reset_index(drop=True)
    assert set(dt.build_view(pool, "bd05_B").image) == set(view.image)
    logger.info("view: %d images; reading ball sizes and GSR train tracks", len(view))
    bins = ball_bins(view, smallest_ball(view), gsr_blur(view))
    reference = json.loads(dt.view_json("run2c").read_text())["train"]["list_sha256"]
    for run, factor in ARMS.items():
        times = arm_times(
            view, bins.in_bin.to_numpy(), dt.RUNS[run]["oversample"], factor
        )
        record = write_arm(run, view, times, bins, len(pool))
        if factor == 1 and record["train"]["list_sha256"] != reference:
            raise SystemExit(
                f"{run}: the list differs from run 2c's ({reference[:12]})"
            )
        if factor > 1:
            base = base_times(view, dt.RUNS[run]["oversample"])
            bins.assign(times_a=base, times_b=times).to_parquet(
                dt.run_dir(run) / "bins.parquet", index=False
            )
        summary = record["bins"]["all"]
        logger.info(
            "%s: %d lines, sha %s; %s",
            run,
            record["train"]["images_per_epoch"],
            record["train"]["list_sha256"][:12],
            json.dumps(summary),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    b = commands.add_parser("build")
    b.set_defaults(handler=build)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.handler(args)


if __name__ == "__main__":
    main()
