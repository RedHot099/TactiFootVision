"""ball-detection 16: the xG read (descriptive) of the ball pick at the shot frames, phase 1 vs ticket 13's second stage.

    XG=/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg.py detect   # dd2c >= 0.01 + second stage, -800 ms frames (GPU)
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg.py read     # arms -> feet_t0 features -> table (CPU)

Both commands run in the xG experiment's environment (``run.sh``: xg-detector's ``src`` and ``xgpoc`` on the path),
because the read replays the xG cache with the frozen xG code (``xgpoc.detector.location.FEET_T0``) and the detector
must be the one the cache was built with.

Population: the 7,062 non-penalty shots of the 328 development matches (``plan.csv`` role ``training``, the stack's
training population); none of the 172 held-out matches. Only label-free index columns are read (``INDEX_COLUMNS``).
StatsBomb is not read: its pairing covers only the 135 evaluation matches, all inside the 172.

Arms (one frame each, at 0, -400 and -800 ms; people, teams, keypoints and homographies stay the cache's, only the
ball is swapped, placed at its box centre by the frame's own homography as the package does):

    X0  xG today: dd2c 1280's most confident ball >= 0.15, no filter (must reproduce ``cache/xgd_full4_feet_t0``)
    P1  phase 1's single-frame candidate: X0 after the 45 px post-filter (long side at 1080p)
    S2  ticket 13's second stage, row (ii): dd2c's ball candidates >= 0.01 rescored by the crop net and the fusion,
        boxes moved by the offset head, the 45 px post-filter, the most confident fused ball >= 0.15

Registration: ticket 16's comment of 2026-10-26 22:16.
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402

logger = logging.getLogger("ball_xg")

XG = Path(
    "/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable"
)
CACHE = XG / "cache/xgd_full4/E3"
FEET_CACHE = XG / "cache/xgd_full4_feet_t0/E3"
PLAN = XG / "results/xgv_split_plan/plan.csv"
OUT = be.ROOT / "ball/16/xg"
FUSION = be.ROOT / "ball/13/fusion/fusion.json"
INDEX_COLUMNS = [
    "shot_id",
    "game",
    "is_penalty",
    "frame_path",
    "frame_path_-400",
    "frame_path_-800",
]  # no label
FRAME_COLUMNS = {0: "frame_path", -400: "frame_path_-400", -800: "frame_path_-800"}
OFFSETS = (0, -400, -800)
DETECTOR = {
    "conf": 0.3,
    "iou": 0.5,
    "imgsz": 1280,
    "ball_conf": 0.01,
}  # xG's dd2c_1280, ball floor lowered
TOP1 = 0.15  # xG's ball class_conf and ticket 13's single-frame threshold
POST_FILTER = 45.0  # px of the long side at 1080p (ticket 03 / phase 1)
ARMS = ("X0", "P1", "S2")
BOX = ["x1", "y1", "x2", "y2"]
MOVED = ["mx1", "my1", "mx2", "my2"]
NEAR_M = 2.0  # proxy: shooter's feet within 2 m of the ball's projection
DRAWS, SEED = 2000, 0


# ------------------------------------------------------------------ pure parts
def population(index: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """The non-penalty shots of the plan's training matches, index shot_id, sorted."""
    role = index.game.map(plan.set_index("game").role)
    shots = index[(role == "training") & ~index.is_penalty.astype(bool)]
    return shots.drop(columns="is_penalty").set_index("shot_id").sort_index()


def long_side_1080p(boxes: pd.DataFrame, height: pd.Series | np.ndarray) -> np.ndarray:
    return (
        np.maximum(boxes.x2 - boxes.x1, boxes.y2 - boxes.y1).to_numpy(float)
        * 1080
        / np.asarray(height, float)
    )


def choose(candidates: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per (shot_id, offset_ms) the arm's ball: ``BOX`` and ``conf`` (S2: the moved box and the fused confidence).
    ``candidates`` holds one row per ball candidate in the detector's order (most confident first): shot_id,
    offset_ms, height, ``BOX``, conf, fused, ``MOVED``. Ties keep the first row, as ``FrameAnalyzer`` does."""
    c = candidates
    if arm == "S2":
        c = c.drop(columns=BOX + ["conf"]).rename(
            columns=dict(zip(MOVED, BOX, strict=True)) | {"fused": "conf"}
        )
    if arm in ("P1", "S2"):
        c = c[long_side_1080p(c, c.height) <= POST_FILTER]
    c = c[c.conf >= TOP1]
    best = c.loc[c.groupby(["shot_id", "offset_ms"], sort=True).conf.idxmax()]
    return best.set_index(["shot_id", "offset_ms"])[BOX + ["conf"]].astype(float)


def swap_ball(row: dict, ball: pd.Series | None, project) -> dict:
    """A frames.parquet row with its ball replaced by ``ball`` (None: no ball). ``project(points, H)`` places the
    box centre with the row's homography (NaN without one, as ``project_detections``)."""
    out = dict(row)
    if not row["readable"]:
        return out
    if ball is None:
        out |= {"ball_conf": [], "ball_xyxy": [], "ball_pitch_xy": []}
        return out
    box = [float(ball[k]) for k in BOX]
    centre = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
    homography = row["homography"]
    if homography is None or (isinstance(homography, float) and np.isnan(homography)):
        pitch = [float("nan"), float("nan")]
    else:
        pitch = [
            float(v)
            for v in np.asarray(
                project(centre, np.asarray(homography, float).reshape(3, 3))
            ).ravel()
        ]
    out |= {
        "ball_conf": [float(ball["conf"])],
        "ball_xyxy": box,
        "ball_pitch_xy": pitch,
    }
    return out


def has_homography(rows: pd.DataFrame) -> pd.Series:
    """Per frames row: readable with a homography (a ball there lands on the pitch)."""
    return rows.readable.astype(bool) & rows.homography.map(
        lambda h: h is not None and not (isinstance(h, float) and np.isnan(h))
    )


def needs_800(chosen: pd.DataFrame, frames: pd.DataFrame) -> set[str]:
    """Shots whose -400 ms row has no ball on the pitch under the arm, so the feet_t0 order (both rows) reads -800."""
    rows = frames[frames.offset_ms == -400].set_index("shot_id")
    on_pitch = has_homography(rows)
    with_ball = (
        rows.index.isin(chosen.xs(-400, level="offset_ms").index)
        if len(chosen)
        else np.zeros(len(rows), bool)
    )
    return set(rows.index[~(on_pitch.to_numpy() & with_ball)])


def same_ball(a: pd.DataFrame, b: pd.DataFrame, scale: float) -> np.ndarray:
    """Whether b's centre lies within r = max(3 px at 1080p, half a's diagonal) of a's (rows aligned)."""
    ca = np.c_[(a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2]
    cb = np.c_[(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]
    r = np.maximum(3.0 * scale, np.hypot(a.x2 - a.x1, a.y2 - a.y1) / 2)
    return np.hypot(*(ca - cb).T) <= r


def paired_interval(
    diff: np.ndarray, games: np.ndarray, draws: int = DRAWS, seed: int = SEED
) -> tuple[float, float]:
    """95% interval of the mean of per-shot ``diff`` by resampling matches (pairs kept)."""
    keep = np.isfinite(diff)
    g = pd.Series(diff[keep]).groupby(games[keep])
    sums, counts = g.sum().to_numpy(), g.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), (draws, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def shot_measures(t0: pd.DataFrame) -> pd.DataFrame:
    """Per-shot indicators of a t0 feature table (index shot_id): ball_ok, own / -400 / -800 / none, shooter-ball
    proxy."""
    ok = t0.ball_ok == 1
    source = t0.ball_source.where(ok)
    out = pd.DataFrame(index=t0.index)
    out["ball_ok"] = ok.astype(float)
    out["own_t0"] = (source == 0).astype(float)
    out["fallback_m400"] = (source == 1).astype(float)
    out["fallback_m800"] = (source == 2).astype(float)
    out["no_ball"] = (~ok).astype(float)
    out["shooter_ball_dist"] = t0.shooter_ball_dist.where(ok)
    out["shooter_near_ball"] = (
        (t0.shooter_ball_dist <= NEAR_M)
        .astype(float)
        .where(ok & t0.shooter_ball_dist.notna())
    )
    return out


def location_change(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per shot with a shot location in both arms: the shift (m, aligned frame) and whether the shooter changed."""
    both = (a.shot_ok == 1) & (b.shot_ok == 1)
    shift = np.hypot(a.shot_x - b.shot_x, a.shot_y - b.shot_y).where(both)
    return pd.DataFrame(
        {
            "shift_m": shift,
            "moved": (shift > 0.01).astype(float).where(both),
            "over_1m": (shift > 1).astype(float).where(both),
            "over_5m": (shift > 5).astype(float).where(both),
        }
    )


# ------------------------------------------------------------------ inputs
def read_population(limit: int | None = None) -> pd.DataFrame:
    """The population; ``limit``: its first shots only (smoke runs)."""
    from xgpoc.paths import INDEX

    shots = population(
        pd.read_parquet(INDEX, columns=INDEX_COLUMNS),
        pd.read_csv(PLAN, usecols=["game", "role"]),
    )
    return shots.iloc[:limit] if limit else shots


def read_frames(ids: list[str]) -> pd.DataFrame:
    frames = pd.read_parquet(CACHE / "frames.parquet", filters=[("shot_id", "in", ids)])
    extra = OUT / "extra_frames.parquet"
    if extra.is_file():
        frames = pd.concat([frames, pd.read_parquet(extra)], ignore_index=True)
    if frames.duplicated(["shot_id", "offset_ms"]).any():
        raise SystemExit("a shot frame is twice in the frames table")
    return frames


# ------------------------------------------------------------------ detect (GPU)
def detect(args: argparse.Namespace) -> None:
    """dd2c's ball candidates >= 0.01 on every population frame (0, -400, -800 ms) with the second stage's fused
    confidence and moved box, then every -800 ms frame an arm needs and the cache lacks, analysed as xG does."""
    import ball_crops as bc
    import ball_temporal as bt
    import detector_bench as bench
    import torch
    from xgpoc.detector import extract as ex
    from xgpoc.detector import extract3 as x3

    import tactifoot_vision as tv

    shots = read_population(args.limit)
    weights = x3.DETECTORS["dd2c_1280"].weights
    fusion = json.loads(FUSION.read_text())
    if bench.sha256_file(Path(fusion["weights"])) != fusion["weights_sha256"]:
        raise SystemExit("the second stage's weights changed since ticket 13")
    OUT.mkdir(parents=True, exist_ok=True)
    parts = OUT / "candidates_parts"
    parts.mkdir(exist_ok=True)
    jobs = [
        (sid, off, shots.at[sid, FRAME_COLUMNS[off]])
        for sid in shots.index
        for off in OFFSETS
    ]
    model = tv.load_model(
        "yolo",
        weights,
        conf=DETECTOR["conf"],
        iou=DETECTOR["iou"],
        imgsz=DETECTOR["imgsz"],
        class_conf={"ball": DETECTOR["ball_conf"]},
        device=args.device,
    )
    stage = bc.SecondStage(Path(fusion["weights"]), fusion["fusion"], args.device)
    det_ms, stage_ms, unreadable = [], [], []
    for start in range(0, len(jobs), args.chunk):
        path = parts / f"{start:06d}.parquet"
        if path.is_file():
            continue
        chunk = jobs[start : start + args.chunk]
        tables = []
        for (sid, off, _), (_, image) in zip(
            chunk, bt.read_ahead([Path(p) for _, _, p in chunk]), strict=True
        ):
            if image is None:
                unreadable.append((sid, off))
                continue
            torch.cuda.synchronize()
            clock = time.perf_counter()
            found = model.predict(image)
            torch.cuda.synchronize()
            det_ms.append(time.perf_counter() - clock)
            balls = found[found.data["class_name"] == "ball"]
            balls = balls[np.argsort(-balls.confidence, kind="stable")]
            if not len(balls):
                continue
            boxes, conf = (
                np.asarray(balls.xyxy, float),
                np.asarray(balls.confidence, float),
            )
            clock = time.perf_counter()
            fused, moved, p_crop, _ = stage(image, boxes, conf)
            torch.cuda.synchronize()
            stage_ms.append(time.perf_counter() - clock)
            table = pd.DataFrame(boxes.astype(np.float32), columns=BOX)
            table[MOVED] = moved.astype(np.float32)
            table["conf"], table["fused"], table["p_crop"] = (
                conf.astype(np.float32),
                fused,
                p_crop,
            )
            table.insert(0, "height", image.shape[0])
            table.insert(0, "offset_ms", off)
            table.insert(0, "shot_id", sid)
            tables.append(table)
        pd.concat(tables, ignore_index=True).to_parquet(path, index=False)
        logger.info("%d / %d frames", min(start + args.chunk, len(jobs)), len(jobs))
    candidates = pd.concat(
        [pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True
    )
    candidates.to_parquet(OUT / "candidates.parquet", index=False)

    frames = read_frames(sorted(shots.index))
    cached = set(zip(frames.shot_id, frames.offset_ms, strict=True))
    wanted = sorted(
        {
            sid
            for arm in ARMS
            for sid in needs_800(choose(candidates, arm), frames)
            if (sid, -800) not in cached
        }
    )
    logger.info("%d -800 ms frames to analyse", len(wanted))
    extra_rows, extra_ms = [], []
    if wanted:
        analyzer = x3.ball_analyzer(args.device, x3.DETECTORS["dd2c_1280"])
        paths = [shots.at[sid, FRAME_COLUMNS[-800]] for sid in wanted]
        for sid, (path, image) in zip(
            wanted, bt.read_ahead([Path(p) for p in paths]), strict=True
        ):
            clock = time.perf_counter()
            result = analyzer(image) if image is not None else None
            extra_ms.append(time.perf_counter() - clock)
            extra_rows.append(ex.frame_row(sid, -800, str(path), result))
        pd.DataFrame(extra_rows).to_parquet(OUT / "extra_frames.parquet", index=False)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 16",
        "shots": len(shots),
        "matches": int(shots.game.nunique()),
        "frames": len(jobs),
        "unreadable_this_run": len(unreadable),
        "detector": {"weights": str(weights), "sha256": bench.sha256_file(weights)}
        | DETECTOR,
        "second_stage": {k: fusion[k] for k in ("weights", "weights_sha256", "fusion")}
        | {"crop": bc.CROP},
        "candidates": len(candidates),
        "extra_m800_frames": len(wanted),
        "ms_per_frame": {
            "detector": round(1000 * float(np.mean(det_ms[10:])), 2)
            if len(det_ms) > 20
            else None,
            "second_stage_with_candidates": round(1000 * float(np.mean(stage_ms)), 2)
            if stage_ms
            else None,
            "extra_analyser": round(1000 * float(np.mean(extra_ms)), 2)
            if extra_ms
            else None,
        },
    }
    (OUT / "detect.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


# ------------------------------------------------------------------ read (CPU)
def arm_frames(frames: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    from tactifoot_vision.pitch.homography import frame_to_pitch

    rows = []
    for row in frames.to_dict("records"):
        key = (row["shot_id"], int(row["offset_ms"]))
        rows.append(
            swap_ball(
                row, chosen.loc[key] if key in chosen.index else None, frame_to_pitch
            )
        )
    return pd.DataFrame(rows, columns=frames.columns)


def fmt_share(v: float) -> str:
    return f"{100 * v:.2f}%".replace(".", ",")


def fmt_pp(v: float, lo: float, hi: float) -> str:
    def f(x):
        return f"{100 * x:+.2f}".replace(".", ",").replace("-", "−")

    return f"{f(v)} [{f(lo)}; {f(hi)}]"


def read(args: argparse.Namespace) -> None:
    import cv2
    from xgpoc.detector import extract as ex
    from xgpoc.detector import location as loc
    from xgpoc.detector.features import COLUMNS

    shots = read_population(args.limit)
    ids = sorted(shots.index)
    frames = read_frames(ids)
    candidates = pd.read_parquet(OUT / "candidates.parquet")
    chosen, features = {}, {}
    for arm in ARMS:
        chosen[arm] = choose(candidates, arm)
        features[arm] = loc.feature_rows(arm_frames(frames, chosen[arm]), loc.FEET_T0)
    if not features["X0"].shot_id.drop_duplicates().sort_values().tolist() == ids:
        raise SystemExit("the arms do not cover the population")

    # control: X0 reproduces the xG cache (feet_t0 features, t0 and -400 ms rows)
    cache = pd.read_parquet(
        FEET_CACHE / "features.parquet", filters=[("shot_id", "in", ids)]
    )
    cache = cache.set_index(["shot_id", "offset_ms"]).sort_index()
    x0 = features["X0"].set_index(["shot_id", "offset_ms"]).sort_index()
    if not cache.index.equals(x0.index):
        raise SystemExit("X0's feature rows are not the cache's rows")
    differ = ex.feature_differences(cache, x0, COLUMNS, 1e-6).any(axis=1)
    control = {
        "rows": len(x0),
        "rows_differing": int(differ.sum()),
        "t0_rows_differing": int(differ.xs(0, level="offset_ms").sum()),
        "t0_ball_ok_differing": int(
            (
                cache.xs(0, level="offset_ms").ball_ok
                != x0.xs(0, level="offset_ms").ball_ok
            ).sum()
        ),
    }

    t0 = {
        arm: f[f.offset_ms == 0].set_index("shot_id").sort_index()
        for arm, f in features.items()
    }
    games = shots.game.reindex(ids).to_numpy()
    per = {arm: shot_measures(t0[arm]) for arm in ARMS}
    names = [
        "ball_ok",
        "own_t0",
        "fallback_m400",
        "fallback_m800",
        "no_ball",
        "shooter_near_ball",
    ]
    table = []
    for name in names:
        row = {"measure": name} | {arm: float(per[arm][name].mean()) for arm in ARMS}
        for a, b in (("P1", "X0"), ("S2", "P1")):
            d = (per[a][name] - per[b][name]).to_numpy(float)
            row[f"{a}-{b}"] = float(np.nanmean(d))
            row[f"{a}-{b}_ci"] = paired_interval(d, games)
        table.append(row)
    medians = {arm: float(per[arm].shooter_ball_dist.median()) for arm in ARMS}

    change = {
        f"{a}_vs_{b}": location_change(t0[a], t0[b])
        for a, b in (("P1", "X0"), ("S2", "P1"))
    }
    shifts = {}
    for key, c in change.items():
        entry = {"shots_with_both": int(c.shift_m.notna().sum())}
        for col in ("moved", "over_1m", "over_5m"):
            entry[col] = float(c[col].mean())
            entry[f"{col}_ci"] = paired_interval(c[col].to_numpy(float), games)
        entry["median_shift_when_moved_m"] = float(c.shift_m[c.moved == 1].median())
        shifts[key] = entry

    # the t0 ball itself (own frame): agreement P1 vs S2
    scale = 720 / 1080
    p1 = chosen["P1"].xs(0, level="offset_ms").reindex(ids)
    s2 = chosen["S2"].xs(0, level="offset_ms").reindex(ids)
    both = p1.conf.notna() & s2.conf.notna()
    same = pd.Series(False, index=ids, dtype=bool)
    same[both] = np.asarray(same_ball(p1[both], s2[both], scale), bool)
    kinds = pd.Series("neither", index=ids)
    kinds[both & same] = "same"
    kinds[both & ~same] = "different"
    kinds[p1.conf.notna() & s2.conf.isna()] = "P1 only"
    kinds[p1.conf.isna() & s2.conf.notna()] = "S2 only"
    agreement = {k: int(v) for k, v in kinds.value_counts().items()}
    x0b = chosen["X0"].xs(0, level="offset_ms").reindex(ids)
    filtered = x0b.conf.notna() & ~(
        p1.conf.notna() & (x0b[BOX].to_numpy() == p1[BOX].to_numpy()).all(axis=1)
    )
    change_s2 = change["S2_vs_P1"]
    by_kind = {}
    for kind, idx in kinds.groupby(kinds).groups.items():
        by_kind[kind] = {
            "shots": len(idx),
            "location_moved": float(change_s2.moved.loc[idx].mean()),
            "location_over_5m": float(change_s2.over_5m.loc[idx].mean()),
            "p1_ball_source": {
                str(k): int(v)
                for k, v in t0["P1"]
                .ball_source.loc[idx]
                .value_counts(dropna=False)
                .items()
            },
            "shooter_near_ball_p1": float(per["P1"].shooter_near_ball.loc[idx].mean()),
            "shooter_near_ball_s2": float(per["S2"].shooter_near_ball.loc[idx].mean()),
            "median_conf_p1": float(p1.conf.loc[idx].median()),
            "median_fused_s2": float(s2.conf.loc[idx].median()),
        }

    # gallery of t0 disagreements (for Jakub's eye, no labels)
    gallery = OUT / "gallery"
    gallery.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    picks = []
    for kind, n in (("different", 24), ("P1 only", 8), ("S2 only", 8)):
        pool = kinds.index[kinds == kind].to_numpy()
        picks += [(kind, s) for s in rng.choice(pool, min(n, len(pool)), replace=False)]
    listing = []
    for i, (kind, sid) in enumerate(picks):
        image = cv2.imread(shots.at[sid, "frame_path"])
        for arm, colour in (("P1", (0, 140, 255)), ("S2", (255, 120, 0))):
            box = chosen[arm].loc[(sid, 0)] if (sid, 0) in chosen[arm].index else None
            if box is None:
                continue
            cx, cy = int((box.x1 + box.x2) / 2), int((box.y1 + box.y2) / 2)
            cv2.circle(image, (cx, cy), 14, colour, 2)
            cv2.putText(
                image,
                f"{arm} {box.conf:.2f}",
                (cx + 16, cy - 8 if arm == "P1" else cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )
        name = f"{i:02d}_{kind.replace(' ', '_')}.jpg"
        cv2.imwrite(str(gallery / name), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        listing.append({"file": name, "kind": kind, "shot_id": sid})
    pd.DataFrame(listing).to_csv(gallery / "index.csv", index=False)

    result = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 16",
        "registration": "ticket 16 comment 2026-10-26 22:16",
        "population": {"shots": len(ids), "matches": int(shots.game.nunique())},
        "control_x0_vs_xg_cache": control,
        "t0_rule": "feet_t0 (order 0, -400, -800 ms)",
        "measures": table,
        "shooter_ball_dist_median_m": medians,
        "location_change": shifts,
        "t0_ball_agreement_p1_s2": agreement,
        "t0_ball_removed_by_filter_x0_to_p1": int(filtered.sum()),
        "by_t0_agreement": by_kind,
        "m800_frames_read": int((frames.offset_ms == -800).sum()),
        "gallery": str(gallery),
    }
    (OUT / "read.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "| miara (wiersz t0, feet_t0) | X0 xG dziś | P1 faza 1 | S2 drugi stopień 13 | P1 − X0, pp [95%] | "
        "S2 − P1, pp [95%] |",
        "|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['measure']} | {fmt_share(r['X0'])} | {fmt_share(r['P1'])} | {fmt_share(r['S2'])} | "
            f"{fmt_pp(r['P1-X0'], *r['P1-X0_ci'])} | {fmt_pp(r['S2-P1'], *r['S2-P1_ci'])} |"
        )
    (OUT / "read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "control_x0_vs_xg_cache",
                    "shooter_ball_dist_median_m",
                    "location_change",
                    "t0_ball_agreement_p1_s2",
                    "t0_ball_removed_by_filter_x0_to_p1",
                    "by_t0_agreement",
                )
            },
            indent=2,
        )
    )


def main() -> None:
    global OUT
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=2000)
    p.set_defaults(func=detect)
    p = sub.add_parser("read")
    p.set_defaults(func=read)
    for p in sub.choices.values():
        p.add_argument(
            "--out",
            type=Path,
            default=OUT,
            help="results folder (smoke runs elsewhere)",
        )
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="first N population shots only (smoke)",
        )
    args = parser.parse_args()
    OUT = args.out
    args.func(args)


if __name__ == "__main__":
    main()
