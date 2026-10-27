"""ball-detection 23: does the 45 px post-filter survive other resolutions and scene scales, and is a filter relative to
the people's height sturdier? (registration: ``.scratch/ball-detection/issues/23-postfilter-scale.md``)

    uv run python scripts/ball_filter_scale.py dump       # GPU: dd2c 1280 on games 2+3 frames at 1080p/720p/540p, xG frames 720p
    uv run python scripts/ball_filter_scale.py read       # CPU: the filters on the dumps -> ball/23/read.json, read.md

Dump boxes are in the pixels of the (resized) frame; every size below is converted to 1080p pixels (x 1080 / frame
height), the unit of the 45 px rule. Filters: ``none``, ``abs45`` (long side > 45 px), ``rel`` (long side > k x median
person height of the frame, k a multiple of k0 = 45 / median person height of the native games 2+3 placement frames).
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_crops as bc  # noqa: E402
import ball_errors as be  # noqa: E402
import ball_projection as bp  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

OUT = be.ROOT / "ball/23"
WEIGHTS = bt.WEIGHTS
SCALES = {"1080": 1080, "720": 720, "540": 540}  # frame heights; 1080 = native
LIMIT = bs.RULE["post_filter_long_side_px"]
DUMP_CONF = bt.DUMP_CONF
PEOPLE_CONF = bench.OPERATING["conf"]
BALL_CONF = bs.BALL_CONF
MIN_PEOPLE = 3
MULTIPLES = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0)  # of k0, registered
TIE_PP = 0.05
LARGE_FACTOR = (
    2.0  # "large" false ball: long side > this x the truth ball's in the frame
)
XG_DIR = be.ROOT / "ball/16/xg"
XG_FRAMES = Path(
    "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/frames"
)
DUMP_02 = be.ROOT / "ball/02/dd2c/dump"
TOP1_17 = be.ROOT / "ball/17/placement_games_2_3_top1.parquet"
COLUMNS = ["x1", "y1", "x2", "y2"]

logger = logging.getLogger("ball_filter_scale")


# ------------------------------------------------------------------ pure helpers
def long_side(table: pd.DataFrame) -> np.ndarray:
    """Long side of the boxes, in the table's pixels."""
    return np.maximum(table.x2 - table.x1, table.y2 - table.y1).to_numpy(float)


def to_1080(table: pd.DataFrame) -> pd.DataFrame:
    """Boxes of a table with a ``height`` column in 1080p pixels."""
    out = table.copy()
    scale = 1080 / out["height"].to_numpy(float)
    for c in COLUMNS:
        out[c] = out[c].to_numpy(float) * scale
    return out


def person_heights(
    people: pd.DataFrame, keys: list[str], group: str | None
) -> pd.DataFrame:
    """Per frame (``keys``) the median person height (1080p px, ``people`` in 1080p, conf >= ``PEOPLE_CONF``);
    frames with fewer than ``MIN_PEOPLE`` people take the median of the frame medians of their ``group`` (None: all)."""
    people = people[people.conf >= PEOPLE_CONF]
    height = (people.y2 - people.y1).rename("h")
    grouped = pd.concat([people[keys], height], axis=1).groupby(keys)["h"]
    out = pd.DataFrame({"hmed": grouped.median(), "n_people": grouped.size()})
    return out.reset_index()


def fill_heights(
    frames: pd.DataFrame, heights: pd.DataFrame, keys: list[str], group: str | None
) -> pd.DataFrame:
    """``frames`` (every dumped frame) with ``hmed``: the frame's median, or the group's median of frame medians when
    the frame has < ``MIN_PEOPLE`` people."""
    out = frames.merge(heights, on=keys, how="left")
    out["n_people"] = out["n_people"].fillna(0).astype(int)
    ok = out.n_people >= MIN_PEOPLE
    out.loc[~ok, "hmed"] = np.nan
    if group is None:
        out["hmed"] = out["hmed"].fillna(out.loc[ok, "hmed"].median())
    else:
        out["hmed"] = out["hmed"].fillna(out.groupby(group)["hmed"].transform("median"))
        out["hmed"] = out["hmed"].fillna(out.loc[ok, "hmed"].median())
    return out


def keep_mask(rule: tuple, side_1080: np.ndarray, hmed: np.ndarray) -> np.ndarray:
    """Which ball candidates survive: ``("none",)``, ``("abs", limit)`` or ``("rel", k)``."""
    kind = rule[0]
    if kind == "none":
        return np.ones(len(side_1080), bool)
    if kind == "abs":
        return side_1080 <= rule[1]
    if kind == "rel":
        return side_1080 <= rule[1] * hmed
    raise ValueError(rule)


def top1(balls: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The most confident row per frame (``keys``) of ``balls`` (already filtered to >= ``BALL_CONF``)."""
    if balls.empty:
        return balls
    return balls.loc[balls.groupby(keys)["conf"].idxmax()]


def choose_multiple(well: dict[float, float]) -> float:
    """The multiple of k0 with the highest well-placed share (pp); ties within ``TIE_PP`` -> the smaller multiple."""
    best = max(well.values())
    return min(m for m, v in well.items() if 100 * (best - v) < TIE_PP)


# ------------------------------------------------------------------ dump (GPU)
def gsr_frames(clips: list[str]) -> list[tuple[str, Path]]:
    out = []
    for clip in clips:
        out += [
            (clip, p)
            for p in sorted((bench.GSR / "valid" / clip / "img1").glob("*.jpg"))
            if (int(p.stem) - 1) % bench.STRIDE == 0
        ]
    return out


def resize(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = round(image.shape[1] * height / image.shape[0])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def table_of(found, **keys) -> pd.DataFrame:
    found = found[found.confidence >= DUMP_CONF]
    table = pd.DataFrame(
        np.asarray(found.xyxy, np.float32).reshape(-1, 4), columns=COLUMNS
    )
    table["conf"] = np.asarray(found.confidence, np.float32)
    table["class_id"] = np.asarray(found.class_id, np.int8)
    for k, v in keys.items():
        table.insert(0, k, v)
    return table


def dump(args: argparse.Namespace) -> None:
    clips = bt.games_clips(bt.SELECTION)
    predict = tv.load_model(
        "yolo",
        WEIGHTS,
        conf=DUMP_CONF,
        iou=bench.ROWS["1280"]["iou"],
        imgsz=1280,
        device=args.device,
    ).predict
    args.out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, list[float]] = {}
    paths = gsr_frames(clips)
    for name, height in SCALES.items():
        part = args.out / f"gsr_{name}.parquet"
        if part.is_file():
            continue
        tables, times = [], []
        for (clip, path), image in zip(
            paths,
            (im for _, im in bt.read_ahead([p for _, p in paths])),
            strict=True,
        ):
            image = resize(image, height)
            start = time.perf_counter()
            found = predict(image)
            times.append(time.perf_counter() - start)
            tables.append(
                table_of(found, clip=clip, frame=np.int16(int(path.stem))).assign(
                    height=np.int16(image.shape[0])
                )
            )
        pd.concat(tables, ignore_index=True).to_parquet(part, index=False)
        timings[name] = times
        logger.info("gsr %s: %d frames", name, len(paths))
    part = args.out / "xg_720.parquet"
    if not part.is_file():
        shots = pd.read_parquet(
            XG_DIR / "candidates.parquet", columns=["shot_id", "offset_ms"]
        )
        frames = shots.drop_duplicates().sort_values(["shot_id", "offset_ms"])
        keys = list(zip(frames.shot_id, frames.offset_ms, strict=True))
        files = [XG_FRAMES / s / f"{o:+d}.jpg" for s, o in keys]
        tables = []
        for i, ((shot, offset), (_, image)) in enumerate(
            zip(keys, bt.read_ahead(files), strict=True)
        ):
            if image is None:
                continue
            tables.append(
                table_of(
                    predict(image), shot_id=shot, offset_ms=np.int16(offset)
                ).assign(height=np.int16(image.shape[0]))
            )
            if i % 2000 == 0:
                logger.info("xg %d / %d", i, len(keys))
        pd.concat(tables, ignore_index=True).to_parquet(part, index=False)
    (args.out / "dump.json").write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "weights_sha256": bench.sha256_file(WEIGHTS),
                "conf": DUMP_CONF,
                "imgsz": 1280,
                "iou": 0.7,
                "scales": SCALES,
                "resize": "cv2.INTER_AREA",
                "gsr_frames": len(paths),
                "clips": len(clips),
                "ms_per_frame": {
                    k: round(1000 * float(np.mean(v[5:])), 2)
                    for k, v in timings.items()
                    if len(v) > 5
                },
                "gpu": bench.gpu_name(args.device),
            },
            indent=2,
        )
        + "\n"
    )


# ------------------------------------------------------------------ read (CPU)
def gsr_scale(name: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(balls >= BALL_CONF in 1080p, frames with ``hmed``, dumped frame list) of one scale."""
    table = to_1080(pd.read_parquet(root / f"gsr_{name}.parquet"))
    frames = table[["clip", "frame"]].drop_duplicates().reset_index(drop=True)
    # frames with no candidate at all would be missing: take the list from the file listing of the clips
    people = table[table.class_id > 0]
    heights = person_heights(people, ["clip", "frame"], None)
    frames = fill_heights(listed_frames(), heights, ["clip", "frame"], "clip")
    balls = table[(table.class_id == 0) & (table.conf >= BALL_CONF)].reset_index(
        drop=True
    )
    return balls, frames, table


def listed_frames() -> pd.DataFrame:
    clips = bt.games_clips(bt.SELECTION)
    return pd.DataFrame(
        [{"clip": c, "frame": np.int16(int(p.stem))} for c, p in gsr_frames(clips)]
    )


def scored(
    balls: pd.DataFrame, frames: pd.DataFrame, rule: tuple, extra: float = 0.0
) -> pd.DataFrame:
    """``balls`` with ``hmed``, ``side`` (1080p) and ``keep`` under ``rule``."""
    out = balls.merge(
        frames[["clip", "frame", "hmed"]], on=["clip", "frame"], how="left"
    )
    out["side"] = long_side(out)
    out["keep"] = keep_mask(rule, out.side.to_numpy(), out.hmed.to_numpy())
    return out


def tracks_table(balls: pd.DataFrame) -> pd.DataFrame:
    best = top1(balls[balls.keep], ["clip", "frame"]).copy()
    best["status"] = "measured"
    return best[["clip", "frame", "status", *COLUMNS, "conf"]].reset_index(drop=True)


def match_truth(balls: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Ball candidates (1080p) on the truth frames with the centre rule's ``on_truth`` and ``large`` (false and
    long side > ``LARGE_FACTOR`` x the truth's)."""
    joined = balls.merge(truth, on=["clip", "frame"], how="inner")
    xy = be.centres(joined[COLUMNS].to_numpy(float))
    true_xy = be.centres(joined[["tx1", "ty1", "tx2", "ty2"]].to_numpy(float))
    joined["on_truth"] = (
        np.linalg.norm(xy - true_xy, axis=1) <= joined.radius.to_numpy()
    )
    truth_side = np.maximum(joined.tx2 - joined.tx1, joined.ty2 - joined.ty1)
    joined["large_false"] = (~joined.on_truth) & (
        joined.side > LARGE_FACTOR * truth_side
    )
    return joined


def ratio(draws: bs.Draws, part: pd.DataFrame, num, den) -> tuple[float, np.ndarray]:
    return bs.ratio_draws(
        draws,
        draws.per_clip(part["clip"], np.asarray(num, float)),
        draws.per_clip(part["clip"], np.asarray(den, float)),
    )


def removal_numbers(
    matched: pd.DataFrame, truth: pd.DataFrame, draws: bs.Draws
) -> dict:
    """Removal shares and top-1 trueness of one scored, matched candidate table (``keep`` already set)."""
    drop = ~matched.keep.to_numpy()
    on, large = matched.on_truth.to_numpy(), matched.large_false.to_numpy()
    out = {
        "true_removed": ratio(draws, matched, drop & on, on),
        "large_false_removed": ratio(draws, matched, drop & large, large),
        "false_removed": ratio(draws, matched, drop & ~on, ~on),
        "all_removed": ratio(draws, matched, drop, np.ones(len(matched))),
    }
    kept = matched[matched.keep]
    best = top1(kept, ["clip", "frame"])
    hit = best[best.on_truth][["clip", "frame"]].drop_duplicates()
    flag = truth.merge(
        hit.assign(hit=1.0), on=["clip", "frame"], how="left"
    ).hit.fillna(0.0)
    out["top1_on_truth"] = ratio(draws, truth, flag, np.ones(len(truth)))
    return out


def placed(
    rows_table: pd.DataFrame, homographies: dict, draws: bs.Draws
) -> tuple[pd.DataFrame, dict]:
    rows = bp.placement(rows_table, bp.CENTRE, homographies)
    return rows, bp.numbers(rows, draws)


def summary_of(found: dict) -> dict:
    return bs.summary({k: v for k, v in found.items()})


def read(args: argparse.Namespace) -> None:
    root = args.root
    clips, homographies = bc.selection_homographies()
    draws = bs.Draws(clips)
    truth = bt.truth_frames(clips)
    truth = truth[truth.bench].reset_index(drop=True)
    data = {n: gsr_scale(n, root) for n in SCALES}
    # k0 from the native placement frames
    native_frames = data["1080"][1]
    probe = placed(
        tracks_table(scored(data["1080"][0], native_frames, ("none",))),
        homographies,
        draws,
    )[0]
    keys = probe[["clip", "frame"]]
    h_native = native_frames.merge(keys, on=["clip", "frame"])["hmed"]
    k0 = LIMIT / float(h_native.median())
    rules = {"none": ("none",), "abs45": ("abs", LIMIT)} | {
        f"rel{m:g}k0": ("rel", m * k0) for m in MULTIPLES
    }
    # scene groups: clip median of frame medians (native), terciles
    clip_h = native_frames.groupby("clip")["hmed"].median()
    group = pd.qcut(clip_h, 3, labels=["small", "middle", "large"])
    result: dict = {
        "k0": k0,
        "median_person_height_native": float(h_native.median()),
        "multiples": MULTIPLES,
        "k": {name: rule[1] for name, rule in rules.items() if rule[0] == "rel"},
        "clip_height": {
            "quantiles": clip_h.quantile([0, 0.33, 0.67, 1]).tolist(),
            "groups": {g: int((group == g).sum()) for g in group.cat.categories},
        },
    }
    per = {}
    for scale, (balls, frames, _) in data.items():
        for name, rule in rules.items():
            scored_balls = scored(balls, frames, rule)
            rows, nums = placed(tracks_table(scored_balls), homographies, draws)
            matched = match_truth(scored_balls, truth)
            per[(scale, name)] = {
                "rows": rows,
                "numbers": nums,
                "removal": removal_numbers(matched, truth, draws),
                "matched": matched,
            }
    well = {
        m: per[("1080", f"rel{m:g}k0")]["numbers"]["well_placed"][0] for m in MULTIPLES
    }
    chosen = choose_multiple(well)
    result["chosen_multiple"] = chosen
    result["chosen_k"] = chosen * k0
    result["native_well_by_multiple"] = well
    star = f"rel{chosen:g}k0"
    # control against 02 / 17
    result["control"] = controls(
        data["1080"], per[("1080", "none")]["rows"], rules["none"]
    )
    table = {}
    for scale in SCALES:
        entry = {}
        for name in ("none", "abs45", star):
            p = per[(scale, name)]
            entry[name] = {
                "placement": summary_of(p["numbers"]),
                "removal": summary_of(p["removal"]),
            }
        entry["abs45_minus_none"] = bs.paired(
            per[(scale, "abs45")]["numbers"], per[(scale, "none")]["numbers"]
        )
        entry["rel_minus_abs45"] = bs.paired(
            per[(scale, star)]["numbers"], per[(scale, "abs45")]["numbers"]
        )
        entry["truth_long_side_px"] = {
            "scaled": float(
                np.median(np.maximum(truth.tx2 - truth.tx1, truth.ty2 - truth.ty1))
                * SCALES[scale]
                / 1080
            ),
            "1080": float(
                np.median(np.maximum(truth.tx2 - truth.tx1, truth.ty2 - truth.ty1))
            ),
        }
        entry["all_multiples"] = {
            name: {
                "well_placed": per[(scale, name)]["numbers"]["well_placed"][0],
                "far_of_all": per[(scale, name)]["numbers"]["far_of_all"][0],
                "true_removed": per[(scale, name)]["removal"]["true_removed"][0],
                "large_false_removed": per[(scale, name)]["removal"][
                    "large_false_removed"
                ][0],
            }
            for name in rules
        }
        table[scale] = entry
    result["scales"] = table
    # scene groups
    groups = {}
    for g in group.cat.categories:
        gclips = list(group.index[group == g])
        gdraws = bs.Draws(gclips)
        gtruth = truth[truth["clip"].isin(gclips)].reset_index(drop=True)
        groups[g] = {}
        for scale in SCALES:
            entry = {}
            for name in ("none", "abs45", star):
                p = per[(scale, name)]
                rows = p["rows"][p["rows"]["clip"].isin(gclips)].reset_index(drop=True)
                nums = bp.numbers(rows, gdraws)
                matched = p["matched"][p["matched"]["clip"].isin(gclips)]
                entry[name] = {
                    "placement": summary_of(nums),
                    "removal": summary_of(removal_numbers(matched, gtruth, gdraws)),
                    "_nums": nums,
                }
            entry["abs45_minus_none"] = bs.paired(
                entry["abs45"]["_nums"], entry["none"]["_nums"]
            )
            entry["rel_minus_abs45"] = bs.paired(
                entry[star]["_nums"], entry["abs45"]["_nums"]
            )
            for name in ("none", "abs45", star):
                del entry[name]["_nums"]
            groups[g][scale] = entry
    result["groups"] = groups
    result["rule_reading"] = reading(result, star)
    result["xg"] = xg_read(root, rules, star)
    result["ekstraklasa"] = ekstraklasa_read(rules, star)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "read.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n"
    )
    text = markdown(result, star)
    (args.out / "read.md").write_text(text)
    print(text)


def controls(native, rows: pd.DataFrame, rule) -> dict:
    """Native scale = ticket 02's dump (top-1 boxes) and ticket 17's top-1 placement."""
    balls, frames, _ = native
    mine = top1(scored(balls, frames, rule), ["clip", "frame"])
    mine = mine[["clip", "frame", *COLUMNS, "conf"]]
    sets = pd.read_parquet(
        DUMP_02 / "candidates.parquet",
        filters=[
            ("set", "==", "gsr"),
            ("row", "==", "1280"),
            ("class_name", "==", "ball"),
        ],
    )
    sets = sets[sets.conf >= BALL_CONF].copy()
    sets["clip"] = sets["image"].str.slice(0, 8)
    sets["frame"] = sets["image"].str.slice(9).astype(int)
    keep = sets[sets["clip"].isin(mine["clip"].unique())]
    ref = top1(keep, ["clip", "frame"])[["clip", "frame", *COLUMNS, "conf"]]
    ref["frame"] = ref["frame"].astype(mine.frame.dtype)
    merged = mine.merge(
        ref, on=["clip", "frame"], how="outer", suffixes=("", "_02"), indicator=True
    )
    both = merged[merged._merge == "both"]
    diff = np.abs(
        both[COLUMNS].to_numpy(float)
        - both[[c + "_02" for c in COLUMNS]].to_numpy(float)
    )
    reference17 = pd.read_parquet(TOP1_17)
    joined = rows.merge(reference17, on=["clip", "frame"], suffixes=("", "_17"))
    err = joined.error_m.to_numpy(float), joined.error_m_17.to_numpy(float)
    same = np.isclose(err[0], err[1], atol=1e-3, equal_nan=True)
    return {
        "frames_with_top1_mine": len(mine),
        "frames_with_top1_02": len(ref),
        "frames_only_one_side": int((merged._merge != "both").sum()),
        "max_box_diff_px": float(diff.max()) if diff.size else None,
        "placement_rows": len(rows),
        "placement_rows_17": len(reference17),
        "placement_error_mismatches": int((~same).sum()),
        "clips": int(rows["clip"].nunique()),
    }


def reading(result: dict, star: str) -> dict:
    out = {}
    native = result["scales"]["1080"]
    native_true = native["abs45"]["removal"]["true_removed"]["value"]
    for scale, entry in result["scales"].items():
        gain = entry["abs45_minus_none"]["well_placed"]
        true_removed = entry["abs45"]["removal"]["true_removed"]["value"]
        rel = entry["rel_minus_abs45"]["well_placed"]
        out[scale] = {
            "abs45_works": bool(
                gain["ci"][0] > 0 and 100 * (true_removed - native_true) <= 2.0
            ),
            "abs45_gain_pp": 100 * gain["difference"],
            "abs45_gain_ci_pp": [100 * x for x in gain["ci"]],
            "abs45_true_removed_vs_native_pp": 100 * (true_removed - native_true),
            "rel_minus_abs45_pp": 100 * rel["difference"],
            "rel_ci_low_above_zero": bool(rel["ci"][0] > 0),
        }
    non_native = [s for s in out if s != "1080"]
    out["relative_more_robust"] = bool(
        any(out[s]["rel_ci_low_above_zero"] for s in non_native)
        and out["1080"]["rel_minus_abs45_pp"] >= -1.0
    )
    return out


def xg_read(root: Path, rules: dict, star: str) -> dict:
    """Shares of removed candidates >= BALL_CONF on the xG shot frames (ticket 16's candidates, no truth), by tercile of
    the frame's median person height (this run's people)."""
    people = to_1080(pd.read_parquet(root / "xg_720.parquet"))
    keys = ["shot_id", "offset_ms"]
    all_frames = people[keys].drop_duplicates().reset_index(drop=True)
    heights = person_heights(people[people.class_id > 0], keys, None)
    frames = fill_heights(all_frames, heights, keys, None)
    cand = to_1080(pd.read_parquet(XG_DIR / "candidates.parquet"))
    mine = people[(people.class_id == 0) & (people.conf >= BALL_CONF)]
    balls = cand[cand.conf >= BALL_CONF].merge(
        frames[[*keys, "hmed"]], on=keys, how="left"
    )
    balls["side"] = long_side(balls)
    balls["tercile"] = pd.qcut(balls.hmed, 3, labels=["small", "middle", "large"])
    frame_q = pd.qcut(frames.hmed, 3, labels=["small", "middle", "large"], retbins=True)
    out = {
        "frames": len(frames),
        "ball_candidates_ge_0.15": len(balls),
        "frames_hmed_quantiles": frames.hmed.quantile(
            [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
        ).tolist(),
        "tercile_edges": frame_q[1].tolist(),
        "control_16_candidates_ge_0.15_vs_this_run": [len(balls), len(mine)],
        "control_top1_conf_vs_16": top1_agreement(cand, people),
    }
    for name in ("abs45", star):
        keep = keep_mask(rules[name], balls.side.to_numpy(), balls.hmed.to_numpy())
        balls[f"drop_{name}"] = ~keep
        best_all = top1(balls, keys)
        best_kept = top1(balls[keep], keys)
        changed = best_all.merge(best_kept[[*keys]].assign(kept=1), on=keys, how="left")
        changed2 = best_all.merge(
            best_kept[[*keys, "conf"]], on=keys, how="left", suffixes=("", "_k")
        )
        out[name] = {
            "dropped_share": float((~keep).mean()),
            "by_tercile": {
                t: float(balls.loc[balls.tercile == t, f"drop_{name}"].mean())
                for t in ("small", "middle", "large")
            },
            "top1_lost_or_changed_share_of_frames_with_top1": float(
                ((changed2.conf_k != changed2.conf) | changed2.conf_k.isna()).mean()
            ),
            "top1_removed_share_by_tercile": {
                t: float(
                    best_all[best_all.tercile == t]
                    .merge(best_kept[keys].assign(kept=True), on=keys, how="left")
                    .kept.isna()
                    .mean()
                )
                for t in ("small", "middle", "large")
            },
        }
        del changed
    out["disagree_share"] = float((balls.drop_abs45 != balls[f"drop_{star}"]).mean())
    return out


def top1_agreement(cand: pd.DataFrame, people: pd.DataFrame) -> dict:
    """Ticket 16's frozen candidates vs this run's: the top-1 ball of each frame (conf and box) must be the same."""
    keys = ["shot_id", "offset_ms"]
    mine = people[people.class_id == 0]
    a = top1(cand, keys).set_index(keys)
    b = top1(mine, keys).set_index(keys)
    both = a.join(b, rsuffix="_run", how="outer", lsuffix="")
    paired = both.dropna(subset=["conf", "conf_run"])
    return {
        "frames_16": len(a),
        "frames_run": len(b),
        "frames_both": len(paired),
        "max_conf_diff": float((paired.conf - paired.conf_run).abs().max()),
        "max_box_diff_px": float(
            max((paired[c] - paired[c + "_run"]).abs().max() for c in COLUMNS)
        ),
        "note": "the lower candidates differ: ticket 16 ran NMS IoU 0.5, this run 0.7 (as on GSR)",
    }


def ekstraklasa_read(rules: dict, star: str) -> dict:
    images = pd.read_parquet(DUMP_02 / "images.parquet")
    images = images[images.set == "ekstraklasa"]
    cand = pd.read_parquet(
        DUMP_02 / "candidates.parquet",
        filters=[("set", "==", "ekstraklasa"), ("row", "==", "1280")],
    ).merge(images[["image", "height"]], on="image")
    cand = to_1080(cand)
    people = cand[(cand.class_id > 0) & (cand.conf >= PEOPLE_CONF)]
    heights = person_heights(people, ["image"], None)
    frames = fill_heights(images[["image"]], heights, ["image"], None)
    balls = cand[(cand.class_id == 0) & (cand.conf >= BALL_CONF)].merge(
        frames, on="image"
    )
    balls["side"] = long_side(balls)
    out = {
        "images": len(images),
        "ball_candidates_ge_0.15": len(balls),
        "hmed_median": float(frames.hmed.median()),
    }
    for name in ("abs45", star):
        keep = keep_mask(rules[name], balls.side.to_numpy(), balls.hmed.to_numpy())
        best_all = top1(balls, ["image"])
        best_kept = top1(balls[keep], ["image"])
        out[name] = {
            "dropped": int((~keep).sum()),
            "dropped_share": float((~keep).mean()) if len(balls) else None,
            "top1_removed_frames": int(
                len(best_all) - len(best_all.merge(best_kept[["image"]], on="image"))
            ),
            "frames_with_top1": len(best_all),
        }
    return out


# ------------------------------------------------------------------ report
def pc(entry: dict, scale: float = 100.0) -> str:
    lo, hi = entry["ci"]
    return f"{scale * entry['value']:.1f} [{scale * lo:.1f}; {scale * hi:.1f}]".replace(
        ".", ","
    )


def pd_(entry: dict) -> str:
    lo, hi = entry["ci"]
    return (
        f"{100 * entry['difference']:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}]".replace(
            ".", ","
        )
    )


def markdown(r: dict, star: str) -> str:
    L = [
        "# ball-detection 23: filtr 45 px a rozdzielczość i skala (odczyt)",
        "",
        f"k0 = {r['k0']:.3f} (45 px / mediana wysokości ludzi {r['median_person_height_native']:.1f} px 1080p w klatkach placement gier 2+3). "
        f"Dobór na skali natywnej: mnożnik {r['chosen_multiple']:g} k0, k = {r['chosen_k']:.3f}.",
        "",
        "## Dobrze umieszczone (≤ 5 m, % wszystkich klatek placement, gry 2+3, 39 klipów) i usunięcia",
        "",
        "| skala | filtr | dobrze umieszczone | > 5 m wśród wszystkich | prawdziwe usunięte % | duże fałszywe usunięte % | top-1 na prawdzie % |",
        "|---|---|---|---|---|---|---|",
    ]
    for scale, e in r["scales"].items():
        for name in ("none", "abs45", star):
            p, m = e[name]["placement"], e[name]["removal"]
            L.append(
                f"| {scale}p | {name} | {pc(p['well_placed'])} | {pc(p['far_of_all']) if 'far_of_all' in p else '-'} | "
                f"{pc(m['true_removed'])} | {pc(m['large_false_removed'])} | {pc(m['top1_on_truth'])} |"
            )
    L += [
        "",
        "## Różnice sparowane (pp, bootstrap po klipach)",
        "",
        "| skala | abs45 − none | rel − abs45 |",
        "|---|---|---|",
    ]
    for scale, e in r["scales"].items():
        L.append(
            f"| {scale}p | {pd_(e['abs45_minus_none']['well_placed'])} | {pd_(e['rel_minus_abs45']['well_placed'])} |"
        )
    L += [
        "",
        "## Siatka k (punkty, dobrze umieszczone % / prawdziwe usunięte %)",
        "",
        "| filtr | " + " | ".join(f"{s}p" for s in r["scales"]) + " |",
        "|---|---|---|---|",
    ]
    names = next(iter(r["scales"].values()))["all_multiples"]
    for name in names:
        cells = [
            f"{100 * e['all_multiples'][name]['well_placed']:.1f} / {100 * e['all_multiples'][name]['true_removed']:.1f}".replace(
                ".", ","
            )
            for e in r["scales"].values()
        ]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += [
        "",
        "## Grupy scenowe (terciły klipów wg mediany wysokości ludzi, skala natywna)",
        "",
    ]
    L += [
        "| grupa | skala | none | abs45 | rel | abs45 − none | rel − abs45 | prawdziwe usunięte abs45 / rel % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g, by in r["groups"].items():
        for scale, e in by.items():
            L.append(
                f"| {g} | {scale}p | {pc(e['none']['placement']['well_placed'])} | {pc(e['abs45']['placement']['well_placed'])} | "
                f"{pc(e[star]['placement']['well_placed'])} | {pd_(e['abs45_minus_none']['well_placed'])} | {pd_(e['rel_minus_abs45']['well_placed'])} | "
                f"{pc(e['abs45']['removal']['true_removed'])} / {pc(e[star]['removal']['true_removed'])} |"
            )
    L += [
        "",
        f"klipy w grupach: {r['clip_height']['groups']}; kwantyle mediany wysokości ludzi klipów: {r['clip_height']['quantiles']}",
        "",
    ]
    L += [
        "## Reguła czytania",
        "",
        "```",
        json.dumps(r["rule_reading"], indent=1, default=float),
        "```",
        "",
    ]
    L += [
        "## Klatki strzałów xG (bez prawdy)",
        "",
        "```",
        json.dumps(r["xg"], indent=1, default=float),
        "```",
        "",
    ]
    L += [
        "## Ekstraklasa dev (opisowo)",
        "",
        "```",
        json.dumps(r["ekstraklasa"], indent=1, default=float),
        "```",
        "",
    ]
    L += [
        "## Kontrole",
        "",
        "```",
        json.dumps(r["control"], indent=1, default=float),
        "```",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("dump", help="GPU: dumps at 3 scales + xG frames")
    d.add_argument("--out", type=Path, default=OUT / "dump")
    d.add_argument("--device", default=None)
    r = commands.add_parser("read", help="CPU: the filters on the dumps")
    r.add_argument("--root", type=Path, default=OUT / "dump")
    r.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    tv.setup_logging("INFO")
    {"dump": dump, "read": read}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
