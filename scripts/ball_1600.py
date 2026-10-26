"""ball-detection 15: 14's arm B (the recipe for 15) trained and run at imgsz 1600.

    uv run python scripts/ball_1600.py build     # runs/bd15_1600: B's list, val list and class masks, hash-checked
    uv run python scripts/detector_train.py train --run bd15_1600                      # see runs/bd15_1600/RUN.md
    uv run python scripts/ball_temporal.py dump --variant 1600 --weights W --out ball/15/1600/dump25   # 25 fps (GPU)
    uv run python scripts/ball_scorecard.py card --weights W --rows 1600 --primary-row 1600 --out ball/15/1600/card
    uv run python scripts/ball_1600.py read      # games 2+3: pipeline, single frame, ball size, filter, cost, guards

The only change against 14's B is the resolution: ``RUNS["bd15_1600"]`` is ``RUNS["bd14_B"]`` with imgsz 1600 (and
batch 3, the largest that fits; Ultralytics accumulates gradients to its nominal batch 64 either way), the list is
B's list byte for byte (``build`` checks the hashes), the checkpoint is the last EMA after the epoch (`last.pt`).
Inference in 1600 too: the bench's row 1600 (native 1920x1080 frame, imgsz 1600, NMS IoU 0.7).

`read` reads games 2+3 only, with 14's pipeline unchanged (25 fps dump, the 45 px post-filter, `track_ball` with
ticket 04's setting, ticket 04's camera motion, the box centre on the frozen homographies):

* **primary:** 1600 - B 1280 (pipeline); the 1600 detector is better when the lower bound is > 0;
* **ticket win:** 1600 - phase 1's candidate (ticket 09's tracks): ticket 11's rule (lower bound > 0, > 5 m of all
  up by <= 1 pp) plus the guards of the 1600 card against dd2c (people >= -0.01, Ekstraklasa >= -0.02);
* single frame (top-1 >= 0.15 of the 25 fps dump, no filter): 1600 - B top-1, 1600 - dd2c top-1;
* per game; per truth ball size (``SIZE_BINS``: long side at 1080p < 8, 8-14, > 14 px): the paired gains per bin and
  the net frames gained per bin; the gain "sits in small balls" (``small_balls``) when the < 8 px bin's lower bound
  is > 0 and its gain per frame is above that of the balls >= 8 px together;
* the 45 px filter at 1600 (``filter_check``): the dumps' boxes are native frame pixels for every imgsz, so the
  filter (long side x 1080 / frame height) is the same rule; the check measures it on the dumps: for the ball
  candidates >= 0.15 on a truth ball (centre rule), the ratio of their long side to the truth's and the share the
  filter drops, and the share of all ball candidates >= 0.15 it drops;
* cost (ms per frame of the 25 fps dumps and of the cards, the 40 ms limit), false balls on referees as in 14.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_crops as bc  # noqa: E402
import ball_errors as be  # noqa: E402
import ball_projection as bp  # noqa: E402
import ball_recipe as br  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_train as dt  # noqa: E402

RUN, SOURCE = "bd15_1600", "bd14_B"
ROW = "1600"
OUT = be.ROOT / "ball/15"
B_ROOT = br.OUT / "B"  # 14's arm B: dump25, card
B_TRACKS = br.OUT / "read/track_B/tracks.parquet"  # 14's tracks of B (the control)
SIZE_BINS = ("<8", "8-14", ">14")  # truth ball long side, px at 1080p
COST_LIMIT_MS = 40.0
GAMES = bt.SELECTION

logger = logging.getLogger("ball_1600")


# ------------------------------------------------------------------ build
def changed_settings(a: dict, b: dict) -> set[str]:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def build(args: argparse.Namespace) -> None:
    """runs/bd15_1600: B's training list, validation list and class masks, unchanged (hashes checked)."""
    source = dt.run_dir(SOURCE)
    record_b = json.loads(dt.view_json(SOURCE).read_text())
    lines = (source / "train.txt").read_text().split()
    val = (source / "val.txt").read_text().split()
    classes = pd.read_parquet(source / "annotated_classes.parquet")
    if dt.sha256_lines(lines) != record_b["train"]["list_sha256"]:
        raise SystemExit("B's train.txt is not its recorded list")
    if dt.sha256_lines(val) != record_b["val"]["list_sha256"]:
        raise SystemExit("B's val.txt is not its recorded list")
    names = br.holdout_names()
    br.check_view(lines, names)
    br.check_view(val, names, ("test",))
    changed = changed_settings(dt.RUNS[RUN]["settings"], dt.RUNS[SOURCE]["settings"])
    if (
        changed != {"imgsz", "batch_size"}
        or dt.RUNS[RUN]["init"] != dt.RUNS[SOURCE]["init"]
    ):
        raise SystemExit(
            f"{RUN} differs from {SOURCE} in more than imgsz/batch: {changed}"
        )
    extra = {
        "list_of": SOURCE,
        "source_view": str(dt.view_json(SOURCE)),
        "changed_settings": sorted(changed),
        "negatives": record_b["rule"]["negatives"],
    }
    record = br.write_arm(RUN, lines, classes, val, extra)
    out = dt.run_dir(RUN)
    for name in ("train.txt", "val.txt"):
        if (out / name).read_bytes() != (source / name).read_bytes():
            raise SystemExit(f"{name} is not B's")
    logger.info(
        "%s: %d lines, sha %s (= %s's)",
        RUN,
        record["train"]["lines_per_epoch"],
        record["train"]["list_sha256"][:12],
        SOURCE,
    )


# ------------------------------------------------------------------ read: ball size, filter
def size_bin(long_side: np.ndarray) -> np.ndarray:
    """``SIZE_BINS`` of truth long sides (px at 1080p): < 8, 8 to 14 (both included), > 14."""
    side = np.asarray(long_side, float)
    return np.select([side < 8, side <= 14], SIZE_BINS[:2], SIZE_BINS[2])


def ball_sizes(rows: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    """The size bin of each placement row's truth ball (``bt.truth_frames``: the denominator's single ball)."""
    side = np.maximum(truth.tx2 - truth.tx1, truth.ty2 - truth.ty1)
    found = rows[["clip", "frame"]].merge(
        truth[["clip", "frame"]].assign(side=side.to_numpy()),
        on=["clip", "frame"],
        how="left",
    )
    if found.side.isna().any():
        raise SystemExit("a placement frame without its truth ball")
    return size_bin(found.side.to_numpy())


def size_comparisons(
    rows: dict[str, pd.DataFrame],
    pairs: dict[str, tuple[str, str]],
    sizes: np.ndarray,
    clips: list[str],
) -> dict:
    """Per size bin and pair: the paired comparison on the bin's frames (bootstrap over every clip) and the net
    frames well placed (variant - reference)."""
    draws = bs.Draws(clips)
    out = {}
    for level in SIZE_BINS:
        keep = sizes == level
        out[level] = {"frames": int(keep.sum())}
        for name, (variant, reference) in pairs.items():
            v = rows[variant][keep].reset_index(drop=True)
            r = rows[reference][keep].reset_index(drop=True)
            found = bc.compare(v, r, draws)
            found["net_frames"] = int(
                (v.error_m <= bc.FAR).sum() - (r.error_m <= bc.FAR).sum()
            )
            out[level][name] = found
    return out


def small_balls(by_size: dict, pair: str) -> dict:
    """The registered reading of "the gain sits in small balls": the < 8 px bin's lower bound > 0 and its gain per
    frame above that of the balls >= 8 px together (net frames / frames)."""
    small = by_size["<8"][pair]
    rest = [by_size[level] for level in SIZE_BINS[1:]]
    per_frame_rest = sum(r[pair]["net_frames"] for r in rest) / max(
        sum(r["frames"] for r in rest), 1
    )
    per_frame_small = small["net_frames"] / max(by_size["<8"]["frames"], 1)
    total = sum(by_size[level][pair]["net_frames"] for level in SIZE_BINS)
    return {
        "small_ci_low_above_zero": bool(small["gain"]["well_placed"]["ci"][0] > 0),
        "gain_per_frame_small": per_frame_small,
        "gain_per_frame_rest": per_frame_rest,
        "net_frames_total": total,
        "net_frames_small": small["net_frames"],
        "small_frames_share": by_size["<8"]["frames"]
        / sum(by_size[level]["frames"] for level in SIZE_BINS),
        "sits_in_small_balls": bool(
            small["gain"]["well_placed"]["ci"][0] > 0
            and per_frame_small > per_frame_rest
        ),
    }


def matched_balls(candidates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Ball candidates >= 0.15 of the ``truth`` frames with their long side at 1080p and ``on_truth`` (centre within
    the truth ball's centre-rule radius) and the truth's long side."""
    balls = candidates[
        (candidates.class_id == br.BALL) & (candidates.conf >= bs.BALL_CONF)
    ]
    joined = balls.merge(truth, on="frame", how="inner")
    xy = be.centres(joined[bt.COLUMNS].to_numpy(float))
    true_xy = be.centres(joined[["tx1", "ty1", "tx2", "ty2"]].to_numpy(float))
    return pd.DataFrame(
        {
            "long_side": np.maximum(joined.x2 - joined.x1, joined.y2 - joined.y1),
            "truth_long_side": np.maximum(
                joined.tx2 - joined.tx1, joined.ty2 - joined.ty1
            ),
            "on_truth": np.linalg.norm(xy - true_xy, axis=1)
            <= joined.radius.to_numpy(),
        }
    )


def filter_numbers(table: pd.DataFrame, limit: float) -> dict:
    """Numbers of ``matched_balls`` rows (1080p frames): what the ``limit`` px filter drops."""
    hit = table[table.on_truth]
    ratio = hit.long_side / hit.truth_long_side
    return {
        "candidates": len(table),
        "on_truth": len(hit),
        "long_side_ratio_on_truth_median": float(ratio.median()),
        "long_side_ratio_on_truth_p05_p95": ratio.quantile([0.05, 0.95]).tolist(),
        "dropped_of_all": float((table.long_side > limit).mean()),
        "dropped_on_truth": float((hit.long_side > limit).mean()),
        "dropped_off_truth": float((table[~table.on_truth].long_side > limit).mean()),
    }


def filter_check(dumps: dict[str, Path], clips: list[str]) -> dict:
    """``filter_numbers`` per dump over every frame of ``clips`` with one truth ball (GSR frames are 1920x1080, so
    dump pixels are 1080p pixels), plus the dump's box convention."""
    truth = bt.truth_frames(clips)
    out = {}
    for name, folder in dumps.items():
        record = json.loads((folder / "dump.json").read_text())
        tables = [
            matched_balls(
                bt.clip_candidates(clip, folder), truth[truth["clip"] == clip]
            )
            for clip in clips
        ]
        out[name] = {
            "row": record["row"],
            "imgsz": record["settings"]["imgsz"],
            "boxes": record["boxes"],
        } | filter_numbers(pd.concat(tables, ignore_index=True), bt.POST_FILTER)
    return out


# ------------------------------------------------------------------ read
def dump_cost(folder: Path) -> float | None:
    return json.loads((folder / "dump.json").read_text()).get("ms_per_frame")


def read(args: argparse.Namespace) -> None:
    setting = json.loads(bc.SETTING.read_text())
    clips, homographies = bc.selection_homographies()
    dumps = {"dd2c": bt.DUMP, "B": B_ROOT / "dump25", "1600": args.root / "1600/dump25"}
    tracks = {"dd2c": pd.read_parquet(bc.REFERENCE_TRACKS / "tracks.parquet")}
    top1 = {"dd2c": pd.read_parquet(bc.REFERENCE_TRACKS / "top1.parquet")}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name in ("B", "1600"):
        top, (found,) = bt.run_tracks(
            clips,
            [setting],
            args.workers,
            dumps[name],
            "valid",
            bt.MOTION_DIR,
            bt.POST_FILTER,
        )
        (out / f"track_{name}").mkdir(parents=True, exist_ok=True)
        found.to_parquet(out / f"track_{name}/tracks.parquet", index=False)
        top.to_parquet(out / f"track_{name}/top1.parquet", index=False)
        tracks[name], top1[name] = found, top
    control = pd.read_parquet(B_TRACKS).drop(columns="seconds")  # seconds: wall time
    b_same = bool(
        control.reset_index(drop=True).equals(
            tracks["B"].drop(columns="seconds").reset_index(drop=True)
        )
    )
    if not b_same:
        raise SystemExit("B's tracks differ from 14's")
    rows = {
        f"{k} pipeline": br.selection_rows(v, homographies, clips)
        for k, v in tracks.items()
    }
    rows |= {
        f"{k} top-1": br.selection_rows(v, homographies, clips) for k, v in top1.items()
    }
    pipeline = {
        "1600 - B 1280 (primary)": ("1600 pipeline", "B pipeline"),
        "1600 - phase 1 (rule 11)": ("1600 pipeline", "dd2c pipeline"),
        "B - phase 1 (14's row)": ("B pipeline", "dd2c pipeline"),
    }
    single = {
        "1600 - B top-1": ("1600 top-1", "B top-1"),
        "1600 - dd2c top-1": ("1600 top-1", "dd2c top-1"),
        "B - dd2c top-1 (14's row)": ("B top-1", "dd2c top-1"),
    }
    found = {"games 2+3": br.comparisons(rows, pipeline | single, clips)} | {
        f"game {g}": br.comparisons(rows, pipeline | single, bt.games_clips((g,)))
        for g in GAMES
    }
    sizes = ball_sizes(rows["1600 pipeline"], bt.truth_frames(clips))
    size_pairs = {
        "1600 - B 1280 (primary)": pipeline["1600 - B 1280 (primary)"],
        "1600 - phase 1 (rule 11)": pipeline["1600 - phase 1 (rule 11)"],
        "1600 - B top-1": single["1600 - B top-1"],
    }
    by_size = size_comparisons(rows, size_pairs, sizes, clips)
    small = {name: small_balls(by_size, name) for name in size_pairs}
    guards = {
        "1600 vs dd2c": br.card_guards(args.root / "1600/card", ROW),
        "1600 vs B": br.card_guards(args.root / "1600_vs_B_card", ROW),
        "B vs dd2c (14)": br.card_guards(B_ROOT / "card"),
    }
    cost = {
        "dump25_ms_per_frame": {k: dump_cost(v) for k, v in dumps.items()},
        "card_ms_per_frame": {k: g["ms_per_frame"] for k, g in guards.items()},
        "limit_ms": COST_LIMIT_MS,
    }
    cost["within_limit"] = bool(
        cost["dump25_ms_per_frame"]["1600"] is not None
        and cost["dump25_ms_per_frame"]["1600"] <= COST_LIMIT_MS
    )
    filters = filter_check({k: dumps[k] for k in ("B", "1600")}, clips)
    referees = br.referee_rows(dumps, tracks)
    primary = found["games 2+3"]["1600 - B 1280 (primary)"]
    winner = found["games 2+3"]["1600 - phase 1 (rule 11)"]
    guard_ok = (
        guards["1600 vs dd2c"]["guard_people"]
        and guards["1600 vs dd2c"]["guard_ekstraklasa"]
    )
    verdict = {
        "primary": "1600 - B 1280, well placed, games 2+3, pipeline (45 px filter + track_ball 04, box centre)",
        "primary_difference": primary["gain"]["well_placed"]["difference"],
        "primary_ci": primary["gain"]["well_placed"]["ci"],
        "primary_barrier": primary["verdict"]["barrier"],
        "1600_better_than_B": bool(primary["gain"]["well_placed"]["ci"][0] > 0),
        "vs_phase1": winner["gain"]["well_placed"],
        "vs_phase1_rule_11": winner["verdict"],
        "guards_vs_dd2c": {
            "people": guards["1600 vs dd2c"]["guard_people"],
            "ekstraklasa": guards["1600 vs dd2c"]["guard_ekstraklasa"],
        },
        "cost_within_limit": cost["within_limit"],
        "ticket_wins": bool(winner["verdict"]["wins"] and guard_ok),
        "p2_suggested": small["1600 - B 1280 (primary)"]["sits_in_small_balls"],
    }
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "games": list(GAMES),
        "clips": len(clips),
        "setting": setting,
        "reference_tracks": str(bc.REFERENCE_TRACKS),
        "dumps": {k: str(v) for k, v in dumps.items()},
        "control_B_tracks_identical_to_14": b_same,
        "bootstrap": bs.BOOTSTRAP,
        "verdict": verdict,
        "comparisons": found,
        "by_size": by_size,
        "small_balls": small,
        "filter_check": filters,
        "cost": cost,
        "guards": guards,
        "referees": referees,
    }
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    br.table_rows(found).to_csv(out / "table.csv", index=False)
    size_table(by_size).to_csv(out / "by_size.csv", index=False)
    pd.DataFrame(referees).to_csv(out / "referees.csv", index=False)
    text = markdown(record)
    (out / "read.md").write_text(text)
    print(text)


def size_table(by_size: dict) -> pd.DataFrame:
    rows = []
    for level, entry in by_size.items():
        for name, c in entry.items():
            if name == "frames":
                continue
            gain, far = c["gain"]["well_placed"], c["gain"]["far_of_all"]
            rows.append(
                {
                    "size": level,
                    "frames": entry["frames"],
                    "comparison": name,
                    "reference_well_placed": c["reference"]["well_placed"]["value"],
                    "variant_well_placed": c["variant"]["well_placed"]["value"],
                    "gain_pp": 100 * gain["difference"],
                    "gain_ci_low_pp": 100 * gain["ci"][0],
                    "gain_ci_high_pp": 100 * gain["ci"][1],
                    "far_of_all_pp": 100 * far["difference"],
                    "net_frames": c["net_frames"],
                }
            )
    return pd.DataFrame(rows)


def markdown(record: dict) -> str:
    v = record["verdict"]
    low, high = (100 * x for x in v["primary_ci"])
    w = v["vs_phase1"]
    lines = [
        "# ball-detection 15: 14's arm B at imgsz 1600 (games 2+3 of GSR valid)",
        "",
        f"Primary 1600 - B 1280 (pipeline): {100 * v['primary_difference']:+.2f} pp [{low:+.2f}; {high:+.2f}] -> "
        f"1600 better: {v['1600_better_than_B']}. Against phase 1: {100 * w['difference']:+.2f} pp "
        f"[{100 * w['ci'][0]:+.2f}; {100 * w['ci'][1]:+.2f}], rule 11: {v['vs_phase1_rule_11']}, guards vs dd2c: "
        f"{v['guards_vs_dd2c']}, cost within {record['cost']['limit_ms']:g} ms: {v['cost_within_limit']} -> "
        f"**ticket {'WINS' if v['ticket_wins'] else 'does not win'}**. P2 suggested: {v['p2_suggested']}.",
        "",
        f"Control: B's tracks recomputed = 14's: {record['control_B_tracks_identical_to_14']}.",
        "",
    ]
    for group, found in record["comparisons"].items():
        lines += [f"## {group}", "", *bc.comparison_lines(found), ""]
    lines += [
        "## By truth ball size (long side, px at 1080p)",
        "",
        "| size | frames | comparison | reference | variant | gain, pp [95%] | > 5 m of all, pp | net frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in size_table(record["by_size"]).itertuples():
        lines.append(
            f"| {r.size} | {r.frames} | {r.comparison} | {100 * r.reference_well_placed:.2f}% | "
            f"{100 * r.variant_well_placed:.2f}% | {r.gain_pp:+.2f} [{r.gain_ci_low_pp:+.2f}; "
            f"{r.gain_ci_high_pp:+.2f}] | {r.far_of_all_pp:+.2f} | {r.net_frames:+d} |"
        )
    lines += [
        "",
        "Small balls: `" + json.dumps(record["small_balls"], default=float) + "`",
        "",
    ]
    lines += [
        "## The 45 px filter (ball candidates >= 0.15, every single-ball frame of games 2+3)",
        "",
        "| dump | imgsz | candidates | on truth | long side / truth (median) [p5; p95] | dropped: all / on truth / "
        "off truth |",
        "|---|---|---|---|---|---|",
    ]
    for name, f in record["filter_check"].items():
        p5, p95 = f["long_side_ratio_on_truth_p05_p95"]
        lines.append(
            f"| {name} | {f['imgsz']} | {f['candidates']} | {f['on_truth']} | "
            f"{f['long_side_ratio_on_truth_median']:.3f} [{p5:.3f}; {p95:.3f}] | {100 * f['dropped_of_all']:.2f}% / "
            f"{100 * f['dropped_on_truth']:.3f}% / {100 * f['dropped_off_truth']:.2f}% |"
        )
    c = record["cost"]
    lines += [
        "",
        f"Cost, ms per frame: 25 fps dumps {c['dump25_ms_per_frame']}, cards {c['card_ms_per_frame']}; limit "
        f"{c['limit_ms']:g} ms.",
        "",
        "## Guards and cards (games 2+3; Ekstraklasa dev)",
        "",
        "| card | people mAP50-95 (diff) | Eks. mAP50-95 (diff) | Eks. ball AP50 | ball AP50 (diff) | false balls / "
        "frame | single frame well placed vs reference, pp | guards | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in record["guards"].items():
        w = g["well_placed_vs_reference"]
        lines.append(
            f"| {name} | {g['people_map50_95']:.4f} ({g['people_map50_95_diff']:+.4f}) | {g['eks_map50_95']:.4f} "
            f"({g['eks_map50_95_diff']:+.4f}) | {g['eks_ball_ap50']:.3f} | {g['ball_ap50']:.3f} "
            f"({g['ball_ap50_diff']:+.3f}) | {g['false_balls_per_frame']:.3f} | {100 * w['difference']:+.2f} "
            f"[{100 * w['ci'][0]:+.2f}; {100 * w['ci'][1]:+.2f}] | people {g['guard_people']}, Eks. "
            f"{g['guard_ekstraklasa']} | {g['ms_per_frame']} |"
        )
    lines += [
        "",
        "## False balls on referees (every annotated frame, 25 fps)",
        "",
        "| game | detector | balls | frames | on referee / frame | frames with one | false / frame |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in record["referees"]:
        lines.append(
            f"| {r['game']} | {r['detector']} | {r['balls']} | {r['frames']} | {r['on_referee_per_frame']:.4f} | "
            f"{100 * r['frames_with_on_referee']:.2f}% | {r['false_per_frame']:.3f} |"
        )
    lines += [
        "",
        f"Bootstrap {bs.BOOTSTRAP['draws']} draws over clips, seed {bs.BOOTSTRAP['seed']}. Placement: box centre, "
        "frozen homographies (games 2+3 only). Barrier: > 5 m of all denominator frames up by <= "
        f"{bp.BARRIER_PP:g} pp.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    b = commands.add_parser("build", help="runs/bd15_1600: B's lists")
    b.set_defaults(handler=build)
    r = commands.add_parser(
        "read", help="games 2+3: pipeline, single frame, size, filter"
    )
    r.add_argument("--root", type=Path, default=OUT)
    r.add_argument("--out", type=Path, default=OUT / "read")
    r.add_argument("--workers", type=int, default=12)
    r.set_defaults(handler=read)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.handler(args)


if __name__ == "__main__":
    main()
