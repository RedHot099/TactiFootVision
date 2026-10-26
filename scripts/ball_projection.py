"""ball-detection 11: how much of the score the projection point costs (GSR valid, games 2+3 only, CPU).

    uv run python scripts/ball_projection.py read           # the variants, the centre check and the diagnostics

The truth ball is the bottom middle of its box (``bbox_pitch``); the pipeline places the ball at its box centre. A
projection point is a pipeline choice made in the ball-selection dump: ``with_point`` writes the point
(cx, cy + k * h / 2) in 1080p pixels of each frame's chosen box (k = 0 the centre, k = 1 the bottom) into the dump's
``point_x`` / ``point_y`` columns, and ``dump_top_ball`` hands that point to the unchanged read
(``detector_bench.placement_rows``, ``ball_scorecard.Draws`` / ``placement_draws`` / ``paired``). The tracker is not
re-run: ``track_ball`` gates on box centres (the package's ``reproject``), so every frame keeps ticket 09's ball and
a variant moves only the point.

Reference: ticket 09's tracks on GSR valid (``ball/09/track_valid``: dd2c 1280, candidates >= 0.01 at 25 fps from
``ball/04/dump25``, the 45 px post-filter, ``track_ball`` with ``ball/04/tune/chosen.json``, the camera motion of
``ball/04/motion``) and dd2c's top-1 ball >= 0.15 of the same dump. Only games 2+3 are read: the frozen homographies
are kept for their clips only, so every other clip's frames have no homography, and their rows are dropped before any
number. Variant (a) is k = 1; variant (b) is the k of ``GRID`` with the largest paired point gain over the centre in
well placed (ties to the smaller k), a read after selection. A variant wins when the 95% interval of its paired gain
has a lower bound > 0 and the share of denominator frames placed > 5 m grows by at most ``BARRIER_PP`` (point).

The diagnostics split the pitch error (prediction - truth) of the tracking's measured frames into the component
along the ray from the camera's ground point (+ = away from the camera) and across it, by the truth's distance from
the camera. The camera comes from the frozen homography (``camera``: square pixels, principal point at the image
centre, focal length from r1 orthogonal to r2), its ground point and height the clip's median.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402

OUT = be.ROOT / "ball/11"
TRACKS = be.ROOT / "ball/09/track_valid"
REFERENCE_READ = be.ROOT / "ball/09/read_valid/read.json"
GAMES = bt.SELECTION  # selection only: games 2+3
CENTRE, BOTTOM = 0.0, 1.0
GRID = (0.25, 0.5, 0.75, 1.0)  # variant (b), registered 2026-10-26 09:42
BARRIER_PP = 1.0
BALL_RADIUS_M = 0.11
FAR = bench.PLACEMENT["far_m"]
NATIVE_WIDTH = 1920


# ------------------------------------------------------------------ the point in the dump
def with_point(table: pd.DataFrame, k: float) -> pd.DataFrame:
    """The dump with each chosen box's projection point (1080p pixels): (cx, cy + k * h / 2), NaN without a ball."""
    out = table.copy()
    out["point_x"] = (out.x1 + out.x2) / 2
    out["point_y"] = (out.y1 + out.y2) / 2 + k * (out.y2 - out.y1) / 2
    none = (out.status == "none").to_numpy()
    out.loc[none, ["point_x", "point_y"]] = np.nan
    return out


def dump_top_ball(table: pd.DataFrame):
    """``detector_bench.placement_rows``' ``top_ball`` from a dump with ``point_x`` / ``point_y``: the frame's point
    in the homographies' ``PLACEMENT["input_size"]`` pixels, None without a ball."""
    scale = NATIVE_WIDTH / bench.PLACEMENT["input_size"][0]
    balls = table[table.status != "none"]
    points = {
        (c, int(f)): (x / scale, y / scale)
        for c, f, x, y in zip(
            balls["clip"], balls.frame, balls.point_x, balls.point_y, strict=True
        )
    }

    def top_ball(clip: str, frame: int, path: Path):
        point = points.get((clip, frame))
        return None if point is None else np.asarray(point)

    return top_ball


def placement(table: pd.DataFrame, k: float, homographies: dict) -> pd.DataFrame:
    """The placement rows of games ``GAMES`` with the dump's ball placed at point ``k``."""
    rows = bench.placement_rows(
        dump_top_ball(with_point(table, k)), homographies, "valid"
    )
    return rows[rows.game.isin(GAMES)].reset_index(drop=True)


def numbers(rows: pd.DataFrame, draws: bs.Draws) -> dict:
    """The scorecard's placement numbers plus ``far_of_all``: the share of denominator frames placed > 5 m."""
    clip = rows["clip"]
    far = rows.error_m.to_numpy(float) > FAR
    return bs.placement_draws(rows, draws) | {
        "far_of_all": bs.ratio_draws(
            draws, draws.per_clip(clip, far), draws.per_clip(clip, np.ones(len(rows)))
        )
    }


def verdict(gain: dict) -> dict:
    """The ticket's rule on a paired entry: lower bound > 0 and > 5 m of all frames up by at most ``BARRIER_PP``."""
    low = gain["well_placed"]["ci"][0]
    far_pp = 100 * gain["far_of_all"]["difference"]
    return {
        "ci_low_above_zero": bool(low > 0),
        "far_of_all_pp": far_pp,
        "barrier": bool(far_pp <= BARRIER_PP),
        "wins": bool(low > 0 and far_pp <= BARRIER_PP),
    }


def choose_k(variants: dict[float, dict]) -> float:
    """Variant (b): the ``GRID`` k with the largest point gain in well placed (ties to the smaller k)."""
    gains = {
        k: variants[k]["track_vs_centre"]["well_placed"]["difference"] for k in GRID
    }
    best = max(gains.values())
    return min(k for k, g in gains.items() if g == best)


def read_variants(
    tracks: pd.DataFrame, top1: pd.DataFrame, homographies: dict, clips: list[str]
) -> dict:
    """Every k of ``CENTRE`` + ``GRID`` on games ``GAMES``: tracking and top-1 at k, paired with the centre."""
    draws = bs.Draws(clips)
    keys = ["clip", "frame"]
    found: dict[float, dict] = {}
    for k in (CENTRE, *GRID):
        rows, top_rows = (
            placement(tracks, k, homographies),
            placement(top1, k, homographies),
        )
        if not rows[keys].equals(top_rows[keys]):
            raise SystemExit("the two placements have other frames")
        found[k] = {
            "rows": rows,
            "top1_rows": top_rows,
            "track": numbers(rows, draws),
            "top1": numbers(top_rows, draws),
        }
    centre = found[CENTRE]
    for k, entry in found.items():
        if not entry["rows"][keys].equals(centre["rows"][keys]):
            raise SystemExit(f"k = {k}: other placement frames than the centre's")
        entry["track_vs_centre"] = bs.paired(entry["track"], centre["track"])
        entry["top1_vs_centre"] = bs.paired(entry["top1"], centre["top1"])
        entry["track_vs_top1_centre"] = bs.paired(entry["track"], centre["top1"])
    return found


# ------------------------------------------------------------------ the centre check
def leaves(value, path: str = "") -> dict:
    if isinstance(value, dict):
        return {
            k: v
            for key, item in value.items()
            for k, v in leaves(item, f"{path}/{key}").items()
        }
    if isinstance(value, list):
        return {
            k: v
            for i, item in enumerate(value)
            for k, v in leaves(item, f"{path}/{i}").items()
        }
    return {path: value}


def centre_check(centre: dict, tracks: pd.DataFrame, reference: dict) -> dict:
    """k = 0 against ticket 09's ``selection`` group: every number of its card (track, top1, paired, status rows)."""
    mine = {
        "counts": bs.placement_counts(centre["rows"]),
        "track": bs.summary(
            {k: v for k, v in centre["track"].items() if k != "far_of_all"}
        ),
        "top1": bs.summary(
            {k: v for k, v in centre["top1"].items() if k != "far_of_all"}
        ),
        "track_vs_top1": bs.paired(
            {k: v for k, v in centre["track"].items() if k != "far_of_all"},
            {k: v for k, v in centre["top1"].items() if k != "far_of_all"},
        ),
        "status": bt.status_rows(centre["rows"], centre["top1_rows"], tracks),
    }
    mine = json.loads(json.dumps(mine, default=float))
    theirs = {k: reference[k] for k in mine}
    a, b = leaves(mine), leaves(theirs)
    differ = sorted(
        k
        for k in set(a) | set(b)
        if not (a.get(k) == b.get(k) or (_nan(a.get(k)) and _nan(b.get(k))))
    )
    return {"numbers": len(b), "identical": not differ, "differ": differ[:20]}


def _nan(value) -> bool:
    return isinstance(value, float) and np.isnan(value)


# ------------------------------------------------------------------ diagnostics
def camera(frame_to_pitch: np.ndarray, size=bench.PLACEMENT["input_size"]):
    """``(focal px, ground x, ground y, height)`` of the camera of an image -> pitch homography (square pixels,
    principal point at the image centre, focal length from r1 orthogonal to r2), None when that has no solution."""
    a = np.array(
        [[1, 0, -size[0] / 2], [0, 1, -size[1] / 2], [0, 0, 1.0]]
    ) @ np.linalg.inv(frame_to_pitch)
    a1, a2 = a[:, 0], a[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = -(a1[0] * a2[0] + a1[1] * a2[1]) / (a1[2] * a2[2])
    if not np.isfinite(f2) or f2 <= 0:
        return None
    m = np.diag([1 / np.sqrt(f2), 1 / np.sqrt(f2), 1.0]) @ a
    lam = 2 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    centre = -np.stack([r1, r2, np.cross(r1, r2)], 1).T @ t
    return float(np.sqrt(f2)), float(centre[0]), float(centre[1]), float(abs(centre[2]))


def clip_cameras(homographies: dict) -> pd.DataFrame:
    """Per clip: the median camera ground point and height over its frozen homographies, and how many solved."""
    rows = []
    for (clip, frame), matrix in homographies.items():
        found = camera(matrix)
        rows.append({"clip": clip, "frame": frame, "solved": found is not None})
        if found is not None:
            rows[-1] |= dict(
                zip(("focal_px", "cam_x", "cam_y", "cam_h"), found, strict=True)
            )
    table = pd.DataFrame(rows)
    out = table.groupby("clip")[["cam_x", "cam_y", "cam_h"]].median()
    out["frames"] = table.groupby("clip").size()
    out["solved"] = table.groupby("clip").solved.mean()
    return out.reset_index()


def error_components(rows: pd.DataFrame, cameras: pd.DataFrame) -> pd.DataFrame:
    """Placed rows with the error along (+ = away from the camera) and across the camera ray through the truth,
    the truth's ground distance from the camera and the expected centre offset of a ball on the ground."""
    table = rows[np.isfinite(rows.error_m)].merge(cameras, on="clip", how="inner")
    truth = table[["truth_x", "truth_y"]].to_numpy(float)
    ray = truth - table[["cam_x", "cam_y"]].to_numpy(float)
    distance = np.linalg.norm(ray, axis=1)
    along_unit = ray / distance[:, None]
    across_unit = np.stack([-along_unit[:, 1], along_unit[:, 0]], 1)
    error = table[["x", "y"]].to_numpy(float) - truth
    table["distance_m"] = distance
    table["along_m"] = (error * along_unit).sum(1)
    table["across_m"] = (error * across_unit).sum(1)
    table["expected_centre_offset_m"] = (
        distance * BALL_RADIUS_M / (table.cam_h - BALL_RADIUS_M)
    )
    return table


def describe(part: pd.DataFrame) -> dict:
    out = {"frames": len(part)}
    for column in ("along_m", "across_m"):
        values = part[column].to_numpy(float)
        q = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else [np.nan] * 3
        out[column] = {
            "mean": float(values.mean()) if len(values) else np.nan,
            "q25": float(q[0]),
            "median": float(q[1]),
            "q75": float(q[2]),
        }
    out["expected_centre_offset_m"] = (
        float(part.expected_centre_offset_m.median()) if len(part) else np.nan
    )
    return out


def diagnostics(
    found: dict, tracks: pd.DataFrame, cameras: pd.DataFrame, ks: list[float]
) -> dict:
    """The along / across error of the tracking's measured frames at each k of ``ks``, by distance quartile."""
    measured = tracks.loc[tracks.status == "measured", ["clip", "frame"]]
    first = error_components(
        found[ks[0]]["rows"].merge(measured, on=["clip", "frame"]), cameras
    )
    edges = np.quantile(first.distance_m, [0, 0.25, 0.5, 0.75, 1.0])
    out = {"distance_edges_m": edges.tolist(), "variants": {}}
    for k in ks:
        table = error_components(
            found[k]["rows"].merge(measured, on=["clip", "frame"]), cameras
        )
        band = np.clip(np.searchsorted(edges, table.distance_m, side="right") - 1, 0, 3)
        groups = {}
        for subset, keep in (
            ("all", np.ones(len(table), bool)),
            ("within_5m", table.error_m <= FAR),
        ):
            groups[subset] = {"all": describe(table[keep])} | {
                f"q{b + 1}": describe(table[keep & (band == b)]) for b in range(4)
            }
        out["variants"][f"{k:g}"] = groups
    return out


def switches(found: dict, tracks: pd.DataFrame, k: float) -> list[dict]:
    """Per tracking status: the frames well placed at the centre and > 5 m at ``k`` (lost) and the reverse (won)."""
    centre, moved = found[CENTRE]["rows"], found[k]["rows"]
    status = centre[["clip", "frame"]].merge(
        tracks[["clip", "frame", "status"]], how="left"
    )["status"]
    lost = (centre.error_m <= FAR) & (moved.error_m > FAR)
    won = (centre.error_m > FAR) & (moved.error_m <= FAR)
    return [
        {
            "status": level,
            "lost": int((lost & (status == level)).sum()),
            "won": int((won & (status == level)).sum()),
            "lost_centre_error_m": float(
                centre.error_m[lost & (status == level)].median()
            ),
        }
        for level in ("measured", "interpolated")
    ]


# ------------------------------------------------------------------ report
def pct(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['value']:.2f}% [{low:.1f}; {high:.1f}]"


def pp(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['difference']:+.2f} [{low:+.2f}; {high:+.2f}]"


def markdown(record: dict) -> str:
    v = record["variants"]
    lines = [
        "# ball-detection 11: projection point of the ball, games 2+3 of GSR valid",
        "",
        f"Reference: ticket 09's tracks (k = 0, box centre). Point = (cx, cy + k * h / 2). Variant (a): k = 1; "
        f"variant (b): k = {record['chosen_k']:g} (chosen on games 2+3 from {list(GRID)}, a read after selection). "
        f"Centre check vs ticket 09's card: {'identical' if record['centre_check']['identical'] else 'DIFFERS'} "
        f"({record['centre_check']['numbers']} numbers).",
        "",
        "| k | tracking well placed | gain vs centre, pp [95%] | > 5 m of all frames: pp change | barrier | wins "
        "| > 5 m of placed | median m | vs top-1 centre, pp | single frame: top-1 well | top-1 gain vs centre, pp |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, e in v.items():
        g, t, s = e["track_vs_centre"], e["track"], e["top1_vs_centre"]
        w = e["verdict"]
        lines.append(
            f"| {key} | {pct(t['well_placed'])} | **{pp(g['well_placed'])}** | "
            f"{100 * t['far_of_all']['value']:.2f}% ({pp(g['far_of_all'])}) | "
            f"{'ok' if w['barrier'] else 'NO'} | {'yes' if w['wins'] else 'no'} | "
            f"{100 * t['far_of_placed']['value']:.2f}% | {t['median_m']['value']:.3f} | "
            f"{pp(e['track_vs_top1_centre']['well_placed'])} | {pct(e['top1']['well_placed'])} | "
            f"{pp(s['well_placed'])} |"
        )
    d = record["diagnostics"]
    edges = d["distance_edges_m"]
    names = ["all"] + [
        f"q{b + 1} ({edges[b]:.0f}-{edges[b + 1]:.0f} m)" for b in range(4)
    ]
    lines += [
        "",
        "Diagnostics: tracking's measured, placed frames; error = prediction - truth, along the camera ray "
        "(+ = away from the camera) and across it, by the truth's distance from the camera's ground point.",
        "",
        "| k | frames | distance | n | along: median [q25; q75] | along: mean | across: median [q25; q75] | "
        "expected centre offset m |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, groups in d["variants"].items():
        for subset, bands in groups.items():
            for name, b in zip(names, bands.values(), strict=True):
                a, c = b["along_m"], b["across_m"]
                lines.append(
                    f"| {key} | {subset} | {name} | {b['frames']} | {a['median']:+.3f} [{a['q25']:+.2f}; "
                    f"{a['q75']:+.2f}] | {a['mean']:+.3f} | {c['median']:+.3f} [{c['q25']:+.2f}; "
                    f"{c['q75']:+.2f}] | {b['expected_centre_offset_m']:.3f} |"
                )
    lines += [
        "",
        "Frames that cross 5 m against the centre (tracking): lost = well at the centre, > 5 m at k; won = reverse.",
        "",
        "| k | status | lost | won | median centre error of the lost, m |",
        "|---|---|---|---|---|",
    ]
    for key, rows in record["switches"].items():
        for r in rows:
            lines.append(
                f"| {key} | {r['status']} | {r['lost']} | {r['won']} | {r['lost_centre_error_m']:.2f} |"
            )
    cams = record["cameras"]
    lines += [
        "",
        f"Cameras (median per clip): ground x {cams['cam_x']['median']:.1f} m, y {cams['cam_y']['median']:.1f} m, "
        f"height {cams['cam_h']['median']:.1f} m [{cams['cam_h']['min']:.1f}; {cams['cam_h']['max']:.1f}]; "
        f"{100 * cams['solved']:.1f}% of the frozen homographies solved. Bootstrap {bs.BOOTSTRAP['draws']} draws "
        f"over clips, seed {bs.BOOTSTRAP['seed']}.",
        "",
    ]
    return "\n".join(lines)


def read(args: argparse.Namespace) -> None:
    tracks = pd.read_parquet(args.tracks / "tracks.parquet")
    top1 = pd.read_parquet(args.tracks / "top1.parquet")
    clips = bt.games_clips(GAMES)
    keep = set(clips)
    homographies = {
        key: m
        for key, m in bench.frozen_homographies("valid").items()
        if key[0] in keep
    }
    found = read_variants(tracks, top1, homographies, clips)
    chosen = choose_k(found)
    reference = json.loads(args.reference.read_text())["groups"]["selection"]
    check = centre_check(found[CENTRE], tracks, reference)
    cameras = clip_cameras(homographies)
    ks = sorted({CENTRE, BOTTOM, chosen})
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "tracks": str(args.tracks),
        "reference_read": str(args.reference),
        "games": list(GAMES),
        "clips": len(clips),
        "homographies": bench.homography_file("valid"),
        "bootstrap": bs.BOOTSTRAP,
        "grid": list(GRID),
        "chosen_k": chosen,
        "barrier_pp": BARRIER_PP,
        "centre_check": check,
        "variants": {},
        "cameras": {
            c: {
                "median": float(cameras[c].median()),
                "min": float(cameras[c].min()),
                "max": float(cameras[c].max()),
            }
            for c in ("cam_x", "cam_y", "cam_h")
        }
        | {
            "solved": float(
                (cameras.solved * cameras.frames).sum() / cameras.frames.sum()
            )
        },
        "diagnostics": diagnostics(found, tracks, cameras, ks),
        "switches": {f"{k:g}": switches(found, tracks, k) for k in ks if k != CENTRE},
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cameras.to_parquet(out / "cameras.parquet", index=False)
    for k, entry in found.items():
        entry.pop("rows").to_parquet(
            out / f"placement_k{k:g}_track.parquet", index=False
        )
        entry.pop("top1_rows").to_parquet(
            out / f"placement_k{k:g}_top1.parquet", index=False
        )
        record["variants"][f"{k:g}"] = {
            "track": bs.summary(entry["track"]),
            "top1": bs.summary(entry["top1"]),
            "track_vs_centre": entry["track_vs_centre"],
            "top1_vs_centre": entry["top1_vs_centre"],
            "track_vs_top1_centre": entry["track_vs_top1_centre"],
            "verdict": verdict(entry["track_vs_centre"]),
        }
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    text = markdown(record)
    (out / "read.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("read", help="the projection-point variants on games 2+3")
    r.add_argument(
        "--tracks", type=Path, default=TRACKS, help="`ball_temporal track`'s --out"
    )
    r.add_argument(
        "--reference", type=Path, default=REFERENCE_READ, help="ticket 09's read.json"
    )
    r.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    {"read": read}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
