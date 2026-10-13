"""SoccerNet line annotations -> a 32-vertex YOLO-pose pitch keypoint dataset (pitch-keypoints 01).

    ./run.sh --no-sync python scripts/pk_build_dataset.py [--out results/pk_dataset] [--dataset <data>/pitch_keypoints_sn]
                                                          [--workers 14] [--overlays 30] [--refit]

Sources (downloaded by `scripts/pk_fetch_soccernet.py`): SoccerNet GSR 2024 train / valid / test (every 5th frame of
each clip, frames 1, 6, 11, ...) and Camera Calibration 2023 train / valid / test. Nothing else is read: no SoccerNet
action-spotting video, StatsBomb data or goal label.

1. **Fit** (`fits.parquet`, cached; `--refit` recomputes): per image, the frame -> pitch homography fitted to the
   annotated pitch lines (`xgpoc.pitchkp.fit`), the 32 package vertices projected into the frame (visibility 2 in the
   frame and in front of the camera, else 0), the residuals, the vertex uncertainty, the rejection rule
   (`xgpoc.pitchkp.dataset.rejection`), and the checks:
   - (b) GSR: persons' `bbox_image` bottom-middle mapped through the line fit vs their `bbox_pitch` (and
     `bbox_pitch_raw`) bottom-middle, centre origin converted to the corner origin;
   - (c) the horizontal-flip check on every 10th image: the fit of the mirrored annotations (x -> 1 - x, left/right
     names swapped) must give the original labels re-indexed by `flip_idx` with x -> W - x.
2. **Axis check** (`axis_check.json`): on GSR valid, a homography fitted directly from persons' feet pixels to their
   GSR positions under each of the four axis sign conventions, against the line-fit vertices.
3. **Export** of the accepted images to `--dataset`: 960 x 540 JPEGs and YOLO-pose labels (`kpt_shape [32, 3]`, the
   `flip_idx` of `data/keypoints/data.yaml`) in `train` (GSR train + Calibration train), `val` (GSR valid + Calibration
   valid), `test_gsr` and `test_calib` (evaluation only, not in `data.yaml`). A GSR clip lives in one split.
4. **Summary** (`summary.json`), the data card (`data_card.md`), 30 random labelled overlays (`overlays/`), and
   `complete.json`, last.
"""

import argparse
import json
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import records
from xgpoc.pitchkp import dataset as D
from xgpoc.pitchkp import fit as F
from xgpoc.pitchkp import sources as S

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/kuba/projects/tactifoot_vision/data")
GSR_ROOT, CALIB_ROOT = DATA / "SoccerNetGS", DATA / "SoccerNetCalibration"
SPLITS = ("train", "valid", "test")
EXPORT_SPLIT = {("gsr", "train"): "train", ("calib", "train"): "train", ("gsr", "valid"): "val",
                ("calib", "valid"): "val", ("gsr", "test"): "test_gsr", ("calib", "test"): "test_calib"}
FLIP_EVERY = 10
VENDOR_PLAN = ROOT / "results/xgv_split_plan/plan.csv"     # xg-vendor split plan: evaluation + gate = held out
_PLAN_GAME = re.compile(r"^(?P<league>[^/]+)/(?P<season>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2}) - \d{2}-\d{2} "
                        r"(?P<home>.+?) \d+ - \d+ (?P<away>.+)$")
_CALIB_GAME = re.compile(r"^(?P<league>[^/]+)/(?P<season>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2}) (?P<home>.+?) - (?P<away>.+)$")


# --------------------------------------------------------------------------- 1. fit
def _gsr_job(args: tuple[str, str]) -> list[dict]:
    clip, split = args
    return [D.process(f, flip_check=f.frame % (S.FRAME_STEP * FLIP_EVERY) == 1) for f in S.gsr_clip(Path(clip), split)]


def _calib_job(args: tuple[str, str, list[str]]) -> list[dict]:
    split, game, stems = args
    return [D.process(f, flip_check=int(f.image.stem) % FLIP_EVERY == 0)
            for f in S.calibration_images(CALIB_ROOT, split, game, stems)]


def fit_all(workers: int) -> pd.DataFrame:
    jobs = []
    for split in SPLITS:
        jobs += [(_gsr_job, (str(c), split)) for c in S.gsr_clips(GSR_ROOT, split)]
        if (CALIB_ROOT / split / "match_info.json").exists():
            jobs += [(_calib_job, (split, game, stems)) for game, stems in S.calibration_games(CALIB_ROOT, split).items()]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(fn, arg) for fn, arg in jobs]
        for k, fut in enumerate(futures):
            rows += fut.result()
            if k % 20 == 0:
                print(f"{datetime.now(UTC):%H:%M:%S} fit {k + 1}/{len(futures)} jobs, {len(rows)} images", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. axis check
def axis_check(frames_per_clip: int = 3) -> dict:
    """Feet pixels -> GSR positions under each sign convention vs the line-fit vertices (GSR valid, a few frames/clip).

    For a frame, a homography is fitted from the persons' feet pixels to (s_x x + 52.5, s_y y + 34) without the lines;
    the line fit's visible vertex pixels are mapped through it and compared, in metres, with the package vertices.
    The right convention gives small errors; a flipped axis gives errors of tens of metres.
    """
    out = {}
    errors: dict[str, list[float]] = {f"{sx:+d}{sy:+d}": [] for sx in (1, -1) for sy in (1, -1)}
    for clip in S.gsr_clips(GSR_ROOT, "valid"):
        frames = [f for f in S.gsr_clip(clip, "valid") if len(f.feet_image) >= 8]
        for frame in frames[:: max(len(frames) // frames_per_clip, 1)][:frames_per_clip]:
            fit = F.fit_lines(F.annotation_points(frame.lines))
            if not fit.ok:
                continue
            lab = F.labels(fit.H, fit.front)
            if lab.visible.sum() < 2:
                continue
            centred = frame.feet_pitch - S.GSR_HALF
            for sx in (1, -1):
                for sy in (1, -1):
                    target = centred * (sx, sy) + S.GSR_HALF
                    Hf, _ = cv2.findHomography(frame.feet_image, target, 0)
                    if Hf is None:
                        continue
                    q = np.column_stack([lab.xy[lab.visible], np.ones(lab.visible.sum())]) @ Hf.T
                    err = np.linalg.norm(q[:, :2] / q[:, 2:] - F.VERTICES[lab.visible], axis=1)
                    errors[f"{sx:+d}{sy:+d}"].append(float(np.median(err)))
    for name, values in errors.items():
        out[name] = {"frames": len(values), "median_m": float(np.median(values)) if values else None,
                     "p90_m": float(np.quantile(values, 0.9)) if values else None}
    out["convention"] = "corner = (s_x x + 52.5, s_y y + 34); the package conversion is s_x = s_y = +1"
    return out


# --------------------------------------------------------------------------- game identity
def game_key(game: str, pattern: re.Pattern) -> tuple | None:
    m = pattern.match(game)
    return None if m is None else (m["league"], m["season"], m["date"], m["home"].strip(), m["away"].strip())


def vendor_overlap(fits: pd.DataFrame, plan_path: Path = VENDOR_PLAN) -> dict:
    """Calibration games matched to the xg-vendor split plan (league, season, date, home, away), per split and role.

    GSR clips come from 12 games of the 2019 Swiss Super League (SoccerNet-Tracking paper, arXiv 2204.06918), a league
    the plan (SoccerNet-v2 games, 2014-2017) does not hold; GSR `game_id` is an index without a match name.
    """
    plan = pd.read_csv(plan_path)
    role = {game_key(g, _PLAN_GAME): r for g, r in zip(plan["game"], plan["role"], strict=True)}
    calib = fits[fits["source"] == "calib"].drop_duplicates("group")
    out: dict = {"plan": str(plan_path), "plan_sha256": sha256_file(plan_path), "unparsed_plan_games":
                 int(sum(k is None for k in role)), "calibration": {}}
    for split, g in calib.groupby("split"):
        roles = pd.Series([role.get(game_key(x, _CALIB_GAME), "not in plan") for x in g["group"]])
        kept = fits[(fits["source"] == "calib") & (fits["split"] == split) & (fits["reject"] == "")]
        kept_roles = pd.Series([role.get(game_key(x, _CALIB_GAME), "not in plan") for x in kept["group"]])
        out["calibration"][split] = {"games": int(len(g)), "by_role": roles.value_counts().to_dict(),
                                     "kept_images_by_role": kept_roles.value_counts().to_dict()}
    gsr = fits[fits["source"] == "gsr"]
    out["gsr"] = {"games_per_split": {s: sorted(set(g["game"])) for s, g in gsr.groupby("split")},
                  "note": "2019 Swiss Super League (SoccerNet-Tracking); not in the plan"}
    return out


# --------------------------------------------------------------------------- 3. export
def export(fits: pd.DataFrame, dataset: Path, workers: int) -> pd.DataFrame:
    kept = fits[fits["reject"] == ""].copy()
    kept["export_split"] = [EXPORT_SPLIT[(s, p)] for s, p in zip(kept["source"], kept["split"], strict=True)]
    kept["name"] = [k.replace("/", "_") for k in kept["key"]]
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in sorted(set(EXPORT_SPLIT.values())):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    jobs = [(row.image, str(dataset / "images" / row.export_split / f"{row.name}.jpg"),
             str(dataset / "labels" / row.export_split / f"{row.name}.txt"), D.yolo_line(row._asdict()))
            for row in kept.itertuples(index=False)]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_write_one, jobs, chunksize=64))
    flip = list(D.FLIP_IDX)
    (dataset / "data.yaml").write_text(
        "# SoccerNet GSR 2024 + Camera Calibration 2023 pitch keypoints (pitch-keypoints 01), labels from line fits.\n"
        "# Evaluation only (not used in training or model choice): images/test_gsr, images/test_calib.\n"
        f"path: {dataset}\ntrain: images/train\nval: images/val\n\n"
        f"kpt_shape: [32, 3]\nflip_idx: {flip}\n\nnc: 1\nnames: ['pitch']\n")
    return kept


def _write_one(job: tuple[str, str, str, str]) -> None:
    src, image_out, label_out, line = job
    image = cv2.imread(src)
    if image.shape[1] != D.IMAGE_SIZE[0]:
        image = cv2.resize(image, D.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_out, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    Path(label_out).write_text(line + "\n")


# --------------------------------------------------------------------------- 4. summary
def _quantiles(values, qs=(0.5, 0.9, 0.95, 0.99)) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()), **{f"p{int(q * 100)}": float(np.quantile(values, q))
                                                                    for q in qs}, "max": float(values.max())}


def summarise(fits: pd.DataFrame, kept: pd.DataFrame, axis: dict, overlap: dict) -> dict:
    by = fits.groupby(["source", "split"])
    counts = {f"{s}/{p}": {"images": int(len(g)), "groups": int(g["group"].nunique()),
                           "kept": int((g["reject"] == "").sum()),
                           "rejected": {k: int(v) for k, v in g.loc[g["reject"] != "", "reject"].value_counts().items()}}
              for (s, p), g in by}
    export_counts = {s: {"images": int(len(g)), "by_source": {k: int(v) for k, v in g["source"].value_counts().items()}}
                     for s, g in kept.groupby("export_split")}
    fitted = fits[fits["fitted"]]
    residuals = {f"{s}/{p}": {"rms_px": _quantiles(g["rms"]), "max_px": _quantiles(g["max_residual"]),
                              "sd_max_px": _quantiles(g["sd_max"])} for (s, p), g in fitted.groupby(["source", "split"])}
    kept_residuals = {f"{s}/{p}": {"rms_px": _quantiles(g["rms"]), "sd_max_px": _quantiles(g["sd_max"])}
                      for (s, p), g in kept.groupby(["source", "split"])}
    visible_hist = {f"{s}/{p}": {int(k): int(v) for k, v in g["n_visible"].value_counts().sort_index().items()}
                    for (s, p), g in kept.groupby(["source", "split"])}
    per_vertex = np.array([v for v in kept["visible"]]).mean(axis=0).round(3).tolist()
    gsr = fits[(fits["source"] == "gsr") & fits["fitted"]]
    feet = {}
    for label, frame in (("all fitted", gsr), ("kept", gsr[gsr["reject"] == ""]),
                         ("rejected", gsr[gsr["reject"] != ""])):
        for split, g in frame.groupby("split"):
            med = g["feet_median_m"]
            feet[f"{label}/{split}"] = {"frames": int(med.notna().sum()), "frame_median_m": _quantiles(med),
                                        "share_under_0.5m": float((med < 0.5).mean()), "share_under_1m":
                                        float((med < 1.0).mean()), "share_under_2m": float((med < 2.0).mean()),
                                        "raw_frame_median_m": _quantiles(g["feet_raw_median_m"])}
    flipped = fits[fits["flip_visible_equal"].notna()]
    views = {f"{s}/{p}": g["view"].value_counts().to_dict() for (s, p), g in fitted.groupby(["source", "split"])}
    up_dy = {f"{s}/{p}": _quantiles(g["up_dy"], (0.01, 0.5, 0.99)) for (s, p), g in kept.groupby(["source", "split"])}
    return {
        "rule": {"rms_max_px": D.RMS_MAX_PX, "max_residual_px": D.MAX_RESIDUAL_PX, "sd_max_px": D.SD_MAX_PX,
                 "main_side_up_dy": D.MAIN_SIDE_UP_DY, "sd_floor_px": F.SD_FLOOR_PX, "units": "720p pixels (1280 x 720)"},
        "counts": counts, "export": export_counts, "residuals_fitted": residuals, "residuals_kept": kept_residuals,
        "visible_vertices_hist_kept": visible_hist, "vertex_visible_share_kept": per_vertex,
        "gsr_feet_check": feet, "axis_check": axis, "views_fitted": views, "up_dy_kept": up_dy,
        "flip_check": {"images": int(len(flipped)), "visibility_equal": int(flipped["flip_visible_equal"].sum()),
                       "max_px": _quantiles(flipped["flip_max_px"]),
                       "max_px_kept": _quantiles(flipped.loc[flipped["reject"] == "", "flip_max_px"])},
        "mirror_fits": "none possible: the fit only keeps camera views (det(H) w > 0 at every annotated point)",
        "image_size": list(D.IMAGE_SIZE), "vendor_overlap": overlap,
    }


def _q(d: dict, key: str = "p50", digits: int = 2) -> str:
    return "-" if not d or d.get("n", 0) == 0 else f"{d[key]:.{digits}f}"


def data_card(summary: dict, record: dict) -> str:
    """The dataset's data card (markdown) from the summary and the completion record."""
    rule = summary["rule"]
    lines = [
        "# Data card: SoccerNet pitch keypoints (pitch-keypoints 01)", "",
        f"Built {record['ts']} by `{record['script']}` at git {record['git'].get('commit', '?')[:7]}"
        f"{' (dirty)' if record['git'].get('dirty') else ''}; dataset `{record['dataset']['path']}`; "
        f"labels digest `{record['dataset']['labels_digest'][:16]}`.", "",
        "## What it is", "",
        "One instance per image (class `pitch`): the box of the pitch area in the frame and the 32 vertices of "
        "`tv.pitch.SoccerPitch()` (corner origin, x along the length, y = 0 at the far touchline; the Roboflow "
        "layout of `data/keypoints`) as (x, y, v), v = 2 in the frame and in front of the camera, else 0 with x = y "
        "= 0. Labels are projections through a homography fitted to SoccerNet's annotated pitch lines "
        "(`xgpoc/pitchkp/fit.py`); nothing is hand-labelled. `kpt_shape [32, 3]`, `flip_idx` as in "
        f"`data/keypoints/data.yaml`. Images stored at {summary['image_size'][0]} x {summary['image_size'][1]} "
        "(GSR frames downscaled from 1920 x 1080; Calibration images are 960 x 540 originals).", "",
        "## Sources and licences", "",
        "- SoccerNet Game State Reconstruction 2024 (`gamestate-2024`): 30 s clips of 12 games of the 2019 Swiss "
        "Super League (SoccerNet-Tracking), main camera; every 5th frame (1, 6, 11, ...) kept. Per-frame pitch lines "
        "and persons' `bbox_pitch`.",
        "- SoccerNet Camera Calibration 2023 (`calibration-2023`): broadcast frames of SoccerNet-v2 games (EPL, "
        "La Liga, Ligue 1, Bundesliga, Serie A, UCL; 2014-2017), live and replay views, per-image pitch lines.",
        "- Both via the `SoccerNet` package downloader (public data, no NDA password). SoccerNet data are for "
        "research use under the SoccerNet terms (non-commercial; broadcast rights stay with their owners); the "
        "derived labels inherit those terms.", "",
        "## Rejection rule (720p pixels)", "",
        f"An image is kept when its line fit exists (>= 9 annotated ground-plane points incl. a straight line, a "
        f"camera view: no mirror image), RMS residual <= {rule['rms_max_px']} px, largest residual <= "
        f"{rule['max_residual_px']} px, every visible vertex's 2-D standard deviation <= {rule['sd_max_px']} px "
        f"(residual covariance, floor {rule['sd_floor_px']} px; underdetermined fits are infinite), the pitch area is "
        "in the frame, and the view is from the main side: up the frame at its bottom centre points to the far "
        f"touchline within 66 degrees (pitch direction y component < {rule['main_side_up_dy']}). SoccerNet names "
        "lines in pitch coordinates, so reverse-angle and along-the-pitch views would carry labels rotated by 180 or "
        "90 degrees against main-camera views of look-alike pictures.", "",
        "## Counts", "",
        "| source/split | images | groups | kept | rejected (by first rule broken) |", "|---|---|---|---|---|",
    ]
    for key, c in summary["counts"].items():
        rejected = ", ".join(f"{k} {v}" for k, v in sorted(c["rejected"].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {key} | {c['images']} | {c['groups']} | {c['kept']} | {rejected} |")
    lines += ["", "Groups are GSR clips and Calibration games; a group never spans two splits.", "",
              "| export split | images | by source |", "|---|---|---|"]
    for split, c in summary["export"].items():
        lines.append(f"| {split} | {c['images']} | {', '.join(f'{k} {v}' for k, v in c['by_source'].items())} |")
    lines += ["", "`train` and `val` are in `data.yaml`; `test_gsr` and `test_calib` are for evaluation only.", "",
              "## Residuals of the kept images (720p px)", "",
              "| source/split | RMS p50 | RMS p95 | vertex sd p50 | vertex sd p95 |", "|---|---|---|---|---|"]
    for key, r in summary["residuals_kept"].items():
        lines.append(f"| {key} | {_q(r['rms_px'])} | {_q(r['rms_px'], 'p95')} | {_q(r['sd_max_px'])} | "
                     f"{_q(r['sd_max_px'], 'p95')} |")
    lines += ["", "## Checks", "",
              "GSR positions: each kept frame's persons (`bbox_image` bottom-middle through the line fit vs "
              "`bbox_pitch` bottom-middle, centre origin + (52.5, 34)); per-frame median error in metres:", "",
              "| frames | n | p50 | p90 | p99 | < 0.5 m | < 1 m | < 2 m |", "|---|---|---|---|---|---|---|---|"]
    for key, f in summary["gsr_feet_check"].items():
        q = f["frame_median_m"]
        lines.append(f"| {key} | {q.get('n', 0)} | {_q(q)} | {_q(q, 'p90')} | {_q(q, 'p99')} | "
                     f"{f['share_under_0.5m']:.3f} | {f['share_under_1m']:.3f} | {f['share_under_2m']:.3f} |")
    axis = summary["axis_check"]
    lines += ["", "Axis directions (GSR valid): homography from persons' feet pixels to (s_x x + 52.5, s_y y + 34), "
              "then the line-fit vertices mapped through it vs the package vertices, median of per-frame medians: "
              + "; ".join(f"s = {k}: {v['median_m']:.2f} m" for k, v in axis.items() if isinstance(v, dict)
                          and v.get("median_m") is not None) + ".", ""]
    flip = summary["flip_check"]
    lines += [f"Horizontal flip (`flip_idx`): {flip['images']} images refitted from mirrored annotations; "
              f"visibility equal on {flip['visibility_equal']}; kept images' largest vertex difference "
              f"{_q(flip['max_px_kept'], 'max', 6)} px.", "",
              "Mirror images: none can be kept (the fit only accepts det(H) w > 0 at every annotated point).", "",
              "## Visible vertices per kept image", "", "| source/split | " + " | ".join(str(i) for i in range(33))
              + " |", "|---|" + "---|" * 33]
    for key, h in summary["visible_vertices_hist_kept"].items():
        lines.append(f"| {key} | " + " | ".join(str(h.get(i, h.get(str(i), 0))) for i in range(33)) + " |")
    shares = summary["vertex_visible_share_kept"]
    lines += ["", "Share of kept images where each vertex is visible: "
              + ", ".join(f"{i}: {v:.2f}" for i, v in enumerate(shares)) + ".", "",
              "## Game identity and the xg-vendor split plan", ""]
    ov = summary["vendor_overlap"]
    for split, c in ov["calibration"].items():
        lines.append(f"- Calibration {split}: {c['games']} games, plan roles {c['by_role']}; kept images by role "
                     f"{c['kept_images_by_role']}.")
    lines += [f"- GSR: {ov['gsr']['note']}; game ids per split {ov['gsr']['games_per_split']}.", "",
              "The plan's held-out matches are its `evaluation` and `gate` roles. The keypoint model sees no goal "
              "label and no StatsBomb data, so their frames may train it (spec, Data); they are recorded here.", ""]
    return "\n".join(lines)


def overlays(fits: pd.DataFrame, folder: Path, n: int, seed: int = 0) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    kept = fits[fits["reject"] == ""]
    pick = kept.sample(n=min(n, len(kept)), random_state=seed)
    names = []
    for row in pick.itertuples(index=False):
        rec = row._asdict()
        frame_lines = _lines_of(rec)
        image = cv2.imread(rec["image"])
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        name = rec["key"].replace("/", "_") + ".jpg"
        cv2.imwrite(str(folder / name), D.draw_overlay(image, rec, frame_lines), [cv2.IMWRITE_JPEG_QUALITY, 88])
        names.append(name)
    return names


def _lines_of(rec: dict) -> dict:
    if rec["source"] == "calib":
        return json.loads(Path(rec["image"]).with_suffix(".json").read_text())
    clip = Path(rec["image"]).parents[1]
    for frame in S.gsr_clip(clip, rec["split"]):
        if frame.frame == rec["frame"]:
            return frame.lines
    return {}


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results/pk_dataset")
    parser.add_argument("--dataset", type=Path, default=DATA / "pitch_keypoints_sn")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--overlays", type=int, default=30)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    threadpool_limits(1)                      # one BLAS thread per process; the workers are forked from here
    start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "complete.json").unlink(missing_ok=True)
    fits_path = args.out / "fits.parquet"
    if args.refit or not fits_path.exists():
        fits = fit_all(args.workers)
        fits["box"] = fits["box"].astype(object)
        fits.to_parquet(fits_path)
    fits = pd.read_parquet(fits_path)
    fits["box"] = [None if b is None or (isinstance(b, float) and np.isnan(b)) else list(b) for b in fits["box"]]
    fits["reject"] = [D.rejection(r) for r in fits.to_dict("records")]
    print(f"{datetime.now(UTC):%H:%M:%S} {len(fits)} images fitted; axis check", flush=True)
    axis = axis_check()
    (args.out / "axis_check.json").write_text(json.dumps(axis, indent=2) + "\n")
    kept = export(fits, args.dataset, args.workers)
    kept[["key", "source", "split", "export_split", "group", "frame", "game", "name", "rms", "sd_max",
          "n_visible"]].to_csv(args.out / "exported.csv", index=False)
    summary = summarise(fits, kept, axis, vendor_overlap(fits))
    summary["overlays"] = overlays(fits, args.out / "overlays", args.overlays)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    shutil.copy(args.dataset / "data.yaml", args.out / "data.yaml")
    labels = sorted((args.dataset / "labels").rglob("*.txt"))
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"), "elapsed_s": round(time.time() - start, 1),
        "script": str(Path(__file__).resolve().relative_to(ROOT)), "git": records.git_state(),
        "code": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                 [*sorted((ROOT / "xgpoc/pitchkp").glob("*.py")), Path(__file__).resolve()]},
        "dataset": {"path": str(args.dataset), "images": int(len(kept)),
                    "labels_digest": canonical_digest({str(p.relative_to(args.dataset)): sha256_file(p)
                                                       for p in labels})},
        "sources": {"gsr": str(GSR_ROOT), "calib": str(CALIB_ROOT)},
    }
    (args.out / "data_card.md").write_text(data_card(summary, record) + "\n")
    record["files"] = records.output_hashes(args.out)
    (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "export")}, indent=1))


if __name__ == "__main__":
    main()
