"""Automatic attributes Xa, roles and mask regions of xg-dino-info (ticket 09; `xgpoc.dino.attributes`,
`xgpoc.dino.regions`, spec-info §Attributes and council C3). CPU only, no model is loaded.

    ./run.sh --no-sync python scripts/xgdino_info_attributes.py

Population: the 7,062 non-penalty shots of the 328 training matches (`stack.inputs.training_shots`, checked against the
split plan). Inputs: the feet_t0 E3 cache (`cache/xgd_full4_feet_t0/E3`, files checked against its completion record),
the SoccerNet-v2 labels, and `cut_between_anchor` of the -400 ms offset frame from the xg-dino layer index. Everything
runs inside `attributes.no_label_reads()` (no StatsBomb, caption or GPT path can be opened).

Checks: the replay reproduces the cached shot_x / shot_y (share reported; a row that does not keeps its roles missing);
on `ALIGN_SAMPLE` seeded shots the 720p detector frame, resized to 448 x 252 as G-combo's input, matches the stored
offset frame best at zero shift (the boxes are mapped by a plain 0.35 scale).

Outputs (`OUT`, small): attributes.parquet (Xa, index shot_id), roles.parquet (shooter / goalkeeper / own-ball indices
and 720p boxes per shot and offset), region_summary.parquet (shot x region: present, reason, size in patches, overlay
share, placebo status), goal_geometry.parquet, coverage_columns.csv, coverage_regions.csv, coverage.json,
gallery_index.csv, complete.json last. On Storage (`STORAGE`): region_masks.parquet (packed 18 x 32 patch masks of
every region and placebo draw) and gallery/ (`GALLERY_N` seeded PNGs: the model input with the region tinted and the
placebo draws outlined, beside the region filled with the frame's mean colour, a preview of ticket 14's fill).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.data.datasets import FILL, overlay_mask
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.dino import attributes as at
from xgpoc.dino import regions as rg
from xgpoc.dino.records import import_closure
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.stack import inputs as si

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_info_attributes"
STORAGE = Path("/mnt/storage/tactifoot_research/xg_dino/info")
E3 = ROOT / "cache/xgd_full4_feet_t0/E3"
LAYER_INDEX = Path("/mnt/storage/tactifoot_research/xg_dino/layers/index.parquet")
OFFSET_FRAMES = ROOT / "cache/xgc/offset_frames_v1"
ALIGN_SAMPLE, ALIGN_SHIFT = 50, 3
GALLERY_N, GALLERY_SEED, GALLERY_ZOOM = 50, 9, 2
EXPECTED_SHOTS = 7062


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def offset_frame_paths(ids: list[str]) -> pd.DataFrame:
    """t0 offset frames (448 x 252 PNG) of `ids`: index shot_id, path, sha256."""
    frames = pd.read_csv(OFFSET_FRAMES / "frames.csv", usecols=["shot_id", "offset_ms", "status", "frame_path", "file_sha256"])
    frames = frames[(frames.offset_ms == 0) & frames.shot_id.isin(ids) & frames.frame_path.notna()]
    return frames.set_index("shot_id").assign(path=lambda f: f.frame_path.map(lambda p: str(OFFSET_FRAMES / p)))


def read_offset_frame(row) -> np.ndarray:
    data = Path(row.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != row.file_sha256:
        raise ValueError(f"{row.path} does not match its recorded sha256")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if (img.shape[1], img.shape[0]) != rg.INPUT_WH:
        raise ValueError(f"{row.path} is not {rg.INPUT_WH}")
    return img


def alignment_check(frames: pd.DataFrame, stored: pd.DataFrame, ids: list[str]) -> dict:
    """720p detector frame resized like G-combo's input vs the stored t0 offset frame, over shifts of up to
    `ALIGN_SHIFT` input pixels: the share of sampled shots best at zero shift, the median mean absolute difference."""
    rng = np.random.default_rng(rg.SEED)
    sample = [ids[i] for i in sorted(rng.choice(len(ids), ALIGN_SAMPLE, replace=False))]
    t0 = frames[frames.offset_ms == 0].set_index("shot_id").frame_path
    best, mad0, sizes = [], [], set()
    s = ALIGN_SHIFT
    for shot_id in sample:
        big = cv2.imread(t0[shot_id], cv2.IMREAD_GRAYSCALE)
        sizes.add((big.shape[1], big.shape[0]))
        small = cv2.resize(big, rg.INPUT_WH, interpolation=cv2.INTER_AREA).astype(float)
        ref = cv2.cvtColor(read_offset_frame(stored.loc[shot_id]), cv2.COLOR_BGR2GRAY).astype(float)
        h, w = ref.shape
        mad = {(dx, dy): float(np.abs(small[s + dy:h - s + dy, s + dx:w - s + dx] - ref[s:h - s, s:w - s]).mean())
               for dx in range(-s, s + 1) for dy in range(-s, s + 1)}
        best.append(min(mad, key=mad.get))
        mad0.append(mad[(0, 0)])
    return {"shots": len(sample), "detector_frame_sizes": sorted(sizes), "best_at_zero_shift": int(sum(b == (0, 0) for b in best)),
            "best_shifts": {f"{dx},{dy}": int(sum(b == (dx, dy) for b in best)) for dx, dy in sorted(set(best))},
            "mad_zero_shift_median": float(np.median(mad0)), "mad_zero_shift_max": float(np.max(mad0))}


# ------------------------------------------------------------------ gallery
COLOURS = {"region": (40, 40, 230), "placebo": [(230, 220, 40), (40, 220, 230), (220, 60, 220)]}


def patch_outline(img: np.ndarray, mask: np.ndarray, colour, zoom: int, thickness: int = 2) -> None:
    contours, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), None, fx=rg.PATCH * zoom, fy=rg.PATCH * zoom,
                                              interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, thickness)


def gallery_image(frame: np.ndarray, region: str, masks: pd.DataFrame, info: dict, roles: pd.Series) -> np.ndarray:
    """[model input with the region tinted, the placebos outlined, the roles' boxes | region filled with the mean
    colour], with a caption bar."""
    z = GALLERY_ZOOM
    model_in = frame.copy()
    model_in[overlay_mask(rg.INPUT_WH, "overlay")] = FILL
    region_mask = rg.unpack(masks[masks.kind == "region"]["mask"].iloc[0])
    pixels = cv2.resize(region_mask.astype(np.uint8), rg.INPUT_WH, interpolation=cv2.INTER_NEAREST).astype(bool)
    filled = model_in.copy()
    filled[pixels] = model_in.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    left = cv2.resize(model_in, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    tint = left.copy()
    tint[cv2.resize(pixels.astype(np.uint8), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST).astype(bool)] = COLOURS["region"]
    left = cv2.addWeighted(left, 0.55, tint, 0.45, 0)
    for k, row in enumerate(masks[masks.kind == "placebo"].itertuples()):
        patch_outline(left, rg.unpack(row.mask), COLOURS["placebo"][k % 3], z)
    for role, colour in (("shooter", (0, 255, 0)), ("gk", (255, 128, 0)), ("ball", (0, 255, 255))):
        box = roles[[f"{role}_{k}" for k in ("x0", "y0", "x1", "y1")]].to_numpy(float)
        if np.isfinite(box).all():
            p = np.round(box * rg.SCALE * z).astype(int)
            cv2.rectangle(left, (p[0], p[1]), (p[2], p[3]), colour, 1)
    right = cv2.resize(filled, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    body = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
    bar = np.full((44, body.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{info['item']:02d} {region}: {info['n_patches']} patches, overlay {info['overlay_share']:.2f}, "
                     f"placebo {info['placebo_status']} ({info['n_feasible']} feasible)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{info['shot_id'][-70:]}   left: model input, red region, placebos outlined; right: mean-colour fill",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([bar, body])


def gallery(summary: pd.DataFrame, masks: pd.DataFrame, roles: pd.DataFrame, stored: pd.DataFrame, out: Path) -> pd.DataFrame:
    """`GALLERY_N` seeded items cycling through the regions, each a shot (not used before) with the region present;
    for regions with placebos, every third item is drawn from shots without a valid placebo."""
    rng = np.random.default_rng(GALLERY_SEED)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    t0_roles = roles[roles.offset_ms == 0].set_index("shot_id")
    present = summary[summary.present & summary.shot_id.isin(stored.index)]
    used, items = set(), []
    for k in range(GALLERY_N):
        region = rg.REGIONS[k % len(rg.REGIONS)]
        pool = present[(present.region == region) & ~present.shot_id.isin(used)]
        if region in rg.PLACEBO_REGIONS:
            want = "no_valid_placebo" if (k // len(rg.REGIONS)) % 3 == 2 else "ok"
            pool = pool[pool.placebo_status == want] if (pool.placebo_status == want).any() else pool
        row = pool.iloc[int(rng.integers(len(pool)))]
        used.add(row.shot_id)
        info = {"item": k, "region": region} | row[["shot_id", "n_patches", "overlay_share", "placebo_status", "n_feasible"]].to_dict()
        shot_masks = masks[(masks.shot_id == row.shot_id) & (masks.region == region)]
        image = gallery_image(read_offset_frame(stored.loc[row.shot_id]), region, shot_masks, info, t0_roles.loc[row.shot_id])
        name = f"g{k:02d}_{region}.png"
        if not cv2.imwrite(str(out / name), image):
            raise OSError(f"cannot write {out / name}")
        items.append(info | {"file": name})
    index = pd.DataFrame(items)
    index.to_csv(out / "index.csv", index=False)
    return index


# ------------------------------------------------------------------ main
def code_hashes() -> dict:
    roots = [ROOT / "xgpoc/dino/attributes.py", ROOT / "xgpoc/dino/regions.py", Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in import_closure(roots)}
    return {"digest": canonical_digest(hashes), "files": hashes}


def main() -> int:
    if (OUT / "complete.json").exists():
        raise SystemExit(f"{OUT} is complete; nothing to do")
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    with at.no_label_reads():
        plan = a1.verify_plan(INDEX)
        shots = si.training_shots(INDEX, si.OOF, plan)
        if len(shots) != EXPECTED_SHOTS:
            raise ValueError(f"{len(shots)} training shots, expected {EXPECTED_SHOTS}")
        ids = sorted(shots.index)
        inputs = {name: {"path": str((E3 / name).relative_to(ROOT)), "sha256": records.verified_file(E3 / name)}
                  for name in ("frames.parquet", "features.parquet")}
        frames = pd.read_parquet(E3 / "frames.parquet", filters=[("shot_id", "in", ids)])
        features = pd.read_parquet(E3 / "features.parquet", filters=[("shot_id", "in", ids)])
        layer_index = pd.read_parquet(LAYER_INDEX)
        cuts = layer_index[(layer_index.set == "training") & (layer_index.offset_ms == -400)].set_index("shot_id").cut_between_anchor
        if not cuts.index.is_unique or set(cuts.index) != set(ids):
            raise ValueError("the layer index does not hold one -400 ms row per training shot")
        games = sorted(shots.game.unique())
        annotations = {g: si.read_annotations(g, LABELS_ROOT) for g in games}
        inputs |= {"index": {"path": str(INDEX), "sha256": sha256_file(INDEX)},
                   "oof": {"path": str(si.OOF.relative_to(ROOT)), "sha256": records.verified_file(si.OOF)},
                   "plan_sha256": a1.PLAN_SHA256,
                   "layer_index": {"path": str(LAYER_INDEX), "sha256": sha256_file(LAYER_INDEX)},
                   "labels_v2": {"root": str(LABELS_ROOT), "files": len(games),
                                 "digest": canonical_digest({g: sha256_file(si.labels_file(g, LABELS_ROOT)) for g in games})},
                   "offset_frames": {"path": str((OFFSET_FRAMES / "frames.csv").relative_to(ROOT)),
                                     "sha256": sha256_file(OFFSET_FRAMES / "frames.csv")}}
        log(f"{len(ids)} shots, {len(games)} matches; building attributes")
        table, roles, t0 = at.build(frames, features, shots, annotations, cuts)
        log("regions")
        summary, masks, goals = rg.build(t0)
        stored = offset_frame_paths(ids)
        log("alignment check")
        alignment = alignment_check(frames, stored, ids)
        log("gallery")
        gallery_index = gallery(summary, masks, roles, stored, STORAGE / "gallery")

    replay = at.replay_match(roles)
    columns, region_cov = at.coverage(table), rg.coverage(summary)
    coverage = {
        "shots": len(table), "matches": len(games), "replay_reproduces_shot_xy": replay,
        "alignment_check": alignment,
        "motion_status": {k: int(v) for k, v in table.motion_status.value_counts().items()},
        "goal_status": {k: int(v) for k, v in goals.goal_status.value_counts().items()},
        "flags": {f: int(table[f].sum()) for f in at.FLAGS},
        "blocks": at.BLOCKS, "content_blocks": list(at.CONTENT_BLOCKS),
        "columns": columns.to_dict("records"), "regions": region_cov.to_dict("records"),
        "region_constants": {"input_wh": rg.INPUT_WH, "patch": rg.PATCH, "grid_hw": rg.GRID_HW, "scale": rg.SCALE,
                             "margin": rg.MARGIN, "y_band": rg.Y_BAND, "pitch_share": rg.PITCH_SHARE,
                             "n_placebo": rg.N_PLACEBO, "seed": rg.SEED, "min_goal_px": rg.MIN_GOAL_PX},
        "attribute_constants": {"near_m": at.NEAR_M, "max_step_m": at.MAX_STEP_M, "box_m": [at.BOX_DEPTH, at.BOX_WIDTH],
                                "flag_windows": at.FLAG_WINDOWS},
    }
    if not all(v["share"] >= 0.99 for v in replay.values()):
        log(f"WARNING: replay reproduces shot_x/shot_y for less than 99% of rows: {replay}")

    table.to_parquet(OUT / "attributes.parquet")
    roles.to_parquet(OUT / "roles.parquet", index=False)
    summary.to_parquet(OUT / "region_summary.parquet", index=False)
    goals.to_parquet(OUT / "goal_geometry.parquet", index=False)
    columns.to_csv(OUT / "coverage_columns.csv", index=False)
    region_cov.to_csv(OUT / "coverage_regions.csv", index=False)
    gallery_index.to_csv(OUT / "gallery_index.csv", index=False)
    (OUT / "coverage.json").write_text(json.dumps(jsonable(coverage), indent=1))
    masks.to_parquet(STORAGE / "region_masks.parquet", index=False)

    storage_files = {"region_masks.parquet": sha256_file(STORAGE / "region_masks.parquet")}
    storage_files |= {f"gallery/{p.name}": sha256_file(p) for p in sorted((STORAGE / "gallery").iterdir())}
    complete = {"ticket": "xg-dino 09", "inputs": inputs, "code": code_hashes(), "git": records.git_state(),
                "storage": {"root": str(STORAGE), "files": storage_files},
                "files": records.output_hashes(OUT), "seconds": round(time.time() - started, 1)}
    (OUT / "complete.json").write_text(json.dumps(jsonable(complete), indent=1))
    log("done", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
