"""Masked re-embedding of xg-dino-info (ticket 14; spec-info §(d), council C3.5, protocol `frozen.masking`).

    ./run.sh --no-sync python scripts/xgdino_mask_embed.py --check-only                       # inputs, regions, plan
    ./run.sh --no-sync python scripts/xgdino_mask_embed.py --gpu-approved --max-minutes 8    # resumable; rerun to go on

Every training shot's t0 frame (7,062 shots, 328 matches) goes through its own out-of-fold G-combo fold models (both
seeds of the fold that held its match out, per-member temperature) once unmasked and once per masked variant and fill
(`xgpoc/dino/masking.py`). The frame is the index JPEG (`frame_path` of the v1.2 index), the frame the fold models were
trained and scored on: the stored offset PNG differs from it (xg-temporal 01b: up to 0.84 logit at offset 0), so only
the JPEG can reproduce `z_g_combo`. Region masks are ticket 09's (`region_masks.parquet`, pinned by the protocol), in
patch space of the same 448 x 252 input.

Order of work:
    1. Regions. The feet_t0 roles of offset 0 are replayed from the E3 cache and ticket 09's regions rebuilt with
       `regions.shot_regions`; every stored region and placebo row must be reproduced bit for bit (else stop). The
       placebo domain `masking.allowed_patches` must reproduce every stored placebo draw; the shooter+ball placebos are
       drawn from it (`masking.union_placebos`).
    2. Gate. The unmasked frames of all shots are scored first; own-fold `z` must reproduce `z_g_combo` of
       `results/xgd_a1_ensemble/oof.csv` with max |Δ| <= `GATE_MAX` and mean |Δ| <= `GATE_MEAN` (orchestrator rule,
       spec-info "Notes after ticket 14 gate": the max-only bound 1e-3 of the xg-temporal 01b spot check failed on one
       shot by batch-composition noise). `gate.json` records the rule, max / mean |Δ|, r and every shot above
       `LISTED_ABOVE`; a failed gate stops
       the run before any masked frame is scored.
    3. Masked variants, chunk by chunk (whole shots of one fold per chunk).

Outputs in --out (default `OUT`, ntfs3: plain names, no symlinks):
    plan.parquet          one row per shot x variant x fill: shot_id, fold, variant, kind, region, draw, dx, dy, parts,
                          fill ("none" for unmasked), n_patches, mask (packed 18 x 32), row, chunk, phase (gate / mean / blur)
    scores.parquet        the plan without the mask, plus z (mean over the two seeds of logit / T), logit_seed<k>
                          (raw), on_target (mean raw), jpeg_sha256; `row` indexes pool.npy
    pool.npy              float16 [rows, seeds, 5 x 1024]: each seed's attention-pool vector
    unmasked.parquet      the replayed unmasked z per shot (shot_id, fold, row, z, logit_seed<k>, on_target, z_g_combo,
                          abs_delta): the unmasked reference of ticket 15 (same pipeline and batch policy as the masked
                          variants); z_g_combo only for the continuity check
    union_placebos.parquet  the shooter+ball region per shot: present, n_patches, placebo_status, n_feasible
    gate.json, complete.json (written last)
Summary in `SUMMARY` (results/xgdino_info_masked): summary.json, variant_counts.csv, complete.json last.
Resume: a chunk counts when its marker exists and its table and pool slice hash as recorded; the identity of the run
(inputs, bundles, plan) must not change between invocations.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from run_xgc_g import OVERLAY, own_sources
from run_xgc_transfer import check_gpu_free
from score_offset_frames import (
    XGT_G_COMBO,
    XGT_SHOTS,
    check_layout,
    json_text,
    runtime,
    verify_bundles,
    write_atomic,
    xgt_reference,
    xgt_roles,
)
from torch.utils.data import DataLoader, Dataset
from xgpoc.bundle import sha256_file
from xgpoc.data.datasets import FILL, overlay_mask, to_tensor
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import attributes as at
from xgpoc.dino import masking as mk
from xgpoc.dino import regions as rg
from xgpoc.models import g_bundle
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
VERSION = "xgdino_info_masked_v1"
OUT = Path("/mnt/storage/tactifoot_research/xg_dino/info/masked")
SUMMARY = ROOT / "results/xgdino_info_masked"
PROTOCOL = ROOT / "research/xgdino_info_protocol.json"
PROTOCOL_SHA256 = "39a547b5b7b7642e5327a1a0de362faa46b3043dc79189736757b2a37bba3298"
REGION_MASKS = Path("/mnt/storage/tactifoot_research/xg_dino/info/region_masks.parquet")
E3 = ROOT / "cache/xgd_full4_feet_t0/E3"
GATE_MAX, GATE_MEAN = 2e-3, 1e-5              # orchestrator rule after the first gate (module docstring)
LISTED_ABOVE = 1e-6
EXPECTED_SHOTS = 7062
MIN_FREE_BYTES = 5 * 2**30


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --- inputs --------------------------------------------------------------------------------------------------------------

def check_protocol(path: Path = PROTOCOL) -> dict:
    """The pinned protocol, unchanged, and its record of the region masks."""
    if sha256_file(path) != PROTOCOL_SHA256:
        raise ValueError(f"{path} is not the pinned protocol {PROTOCOL_SHA256[:8]}")
    protocol = json.loads(path.read_text())
    entry = protocol["inputs"]["region_masks"]
    if sha256_file(REGION_MASKS) != entry["sha256"]:
        raise ValueError(f"{REGION_MASKS} is not the region masks the protocol pinned")
    return protocol


def stored_masks(path: Path = REGION_MASKS) -> dict[str, dict[tuple[str, str, int], tuple[np.ndarray, int, int]]]:
    """shot_id -> {(region, kind, draw): (mask, dx, dy)} of ticket 09."""
    table = pd.read_parquet(path)
    out: dict[str, dict] = {}
    for row in table.itertuples(index=False):
        out.setdefault(row.shot_id, {})[(row.region, row.kind, int(row.draw))] = (rg.unpack(row.mask), int(row.dx), int(row.dy))
    return out


def t0_roles(ids: list[str]) -> dict[str, at.Roles]:
    """The feet_t0 roles of offset 0 replayed from the E3 cache (files checked against its completion record)."""
    for name in ("frames.parquet", "features.parquet"):
        records.verified_file(E3 / name)
    frames = pd.read_parquet(E3 / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(E3 / "features.parquet", filters=[("shot_id", "in", ids)])
    cached = features.set_index(["shot_id", "offset_ms"]).sort_index()
    roles = {shot_id: by_offset[0] for shot_id, by_offset in at.replay_roles(frames, cached)}
    if set(roles) != set(ids):
        raise ValueError(f"{len(set(ids) - set(roles))} shots have no replayed t0 row")
    return roles


def rebuild_regions(roles: dict[str, at.Roles], stored: dict) -> tuple[dict[str, np.ndarray], dict]:
    """(allowed placebo patches per shot, check record). Raises unless ticket 09's regions and placebos are reproduced
    bit for bit and `allowed_patches` reproduces every stored placebo draw."""
    overlay = rg.overlay_patches()
    allowed, region_rows, placebo_shots = {}, 0, 0
    for shot_id in sorted(roles):
        _, rows, _ = rg.shot_regions(shot_id, roles[shot_id], overlay)
        mine = {(r["region"], r["kind"], r["draw"]): (r["mask"], r["dx"], r["dy"]) for r in rows}
        theirs = {k: (rg.pack(m), dx, dy) for k, (m, dx, dy) in stored.get(shot_id, {}).items()}
        if mine != theirs:
            raise ValueError(f"{shot_id}: the replayed regions differ from region_masks.parquet")
        region_rows += len(rows)
        allowed[shot_id] = mk.allowed_patches(roles[shot_id], overlay)
        for region in rg.PLACEBO_REGIONS:
            if (region, "region", 0) not in stored.get(shot_id, {}):
                continue
            shifts, _ = rg.placebos(stored[shot_id][(region, "region", 0)][0], allowed[shot_id], rg.rng_for(shot_id, region))
            recorded = [stored[shot_id][(region, "placebo", k)][1:] for k in range(rg.N_PLACEBO)
                        if (region, "placebo", k) in stored[shot_id]]
            if shifts != recorded:
                raise ValueError(f"{shot_id}/{region}: the rebuilt placebo domain does not reproduce the stored draws")
            placebo_shots += bool(recorded)
    return allowed, {"region_mask_rows_reproduced": region_rows, "placebo_sets_reproduced": placebo_shots,
                     "rule": "regions.shot_regions on the replayed t0 roles == region_masks.parquet (every row, packed "
                             "mask and shift); masking.allowed_patches reproduces every stored placebo draw"}


def build_plan(shots: pd.DataFrame, stored: dict, allowed: dict, gate_shots: int,
               chunk_shots: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(plan, shooter+ball record). `shots`: shot_id, fold, jpeg. Rows: the gate phase (unmasked, chunks of
    `gate_shots`) first, then the mean-fill phase (the primary fill), then the blur-fill phase (chunks of `chunk_shots`
    whole shots of one fold); within a phase by fold
    then shot."""
    rows, unions = [], []
    for shot in shots.sort_values(["fold", "shot_id"]).itertuples(index=False):
        variants, union = mk.shot_variants(shot.shot_id, stored.get(shot.shot_id, {}), allowed[shot.shot_id])
        unions.append(union)
        for v in variants:
            base = {"shot_id": shot.shot_id, "fold": int(shot.fold), "variant": v.name, "kind": v.kind, "region": v.region,
                    "draw": v.draw, "dx": v.dx, "dy": v.dy, "parts": v.parts,
                    "n_patches": int(v.mask.sum()) if v.mask is not None else 0,
                    "mask": rg.pack(v.mask) if v.mask is not None else b"", "jpeg": shot.jpeg}
            for fill in (("none",) if v.mask is None else mk.FILLS):
                rows.append(base | {"fill": fill, "phase": "gate" if v.mask is None else fill})
    plan = pd.DataFrame(rows)
    plan["phase_rank"] = plan.phase.map({"gate": 0, "mean": 1, "blur": 2})
    plan = plan.sort_values(["phase_rank", "fold", "shot_id"], kind="stable").reset_index(drop=True)
    plan["row"] = np.arange(len(plan))
    chunk, numbers = 0, []
    for _, group in plan.groupby(["phase_rank", "fold"], sort=True):
        size = gate_shots if group.phase.iloc[0] == "gate" else chunk_shots
        order = pd.factorize(group.shot_id)[0] // size
        numbers.append(pd.Series(order + chunk, index=group.index))
        chunk += int(order.max()) + 1
    plan["chunk"] = pd.concat(numbers).sort_index()
    return plan.drop(columns="phase_rank"), pd.DataFrame(unions)


def plan_digest(plan: pd.DataFrame) -> str:
    keys = plan.drop(columns=["mask"]).astype(str)
    data = pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes() + b"".join(plan["mask"])
    return hashlib.sha256(data).hexdigest()


# --- frames --------------------------------------------------------------------------------------------------------------

class ShotFrames(Dataset):
    """One item per shot of a chunk: the stacked model inputs of its plan rows (in plan order) and the JPEG's sha256.
    The model input is built as `FrameDataset` / `datasets.load_image` (resize INTER_AREA, overlay grey)."""

    def __init__(self, rows: pd.DataFrame, size_wh: tuple[int, int], mask: str):
        self.groups = [g for _, g in rows.groupby("shot_id", sort=False)]
        self.size = tuple(size_wh)
        self.blank = overlay_mask(self.size, mask)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int):
        group = self.groups[i]
        data = Path(group.jpeg.iloc[0]).read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot decode {group.jpeg.iloc[0]}")
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        img[self.blank] = FILL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sources = mk.fill_sources(img) if group.fill.ne("none").any() else {}
        tensors = [to_tensor(mk.apply(img, rg.unpack(m) if fill != "none" else None, sources.get(fill), self.blank))
                   for m, fill in zip(group["mask"], group.fill, strict=True)]
        return torch.stack(tensors), hashlib.sha256(data).hexdigest(), group.row.to_numpy()


def score_chunk(rows: pd.DataFrame, scorer: mk.FoldScorer, batch_size: int, workers: int, size_wh, mask) -> tuple[pd.DataFrame, np.ndarray]:
    """(table rows with scores, pool [rows, seeds, 5 D] float16) of one chunk, in plan order."""
    fold = int(rows.fold.iloc[0])
    if rows.fold.nunique() != 1:
        raise ValueError("a chunk holds shots of several folds")
    loader = DataLoader(ShotFrames(rows, size_wh, mask), batch_size=None, shuffle=False, num_workers=workers)
    seeds = scorer.seeds[fold]
    parts = {"row": [], "z": [], "on_target": [], "pool": [], **{f"logit_seed{s}": [] for s in seeds}}
    shas: dict[int, str] = {}
    buffer: list[tuple[torch.Tensor, np.ndarray]] = []

    def flush(items):
        images = torch.cat([t for t, _ in items])
        out = scorer.score(images, fold)
        parts["row"].append(np.concatenate([r for _, r in items]))
        parts["z"].append(out["z"])
        parts["on_target"].append(out["on_target"])
        for k, s in enumerate(seeds):
            parts[f"logit_seed{s}"].append(out["logit"][:, k])
        if not np.isfinite(out["pool"]).all() or np.abs(out["pool"]).max() > np.finfo(np.float16).max:
            raise ValueError("non-finite or float16-overflowing pool vector")
        parts["pool"].append(out["pool"].astype(np.float16))

    pending = 0
    for tensors, sha, row_ids in loader:
        row_ids = np.asarray(row_ids)
        shas.update(dict.fromkeys(row_ids.tolist(), sha))
        start = 0
        while start < len(tensors):
            take = min(batch_size - pending, len(tensors) - start)
            buffer.append((tensors[start:start + take], row_ids[start:start + take]))
            pending += take
            start += take
            if pending == batch_size:
                flush(buffer)
                buffer, pending = [], 0
    if buffer:
        flush(buffer)
    order = np.concatenate(parts["row"])
    if not np.array_equal(order, rows.row.to_numpy()):
        raise ValueError("scored rows are not the chunk's plan rows in order")
    table = rows.drop(columns=["mask", "jpeg"]).reset_index(drop=True)
    table["z"] = np.concatenate(parts["z"])
    table["on_target"] = np.concatenate(parts["on_target"])
    for s in seeds:
        table[f"logit_seed{s}"] = np.concatenate(parts[f"logit_seed{s}"])
    table["jpeg_sha256"] = [shas[r] for r in order]
    if not np.isfinite(table[["z", "on_target"]].to_numpy()).all():
        raise ValueError("non-finite score")
    return table, np.concatenate(parts["pool"])


# --- chunks --------------------------------------------------------------------------------------------------------------

def chunk_files(out: Path, number: int) -> tuple[Path, Path]:
    return out / "chunks" / f"chunk_{number:05d}.parquet", out / "chunks" / f"chunk_{number:05d}.json"


def slice_hash(array: np.ndarray, start: int, end: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(array[start:end]).tobytes()).hexdigest()


def span(rows: pd.DataFrame) -> tuple[int, int]:
    start, end = int(rows.row.iloc[0]), int(rows.row.iloc[-1]) + 1
    if end - start != len(rows):
        raise ValueError("plan rows of a chunk are not contiguous")
    return start, end


def committed(out: Path, number: int, rows: pd.DataFrame, pool: np.ndarray) -> bool:
    table_file, marker_file = chunk_files(out, number)
    try:
        marker = json.loads(marker_file.read_text())
        start, end = span(rows)
        return marker == {"table_sha256": sha256_file(table_file), "start": start, "end": end,
                          "pool_sha256": slice_hash(pool, start, end)}
    except (OSError, ValueError, KeyError):
        return False


def commit(out: Path, number: int, rows: pd.DataFrame, table: pd.DataFrame, values: np.ndarray, pool: np.ndarray) -> None:
    start, end = span(rows)
    pool[start:end] = values
    pool.flush()
    table_file, marker_file = chunk_files(out, number)
    table_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = table_file.with_name("." + table_file.name + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.rename(table_file)
    write_atomic(marker_file, json_text({"table_sha256": sha256_file(table_file), "start": start, "end": end,
                                         "pool_sha256": slice_hash(pool, start, end)}).encode())


def open_pool(out: Path, rows: int, seeds: int, width: int) -> np.ndarray:
    path, shape = out / "pool.npy", (rows, seeds, width)
    if not path.exists():
        temporary = path.with_name("pool.npy.tmp")
        np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape).flush()
        temporary.rename(path)
    pool = np.lib.format.open_memmap(path, mode="r+")
    if pool.shape != shape or pool.dtype != np.float16:
        raise ValueError(f"{path} has shape {pool.shape} {pool.dtype}, expected {shape} float16")
    return pool


# --- gate ----------------------------------------------------------------------------------------------------------------

def gate_record(table: pd.DataFrame, reference: pd.Series, max_tolerance: float = GATE_MAX,
                mean_tolerance: float = GATE_MEAN) -> dict:
    """Unmasked own-fold z against `z_g_combo`: passes when every shot has a reference, max |Δ| <= `max_tolerance`
    and mean |Δ| <= `mean_tolerance` (orchestrator rule, spec-info "Notes after ticket 14 gate"); every shot above
    `LISTED_ABOVE` is listed."""
    unmasked = table[table.kind.eq("unmasked")]
    ref = reference.reindex(unmasked.shot_id).to_numpy(np.float64)
    ours = unmasked.z.to_numpy(np.float64)
    diff = np.abs(ours - ref)
    usable = np.isfinite(diff)
    listed = unmasked.assign(z_g_combo=ref, abs_delta=diff)[diff > LISTED_ABOVE].sort_values("abs_delta", ascending=False)
    record = {"shots": int(len(unmasked)), "missing_reference": int((~np.isfinite(ref)).sum()),
              "max_abs_delta": float(diff[usable].max()) if usable.any() else None,
              "mean_abs_delta": float(diff[usable].mean()) if usable.any() else None,
              "pearson_r": float(np.corrcoef(ours[usable], ref[usable])[0, 1]) if usable.sum() > 1 else None,
              "rule": {"max_abs_delta_at_most": max_tolerance, "mean_abs_delta_at_most": mean_tolerance,
                       "source": "orchestrator decision 2026-10-28, spec-info \"Notes after ticket 14 gate\": "
                                 "batch-composition noise of the frozen G (replays need identical batch order); the "
                                 "earlier max-only bound 1e-3 (xgpoc.probe.table.SPOT_TOLERANCE) failed on one shot"},
              "counts_above": {str(t): int((diff[usable] > t).sum()) for t in (1e-6, 1e-4, 1e-3, max_tolerance)},
              "listed_above": LISTED_ABOVE,
              "listed": [{"shot_id": r.shot_id, "fold": int(r.fold), "row": int(r.row), "z": float(r.z),
                          "z_g_combo": float(r.z_g_combo), "abs_delta": float(r.abs_delta)} for r in listed.itertuples()],
              "definition": "own-fold z (mean over both seeds of logit / T) of the unmasked index JPEG vs z_g_combo of "
                            "results/xgd_a1_ensemble/oof.csv, fp32, batch g_bundle.EVAL_BATCH"}
    record["passed"] = bool(len(unmasked) and record["missing_reference"] == 0 and record["max_abs_delta"] <= max_tolerance
                            and record["mean_abs_delta"] <= mean_tolerance)
    return record


# --- run -----------------------------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--gpu-approved", action="store_true", help="required on the GPU (lock taken with agent-state)")
    ap.add_argument("--batch-size", type=int, default=g_bundle.EVAL_BATCH, help="the training-time eval batch: bit-level replay")
    ap.add_argument("--gate-shots", type=int, default=256, help="shots per gate chunk")
    ap.add_argument("--chunk-shots", type=int, default=24, help="shots per masked chunk, the unit of resume")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-shots", type=int, help="smoke: the first N shots of each fold (not the default --out)")
    ap.add_argument("--max-minutes", type=float, help="stop cleanly after the chunk that passes this budget")
    ap.add_argument("--gate-only", action="store_true", help="stop after the gate (unmasked replay)")
    ap.add_argument("--check-only", action="store_true", help="verify inputs and regions, build the plan, print sizes")
    args = ap.parse_args(argv)
    if args.limit_shots and (args.out.resolve() == OUT.resolve() or args.summary.resolve() == SUMMARY.resolve()):
        ap.error("a smoke subset (--limit-shots) must not be written to the default --out or --summary")
    return args


def prepare(args) -> dict:
    """Everything before the GPU: protocol, shots, folds, bundles, reference, regions, plan."""
    protocol = check_protocol()
    records.verified_file(XGT_SHOTS)
    training = pd.read_csv(XGT_SHOTS)
    folds = pd.read_csv(a1.FOLDS)
    every = verify_bundles(XGT_G_COMBO, "g_combo", INDEX, a1.FOLDS, ROOT / OVERLAY)
    spec, backbone, n_seeds = check_layout({"g_combo": every}, folds, 10)
    xgt_roles(INDEX, a1.FOLDS, training, every)
    reference, reference_record = xgt_reference(ROOT / "results/xgd_a1_ensemble/oof.csv")
    if len(training) != EXPECTED_SHOTS:
        raise ValueError(f"{len(training)} training shots, expected {EXPECTED_SHOTS}")
    jpeg = pd.read_parquet(INDEX, columns=["shot_id", "frame_path"]).set_index("shot_id").frame_path
    shots = training.assign(fold=training.game.map(folds.set_index("game").fold).astype(int),
                            jpeg=training.shot_id.map(jpeg))
    if shots.jpeg.isna().any():
        raise ValueError("a training shot has no index frame")
    if args.limit_shots:
        shots = shots.sort_values("shot_id").groupby("fold").head(args.limit_shots)
    ids = sorted(shots.shot_id)
    plan, unions, region_check = cached_plan(args, shots, ids)
    return {"protocol": protocol, "models": every, "spec": spec, "backbone": backbone, "n_seeds": n_seeds,
            "reference": reference, "reference_record": reference_record, "plan": plan, "unions": unions,
            "region_check": region_check}


PLAN_CACHE = "prepared"          # in --out: the plan of the first invocation, reused while its inputs are unchanged


def cached_plan(args, shots: pd.DataFrame, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(plan, shooter+ball record, region check): built from the replayed roles (`rebuild_regions`, `build_plan`) on
    the first invocation and stored in --out; later invocations reuse it while the region masks, the E3 cache, the
    shot list, the masking code and the plan options hash as recorded (a rerun of the replay takes ~2 min)."""
    key = {"region_masks": sha256_file(REGION_MASKS), "e3": {n: sha256_file(E3 / n) for n in ("frames.parquet", "features.parquet")},
           "shots": hashlib.sha256("\n".join(f"{s}|{f}|{j}" for s, f, j in shots.sort_values("shot_id")[["shot_id", "fold", "jpeg"]]
                                               .itertuples(index=False)).encode()).hexdigest(),
           "code": {f: sha256_file(ROOT / f) for f in ("xgpoc/dino/masking.py", "xgpoc/dino/regions.py", "xgpoc/dino/attributes.py")},
           "gate_shots": args.gate_shots, "chunk_shots": args.chunk_shots}
    folder = args.out / PLAN_CACHE
    if not args.check_only and (folder / "key.json").exists() and json.loads((folder / "key.json").read_text()) == key:
        record = json.loads((folder / "record.json").read_text())
        plan, unions = pd.read_parquet(folder / "plan.parquet"), pd.read_parquet(folder / "unions.parquet")
        if {n: sha256_file(folder / n) for n in ("plan.parquet", "unions.parquet")} != record["files"]:
            raise ValueError(f"{folder} does not match its record")
        log(f"plan reused from {folder}")
        return plan, unions, record["region_check"]
    log(f"{len(ids)} shots; replaying t0 roles and rebuilding ticket 09's regions")
    stored = stored_masks()
    roles = t0_roles(ids)
    allowed, region_check = rebuild_regions(roles, {s: stored.get(s, {}) for s in ids})
    plan, unions = build_plan(shots, stored, allowed, args.gate_shots, args.chunk_shots)
    if not args.check_only:
        folder.mkdir(parents=True, exist_ok=True)
        plan.to_parquet(folder / "plan.parquet", index=False)
        unions.to_parquet(folder / "unions.parquet", index=False)
        record = {"files": {n: sha256_file(folder / n) for n in ("plan.parquet", "unions.parquet")}, "region_check": region_check}
        write_atomic(folder / "record.json", json_text(record).encode())
        write_atomic(folder / "key.json", json_text(key).encode())
    return plan, unions, region_check


def identity_of(args, p: dict) -> dict:
    plan = p["plan"]
    return json.loads(json.dumps({
        "version": VERSION, "limit_shots": args.limit_shots,
        "protocol": {"path": "research/xgdino_info_protocol.json", "sha256": PROTOCOL_SHA256},
        "region_masks": {"path": str(REGION_MASKS), "sha256": p["protocol"]["inputs"]["region_masks"]["sha256"]},
        "e3": {n: sha256_file(E3 / n) for n in ("frames.parquet", "features.parquet")},
        "shots": {"path": "results/xgt_inputs/extraction_shots.csv", "sha256": sha256_file(XGT_SHOTS)},
        "index_sha256": sha256_file(INDEX), "folds_sha256": sha256_file(a1.FOLDS), "plan_sha256": a1.PLAN_SHA256,
        "reference": p["reference_record"],
        "bundles": [{k: m[k] for k in ("fold", "seed", "sha256", "temperature", "commit", "code_digest")} for m in p["models"]],
        "spec": p["spec"], "backbone": p["backbone"],
        "fills": {"mean": "mean colour of the model input after the overlay mask, rounded to uint8",
                  "blur": f"Gaussian blur of the same input, sigma {mk.BLUR_SIGMA} px (reflect border)",
                  "overlay": "pixels the overlay mask blanks keep the overlay grey"},
        "plan": {"rows": len(plan), "shots": int(plan.shot_id.nunique()), "digest": plan_digest(plan),
                 "gate_shots": args.gate_shots, "chunk_shots": args.chunk_shots, "chunks": int(plan.chunk.nunique())},
        "scoring": {"batch_size": args.batch_size, "precision": "float32, eval mode",
                    "z": "mean over the fold's two seeds of logit / T (per-member temperature)",
                    "pool": "float16 per seed, 5 x 1024 (attention pool entering the head MLP)"}}))


def counts(plan: pd.DataFrame) -> dict:
    by = plan.groupby(["kind", "region", "fill"]).agg(rows=("row", "size"), shots=("shot_id", "nunique")).reset_index()
    return {"rows": len(plan), "shots": int(plan.shot_id.nunique()), "by_kind_fill": {
        f"{k}|{f}": int(v) for (k, f), v in plan.groupby(["kind", "fill"]).size().items()},
        "table": by.to_dict(orient="records")}


def run(args: argparse.Namespace) -> int:
    started = time.time()
    with no_statsbomb_reads():
        return guarded_run(args, started)


def guarded_run(args: argparse.Namespace, started: float) -> int:
    out = args.out
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} is complete; nothing to do")
    p = prepare(args)
    plan = p["plan"]
    identity = identity_of(args, p)
    width = 5 * 1024
    pool_bytes = len(plan) * p["n_seeds"] * width * 2
    log(f"{plan.shot_id.nunique()} shots, {len(plan)} rows ({int(plan.phase.eq('gate').sum())} gate), "
        f"{plan.chunk.nunique()} chunks, pool {pool_bytes / 2**30:.2f} GiB; {json.dumps(p['region_check'])}")
    if args.check_only:
        c = counts(plan)
        print(json_text({"by_kind_fill": c["by_kind_fill"], "union": p["unions"].placebo_status.value_counts().to_dict()}))
        return 0
    device = args.device
    if device == "cuda":
        if not args.gpu_approved:
            raise SystemExit("refusing to use the GPU without --gpu-approved")
        check_gpu_free()
    out.mkdir(parents=True, exist_ok=True)
    marker_path = out / "identity.json"
    if marker_path.exists():
        if json.loads(marker_path.read_text()) != identity:
            raise ValueError(f"{out} holds a different run (identity changed); use a new --out")
    elif any(f.name != PLAN_CACHE for f in out.iterdir()):
        raise ValueError(f"{out} is not empty and has no run identity")
    elif shutil.disk_usage(out).free - pool_bytes < MIN_FREE_BYTES:
        raise OSError("the pool array would leave under 5 GiB free; not started")
    write_atomic(marker_path, json_text(identity).encode())

    pretrained, _ = g_bundle.build_g(p["spec"])
    scorer = mk.FoldScorer(pretrained, p["spec"]["model"]["n_blocks"], device)
    for m in p["models"]:
        bundle = g_bundle.load_g(Path(m["path"]))
        if bundle.backbone != m["backbone"] or sha256_file(Path(m["path"])) != m["sha256"]:
            raise ValueError(f"{m['path']} is not the verified bundle")
        scorer.add(m["fold"], m["seed"], m["temperature"], bundle.model)
    if any(len(s) != p["n_seeds"] for s in scorer.seeds.values()):
        raise ValueError(f"every fold needs {p['n_seeds']} seeds: {scorer.seeds}")
    seeds = scorer.seeds[0]
    size_wh, mask = tuple(p["spec"]["input"]["size_wh"]), p["spec"]["input"]["mask"]
    pool = open_pool(out, len(plan), p["n_seeds"], width)
    chunks = {int(n): rows for n, rows in plan.groupby("chunk", sort=True)}
    log(f"models loaded ({time.time() - started:.0f}s)")

    def work(numbers: list[int]) -> bool:
        """Score the uncommitted chunks of `numbers`; False when the time budget stopped it."""
        todo = [n for n in numbers if not committed(out, n, chunks[n], pool)]
        log(f"{len(numbers) - len(todo)} of {len(numbers)} chunks already committed")
        for k, number in enumerate(todo):
            if args.max_minutes is not None and time.time() - started > args.max_minutes * 60:
                log(f"time budget reached; {len(todo) - k} chunks left, rerun to resume")
                return False
            if device == "cuda" and k % 20 == 0:
                check_gpu_free()
            began, rows = time.time(), chunks[number]
            table, values = score_chunk(rows, scorer, args.batch_size, args.workers, size_wh, mask)
            commit(out, number, rows, table, values, pool)
            if k % 10 == 0 or k == len(todo) - 1:
                log(f"chunk {number}: {len(rows)} frames in {time.time() - began:.1f}s "
                    f"({len(rows) / max(time.time() - began, 1e-9):.0f}/s); {len(todo) - k - 1} left")
        return True

    gate_chunks = sorted(plan[plan.phase.eq("gate")].chunk.unique().tolist())
    if not work(gate_chunks):
        return 0
    gate = gate_record(read_chunks(out, gate_chunks), p["reference"])
    write_atomic(out / "gate.json", json_text(gate).encode())
    log(f"gate: max |Δz| {gate['max_abs_delta']:.3g}, mean {gate['mean_abs_delta']:.3g}, r {gate['pearson_r']:.9f}, "
        f"passed {gate['passed']}")
    if not gate["passed"]:
        raise SystemExit("GATE FAILED: the unmasked replay does not reproduce z_g_combo; no masked frame is scored")
    if args.gate_only:
        return 0
    masked_chunks = sorted(plan[plan.phase.ne("gate")].chunk.unique().tolist())
    if not work(masked_chunks):
        return 0

    table = read_chunks(out, sorted(chunks))
    if not np.array_equal(table.row.to_numpy(), plan.row.to_numpy()):
        raise ValueError("the chunk tables do not cover the plan")
    table.to_parquet(out / "scores.parquet", index=False)
    plan.drop(columns="jpeg").to_parquet(out / "plan.parquet", index=False)
    p["unions"].to_parquet(out / "union_placebos.parquet", index=False)
    unmasked_reference(table, p["reference"]).to_parquet(out / "unmasked.parquet", index=False)
    pool.flush()
    pool = None                                             # release the memmap before hashing pool.npy
    shutil.rmtree(out / "chunks")
    shutil.rmtree(out / PLAN_CACHE)
    marker_path.unlink()
    files = {f.name: sha256_file(f) for f in sorted(out.iterdir()) if f.is_file()}
    summary = summarise(table, gate, p, seeds)
    record = {"identity": identity | {"device": device, "runtime": runtime()}, "files": files, "gate": gate,
              "region_check": p["region_check"], "summary": summary,
              "pool": {"shape": [len(plan), p["n_seeds"], width], "dtype": "float16", "axes": ["row", "seed", "dim"],
                       "seeds": seeds},
              "sources": {**own_sources(), "scripts/xgdino_mask_embed.py": sha256_file(Path(__file__))},
              "seconds_last_invocation": round(time.time() - started, 1),
              "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_atomic(out / "complete.json", json_text(record).encode())
    write_summary(args.summary, out, record, table)
    log(f"MASKED DONE: {summary['rows']} rows, {summary['shots']} shots")
    return 0


def unmasked_reference(table: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """The replayed unmasked z per shot (the `unmasked` variant of scores.parquet): ticket 15's unmasked reference, so
    masked-vs-unmasked contrasts carry no replay noise; `z_g_combo` beside it for the continuity check only."""
    rows = table[table.kind.eq("unmasked")]
    out = rows[["shot_id", "fold", "row", "z", *[c for c in rows.columns if c.startswith("logit_")], "on_target"]].copy()
    out["z_g_combo"] = out.shot_id.map(reference).to_numpy(np.float64)
    out["abs_delta"] = (out.z - out.z_g_combo).abs()
    return out.reset_index(drop=True)


def read_chunks(out: Path, numbers: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(chunk_files(out, n)[0]) for n in numbers], ignore_index=True)


def summarise(table: pd.DataFrame, gate: dict, p: dict, seeds: list[int]) -> dict:
    """Counts per variant and fill, and the mean shift of z against the shot's unmasked z (descriptive)."""
    base = table[table.kind.eq("unmasked")].set_index("shot_id").z
    masked = table[table.kind.ne("unmasked")].assign(dz=lambda t: t.z - t.shot_id.map(base))
    groups = masked.assign(group=np.where(masked.kind.eq("placebo"), masked.region + "|placebo", masked.region))
    shift = groups.groupby(["group", "fill"]).agg(rows=("dz", "size"), shots=("shot_id", "nunique"),
                                                  mean_dz=("dz", "mean"), mean_abs_dz=("dz", lambda d: d.abs().mean()))
    return {"rows": len(table), "shots": int(table.shot_id.nunique()), "seeds": seeds,
            **counts(table.drop(columns=[c for c in table.columns if c.startswith("logit_")])),
            "gate": {k: gate[k] for k in ("passed", "max_abs_delta", "mean_abs_delta", "pearson_r", "rule", "counts_above", "shots")},
            "union_placebo_status": {str(k): int(v) for k, v in p["unions"].placebo_status.value_counts().items()},
            "dz_by_variant": [{"variant": g, "fill": f, **{k: (float(v) if k.endswith("dz") else int(v)) for k, v in r.items()}}
                              for (g, f), r in shift.iterrows()]}


def write_summary(folder: Path, out: Path, record: dict, table: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    counts_table = (table.groupby(["kind", "region", "fill"])
                    .agg(rows=("row", "size"), shots=("shot_id", "nunique")).reset_index())
    counts_table.to_csv(folder / "variant_counts.csv", index=False)
    small = {k: record[k] for k in ("gate", "region_check", "summary", "pool", "seconds_last_invocation", "completed_at")}
    small["storage"] = {"folder": str(out), "complete_json_sha256": sha256_file(out / "complete.json"),
                        "files": record["files"]}
    write_atomic(folder / "summary.json", json_text(small).encode())
    write_atomic(folder / "complete.json", json_text({"files": records.output_hashes(folder),
                                                      "completed_at": record["completed_at"]}).encode())


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
