"""Per-layer embeddings of frozen DINOv2-L and the five seed-0 G-combo fold models (xg-dino ticket 05, council C1).

    ./run.sh --no-sync python scripts/xgdino_layers.py --check-only                        # inputs and plan, no scoring
    ./run.sh --no-sync python scripts/xgdino_layers.py --device cuda --gpu-approved        # the full run (resumable)

Shots: the 7,062 training shots of xg-temporal 01b (`results/xgt_inputs/extraction_shots.csv`, every match a
`training` match of the xgv split plan) and the 2,721 paired shots (the non-penalty `primary` rows of
`results/xgv_eval_primary/shots.parquet`; only `key`, `role` and `is_penalty` are read, before the StatsBomb guard is
entered, and the penalty flag is checked against the offset frames). Offsets 0 and -400 ms, frames from the offset-frames
version `cache/xgc/offset_frames_v1` (the xg-temporal / g_xgv path: lossless PNG, sha256 checked per file).

Models: frozen DINOv2-L (the pretrained network of the bundles) and the seed-0 models of folds 0-4 of
`results/xgv_g_combo`. Every frame is embedded by all five fold models (C1 routing); `index.parquet` names each training
shot's own fold. Kinds and shapes: `xgpoc/dino/layers.py`.

Outputs in --out (default /mnt/storage/tactifoot_research/xg_dino/layers; ntfs3, so no symlinks and plain names):
    <kind>.npy           float16, one row per frame with `row` >= 0 in the index
    index.parquet        shot_id, set (training / paired), offset_ms, fold (-1 for paired), game, split, frame_status,
                         decoded_relative_ms, timing_error_ms, cut_between_anchor, file_sha256, row, chunk
    complete.json        identity, sha256 of every file, checks, summary; written last
Checks: on a sample of frames every fold model's own full forward against the shared path (blocks 0-19 against the
frozen ones, tail kinds against the tail path; max |Δ|) and the frozen final-normed tokens against the existing frozen
embedding path (`FrozenTail`); after the run the own-fold `g_pool` at offset 0 of the training shots against the g_xgv
cache (`results/xgt_xgv_embeddings`, seed 0; mean |Δ|, r), with a wrong-fold contrast.
Resume: chunks of plan rows; a chunk counts when its marker exists and its array slices hash as recorded.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_xgc_g import OVERLAY, own_sources
from run_xgc_transfer import check_gpu_free
from score_offset_frames import (
    OFFSET_FRAMES,
    PROTOCOL,
    XGT_G_COMBO,
    XGT_SHOTS,
    check_frame_files,
    check_layout,
    json_text,
    load_offset_frames,
    runtime,
    verify_bundles,
    write_atomic,
    xgt_roles,
)
from torch.utils.data import DataLoader
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import layers
from xgpoc.models import g_bundle
from xgpoc.paths import INDEX
from xgpoc.probe.scoring import FrameDataset
from xgpoc.temporal import inputs as ti

ROOT = Path(__file__).resolve().parents[1]
VERSION = "xgdino_layers_v1"
OUT = Path("/mnt/storage/tactifoot_research/xg_dino/layers")
PAIRED = ROOT / "results/xgv_eval_primary/shots.parquet"
PAIRED_COLUMNS = ["key", "role", "is_penalty"]
OFFSETS = (0, -400)
SEED = 0
FRAME_STATUSES = ("ok", "anchor_off_tolerance")   # as xgpoc.probe.table: a stored frame exists
MIN_FREE_BYTES = 5 * 2**30
INDEX_FILE = "index.parquet"


# --- inputs --------------------------------------------------------------------------------------------------------------

def read_paired(path: Path = PAIRED) -> list[str]:
    """The paired shot ids: non-penalty `primary` rows of the label-free shot list, three columns only."""
    records.verified_file(path)
    shots = pd.read_parquet(path, columns=PAIRED_COLUMNS)
    keys = shots[shots.role.eq("primary") & ~shots.is_penalty.astype(bool)].key
    if keys.duplicated().any():
        raise ValueError(f"{path}: a paired shot appears twice")
    return sorted(keys)


def build_plan(training: pd.DataFrame, paired: list[str], shots: pd.DataFrame, frames: pd.DataFrame, folds: pd.DataFrame,
               frames_root: Path, chunk_rows: int, offsets: tuple[int, ...] = OFFSETS) -> pd.DataFrame:
    """One row per shot and offset, training shots first, then paired; shots by id, offsets in `offsets` order. `row`
    is the array row (-1 without a stored frame), `fold` the training shot's own held-out fold (-1 for paired)."""
    if overlap := set(training.shot_id) & set(paired):
        raise ValueError(f"{len(overlap)} shots are both training and paired")
    meta = shots.set_index("shot_id")
    if missing := (set(training.shot_id) | set(paired)) - set(meta.index):
        raise ValueError(f"{len(missing)} shots are not in the offset frames")
    if meta.loc[paired, "is_penalty"].astype(bool).any() or meta.loc[training.shot_id, "is_penalty"].astype(bool).any():
        raise ValueError("a penalty is in the shot list")
    fold_of = folds.set_index("game").fold
    sets = pd.concat([pd.DataFrame({"shot_id": sorted(training.shot_id), "set": "training"}),
                      pd.DataFrame({"shot_id": paired, "set": "paired"})], ignore_index=True)
    sets["game"] = sets.shot_id.map(meta.game)
    sets["split"] = sets.shot_id.map(meta.split)
    sets["fold"] = np.where(sets.set.eq("training"), sets.game.map(fold_of), layers.NO_FOLD)
    if sets.fold.isna().any():
        raise ValueError("a training shot's game has no fold")
    wanted = frames[frames.shot_id.isin(sets.shot_id) & frames.offset_ms.isin(offsets)]
    if len(wanted) != len(sets) * len(offsets) or wanted.duplicated(["shot_id", "offset_ms"]).any():
        raise ValueError("the offset frames do not hold every offset for every shot exactly once")
    rows = sets.merge(wanted[["shot_id", "offset_ms", "status", "decoded_relative_ms", "timing_error_ms",
                              "cut_between_anchor", "frame_path", "file_sha256", "file_bytes"]], on="shot_id",
                      validate="one_to_many")
    rows["set_rank"] = rows.set.map({"training": 0, "paired": 1})
    rows["offset_rank"] = rows.offset_ms.map({o: i for i, o in enumerate(offsets)})
    rows = rows.sort_values(["set_rank", "shot_id", "offset_rank"]).reset_index(drop=True)
    rows = rows.rename(columns={"status": "frame_status"})
    has = rows.frame_status.isin(FRAME_STATUSES) & rows.frame_path.notna()
    rows["row"] = np.where(has, has.cumsum() - 1, -1)
    rows["chunk"] = np.arange(len(rows)) // chunk_rows
    rows["frame_file"] = np.where(has, rows.frame_path.map(lambda p: str(frames_root / p) if isinstance(p, str) else ""), "")
    rows["has_frame"] = has
    rows["fold"] = rows.fold.astype(int)
    rows["cut_between_anchor"] = rows.cut_between_anchor.astype("boolean")
    return rows.drop(columns=["set_rank", "offset_rank", "frame_path"])


def status_counts(plan: pd.DataFrame) -> dict[str, int]:
    """Frame status per set and offset, keyed "set|offset|status"."""
    counts = plan.groupby(["set", "offset_ms"]).frame_status.value_counts()
    return {"|".join(map(str, k)): int(v) for k, v in counts.items()}


def index_table(plan: pd.DataFrame) -> pd.DataFrame:
    return plan[["shot_id", "set", "offset_ms", "fold", "game", "split", "frame_status", "decoded_relative_ms",
                 "timing_error_ms", "cut_between_anchor", "file_sha256", "row", "chunk"]]


def plan_digest(plan: pd.DataFrame) -> str:
    keys = index_table(plan).astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest()


# --- arrays and chunks ---------------------------------------------------------------------------------------------------

def open_arrays(out: Path, n_rows: int, dims: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    arrays = {}
    for kind, shape in dims.items():
        path, full = out / f"{kind}.npy", (n_rows, *shape)
        if not path.exists():
            temporary = path.with_name(path.name + ".tmp")
            np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=full).flush()
            temporary.rename(path)
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != full or array.dtype != np.float16:
            raise ValueError(f"{path} has shape {array.shape} {array.dtype}, expected {full} float16")
        arrays[kind] = array
    return arrays


def chunk_slice(rows: pd.DataFrame) -> tuple[int, int]:
    kept = rows.row[rows.row >= 0]
    if kept.empty:
        return 0, 0
    start, end = int(kept.iloc[0]), int(kept.iloc[-1]) + 1
    if end - start != len(kept):
        raise ValueError("array rows of a chunk are not contiguous")
    return start, end


def slice_hash(array: np.ndarray, start: int, end: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(array[start:end]).tobytes()).hexdigest()


def marker_file(out: Path, number: int) -> Path:
    return out / "chunks" / f"chunk_{number:05d}.json"


def marker(rows: pd.DataFrame, arrays: dict[str, np.ndarray]) -> dict:
    start, end = chunk_slice(rows)
    return {"start": start, "end": end, "sha256": {k: slice_hash(a, start, end) for k, a in arrays.items()}}


def committed(out: Path, number: int, rows: pd.DataFrame, arrays: dict[str, np.ndarray]) -> bool:
    try:
        return json.loads(marker_file(out, number).read_text()) == marker(rows, arrays)
    except (OSError, ValueError):
        return False


def encode_chunk(rows: pd.DataFrame, encoder: layers.LayerEncoder, batch_size: int, workers: int,
                 size_wh: tuple[int, int], mask: str) -> dict[str, np.ndarray]:
    images = rows[rows.has_frame]
    loader = DataLoader(FrameDataset([Path(p) for p in images.frame_file], images.file_sha256.tolist(), size_wh, mask),
                        batch_size, shuffle=False, num_workers=workers)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in layers.KINDS}
    for batch in loader:
        for kind, value in encoder.encode(batch).items():
            if not np.isfinite(value).all() or np.abs(value).max() > np.finfo(np.float16).max:
                raise ValueError(f"{kind}: non-finite or float16-overflowing value")
            parts[kind].append(value.astype(np.float16))
    return {k: np.concatenate(v) for k, v in parts.items()}


# --- checks --------------------------------------------------------------------------------------------------------------

def sample_rows(plan: pd.DataFrame, n: int) -> pd.DataFrame:
    """`n` evenly spaced plan rows with a frame (training and paired, both offsets)."""
    have = plan[plan.has_frame]
    return have.iloc[np.unique(np.linspace(0, len(have) - 1, min(n, len(have))).round().astype(int))]


def sample_check(rows: pd.DataFrame, encoder: layers.LayerEncoder, models: dict[int, torch.nn.Module], size_wh, mask) -> dict:
    """Each fold model's own full forward against the shared path, and the frozen final-normed tokens against
    `FrozenTail` (the existing frozen-embedding path), on the same batch and device."""
    images = torch.stack(list(FrameDataset([Path(p) for p in rows.frame_file], rows.file_sha256.tolist(), size_wh, mask)))
    device = encoder.scorer.device
    per_fold = {}
    for fold, model in models.items():
        full = copy.deepcopy(model).to(device)
        per_fold[str(fold)] = layers.full_forward_gap(encoder, fold, full, images)
        del full
        if device == "cuda":
            torch.cuda.empty_cache()
    encoded = encoder.encode(images)
    _, frozen = encoder.scorer.score(images, [], frozen=True)
    d = encoded["dino_final"].shape[-1]
    frozen_gap = float(np.abs(encoded["dino_final"].reshape(len(rows), 2 * d).astype(np.float64) - frozen).max())
    worst = {k: max(v[k] for v in per_fold.values()) for k in ("shared_blocks", "g_blocks", "g_final", "g_pool")}
    return {"frames": len(rows), "shot_offsets": [f"{s}@{o}" for s, o in zip(rows.shot_id, rows.offset_ms, strict=True)],
            "max_abs_delta_per_fold": per_fold, "max_abs_delta": worst,
            "frozen_final_vs_frozen_tail_max_abs_delta": frozen_gap,
            "definition": "shared_blocks: each fold model's own blocks 0-19 vs the frozen DINOv2 blocks (CLS, mean patch, "
                          "before the final norm); tail kinds: the model's own full forward vs the shared-trunk tail path"}


def g_xgv_check(plan: pd.DataFrame, g_pool: np.ndarray, model_folds: list[int]) -> dict:
    """Own-fold `g_pool` at offset 0 of the training shots against the seed-0 g_xgv cache; wrong fold as contrast."""
    record = json.loads((ti.XGV_EMBEDDINGS / "complete.json").read_text())
    if sha256_file(ti.XGV_EMBEDDINGS / ti.TABLE) != record["files"][ti.TABLE]:
        raise ValueError("the g_xgv table does not match its completion record")
    entry = record["embeddings"]["g_combo"]
    if entry["seed"] != SEED or sha256_file(Path(entry["path"])) != entry["sha256"]:
        raise ValueError("the g_xgv seed-0 array is not the recorded one")
    table = pd.read_parquet(ti.XGV_EMBEDDINGS / ti.TABLE, columns=["shot_id", "offset_ms", "encoder", "fold_source", "embedding_ref"])
    table = table[table.encoder.eq("g_combo") & table.offset_ms.eq(0) & table.embedding_ref.ge(0)].set_index("shot_id")
    ours = plan[plan.set.eq("training") & plan.offset_ms.eq(0) & plan.has_frame & plan.shot_id.isin(table.index)]
    if (table.loc[ours.shot_id, "fold_source"].to_numpy() != ours.fold.to_numpy()).any():
        raise ValueError("g_xgv and the plan disagree on a shot's fold")
    reference = np.load(entry["path"], mmap_mode="r")[table.loc[ours.shot_id, "embedding_ref"].to_numpy()].astype(np.float64)

    def compare(folds: np.ndarray) -> dict:
        mine = layers.own_fold(g_pool, ours.row.to_numpy(), folds, model_folds).astype(np.float64)
        delta = np.abs(mine - reference)
        return {"mean_abs_delta": float(delta.mean()), "max_abs_delta": float(delta.max()),
                "pearson_r": float(np.corrcoef(mine.ravel(), reference.ravel())[0, 1])}

    own = ours.fold.to_numpy()
    wrong = np.array([model_folds[(model_folds.index(f) + 1) % len(model_folds)] for f in own])
    return {"shots": len(ours), "reference": {"table_sha256": record["files"][ti.TABLE], "array_sha256": entry["sha256"]},
            "own_fold": compare(own), "next_fold_contrast": compare(wrong),
            "reference_mean_abs": float(np.abs(reference).mean())}


# --- run -----------------------------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--gpu-approved", action="store_true", help="required on the GPU (lock taken with agent-state)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-rows", type=int, default=512, help="plan rows per chunk, the unit of resume")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sample-frames", type=int, default=32, help="frames of the full-forward check")
    ap.add_argument("--limit-shots", type=int, help="smoke: the first N training and N paired shots (not the default --out)")
    ap.add_argument("--max-chunks", type=int, help="stop cleanly after this many new chunks (rerun to resume)")
    ap.add_argument("--check-only", action="store_true", help="verify the inputs, build the plan, print sizes")
    args = ap.parse_args(argv)
    if args.limit_shots and args.out.resolve() == OUT.resolve():
        ap.error("a smoke subset (--limit-shots) must not be written to the default --out")
    return args


def identity_of(args, offset_provenance: dict, models: list[dict], spec: dict, backbone: dict, plan: pd.DataFrame,
                paired_sha: str, sources: dict) -> dict:
    return json.loads(json.dumps({
        "version": VERSION, "offsets_ms": list(OFFSETS), "limit_shots": args.limit_shots,
        "shots": {"training": {"path": ti.recorded_path(XGT_SHOTS), "sha256": sha256_file(XGT_SHOTS)},
                  "paired": {"path": ti.recorded_path(PAIRED), "sha256": paired_sha, "columns": PAIRED_COLUMNS,
                             "rule": "role == primary and not is_penalty"}},
        "offset_frames": offset_provenance, "index_sha256": sha256_file(INDEX), "folds_sha256": sha256_file(a1.FOLDS),
        "bundles": [{k: m[k] for k in ("fold", "seed", "sha256", "commit", "code_digest", "plan_sha256")} for m in models],
        "spec": spec, "backbone": backbone, "kinds": list(layers.KINDS),
        "plan": {"rows": len(plan), "frames": int(plan.has_frame.sum()), "digest": plan_digest(plan),
                 "chunk_rows": args.chunk_rows},
        "encoding": {"batch_size": args.batch_size, "dtype": "float16 (computed in float32, eval mode)",
                     "tokens": "CLS token and mean of the patch tokens (prefix tokens excluded)"},
        "sources": sources, "runtime": runtime()}))


def run(args: argparse.Namespace) -> int:
    started = time.time()
    paired = read_paired()                                  # before the guard: the list lives next to StatsBomb tables
    paired_sha = sha256_file(PAIRED)
    with no_statsbomb_reads():
        return guarded_run(args, paired, paired_sha, started)


def guarded_run(args: argparse.Namespace, paired: list[str], paired_sha: str, started: float) -> int:
    device = "cpu" if args.check_only else args.device
    records.verified_file(XGT_SHOTS)
    training = pd.read_csv(XGT_SHOTS)
    shots, frames, offset_provenance = load_offset_frames(OFFSET_FRAMES, PROTOCOL, INDEX)
    folds = pd.read_csv(a1.FOLDS)
    every = verify_bundles(XGT_G_COMBO, "g_combo", INDEX, a1.FOLDS, ROOT / OVERLAY)
    spec, backbone, _ = check_layout({"g_combo": every}, folds, 10)
    models = [m for m in every if m["seed"] == SEED]
    xgt_roles(INDEX, a1.FOLDS, training, every)              # every training shot in a training match of the plan
    if args.limit_shots:
        training, paired = training.sort_values("shot_id").head(args.limit_shots), paired[: args.limit_shots]
    elif len(training) != 7062 or len(paired) != 2721:
        raise ValueError(f"expected 7,062 training and 2,721 paired shots, found {len(training)} and {len(paired)}")
    plan = build_plan(training, paired, shots, frames, folds, OFFSET_FRAMES, args.chunk_rows)
    sources = {**own_sources(), "scripts/xgdino_layers.py": sha256_file(Path(__file__))}
    identity = identity_of(args, offset_provenance, models, spec, backbone, plan, paired_sha, sources)
    n_rows = int(plan.has_frame.sum())
    width, n_blocks = 1024, 24                              # DINOv2-L; checked against the built encoder below
    per_frame = (n_blocks * 2 + 2 + len(models) * (spec["model"]["n_blocks"] * 2 + 2 + 5)) * width * 2   # float16 bytes
    print(f"{plan.shot_id.nunique()} shots ({plan[plan.set.eq('training')].shot_id.nunique()} training, "
          f"{plan[plan.set.eq('paired')].shot_id.nunique()} paired), {len(plan)} rows, {n_rows} frames, "
          f"{plan.chunk.nunique()} chunks, arrays {n_rows * per_frame / 2**30:.2f} GiB, device {device}", flush=True)
    if args.check_only:
        print(json_text({"frame_status": status_counts(plan),
                         "cut_between_anchor": {k: int(v) for k, v in plan.groupby("set").cut_between_anchor.sum().items()},
                         "folds": {int(k): int(v) for k, v in
                                   plan[plan.set.eq("training")].drop_duplicates("shot_id").fold.value_counts().sort_index().items()},
                         "plan_digest": identity["plan"]["digest"]}))
        return 0
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; nothing to do")
    check_frame_files(plan)
    if device == "cuda":
        if not args.gpu_approved:
            raise SystemExit("refusing to use the GPU without --gpu-approved")
        check_gpu_free()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    marker_path = out / "identity.json"
    if marker_path.exists():
        if json.loads(marker_path.read_text()) != identity:
            raise ValueError(f"{out} holds a different run (identity changed); use a new --out")
    elif any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no run identity")
    elif shutil.disk_usage(out).free - n_rows * per_frame < MIN_FREE_BYTES:
        raise OSError("the arrays would leave under 5 GiB free; not started")
    write_atomic(marker_path, json_text(identity).encode())

    pretrained, _ = g_bundle.build_g(spec)
    encoder = layers.LayerEncoder(pretrained, spec["model"]["n_blocks"], device)
    loaded = {}
    for m in models:
        bundle = g_bundle.load_g(Path(m["path"]))
        if bundle.backbone != m["backbone"] or sha256_file(Path(m["path"])) != m["sha256"]:
            raise ValueError(f"{m['path']} is not the verified bundle")
        encoder.add(m["fold"], m["seed"], bundle.model)
        loaded[m["fold"]] = bundle.model
    if sum(int(np.prod(d)) for d in encoder.dims.values()) * 2 != per_frame:
        raise ValueError("array sizes disagree with the estimate the disk check used")
    size_wh, mask = tuple(spec["input"]["size_wh"]), spec["input"]["mask"]
    check = sample_check(sample_rows(plan, args.sample_frames), encoder, loaded, size_wh, mask)
    write_atomic(out / "sample_check.json", json_text(check).encode())
    print(f"sample check: {check['max_abs_delta']}, frozen tail {check['frozen_final_vs_frozen_tail_max_abs_delta']:.2e}",
          flush=True)
    if check["max_abs_delta"]["shared_blocks"] != 0.0:
        raise ValueError("blocks 0-19 of a fold model differ from the frozen DINOv2 blocks")

    arrays = open_arrays(out, n_rows, encoder.dims)
    chunks = {int(n): rows for n, rows in plan.groupby("chunk", sort=True)}
    todo = [n for n, rows in chunks.items() if not committed(out, n, rows, arrays)]
    print(f"{len(chunks) - len(todo)} of {len(chunks)} chunks already committed", flush=True)
    for done, number in enumerate(todo):
        if args.max_chunks is not None and done >= args.max_chunks:
            print(f"stopping after {done} chunks as asked; rerun to resume")
            return 0
        if device == "cuda":
            check_gpu_free()
        began, rows = time.time(), chunks[number]
        values = encode_chunk(rows, encoder, args.batch_size, args.workers, size_wh, mask)
        start, end = chunk_slice(rows)
        for kind, array in arrays.items():
            if len(values[kind]) != end - start:
                raise ValueError("encoded rows do not match the plan")
            array[start:end] = values[kind]
            array.flush()
        write_atomic(marker_file(out, number), json_text(marker(rows, arrays)).encode())
        print(f"chunk {number}: {end - start} frames {time.time() - began:.0f}s; {len(todo) - done - 1} left "
              f"({time.strftime('%H:%M')})", flush=True)

    index_table(plan).to_parquet(out / INDEX_FILE, index=False)
    g_check = g_xgv_check(plan, arrays["g_pool"], encoder.folds)
    for array in arrays.values():
        array.flush()
    arrays.clear()
    shutil.rmtree(out / "chunks")
    marker_path.unlink()
    files = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    shapes = {k: list(np.load(out / f"{k}.npy", mmap_mode="r").shape) for k in layers.KINDS}
    record = {"identity": identity | {"device": device}, "files": files,
              "kinds": {k: {"shape": shapes[k], "dtype": "float16", "axes": axes} for k, axes in KIND_AXES.items()},
              "model_folds": encoder.folds,
              "checks": {"sample": check, "g_xgv_offset0": g_check},
              "summary": {"frames": n_rows, "rows": len(plan), "shots": int(plan.shot_id.nunique()),
                          "frame_status": status_counts(plan),
                          "bytes": sum((out / f).stat().st_size for f in files),
                          "seconds_this_run": round(time.time() - started, 1),
                          "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None},
              "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_atomic(out / "complete.json", json_text(record).encode())
    print(json_text(record["checks"]["g_xgv_offset0"]))
    print(f"LAYERS DONE ({time.strftime('%H:%M')}): {record['summary']}")
    return 0


KIND_AXES = {"dino_blocks": ["row", "block 0-23", "token (CLS, mean patch)", "dim"],
             "dino_final": ["row", "token (CLS, mean patch)", "dim"],
             "g_blocks": ["row", "model (model_folds)", "block 20-23", "token (CLS, mean patch)", "dim"],
             "g_final": ["row", "model (model_folds)", "token (CLS, mean patch)", "dim"],
             "g_pool": ["row", "model (model_folds)", "dim (CLS + 4 queries, 5 x 1024)"]}


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
