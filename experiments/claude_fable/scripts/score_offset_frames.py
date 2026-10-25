"""Score every offset frame into the per-shot prediction table of the direction probe (xg-credibility ticket 08).

    ./run.sh python scripts/score_offset_frames.py --gpu-approved          # full run -> results/xgc_offset_scores
    ./run.sh python scripts/score_offset_frames.py --check-only            # verify inputs/plan, print sizes, no scoring
    ./run.sh python scripts/score_offset_frames.py --spot-check            # offset-0 check of a finished table, no scoring

Every non-penalty shot (goals included: ticket 10's attribution rows need them) is scored at the 30 offsets of the
approved grid (`research/xgc_direction_probe_protocol.md`) by three encoders:

    g_combo           ticket 04 ten fold models (xG logit, on-target logit, 5120-dim embedding)
    g_combo_no_head   ticket 05 control, same layout (xG logit, embedding)
    dinov2_frozen     the pretrained DINOv2-L the G models start from, untouched (2048-dim embedding)

Out-of-fold rule. A train or valid shot is scored by the models of the fold that held out its match (two seeds, logits
averaged); a test shot by all ten models of the encoder, logits averaged before any calibration. The embedding of a
train/valid shot is the one of the seed-0 model of its fold (protocol, "Gp i G0p"); test shots get no G embedding.
`xgpoc/probe/table.py` documents every column of the table.

Efficiency. Every frame is decoded once and goes through the frozen DINOv2 trunk once (the 20 frozen blocks are the
same in all models, checked tensor by tensor); then through the trained tail of each model that needs it
(`xgpoc/probe/scoring.py`). A model is loaded when the first chunk that needs it comes up and then kept (the test
chunks need all of them; 20 tails are about 4.2 GB on the GPU).

Outputs in --out (all listed in complete.json with their sha256):
    predictions.parquet                  the table (typed columns; CSV would triple the size and lose dtypes)
    embeddings_<encoder>.npy             float16 [rows, dim], one array per encoder, rows referenced by `embedding_ref`
    complete.json                        identity (protocol hash, input hashes, bundle provenance, source hashes), file
                                         hashes, spot check against the ticket-04/05 predictions, summary
Resume. Work is done in chunks of whole shots of one fold. A chunk is committed by a marker file written after its rows
and embeddings are on disk; a rerun re-checks every committed chunk (parquet and embedding hashes), redoes the others,
and refuses to continue when the recorded identity differs from the requested one. Disk: at least 5 GiB stay free.
The GPU is used only with --gpu-approved and only while no other python process holds it.

Layout xgt (xg-temporal ticket 01b, `--layout xgt`): the G-combo embeddings of the xgv fold models for the window of
xg-temporal, in the folder contract of `xgpoc/temporal/inputs.py` (source g_xgv).

    ./run.sh --no-sync python scripts/score_offset_frames.py --layout xgt --check-only --device cpu
    ./run.sh --no-sync python scripts/score_offset_frames.py --layout xgt --embed-seeds 0 1 --device cuda --gpu-approved

    encoders      g_combo only (no head-free control, no frozen DINOv2), models `results/xgv_g_combo` (10 bundles)
    shots         --shots-csv, by default the 01b extraction list `results/xgt_inputs/extraction_shots.csv` (checked
                  against its completion record); every shot's match must be a `training` match of the xgv split plan
                  (`xgpoc.detector.approach1.verify_plan`: plan.csv hash, folds file = the plan's training folds), and
                  the bundles must name that plan
    offsets       by default the 11 offsets -1600 ... +400 ms of `xgpoc.temporal.inputs.GRID`
    folds         `cache/folds_xgv_home5.csv`; there is no test group: every shot is scored by the two models of the
                  fold that held out its match (also the 31 matches of the official v1.2 test split), logits averaged
                  over the two seeds exactly as `z_g_combo` (`xg_logit` = mean of logit / T)
    embeddings    `--embed-seeds` (default 0): seed 0 is `embeddings_g_combo.npy` (the contract's array); every further
                  seed k is `embeddings_g_combo_seed<k>.npy` with the same `embedding_ref` rows (no extra GPU pass: the
                  tails of both seeds run for the logits anyway)
    outputs       --out (default `results/xgt_xgv_embeddings`): predictions.parquet and complete.json only; the arrays
                  in --embeddings-dir (default `outputs/xgt_xgv_embeddings`), named with absolute path and sha256 in
                  complete.json `embeddings` (`results/` is shared between worktrees through a symlink, `outputs/`
                  is not, so a relative path would point elsewhere from another worktree)
    spot check    against `z_g_combo` of `results/xgd_a1_ensemble/oof.csv` (only shot_id and z_g_combo are read, after
                  checking the file against its completion record), never against the training-time prediction files:
                  offset 0 on the stored PNG (descriptive: mean and largest |Δlogit| and the correlation), and the
                  index-JPEG control, which must agree within the batch-composition bound (exit code)
A smoke subset (another shot list or other offsets) must not be written to the default --out or --embeddings-dir.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from build_offset_frames import PROTOCOL_APPROVAL
from run_xgc_g import FOLDS_FILE, OVERLAY, own_sources
from run_xgc_transfer import check_gpu_free
from torch.utils.data import DataLoader
from xgpoc.bundle import canonical_digest, check_data_files, read_bundle, sha256_file
from xgpoc.data.datasets import STD
from xgpoc.data.offset_frames import GRID_MS
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.models import g_bundle
from xgpoc.paths import INDEX
from xgpoc.probe.scoring import (
    FrameDataset,
    GMember,
    SharedScorer,
    ensemble_logits,
)
from xgpoc.probe.table import (
    G_ENCODERS,
    TEST_GROUP,
    build_plan,
    chunk_table,
    reference_check,
    spot_check,
    validate_table,
)
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp

ROOT = Path(__file__).resolve().parents[1]
VERSION = "xgc_offset_scores_v1"
PROTOCOL = ROOT / "research/xgc_direction_probe_protocol.md"
OFFSET_FRAMES = ROOT / "cache/xgc/offset_frames_v1"
G_ROOTS = {"g_combo": ROOT / "results/xgc_g_combo", "g_combo_no_head": ROOT / "results/xgc_g_combo_no_head"}
AUX_WEIGHT = {"g_combo": 0.5, "g_combo_no_head": 0.0}
OUT = ROOT / "results/xgc_offset_scores"
MIN_FREE_BYTES = 5 * 2**30
OFFSET_FILES = ("frames.csv", "shots.csv", "summary.json", "coverage_by_offset.csv")
TABLE_FILE = "predictions.parquet"
FROZEN = "dinov2_frozen"

XGT_VERSION = "xgt_xgv_embeddings_v1"
XGT_OUT = ti.XGV_EMBEDDINGS                                 # results/xgt_xgv_embeddings: table and record only
XGT_EMBEDDINGS = ROOT / "outputs/xgt_xgv_embeddings"        # the arrays, named by path and sha256 in complete.json
XGT_SHOTS = ROOT / "results/xgt_inputs/extraction_shots.csv"
XGT_G_COMBO = ROOT / "results/xgv_g_combo"
XGT_REFERENCE = ti.OOF                                      # z_g_combo, read with shot_id only
XGT_REFERENCE_COLUMNS = ["shot_id", "z_g_combo"]


@dataclass(frozen=True)
class Layout:
    """What a run scores and where it writes: the direction probe of xg-credibility (xgc) or the xgv fold embeddings
    of xg-temporal (xgt, module docstring)."""

    name: str
    version: str
    g_encoders: tuple[str, ...]
    frozen: bool
    test_group: bool

    @property
    def encoders(self) -> tuple[str, ...]:
        return (*self.g_encoders, FROZEN) if self.frozen else self.g_encoders


LAYOUTS = {"xgc": Layout("xgc", VERSION, G_ENCODERS, True, True),
           "xgt": Layout("xgt", XGT_VERSION, ("g_combo",), False, False)}


def array_name(encoder: str, seed: int) -> str:
    """The embedding array of an encoder and seed: seed 0 is `embeddings_<encoder>.npy`, seed k `<encoder>_seed<k>`."""
    return encoder if seed == 0 else f"{encoder}_seed{seed}"


def array_names(layout: Layout, embed_seeds: tuple[int, ...]) -> dict[str, str]:
    """array name -> encoder, in layout order."""
    names = {array_name(e, s): e for e in layout.g_encoders for s in embed_seeds}
    return names | ({FROZEN: FROZEN} if layout.frozen else {})


def resolved(path: str | Path) -> Path:
    """A recorded path (relative to the experiment, or absolute)."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# --- inputs --------------------------------------------------------------------------------------------------------------

def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def without_outcome(column: str) -> bool:
    """A shots.csv column that may be read: none whose name holds a banned fragment of xg-temporal (`label`, `goal`,
    ...); nothing here scores by outcome."""
    return not any(fragment in column.lower() for fragment in tp.BANNED)


def load_offset_frames(root: Path, protocol: Path, index: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The shots and frames tables of the offset-frames version after checking its completion record: tables unchanged,
    approved protocol, 30-offset grid, lossless model-input storage, built from this index, and the protocol's stop
    rule not triggered. Frame files are checked one by one when they are read. The shots table comes without its
    outcome columns (`without_outcome`)."""
    record = json.loads((root / "complete.json").read_text())
    identity = record["identity"]
    for name in OFFSET_FILES:
        if sha256_file(root / name) != record["files"][name]:
            raise ValueError(f"{root / name} changed since the offset frames were built")
    if identity["protocol"]["sha256"] != sha256_file(protocol):
        raise ValueError("the offset frames were built under another protocol file")
    if identity["grid_ms"] != list(GRID_MS) or identity["storage"] != "model_input_png":
        raise ValueError("the offset frames are not the approved grid in model-input storage")
    if identity["sources"].get(str(index.resolve())) != sha256_file(index):
        raise ValueError("the offset frames were built from another index")
    summary = json.loads((root / "summary.json").read_text())
    if summary["protocol_population"]["stop_rule_triggered"] is not False:
        raise SystemExit("the offset frames triggered the protocol's stop rule (more than 5% of the population excluded); "
                         "this returns to the owner, nothing is scored")
    shots = pd.read_csv(root / "shots.csv", usecols=without_outcome)    # shots.csv carries label and goal: never read
    frames = pd.read_csv(root / "frames.csv", dtype={"cut_between_anchor": "boolean", "frame_path": str, "file_sha256": str})
    provenance = {"complete_json_sha256": sha256_file(root / "complete.json"),
                  **{name: record["files"][name] for name in OFFSET_FILES},
                  "protocol_population": summary["protocol_population"], "protocol_approval": identity["protocol"]["approval"]}
    return shots, frames, provenance


def verify_bundles(root: Path, encoder: str, index: Path, folds: Path, overlay: Path) -> list[dict]:
    """Completion records, file hashes, manifest, ensemble file, provenance and recipe of one encoder's fold models.
    Returns one record per model, sorted by (fold, seed)."""
    records = {p.parent: json.loads(p.read_text()) for p in sorted(root.glob("fold*/seed*/complete.json"))}
    if not records:
        raise ValueError(f"no finished runs under {root}")
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t").set_index(["fold", "seed"])
    ensemble = json.loads((root / "ensemble.json").read_text())
    members = {(m["fold"], m["seed"]): m for m in ensemble["members"]}
    gate = json.loads((root / "gate.json").read_text())
    if encoder == "g_combo" and gate["passed"] is not True:
        raise ValueError(f"{root}: the ticket-04 reproduction gate did not pass")
    out = []
    for run, record in records.items():
        for filename, expected in record["files"].items():
            if sha256_file(run / filename) != expected:
                raise ValueError(f"{run / filename} changed since it was written")
        ident = record["identity"]
        key = (ident["fold"], ident["seed"])
        bundle = read_bundle(run / "bundle.pt")
        provenance = bundle["provenance"]
        if provenance["code"]["digest"] != canonical_digest(record["code_sha256"]) or provenance["data"] != {
                "index_sha256": ident["index"]["sha256"], "folds_sha256": ident["folds"]["sha256"],
                "overlay_sha256": ident["overlay_sha256"]}:
            raise ValueError(f"{run}: bundle provenance disagrees with its completion record")
        check_data_files(provenance, index_sha256=index, folds_sha256=folds, overlay_sha256=overlay)
        sha = record["files"]["bundle.pt"]
        if manifest.loc[key, "sha256"] != sha or members[key]["sha256"] != sha:
            raise ValueError(f"{run}: manifest or ensemble file disagree on the bundle hash")
        temperature = bundle["calibration"]["temperature"]
        if not np.isclose(members[key]["temperature"], temperature, rtol=0, atol=1e-12):
            raise ValueError(f"{run}: ensemble file and bundle disagree on the temperature")
        if bundle["recipe"]["cfg"]["aux_weight"] != AUX_WEIGHT[encoder]:
            raise ValueError(f"{run}: auxiliary weight {bundle['recipe']['cfg']['aux_weight']} is not that of {encoder}")
        if bundle["recipe"]["fold"] != key[0] or bundle["recipe"]["seed"] != key[1]:
            raise ValueError(f"{run}: bundle recipe names another fold or seed")
        out.append({"encoder": encoder, "fold": key[0], "seed": key[1], "path": str(run / "bundle.pt"), "sha256": sha,
                    "temperature": float(temperature), "spec": bundle["spec"], "backbone": bundle["backbone"],
                    "commit": bundle["commit"], "code_digest": provenance["code"]["digest"],
                    "data": provenance["data"], "plan_sha256": ident.get("plan", {}).get("sha256")})
    return sorted(out, key=lambda m: (m["fold"], m["seed"]))


def check_layout(models: dict[str, list[dict]], folds: pd.DataFrame, expect_models: int) -> tuple[dict, dict, int]:
    """Same architecture and pretrained backbone everywhere; one model per fold and seed; folds as in the folds file.
    Returns (spec, backbone, number of seeds)."""
    every = [m for ms in models.values() for m in ms]
    spec, backbone = every[0]["spec"], every[0]["backbone"]
    if any(m["spec"] != spec or m["backbone"] != backbone for m in every):
        raise ValueError("the bundles do not share one architecture and pretrained backbone")
    fold_ids, seeds = sorted(folds.fold.unique().tolist()), None
    for encoder, ms in models.items():
        keys = [(m["fold"], m["seed"]) for m in ms]
        if len(keys) != len(set(keys)) or len(ms) != expect_models:
            raise ValueError(f"{encoder}: expected {expect_models} distinct fold models, found {len(ms)}")
        if sorted({k[0] for k in keys}) != fold_ids:
            raise ValueError(f"{encoder}: bundle folds differ from the folds file")
        by_fold = {f: sorted(s for g, s in keys if g == f) for f in fold_ids}
        seeds = seeds or by_fold[fold_ids[0]]
        if any(s != seeds for s in by_fold.values()):
            raise ValueError(f"{encoder}: every fold must have the same seeds")
    if 0 not in seeds:
        raise ValueError("the protocol's embeddings come from seed 0, which no model has")
    return spec, backbone, len(seeds)


def xgt_roles(index: Path, folds: Path, shots: pd.DataFrame, models: list[dict]) -> dict:
    """Layout xgt: the xgv split plan is the approved one and the folds file its training folds
    (`approach1.verify_plan`), every bundle was trained under that plan, and every requested shot lies in a `training`
    match (none of the 172 held-out matches). Returns the plan record of the run identity."""
    plan = a1.verify_plan(index, folds=folds)
    if wrong := [(m["fold"], m["seed"]) for m in models if m["plan_sha256"] != a1.PLAN_SHA256]:
        raise ValueError(f"bundles {wrong} were not trained under the xgv split plan {a1.PLAN_SHA256[:12]}")
    role = shots.game.map(plan.set_index("game").role)
    if (outside := shots[role.ne(tp.ROLE)]).shape[0]:
        raise ValueError(f"{len(outside)} shots lie outside the plan's training role, e.g. {outside.shot_id.iloc[0]} "
                         f"({role[outside.index[0]]})")
    return {"plan": {"path": ti.recorded_path(a1.PLAN_DIR / "plan.csv"), "sha256": a1.PLAN_SHA256},
            "roles": {str(k): int(v) for k, v in role.value_counts().items()},
            "matches": int(shots.game.nunique())}


def xgt_table_check(table: pd.DataFrame, index: Path, folds: Path) -> dict:
    """Layout xgt, on a finished table: the role of every match in the xgv split plan and whether every row was scored
    by the models of its match's xgv fold. Passes when both hold for every row."""
    plan = a1.verify_plan(index, folds=folds).set_index("game")
    games = table.drop_duplicates("game").game
    roles = games.map(plan.role).fillna("not in the plan")
    fold_ok = bool((table.fold_source.astype(int) == table.game.map(plan.fold)).all())
    return {"matches": int(len(games)), "shots": int(table.shot_id.nunique()),
            "match_roles": {str(k): int(v) for k, v in roles.value_counts().items()},
            "fold_source_is_the_xgv_fold": fold_ok, "passed": bool(roles.eq(tp.ROLE).all() and fold_ok)}


def xgt_reference(path: Path) -> tuple[pd.Series, dict]:
    """shot_id -> z_g_combo of the out-of-fold table, read with these two columns only after checking the file against
    its completion record; and its record for the identity."""
    digest = records.verified_file(path)
    table = pd.read_csv(path, usecols=XGT_REFERENCE_COLUMNS)
    if table.shot_id.duplicated().any():
        raise ValueError(f"{path}: a shot appears twice")
    return table.set_index("shot_id").z_g_combo, {"path": ti.recorded_path(path), "sha256": digest,
                                                  "columns": XGT_REFERENCE_COLUMNS}


# --- models --------------------------------------------------------------------------------------------------------------

class ModelBank:
    """The shared trunk and fold tails of both G encoders, loaded when a group first needs them."""

    def __init__(self, models: dict[str, list[dict]], spec: dict, device: str, layout: Layout, embed_seeds: tuple[int, ...]):
        self.models, self.device, self.layout, self.embed_seeds = models, device, layout, tuple(embed_seeds)
        self.arrays = array_names(layout, self.embed_seeds)
        pretrained, _ = g_bundle.build_g(spec)    # the untouched pretrained network: trunk and frozen encoder
        self.scorer = SharedScorer(pretrained.eval(), spec["model"]["n_blocks"], device)
        self.size_wh, self.mask = tuple(spec["input"]["size_wh"]), spec["input"]["mask"]
        width = pretrained.backbone.embed_dim   # 1024 for ViT-L: 5 x 1024 for G (CLS + 4 queries), 2 x 1024 for the frozen encoder
        self.dims = {"g_combo": 5 * width, "g_combo_no_head": 5 * width, FROZEN: 2 * width}

    def members(self, group: str) -> tuple[list[GMember], list[GMember]]:
        """(models that score the group, those whose embedding is kept: the `embed_seeds` models of the fold). A fold
        group needs that fold's models; the test group all of them, and keeps no G embedding."""
        wanted = [(m["encoder"], m["fold"], m["seed"]) for ms in self.models.values() for m in ms
                  if group == TEST_GROUP or group == f"fold{m['fold']}"]
        out = []
        for key in wanted:
            if key not in self.scorer.members:
                record = next(m for m in self.models[key[0]] if (m["fold"], m["seed"]) == key[1:])
                loaded = g_bundle.load_g(Path(record["path"]))
                if loaded.backbone != record["backbone"] or sha256_file(Path(record["path"])) != record["sha256"]:
                    raise ValueError(f"{record['path']} is not the verified bundle")
                self.scorer.add(key[0], key[1], key[2], record["temperature"], loaded.model)
            out.append(self.scorer.members[key])
        embed = [] if group == TEST_GROUP else [m for m in out if m.seed in self.embed_seeds]
        return out, embed


# --- run state -----------------------------------------------------------------------------------------------------------

def runtime() -> dict:
    return {"torch": torch.__version__, "timm": timm.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}


def plan_digest(plan: pd.DataFrame) -> str:
    keys = plan[["shot_id", "offset_ms", "group", "chunk", "file_sha256", "emb_row_g", "emb_row_frozen"]].fillna("")
    return hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest()


def requested_identity(args, offset_provenance: dict, models: dict[str, list[dict]], spec: dict, backbone: dict,
                       plan: pd.DataFrame, full: bool, sources: dict, xgt: dict | None = None) -> dict:
    """Everything that makes a table this table. A rerun with any of it changed is refused, not merged. `xgt`: the
    layout-xgt part (shot list, plan, reference, embedding seeds and folder)."""
    shots = plan.drop_duplicates("shot_id").shot_id
    return json.loads(json.dumps({
        "version": LAYOUTS[args.layout].version, **({"xgt": xgt} if xgt is not None else {}),
        "protocol": {"sha256": sha256_file(args.protocol), "approval": PROTOCOL_APPROVAL},
        "offset_frames": offset_provenance,
        "index_sha256": sha256_file(args.index), "folds_sha256": sha256_file(args.folds), "overlay_sha256": sha256_file(args.overlay),
        "bundles": {e: [{k: m[k] for k in ("fold", "seed", "sha256", "temperature", "commit", "code_digest", "data")}
                        | ({"plan_sha256": m["plan_sha256"]} if xgt is not None else {}) for m in ms]
                    for e, ms in models.items()},
        "spec": spec, "backbone": backbone,
        "plan": {"shots": len(shots), "rows": len(plan), "offsets_ms": sorted(plan.offset_ms.unique().tolist()),
                 "chunk_shots": args.chunk_shots, "shot_ids_sha256": hashlib.sha256("\n".join(shots).encode()).hexdigest(),
                 "digest": plan_digest(plan), "full": full},
        "scoring": {"batch_size": args.batch_size, "device": args.device_type, "embedding_dtype": "float16",
                    "control_shots": args.control_shots,
                    "xg_logit": "mean over the scoring models of logit / temperature (input of the ensemble Platt step)",
                    "xg_logit_raw": "mean over the scoring models of the raw logit",
                    "frozen_embedding": "CLS token and mean patch token after the last norm, 2 x 1024"},
        "sources": sources, "runtime": runtime()}))


def array_rows(plan: pd.DataFrame, encoder: str) -> int:
    return int(((plan.emb_row_frozen if encoder == FROZEN else plan.emb_row_g) >= 0).sum())


def open_embeddings(folder: Path, plan: pd.DataFrame, dims: dict[str, int], names: dict[str, str]) -> dict[str, np.ndarray]:
    """The float16 arrays `names` (array name -> encoder; created at full size, written chunk by chunk) in `folder`;
    sized from the plan."""
    arrays = {}
    for name, encoder in names.items():
        path = folder / f"embeddings_{name}.npy"
        shape = (array_rows(plan, encoder), dims[encoder])
        if not path.exists():
            temporary = path.with_name(path.name + ".tmp")
            np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape).flush()
            temporary.rename(path)
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.float16:
            raise ValueError(f"{path} has shape {array.shape} {array.dtype}, expected {shape} float16")
        arrays[name] = array
    return arrays


def embedding_slice(chunk_rows: pd.DataFrame, name: str) -> tuple[int, int]:
    column = chunk_rows.emb_row_frozen if name == FROZEN else chunk_rows.emb_row_g
    kept = column[column >= 0]
    if kept.empty:
        return 0, 0
    start, end = int(kept.iloc[0]), int(kept.iloc[-1]) + 1
    if end - start != len(kept):
        raise ValueError("embedding rows of a chunk are not contiguous")
    return start, end


def slice_hash(array: np.ndarray, start: int, end: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(array[start:end]).tobytes()).hexdigest()


def chunk_files(out: Path, number: int) -> tuple[Path, Path]:
    return out / "chunks" / f"chunk_{number:05d}.parquet", out / "chunks" / f"chunk_{number:05d}.json"


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.rename(path)


def chunk_is_committed(out: Path, number: int, chunk_rows: pd.DataFrame, arrays: dict[str, np.ndarray]) -> bool:
    """A chunk counts only if its marker exists and the rows and embeddings on disk hash as the marker says."""
    table_file, marker_file = chunk_files(out, number)
    try:
        marker = json.loads(marker_file.read_text())
        if sha256_file(table_file) != marker["table_sha256"]:
            return False
        for encoder, array in arrays.items():
            start, end = embedding_slice(chunk_rows, encoder)
            if marker["embeddings"][encoder] != {"start": start, "end": end, "sha256": slice_hash(array, start, end)}:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


@dataclass
class ChunkResult:
    table: pd.DataFrame
    embeddings: dict[str, np.ndarray]


def score_chunk(chunk_rows: pd.DataFrame, bank: ModelBank, batch_size: int, workers: int, keep_embeddings: bool = True) -> ChunkResult:
    """Decode the chunk's frames once and score them with every model that may see them."""
    group = chunk_rows.group.iloc[0]
    members, embed = bank.members(group)
    embed = embed if keep_embeddings else []
    images = chunk_rows[chunk_rows.has_frame]
    loader = DataLoader(FrameDataset([Path(p) for p in images.frame_file], images.file_sha256.tolist(), bank.size_wh, bank.mask),
                        batch_size, shuffle=False, num_workers=workers)
    layout = bank.layout
    by_encoder = {e: [m for m in members if m.encoder == e] for e in layout.g_encoders}
    collected = {e: {"xg_logit": [], "xg_logit_raw": [], "on_target_logit": []} for e in layout.g_encoders}
    embeddings = {name: [] for name in bank.arrays}
    for batch in loader:
        scores, frozen = bank.scorer.score(batch, members, embed, frozen=layout.frozen)
        lookup = {id(m): s for m, s in zip(members, scores, strict=True)}
        for encoder, ms in by_encoder.items():
            xg_t, xg_raw, on_target = ensemble_logits([lookup[id(m)] for m in ms], ms)
            collected[encoder]["xg_logit"].append(xg_t)
            collected[encoder]["xg_logit_raw"].append(xg_raw)
            collected[encoder]["on_target_logit"].append(on_target)
            for m in ms:
                if any(m is e for e in embed):
                    embeddings[array_name(encoder, m.seed)].append(lookup[id(m)].embedding.astype(np.float16))
        if layout.frozen:
            embeddings[FROZEN].append(frozen.astype(np.float16))
    values = {}
    for encoder in layout.g_encoders:
        values[encoder] = {c: np.concatenate(v) if v else np.zeros(0) for c, v in collected[encoder].items()}
        values[encoder]["on_target_logit"] = values[encoder]["on_target_logit"] if encoder == "g_combo" else None
        n = len(by_encoder[encoder])
        values[encoder] |= {"fold_source": -1 if group == TEST_GROUP else int(group.removeprefix("fold")),
                            "seed_source": ",".join(map(str, sorted({m.seed for m in by_encoder[encoder]}))), "n_models": n}
    if layout.frozen:
        values[FROZEN] = {"fold_source": -1, "seed_source": "", "n_models": 0}
    stacked = {name: np.concatenate(v) if v else np.zeros((0, bank.dims[bank.arrays[name]]), np.float16)
               for name, v in embeddings.items()}
    for name, array in stacked.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: non-finite or overflowing embedding in chunk of {group}")
    for encoder in layout.g_encoders:
        for column in ("xg_logit", "xg_logit_raw"):
            if not np.isfinite(values[encoder][column]).all():
                raise ValueError(f"{encoder}: non-finite {column} in chunk of {group}")
    expected_g = int((chunk_rows.emb_row_g >= 0).sum())
    if any(len(array) != (len(images) if name == FROZEN else expected_g) for name, array in stacked.items()):
        raise ValueError("embedding count does not match the plan")
    return ChunkResult(chunk_table(chunk_rows, values, layout.encoders), stacked)


def commit_chunk(out: Path, number: int, chunk_rows: pd.DataFrame, result: ChunkResult, arrays: dict[str, np.ndarray]) -> None:
    """Embeddings first, then the rows, then the marker that makes the chunk count."""
    marker = {"rows": len(result.table), "embeddings": {}}
    for encoder, array in arrays.items():
        start, end = embedding_slice(chunk_rows, encoder)
        if end > start:
            array[start:end] = result.embeddings[encoder]
            array.flush()
        marker["embeddings"][encoder] = {"start": start, "end": end, "sha256": slice_hash(array, start, end)}
    table_file, marker_file = chunk_files(out, number)
    temporary = table_file.with_name("." + table_file.name + ".tmp")
    table_file.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_parquet(temporary, index=False)
    temporary.rename(table_file)
    marker["table_sha256"] = sha256_file(table_file)
    write_atomic(marker_file, json_text(marker).encode())


def check_frame_files(plan: pd.DataFrame) -> None:
    """Every frame the plan scores exists with its recorded size (a stat per file; contents are hashed when read)."""
    rows = plan[plan.has_frame]
    bad = []
    for path, size in zip(rows.frame_file, rows.file_bytes, strict=True):
        try:
            if os.stat(path).st_size != size:
                bad.append(path)
        except OSError:
            bad.append(path)
    if bad:
        raise ValueError(f"{len(bad)} offset frames are missing or have another size than recorded, e.g. {bad[0]}")


# --- spot check ----------------------------------------------------------------------------------------------------------

def control_rows(plan: pd.DataFrame, index_path: Path, n: int) -> pd.DataFrame:
    """Offset-0 plan rows of `n` evenly spaced shots whose frame is `ok`, with the index frame (the JPEG the models were
    trained and evaluated on) in place of the stored PNG."""
    zero = plan[plan.offset_ms.eq(0) & plan.frame_status.eq("ok")]
    if zero.empty:
        raise ValueError("no offset-0 frame to run the spot check on (smoke runs need offset 0 in --offsets)")
    zero = zero.iloc[np.unique(np.linspace(0, len(zero) - 1, min(n, len(zero))).round().astype(int))].copy()
    jpeg = pd.read_parquet(index_path, columns=["shot_id", "frame_path"]).set_index("shot_id").frame_path
    zero["png_file"], zero["png_sha256"] = zero.frame_file, zero.file_sha256
    zero["frame_file"] = zero.shot_id.map(jpeg)
    zero["file_sha256"] = zero.frame_file.map(lambda f: sha256_file(Path(f)))
    zero[["emb_row_g", "emb_row_frozen"]] = -1
    return zero


def input_gap(rows: pd.DataFrame, bank: ModelBank) -> dict:
    """Model input of the stored PNG against the index JPEG for the same anchor frame, in 0-255 RGB units."""
    png = FrameDataset([Path(p) for p in rows.png_file], rows.png_sha256.tolist(), bank.size_wh, bank.mask)
    jpg = FrameDataset([Path(p) for p in rows.frame_file], rows.file_sha256.tolist(), bank.size_wh, bank.mask)
    signed, absolute = np.zeros(3), 0.0
    for i in range(len(rows)):
        difference = (png[i] - jpg[i]).numpy() * STD[:, None, None] * 255
        signed += difference.mean((1, 2))
        absolute += np.abs(difference).mean()
    return {"shots": len(rows), "mean_abs_rgb_units": float(absolute / len(rows)),
            "mean_signed_rgb_units_r_g_b": (signed / len(rows)).round(3).tolist(),
            "definition": "stored PNG minus index JPEG, model input after the overlay mask"}


def full_spot_check(table: pd.DataFrame, plan: pd.DataFrame, bank: ModelBank | None, args: argparse.Namespace,
                    roots: dict[str, Path]) -> dict:
    """The ticket's spot check and its control.

    `offset0_vs_stored`: the table's own offset-0 logits against the training-time predictions of the same models, as the
    ticket words it. The stored predictions were made on the index JPEG frames, while the table's frames are the lossless
    PNGs cut from the clips, so this differs by the frame source as well as by batch composition.
    `index_jpeg_control`: the same pipeline (shared trunk, out-of-fold rule, averaging) on the index JPEG of
    `--control-shots` evenly spaced shots, which isolates the pipeline: it must match within the batch-composition
    bound. The run passes (exit code 0) when the control does."""
    report = {"offset0_vs_stored": spot_check(table, roots)}
    rows = control_rows(plan, args.index, args.control_shots)
    parts = [score_chunk(group, bank, args.batch_size, args.workers, keep_embeddings=False).table for _, group in rows.groupby("group")]
    control = spot_check(pd.concat(parts, ignore_index=True), roots)
    control["shots"] = len(rows)
    control["files_sha256"] = canonical_digest(dict(zip(rows.shot_id, rows.file_sha256, strict=True)))
    report["index_jpeg_control"] = control
    report["input_gap"] = input_gap(rows, bank)
    report["pipeline_passed"] = control["passed"]
    report["offset0_passed"] = report["offset0_vs_stored"]["passed"]
    return report


def xgt_spot_check(table: pd.DataFrame, plan: pd.DataFrame, bank: ModelBank, args: argparse.Namespace,
                   reference: pd.Series) -> dict:
    """Layout xgt: the spot check against `z_g_combo` (module docstring). `offset0_vs_z_g_combo`: the table's offset-0
    `xg_logit` on the stored PNG, descriptive (the PNG cut from the clip differs from the index JPEG the models saw; xgc
    found a mean |Δlogit| of about 0.09). `index_jpeg_control`: the same pipeline on the index JPEG of `--control-shots`
    evenly spaced shots, which must agree within the batch-composition bound. `roles`: every match a training match
    and every row scored by its xgv fold (`xgt_table_check`). The run passes when the control and the roles do."""
    report = {"offset0_vs_z_g_combo": reference_check(table, reference)}
    rows = control_rows(plan, args.index, args.control_shots)
    parts = [score_chunk(group, bank, args.batch_size, args.workers, keep_embeddings=False).table for _, group in rows.groupby("group")]
    control = reference_check(pd.concat(parts, ignore_index=True), reference)
    control["files_sha256"] = canonical_digest(dict(zip(rows.shot_id, rows.file_sha256, strict=True)))
    report["index_jpeg_control"] = control
    report["input_gap"] = input_gap(rows, bank)
    report["roles"] = xgt_table_check(table, args.index, args.folds)
    report["pipeline_passed"] = bool(control["passed"] and report["roles"]["passed"])
    report["offset0_passed"] = report["offset0_vs_z_g_combo"]["passed"]
    return report


# --- completion ----------------------------------------------------------------------------------------------------------

def output_hashes(out: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ("complete.json", "identity.json")}


def without_device(identity: dict) -> dict:
    return {**identity, "scoring": {k: v for k, v in identity["scoring"].items() if k != "device"}}


def verify_complete(out: Path, identity: dict) -> dict:
    """A finished output must carry this identity (the device that scored it aside: verifying needs no GPU), and every
    file must still hash as recorded; extra files are an error."""
    record = json.loads((out / "complete.json").read_text())
    if without_device(record["identity"]) != without_device(identity):
        raise ValueError(f"{out} holds a different run than requested (identity changed); use a new --out")
    present = {p.name for p in out.iterdir() if p.is_file()} - {"complete.json", "identity.json"}
    if present != set(record["files"]):
        raise ValueError(f"{out} files differ from its completion record: {sorted(present ^ set(record['files']))[:3]}")
    for name, expected in record["files"].items():
        if sha256_file(out / name) != expected:
            raise ValueError(f"{out / name} changed since the run completed")
    for entry in record.get("embeddings", {}).values():
        if sha256_file(resolved(entry["path"])) != entry["sha256"]:
            raise ValueError(f"{resolved(entry['path'])} changed since the run completed")
    return record


def finalize(out: Path, embeddings_dir: Path, plan: pd.DataFrame, identity: dict, bank: ModelBank, checker, n_models: int,
             n_seeds: int, arrays: dict, started: float) -> dict:
    """Assemble and validate the table, run the spot check (`checker(table)`), hash every output and write
    complete.json; arrays outside `out` are named in `embeddings` with path and sha256."""
    parts = []
    for number in sorted(plan.chunk.unique()):
        parts.append(pd.read_parquet(chunk_files(out, int(number))[0]))
    table = pd.concat(parts, ignore_index=True)
    validate_table(table, plan, n_models, n_seeds, bank.layout.encoders)
    for array in arrays.values():
        array.flush()
    arrays.clear()
    table.to_parquet(out / TABLE_FILE, index=False)
    check = checker(table)
    files = output_hashes(out)
    paths = {name: embeddings_dir / f"embeddings_{name}.npy" for name in bank.arrays}
    seeds = {array_name(e, s): s for e in bank.layout.g_encoders for s in bank.embed_seeds}    # none for the frozen encoder
    elsewhere = {} if same_path(embeddings_dir, out) else {"embeddings": {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path), "encoder": bank.arrays[name],
               "seed": seeds.get(name), "shape": list(np.load(path, mmap_mode="r").shape), "dtype": "float16"}
        for name, path in paths.items()}}
    scored = table[table.encoder.eq("g_combo")]
    record = {"identity": identity, "files": files, **elsewhere, "spot_check": check,
              "summary": {"rows": len(table), "shots": int(scored.shot_id.nunique()),
                          "frames_scored": int(scored.frame_status.isin(["ok", "anchor_off_tolerance"]).sum()),
                          "rows_without_frame_per_encoder": int(scored.frame_status.isin(["missing", "ambiguous", "decode_failed"]).sum()),
                          "frame_status": scored.frame_status.value_counts().to_dict(),
                          "complete_case_shots": int(scored.drop_duplicates("shot_id").complete_case.sum()),
                          "embedding_dims": {e: bank.dims[e] for e in bank.layout.encoders},
                          "embedding_rows": {n: int(np.load(p, mmap_mode="r").shape[0]) for n, p in paths.items()},
                          "embedding_bytes": {n: p.stat().st_size for n, p in paths.items()},
                          "seconds_this_run": round(time.time() - started, 1),
                          "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None},
              "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_atomic(out / "complete.json", json_text(record).encode())
    shutil.rmtree(out / "chunks")
    (out / "identity.json").unlink()
    return record


# --- main ----------------------------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="xgc", help="xgc: the direction probe; xgt: xg-temporal 01b")
    ap.add_argument("--out", type=Path, help=f"default {OUT.relative_to(ROOT)} (xgc), {XGT_OUT.relative_to(ROOT)} (xgt)")
    ap.add_argument("--embeddings-dir", type=Path,
                    help=f"xgt only: where the arrays go, default {XGT_EMBEDDINGS.relative_to(ROOT)} (xgc: always --out)")
    ap.add_argument("--embed-seeds", type=int, nargs="+", default=[0],
                    help="xgt only: seeds whose G embedding is kept (0 must be one; seed k > 0 goes to embeddings_<encoder>_seed<k>)")
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--offset-frames", type=Path, default=OFFSET_FRAMES)
    ap.add_argument("--g-combo", type=Path, help=f"default {G_ROOTS['g_combo'].relative_to(ROOT)} (xgc), "
                                                 f"{XGT_G_COMBO.relative_to(ROOT)} (xgt)")
    ap.add_argument("--head-free", type=Path, default=G_ROOTS["g_combo_no_head"], help="xgc only")
    ap.add_argument("--folds", type=Path, help=f"default {FOLDS_FILE} (xgc), {a1.FOLDS.relative_to(ROOT)} (xgt)")
    ap.add_argument("--overlay", type=Path, default=ROOT / OVERLAY)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--reference", type=Path, default=XGT_REFERENCE, help="xgt: the out-of-fold table with z_g_combo")
    ap.add_argument("--expect-models", type=int, default=10, help="fold models per G encoder (ten in the real run)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-shots", type=int, default=64, help="shots per chunk, the unit of resume")
    ap.add_argument("--workers", type=int, default=6, help="image decoding workers")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--gpu-approved", action="store_true", help="owner approval to use the GPU; required on a GPU")
    ap.add_argument("--shots-csv", type=Path, help=f"xgc smoke runs: score only the shots listed in the shot_id column; "
                                                   f"xgt: the shot list, default {XGT_SHOTS.relative_to(ROOT)}")
    ap.add_argument("--offsets", type=int, nargs="*", help="xgc smoke runs: score only these grid offsets; "
                                                           f"xgt: default {list(ti.GRID)}")
    ap.add_argument("--control-shots", type=int, default=120, help="shots of the JPEG control of the spot check")
    ap.add_argument("--max-chunks", type=int, help="stop cleanly after this many new chunks (resume by rerunning)")
    ap.add_argument("--check-only", action="store_true", help="verify the inputs, build the plan, print sizes; score nothing")
    ap.add_argument("--spot-check", action="store_true", help="re-run the offset-0 check on a finished table; score nothing")
    args = ap.parse_args(argv)
    xgt = args.layout == "xgt"
    args.out = args.out or (XGT_OUT if xgt else OUT)
    args.embeddings_dir = args.embeddings_dir or (XGT_EMBEDDINGS if xgt else args.out)
    args.g_combo = args.g_combo or (XGT_G_COMBO if xgt else G_ROOTS["g_combo"])
    args.folds = args.folds or (a1.FOLDS if xgt else ROOT / FOLDS_FILE)
    args.embed_seeds = tuple(sorted(set(args.embed_seeds)))
    if 0 not in args.embed_seeds:
        ap.error("--embed-seeds must include 0 (the embedding of the protocol and of the g_xgv contract)")
    if not xgt and (args.embed_seeds != (0,) or not same_path(args.embeddings_dir, args.out)):
        ap.error("--embed-seeds and --embeddings-dir are for layout xgt: the xgc identity records neither")
    if xgt:
        args.shots_csv = args.shots_csv or XGT_SHOTS
        args.offsets = args.offsets or list(ti.GRID)
    return args


def same_path(a: Path, b: Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def check_destination(args: argparse.Namespace, grid: tuple[int, ...]) -> bool:
    """Whether the run is the full one of its layout; a subset must not go to a default output. Layout xgt: full is
    the 01b extraction list (checked against its completion record) at the 11 offsets of the temporal grid."""
    if args.layout == "xgc":
        if (args.shots_csv or args.offsets) and same_path(args.out, OUT):
            raise SystemExit("a smoke subset (--shots-csv / --offsets) must not be written to the default --out; pass another --out")
        return not args.shots_csv and grid == GRID_MS
    full = same_path(args.shots_csv, XGT_SHOTS) and grid == ti.GRID
    if full:
        records.verified_file(XGT_SHOTS)
    elif same_path(args.out, XGT_OUT) or same_path(args.embeddings_dir, XGT_EMBEDDINGS):
        raise SystemExit("a subset (another --shots-csv or --offsets) must not be written to the default --out or "
                         "--embeddings-dir of layout xgt; pass others")
    return full


def plan_summary(plan: pd.DataFrame, bank: ModelBank, identity: dict, n_seeds: int) -> dict:
    """What a run would score (printed by --check-only)."""
    frames = plan[plan.has_frame]
    sizes = {name: array_rows(plan, e) * bank.dims[e] * 2 for name, e in bank.arrays.items()}
    return {"layout": bank.layout.name, "encoders": list(bank.layout.encoders), "shots": identity["plan"]["shots"],
            "matches": int(plan.game.nunique()), "offsets_ms": identity["plan"]["offsets_ms"], "plan_rows": len(plan),
            "frames_to_score": len(frames), "frames_ok_uncut": int((frames.frame_status.eq("ok") & frames.cut_between_anchor.eq(False)).sum()),
            "frame_status": {str(k): int(v) for k, v in plan.frame_status.value_counts().items()},
            "official_split": {str(k): int(v) for k, v in plan.drop_duplicates("shot_id").split.value_counts().items()},
            "shots_per_group": {str(k): int(v) for k, v in plan.drop_duplicates("shot_id").group.value_counts().sort_index().items()},
            "chunks": int(plan.chunk.nunique()), "trunk_passes": len(frames),
            "tail_passes": len(frames) * len(bank.layout.g_encoders) * n_seeds + (len(frames) if bank.layout.frozen else 0),
            "embedding_arrays_gib": {n: round(b / 2**30, 3) for n, b in sizes.items()},
            "bundles": {e: [{"fold": m["fold"], "seed": m["seed"], "sha256": m["sha256"]} for m in ms]
                        for e, ms in identity["bundles"].items()},
            "folds_sha256": identity["folds_sha256"], "index_sha256": identity["index_sha256"], "full": identity["plan"]["full"],
            "plan_digest": identity["plan"]["digest"], "shot_ids_sha256": identity["plan"]["shot_ids_sha256"],
            **({"xgt": identity["xgt"]} if "xgt" in identity else {})}


def run(args: argparse.Namespace) -> int:
    started = time.time()
    layout = LAYOUTS[args.layout]
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.check_only or args.spot_check:
        device = "cpu"    # nothing is scored on a GPU unless the run is approved
    args.device_type = device
    grid = tuple(args.offsets) if args.offsets else GRID_MS
    full = check_destination(args, grid)
    if args.spot_check:
        recorded = json.loads((args.out / "complete.json").read_text())["spot_check"]
        table = pd.read_parquet(args.out / TABLE_FILE)
        if layout.name == "xgt":
            again = {"offset0_vs_z_g_combo": reference_check(table, xgt_reference(args.reference)[0]),
                     "roles": xgt_table_check(table, args.index, args.folds)}
            passed = recorded["pipeline_passed"] and again["roles"]["passed"]
        else:
            again = {"offset0_vs_stored": spot_check(table, {"g_combo": args.g_combo, "g_combo_no_head": args.head_free})}
            passed = recorded["pipeline_passed"]
        print(json_text({"recomputed": again, "recorded_at_completion": recorded}))
        return 0 if passed else 1

    shots, frames, offset_provenance = load_offset_frames(args.offset_frames, args.protocol, args.index)
    folds = pd.read_csv(args.folds)
    roots = {"g_combo": args.g_combo, "g_combo_no_head": args.head_free}
    models = {e: verify_bundles(roots[e], e, args.index, args.folds, args.overlay) for e in layout.g_encoders}
    spec, backbone, n_seeds = check_layout(models, folds, args.expect_models)
    if not set(args.embed_seeds) <= {m["seed"] for m in models["g_combo"]}:
        raise ValueError(f"--embed-seeds {list(args.embed_seeds)} are not seeds of the fold models")
    if args.shots_csv:
        wanted = set(pd.read_csv(args.shots_csv).shot_id)
        if unknown := wanted - set(shots.shot_id):
            raise ValueError(f"{len(unknown)} requested shots are not in the offset frames")
        shots = shots[shots.shot_id.isin(wanted)]
    if not set(grid) <= set(GRID_MS):
        raise ValueError("offsets must be on the approved grid")
    xgt, reference = None, None
    if layout.name == "xgt":
        reference, reference_record = xgt_reference(args.reference)
        xgt = {**xgt_roles(args.index, args.folds, shots, models["g_combo"]),
               "shots_csv": {"path": ti.recorded_path(args.shots_csv), "sha256": sha256_file(args.shots_csv)},
               "reference": reference_record, "embedding_seeds": list(args.embed_seeds),
               "embeddings_dir": str(args.embeddings_dir.resolve()), "encoders": list(layout.encoders)}
    plan = build_plan(shots, frames, folds, args.offset_frames, args.chunk_shots, grid, test_group=layout.test_group)
    sources = {**own_sources(), "scripts/score_offset_frames.py": sha256_file(Path(__file__))}
    if absent := [m for m in ("xgpoc/probe/scoring.py", "xgpoc/probe/table.py") if m not in sources]:
        raise ValueError(f"source hashes lack {absent}; run from the experiment directory (./run.sh)")
    identity = requested_identity(args, offset_provenance, models, spec, backbone, plan, full, sources, xgt)
    if (args.out / "complete.json").exists() and not args.check_only:
        record = verify_complete(args.out, identity)
        print("verified existing output; nothing rescored")
        print(json_text(record["spot_check"]))
        return 0 if record["spot_check"]["pipeline_passed"] else 1
    check_frame_files(plan)
    if device == "cuda":
        if not args.gpu_approved:
            raise SystemExit("refusing to use the GPU without --gpu-approved (owner approval)")
        check_gpu_free()
    bank = ModelBank(models, spec, device, layout, args.embed_seeds)
    sizes = {name: array_rows(plan, e) * bank.dims[e] * 2 for name, e in bank.arrays.items()}
    print(f"{identity['plan']['shots']} shots, {len(plan)} offset rows x {len(layout.encoders)} encoders, {plan.chunk.nunique()} "
          f"chunks, {int(plan.has_frame.sum())} frames, embeddings {sum(sizes.values()) / 2**30:.2f} GiB "
          f"({', '.join(f'{e} {s / 2**30:.2f}' for e, s in sizes.items())}), device {device}", flush=True)
    if args.check_only:
        print(json_text(plan_summary(plan, bank, identity, n_seeds)))
        return 0

    out, embeddings_dir = args.out, args.embeddings_dir
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"{out} holds a different run than requested (identity changed); use a new --out")
    elif any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no run identity")
    elif not same_path(embeddings_dir, out) and embeddings_dir.exists() and any(embeddings_dir.iterdir()):
        raise ValueError(f"{embeddings_dir} is not empty and {out} has no run identity")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    arrays_needed = sum(sizes.values())
    if shutil.disk_usage(embeddings_dir).free - arrays_needed < MIN_FREE_BYTES and not marker.exists():
        raise OSError(f"embeddings need {arrays_needed / 2**30:.1f} GiB, which would leave under 5 GiB free; not started")
    write_atomic(marker, json_text(identity).encode())
    arrays = open_embeddings(embeddings_dir, plan, bank.dims, bank.arrays)

    chunks = {int(n): rows for n, rows in plan.groupby("chunk", sort=True)}
    todo = [n for n, rows in chunks.items() if not chunk_is_committed(out, n, rows, arrays)]
    print(f"{len(chunks) - len(todo)} of {len(chunks)} chunks already committed", flush=True)
    done = 0
    for number in todo:
        if args.max_chunks is not None and done >= args.max_chunks:
            print(f"stopping after {done} chunks as asked; rerun to resume")
            return 0
        if shutil.disk_usage(embeddings_dir).free < MIN_FREE_BYTES:
            raise OSError("free disk fell under 5 GiB; stopped, rerun to resume")
        if device == "cuda":
            check_gpu_free()
        rows = chunks[number]
        began = time.time()
        result = score_chunk(rows, bank, args.batch_size, args.workers)
        commit_chunk(out, number, rows, result, arrays)
        done += 1
        remaining = len(todo) - done
        print(f"chunk {number} ({rows.group.iloc[0]}, {rows.shot_id.nunique()} shots, {int(rows.has_frame.sum())} frames) "
              f"{time.time() - began:.0f}s; {remaining} left ({time.strftime('%H:%M')})", flush=True)
    if layout.name == "xgt":
        def checker(table: pd.DataFrame) -> dict:
            return xgt_spot_check(table, plan, bank, args, reference)
    else:
        def checker(table: pd.DataFrame) -> dict:
            return full_spot_check(table, plan, bank, args, {"g_combo": args.g_combo, "g_combo_no_head": args.head_free})
    record = finalize(out, embeddings_dir, plan, identity, bank, checker, args.expect_models, n_seeds, arrays, started)
    print(json_text(record["spot_check"]))
    print(f"OFFSET SCORES DONE ({time.strftime('%H:%M')}): {record['summary']}")
    if not record["spot_check"]["pipeline_passed"]:
        print("SPOT CHECK CONTROL FAILED: the scoring pipeline does not reproduce the reference logits on the index frames")
    return 0 if record["spot_check"]["pipeline_passed"] else 1


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
