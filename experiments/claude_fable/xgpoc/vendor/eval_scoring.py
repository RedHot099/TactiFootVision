"""Scoring of the primary evaluation shots and the gate shots by the retrained G-combo (xg-vendor ticket 11).

Pure assembly code; `scripts/vendor_score_eval.py` runs the GPU stages and calls it.

Frames. The frames are the ones the retrained models were trained and gated on: the columns `frame_path` (offset 0) and
`frame_path_-400` (offset -400 ms) of index v1.2, read through `g_bundle.g_dataset` (overlay mask, no augmentation).
The archived offset-frame PNGs are not used (their decode differs from the index frames by up to about 1.3 logits).
A shot whose frame at an offset is missing or unreadable keeps its row with a status for that offset; it is never replaced
by another frame, and that offset gets no score.

Models. Every fold model (five folds x two seeds) scores every shot with `g_bundle.g_outputs` in the training-time batch
size and in row order, so a replay in the same order reproduces the logits. Evaluation shots and gate shots are scored in
separate calls, the gate shots in the row order of the retrain (so that their logits are comparable to the stored
gate predictions of each fold model). The single ensemble calibration of `ensemble.json` (temperature-scaled logits
averaged, Platt, prior correction, penalty routing) turns the ten logits into one probability per shot and offset.

What is stored per shot (`scores.parquet`, evaluation and gate shots in one table, `role` primary or gate):
    xg_f<k>s<s>_{0,m400}   raw xG logit of fold model (k, s) at the offset
    ontarget_mean_{0,m400} mean raw on-target logit of the ten models
    raw_mean_{0,m400}      mean raw xG logit of the ten models (the evaluation table's `frame_raw_logit_*`)
    p_{0,m400}             calibrated ensemble probability (the table's `frame_logit_*` is its logit)
    penalty_routed         the shot was routed to the penalty rate (the index penalty flag)
    status_/sha256_<offset> frame status and file hash
The pooled embeddings of the fold 0 and fold 1 seed-0 models and the frozen DINOv2 embeddings of the evaluation shots at
offset 0 are separate float16 matrices of the shots with a scored offset-0 frame, in row order.
"""

import hashlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from xgpoc.eval.calibration import logit
from xgpoc.models.g_bundle import EVAL_BATCH, ensemble_probability, g_dataset, g_outputs
from xgpoc.probe.scoring import SharedScorer
from xgpoc.vendor import evaltable as et

OFFSETS = {"0": "frame_path", "m400": "frame_path_-400"}   # offset name -> index column
OFFSET_MS = {"0": 0, "m400": -400}
MEMBERS = tuple((fold, seed) for fold in range(5) for seed in (0, 1))
EMBEDDING_FOLDS = (0, 1)   # the pre-specified and the sensitivity model, seed 0
NO_PATH, MISSING, UNREADABLE = "no_frame_path", "file_missing", "undecodable"
PRIMARY, SECONDARY, GATE = "primary", "secondary", "gate"   # role of a row = which scoring call it belongs to
REPLAY_ROWS = 64   # a multiple of the batch size: the first batches of the original run, in the same composition
FRAME_SOURCE = "ensemble_10_xgv_g_combo"


def member_name(fold: int, seed: int) -> str:
    return f"f{fold}s{seed}"


# ------------------------------------------------------------------ frames
def _inspect(path: str) -> tuple[str, str]:
    if not isinstance(path, str) or not path:
        return NO_PATH, ""
    file = Path(path)
    if not file.is_file():
        return MISSING, ""
    data = file.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        return UNREADABLE, digest
    return et.OK, digest


def inspect_frames(paths: Sequence[str], threads: int = 8) -> pd.DataFrame:
    """Per path: `status` (ok, no_frame_path, file_missing, undecodable) and the sha256 of the file bytes ("" when absent)."""
    with ThreadPoolExecutor(threads) as pool:
        found = list(pool.map(_inspect, paths))
    return pd.DataFrame(found, columns=["status", "sha256"])


def frame_fingerprint(paths: Sequence[str], sha256: Sequence[str]) -> dict:
    """Content hash of every present frame (path + file hash, sorted), in the manner of the retrain's `frame_fingerprint`."""
    entries = sorted({f"{p}\t{h}" for p, h in zip(paths, sha256, strict=True) if h})
    return {"count": len(entries), "sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}


def shots_frame(index: pd.DataFrame, pairs: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    """Rows to score: the paired shots in the order of `pairs` (role primary), then the gate shots in the order of `gate`
    (role gate). `index` is the full index (shots without a frame file included); a paired shot absent from the index or a
    shot in both groups is an error. Columns: key, role, game, half, split, is_penalty, the two frame columns, then per
    offset `status_<name>` and `sha256_<name>`."""
    paired = pairs[pairs.paired]
    by_id = index.drop_duplicates("shot_id").set_index("shot_id")
    absent = sorted(set(paired.shot_id) - set(by_id.index))
    if absent:
        raise ValueError(f"{len(absent)} paired shots are not in the index, e.g. {absent[0]}")
    if overlap := sorted(set(paired.shot_id) & set(gate.shot_id)):
        raise ValueError(f"{len(overlap)} shots are both evaluation and gate shots, e.g. {overlap[0]}")
    columns = ["game", "half", "split", "is_penalty", *OFFSETS.values()]
    shots = pd.concat([by_id.loc[paired.shot_id, columns].assign(role=PRIMARY).rename_axis("key").reset_index(),
                       gate.set_index("shot_id")[columns].assign(role=GATE).rename_axis("key").reset_index()], ignore_index=True)
    shots["is_penalty"] = shots.is_penalty.astype(bool)
    for name, column in OFFSETS.items():
        found = inspect_frames(shots[column].tolist())
        shots[f"status_{name}"], shots[f"sha256_{name}"] = found.status.to_numpy(), found.sha256.to_numpy()
    return shots[["key", "role", *columns, *[f"{k}_{n}" for n in OFFSETS for k in ("status", "sha256")]]]


def shots_fingerprint(shots: pd.DataFrame) -> dict:
    return frame_fingerprint([p for column in OFFSETS.values() for p in shots[column]],
                             [h for name in OFFSETS for h in shots[f"sha256_{name}"]])


def frames_unchanged(shots: pd.DataFrame) -> bool:
    """True when every frame of `shots` has now the status and file hash recorded in it (re-reads all frames). A row
    without a path keeps the status recorded in it (the secondary population records why a frame was never cut)."""
    now = shots.copy()
    for name, column in OFFSETS.items():
        has_path = shots[column].map(bool).to_numpy()
        found = inspect_frames(shots.loc[has_path, column].tolist())
        now.loc[has_path, f"status_{name}"], now.loc[has_path, f"sha256_{name}"] = found.status.to_numpy(), found.sha256.to_numpy()
    return bool(now.equals(shots))


def scoring_roles(shots: pd.DataFrame) -> list[str]:
    """The roles present in `shots`, in scoring order; each role is scored in its own calls (batch composition)."""
    return [role for role in (PRIMARY, SECONDARY, GATE) if shots.role.eq(role).any()]


def embedding_rows(shots: pd.DataFrame) -> pd.DataFrame:
    """The rows that get embeddings: those of the one role that is not the gate, in table order. A table with two such
    roles is refused (their embeddings would overwrite each other)."""
    roles = [role for role in scoring_roles(shots) if role != GATE]
    if len(roles) > 1:
        raise ValueError(f"embeddings are written for one evaluation role per table, got {roles}")
    return shots[shots.role.ne(GATE)]


def scored_mask(shots: pd.DataFrame, name: str) -> np.ndarray:
    """Rows whose frame at offset `name` is present and readable."""
    return shots[f"status_{name}"].eq(et.OK).to_numpy()


def table_frame_status(status_0: Sequence[str], status_m400: Sequence[str]) -> list[str]:
    """"ok" when both frames were scored, else the failing offsets with their reason, e.g. "-400:file_missing"."""
    return [";".join(f"{OFFSET_MS[n]}:{s}" for n, s in (("0", s0), ("m400", s4)) if s != et.OK) or et.OK
            for s0, s4 in zip(status_0, status_m400, strict=True)]


# ------------------------------------------------------------------ the GPU calls
def unlabeled(rows: pd.DataFrame) -> pd.DataFrame:
    """`rows` for the image dataset, which wants a `goal` column for its training target. Scoring never reads it; the
    column holds a constant, so no label reaches this code."""
    return rows.reset_index(drop=True).assign(goal=False)


def score_rows(model: nn.Module, spec: dict, rows: pd.DataFrame, name: str, *, batch_size: int = EVAL_BATCH, device: str = "cuda",
               num_workers: int = 4, embedding: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(xG logit [N], on-target logit [N], pooled embedding [n scored, 5120] float16 or None) of one fold model at offset
    `name`. Rows without a scored frame (`status_<name>`) get NaN logits and no embedding row; the scored rows go through
    `g_outputs` in row order and in `batch_size`, the batch composition of the model's training-time evaluation."""
    ok = scored_mask(rows, name)
    xg, on_target = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    if not ok.any():
        return xg, on_target, None
    spec = {**spec, "input": {**spec["input"], "frame_col": OFFSETS[name]}}
    z, o, e = g_outputs(model, unlabeled(rows[ok]), spec, device=device, batch_size=batch_size, num_workers=num_workers)
    xg[ok], on_target[ok] = z, o
    return xg, on_target, e.astype(np.float16) if embedding else None


@torch.no_grad()
def frozen_embeddings(scorer: SharedScorer, spec: dict, rows: pd.DataFrame, name: str, *, batch_size: int = EVAL_BATCH,
                      num_workers: int = 4) -> np.ndarray:
    """Frozen DINOv2 embeddings (CLS + mean patch token, 2048 features, float16) of the shots with a scored frame at offset
    `name`, through the same dataset and transform as the fold models."""
    ok = scored_mask(rows, name)
    spec = {**spec, "input": {**spec["input"], "frame_col": OFFSETS[name]}}
    loader = DataLoader(g_dataset(unlabeled(rows[ok]), spec), batch_size, shuffle=False, num_workers=num_workers)
    chunks = [scorer.score(batch["image"], [])[1] for batch in loader]
    return np.concatenate(chunks).astype(np.float16) if chunks else np.zeros((0, 0), np.float16)


# ------------------------------------------------------------------ the ensemble
def ensemble_scores(xg: dict[tuple[int, int], np.ndarray], temperatures: dict[tuple[int, int], float], calibration: dict,
                    is_penalty: np.ndarray) -> dict[str, np.ndarray]:
    """For one offset: from the raw xG logit of every fold model (NaN where the frame was not scored) to the mean raw
    logit and the calibrated ensemble probability (`g_bundle.ensemble_probability`); NaN rows stay NaN."""
    keys = sorted(xg)
    raw = np.stack([np.asarray(xg[k], float) for k in keys])
    if (np.isnan(raw).any(axis=0) != np.isnan(raw).all(axis=0)).any():
        raise ValueError("some fold models scored a shot and others did not")
    ok = ~np.isnan(raw[0])
    p = np.full(raw.shape[1], np.nan)
    if ok.any():
        p[ok] = ensemble_probability([r[ok] for r in raw], [temperatures[k] for k in keys], calibration, np.asarray(is_penalty, bool)[ok])
    return {"raw_mean": raw.mean(axis=0), "p": p}


def assemble_scores(shots: pd.DataFrame, members: dict[tuple[int, int], dict[str, np.ndarray]], temperatures: dict[tuple[int, int], float],
                    calibration: dict) -> pd.DataFrame:
    """The `scores.parquet` table. `members[(fold, seed)]` holds `xg` and `on_target` arrays of shape [2 offsets, N]
    (offset order of `OFFSETS`)."""
    if sorted(members) != sorted(MEMBERS):
        raise ValueError(f"expected the ten fold models, got {sorted(members)}")
    out = pd.DataFrame({"key": shots.key.to_numpy(), "role": shots.role.to_numpy(), "game": shots.game.to_numpy(),
                        "penalty_routed": shots.is_penalty.to_numpy(bool)})
    for column in (f"{k}_{n}" for n in OFFSETS for k in ("status", "sha256")):
        out[column] = shots[column].to_numpy()
    for i, name in enumerate(OFFSETS):
        for k in MEMBERS:
            out[f"xg_{member_name(*k)}_{name}"] = members[k]["xg"][i]
        out[f"ontarget_mean_{name}"] = np.stack([members[k]["on_target"][i] for k in MEMBERS]).mean(axis=0)
        combined = ensemble_scores({k: members[k]["xg"][i] for k in MEMBERS}, temperatures, calibration, shots.is_penalty.to_numpy(bool))
        out[f"raw_mean_{name}"], out[f"p_{name}"] = combined["raw_mean"], combined["p"]
    return out


# ------------------------------------------------------------------ the evaluation table
def embedding_references(ok: np.ndarray) -> np.ndarray:
    """Row of each shot in a matrix that holds only the shots with `ok`, in order; -1 for the others."""
    refs = np.full(len(ok), -1, np.int32)
    refs[ok] = np.arange(int(ok.sum()), dtype=np.int32)
    return refs


def build_eval_table(shots: pd.DataFrame, scores: pd.DataFrame, pairs: pd.DataFrame, comparator: pd.DataFrame) -> pd.DataFrame:
    """The label-free evaluation table of the primary population (`et.LABEL_FREE_COLUMNS`, kind confirmatory).

    `shots`, `scores`: rows of `shots_frame` and `assemble_scores` (the primary rows are used); `pairs`: the frozen
    pairing's pair table; `comparator`: the frozen comparator's predictions (primary rows, joined on the StatsBomb event
    id). Raises ValueError when a primary shot has no comparator prediction or the comparator and the pairing disagree
    on which SoccerNet shot an event belongs to."""
    primary = shots.role.eq(PRIMARY).to_numpy()
    shots, scores = shots[primary].reset_index(drop=True), scores[primary].reset_index(drop=True)
    if not shots.key.equals(scores.key):
        raise ValueError("shots and scores are not in the same order")
    pair = pairs[pairs.paired].set_index("shot_id").loc[shots.key]
    rungs = comparator[comparator.population.eq(PRIMARY)].drop_duplicates("statsbomb_event_id").set_index("statsbomb_event_id")
    missing = sorted(set(pair.sb_event_id) - set(rungs.index))
    if missing:
        raise ValueError(f"{len(missing)} primary shots have no comparator prediction, e.g. {missing[0]}")
    rungs = rungs.loc[pair.sb_event_id]
    if (rungs.soccernet_shot_id.to_numpy() != shots.key.to_numpy()).any():
        raise ValueError("the comparator joins a StatsBomb event to another SoccerNet shot than the pairing")
    table = pd.DataFrame({
        "key": shots.key.to_numpy(), "soccernet_shot_id": shots.key.to_numpy(), "statsbomb_event_id": pair.sb_event_id.to_numpy(),
        "statsbomb_match_id": pd.array(pair.sb_match_id.to_numpy(), dtype="Int64"), "match": shots.game.to_numpy(),
        "half": shots.half.to_numpy(np.int8), "population": PRIMARY, "split": shots.split.to_numpy(),
        "sb_body_part": pair.sb_body_part.to_numpy(), "sb_shot_type": pair.sb_shot_type.to_numpy(),
        "pred_vendor_xg": pair.sb_xg.to_numpy(float), "pred_constant": rungs.p_constant.to_numpy(float),
        "pred_dist_angle": rungs.p_dist_angle.to_numpy(float), "pred_event": rungs.p_event.to_numpy(float),
        "pred_freeze_frame": rungs.p_freeze_frame.to_numpy(float),
        "frame_logit_0": logit(scores.p_0.to_numpy()), "frame_logit_m400": logit(scores.p_m400.to_numpy()),
        "frame_raw_logit_0": scores.raw_mean_0.to_numpy(float), "frame_raw_logit_m400": scores.raw_mean_m400.to_numpy(float),
        "frame_status": table_frame_status(scores.status_0, scores.status_m400), "frame_source": FRAME_SOURCE})
    refs = embedding_references(scored_mask(shots, "0"))   # the three embedding matrices hold the same shots
    table["emb_g_fold0"], table["emb_g_fold1"], table["emb_dinov2"] = refs, refs.copy(), refs.copy()
    table["timing_residual_ms"] = pair.residual_ms.to_numpy(float)
    table["anchor_convention"] = et.ANCHOR_CONVENTION[PRIMARY]
    table["table_kind"] = "confirmatory"
    return table[et.LABEL_FREE_COLUMNS]
