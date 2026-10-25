"""The per-shot prediction table of the direction probe (xg-credibility ticket 08, test seam 1).

One row per shot, offset and encoder. Every downstream analysis reads only this table and the index, so the
meaning of each column is fixed here.

    shot_id, game, half, split   the shot, its match, the half (period) and the official split, from the index
    offset_ms                    nominal offset of the frame on the approved grid, relative to the shot anchor
    frame_status                 ok | anchor_off_tolerance | missing | ambiguous | decode_failed (offset-frames
                                 version); rows without a frame stay in the table with NaN logits and ref -1
    decoded_relative_ms          decoded time of the frame relative to the anchor (NaN without a frame)
    cut_between_anchor           camera-cut flag of the frame (<NA> without a frame)
    complete_case                the shot has an `ok` frame at all 30 offsets (the probe's inclusion rule)
    encoder                      g_combo | g_combo_no_head | dinov2_frozen
    xg_logit                     G encoders: mean over the scoring models of the temperature-scaled xG logit
                                 (the input of the ensemble's Platt step); NaN for dinov2_frozen
    xg_logit_raw                 the same with the raw (uncalibrated) logits
    on_target_logit              g_combo only: mean raw on-target logit; NaN otherwise
    oof                          every scoring model was trained without this shot's match (always true here, by
                                 construction; `validate_table` checks that `fold_source` is the fold of the shot's
                                 match in the folds file, or -1 for test shots, which are in no fold; dinov2_frozen
                                 is not trained at all). Each fold model's epoch and temperature were still
                                 selected on its held-out fold, equally at every offset; prefer `xg_logit_raw`
                                 where the scale cancels after centring
    fold_source                  fold of the scoring models for train/valid shots; -1 for test shots (all folds)
                                 and for dinov2_frozen
    seed_source                  seeds of the scoring models, e.g. "0,1"; empty for dinov2_frozen
    n_models                     number of models averaged (0 for dinov2_frozen)
    embedding_ref                row of the pooled embedding in `embeddings_<encoder>.npy` (float16), -1 if none.
                                 g_combo / g_combo_no_head: 5120 features of the seed-0 model of the shot's fold,
                                 train/valid shots only (the protocol's Gp and G0p read nothing else);
                                 dinov2_frozen: 2048 features (CLS + mean patch token), all shots with a frame.

The xg-temporal layout of `scripts/score_offset_frames.py` (`--layout xgt`, ticket 01b) writes the g_combo rows only
(`encoders` of `chunk_table` / `validate_table`) and has no test group (`build_plan(test_group=False)`): every shot is
in a fold of the xgv folds file, whatever its official `split`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.data.offset_frames import GRID_MS

ENCODERS = ("g_combo", "g_combo_no_head", "dinov2_frozen")
G_ENCODERS = ENCODERS[:2]
TEST_GROUP = "test"
TEST_RANK = 1_000_000  # after every fold
FRAME_STATUSES_WITH_FRAME = ("ok", "anchor_off_tolerance")
TABLE_COLUMNS = ["shot_id", "game", "half", "split", "offset_ms", "frame_status", "decoded_relative_ms",
                 "cut_between_anchor", "complete_case", "encoder", "xg_logit", "xg_logit_raw", "on_target_logit",
                 "oof", "fold_source", "seed_source", "n_models", "embedding_ref"]
PLAN_COLUMNS = ["shot_id", "game", "half", "split", "offset_ms", "frame_status", "decoded_relative_ms", "cut_between_anchor",
                "complete_case", "group", "chunk", "has_frame", "frame_file", "file_sha256", "file_bytes", "emb_row_g",
                "emb_row_frozen"]
SPOT_TOLERANCE = 1e-3  # ticket 08: the batch-composition noise bound of the frozen G


def build_plan(shots: pd.DataFrame, frames: pd.DataFrame, folds: pd.DataFrame, frames_root: Path, chunk_shots: int,
               grid: tuple[int, ...] = GRID_MS, test_group: bool = True) -> pd.DataFrame:
    """Everything to score, one row per non-penalty shot and offset, in processing order.

    Goals are scored too (the attribution-sensitivity rows of ticket 10 need shots that are goals under the main
    rule); penalties are not (protocol). `shots` is the offset-frames `shots.csv`, `frames` its `frames.csv`, `folds`
    the home-team fold file (game, fold). Rows are ordered fold 0, fold 1, ..., test; shots by id; offsets by `grid`.
    Chunks hold at most `chunk_shots` shots of one group, so all frames of a chunk need the same models. Embedding rows
    are assigned here, from the plan alone, so that they do not depend on how far a run got.

    `test_group` False (the xgv folds of xg-temporal, which cover matches of every official split): every shot's game
    must be in `folds` and every shot is scored by its fold's models; the official `split` is then only a column."""
    shots = shots[~shots.is_penalty.astype(bool)]
    if shots.empty:
        raise ValueError("no non-penalty shots to score")
    if folds.game.duplicated().any():
        raise ValueError("a game has two folds in the folds file")
    fold_of = folds.set_index("game").fold
    is_test = shots.split.eq("test") if test_group else pd.Series(False, index=shots.index)
    if shots[is_test].game.isin(fold_of.index).any():
        raise ValueError("a test game is in the folds file")
    missing = sorted(set(shots[~is_test].game) - set(fold_of.index))
    if missing:
        raise ValueError(f"{len(missing)} {'train/valid ' if test_group else ''}games have no fold, e.g. {missing[0]}")
    fold = shots.game.map(fold_of)
    shots = shots.assign(group=np.where(is_test, TEST_GROUP, "fold" + fold.fillna(-1).astype(int).astype(str)),
                         group_rank=np.where(is_test, TEST_RANK, fold.fillna(-1)).astype(int))
    shots = shots.sort_values(["group_rank", "shot_id"]).reset_index(drop=True)
    shots["chunk_in_group"] = shots.groupby("group").cumcount() // chunk_shots
    shots["chunk"] = shots.groupby(["group_rank", "chunk_in_group"], sort=True).ngroup()

    wanted = frames[frames.shot_id.isin(shots.shot_id) & frames.offset_ms.isin(grid)]
    if wanted.groupby("shot_id").size().ne(len(grid)).any() or wanted.shot_id.nunique() != len(shots) or wanted.duplicated(
            ["shot_id", "offset_ms"]).any():
        raise ValueError("the offset frames do not hold every grid offset for every shot exactly once")
    rows = shots[["shot_id", "game", "half", "split", "group", "chunk", "complete_case", "group_rank"]].merge(
        wanted[["shot_id", "offset_ms", "status", "decoded_relative_ms", "cut_between_anchor", "frame_path", "file_sha256",
                "file_bytes"]],
        on="shot_id", validate="one_to_many")
    rows["offset_rank"] = rows.offset_ms.map({o: i for i, o in enumerate(grid)})
    rows = rows.sort_values(["group_rank", "shot_id", "offset_rank"]).reset_index(drop=True)
    rows = rows.rename(columns={"status": "frame_status"})
    rows["has_frame"] = rows.frame_status.isin(FRAME_STATUSES_WITH_FRAME) & rows.frame_path.notna()
    rows["frame_file"] = np.where(rows.has_frame, rows.frame_path.map(lambda p: str(frames_root / p) if isinstance(p, str) else ""), "")
    rows["emb_row_frozen"] = np.where(rows.has_frame, rows.has_frame.cumsum() - 1, -1)
    g_rows = rows.has_frame & rows.group.ne(TEST_GROUP)
    rows["emb_row_g"] = np.where(g_rows, g_rows.cumsum() - 1, -1)
    rows["cut_between_anchor"] = rows.cut_between_anchor.astype("boolean")
    rows["complete_case"] = rows.complete_case.astype(bool)
    return rows[PLAN_COLUMNS]


def chunk_table(rows: pd.DataFrame, values: dict[str, dict], encoders: tuple[str, ...] = ENCODERS) -> pd.DataFrame:
    """Table rows of one chunk. `rows`: its plan rows. `values[encoder]`: arrays aligned to the rows that have a frame
    (`xg_logit`, `xg_logit_raw`, optional `on_target_logit`) plus `fold_source`, `seed_source`, `n_models`."""
    parts = []
    for encoder in encoders:
        v = values[encoder]
        has = rows.has_frame.to_numpy()
        out = rows[["shot_id", "game", "half", "split", "offset_ms", "frame_status", "decoded_relative_ms",
                    "cut_between_anchor", "complete_case"]].copy()
        out["encoder"] = encoder
        for column in ("xg_logit", "xg_logit_raw", "on_target_logit"):
            full = np.full(len(rows), np.nan, np.float32)
            if v.get(column) is not None:
                full[has] = v[column]
            out[column] = full
        out["oof"] = True
        out["fold_source"] = np.int8(v["fold_source"])
        out["seed_source"] = v["seed_source"]
        out["n_models"] = np.int8(v["n_models"])
        out["embedding_ref"] = (rows.emb_row_frozen if encoder == "dinov2_frozen" else rows.emb_row_g).astype(np.int32).to_numpy()
        parts.append(out)
    table = pd.concat(parts, ignore_index=True)
    table["order"] = np.tile(np.arange(len(rows)), len(encoders))
    table = table.sort_values("order", kind="stable").drop(columns="order").reset_index(drop=True)   # row, then encoder order
    table["half"] = table.half.astype(np.int8)
    table["offset_ms"] = table.offset_ms.astype(np.int16)
    return table[TABLE_COLUMNS]


def validate_table(table: pd.DataFrame, plan: pd.DataFrame, n_models: int, n_seeds: int,
                   encoders: tuple[str, ...] = ENCODERS) -> None:
    """Raise unless `table` is exactly what `plan` asks for. `n_models`: models that score a test shot."""
    if list(table.columns) != TABLE_COLUMNS:
        raise ValueError(f"table columns are {list(table.columns)}")
    if len(table) != len(plan) * len(encoders):
        raise ValueError(f"table has {len(table)} rows, the plan asks for {len(plan) * len(encoders)}")
    if set(table.encoder) != set(encoders):
        raise ValueError(f"table encoders are {sorted(set(table.encoder))}, expected {sorted(encoders)}")
    if table.duplicated(["shot_id", "offset_ms", "encoder"]).any():
        raise ValueError("duplicate shot/offset/encoder rows")
    keep = ["shot_id", "game", "half", "split", "offset_ms", "frame_status", "complete_case"]
    plan_view = plan[keep].astype({"offset_ms": int, "half": int}).reset_index(drop=True)
    for encoder in encoders:
        part = table[table.encoder.eq(encoder)]
        if not plan_view.equals(part[keep].astype({"offset_ms": int, "half": int}).reset_index(drop=True)):
            raise ValueError(f"{encoder}: table rows do not follow the plan")
    test = plan.group.eq(TEST_GROUP).to_numpy()
    for encoder, part in table.groupby("encoder"):
        has = part.frame_status.isin(FRAME_STATUSES_WITH_FRAME).to_numpy()
        columns = {"g_combo": ("xg_logit", "xg_logit_raw", "on_target_logit"),
                   "g_combo_no_head": ("xg_logit", "xg_logit_raw"), "dinov2_frozen": ()}[encoder]
        for column in ("xg_logit", "xg_logit_raw", "on_target_logit"):
            finite = np.isfinite(part[column].to_numpy())
            if column in columns and not np.array_equal(finite, has):
                raise ValueError(f"{encoder}: {column} must be finite exactly where a frame was scored")
            if column not in columns and finite.any():
                raise ValueError(f"{encoder}: {column} must be empty")
        refs = part.embedding_ref.to_numpy()
        kept = np.sort(refs[refs >= 0])
        if not np.array_equal(kept, np.arange(len(kept))):
            raise ValueError(f"{encoder}: embedding rows are not 0..n-1 without gaps or repeats")
        if encoder == "dinov2_frozen":
            if not np.array_equal(refs >= 0, has):
                raise ValueError("dinov2_frozen: every scored frame needs an embedding")
            if (part.fold_source != -1).any() or (part.n_models != 0).any() or (part.seed_source != "").any():
                raise ValueError("dinov2_frozen has no fold, seed or model count")
        else:
            if (refs[test] != -1).any() or not np.array_equal(refs[~test] >= 0, has[~test]):
                raise ValueError(f"{encoder}: embeddings belong to scored train/valid frames only")
            if (part.fold_source[test] != -1).any() or (part.n_models[test] != n_models).any():
                raise ValueError(f"{encoder}: test shots are scored by all {n_models} models")
            if (part.fold_source[~test] < 0).any() or (part.n_models[~test] != n_seeds).any():
                raise ValueError(f"{encoder}: train/valid shots are scored by the {n_seeds} models of their fold")
    if not table.oof.all():
        raise ValueError("a row is not out-of-fold")
    expected_fold = plan.group.map(lambda g: -1 if g == TEST_GROUP else int(g.removeprefix("fold"))).to_numpy()
    for encoder in (e for e in encoders if e in G_ENCODERS):
        part = table[table.encoder.eq(encoder)]
        if not np.array_equal(part.fold_source.to_numpy(), expected_fold):
            raise ValueError(f"{encoder}: a shot is scored by the models of another fold than the one that held out its match")


def stored_reference(root: Path) -> tuple[pd.Series, pd.Series, str]:
    """Out-of-fold mean raw logit per shot from the per-model prediction files written at training time (tickets 04
    and 05): the held-out fold's `valid` file of each model for train/valid shots, every model's `test` file for test
    shots. Returns (pool, test, model name)."""
    runs = sorted(root.glob("fold*/seed*/complete.json"))
    if not runs:
        raise ValueError(f"no finished runs under {root}")
    name = json.loads(runs[0].read_text())["identity"]["name"]
    pool, test = [], []
    for record in runs:
        for role, bucket in (("valid", pool), ("test", test)):
            frame = pd.read_csv(record.parent / f"{name}_{role}_predictions.csv").drop_duplicates("shot_id")
            bucket.append(frame.set_index("shot_id").logit)
    return pd.concat(pool).groupby(level=0).mean(), pd.concat(test).groupby(level=0).mean(), name


def spot_check(table: pd.DataFrame, roots: dict[str, Path], tolerance: float = SPOT_TOLERANCE) -> dict:
    """Offset-0 raw xG logits of the table against the stored training-time predictions of the same models (ticket
    08 spot check). Only `ok` offset-0 frames are compared; train/valid and test shots are reported separately.
    Passes when every encoder compares at least one shot and the largest |difference| is within `tolerance`."""
    report = {}
    for encoder, root in roots.items():
        pool_ref, test_ref, name = stored_reference(root)
        zero = table[table.encoder.eq(encoder) & table.offset_ms.eq(0) & table.frame_status.eq("ok")]
        entry = {"model_name": name, "tolerance": tolerance}
        for label, part, ref in (("train_valid", zero[zero.split.ne("test")], pool_ref), ("test", zero[zero.split.eq("test")], test_ref)):
            ref = ref.reindex(part.shot_id).to_numpy()
            diff = np.abs(part.xg_logit_raw.to_numpy(np.float64) - ref)
            usable = diff[np.isfinite(diff)]
            entry[label] = {"shots": int(len(part)), "max_abs_dlogit": float(usable.max()) if len(usable) else None,
                            "mean_abs_dlogit": float(usable.mean()) if len(usable) else None,
                            "missing_reference": int(np.isnan(ref).sum())}
        compared = [entry[k] for k in ("train_valid", "test") if entry[k]["shots"]]
        entry["passed"] = bool(compared and all(c["missing_reference"] == 0 and c["max_abs_dlogit"] is not None
                                                and c["max_abs_dlogit"] <= tolerance for c in compared))
        report[encoder] = entry
    report["passed"] = all(report[e]["passed"] for e in roots)
    return report


def reference_check(table: pd.DataFrame, reference: pd.Series, tolerance: float = SPOT_TOLERANCE,
                    encoder: str = "g_combo") -> dict:
    """Offset-0 `xg_logit` (mean of logit / temperature over the scoring models) of `encoder`'s `ok` rows against a
    per-shot reference logit on the same scale (`reference`: shot_id -> logit, e.g. `z_g_combo` of the out-of-fold
    table, the mean of logit(stored p) = logit / T of the models that held the shot out). Reports the mean and largest
    |difference| and the Pearson correlation; passes when every compared shot has a reference and the largest
    |difference| is within `tolerance`."""
    zero = table[table.encoder.eq(encoder) & table.offset_ms.eq(0) & table.frame_status.eq("ok")]
    ref = reference.reindex(zero.shot_id).to_numpy(np.float64)
    ours = zero.xg_logit.to_numpy(np.float64)
    usable = np.isfinite(ref) & np.isfinite(ours)
    diff = np.abs(ours[usable] - ref[usable])
    entry = {"encoder": encoder, "column": "xg_logit", "tolerance": tolerance, "shots": int(len(zero)),
             "missing_reference": int((~np.isfinite(ref)).sum()),
             "mean_abs_dlogit": float(diff.mean()) if len(diff) else None,
             "max_abs_dlogit": float(diff.max()) if len(diff) else None,
             "pearson_r": float(np.corrcoef(ours[usable], ref[usable])[0, 1]) if usable.sum() > 1 else None}
    entry["passed"] = bool(len(diff) and entry["missing_reference"] == 0 and entry["max_abs_dlogit"] <= tolerance)
    return entry
