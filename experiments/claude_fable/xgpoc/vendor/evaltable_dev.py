"""The development evaluation table, built from the prototype's outputs (xg-vendor ticket 07).

DEVELOPMENT DATA ONLY. The primary population is the prototype's 2,708 paired non-penalty SoccerNet shots; every
G-combo number comes from the archived models, which saw most of these matches in training. Nothing here is a result.

What the development table holds, column by column (the sidecar repeats it):
    pred_*                the prototype's comparator rungs and the vendor xG (`ladder_shots.parquet`, as clipped there)
    frame_logit_0         logit of the prototype's calibrated G-combo probability: the stored 10-model test predictions
                          for test shots, the cross-fitted Platt of the shot's own fold models for train/valid shots
                          (`frame_source` says which)
    frame_logit_m400      not available: the archive has no calibrated logit at -400 ms (declared in `unavailable_columns`)
    frame_raw_logit_*     the archived offset table's uncalibrated mean logits at 0 and -400 ms. They come from the
                          PNG-decoded frames and differ from the stored predictions behind `frame_logit_0` by up to
                          about 1.3 logits (offset-scores spot check, `complete.json` of xgc_offset_scores)
    emb_g_fold0/fold1     rows of the archived `embeddings_g_combo.npy` (seed-0 model, offset 0). The archive stores
                          one embedding per train/valid shot, from the seed-0 model of the fold that held the shot's
                          match out: a shot has `emb_g_fold0` when that fold is 0, `emb_g_fold1` when it is 1, and
                          -1 (missing) in the other column. Test shots and folds 2-4 have no G embedding.
    emb_dinov2            rows of the archived `embeddings_dinov2_frozen.npy`, every shot
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from xgpoc.eval.calibration import EPS, PlattScaler, logit, sigmoid
from xgpoc.vendor import evaltable as et

OFFSETS = (0, -400)
ON_TARGET_LABEL = "Shots on target"
RUNGS = {"constant": "pred_constant", "dist_angle": "pred_dist_angle", "event": "pred_event", "freeze_frame": "pred_freeze_frame",
         "vendor": "pred_vendor_xg", "frame": None}   # prototype rung -> table column; the frame is sigmoid(frame_logit_0)
RECALIBRATED = {"frame_recal": "frame", "positions_recal": "freeze_frame", "vendor_recal": "vendor"}
METRICS = ("log_loss", "brier", "auc", "cal_slope")
NOTES = (
    "DEVELOPMENT table built from the prototype outputs; archived G-combo models saw most of these matches in training",
    "primary population only: the paired non-penalty SoccerNet shots of the prototype (index v1.2 labels)",
    "frame_logit_0: logit of the prototype's calibrated G-combo probability (stored test predictions; cross-fitted Platt of the "
    "shot's own fold models for train/valid shots), see frame_source",
    "frame_logit_m400 is unavailable: the archive has no calibrated logit at -400 ms",
    "frame_raw_logit_0/m400: archived offset-table mean raw logits from PNG-decoded frames; not the decode behind frame_logit_0",
    "emb_g_fold0/emb_g_fold1 point into the archived embeddings_g_combo.npy, which holds the seed-0 model of the fold that held "
    "the shot's match out: fold-0 references exist for fold-0 train/valid shots, fold-1 references for fold-1 shots, all other "
    "references and every test shot are -1 (missing)",
    "emb_dinov2 points into the archived embeddings_dinov2_frozen.npy for every shot",
)


def archive_view(archive: pd.DataFrame) -> pd.DataFrame:
    """One row per shot from the archived per-shot prediction table: raw G-combo logits and frame status at the two offsets,
    the fold of the shot's G models and the embedding rows at offset 0."""
    g = archive[archive.encoder.eq("g_combo") & archive.offset_ms.isin(OFFSETS)]
    wide = g.pivot(index="shot_id", columns="offset_ms", values=["xg_logit_raw", "frame_status"])
    out = pd.DataFrame({"raw_0": wide[("xg_logit_raw", 0)], "raw_m400": wide[("xg_logit_raw", -400)],
                        "status_0": wide[("frame_status", 0)], "status_m400": wide[("frame_status", -400)]})
    zero = g[g.offset_ms.eq(0)].set_index("shot_id")
    out["g_fold"], out["g_ref"] = zero.fold_source, zero.embedding_ref
    dino = archive[archive.encoder.eq("dinov2_frozen") & archive.offset_ms.eq(0)].set_index("shot_id")
    out["dino_ref"] = dino.embedding_ref
    return out


def frame_status(status_0: pd.Series, status_m400: pd.Series) -> list[str]:
    """"ok" when both frames were scored cleanly, else the offsets' statuses, e.g. "0:anchor_off_tolerance"."""
    missing = "no_archive_row"
    return [";".join(f"{offset}:{s}" for offset, s in ((0, s0), (-400, s400)) if s != et.OK) or et.OK
            for s0, s400 in zip(status_0.fillna(missing), status_m400.fillna(missing), strict=True)]


def build_development_tables(ladder: pd.DataFrame, pairs: pd.DataFrame, archive: pd.DataFrame, embedding_sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(label-free table, labels) of the development population. `ladder`: the prototype's `ladder_shots.parquet`;
    `pairs`: its `pairs.parquet`; `archive`: the archived per-shot prediction table (`xgpoc.probe.table`);
    `embedding_sets`: the sidecar description of the embedding files, which this function does not read."""
    extra = pairs.set_index("shot_id")[["sb_match_id", "sb_shot_type", "pen_v12", "sb_pen"]].reindex(ladder.shot_id)
    view = archive_view(archive).reindex(ladder.shot_id)
    n = len(ladder)
    table = pd.DataFrame({
        "key": ladder.shot_id.to_numpy(), "soccernet_shot_id": ladder.shot_id.to_numpy(), "statsbomb_event_id": ladder.sb_event_id.to_numpy(),
        "statsbomb_match_id": pd.array(extra.sb_match_id.to_numpy(), dtype="Int64"), "match": ladder.game.to_numpy(),
        "half": ladder.half.to_numpy(np.int8), "population": "primary", "split": ladder.split.to_numpy(),
        "sb_body_part": ladder.sb_body_part.to_numpy(), "sb_shot_type": extra.sb_shot_type.to_numpy(),
        "pred_vendor_xg": ladder.r_vendor.to_numpy(), "pred_constant": ladder.r_constant.to_numpy(),
        "pred_dist_angle": ladder.r_dist_angle.to_numpy(), "pred_event": ladder.r_event.to_numpy(),
        "pred_freeze_frame": ladder.r_freeze_frame.to_numpy(),
        "frame_logit_0": logit(ladder.r_frame), "frame_logit_m400": np.full(n, np.nan),
        "frame_raw_logit_0": view.raw_0.to_numpy(float), "frame_raw_logit_m400": view.raw_m400.to_numpy(float),
        "frame_status": frame_status(view.status_0, view.status_m400), "frame_source": ladder.src.to_numpy()})
    fold, ref = view.g_fold.to_numpy(), view.g_ref.fillna(-1).to_numpy()
    table["emb_g_fold0"] = np.where(fold == 0, ref, -1).astype(np.int32)
    table["emb_g_fold1"] = np.where(fold == 1, ref, -1).astype(np.int32)
    table["emb_dinov2"] = view.dino_ref.fillna(-1).to_numpy().astype(np.int32)
    table["timing_residual_ms"] = ladder.res_ms.to_numpy(float)
    table["anchor_convention"] = et.ANCHOR_CONVENTION["primary"]
    table["table_kind"] = "development"
    labels = pd.DataFrame({
        "key": ladder.shot_id.to_numpy(), "goal_sn": ladder.goal.to_numpy().astype(bool),
        "on_target_sn": pd.array(ladder.label.eq(ON_TARGET_LABEL).to_numpy(), dtype="boolean"),
        "sb_outcome": ladder.sb_outcome.to_numpy(), "is_penalty": (extra.pen_v12.astype(bool) | extra.sb_pen.astype(bool)).to_numpy()})
    return table[et.LABEL_FREE_COLUMNS], labels


# ------------------------------------------------------------------ reproduction of the prototype's ladder
def calibration_slope(y: np.ndarray, p: np.ndarray, iterations: int = 50) -> float:
    """Slope of the logistic recalibration y ~ a + b * logit(p) (Newton, as in the prototype)."""
    z = logit(p)
    x = np.column_stack([np.ones_like(z), z])
    beta = np.array([0.0, 1.0])
    for _ in range(iterations):
        q = sigmoid(x @ beta)
        step = np.linalg.solve(x.T @ (x * (q * (1 - q))[:, None]) + 1e-9 * np.eye(2), x.T @ (y - q))
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return float(beta[1])


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "brier": float(np.mean((p - y) ** 2)),
            "auc": float(roc_auc_score(y, p)), "cal_slope": calibration_slope(y, p)}


def crossfit_recalibration(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold probabilities of a Platt recalibration of `p`, five match-grouped folds (the prototype's single-input
    recalibration; the analysis tickets define the confirmatory one)."""
    z = logit(p)
    out = np.zeros(len(y))
    for fit, test in GroupKFold(5).split(z, y, groups):
        out[test] = PlattScaler().fit(z[fit], y[fit])(z[test])
    return out


def ladder_rungs(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """The prototype's single-input rungs as arrays over the rows of `table` (labels joined), recalibration cross-fitted per split group."""
    rungs = {name: (table[column].to_numpy(float) if column else sigmoid(table.frame_logit_0)).clip(EPS, 1 - EPS) for name, column in RUNGS.items()}
    group = np.where(table.split.eq("test"), "test", "trainvalid")
    y, matches = table.goal_sn.to_numpy(float), table.match.to_numpy()
    for name, base in RECALIBRATED.items():
        out = np.zeros(len(table))
        for g in ("test", "trainvalid"):
            rows = group == g
            out[rows] = crossfit_recalibration(y[rows], rungs[base][rows], matches[rows])
        rungs[name] = out
    return rungs


def prototype_comparison(table: pd.DataFrame, prototype_metrics: pd.DataFrame) -> pd.DataFrame:
    """Every shared (population, rung, metric) cell of the table's ladder and the prototype's `ladder_metrics.csv`, with the
    absolute difference. The populations are the prototype's: "test" (official test matches) and "trainvalid"."""
    rungs = ladder_rungs(table)
    group = np.where(table.split.eq("test"), "test", "trainvalid")
    y = table.goal_sn.to_numpy(float)
    rows = []
    for population in ("test", "trainvalid"):
        mask = group == population
        for name, p in rungs.items():
            ref = prototype_metrics[prototype_metrics.population.eq(population) & prototype_metrics.rung.eq("r_" + name)]
            if len(ref) != 1:
                raise ValueError(f"the prototype has no single row for {population}/{name}")
            mine = point_metrics(y[mask], p[mask])
            counts = {"n": int(mask.sum()), "goals": int(y[mask].sum()), "matches": int(table.match[mask].nunique())}
            for metric, value in {**counts, **mine}.items():
                theirs = float(ref.iloc[0][metric])
                rows.append({"population": population, "rung": name, "metric": metric, "table": value, "prototype": theirs,
                             "abs_diff": abs(value - theirs)})
    return pd.DataFrame(rows)
