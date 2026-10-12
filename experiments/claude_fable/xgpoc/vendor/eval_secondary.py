"""The secondary population in the evaluation table (xg-vendor ticket 13).

The secondary population is the 50 recovered goals and the 797 StatsBomb-only shots of the frozen pairing, with frames
cut by the re-download (`xgpoc.vendor.redownload`, data version `xgv_secondary_frames_v1`) at StatsBomb time plus the
half's frozen offset. They are scored by the same ten models, at offsets 0 and -400 ms, with the same pipeline as the
primary rows (`xgpoc.vendor.eval_scoring`), and appended to the primary table of ticket 11.

Rows. One row per target of the pairing, whether or not it has frames. A target whose frame was never cut (in this
run all 147 lie beyond the end of the half video) keeps its row with the re-download's reason as the frame status and NaN frame logits; the
analyses leave such rows out (`ladder.analysis_population`). The key is the StatsBomb event id and the anchor convention
is `statsbomb_time` (the frame time is StatsBomb's shot time plus the half offset; the re-download calls it
`statsbomb_time_plus_half_offset`, which `secondary_shots` requires).

Sources, by column:
    population, half, game   re-download frame index (recovered goal | secondary -> statsbomb_only); cross-checked with the
                             pairing's population table
    split                    the official split of the game in index v1.2
    sb_body_part, sb_shot_type, pred_vendor_xg
                             the pairing's population table `sb_unpaired.parquet` (StatsBomb xG unclipped, as for the
                             primary rows); `is_penalty` for the penalty routing is `sb_shot_type == "Penalty"` (on all 847 targets it equals the
                             pairing's StatsBomb `penalty` flag, which the label step uses)
    pred_constant ... pred_freeze_frame
                             the frozen comparator's predictions (ticket 06), by StatsBomb event id
    frame_*                  the ten-model ensemble of ticket 11's calibration
    emb_*                    rows of the combined embedding matrices, which hold the primary rows first (unchanged)

No label is read here: the outcome, goal and flag columns of the population table are never selected (`UNPAIRED_COLUMNS`).
"""

import numpy as np
import pandas as pd

from xgpoc.eval.calibration import logit
from xgpoc.vendor import eval_scoring as es
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import redownload as rd

POPULATION_OF_KIND = {"recovered_goal": "recovered_goal", "secondary": "statsbomb_only"}   # re-download kind -> table population
PAIRING_POPULATION = {"recovered_goal": "recovered_goal", "statsbomb_only": "secondary"}   # table population -> sb_unpaired population
UNPAIRED_COLUMNS = ["event_id", "match_id", "game", "period", "statsbomb_xg", "body_part", "shot_type", "population"]   # no label
FRAME_INDEX_COLUMNS = ["shot_id", "kind", "anchor_convention", "sb_event_id", "sb_match_id", "game", "half", "target_position_ms", "frame_path", "frame_path_-400",
                       "frame_status_0", "frame_status_-400", "frame_sha256_0", "frame_sha256_-400", "clip_status", "reason"]
PENALTY_SHOT_TYPE = "Penalty"
CUT_ANCHOR_CONVENTION = "statsbomb_time_plus_half_offset"   # what the re-download calls the frame time rule; the table calls it statsbomb_time
NO_FRAME_REASONS = (rd.BEFORE_START, rd.BEYOND_END, rd.EXTRACT_FAILED, rd.UNREADABLE, rd.HALF_REFUSED, rd.TARGET_UNUSABLE)   # the re-download's statuses


def secondary_shots(frames: pd.DataFrame, unpaired: pd.DataFrame, splits: dict[str, str]) -> pd.DataFrame:
    """The rows to score: one per target of `frames` (the re-download's frame index), in its order, role secondary.

    `unpaired` is the pairing's population table (only `UNPAIRED_COLUMNS` are read); `splits` maps a game to its official
    split. Columns: the ones `eval_scoring.shots_frame` makes (key, role, game, half, split, is_penalty, the two frame
    columns, `status_*` and `sha256_*` per offset) and, for the table and the audit, population, sb_match_id,
    sb_body_part, sb_shot_type, pred_vendor_xg, target_position_ms, clip_status, reason.

    A frame the index calls ok must exist, decode and have the recorded sha256; anything else raises ValueError (a
    missing or changed frame is investigated, never replaced). A target without frames gets no path and the
    re-download's status."""
    frames = frames[FRAME_INDEX_COLUMNS].reset_index(drop=True)
    if frames.sb_event_id.duplicated().any():
        raise ValueError("the frame index lists a StatsBomb event twice")
    if not frames.anchor_convention.eq(CUT_ANCHOR_CONVENTION).all():
        raise ValueError(f"the frame index is not cut at {CUT_ANCHOR_CONVENTION}")
    if not frames.kind.isin(POPULATION_OF_KIND).all():
        raise ValueError(f"unknown kinds in the frame index: {sorted(set(frames.kind) - set(POPULATION_OF_KIND))}")
    known = unpaired[UNPAIRED_COLUMNS].drop_duplicates("event_id").set_index("event_id")
    absent = sorted(set(frames.sb_event_id) - set(known.index))
    if absent:
        raise ValueError(f"{len(absent)} targets are not in the pairing's population table, e.g. {absent[0]}")
    rows = known.loc[frames.sb_event_id]
    population = frames.kind.map(POPULATION_OF_KIND)
    if (rows.population.to_numpy() != population.map(PAIRING_POPULATION).to_numpy()).any():
        raise ValueError("a target has another population in the pairing than in the frame index")
    if (rows.period.to_numpy() != frames.half.to_numpy()).any() or (rows.game.to_numpy() != frames.game.to_numpy()).any() \
            or (rows.match_id.to_numpy() != frames.sb_match_id.to_numpy()).any():
        raise ValueError("a target has another game, half or StatsBomb match in the pairing than in the frame index")
    no_split = sorted(set(frames.game) - set(splits))
    if no_split:
        raise ValueError(f"{len(no_split)} games have no official split, e.g. {no_split[0]}")

    shots = pd.DataFrame({"key": frames.sb_event_id.to_numpy(), "role": es.SECONDARY, "game": frames.game.to_numpy(),
                          "half": frames.half.to_numpy(), "split": frames.game.map(splits).to_numpy(),
                          "is_penalty": rows.shot_type.eq(PENALTY_SHOT_TYPE).to_numpy(bool)})
    for name, column in es.OFFSETS.items():
        suffix = es.OFFSET_MS[name]
        status, recorded = frames[f"frame_status_{suffix}"], frames[f"frame_sha256_{suffix}"]
        wanted = status.eq(et.OK).to_numpy()
        paths = frames[column].where(wanted, "")
        found = es.inspect_frames(paths[wanted].tolist())
        if (found.status != et.OK).any() or (found.sha256.to_numpy() != recorded[wanted].to_numpy()).any():
            raise ValueError(f"offset {name}: {int((found.status != et.OK).sum())} frames the index calls ok are missing or undecodable, "
                             f"{int((found.sha256.to_numpy() != recorded[wanted].to_numpy()).sum())} differ from the recorded sha256")
        other = sorted(set(status[~wanted]) - set(NO_FRAME_REASONS))
        if other:
            raise ValueError(f"offset {name}: unknown frame status {other}")
        sha = pd.Series("", index=frames.index)
        sha[wanted] = found.sha256.to_numpy()
        shots[column], shots[f"status_{name}"], shots[f"sha256_{name}"] = paths.to_numpy(), status.to_numpy(), sha.to_numpy()
    shots = shots[["key", "role", "game", "half", "split", "is_penalty", *es.OFFSETS.values(),
                   *[f"{k}_{n}" for n in es.OFFSETS for k in ("status", "sha256")]]]
    return shots.assign(population=population.to_numpy(), sb_match_id=frames.sb_match_id.to_numpy(), sb_body_part=rows.body_part.to_numpy(),
                        sb_shot_type=rows.shot_type.to_numpy(), pred_vendor_xg=rows.statsbomb_xg.to_numpy(float),
                        target_position_ms=frames.target_position_ms.to_numpy(), clip_status=frames.clip_status.to_numpy(),
                        reason=frames.reason.to_numpy())


def no_frame_report(shots: pd.DataFrame) -> pd.DataFrame:
    """The targets without a scored frame at one or both offsets, with the reason, for the record (never dropped)."""
    missing = ~(es.scored_mask(shots, "0") & es.scored_mask(shots, "m400"))
    return shots.loc[missing, ["key", "population", "game", "half", "target_position_ms", "status_0", "status_m400", "clip_status", "reason"]] \
        .rename(columns={"key": "statsbomb_event_id", "status_0": "frame_status_0", "status_m400": "frame_status_m400"}).reset_index(drop=True)


def frame_clip_disagreements(shots: pd.DataFrame) -> pd.DataFrame:
    """Targets whose frames and clip disagree: frames present and clip not ok, or no frame and a clip ok. Scoring uses the
    frames; the clips are an archive."""
    frames_ok = es.scored_mask(shots, "0") & es.scored_mask(shots, "m400")
    return shots.loc[frames_ok != shots.clip_status.eq(et.OK).to_numpy(), ["key", "population", "game", "half", "target_position_ms",
                                                                          "status_0", "status_m400", "clip_status"]].reset_index(drop=True)


def build_secondary_table(shots: pd.DataFrame, scores: pd.DataFrame, comparator: pd.DataFrame, first_embedding_row: int) -> pd.DataFrame:
    """The label-free rows of the secondary population (`et.LABEL_FREE_COLUMNS`, kind confirmatory).

    `shots`, `scores`: `secondary_shots` and `eval_scoring.assemble_scores` of the same rows; `comparator`: the frozen
    comparator's predictions (rows of the secondary populations, joined on the StatsBomb event id, population and
    StatsBomb match must agree); `first_embedding_row`: the row of the combined embedding matrices at which the secondary
    shots' embeddings start (the primary rows come first). Raises ValueError on a shot without comparator prediction."""
    if not shots.key.equals(scores.key):
        raise ValueError("shots and scores are not in the same order")
    rungs = comparator[comparator.population.ne("primary")].drop_duplicates("statsbomb_event_id").set_index("statsbomb_event_id")
    missing = sorted(set(shots.key) - set(rungs.index))
    if missing:
        raise ValueError(f"{len(missing)} secondary shots have no comparator prediction, e.g. {missing[0]}")
    rungs = rungs.loc[shots.key]
    if (rungs.population.to_numpy() != shots.population.to_numpy()).any() or (rungs.statsbomb_match_id.to_numpy() != shots.sb_match_id.to_numpy()).any():
        raise ValueError("the comparator has another population or StatsBomb match for a secondary shot than the pairing")
    if rungs.soccernet_shot_id.notna().any():
        raise ValueError("the comparator joins a SoccerNet shot to a secondary row")
    refs = es.embedding_references(es.scored_mask(shots, "0"))
    refs = np.where(refs >= 0, refs + first_embedding_row, -1).astype(np.int32)
    table = pd.DataFrame({
        "key": shots.key.to_numpy(), "soccernet_shot_id": pd.Series([None] * len(shots), dtype=object).to_numpy(),
        "statsbomb_event_id": shots.key.to_numpy(), "statsbomb_match_id": pd.array(shots.sb_match_id.to_numpy(), dtype="Int64"),
        "match": shots.game.to_numpy(), "half": shots.half.to_numpy(np.int8), "population": shots.population.to_numpy(), "split": shots.split.to_numpy(),
        "sb_body_part": shots.sb_body_part.to_numpy(), "sb_shot_type": shots.sb_shot_type.to_numpy(),
        "pred_vendor_xg": shots.pred_vendor_xg.to_numpy(float), "pred_constant": rungs.p_constant.to_numpy(float),
        "pred_dist_angle": rungs.p_dist_angle.to_numpy(float), "pred_event": rungs.p_event.to_numpy(float),
        "pred_freeze_frame": rungs.p_freeze_frame.to_numpy(float),
        "frame_logit_0": logit(scores.p_0.to_numpy()), "frame_logit_m400": logit(scores.p_m400.to_numpy()),
        "frame_raw_logit_0": scores.raw_mean_0.to_numpy(float), "frame_raw_logit_m400": scores.raw_mean_m400.to_numpy(float),
        "frame_status": es.table_frame_status(scores.status_0, scores.status_m400), "frame_source": es.FRAME_SOURCE})
    table["emb_g_fold0"], table["emb_g_fold1"], table["emb_dinov2"] = refs, refs.copy(), refs.copy()
    table["timing_residual_ms"] = np.nan
    table["anchor_convention"] = et.ANCHOR_CONVENTION["statsbomb_only"]
    table["table_kind"] = "confirmatory"
    return table[et.LABEL_FREE_COLUMNS]


def combine_tables(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """The primary rows (exactly as given, first) followed by the secondary rows. Raises ValueError unless the primary
    rows come out of the concatenation unchanged, dtypes included."""
    combined = pd.concat([primary, secondary], ignore_index=True)
    if not (combined.dtypes == primary.dtypes).all():
        raise ValueError(f"the secondary rows change the column types: {dict(combined.dtypes[combined.dtypes != primary.dtypes])}")
    if not combined.iloc[: len(primary)].equals(primary):
        raise ValueError("the primary rows changed in the combined table")
    return combined


def primary_rows_sha256(combined: pd.DataFrame) -> str:
    """sha256 of the parquet bytes of the primary rows of `combined`, serialised alone by the table writer: equal to the
    sha256 of ticket 11's label-free table when those rows are byte-for-byte the same table."""
    return et.table_sha256(combined[combined.population.eq("primary")].reset_index(drop=True))
