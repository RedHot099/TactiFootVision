"""Release package of the xg-vendor study (ticket 16): ids, our pairing fields and our predictions only.

Owner decision 2026-10-11 (decision log `xgv-release-no-statsbomb`, it narrows D1): the package ships no
StatsBomb data. It holds our code, our predictions and StatsBomb match and event ids as join keys. Everything
StatsBomb-derived is rebuilt by the user from StatsBomb open data with our code: vendor xG, outcomes, locations,
freeze-frame features and the probe targets made from them, the positional comparator's predictions (functions
of StatsBomb data only), StatsBomb timestamps and anything that gives them back (half offsets and timing
residuals: SoccerNet time minus offset minus residual is the StatsBomb time), and the reason a row has no frame
(the StatsBomb time lies after the end of the half video), so `frame_status` is only ok or no_frame. From SoccerNet
the package keeps game ids, shot ids, halves and times in the half video; the study's notes decided nothing about
releasing SoccerNet annotations, so their content (shot label, team, goal flag of a shot, penalty flag, official
split, Labels-v2 metadata) stays out. What a table is says which annotation class its times belong to: `shots.csv`
lists the shot annotations of the index, `recovered_goals.csv` the Goal annotations without a shot annotation (the
join keys of the recovered goals). No frames, clips, embeddings or weights.

Every output column is declared in `TABLES` with its source column and provenance, and every column of every
source table is either shipped or listed in `EXCLUDED` with the reason, so the manifest is complete by
construction. Three guards refuse a package that breaks the rule:

- `check_columns`: a shipped column whose name or name token is forbidden (`FORBIDDEN_NAMES`,
  `FORBIDDEN_TOKENS`), or whose provenance is not one of `PROVENANCE`;
- `check_source_coverage`: a source column that is neither shipped nor excluded with a reason;
- `check_no_copied_values`: a shipped numeric column equal, on more than half of the rows of a join, to a
  numeric StatsBomb or comparator column.

Frame predictions are shipped unrouted: the deployed G-combo probability sets penalties to the pool penalty
rate by a flag that is a SoccerNet annotation (primary rows) or the StatsBomb shot type (7 recovered goals), so
the routed value would publish that flag. `unrouted_frame_logits` recomputes the ensemble probability from the
ten member logits with every row treated as a non-penalty and refuses unless it equals the analysed logit on
every non-penalty row.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.eval.calibration import logit, sigmoid

PROVENANCE = {
    "soccernet_id": "SoccerNet identifier or time in the SoccerNet half video; no annotation content",
    "statsbomb_id": "StatsBomb open-data identifier, used only as a join key",
    "ours": "computed by this study's code from SoccerNet frames, SoccerNet ids or the pairing rule; "
            "holds no StatsBomb value",
}
# Exact column names that are labels or StatsBomb values, and name tokens (split on "_") that mark them.
FORBIDDEN_NAMES = {"goal", "goal_sn", "sb_goal", "on_target_sn", "is_penalty", "penalty", "sb_outcome", "outcome",
                   "label", "label_exclusion", "split", "official_split", "key_pass", "sb_goal_without_sn_goal"}
FORBIDDEN_TOKENS = {"xg", "outcome", "freeze", "ff", "x", "y", "location", "timestamp", "body", "technique",
                    "pressure", "pattern", "residual", "offset", "lag", "team", "side", "home", "away", "score",
                    "player", "dist", "angle", "cone", "gk", "opp", "head", "emb", "sb", "pred", "probe", "target",
                    "keeper", "end", "height", "deflected", "aerial", "sim", "constant", "event", "positions"}
# "event" alone is forbidden ("event features"), but `statsbomb_event_id` is the join key: tokens are checked
# outside the allowed id columns.
ID_COLUMNS = {"statsbomb_event_id", "statsbomb_match_id", "soccernet_shot_id", "soccernet_game"}


@dataclass(frozen=True)
class Column:
    name: str          # name in the package
    source: str        # column of the source table it is taken from ("" when computed here)
    provenance: str    # key of PROVENANCE
    description: str


@dataclass(frozen=True)
class Table:
    file: str
    source: str        # source table (key of SOURCES)
    columns: tuple[Column, ...]
    description: str


SOURCES = {
    "match_map": "results/xgv_pairing/match_map.csv",
    "pairs": "results/xgv_pairing/pairs.parquet",
    "recovered_goals": "results/xgv_pairing/recovered_goals.csv",
    "eval_table": "results/xgv_eval_combined/eval_table_labels_absent.parquet",
    "split_plan": "results/xgv_split_plan/plan.csv",
}

C = Column
TABLES = (
    Table("match_map.csv", "match_map", (
        C("soccernet_game", "game", "soccernet_id", "SoccerNet game id (league/season/date - kick-off home score - score away)"),
        C("league", "league", "soccernet_id", "first folder of the game id"),
        C("season", "season", "soccernet_id", "second folder of the game id"),
        C("statsbomb_match_id", "sb_match_id", "statsbomb_id", "paired StatsBomb match; empty when unpaired"),
        C("pairing", "reason", "ours", "match rule outcome: paired | no_candidate_within_1_day | "
                                       "no_candidate_with_these_teams"),
    ), "every SoccerNet game with its StatsBomb match, or why it has none (match rule, ticket 02)"),
    Table("shots.csv", "pairs", (
        C("soccernet_shot_id", "shot_id", "soccernet_id", "shot id of the SoccerNet shot index v1.2"),
        C("soccernet_game", "game", "soccernet_id", "SoccerNet game id"),
        C("half", "half", "soccernet_id", "half of the game (1 or 2)"),
        C("position_ms", "position_ms", "soccernet_id", "shot anchor: time of the SoccerNet annotation in the half video"),
        C("paired", "paired", "ours", "shot rule outcome: paired one to one with a StatsBomb shot"),
        C("statsbomb_event_id", "sb_event_id", "statsbomb_id", "paired StatsBomb shot event; empty when unpaired"),
        C("statsbomb_match_id", "sb_match_id", "statsbomb_id", "StatsBomb match of the game"),
    ), "the pair table: every index shot of the 135 paired games and its StatsBomb partner (shot rule, ticket 02)"),
    Table("recovered_goals.csv", "recovered_goals", (
        C("soccernet_game", "game", "soccernet_id", "SoccerNet game id"),
        C("half", "half", "soccernet_id", "half of the game"),
        C("goal_label_ms", "goal_label_ms", "soccernet_id", "time of the SoccerNet Goal label in the half video "
                                                            "(a goal without a shot annotation)"),
        C("recovered", "recovered", "ours", "recovery rule outcome: matched one to one to a StatsBomb shot"),
        C("statsbomb_event_id", "sb_event_id", "statsbomb_id", "StatsBomb shot the goal was recovered to; empty if not"),
        C("statsbomb_match_id", "sb_match_id", "statsbomb_id", "StatsBomb match of the game"),
    ), "SoccerNet goals without a shot annotation and the StatsBomb shot each was recovered to (ticket 02)"),
    Table("predictions.csv", "eval_table", (
        C("key", "key", "ours", "row key: SoccerNet shot id (primary) or StatsBomb event id (secondary rows)"),
        C("population", "population", "ours", "primary | recovered_goal | statsbomb_only (pairing populations)"),
        C("soccernet_shot_id", "soccernet_shot_id", "soccernet_id", "SoccerNet shot id; empty for secondary rows"),
        C("statsbomb_event_id", "statsbomb_event_id", "statsbomb_id", "StatsBomb shot event"),
        C("statsbomb_match_id", "statsbomb_match_id", "statsbomb_id", "StatsBomb match"),
        C("soccernet_game", "match", "soccernet_id", "SoccerNet game id"),
        C("half", "half", "soccernet_id", "half of the game"),
        C("anchor_convention", "anchor_convention", "ours", "frame time: soccernet_anchor (primary) or "
                                                            "statsbomb_time (StatsBomb time plus the half offset)"),
        C("frame_status", "", "ours", "ok, or no_frame (the reason names the StatsBomb time, so it is not shipped)"),
        C("frame_exclusion", "", "ours", "pre-registered frame exclusion (decision B): decode_corrupted_frame, else empty"),
        C("frame_logit_0", "", "ours", "G-combo logit at the anchor: ten fold models, one calibration, no penalty routing"),
        C("frame_logit_m400", "", "ours", "the same at -400 ms"),
        C("frame_p_0", "", "ours", "sigmoid of frame_logit_0"),
        C("frame_p_m400", "", "ours", "sigmoid of frame_logit_m400"),
        C("frame_raw_logit_0", "frame_raw_logit_0", "ours", "mean uncalibrated logit of the ten fold models at the anchor"),
        C("frame_raw_logit_m400", "frame_raw_logit_m400", "ours", "the same at -400 ms"),
        C("frame_source", "frame_source", "ours", "which models made the frame columns"),
    ), "G-combo predictions for every evaluation row of the combined label-free table (tickets 11 and 13)"),
    Table("split_plan.csv", "split_plan", (
        C("soccernet_game", "game", "soccernet_id", "SoccerNet game id"),
        C("role", "role", "ours", "evaluation | gate | training in the retrain without the overlap (ticket 05)"),
        C("fold", "fold", "ours", "home-team fold of a training game, -1 otherwise"),
    ), "the retraining split plan, seed 0 (ticket 05)"),
)
del C

_LABELS = "a SoccerNet annotation; the study's notes decided nothing about releasing SoccerNet labels"
_SB = "a StatsBomb value"
_SB_TIME = "with position_ms it gives back the StatsBomb timestamp"
_COMPARATOR = ("a positional comparator prediction: a function of StatsBomb data only; rebuilt by "
               "scripts/vendor_comparator.py (sha256 of the expected file in the README)")
EXCLUDED = {
    "match_map": {
        "sn_home": f"Labels-v2 metadata, {_LABELS}", "sn_away": f"Labels-v2 metadata, {_LABELS}",
        "sn_score": f"Labels-v2 metadata, {_LABELS}", "sb_home": _SB, "sb_away": _SB,
        "name_sim": "computed from StatsBomb team names",
    },
    "pairs": {
        "league": "redundant with the game id", "split": f"the official SoccerNet split, {_LABELS}",
        "label": _LABELS, "team": _LABELS, "goal": _LABELS, "is_penalty": _LABELS,
        "residual_ms": f"timing residual: {_SB_TIME}", "offset_ms": f"half offset: {_SB_TIME}",
        "nearest_sb_ms": "distance to the nearest StatsBomb timestamp", "sb_timestamp_ms": _SB, "sb_team": _SB,
        "sb_side": _SB, "sb_xg": _SB, "sb_outcome": _SB, "sb_goal": _SB, "sb_penalty": _SB, "sb_body_part": _SB,
        "sb_shot_type": _SB, "sb_technique": _SB, "sb_first_time": _SB, "sb_under_pressure": _SB, "sb_x": _SB,
        "sb_y": _SB, "sb_play_pattern": _SB, "sb_n_freeze_frame": _SB,
        "team_agree": "compares the SoccerNet team annotation with the StatsBomb team",
    },
    "recovered_goals": {
        "anchor_ms": "the goal label minus 1000 ms (redundant)", "team": _LABELS, "offset_ms": f"half offset: {_SB_TIME}",
        "any_candidate": "whether a StatsBomb goal lies in the window: from StatsBomb outcomes",
        "sb_timestamp_ms": _SB, "lag_ms": "StatsBomb time minus the attempts anchor: gives back the StatsBomb timestamp",
        "label_lag_ms": "StatsBomb time minus the Goal label time: gives back the StatsBomb timestamp",
        "sb_side": _SB, "team_agree": "compares the SoccerNet team annotation with the StatsBomb team",
        "reason": "names StatsBomb event types (own goals)", "target_position_ms": "StatsBomb time plus the half offset",
    },
    "eval_table": {
        "split": f"the official SoccerNet split, {_LABELS}", "sb_body_part": _SB, "sb_shot_type": _SB,
        "pred_vendor_xg": _SB, "pred_constant": f"the StatsBomb pool goal rate; {_COMPARATOR}",
        "pred_dist_angle": _COMPARATOR, "pred_event": _COMPARATOR, "pred_freeze_frame": _COMPARATOR,
        "frame_logit_0": "penalty-routed by a SoccerNet annotation or the StatsBomb shot type; shipped unrouted",
        "frame_logit_m400": "penalty-routed by a SoccerNet annotation or the StatsBomb shot type; shipped unrouted",
        "emb_g_fold0": "row reference into an embedding matrix that is not released",
        "emb_g_fold1": "row reference into an embedding matrix that is not released",
        "emb_dinov2": "row reference into an embedding matrix that is not released",
        "timing_residual_ms": f"timing residual: {_SB_TIME}", "table_kind": "internal marker (confirmatory)",
        "frame_status": "the no-frame reason says the StatsBomb time lies after the end of the half video; shipped as ok | no_frame",
    },
    "split_plan": {
        "league": "redundant with the game id", "season": "redundant with the game id",
        "home": "home team parsed from the game id (redundant)", "official_split": f"the official SoccerNet split, {_LABELS}",
    },
}
# Not released at all, whole files: listed in the manifest so a reader sees what stays private and why.
NOT_RELEASED = {
    "SoccerNet half videos, clips, frames": "SoccerNet NDA: research use only, never redistributed",
    "G-combo fold-model weights, embeddings (G-combo fold 0/1, frozen DINOv2)": "derived from SoccerNet video under the NDA",
    "StatsBomb open data (events, shots, freeze frames, matches)": "StatsBomb User Agreement 1.2.1 and 7; fetched by the user",
    "positional comparator features and predictions": _COMPARATOR,
    "labels (SoccerNet goal and on-target, StatsBomb outcome, penalty flag, label exclusions)": "SoccerNet annotations and StatsBomb values",
    "H3 probe targets and probe predictions": "functions of StatsBomb freeze frames and locations, or fitted to them",
    "half offsets and timing residuals": _SB_TIME,
}
PAIRING_CODES = {
    "paired": "paired",
    "no StatsBomb match within 1 day": "no_candidate_within_1_day",
    "StatsBomb has matches that day, none with these teams": "no_candidate_with_these_teams",
}


class ReleaseError(ValueError):
    """The package would break the release rule, or an input is not what the study used."""


def name_tokens(name: str) -> set[str]:
    return set(re.split(r"[_\W]+", name.lower())) - {""}


def check_columns(tables: dict[str, pd.DataFrame]) -> None:
    """Refuse a forbidden or undeclared column, an unknown provenance, or a frame that differs from its spec."""
    specs = {t.file: t for t in TABLES}
    for file, frame in tables.items():
        spec = specs.get(file)
        if spec is None:
            raise ReleaseError(f"{file} is not a declared release table")
        names = [c.name for c in spec.columns]
        if list(frame.columns) != names:
            raise ReleaseError(f"{file}: columns {list(frame.columns)} differ from the declared {names}")
        for col in spec.columns:
            if col.provenance not in PROVENANCE:
                raise ReleaseError(f"{file}.{col.name}: unknown provenance {col.provenance!r}")
            if col.name in FORBIDDEN_NAMES:
                raise ReleaseError(f"{file}.{col.name}: forbidden column (a label or a StatsBomb value)")
            if col.name not in ID_COLUMNS and (bad := name_tokens(col.name) & FORBIDDEN_TOKENS):
                raise ReleaseError(f"{file}.{col.name}: forbidden name token(s) {sorted(bad)}")


def check_source_coverage(sources: dict[str, pd.DataFrame]) -> None:
    """Every source column is shipped by some table or excluded with a reason; nothing is both."""
    for key, frame in sources.items():
        shipped = {c.source for t in TABLES if t.source == key for c in t.columns if c.source}
        excluded = set(EXCLUDED.get(key, {}))
        if both := sorted(shipped & excluded):
            raise ReleaseError(f"{key}: columns both shipped and excluded: {both}")
        if missing := sorted(set(frame.columns) - shipped - excluded):
            raise ReleaseError(f"{key}: columns neither shipped nor excluded with a reason: {missing}")
        if absent := sorted((shipped | excluded) - set(frame.columns)):
            raise ReleaseError(f"{key}: declared columns absent from the source: {absent}")


def check_no_copied_values(table: pd.DataFrame, join: str, reference: pd.DataFrame, share: float = 0.5) -> None:
    """Refuse when a numeric column of `table` equals a numeric column of `reference` on more than `share` of
    the rows both have (joined on `join`): a StatsBomb value copied under another name."""
    merged = table.merge(reference, on=join, how="inner", suffixes=("", "__ref"))
    ours = [c for c in table.columns if c != join and c not in ID_COLUMNS and pd.api.types.is_numeric_dtype(table[c])
            and not pd.api.types.is_bool_dtype(table[c])]
    refs = [c for c in reference.columns if c != join and pd.api.types.is_numeric_dtype(reference[c])
            and not pd.api.types.is_bool_dtype(reference[c])]
    for a in ours:
        for b in refs:
            col_b = b if b not in table.columns else f"{b}__ref"
            x = merged[a].to_numpy(float, na_value=np.nan)
            y = merged[col_b].to_numpy(float, na_value=np.nan)
            both = np.isfinite(x) & np.isfinite(y)
            if both.sum() and np.mean(np.abs(x[both] - y[both]) <= 1e-9) > share:
                raise ReleaseError(f"{a} equals the reference column {b} on most rows")


def check_copied_values(tables: dict[str, pd.DataFrame], sources: dict[str, pd.DataFrame]) -> None:
    """`check_no_copied_values` for every table that has StatsBomb or comparator values in its source."""
    pairs, goals, table = sources["pairs"], sources["recovered_goals"], sources["eval_table"]
    check_no_copied_values(tables["shots.csv"], "soccernet_shot_id", pairs.rename(columns={"shot_id": "soccernet_shot_id"})[
        ["soccernet_shot_id", "sb_timestamp_ms", "sb_xg", "sb_x", "sb_y", "residual_ms", "offset_ms", "nearest_sb_ms"]])
    check_no_copied_values(tables["recovered_goals.csv"], "statsbomb_event_id", goals.rename(
        columns={"sb_event_id": "statsbomb_event_id"})[["statsbomb_event_id", "sb_timestamp_ms", "lag_ms", "label_lag_ms",
                                                         "target_position_ms", "offset_ms"]].dropna(subset=["statsbomb_event_id"]))
    check_no_copied_values(tables["predictions.csv"], "key", table[
        ["key", "pred_vendor_xg", "pred_constant", "pred_dist_angle", "pred_event", "pred_freeze_frame", "timing_residual_ms"]])


def match_map_table(match_map: pd.DataFrame) -> pd.DataFrame:
    if unknown := sorted(set(match_map.reason) - set(PAIRING_CODES)):
        raise ReleaseError(f"match map reasons without a release code: {unknown}")
    out = match_map.assign(reason=match_map.reason.map(PAIRING_CODES))
    out["sb_match_id"] = out.sb_match_id.astype("Int64")
    return _select(out, "match_map.csv")


def predictions_table(table: pd.DataFrame, frames: pd.DataFrame, frame_exclusions: dict[str, str]) -> pd.DataFrame:
    """The evaluation rows with the unrouted frame logits of `frames` (key, logit_0, logit_m400)."""
    if missing := sorted(set(frame_exclusions) - set(table.key)):
        raise ReleaseError(f"frame exclusions name rows absent from the table: {missing}")
    out = table.merge(frames, on="key", how="left", validate="one_to_one", suffixes=("__table", ""))
    out = out.drop(columns=["frame_logit_0__table", "frame_logit_m400__table"])
    out["frame_exclusion"] = out.key.map(frame_exclusions).fillna("")
    out["frame_status"] = np.where(out.frame_status == "ok", "ok", "no_frame")
    out["frame_p_0"], out["frame_p_m400"] = sigmoid(out.frame_logit_0), sigmoid(out.frame_logit_m400)
    return _select(out, "predictions.csv")


def unrouted_frame_logits(scores: pd.DataFrame, ensemble: dict, table: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Ensemble logits at offsets 0 and -400 ms with no row routed as a penalty.

    `scores` holds the member logits `xg_f{fold}s{seed}_{0,m400}`, `key` and `penalty_routed`; `ensemble` is the
    retrain's ensemble.json. Refuses unless the result equals `table.frame_logit_*` on every row that was not
    routed (so the shipped value is the analysed prediction wherever the analyses used a model output)."""
    # g_bundle imports torch; only the builder needs it, so the import stays here.
    from xgpoc.models.g_bundle import ensemble_probability

    members = ensemble["members"]
    temperatures = [m["temperature"] for m in members]
    out = pd.DataFrame({"key": scores.key.to_numpy()})
    for offset in ("0", "m400"):
        logits = [scores[f"xg_f{m['fold']}s{m['seed']}_{offset}"].to_numpy(float) for m in members]
        p = ensemble_probability(logits, temperatures, ensemble["calibration"], np.zeros(len(scores), bool))
        out[f"frame_logit_{offset}"] = logit(p)
    check = table[["key", "frame_logit_0", "frame_logit_m400"]].merge(
        out.assign(routed=scores.penalty_routed.to_numpy(bool)), on="key", how="left", validate="one_to_one",
        suffixes=("", "_new"))
    if check.routed.isna().any():
        raise ReleaseError("table rows without member scores")
    for offset in ("0", "m400"):
        a, b = check[f"frame_logit_{offset}"].to_numpy(float), check[f"frame_logit_{offset}_new"].to_numpy(float)
        keep = ~check.routed.to_numpy(bool)
        if not np.array_equal(np.isnan(a[keep]), np.isnan(b[keep])) or np.nanmax(np.abs(a[keep] - b[keep]), initial=0) > tol:
            raise ReleaseError(f"unrouted logits at offset {offset} differ from the analysed ones on non-penalty rows")
    return out


def _select(frame: pd.DataFrame, file: str) -> pd.DataFrame:
    spec = next(t for t in TABLES if t.file == file)
    return pd.DataFrame({c.name: frame[c.source if c.source else c.name] for c in spec.columns})


def build_tables(sources: dict[str, pd.DataFrame], frames: pd.DataFrame, frame_exclusions: dict[str, str]) -> dict[str, pd.DataFrame]:
    """All release tables from the source tables (keys of SOURCES), checked by `check_source_coverage`,
    `check_columns` and `check_copied_values`. `frames` comes from `unrouted_frame_logits`."""
    check_source_coverage(sources)
    tables = {
        "match_map.csv": match_map_table(sources["match_map"]),
        "shots.csv": _select(sources["pairs"], "shots.csv"),
        "recovered_goals.csv": _select(sources["recovered_goals"], "recovered_goals.csv"),
        "predictions.csv": predictions_table(sources["eval_table"], frames, frame_exclusions),
        "split_plan.csv": _select(sources["split_plan"], "split_plan.csv"),
    }
    check_columns(tables)
    check_copied_values(tables, sources)
    return tables


def manifest(tables: dict[str, pd.DataFrame], hashes: dict[str, str]) -> dict:
    """Column-by-column manifest: per table its file hash, rows, source and every column with its provenance;
    then the excluded source columns with reasons and the material not released at all."""
    out = {"provenance": PROVENANCE, "tables": [], "excluded_columns": {}, "not_released": NOT_RELEASED}
    for spec in TABLES:
        frame = tables[spec.file]
        out["tables"].append({
            "file": f"data/{spec.file}", "sha256": hashes[spec.file], "rows": len(frame), "description": spec.description,
            "source": SOURCES[spec.source],
            "columns": [{"name": c.name, "dtype": str(frame[c.name].dtype), "provenance": c.provenance,
                         "source_column": c.source or None, "description": c.description} for c in spec.columns],
        })
    for key, columns in EXCLUDED.items():
        out["excluded_columns"][SOURCES[key]] = dict(sorted(columns.items()))
    return out


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def manifest_markdown(record: dict) -> str:
    """MANIFEST.md: the manifest as tables (a "|" inside a cell is escaped)."""
    lines = ["# Manifest of the xg-vendor release package", "",
             "Generated by `scripts/vendor_release.py` from `manifest.json`. Provenance codes:", ""]
    lines += [f"- `{k}`: {v}." for k, v in record["provenance"].items()]
    for t in record["tables"]:
        lines += ["", f"## `{t['file']}`", "", f"{t['description']}. {t['rows']} rows; source `{t['source']}`; "
                  f"sha256 `{t['sha256']}`.", "", "| Column | Type | Provenance | Source column | Meaning |",
                  "|---|---|---|---|---|"]
        lines += [f"| `{c['name']}` | {c['dtype']} | {c['provenance']} | {('`' + c['source_column'] + '`') if c['source_column'] else 'computed'} "
                  f"| {_cell(c['description'])} |" for c in t["columns"]]
    lines += ["", "## Source columns not released", "", "| Source | Column | Why not |", "|---|---|---|"]
    for source, columns in record["excluded_columns"].items():
        lines += [f"| `{source}` | `{name}` | {_cell(why)} |" for name, why in columns.items()]
    lines += ["", "## Not released at all", "", "| Material | Why not |", "|---|---|"]
    lines += [f"| {_cell(k)} | {_cell(v)} |" for k, v in record["not_released"].items()]
    return "\n".join(lines) + "\n"
