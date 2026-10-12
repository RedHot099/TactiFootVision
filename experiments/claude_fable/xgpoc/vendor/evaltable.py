"""The evaluation table of the vendor comparison (xg-vendor ticket 07, test seam 2).

One row per evaluation shot. Every downstream analysis (power, ladder, H2, H3, H4, report numbers) is a pure
function of this table, the embeddings it references and the pre-registration parameters, so the meaning of each
column is fixed here. The table is stored as `<name>.parquet` plus a JSON sidecar `<name>.json`.

Keys
    key                 unique row key: the SoccerNet shot id of a primary row, the StatsBomb event id otherwise
    soccernet_shot_id   SoccerNet shot id (primary rows only, else null)
    statsbomb_event_id  StatsBomb event id of the shot (every row; the vendor xG belongs to it)
    statsbomb_match_id  StatsBomb match id (nullable integer)
    match, half         SoccerNet game name and half (period 1-4)
    population          primary | recovered_goal | statsbomb_only
    split               official split of the match: train | valid | test

Covariates (known before the strike; the H4 controls)
    sb_body_part, sb_shot_type

Predictions (every probability in [0, 1], one value per row)
    pred_vendor_xg, pred_constant, pred_dist_angle, pred_event, pred_freeze_frame
                        vendor xG and the positional-comparator rungs, uncalibrated
    frame_logit_0, frame_logit_m400
                        G-combo logit at the shot anchor and at -400 ms after the one calibration of the pool
    frame_raw_logit_0, frame_raw_logit_m400
                        the same models' uncalibrated mean logits (the scale cancels in the offset contrast)
    frame_status        "ok" when the frame was scored; anything else is the listed reason (logits may be NaN)
    frame_source        which models produced the frame logits (free text, e.g. "ensemble_10")

Embedding references (rows of the arrays named in the sidecar's `embedding_sets`; -1 = no embedding)
    emb_g_fold0, emb_g_fold1   pooled embedding of the fold-0 / fold-1 seed-0 G-combo model
    emb_dinov2                 frozen DINOv2 embedding

Pairing data
    timing_residual_ms  SoccerNet anchor minus aligned StatsBomb time (primary rows only, else NaN)
    anchor_convention   soccernet_anchor (primary) | statsbomb_time (secondary populations)
    table_kind          development | dryrun | confirmatory (also in the sidecar; the two must agree). A dryrun table is
                        the confirmatory label-free table with labels permuted within matches (ticket 14's dry run):
                        it runs the confirmatory code path and its numbers are never results.

Labels (absent from a label-free table; added by `join_labels`)
    goal_sn             SoccerNet goal (decision D4)
    on_target_sn        SoccerNet "Shots on target" (primary rows only; counts blocked shots)
    sb_outcome          StatsBomb outcome of the shot
    is_penalty          a pre-shot condition, not an outcome: the shot is a penalty, known before the strike. It sits
                        with the labels because it comes from the same sources at the join (index v1.2 for primary
                        rows, StatsBomb for secondary rows). It is used only for routing: the frame logits of
                        penalties are the routed penalty rate of the G-combo calibration, and the analyses take
                        penalties out of H1-H4 and report them apart. No metric uses it as a target.
    label_exclusion     null for a row that enters the analyses; otherwise the reason code (`LABEL_EXCLUSIONS`) of a
                        row that stays in the table, listed, and is left out of every analysis (spec user story 74)

Not in the table: the H3 probe targets (distance, angle, defenders in the cone, goalkeeper position, opponents within 3 m,
header). They are functions of the StatsBomb event and freeze frame, so the probes read them from the derived StatsBomb
features by `statsbomb_event_id`.

Writing
    Predictions are written first, without the label columns (`labels_state` "absent"). `join_labels` joins the
    labels in a separate step: it records the sha256 of the label-free table and the time, appends both to a log
    and refuses a confirmatory join that is earlier than the pre-registration record (`PreRegistration`: the
    pre-registration document's hash, the time it was committed, the hash of the label-free table it pins and the
    analysis parameters it fixes: margin, bootstrap draws, seeds, the bootstrap of the recalibrated rungs and the frame
    exclusions). A draft record (written from an uncommitted
    document, for the dry run) never admits a confirmatory join.
    The times are those the caller reports (the commit time ticket 14 records); nothing here anchors them to a clock.
    Analyses open a table with `load_table`, which validates it. `validate_table` rejects duplicate keys,
    missing or unexpected columns, labels in a label-free table, a joined table whose join record does not match
    its content, and a confirmatory table joined before the pre-registration record.
"""

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

SCHEMA_VERSION = "xgv-eval-table-1"
KINDS = ("development", "dryrun", "confirmatory")
RUN_LABELS = {"development": "DEVELOPMENT numbers, not results", "dryrun": "DRY RUN on labels permuted within matches, not results",
              "confirmatory": "confirmatory run"}
POPULATIONS = ("primary", "recovered_goal", "statsbomb_only")
SPLITS = ("train", "valid", "test")
ANCHOR_CONVENTION = {"primary": "soccernet_anchor", "recovered_goal": "statsbomb_time", "statsbomb_only": "statsbomb_time"}
LABELS_STATES = ("absent", "joined")

KEY_COLUMNS = ["key", "soccernet_shot_id", "statsbomb_event_id", "statsbomb_match_id", "match", "half", "population", "split"]
COVARIATE_COLUMNS = ["sb_body_part", "sb_shot_type"]
PROBABILITY_COLUMNS = ["pred_vendor_xg", "pred_constant", "pred_dist_angle", "pred_event", "pred_freeze_frame"]
FRAME_LOGIT_COLUMNS = ["frame_logit_0", "frame_logit_m400", "frame_raw_logit_0", "frame_raw_logit_m400"]
FRAME_COLUMNS = [*FRAME_LOGIT_COLUMNS, "frame_status", "frame_source"]
EMBEDDING_COLUMNS = ["emb_g_fold0", "emb_g_fold1", "emb_dinov2"]
PAIRING_COLUMNS = ["timing_residual_ms", "anchor_convention", "table_kind"]
LABEL_COLUMNS = ["goal_sn", "on_target_sn", "sb_outcome", "is_penalty", "label_exclusion"]
LABEL_EXCLUSIONS = {"sb_goal_without_sn_goal": "a StatsBomb goal of the secondary population without any SoccerNet goal (label conflict)"}
# pre-registered, label-blind exclusions of single secondary rows whose scored frame does not show the shot (H1/H2 treat
# them like rows without a frame); the reason codes a pre-registration may use
FRAME_EXCLUSIONS = {"decode_corrupted_frame": "the half video is decode-corrupted at the target: the scored frame shows another moment"}
# the largest R^2 the H3 shuffled-target control may reach (point estimate) before the gate fails an encoder
MAX_CONTROL_R2_LIMIT = 0.05
# the match bootstrap of the recalibrated rungs and stacks: models refitted in every draw, or out-of-fold predictions held fixed
RECALIBRATED_BOOTSTRAPS = ("refit", "fixed")
LABEL_FREE_COLUMNS = [*KEY_COLUMNS, *COVARIATE_COLUMNS, *PROBABILITY_COLUMNS, *FRAME_COLUMNS, *EMBEDDING_COLUMNS, *PAIRING_COLUMNS]
JOINED_COLUMNS = [*LABEL_FREE_COLUMNS, *LABEL_COLUMNS]

OK = "ok"
EMBEDDING_SET_FIELDS = ("file", "sha256", "dim", "rows", "model")
JOIN_FIELDS = ("joined_at", "label_free_sha256", "labels_sha256", "prereg_sha256")


class EvalTableError(ValueError):
    """The table, its sidecar or a join request breaks the evaluation-table contract."""


@dataclass(frozen=True)
class PreRegistration:
    """The pre-registration record a confirmatory join is checked against and the analyses read their parameters from:
    the hash of the pre-registration document, the time it was committed (timezone-aware), the hash of the label-free
    evaluation table that the document pins, the H2 equivalence margin (log-loss), the number of match-bootstrap draws,
    the cross-fit fold seed and the bootstrap seed, the H4 offset logits ("calibrated" or "raw"), the H3 thresholds
    {target: threshold}, confirmatory targets and control-gate threshold (largest control R^2), the sha256 of the timing rule the H4 verdict must have been reached
    under, the bootstrap of the recalibrated rungs (`RECALIBRATED_BOOTSTRAPS`), the frame exclusions {row key: reason
    code of `FRAME_EXCLUSIONS`}, whether the record is a draft (an uncommitted document, dry run only) and, optionally,
    the commit."""

    sha256: str
    recorded_at: datetime
    label_free_table_sha256: str
    margin: float
    n_boot: int
    fold_seed: int
    boot_seed: int
    offset_logits: str
    h3_thresholds: dict
    h3_confirmatory_targets: tuple[str, ...]
    h3_control_max_r2: float
    timing_rule_sha256: str
    recalibrated_bootstrap: str
    frame_exclusions: dict
    draft: bool
    commit: str | None = None


def parse_time(text: str) -> datetime:
    when = datetime.fromisoformat(text)
    if when.tzinfo is None:
        raise EvalTableError(f"time {text!r} has no timezone")
    return when


def format_time(when: datetime) -> str:
    if when.tzinfo is None:
        raise EvalTableError("time has no timezone")
    return when.astimezone(UTC).isoformat()


def is_sha256(text) -> bool:
    return isinstance(text, str) and len(text) == 64 and set(text) <= set("0123456789abcdef")


PREREG_FIELDS = ("sha256", "recorded_at", "label_free_table_sha256", "margin", "n_boot", "fold_seed", "boot_seed", "offset_logits",
                 "h3_thresholds", "h3_confirmatory_targets", "h3_control_max_r2", "timing_rule_sha256", "recalibrated_bootstrap", "frame_exclusions", "draft")


def load_preregistration(path: Path) -> PreRegistration:
    """Read the JSON record `preregistration_record` writes: every field of `PREREG_FIELDS` (`recorded_at` an ISO time
    with timezone) and optionally `commit`."""
    record = json.loads(Path(path).read_text())
    for field in PREREG_FIELDS:
        if field not in record:
            raise EvalTableError(f"pre-registration record lacks {field!r}")
    for field in ("sha256", "label_free_table_sha256", "timing_rule_sha256"):
        if not is_sha256(record[field]):
            raise EvalTableError(f"pre-registration {field} is not a sha256 hex digest")
    margin = record["margin"]
    if isinstance(margin, bool) or not isinstance(margin, int | float) or not 0 < margin <= 0.01:
        raise EvalTableError(f"pre-registration margin {margin!r} is not a log-loss margin in (0, 0.01]")
    for field, low in (("n_boot", 1), ("fold_seed", 0), ("boot_seed", 0)):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise EvalTableError(f"pre-registration {field} {value!r} is not an integer >= {low}")
    if record["offset_logits"] not in ("calibrated", "raw"):
        raise EvalTableError(f"pre-registration offset_logits {record['offset_logits']!r} is not calibrated or raw")
    thresholds, targets = record["h3_thresholds"], record["h3_confirmatory_targets"]
    if not isinstance(thresholds, dict) or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in thresholds.values()):
        raise EvalTableError("pre-registration h3_thresholds must map targets to numbers")
    if not isinstance(targets, list) or not set(targets) <= set(thresholds):
        raise EvalTableError("pre-registration h3_confirmatory_targets must be a list of thresholded targets")
    gate = record["h3_control_max_r2"]
    if isinstance(gate, bool) or not isinstance(gate, int | float) or not 0 <= gate <= MAX_CONTROL_R2_LIMIT:
        raise EvalTableError(f"pre-registration h3_control_max_r2 {gate!r} is not an R^2 in [0, {MAX_CONTROL_R2_LIMIT}]")
    if record["recalibrated_bootstrap"] not in RECALIBRATED_BOOTSTRAPS:
        raise EvalTableError(f"pre-registration recalibrated_bootstrap {record['recalibrated_bootstrap']!r} is not one of {RECALIBRATED_BOOTSTRAPS}")
    exclusions = record["frame_exclusions"]
    if not isinstance(exclusions, dict) or not set(exclusions.values()) <= set(FRAME_EXCLUSIONS):
        raise EvalTableError(f"pre-registration frame_exclusions must map row keys to reason codes of {sorted(FRAME_EXCLUSIONS)}")
    if not isinstance(record["draft"], bool):
        raise EvalTableError("pre-registration draft must be true or false")
    return PreRegistration(sha256=record["sha256"], recorded_at=parse_time(record["recorded_at"]),
                           label_free_table_sha256=record["label_free_table_sha256"], margin=float(margin), n_boot=record["n_boot"],
                           fold_seed=record["fold_seed"], boot_seed=record["boot_seed"], offset_logits=record["offset_logits"],
                           h3_thresholds={k: float(v) for k, v in thresholds.items()}, h3_confirmatory_targets=tuple(targets),
                           h3_control_max_r2=float(gate), timing_rule_sha256=record["timing_rule_sha256"], recalibrated_bootstrap=record["recalibrated_bootstrap"],
                           frame_exclusions=dict(exclusions), draft=record["draft"], commit=record.get("commit"))


def preregistration_record(prereg: PreRegistration) -> dict:
    """The JSON object `load_preregistration` reads."""
    return {"sha256": prereg.sha256, "recorded_at": format_time(prereg.recorded_at), "label_free_table_sha256": prereg.label_free_table_sha256,
            "margin": prereg.margin, "n_boot": prereg.n_boot, "fold_seed": prereg.fold_seed, "boot_seed": prereg.boot_seed,
            "offset_logits": prereg.offset_logits, "h3_thresholds": dict(prereg.h3_thresholds),
            "h3_confirmatory_targets": list(prereg.h3_confirmatory_targets), "h3_control_max_r2": prereg.h3_control_max_r2,
            "timing_rule_sha256": prereg.timing_rule_sha256,
            "recalibrated_bootstrap": prereg.recalibrated_bootstrap, "frame_exclusions": dict(prereg.frame_exclusions),
            "draft": prereg.draft, "commit": prereg.commit}


# ------------------------------------------------------------------ storage
def table_bytes(table: pd.DataFrame) -> bytes:
    """The one serialisation of a table; the file on disk is exactly these bytes, so a table's hash is its file's."""
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False, compression="zstd")
    return buffer.getvalue()


def table_sha256(table: pd.DataFrame) -> str:
    return hashlib.sha256(table_bytes(table)).hexdigest()


def labels_sha256(table: pd.DataFrame) -> str:
    """Hash of the label columns of a joined table, by key."""
    return table_sha256(table[["key", *LABEL_COLUMNS]].sort_values("key").reset_index(drop=True))


def counts(table: pd.DataFrame) -> dict:
    by_both = table.groupby(["population", "split"]).size()
    return {"rows": int(len(table)), "by_population": {k: int(v) for k, v in table.population.value_counts().sort_index().items()},
            "by_split": {k: int(v) for k, v in table.split.value_counts().sort_index().items()},
            "by_population_split": {f"{p}|{s}": int(v) for (p, s), v in by_both.items()}}


def labels_state(table: pd.DataFrame) -> str:
    return "joined" if set(LABEL_COLUMNS) & set(table.columns) else "absent"


def write_table(table: pd.DataFrame, directory: Path, name: str, *, kind: str, embedding_sets: dict,
                unavailable_columns: tuple[str, ...] = (), notes: tuple[str, ...] = (), sources: dict | None = None,
                label_join: dict | None = None, prereg: PreRegistration | None = None) -> dict:
    """Write `<name>.parquet` and `<name>.json` into `directory`; return the sidecar. The table is validated first
    (against `prereg` when it is a confirmatory table with labels)."""
    directory = Path(directory)
    meta = {"schema_version": SCHEMA_VERSION, "table_kind": kind, "labels_state": labels_state(table), **counts(table),
            "table_sha256": table_sha256(table), "columns": list(table.columns), "embedding_sets": embedding_sets,
            "unavailable_columns": list(unavailable_columns), "notes": list(notes), "sources": sources or {},
            "label_join": label_join,
            "writer": {"pandas": pd.__version__, "pyarrow": pyarrow.__version__}}   # the table hash is of the parquet bytes
    validate_table(table, meta, prereg)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.parquet").write_bytes(table_bytes(table))
    (directory / f"{name}.json").write_text(json.dumps(meta, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return meta


def read_table(directory: Path, name: str) -> tuple[pd.DataFrame, dict]:
    """Table and sidecar; raises when the parquet file does not match the hash in its sidecar."""
    directory = Path(directory)
    meta = json.loads((directory / f"{name}.json").read_text())
    payload = (directory / f"{name}.parquet").read_bytes()
    if hashlib.sha256(payload).hexdigest() != meta["table_sha256"]:
        raise EvalTableError(f"{name}.parquet does not match the hash in {name}.json")
    return pd.read_parquet(io.BytesIO(payload)), meta


def load_table(directory: Path, name: str, prereg: PreRegistration | None = None) -> tuple[pd.DataFrame, dict]:
    """`read_table` and `validate_table`: the way analyses open a table. A confirmatory table with labels needs `prereg`."""
    table, meta = read_table(directory, name)
    validate_table(table, meta, prereg)
    return table, meta


def load_embeddings(table: pd.DataFrame, meta: dict, column: str, root: Path | None = None, verify_hash: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings referenced by `column`: (positions of the table rows that have one, matrix in that order).
    Relative embedding files are read from `root`; `verify_hash` also checks the file against the sidecar's sha256."""
    if column not in EMBEDDING_COLUMNS:
        raise EvalTableError(f"{column} is not an embedding column")
    entry = meta["embedding_sets"][column]
    path = Path(entry["file"])
    path = path if path.is_absolute() or root is None else Path(root) / path
    if verify_hash and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise EvalTableError(f"{entry['file']} does not match the sidecar's sha256")
    array = np.load(path, mmap_mode="r")
    if array.shape != (entry["rows"], entry["dim"]):
        raise EvalTableError(f"{entry['file']} has shape {array.shape}, the sidecar says {(entry['rows'], entry['dim'])}")
    refs = table[column].to_numpy()
    positions = np.flatnonzero(refs >= 0)
    return positions, np.asarray(array[refs[positions]])


# ------------------------------------------------------------------ validation
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalTableError(message)


def _check_columns(table: pd.DataFrame, meta: dict) -> None:
    state = meta["labels_state"]
    _require(state in LABELS_STATES, f"unknown labels_state {state!r}")
    present = set(LABEL_COLUMNS) & set(table.columns)
    if state == "absent":
        _require(not present, f"labels present in a table marked labels absent: {sorted(present)}")
    expected = LABEL_FREE_COLUMNS if state == "absent" else JOINED_COLUMNS
    missing = [c for c in expected if c not in table.columns]
    _require(not missing, f"missing required columns: {missing}")
    extra = [c for c in table.columns if c not in expected]
    _require(not extra, f"unexpected columns: {extra}")
    _require(list(table.columns) == expected, "columns are not in the schema order")


def _check_meta(table: pd.DataFrame, meta: dict) -> None:
    _require(meta.get("schema_version") == SCHEMA_VERSION, f"schema version {meta.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    kind = meta["table_kind"]
    _require(kind in KINDS, f"unknown table_kind {kind!r}")
    _require(table.table_kind.eq(kind).all(), "table_kind column disagrees with the sidecar")
    if kind != "development":
        _require(not meta["unavailable_columns"], f"a {kind} table has no unavailable columns")
    _require(set(meta["unavailable_columns"]) <= set(FRAME_LOGIT_COLUMNS), "only frame logit columns can be unavailable")
    for field in ("rows", "by_population", "by_split", "by_population_split"):
        _require(meta[field] == counts(table)[field], f"sidecar count {field} disagrees with the table")


def _check_keys(table: pd.DataFrame) -> None:
    _require(not table.key.isna().any() and table.key.ne("").all(), "empty key")
    duplicated = table.key[table.key.duplicated()]
    _require(duplicated.empty, f"duplicate keys, e.g. {duplicated.iloc[0] if len(duplicated) else None}")
    _require(table.statsbomb_event_id.notna().all(), "every row needs a StatsBomb event id")
    _require(not table.statsbomb_event_id.duplicated().any(), "a StatsBomb event appears in two rows")
    sn = table.soccernet_shot_id.dropna()
    _require(not sn.duplicated().any(), "a SoccerNet shot appears in two rows")
    primary = table.population.eq("primary")
    _require(table.population.isin(POPULATIONS).all(), "unknown population")
    _require(table.split.isin(SPLITS).all(), "unknown split")
    _require(table.half.isin([1, 2, 3, 4]).all(), "half must be a period 1-4")
    _require(table.soccernet_shot_id.notna().eq(primary).all(), "a SoccerNet shot id belongs to primary rows and only to them")
    _require(table.key.eq(table.soccernet_shot_id.where(primary, table.statsbomb_event_id)).all(),
             "key must be the SoccerNet shot id for primary rows and the StatsBomb event id otherwise")


def _check_predictions(table: pd.DataFrame, meta: dict) -> None:
    for column in PROBABILITY_COLUMNS:
        values = table[column].to_numpy(float)
        _require(bool(np.all((values >= 0) & (values <= 1))), f"{column} must be a probability in [0, 1] without gaps")
    unavailable = set(meta["unavailable_columns"])
    for column in FRAME_LOGIT_COLUMNS:
        values = table[column].to_numpy(float)
        _require(not np.isinf(values).any(), f"{column} has infinite values")
        if column in unavailable:
            _require(bool(np.isnan(values).all()), f"{column} is declared unavailable but holds values")
        else:
            _require(bool(np.isfinite(values[table.frame_status.eq(OK).to_numpy()]).all()), f"{column} is missing where frame_status is ok")
    _require(table.frame_status.notna().all() and table.frame_status.ne("").all(), "frame_status needs a value or a reason")
    _require(table.frame_source.notna().all(), "frame_source needs a value")
    primary = table.population.eq("primary")
    residual = table.timing_residual_ms
    _require(residual.notna().eq(primary).all(), "the timing residual belongs to primary rows and only to them")
    _require(table.anchor_convention.eq(table.population.map(ANCHOR_CONVENTION)).all(), "anchor_convention disagrees with the population")


def _check_embeddings(table: pd.DataFrame, meta: dict) -> None:
    _require(set(meta["embedding_sets"]) == set(EMBEDDING_COLUMNS), "the sidecar must describe exactly the three embedding sets")
    for column in EMBEDDING_COLUMNS:
        entry = meta["embedding_sets"][column]
        _require(all(f in entry for f in EMBEDDING_SET_FIELDS), f"embedding set {column} lacks one of {EMBEDDING_SET_FIELDS}")
        refs = table[column].to_numpy()
        kept = refs[refs >= 0]
        _require(bool((refs >= -1).all()), f"{column}: references below -1")
        _require(len(np.unique(kept)) == len(kept), f"{column}: two rows reference the same embedding")
        _require(len(kept) == 0 or kept.max() < entry["rows"], f"{column}: a reference points past the {entry['rows']} stored rows")


def _check_labels(table: pd.DataFrame, meta: dict, prereg: PreRegistration | None) -> None:
    join = meta["label_join"]
    _require(join is not None and all(f in join for f in JOIN_FIELDS), "a joined table needs the label-join record")
    _require(table_sha256(table.drop(columns=LABEL_COLUMNS)) == join["label_free_sha256"],
             "the table without its labels does not match the hash recorded at the label join")
    _require(labels_sha256(table) == join["labels_sha256"], "the labels do not match the hash recorded at the label join")
    _require(table.goal_sn.notna().all() and table.sb_outcome.notna().all() and table.is_penalty.notna().all(), "labels have gaps")
    unknown = set(table.label_exclusion.dropna()) - set(LABEL_EXCLUSIONS)
    _require(not unknown, f"unknown label exclusion codes: {sorted(unknown)}")
    primary = table.population.eq("primary")
    _require(table.on_target_sn.notna().eq(primary).all(), "on_target_sn belongs to primary rows and only to them")
    if meta["table_kind"] == "confirmatory":
        _require(prereg is not None, "a confirmatory table with labels needs the pre-registration record")
        _require(not prereg.draft, "a draft pre-registration record cannot admit a confirmatory table")
        _require(parse_time(join["joined_at"]) >= prereg.recorded_at,
                 f"the label join ({join['joined_at']}) predates the pre-registration record ({format_time(prereg.recorded_at)})")
        _require(join["prereg_sha256"] == prereg.sha256, "the label join recorded another pre-registration hash")
        _require(join["label_free_sha256"] == prereg.label_free_table_sha256, "the label-free table is not the one the pre-registration pins")


def validate_table(table: pd.DataFrame, meta: dict, prereg: PreRegistration | None = None) -> None:
    """Raise `EvalTableError` unless `table` and its sidecar `meta` meet the contract (module docstring)."""
    _check_columns(table, meta)
    _check_meta(table, meta)
    _check_keys(table)
    _check_predictions(table, meta)
    _check_embeddings(table, meta)
    if meta["labels_state"] == "joined":
        _check_labels(table, meta, prereg)
    else:
        _require(meta["label_join"] is None, "a label-free table has no label-join record")


# ------------------------------------------------------------------ label join
def join_labels(directory: Path, name: str, labels: pd.DataFrame, out_name: str, log_path: Path, *,
                prereg: PreRegistration | None = None, now: datetime | None = None) -> dict:
    """Join `labels` (key + `LABEL_COLUMNS`, exactly the table's keys) to the label-free table `name`.

    Writes the joined table as `out_name` with a label-join record (time, sha256 of the label-free table, sha256 of the
    labels, pre-registration hash) in its sidecar and appends the same record to the JSON-lines `log_path`. A
    confirmatory join needs `prereg` and refuses a `now` earlier than its record."""
    directory = Path(directory)
    table, meta = read_table(directory, name)
    _require(meta["labels_state"] == "absent", f"{name} already has labels")
    now = now or datetime.now(UTC)
    format_time(now)   # a naive time cannot be compared with the pre-registration record
    if meta["table_kind"] == "confirmatory":
        _require(prereg is not None, "a confirmatory join needs the pre-registration record")
        _require(not prereg.draft, "a draft pre-registration record cannot admit a confirmatory join")
        _require(now >= prereg.recorded_at, f"refusing to join labels at {format_time(now)}, before the pre-registration record at {format_time(prereg.recorded_at)}")
    _require(list(labels.columns) == ["key", *LABEL_COLUMNS], f"labels need the columns {['key', *LABEL_COLUMNS]}")
    _require(not labels.key.duplicated().any(), "duplicate keys in the labels")
    _require(set(labels.key) == set(table.key), "the labels do not cover exactly the table's keys")
    joined = table.merge(labels, on="key", how="left", validate="one_to_one")[JOINED_COLUMNS]
    _require(joined.key.equals(table.key), "the join changed the row order")
    join = {"joined_at": format_time(now), "label_free_sha256": meta["table_sha256"], "labels_sha256": labels_sha256(joined),
            "prereg_sha256": prereg.sha256 if prereg else None}
    result = write_table(joined, directory, out_name, kind=meta["table_kind"], embedding_sets=meta["embedding_sets"],
                         unavailable_columns=tuple(meta["unavailable_columns"]), notes=tuple(meta["notes"]), sources=meta["sources"],
                         label_join=join, prereg=prereg)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(json.dumps({**join, "label_free_table": name, "joined_table": out_name, "joined_sha256": result["table_sha256"],
                              "table_kind": meta["table_kind"]}, sort_keys=True) + "\n")
    return result
