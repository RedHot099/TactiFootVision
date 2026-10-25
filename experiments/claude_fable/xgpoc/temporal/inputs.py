"""The label-free inputs of xg-temporal (ticket 01): the window, the embedding sources, the population and the
extraction list of ticket 01b. No outcome is read here, not even through an old result file: the out-of-fold table
gives only `shot_id` and `z_g_combo`, the index only `shot_id`, `game` and `is_penalty`, the plan only its role and
fold columns.

Window (`Window`)
    the 200 ms grid offsets in [start, o_max] (`GRID`: -1600 ... +400 ms, the 11 offsets of the 01b extraction); main
    read-out start -1600, o_max -600 (6 frames). The middle frame is the grid offset nearest (start + o_max) / 2, the
    earlier one on a tie.
Embedding sources (`Source`, `SOURCES`; the source is a parameter of every run)
    g_xgv   G-combo embeddings of the xgv fold models (ticket 01b), `results/xgt_xgv_embeddings`, encoder "g_combo";
            every embedding must come from the model of the fold that held out the shot's match (fold_source == fold)
    g_xgc   G-combo embeddings of the xgc (v1.2 fold) models, `results/xgc_offset_scores`, encoder "g_combo"
    dinov2  frozen DINOv2-L (CLS + mean patch token), `results/xgc_offset_scores`, encoder "dinov2_frozen"
    The folder contract, which ticket 01b writes for g_xgv (the layout of `scripts/score_offset_frames.py`):
        predictions.parquet   one row per (shot_id, offset_ms, encoder) with at least `FRAME_COLUMNS`:
                                  shot_id (str), game (str), offset_ms (int, ms relative to the shot anchor),
                                  encoder (str), frame_status (str, "ok" for a decoded frame), cut_between_anchor
                                  (bool, null without a frame), embedding_ref (int, the row of the embedding array,
                                  -1 without an embedding), fold_source (int, the fold of the model that produced the
                                  embedding: for g_xgv the xgv host fold of the shot's match)
                              no column name may contain a banned fragment (`provenance.BANNED`); other columns
                              (xg_logit, on_target_logit, ...) are allowed and are not read here
        complete.json         `files`: {"predictions.parquet": sha256, ...}; the embedding array either beside the
                              table, `files["embeddings_<encoder>.npy"]`, or kept elsewhere (outputs/, ~/tactifoot-jobs/)
                              and named by `embeddings`: {"<encoder>": {"path": absolute or relative to
                              experiments/claude_fable, "sha256": ...}}
        embeddings            .npy, float16 or float32, [rows, dim]; row r is the embedding of `embedding_ref` r
    Both files are checked against complete.json before anything is read; the array is opened as a memmap and only
    the population's window rows are read.
Usable frame
    frame_status "ok", no cut between the frame and the anchor (`cut_between_anchor` False), and an embedding
    (embedding_ref >= 0). A cut makes every earlier frame unusable too, so the unusable frames of a window are mostly
    a leading run.
Population (`build_population`, `population_from`)
    the non-penalty shots of the out-of-fold G-combo table (`results/xgd_a1_ensemble/oof.csv`, checked against its
    completion record) in the plan's `training` role (328 matches, 7,062 shots; fold from the plan's training folds),
    with at least one usable frame in the window. One row per shot, sorted by shot id. The out-of-fold table's own
    `fold` column is not read (only shot_id and z_g_combo are, by the phase's rule); that its folds are the plan's xgv
    folds was checked by xg-stack (`xgpoc.stack.inputs.training_shots`) on the file its completion record pins.
Imputation (`impute`)
    an unusable window frame takes the nearest earlier usable frame of the window; when there is none (the leading run
    a cut leaves) the nearest later one. Every window offset has a missing indicator. Row-wise, no fitted state.
Extraction list of ticket 01b (`extraction_shots`)
    every non-penalty `training` shot with at least one decoded frame without a cut at the 11 `GRID` offsets, read
    from the frame mask of the DINOv2 rows of `results/xgc_offset_scores` (scored for every shot, so the mask does not
    depend on which G models had an embedding).
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.paths import INDEX
from xgpoc.temporal import provenance as tp

EXPERIMENT = Path(__file__).resolve().parents[2]
OOF = EXPERIMENT / "results/xgd_a1_ensemble/oof.csv"
PLAN_CSV = a1.PLAN_DIR / "plan.csv"
XGC_SCORES = EXPERIMENT / "results/xgc_offset_scores"
XGV_EMBEDDINGS = EXPERIMENT / "results/xgt_xgv_embeddings"
TABLE = "predictions.parquet"
RECORD = "complete.json"

STEP = 200
GRID = tuple(range(-1600, 401, STEP))           # the 11 offsets of the 01b extraction
START, O_MAX = -1600, -600                      # the main read-out's window (decision D1)
EXPECTED_BASE = {"shots": 7062, "matches": 328, "folds": 5}
FRAME_COLUMNS = ["shot_id", "game", "offset_ms", "encoder", "frame_status", "cut_between_anchor", "embedding_ref",
                 "fold_source"]


# ------------------------------------------------------------------ window
@dataclass(frozen=True)
class Window:
    """The grid offsets in [start, o_max] (module docstring); `control` marks a leakage-control row (o_max > -600)."""

    start: int = START
    o_max: int = O_MAX
    control: bool = False

    def __post_init__(self) -> None:
        if self.start not in GRID or self.o_max not in GRID or self.start >= self.o_max:
            raise ValueError(f"window ({self.start}, {self.o_max}) is not two increasing offsets of {GRID}")
        tp.check_window(self.o_max, self.control)
        if self.control and self.o_max <= tp.LEAK_LIMIT:
            raise ValueError(f"o_max {self.o_max} <= {tp.LEAK_LIMIT} ms is no leakage-control row")

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.o_max + 1, STEP))

    @property
    def middle(self) -> int:
        """The offset nearest (start + o_max) / 2, the earlier one on a tie."""
        centre = (self.start + self.o_max) / 2
        return min(self.offsets, key=lambda o: (abs(o - centre), o))

    def label(self) -> str:
        return f"{self.start}_{self.o_max}" + ("_control" if self.control else "")


# ------------------------------------------------------------------ sources
@dataclass(frozen=True)
class Source:
    """An embedding source (module docstring): its folder, the encoder value of its rows and whether every embedding
    must come from the xgv fold model that held out the shot's match."""

    name: str
    folder: Path
    encoder: str
    xgv_folds: bool


SOURCES = {s.name: s for s in (
    Source("g_xgv", XGV_EMBEDDINGS, "g_combo", True),
    Source("g_xgc", XGC_SCORES, "g_combo", False),
    Source("dinov2", XGC_SCORES, "dinov2_frozen", False),
)}


@dataclass(frozen=True)
class SourceFiles:
    """The verified table and embedding array of a source."""

    source: Source
    table: Path
    table_sha256: str
    embeddings: Path
    embeddings_sha256: str

    def record(self) -> dict:
        return {"source": self.source.name, "encoder": self.source.encoder, "xgv_folds": self.source.xgv_folds,
                "table": {"path": str(self.table), "sha256": self.table_sha256},
                "embeddings": {"path": str(self.embeddings), "sha256": self.embeddings_sha256}}


def verify_source(source: Source) -> SourceFiles:
    """The source's table and embedding array, refused unless both hash to the folder's completion record and the
    table carries `FRAME_COLUMNS` and no banned column name (module docstring)."""
    record = json.loads((source.folder / RECORD).read_text())
    table = source.folder / TABLE
    if (digest := sha256_file(table)) != record["files"].get(TABLE):
        raise ValueError(f"{table} does not match its completion record")
    names = pq.read_schema(table).names
    if missing := sorted(set(FRAME_COLUMNS) - set(names)):
        raise ValueError(f"{table} lacks the columns {missing}")
    tp.check_names(names)
    named = record.get("embeddings", {}).get(source.encoder)
    if named is not None:
        path = Path(named["path"])
        embeddings, expected = (path if path.is_absolute() else EXPERIMENT / path), named["sha256"]
    else:
        embeddings = source.folder / f"embeddings_{source.encoder}.npy"
        expected = record["files"].get(embeddings.name)
    if (found := sha256_file(embeddings)) != expected:
        raise ValueError(f"{embeddings} hashes to {found}, its completion record names {expected}")
    return SourceFiles(source, table, digest, embeddings, found)


def read_frames(files: SourceFiles, shot_ids, offsets) -> pd.DataFrame:
    """`FRAME_COLUMNS` of the source's rows for `shot_ids` at `offsets` (one row per shot and offset)."""
    frames = pd.read_parquet(files.table, columns=FRAME_COLUMNS,
                             filters=[("encoder", "==", files.source.encoder), ("shot_id", "in", sorted(shot_ids)),
                                      ("offset_ms", "in", [int(o) for o in offsets])])
    frames["offset_ms"] = frames.offset_ms.astype(int)
    if frames.duplicated(["shot_id", "offset_ms"]).any():
        raise ValueError(f"{files.table}: more than one {files.source.encoder} row for a shot and offset")
    return frames.reset_index(drop=True)


def usable(frames: pd.DataFrame) -> pd.Series:
    """The usable-frame mask (module docstring)."""
    return (frames.frame_status == tp.OK) & tp.uncut(frames) & (frames.embedding_ref >= 0)


# ------------------------------------------------------------------ base shots
def training_folds(plan: pd.DataFrame) -> pd.Series:
    """game -> xgv fold of the plan's training matches."""
    return plan.loc[plan.role == tp.ROLE].set_index("game").fold.astype(int)


def base_shots(plan: pd.DataFrame, index: Path = INDEX, oof: Path = OOF) -> pd.DataFrame:
    """The non-penalty out-of-fold shots of the training role (index shot_id, sorted; game, fold, z_g_combo). Reads
    only `shot_id` and `z_g_combo` of `oof` and `shot_id`, `game`, `is_penalty` of the index."""
    records.verified_file(oof)
    scored = pd.read_csv(oof, usecols=["shot_id", "z_g_combo"])
    shots = pd.read_parquet(index, columns=["shot_id", "game", "is_penalty"])
    table = scored.merge(shots, on="shot_id", how="left", validate="one_to_one")
    if table.game.isna().any():
        raise ValueError("an out-of-fold shot is not in the index")
    table = table[~table.is_penalty.astype(bool)]
    folds = training_folds(plan)
    if not table.game.isin(folds.index).all():
        raise ValueError("an out-of-fold shot lies outside the plan's training matches")
    table = table.assign(fold=table.game.map(folds))[["shot_id", "game", "fold", "z_g_combo"]]
    table = table.set_index("shot_id").sort_index()
    counts = {"shots": len(table), "matches": table.game.nunique(), "folds": table.fold.nunique()}
    if counts != EXPECTED_BASE:
        raise ValueError(f"base shots {counts} are not {EXPECTED_BASE}")
    return table


# ------------------------------------------------------------------ population
@dataclass(frozen=True)
class Population:
    """The shots of a source and window (`table`: index shot_id, sorted; game, fold, z_g_combo, frames, complete),
    the embedding row of each usable window frame (`refs`, [shots, window offsets], -1 where unusable), the window's
    frame rows with their `used` flag (`frames`), input hashes and counts. No outcome."""

    source: str
    window: Window
    table: pd.DataFrame
    refs: np.ndarray
    frames: pd.DataFrame
    inputs: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    @property
    def observed(self) -> np.ndarray:
        return self.refs >= 0

    @property
    def games(self) -> np.ndarray:
        return self.table.game.to_numpy()

    @property
    def folds(self) -> np.ndarray:
        return self.table.fold.to_numpy()

    @property
    def z(self) -> np.ndarray:
        return self.table.z_g_combo.to_numpy(float)


def population_from(base: pd.DataFrame, frames: pd.DataFrame, window: Window, source: Source) -> Population:
    """The population of `window` from the base shots and the source's frame rows (module docstring)."""
    frames = frames[frames.shot_id.isin(base.index) & frames.offset_ms.isin(window.offsets)].copy()
    shot_game = frames.shot_id.map(base.game)
    if (frames.game != shot_game).any():
        raise ValueError("a frame row's game differs from its shot's")
    frames["used"] = usable(frames).to_numpy()
    if source.xgv_folds:
        embedded = frames[frames.embedding_ref >= 0]
        if (embedded.fold_source.astype(int) != embedded.shot_id.map(base.fold)).any():
            raise ValueError(f"{source.name}: an embedding does not come from the fold model that held out its match")
    used = frames[frames.used]
    keep = base.index[base.index.isin(used.shot_id.unique())]
    table = base.loc[keep].copy()
    column_of = {o: i for i, o in enumerate(window.offsets)}
    refs = np.full((len(table), len(window.offsets)), -1, dtype=np.int64)
    rows = table.index.get_indexer(used.shot_id)
    refs[rows, used.offset_ms.map(column_of).to_numpy()] = used.embedding_ref.to_numpy(np.int64)
    table["frames"] = (refs >= 0).sum(axis=1)
    table["complete"] = table.frames == len(window.offsets)
    frames = frames[frames.shot_id.isin(table.index)].sort_values(["shot_id", "offset_ms"]).reset_index(drop=True)
    counts = {"shots": len(table), "matches": int(table.game.nunique()), "complete_shots": int(table.complete.sum()),
              "base_shots": len(base), "base_matches": int(base.game.nunique()),
              "shots_per_fold": {str(k): int(v) for k, v in table.fold.value_counts().sort_index().items()},
              "matches_per_fold": {str(k): int(v) for k, v in table.groupby("fold").game.nunique().items()},
              "usable_frames": int((refs >= 0).sum()),
              "usable_by_offset": {str(o): int((refs[:, i] >= 0).sum()) for i, o in enumerate(window.offsets)}}
    return Population(source.name, window, table, refs, frames, counts=counts)


def recorded_path(path: Path) -> str:
    """`path` relative to the experiment when inside it, else absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(EXPERIMENT)) if path.is_relative_to(EXPERIMENT) else str(path)


def build_population(source: Source, window: Window | None = None, plan: pd.DataFrame | None = None, index: Path = INDEX,
                     oof: Path = OOF, files: SourceFiles | None = None) -> tuple[Population, SourceFiles]:
    """The population of `source` and `window` (the main window by default) from the frozen inputs, every input
    hashed into `inputs`; `files` (the source verified by `verify_source`) is verified here when not given."""
    window = Window() if window is None else window
    plan = a1.verify_plan(index) if plan is None else plan
    files = verify_source(source) if files is None else files
    base = base_shots(plan, index, oof)
    population = population_from(base, read_frames(files, base.index, window.offsets), window, source)
    inputs = {"plan": {"path": recorded_path(PLAN_CSV), "sha256": a1.PLAN_SHA256},
              "folds": {"path": recorded_path(a1.FOLDS), "sha256": sha256_file(a1.FOLDS)},
              "oof": {"path": recorded_path(oof), "sha256": sha256_file(oof), "columns": ["shot_id", "z_g_combo"]},
              "index": {"path": recorded_path(index), "sha256": sha256_file(index), "columns": ["shot_id", "game", "is_penalty"]},
              "source": files.record()}
    return replace(population, inputs=inputs), files


# ------------------------------------------------------------------ imputation and embeddings
def impute(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(take, missing): for each shot and window position, the position whose frame fills it (itself when usable,
    else the nearest earlier usable position, else the nearest later one) and the missing indicator. Raises when a
    shot has no usable frame."""
    observed = np.asarray(observed, bool)
    if not observed.any(axis=1).all():
        raise ValueError("a shot has no usable frame in the window")
    n, m = observed.shape
    positions = np.arange(m)
    earlier = np.maximum.accumulate(np.where(observed, positions, -1), axis=1)
    later = np.minimum.accumulate(np.where(observed, positions, m)[:, ::-1], axis=1)[:, ::-1]
    take = np.where(earlier >= 0, earlier, later)
    return take, ~observed


def load_embeddings(files: SourceFiles, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the embedding rows of the usable frames, [u, dim] as stored, and `refs` re-indexed into them, -1 kept); the
    array is opened as a memmap and only these rows are read."""
    array = np.load(files.embeddings, mmap_mode="r")
    wanted = np.unique(refs[refs >= 0])
    if len(wanted) and wanted[-1] >= len(array):
        raise ValueError(f"{files.embeddings}: embedding_ref {wanted[-1]} beyond its {len(array)} rows")
    rows = np.asarray(array[wanted])
    position = np.where(refs >= 0, np.searchsorted(wanted, np.where(refs >= 0, refs, 0)), -1)
    return rows, position


# ------------------------------------------------------------------ extraction list of ticket 01b
def extraction_shots(base: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(the 01b extraction list: shot_id and game, sorted, label-free; counts): the base shots with at least one
    decoded frame without a cut at the `GRID` offsets of `frames` (module docstring)."""
    frames = frames[frames.shot_id.isin(base.index) & frames.offset_ms.isin(GRID)]
    decoded = frames.frame_status == tp.OK
    uncut = decoded & tp.uncut(frames)
    keep = sorted(frames.shot_id[uncut].unique())
    shots = base.loc[keep, ["game"]].reset_index()
    chosen = frames[frames.shot_id.isin(keep)]
    counts = {"shots": len(shots), "matches": int(shots.game.nunique()), "offsets": list(GRID),
              "frames_grid": len(keep) * len(GRID), "frames_decoded": int((chosen.frame_status == tp.OK).sum()),
              "frames_decoded_uncut": int(uncut[frames.shot_id.isin(keep)].sum()),
              "base_shots_without_frame": len(base) - len(keep)}
    return shots, counts
