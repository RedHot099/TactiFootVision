"""The population, the E3 and SoccerNet inputs and the outcomes of xg-stack (ticket 01, council C5). Label-free except
`load_outcomes`.

Population
    the 7,062 non-penalty shots of the 328 training matches: `z_g_combo` (the frozen out-of-fold G-combo logit) and the
    fold from `results/xgd_a1_ensemble/oof.csv`, checked against its completion record, against the host folds of
    `cache/folds_xgv_home5.csv` and against the split plan (`approach1.verify_plan`, which also pins the index and the
    folds file). One row per shot, sorted by shot id, index `shot_id`. No outcome is a column of it.
E3 inputs (`e3_table`)
    the E3 frames (`cache/xgd_full3/E3/frames.parquet`) replayed without a GPU with E3's ball-source rule
    (`extract3.row_features`): the replayed raw features must equal the cached `features.parquet` exactly, else nothing
    is built. The frozen sanitation rule (`detector.sanitation.sanitised_features`) is then applied to the same
    replayed frames, as for D3. Per offset (t0 and -400 ms; every column is named `<name>@<offset>`):
        dist, angle          sanitised; missing when the shot location is missing or masked
        shot_x               aligned shot x of the extraction, masked like the geometry (sanitised shot_ok = 0)
        shot_missing         1 - sanitised shot_ok (the missing indicator of dist, angle and shot_x)
        ball_<level>         the sanitised ball source one-hot: none, own (the row's own frame), m<k> (the fallback frame
                             at -k ms), in E3's source order for that offset
        shot_from_ball, shooter_by_net
                             sanitised `shot_from_ball` and `shooter_by_goal` (renamed: the provenance check bans the
                             fragment "goal" in a design column name)
        kp_instance_conf     the first keypoint instance's confidence of the row's frame (`frames.parquet`), with
                             kp_conf_missing
        n_keypoints, n_players
                             the extraction's counts (missing only for an unreadable frame), with frame_missing
        extras (ALL only)    the remaining sanitised geometry with a `<column>_missing` indicator each (`gk_dist_goal`
                             renamed `gk_dist_net`), the remaining sanitised flags, the sanitation indicators and counts
SoccerNet flags (`shot_flags`, from `Labels-v2.json`)
    `parse_annotations` drops every "Goal" annotation and maps both shot label types to one kind, "shot", before any
    rule runs, so no flag can depend on "Goal" or on a shot's own label type. The shooter's team is the team of the
    shot's own annotation (kind "shot", same half and position as the index row); when it is missing, not home or away,
    or ambiguous, every flag is 0 and `sn_team_unknown` is 1. The windows (`WINDOWS`, ms relative to the shot
    annotation, both ends inclusive) take same-team annotations of the same half only.
Outcomes (`load_outcomes`)
    the index `goal` column of the population's shots, read inside the held-out label seal (`seal.
    development_labels_only`), the training-label source of A1's development (`scripts/xgd_a1_ensemble.py`). They are
    an object of their own; fitting code receives y only as an explicit argument.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tactifoot_vision.pipeline import FrameResult
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as x3
from xgpoc.detector import records, seal
from xgpoc.detector import sanitation as sn
from xgpoc.detector.features import COLUMNS, missing_features, shot_features, situation
from xgpoc.paths import INDEX, LABELS_ROOT

EXPERIMENT = Path(__file__).resolve().parents[2]
OOF = EXPERIMENT / "results/xgd_a1_ensemble/oof.csv"
FOLDS = a1.FOLDS
E3 = EXPERIMENT / "cache/xgd_full3/E3"
E3_CONFIG = x3.CONFIGS["E3"]
OFFSETS = x3.STUDY_OFFSETS                      # t0, -400 ms
EXPECTED = {"shots": 7062, "matches": 328, "folds": 5}
INDEX_COLUMNS = ["shot_id", "game", "half", "position_ms", "is_penalty"]   # never `goal`, `label` or `visibility`

# ------------------------------------------------------------------ E3 columns
RENAMED = {"shooter_by_goal": "shooter_by_net", "gk_dist_goal": "gk_dist_net"}
EXTRA_GEOMETRY = ["ff_def_cone", "def_cone_ge1", "def_cone_ge2", "ff_near_opp", "ff_opp_3m", "gk_depth", "gk_off_line_abs",
                  "gk_dist_net"]
EXTRA_FLAGS = ["homography_ok", "shooter_ok", "teams_ok", "gk_ok", "gk_detected", "gk_proxy"]   # shot_ok, ball_ok and
EXTRA_COUNTS = [*sn.INDICATORS, *sn.COUNTS]     # ball_fallback are the shot_missing indicator and the ball one-hot


def col(name: str, offset: int) -> str:
    return f"{name}@{offset}"


def ball_levels(offset: int) -> tuple[str, ...]:
    """The one-hot levels of the sanitised ball source at `offset`: none, own, then E3's fallback frames in order."""
    others = [o for o in E3_CONFIG.order(offset) if o != offset]
    return ("none", "own", *[f"m{-o}" for o in others])


def base_columns(offset: int) -> list[str]:
    """The G columns of one offset, before the spline basis (module docstring)."""
    names = ["dist", "angle", "shot_x", "shot_missing", *[f"ball_{level}" for level in ball_levels(offset)],
             "shot_from_ball", "shooter_by_net", "kp_instance_conf", "kp_conf_missing", "n_keypoints", "n_players",
             "frame_missing"]
    return [col(n, offset) for n in names]


def extra_columns(offset: int) -> list[str]:
    names = [n for c in EXTRA_GEOMETRY for n in (c, f"{c}_missing")] + EXTRA_FLAGS + EXTRA_COUNTS
    return [col(n, offset) for n in names]


def missing_groups(offset: int) -> dict[str, list[str]]:
    """indicator column -> the columns that are missing exactly where it is 1 (checked by `check_missing_groups`)."""
    groups = {"shot_missing": ["dist", "angle", "shot_x"], "kp_conf_missing": ["kp_instance_conf"],
              "frame_missing": ["n_keypoints", "n_players"]} | {f"{c}_missing": [c] for c in EXTRA_GEOMETRY}
    return {col(k, offset): [col(c, offset) for c in v] for k, v in groups.items()}


E3_COLUMNS = [c for o in OFFSETS for c in (*base_columns(o), *extra_columns(o))]


def filled_frame(offset: int, cached: dict[int, FrameResult | None]) -> FrameResult | None:
    """The row's frame at `offset` with E3's ball (`extract3.row_features`, step for step), None when unreadable. It is
    a copy, not a call, because `extract3` returns features, not the filled frame, and changing it would change the code
    the E3 cache records; `e3_table` refuses any row whose features differ from the cache, so the copy cannot drift."""
    result = cached.get(offset)
    if result is None:
        return None
    order = E3_CONFIG.order(offset)
    picked = x3.chosen_ball(cached, order)
    if picked is None:
        return x3.without_ball(result)
    others = [o for o in order if o != offset]
    return x3.with_ball(result, cached[picked[1]], 0 if picked[1] == offset else others.index(picked[1]) + 1)


def replay(frames: pd.DataFrame) -> Iterator[tuple[str, int, FrameResult | None, float]]:
    """(shot_id, offset, filled frame or None, keypoint instance confidence) for every shot (sorted) and study offset."""
    for shot_id, group in frames.groupby("shot_id", sort=True):
        rows = {int(r["offset_ms"]): r for r in group.to_dict("records")}
        cached = {o: ex.frame_from_row(r) for o, r in rows.items()}
        for offset in OFFSETS:
            confidence = float(rows[offset]["kp_instance_conf"]) if rows[offset]["readable"] else np.nan
            yield shot_id, offset, filled_frame(offset, cached), confidence


def offset_row(offset: int, raw: dict, clean: dict, kp_confidence: float) -> dict[str, float]:
    """The `base_columns` and `extra_columns` of one offset from its raw and sanitised feature rows."""
    shot_ok = clean["shot_ok"] == 1.0
    source = clean["ball_source"]
    levels = ball_levels(offset)
    level = "none" if np.isnan(source) else levels[1 + int(source)]
    out = {"dist": clean["dist"], "angle": clean["angle"], "shot_x": raw["shot_x"] if shot_ok else np.nan,
           "shot_missing": float(not shot_ok)} | {f"ball_{name}": float(name == level) for name in levels}
    out |= {"shot_from_ball": clean["shot_from_ball"], "shooter_by_net": clean["shooter_by_goal"],
            "kp_instance_conf": kp_confidence, "kp_conf_missing": float(np.isnan(kp_confidence)),
            "n_keypoints": raw["n_keypoints"], "n_players": raw["n_players"], "frame_missing": float(np.isnan(raw["n_players"]))}
    renamed = {RENAMED.get(k, k): v for k, v in clean.items()}
    for c in EXTRA_GEOMETRY:
        out |= {c: renamed[c], f"{c}_missing": float(np.isnan(renamed[c]))}
    out |= {c: renamed[c] for c in [*EXTRA_FLAGS, *EXTRA_COUNTS]}
    return {col(k, offset): float(v) for k, v in out.items()}


def e3_table(frames: pd.DataFrame, cached_features: pd.DataFrame) -> pd.DataFrame:
    """`E3_COLUMNS`, one row per shot of `frames` (index shot_id, sorted). Raises unless the replayed raw features equal
    `cached_features` (the E3 `features.parquet` rows of the same shots) exactly."""
    raw_rows, rows = [], {}
    for shot_id, offset, filled, kp_confidence in replay(frames):
        raw = shot_features(filled, x3.COMMON["homography_min_confidence"]) if filled is not None else missing_features()
        clean = sn.sanitised_features(situation(filled), filled.homography is not None) if filled is not None else sn.missing_row()
        raw_rows.append({"shot_id": shot_id, "offset_ms": offset} | raw)
        rows.setdefault(shot_id, {}).update(offset_row(offset, raw, clean, kp_confidence))
    replayed = pd.DataFrame(raw_rows).set_index(["shot_id", "offset_ms"]).sort_index()
    cached = cached_features.set_index(["shot_id", "offset_ms"]).sort_index()
    cached = cached[cached.index.get_level_values("offset_ms").isin(OFFSETS)]
    if not cached.index.equals(replayed.index):
        raise ValueError("the replayed E3 rows are not the cached rows")
    if n := int(ex.feature_differences(cached, replayed, COLUMNS, 0.0).any(axis=1).sum()):
        raise ValueError(f"the replayed E3 features differ from the cache in {n} rows")
    table = pd.DataFrame.from_dict(rows, orient="index")[E3_COLUMNS]
    table.index.name = "shot_id"
    check_missing_groups(table)
    return table.sort_index()


def check_missing_groups(table: pd.DataFrame) -> None:
    """Raise unless every indicator of `missing_groups` is 1 exactly where each of its columns is missing."""
    for offset in OFFSETS:
        for indicator, columns in missing_groups(offset).items():
            for c in columns:
                if not (table[c].isna() == (table[indicator] == 1.0)).all():
                    raise ValueError(f"{c} is not missing exactly where {indicator} is 1")


# ------------------------------------------------------------------ SoccerNet flags
SHOT_TYPES = frozenset({"Shots on target", "Shots off target"})
TEAMS = frozenset({"home", "away"})
SHOT = "shot"
WINDOWS = {                                     # flag -> (annotation kind, window start, window end), ms, inclusive
    "sn_direct_fk": ("Direct free-kick", -3000, 300),
    "sn_indirect_fk_3s": ("Indirect free-kick", -3000, 0),
    "sn_corner_20s": ("Corner", -20000, 0),
    "sn_prev_shot_5s": (SHOT, -5000, -300),     # hindsight-prone (C5)
}
S_FLAGS = [*WINDOWS, "sn_team_unknown"]


@dataclass(frozen=True)
class Annotation:
    half: int
    position_ms: int
    kind: str                                   # the SoccerNet label, both shot label types as SHOT
    team: str | None                            # home / away, None otherwise


def parse_annotations(raw: list[dict]) -> list[Annotation]:
    """The annotations of a `Labels-v2.json` without "Goal", the two shot label types made one kind (module docstring)."""
    return [Annotation(int(a["gameTime"].split(" - ")[0]), int(a["position"]), SHOT if a["label"] in SHOT_TYPES else a["label"],
                       a["team"] if a["team"] in TEAMS else None)
            for a in raw if a["label"] != "Goal"]


def shot_flags(annotations: list[Annotation], half: int, position_ms: int) -> dict[str, float]:
    """`S_FLAGS` of the shot at (`half`, `position_ms`), from its game's parsed annotations (module docstring)."""
    teams = {a.team for a in annotations if a.kind == SHOT and a.half == half and a.position_ms == position_ms}
    if len(teams) != 1 or None in teams:
        return dict.fromkeys(WINDOWS, 0.0) | {"sn_team_unknown": 1.0}
    team = teams.pop()
    same = [a for a in annotations if a.half == half and a.team == team]
    return {flag: float(any(a.kind == kind and lo <= a.position_ms - position_ms <= hi for a in same))
            for flag, (kind, lo, hi) in WINDOWS.items()} | {"sn_team_unknown": 0.0}


def labels_file(game: str, root: Path = LABELS_ROOT) -> Path:
    return Path(root) / game / "Labels-v2.json"


def read_annotations(game: str, root: Path = LABELS_ROOT) -> list[dict]:
    return json.loads(labels_file(game, root).read_text())["annotations"]


def flags_table(shots: pd.DataFrame, annotations: dict[str, list[dict]]) -> pd.DataFrame:
    """`S_FLAGS` per shot (index shot_id as `shots`); `shots` has game, half and position_ms, `annotations` maps a game
    to its raw annotation list."""
    parsed = {game: parse_annotations(raw) for game, raw in annotations.items()}
    rows = [shot_flags(parsed[r.game], int(r.half), int(r.position_ms)) for r in shots.itertuples()]
    return pd.DataFrame(rows, index=shots.index, columns=S_FLAGS)


# ------------------------------------------------------------------ population
@dataclass(frozen=True)
class Population:
    """The shots (`table`: index shot_id, sorted; game, half, position_ms, fold, z_g_combo, `E3_COLUMNS`, `S_FLAGS`),
    the hashes of every input (`inputs`) and label-free counts (`counts`). No outcome."""

    table: pd.DataFrame
    inputs: dict
    counts: dict

    @property
    def games(self) -> np.ndarray:
        return self.table.game.to_numpy()

    @property
    def folds(self) -> np.ndarray:
        return self.table.fold.to_numpy()


def training_shots(index: Path, oof: Path, plan: pd.DataFrame) -> pd.DataFrame:
    """The non-penalty shots of `oof` with their index row and fold, checked against the plan's training folds."""
    shots = pd.read_parquet(index, columns=INDEX_COLUMNS)
    scored = pd.read_csv(oof, usecols=["shot_id", "fold", "z_g_combo", "is_penalty"])
    scored = scored[~scored.is_penalty.astype(bool)]
    table = scored.merge(shots, on="shot_id", how="left", validate="one_to_one", suffixes=("", "_index"))
    if table.game.isna().any() or (table.is_penalty_index.astype(bool) != table.is_penalty.astype(bool)).any():
        raise ValueError("an out-of-fold shot is not in the index or its penalty flag differs")
    folds = plan.loc[plan.role == "training"].set_index("game").fold
    if not table.game.isin(folds.index).all():
        raise ValueError("an out-of-fold shot lies outside the plan's training matches")
    if (table.game.map(folds) != table.fold).any():
        raise ValueError("an out-of-fold shot's fold is not its match's host fold")
    keep = ["shot_id", "game", "half", "position_ms", "fold", "z_g_combo"]
    return table[keep].set_index("shot_id").sort_index()


def build_population(index: Path = INDEX, oof: Path = OOF, e3: Path = E3, labels_root: Path = LABELS_ROOT,
                     plan: pd.DataFrame | None = None) -> Population:
    """The population of C5 from the frozen inputs (module docstring); every input is hashed into `inputs`."""
    plan = a1.verify_plan(index) if plan is None else plan
    inputs = {"index": {"path": str(index), "sha256": sha256_file(index)},
              "folds": {"path": str(FOLDS.relative_to(EXPERIMENT)), "sha256": sha256_file(FOLDS)},
              "plan": {"path": str((a1.PLAN_DIR / "plan.csv").relative_to(EXPERIMENT)), "sha256": a1.PLAN_SHA256},
              "oof": {"path": str(oof.relative_to(EXPERIMENT)), "sha256": records.verified_file(oof)}}
    for name in ("frames.parquet", "features.parquet"):
        inputs[f"e3_{name.split('.')[0]}"] = {"path": str((e3 / name).relative_to(EXPERIMENT)), "sha256": records.verified_file(e3 / name)}
    shots = training_shots(index, oof, plan)
    ids = sorted(shots.index)
    frames = pd.read_parquet(e3 / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(e3 / "features.parquet", filters=[("shot_id", "in", ids)])
    detector = e3_table(frames, features)
    if not detector.index.equals(shots.index):
        raise ValueError("E3 does not cover the population exactly")
    games = sorted(shots.game.unique())
    annotations = {g: read_annotations(g, labels_root) for g in games}
    inputs["labels_v2"] = {"root": str(labels_root), "files": len(games),
                           "digest": canonical_digest({g: sha256_file(labels_file(g, labels_root)) for g in games})}
    table = pd.concat([shots, detector, flags_table(shots, annotations)], axis=1)
    counts = {"shots": len(table), "matches": int(table.game.nunique()), "folds": int(table.fold.nunique()),
              "shots_per_fold": {str(k): int(v) for k, v in table.fold.value_counts().sort_index().items()},
              "matches_per_fold": {str(k): int(v) for k, v in table.groupby("fold").game.nunique().items()},
              "flags": {f: int(table[f].sum()) for f in S_FLAGS},
              "e3_t0": {"shot_missing": int(table[col("shot_missing", 0)].sum()),
                        "frame_missing": int(table[col("frame_missing", 0)].sum()),
                        "kp_conf_missing": int(table[col("kp_conf_missing", 0)].sum())}}
    if {k: counts[k] for k in EXPECTED} != EXPECTED:
        raise ValueError(f"population {counts} is not C5's {EXPECTED}")
    return Population(table, inputs, counts)


# ------------------------------------------------------------------ outcomes
@dataclass(frozen=True)
class Outcomes:
    """Goal labels (0/1) by shot id, kept apart from the population; `aligned` returns them in the population's order."""

    shot_id: np.ndarray
    y: np.ndarray

    def aligned(self, population: Population) -> np.ndarray:
        if not np.array_equal(self.shot_id, population.table.index.to_numpy()):
            raise ValueError("the outcomes are not in the population's shot order")
        return self.y


def load_outcomes(population: Population, index: Path = INDEX, plan_csv: Path = a1.PLAN_DIR / "plan.csv") -> Outcomes:
    """The index `goal` of the population's shots, read inside the held-out label seal (module docstring)."""
    with seal.development_labels_only(plan_csv, index):
        labels = pd.read_parquet(index, columns=["shot_id", "game", "goal"]).set_index("shot_id")
    ids = population.table.index
    if not ids.isin(labels.index).all():
        raise ValueError("a population shot has no label")
    goal = labels.goal.loc[ids]
    if goal.isna().any():
        raise ValueError("a population shot has a missing label")
    return Outcomes(ids.to_numpy(), goal.astype(int).to_numpy())
