"""The A/B/C ablation of xg-dino on the 7,062 training shots, out of fold (xg-dino ticket 01; spec §Arms, owner
decisions D1-D3, council decision C1 of 2026-10-27, which wins where they differ).

Inputs (`load_inputs`, label-free, inside `no_statsbomb_reads()`)
    primary     `xgpoc.stack_location.inputs.build_population` under `location.FEET_T0` on `cache/xgd_full4_feet_t0/E3`
                (the frozen feet_t0 rule with E3's t0 ball-source flags; results/xgs_stack_feet_t0 used the historical
                own-flags variant, `feet_rerun.py` `feet_t0_own_flags`, so it is not an exact baseline)
    secondary   `xgpoc.stack.inputs.build_population` on `cache/xgd_full3/E3` (the old detector, never saw v3)
    subsets     `no_v3`: matches without any SoccerNet-v3 image in the detector pool (`pool_index.parquet`, source
                soccernet_v3); `unseen`: `results/xgs_stack_dd2c_leak/seen.csv` (window 3 s) false; their complements
                `v3` and `seen` serve the heterogeneity contrasts
Arms (`arm_set`; the C6 spec's candidate `bc`: dist natural spline 3 df, standardised design, L2 logistic by Newton)
    A        z_g_combo alone, unpenalised: a Platt fit per outer fold
    B_DA     dist spline + angle@0 + shot_missing@0, every column penalised
    B        `GEOMETRY` (C1: location, distance, angle, goalkeeper, opponents, cone at t0 and -400 ms and their missing
             indicators; dist@0 as the spline), every column penalised
    C        z_g_combo + B's columns, every column but z_g_combo penalised
    descriptive: BQ / CQ (B / C + `QUALITY`: ball source, detection flags and counts, keypoint confidence, sanitation
             indicators), B_legacy / C_legacy (xg-stack's ALL with the SoccerNet flags, without / with z), and the
             secondary block (old-E3 geometry: B_DA, B, C)
    The zero-variance rule (`fit.constant_columns`) drops constant inputs per table. The penalty is by column name
    (`NamedModel`): only a column named z_g_combo is free. C is chosen per outer fold by inner match-grouped CV over the
    other host folds (`fit.chosen_c`) on `C_GRID` (C6's grid extended by 3, 10, 30, 100), a grid-edge flag per fold.
    Every arm uses the same outer and inner folds. No arm gets an extra Platt layer.
Read-out (`read_out`, on any outcome vector: real, permuted or simulated)
    per block and arm: log-loss, Brier, AUC, calibration slope and intercept, a 10-bin (equal-count) reliability table;
    per contrast ΔLL, ΔBrier, ΔAUC with the paired match bootstrap (2,000 draws, seed 0): 95% and 90% intervals, one-
    and two-sided p, TOST at δ 0.002 with the equivalence curve and its smallest margin, ΔLL among goals and among
    non-goals and ΔLL standardised to the primary goal rate. Holm over the one-sided superiority tests of C-B and C-A.
    Blocks: primary; secondary; each subset restricted from the primary out-of-fold probabilities (the primary draws,
    the leak controls' estimand) and refitted on its own (descriptive, as xgs_detector_leak.py).
Decision table (`decision`, C1) on the primary block:
    C-B  "DINOv2 adds" when its one-sided superiority test is Holm-rejected (its one-sided 95% upper bound < 0);
         else "equivalent for this stacker at δ = 0.002" when TOST holds (90% interval inside ±0.002); else
         "inconclusive at δ = 0.002; smallest margin shown = X"
    C-A  "geometry adds over the image" when Holm-rejected; B-A two-sided, descriptive
Leak verdict (`leak_verdict`, C1): "not decidable" when a heterogeneity 95% interval (C-B on seen minus unseen, on v3
    minus no-v3 matches, from the same primary draws) excludes 0 or the restricted no_v3 C-B 95% interval excludes the
    primary point estimate; else "decidable". These controls cannot resolve δ.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import location as loc
from xgpoc.detector.guard import check_path, no_statsbomb_reads
from xgpoc.detector.testrule import shot_log_loss
from xgpoc.paths import INDEX
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import provenance as sp
from xgpoc.stack import stats as ss
from xgpoc.stack_location import inputs as sl
from xgpoc.vendor import ladder as ld

EXPERIMENT = si.EXPERIMENT
CANDIDATE = "bc"                                   # the C6 protocol's chosen candidate
Z = sf.Z
GEOMETRY_NAMES = ("dist", "angle", "shot_x", "shot_missing",
                  *(n for c in si.EXTRA_GEOMETRY for n in (c, f"{c}_missing")))
GEOMETRY = tuple(si.col(n, o) for o in si.OFFSETS for n in GEOMETRY_NAMES)     # the C1 allowlist of B and C
QUALITY = tuple(c for c in si.E3_COLUMNS if c not in GEOMETRY)                 # the descriptive Q block
SPLINE = sf.CANDIDATES[CANDIDATE].spline
DF = sf.CANDIDATES[CANDIDATE].df
B_DA_LINEAR = (si.col("angle", 0), si.col("shot_missing", 0))
C_GRID = (*sf.C_GRID, 3.0, 10.0, 30.0, 100.0)
CONFIRMATORY_ARMS = ("A", "B_DA", "B", "C")
DESCRIPTIVE_ARMS = ("BQ", "CQ", "B_legacy", "C_legacy")
SECONDARY_ARMS = CONFIRMATORY_ARMS
SUPERIORITY = {"C-B": ("C", "B"), "C-A": ("C", "A")}              # one-sided, Holm family
CONTRASTS = SUPERIORITY | {"B-A": ("B", "A")}                        # B-A two-sided, descriptive
EQUIVALENCE = ("C-B", "C-A")
DESCRIPTIVE_CONTRASTS = {"B_DA-A": ("B_DA", "A"), "C-B_DA": ("C", "B_DA"), "BQ-B": ("BQ", "B"), "CQ-C": ("CQ", "C"),
                         "CQ-BQ": ("CQ", "BQ"), "C_legacy-B_legacy": ("C_legacy", "B_legacy"),
                         "B_legacy-A": ("B_legacy", "A"), "C_legacy-A": ("C_legacy", "A")}
DELTA = 0.002
TOST_ALPHA = 0.05                                  # each one-sided test: the 90% interval inside ±δ
HOLM_ALPHA = 0.05
MARGINS = ld.MARGIN_GRID
RELIABILITY_BINS = 10
SUBSETS = ("no_v3", "unseen")
HETEROGENEITY = {"seen-unseen": ("seen", "unseen"), "v3-no_v3": ("v3", "no_v3")}
EXPECTED = {"primary": {"shots": 7062, "matches": 328}, "no_v3": {"shots": 1028, "matches": 47},
            "unseen": {"shots": 3249}}
EXPECTED_GOALS = {"primary": 902, "no_v3": 137}    # invariant under a permutation within matches
DISCLOSURES = (
    "Development comparisons on the 328 training matches, not an independent test: the 172 held-out matches are "
    "spent; the descriptive row on them (frozen test models) is the size check.",
    "G-combo's epoch and temperature were chosen on each scored fold (C6 disclosure 3): A and C carry optimism that B "
    "does not, so C-B is tilted toward 'DINOv2 adds'.",
    "Stacking provenance: the training-fold z values of the stacker come from G-combo models that saw the scored fold "
    "(C6 disclosure).",
    "The retrained detector dd2c (feet_t0 geometry) was trained on SoccerNet-v3 pool frames of 3,813 of the 7,062 "
    "shots, more often goals; the no_v3 and unseen controls (D1, C1) cannot resolve δ.",
    "results/xgs_stack_feet_t0 used the historical feet_t0_own_flags variant (scripts/feet_rerun.py), not the frozen "
    "feet_t0 rule with E3's t0 ball-source flags used here; it is not an exact baseline.",
    "E3's configuration was chosen on 135 of the 172 held-out matches by StatsBomb position error, with no outcomes.",
)
TODO = {"status": "TODO: filled by the orchestrator before the freeze (not set by ticket 01)", "value": None}


# ------------------------------------------------------------------ paths and inputs
@dataclass(frozen=True)
class Paths:
    index: Path = INDEX
    oof: Path = si.OOF
    feet: Path = EXPERIMENT / "cache/xgd_full4_feet_t0/E3"
    secondary: Path = si.E3
    seen: Path = EXPERIMENT / "results/xgs_stack_dd2c_leak/seen.csv"
    pool: Path = Path("/mnt/storage/tactifoot_research/detector_data/pool_index.parquet")

    def all(self) -> list[Path]:
        return [self.index, self.oof, self.feet / "frames.parquet", self.feet / "features.parquet",
                self.secondary / "frames.parquet", self.secondary / "features.parquet", self.seen, self.pool]


FROZEN = Paths()


@dataclass(frozen=True)
class Inputs:
    """The primary and secondary populations (same shots, same order), the subsets as masks over the primary rows, the
    hashes of every input and label-free counts."""

    primary: si.Population
    secondary: si.Population | None
    subsets: dict[str, np.ndarray]
    hashes: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)


def subset_counts(table: pd.DataFrame, subsets: dict[str, np.ndarray]) -> dict:
    out = {"primary": {"shots": len(table), "matches": int(table.game.nunique())}}
    return out | {k: {"shots": int(m.sum()), "matches": int(table.game[m].nunique())} for k, m in subsets.items()}


def check_counts(counts: dict) -> None:
    for name, expected in EXPECTED.items():
        if {k: counts[name][k] for k in expected} != expected:
            raise ValueError(f"population {name} {counts[name]} is not the spec's {expected}")


def load_inputs(paths: Paths = FROZEN) -> Inputs:
    """The frozen inputs (module docstring), label-free; every path is checked by the StatsBomb guard first."""
    with no_statsbomb_reads():
        for p in paths.all():
            check_path(p)
        plan = a1.verify_plan(paths.index)
        primary = sl.build_population(loc.FEET_T0, paths.index, paths.oof, e3=paths.feet, plan=plan)
        secondary = si.build_population(paths.index, paths.oof, paths.secondary, plan=plan)
        seen = pd.read_csv(paths.seen, index_col="shot_id").seen
        pool = pd.read_parquet(paths.pool, columns=["source", "game"])
    table = primary.table
    if not secondary.table.index.equals(table.index) or not np.array_equal(secondary.table[Z], table[Z]):
        raise ValueError("the secondary population is not the primary's shots and z")
    if not seen.index.sort_values().equals(table.index) or seen.dtype != bool:
        raise ValueError("seen.csv does not cover the population exactly with booleans")
    v3_games = set(pool.game[pool.source == "soccernet_v3"])
    subsets = {"no_v3": ~table.game.isin(v3_games).to_numpy(), "unseen": ~seen.loc[table.index].to_numpy()}
    counts = subset_counts(table, subsets) | {"populations": {"primary": primary.counts, "secondary": secondary.counts},
                                             "v3_games_in_pool": len(v3_games)}
    check_counts(counts)
    hashes = {"primary": primary.inputs, "secondary": secondary.inputs,
              "seen": {"path": str(paths.seen.relative_to(EXPERIMENT)), "sha256": sha256_file(paths.seen)},
              "pool_index": {"path": str(paths.pool), "sha256": sha256_file(paths.pool)}}
    return Inputs(primary, secondary, subsets, hashes, counts)


def masks(inputs_subsets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The subsets and their complements (`v3`, `seen`)."""
    out = dict(inputs_subsets)
    out |= {"v3": ~out["no_v3"]} if "no_v3" in out else {}
    out |= {"seen": ~out["unseen"]} if "unseen" in out else {}
    return out


# ------------------------------------------------------------------ arms and models
@dataclass(frozen=True)
class NamedModel(sf.Model):
    """`fit.Model` whose penalties follow the design column names: λ = 1 / C on every column except those in `free`
    (λ = 0). `fit.out_of_fold` and `fit.chosen_c` call `penalties(n_columns, c)`; the width must be the arm's."""

    columns: tuple[str, ...] = ()
    free: frozenset[str] = frozenset()

    def penalties(self, n_columns: int, c: float) -> np.ndarray:
        if n_columns != len(self.columns):
            raise ValueError(f"{self.name}: {n_columns} design columns, the model names {len(self.columns)}")
        return np.array([0.0 if name in self.free else 1.0 / c for name in self.columns])

    @property
    def penalised(self) -> bool:
        return any(name not in self.free for name in self.columns)


def model_for(arm: sf.Arm) -> NamedModel:
    """The arm's model: `C_GRID`, z_g_combo (by name) the only unpenalised column."""
    return NamedModel(f"inner_cv_named:{arm.name}", C_GRID, False, tuple(arm.columns), frozenset({Z}) & set(arm.columns))


def _arm(name: str, linear, spline, table: pd.DataFrame, dropped: dict) -> sf.Arm:
    columns = [c for c in sf.unique(linear) if c not in spline]
    dropped[name] = sf.constant_columns(table, [*columns, *spline])
    status = "confirmatory" if name in CONFIRMATORY_ARMS else "descriptive"
    arm = sf.Arm(name, status, tuple(c for c in columns if c not in dropped[name]),
                 tuple(c for c in spline if c not in dropped[name]), DF)
    sf.check_no_constant(table, arm)
    sp.check_names(arm.columns)
    return arm


def arm_set(table: pd.DataFrame, names=(*CONFIRMATORY_ARMS, *DESCRIPTIVE_ARMS)) -> tuple[dict[str, sf.Arm], dict[str, list[str]]]:
    """The arms of `names` on `table` after the zero-variance rule, and the dropped columns per arm."""
    geometry = [c for c in GEOMETRY if c not in SPLINE]
    linear = {"A": (Z,), "B_DA": B_DA_LINEAR, "B": geometry, "C": (Z, *geometry), "BQ": (*geometry, *QUALITY),
              "CQ": (Z, *geometry, *QUALITY)}
    if set(names) & {"B_legacy", "C_legacy"}:
        legacy = sf.spec(CANDIDATE, table).arms["ALL"]
        linear |= {"B_legacy": [c for c in legacy.linear if c != Z], "C_legacy": legacy.linear}
    dropped: dict[str, list[str]] = {}
    arms = {n: _arm(n, linear[n], () if n == "A" else SPLINE, table, dropped) for n in names}
    for n in ("B_DA", "B", "C"):
        if n in arms and (bad := [c for c in arms[n].inputs if c != Z and c not in GEOMETRY]):
            raise ValueError(f"arm {n} holds columns outside the geometry allowlist: {bad}")
    return arms, dropped


# ------------------------------------------------------------------ prepared designs (label-free)
@dataclass(frozen=True)
class Block:
    """One fitted population: `rows` (positions in the primary), its games and folds, its arms with their models and
    fold designs (built once, y-free), the dropped columns and its bootstrap weights."""

    name: str
    rows: np.ndarray
    games: np.ndarray
    folds: np.ndarray
    arms: dict[str, sf.Arm]
    models: dict[str, NamedModel]
    designs: dict[str, list[sf.FoldDesign]]
    dropped: dict[str, list[str]]
    weights: np.ndarray


@dataclass(frozen=True)
class Prepared:
    shot_id: np.ndarray
    blocks: dict[str, Block]
    masks: dict[str, np.ndarray]                   # subsets and complements over the primary rows


def _block(name: str, table: pd.DataFrame, rows: np.ndarray, names, draws: int, seed: int) -> Block:
    sp.check_names([c for c in table.columns if c not in ("game", "half", "position_ms", "fold")])
    arms, dropped = arm_set(table, names)
    models = {n: model_for(a) for n, a in arms.items()}
    folds = table.fold.to_numpy()
    designs = {n: sf.fold_designs(table, folds, a, models[n].penalised and len(a.columns) > 1) for n, a in arms.items()}
    games = table.game.to_numpy()
    return Block(name, rows, games, folds, arms, models, designs, dropped, ss.bootstrap_weights(games, draws, seed))


def prepare(inputs: Inputs, draws: int = ss.N_BOOT, seed: int = ss.BOOT_SEED, primary_arms=None,
            full: bool = True) -> Prepared:
    """Every block's arms and fold designs and the bootstrap weights, from the inputs only (no outcome), so the
    read-out can run on any number of outcome vectors. `full` False builds the primary block only (simulations)."""
    table = inputs.primary.table
    n = len(table)
    names = primary_arms or (*CONFIRMATORY_ARMS, *DESCRIPTIVE_ARMS)
    blocks = {"primary": _block("primary", table, np.arange(n), names, draws, seed)}
    if full and inputs.secondary is not None:
        blocks["secondary"] = _block("secondary", inputs.secondary.table, np.arange(n), SECONDARY_ARMS, draws, seed)
    if full:
        for name, mask in inputs.subsets.items():
            rows = np.flatnonzero(mask)
            blocks[f"refit:{name}"] = _block(f"refit:{name}", table.iloc[rows], rows, CONFIRMATORY_ARMS, draws, seed)
    return Prepared(table.index.to_numpy(), blocks, masks(inputs.subsets) if full else {})


def fold_record(block: Block) -> dict[str, list]:
    """Per arm: each outer fold's training and scored rows (digests) and its inner splits, to show they are shared."""
    def rows(d: sf.FoldDesign) -> dict:
        return {"fold": d.fold, "train": canonical_digest(d.train.tolist()), "test": canonical_digest(d.test.tolist())}
    return {n: [rows(d) | {"inner": [rows(i) for i in d.inner]} for d in ds] for n, ds in block.designs.items()}


def fit_block(block: Block, y: np.ndarray, names=None) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Out-of-fold probabilities and the chosen C per fold of the block's arms (`names`, all by default) on `y` (the
    block's rows)."""
    chosen = {n: [] for n in names or block.arms}
    p = {n: sf.out_of_fold(block.designs[n], y, block.models[n], chosen[n]) for n in chosen}
    return p, chosen


def grid_edges(chosen: list) -> list[bool | None]:
    """Per fold: the inner CV chose the smallest or the largest C of `C_GRID` (None for an arm without CV)."""
    return [None if c is None else bool(c in (C_GRID[0], C_GRID[-1])) for c in chosen]


# ------------------------------------------------------------------ statistics
def _summary(point: float, draws: np.ndarray) -> dict:
    draws = draws[np.isfinite(draws)]
    lo95, hi95, lo90, hi90 = np.percentile(draws, [2.5, 97.5, 5, 95])
    b = len(draws)
    above, below = 1 + np.sum(draws >= 0), 1 + np.sum(draws <= 0)
    return {"delta": float(point), "lo95": float(lo95), "hi95": float(hi95), "lo90": float(lo90), "hi90": float(hi90),
            "upper_one_sided_95": float(hi90), "p_one_sided": float(above / (b + 1)),
            "p_two_sided": float(min(1.0, 2 * min(above, below) / (b + 1))), "boot_sd": float(draws.std(ddof=1)),
            "draws": int(b)}


def _equivalence(draws: np.ndarray) -> tuple[dict, pd.DataFrame]:
    draws = draws[np.isfinite(draws)]
    curve = ld.equivalence_curve(draws, MARGINS, TOST_ALPHA)
    at = ld.equivalence_curve(draws, np.array([DELTA]), TOST_ALPHA).iloc[0]
    return {"delta": DELTA, "p_tost": float(at.p_tost), "equivalent": bool(at.equivalent),
            "smallest_margin": ld.smallest_margin(curve)}, curve


def _auc_draws(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return ld._auc(y, np.clip(p, ss.CLIP, 1 - ss.CLIP), w)


def ll_differences(y: np.ndarray, p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return shot_log_loss(y, p1, ss.CLIP) - shot_log_loss(y, p0, ss.CLIP)


def ll_contrast(y: np.ndarray, d: np.ndarray, weights: np.ndarray, goal_rate: float | None = None) -> tuple[dict, pd.DataFrame]:
    """ΔLL of the per-shot differences `d`: summary, TOST, goal / non-goal split and, with `goal_rate`, ΔLL
    standardised to that goal rate (π Δ_goals + (1 - π) Δ_non-goals, per draw); the equivalence curve."""
    out = _summary(d.mean(), ss.boot_means(weights, d))
    out["tost"], curve = _equivalence(ss.boot_means(weights, d))
    goals, non = y == 1, y == 0
    dg, dn = ss.boot_means(weights, d, goals), ss.boot_means(weights, d, non)
    out["goals"] = _summary(d[goals].mean(), dg)
    out["non_goals"] = _summary(d[non].mean(), dn)
    if goal_rate is not None:
        out["standardised"] = _summary(goal_rate * d[goals].mean() + (1 - goal_rate) * d[non].mean(),
                                       goal_rate * dg + (1 - goal_rate) * dn) | {"goal_rate": goal_rate}
    return out, curve


def contrast(y: np.ndarray, p1: np.ndarray, p0: np.ndarray, weights: np.ndarray,
             goal_rate: float | None = None) -> tuple[dict, pd.DataFrame]:
    """ΔLL (`ll_contrast`), ΔBrier and ΔAUC of p1 - p0 with the paired match bootstrap `weights` (draws, shots)."""
    y = np.asarray(y, int)
    out = {}
    out["log_loss"], curve = ll_contrast(y, ll_differences(y, p1, p0), weights, goal_rate)
    d_br = (p1 - y) ** 2 - (p0 - y) ** 2
    out["brier"] = _summary(d_br.mean(), ss.boot_means(weights, d_br))
    auc_point = ss.scores(y, p1)["auc"] - ss.scores(y, p0)["auc"]
    out["auc"] = _summary(auc_point, _auc_draws(y, p1, weights) - _auc_draws(y, p0, weights))
    return out, curve


def reliability(y: np.ndarray, p: np.ndarray, bins: int = RELIABILITY_BINS) -> pd.DataFrame:
    """Equal-count bins of p: n, mean p, observed rate."""
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    frame = pd.DataFrame({"bin": q.to_numpy(), "p": p, "y": y}).groupby("bin")
    return pd.DataFrame({"n": frame.size(), "mean_p": frame.p.mean(), "observed": frame.y.mean()}).reset_index()


def holm_superiority(contrasts: dict) -> dict[str, bool]:
    return ss.holm({c: contrasts[c]["log_loss"]["p_one_sided"] for c in SUPERIORITY if c in contrasts}, HOLM_ALPHA)


def decision(c_b: dict, c_a: dict, holm: dict[str, bool]) -> dict:
    """The C1 decision table (module docstring) from the C-B and C-A LL summaries and the Holm rejections."""
    tost = c_b["tost"]
    if holm["C-B"] and c_b["upper_one_sided_95"] < 0:
        wording = "DINOv2 adds"
    elif tost["equivalent"]:
        wording = f"equivalent for this stacker at δ = {DELTA}"
    else:
        wording = f"inconclusive at δ = {DELTA}; smallest margin shown = {tost['smallest_margin']:.4f}"
    return {"C-B": wording, "C-A": "geometry adds over the image" if holm["C-A"] and c_a["upper_one_sided_95"] < 0
            else "no detectable gain of geometry over the image", "holm_rejected": holm}


def _read_block(name: str, y: np.ndarray, p: dict[str, np.ndarray], weights: np.ndarray, goal_rate: float) -> dict:
    contrasts, curves, rel = {}, [], []
    for c, (a, b) in (CONTRASTS | DESCRIPTIVE_CONTRASTS).items():
        if a in p and b in p:
            contrasts[c], curve = contrast(y, p[a], p[b], weights, goal_rate)
            curves.append(curve.assign(block=name, contrast=c))
    for arm, q in p.items():
        rel.append(reliability(y, q).assign(block=name, arm=arm))
    return {"scores": {arm: ss.scores(y, q) for arm, q in p.items()}, "contrasts": contrasts,
            "holm_superiority": holm_superiority(contrasts),
            "_curves": pd.concat(curves), "_reliability": pd.concat(rel)}


def heterogeneity(y: np.ndarray, p: dict[str, np.ndarray], weights: np.ndarray, subsets: dict[str, np.ndarray]) -> dict:
    """C-B on one subset minus C-B on its complement, from the same primary draws."""
    d = ll_differences(y, p["C"], p["B"])
    out = {}
    for name, (a, b) in HETEROGENEITY.items():
        if a in subsets and b in subsets:
            ma, mb = subsets[a], subsets[b]
            out[name] = _summary(d[ma].mean() - d[mb].mean(), ss.boot_means(weights, d, ma) - ss.boot_means(weights, d, mb))
    return out


def leak_verdict(primary_c_b: float, no_v3_c_b: dict, het: dict) -> dict:
    """C1's leak verdict (module docstring)."""
    excluding = [k for k, h in het.items() if not h["lo95"] <= 0 <= h["hi95"]]
    outside = not no_v3_c_b["lo95"] <= primary_c_b <= no_v3_c_b["hi95"]
    return {"verdict": "not decidable" if excluding or outside else "decidable",
            "heterogeneity_excluding_0": excluding, "no_v3_interval_excludes_primary_point": bool(outside),
            "note": "these controls cannot resolve δ"}


def read_out(prepared: Prepared, y: np.ndarray, log: Callable[[str], None] | None = None) -> dict:
    """The full read-out (module docstring) on outcomes `y` (in the primary's shot order). Returns {"summary"
    (JSON-able), "probabilities" (block -> DataFrame, no outcome), "curves", "reliability"}. Runs inside
    `no_statsbomb_reads()`."""
    y = np.asarray(y, int)
    if len(y) != len(prepared.shot_id):
        raise ValueError("the outcomes are not the population's length")
    rate = float(y.mean())
    summary, probabilities, curves, rel = {"blocks": {}, "goal_rate": rate}, {}, [], []
    with no_statsbomb_reads():
        primary_p = None
        for name, block in prepared.blocks.items():
            yb = y[block.rows]
            p, chosen = fit_block(block, yb)
            if name == "primary":
                primary_p = p
            elif name == "secondary":
                p = {"A": primary_p["A"]} | {n: q for n, q in p.items() if n != "A"}   # A: same z, same fit
            read = _read_block(name, yb, p, block.weights, rate)
            curves.append(read.pop("_curves"))
            rel.append(read.pop("_reliability"))
            summary["blocks"][name] = read | {
                "status": "primary" if name == "primary" else "descriptive", "shots": len(yb), "goals": int(yb.sum()),
                "matches": int(len(np.unique(block.games))), "dropped": block.dropped, "chosen_c": chosen,
                "grid_edge": {n: grid_edges(c) for n, c in chosen.items()}}
            probabilities[name] = pd.DataFrame({"shot_id": prepared.shot_id[block.rows], "game": block.games,
                                                "fold": block.folds} | {f"p_{n}": q for n, q in p.items()})
            if log:
                log(f"{name}: " + ", ".join(f"{c} {v['log_loss']['delta']:+.5f}" for c, v in read["contrasts"].items()
                                            if c in CONTRASTS))
        weights = prepared.blocks["primary"].weights
        for name, mask in prepared.masks.items():
            p = {n: q[mask] for n, q in primary_p.items()}
            read = _read_block(f"restricted:{name}", y[mask], p, weights[:, mask], rate)
            curves.append(read.pop("_curves"))
            rel.append(read.pop("_reliability"))
            summary["blocks"][f"restricted:{name}"] = read | {"status": "leak control" if name == "no_v3" else "descriptive",
                                                             "shots": int(mask.sum()), "goals": int(y[mask].sum())}
        primary = summary["blocks"]["primary"]["contrasts"]
        summary["decision"] = decision(primary["C-B"]["log_loss"], primary["C-A"]["log_loss"],
                                       summary["blocks"]["primary"]["holm_superiority"])
        if prepared.masks:
            summary["heterogeneity"] = heterogeneity(y, primary_p, weights, prepared.masks)
            summary["leak"] = leak_verdict(primary["C-B"]["log_loss"]["delta"],
                                           summary["blocks"]["restricted:no_v3"]["contrasts"]["C-B"]["log_loss"],
                                           summary["heterogeneity"])
    return {"summary": summary, "probabilities": probabilities, "curves": pd.concat(curves, ignore_index=True),
            "reliability": pd.concat(rel, ignore_index=True)}


def dry_run_check(summary: dict) -> dict:
    """The permuted-label diagnostic (not the validation): no Holm-rejected superiority test, B-A's two-sided p."""
    primary = summary["blocks"]["primary"]
    rejected = [c for c, r in primary["holm_superiority"].items() if r]
    b_a = primary["contrasts"]["B-A"]["log_loss"]
    return {"no_holm_rejection": not rejected, "holm_rejected": rejected, "b_minus_a_p_two_sided": b_a["p_two_sided"],
            "b_minus_a_95_excludes_0": bool(not b_a["lo95"] <= 0 <= b_a["hi95"]), "decision": summary["decision"]}


# ------------------------------------------------------------------ protocol
def protocol(inputs: Inputs) -> dict:
    """The frozen part of the protocol: arms with their exact design columns and free columns per block, the model,
    the folds, the read-out, decision and leak rules, TODO sections for later tickets and the disclosures (written by
    `scripts/xgdino_ablation.py --protocol`)."""
    table = inputs.primary.table
    tables = {"primary": (table, (*CONFIRMATORY_ARMS, *DESCRIPTIVE_ARMS))}
    if inputs.secondary is not None:
        tables["secondary"] = (inputs.secondary.table, SECONDARY_ARMS)
    tables |= {f"refit:{k}": (table[m], CONFIRMATORY_ARMS) for k, m in inputs.subsets.items()}
    arms = {}
    for block, (t, names) in tables.items():
        found, dropped = arm_set(t, names)
        arms[block] = {n: {"status": a.status, "columns": a.columns, "spline": list(a.spline), "df": a.df,
                           "unpenalised": sorted(model_for(a).free), "dropped_zero_variance": dropped[n]}
                       for n, a in found.items()}
    return {
        "population": {"primary": "stack_location.inputs.build_population(location.FEET_T0, e3=cache/xgd_full4_feet_t0/E3)",
                       "secondary": "stack.inputs.build_population(e3=cache/xgd_full3/E3): old-E3 geometry, descriptive",
                       "expected": EXPECTED, "expected_goals": EXPECTED_GOALS},
        "features": {"geometry_allowlist": list(GEOMETRY), "quality_descriptive": list(QUALITY),
                     "excluded_from_confirmatory": "SoccerNet flags and every quality covariate (ball source, "
                                                   "shot_from_ball, shooter_by_net, detection flags, counts, keypoint "
                                                   "confidence, sanitation indicators)"},
        "arms": arms,
        "model": {"class": "L2 logistic regression on the standardised design, Newton (xgpoc.stack.fit.logistic_fit)",
                  "candidate": CANDIDATE, "spline": list(SPLINE), "df": DF, "c_grid": list(C_GRID),
                  "penalty": "lambda = 1/C on every design column except the one named z_g_combo (xgpoc.dino.ablation."
                             "NamedModel); B_DA and B have every column penalised",
                  "inner_cv": "C chosen inside each outer training fold over the other host folds, summed inner held-out "
                              "log-loss, ties to the smaller C (xgpoc.stack.fit.chosen_c); arms without a penalised "
                              "column (A) need none; a grid-edge flag per fold (smallest or largest C)",
                  "recalibration": "none beyond each arm's own per-outer-fold logistic fit: A is a Platt fit on z per "
                                   "outer fold; B and C get no extra Platt layer; calibration slope and intercept reported",
                  "folds": "the host folds of cache/folds_xgv_home5.csv (5), shared by every arm and block"},
        "read_out": {"contrasts": {k: list(v) for k, v in CONTRASTS.items()},
                     "descriptive_contrasts": {k: list(v) for k, v in DESCRIPTIVE_CONTRASTS.items()},
                     "metrics": ["delta log-loss (p clipped 1e-6)", "delta Brier", "delta AUC",
                                 "delta log-loss among goals and among non-goals",
                                 "delta log-loss standardised to the primary goal rate"],
                     "per_arm": ["log-loss", "Brier", "AUC", "calibration slope and intercept",
                                 f"{RELIABILITY_BINS}-bin equal-count reliability table"],
                     "bootstrap": {"draws": ss.N_BOOT, "seed": ss.BOOT_SEED, "unit": "match",
                                   "definition": "xgpoc.detector.testrule.match_weights"},
                     "p_one_sided": "(1 + #{d >= 0}) / (B + 1): superiority of the first arm",
                     "p_two_sided": "min(1, 2 min(1 + #{d >= 0}, 1 + #{d <= 0}) / (B + 1))",
                     "holm": {"family": list(SUPERIORITY), "test": "one-sided superiority", "alpha": HOLM_ALPHA},
                     "tost": {"contrasts": list(EQUIVALENCE), "delta": DELTA, "alpha_each_side": TOST_ALPHA,
                              "rule": "P(d <= -delta) <= alpha and P(d >= delta) <= alpha over the draws "
                                      "(xgpoc.vendor.ladder.equivalence_curve)",
                              "curve": "margins 0 to 0.01 step 1e-4, smallest equivalent margin reported"}},
        "decision_table": {
            "C-B": ["'DINOv2 adds' if the one-sided superiority test is Holm-rejected and the one-sided 95% upper "
                    "bound < 0",
                    f"else 'equivalent for this stacker at δ = {DELTA}' if the 90% interval lies within ±{DELTA}",
                    f"else 'inconclusive at δ = {DELTA}; smallest margin shown = X'"],
            "C-A": "'geometry adds over the image' if Holm-rejected (one-sided)",
            "B-A": "two-sided, descriptive"},
        "leak_controls": {
            "estimand": "the primary out-of-fold predictions restricted to the subsets, no refit, the primary draws",
            "heterogeneity": {k: list(v) for k, v in HETEROGENEITY.items()},
            "verdict": "'not decidable' if a heterogeneity 95% interval excludes 0 or the restricted no_v3 C-B 95% "
                       "interval excludes the primary point estimate; else 'decidable'",
            "unseen": "descriptive (outcome-selected): goal / non-goal split and the sensitivity standardised to the "
                      "primary goal rate",
            "refit": "each subset refitted on its own (as xgs_detector_leak.py), descriptive",
            "limits": "these controls cannot resolve δ",
            "subsets": {"no_v3": "matches without any soccernet_v3 image in the detector pool index",
                        "unseen": "results/xgs_stack_dd2c_leak/seen.csv seen = false (v3 pool image within 3 s)"}},
        "validation": {"controls": "xgpoc.dino.controls: clustered label-free simulations (null, image-specific plant), "
                                   "run before the freeze",
                       "expected_c_minus_b_half_width_90": TODO},
        "dry_run": {"labels": "permuted within matches (xgpoc.compare.stats.permute_within_matches, seed 0)",
                    "role": "diagnostic only (C1)"},
        "probes": {"rules": TODO, "encodes_thresholds": TODO},
        "agreement": {"tolerances": TODO},
        "statsbomb": "never read: every input path and the read-out run inside xgpoc.detector.guard.no_statsbomb_reads",
        "disclosures": list(DISCLOSURES),
    }
