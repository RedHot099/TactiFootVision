"""The pre-registered protocol of xg-dino-info (ticket 11; spec-info §Decisions C3 and "Notes after ticket 09", which
win over the earlier sections): redundancy arms, the C3.2 contract, subgroups, masking and probe families.

Population and model: the xg-dino primary block (`ablation.load_inputs`: 7,062 training shots, 328 matches, 5 host
folds), its folds, inner CV, C grid, penalties by name (`ablation.NamedModel`, only z_g_combo free) and bootstrap (2,000
match draws, seed 0). No StatsBomb read (`no_statsbomb_reads`); caption and GPT labels never enter a design
(`labels.assert_not_design`).

Attribute blocks (`BLOCKS`; columns of `results/xgdino_info_attributes/attributes.parquet`, ticket 09)
    content (C3.1)  crowding, shooter, ball_t0, ball_m400, motion, flags (sn_direct_fk -3000..+300 ms, sn_indirect_fk_3s,
                    sn_corner_20s) = the Holm family `FAMILY`
    Xc              the joint content block (the union of the family): the single primary test
    flags_le0       flags with sn_direct_fk_le0 (-3000..0 ms) instead of sn_direct_fk: sensitivity (C3.3)
    Xq              the quality-like block (framing, fallback-filled ball flags): descriptive, "may carry broadcast /
                    annotation artefacts"
    Q               xg-dino's BQ / CQ (B / C + its quality covariates): the known baseline, repeated beside every read-out
    A column with missing values gets an indicator `<name>_missing@<offset>` (`indicator`); identical indicators inside
    a block are kept once. Missing values are filled with the training mean (`fit.Preprocessor`), as every arm.
Arms (`arm_set`): B and C (and BQ, CQ) are xg-dino's arms with its geometry-allowlist assert (`ablation.arm_set`);
    `B+X` = B's columns + X, `C+X` = C's columns + X, dist@0 as the 3-df spline, every column but z_g_combo penalised.
    Every input of a new arm is z_g_combo, an xg-dino geometry column or on `XA_ALLOWLIST` (assert), and no input is a
    caption or GPT label.
Redundancy contract (C3.2; `vectors`, `read_block`, `verdict`), per bootstrap draw on the same 2,000 match draws:
    G = LL_B - LL_C, G_X = LL_{B+X} - LL_{C+X}, R_X = G - G_X (attenuation), gain_X = LL_{B+X} - LL_B. Each is a mean
    of per-shot log-loss combinations, so every statistic, subset and heterogeneity contrast comes from the same draws.
        explains most      gain_X Holm-rejected in its family (one-sided p = (1 + #{draws >= 0}) / (B + 1)) and its
                           one-sided 95% upper bound < 0, and the one-sided 95% lower bound (5th percentile) of
                           R_X - 0.5 G > 0
        explains part      the same with R_X - 0.25 G
        does not explain   the one-sided 95% upper bound (95th percentile) of R_X - 0.25 G < 0
        inconclusive       otherwise; not evidence against
    Boundary gate (`boundary_gate`, a departure in C2's pattern, ticket 11): the controls showed R_X biased low out of
    fold, so "does not explain" is available only if its false rate at the planted R = 0.25 G (Xc) is <= 0.05; else it
    reads "inconclusive" (the raw verdict beside).
    Holm: over `FAMILY` (alpha 0.05); Xc alone at alpha 0.05; flags_le0 by re-running the family's Holm with it in place
    of flags; Xq and Q unadjusted, descriptive. R_X / G reported unclipped (point and the draws' 2.5 / 97.5
    percentiles), flagged as reportable only when G's 95% interval excludes 0, with the count of draws with G <= 0.
    C - (B+X) with C1's wording (adds / equivalent at δ = 0.002 / inconclusive, smallest margin); C+X - C descriptive.
    Wording: "predictive redundancy with DINOv2", never causal identification.
Leak estimands (C1 table, `leak`): every statistic restricted to no_v3 and unseen from the primary out-of-fold
    probabilities and draws (no refit), unseen also standardised to the primary goal rate; heterogeneity (seen -
    unseen, v3 - no_v3) of R_X, gain_X and C - (B+X); per statistic "not decidable" when a heterogeneity 95% interval
    excludes 0 or the no_v3 95% interval excludes the primary point (`ablation.leak_verdict`).
Subgroups (C3.6; `subgroup_keys`, `subgroups`) on the per-shot out-of-fold d = ℓ_C - ℓ_B of xg-dino's B and C (the
    run refits them and must replay the frozen results/xgdino_ablation/real OOF on the real labels):
    confirmatory (Holm over the two, two-sided): header_gpt head vs foot on the 1,000-shot GPT sample (unclear
    missing) weighted by the stratum weights; set piece (any `FLAGS`) vs open play. Descriptive: dist@0, box_players@0
    and px_per_m@0 top vs bottom tercile (boundaries `tercile_boundaries`, frozen in the protocol: bottom x <= q1/3, top
    x > q2/3), header_caption header vs rest, ball_own@0, gk_detected at t0, |t0 timing error| band (C2 bands and merge),
    cut_between_anchor at -400 ms. Per side: shots, goals, mean d with its 95% interval, share of the total gain
    (sum of d inside / sum over all), d among goals and non-goals and standardised to the primary goal rate. A side
    below `MIN_SUPPORT` shots is not read. Intervals are conditional on the fitted predictions.
Masking (C3.5 + notes after ticket 09; `MASKING`, `region_test`, `masking_family`, `region_overlap`): regions shooter,
    goalkeeper, ball, shooter+ball (the union of the two stored masks, present when both are), goal (test on its
    placebo-feasible subset, labelled); players and background descriptive (no feasible placebo). Fills mean (primary)
    and blur; estimands refit (C on the masked z, geometry fixed; primary) and frozen (the fitted C fold models on the
    masked z, `fold_fits` / `frozen_predict`). Test per region: degradation ΔLL(masked - unmasked) and ΔLL(masked -
    placebo, the per-shot log-loss averaged over the 3 placebo draws), one-sided p = (1 + #{draws <= 0}) / (B + 1)
    each, p = max of the two, Holm over `MASK_FAMILY`; positive = "sensitivity to the intervention". Fills disagree
    on the Holm decision -> inconclusive. Keep-only variants descriptive.
Probes (C3.7; `probe_targets`): every Xa column, header_gpt (primary label, stratum-weighted), header_caption
    (secondary: header 1, foot 0, unknown missing), set_piece, and z_C - z_B (frozen OOF logits); C2 thresholds,
    nested layer selection, BH (q 0.05) within each family.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.bundle import sha256_file
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.detector.testrule import shot_log_loss
from xgpoc.dino import ablation as ab
from xgpoc.dino import attributes as at
from xgpoc.dino import labels as lb
from xgpoc.dino import regions as rg
from xgpoc.eval.calibration import EPS
from xgpoc.stack import fit as sf
from xgpoc.stack import provenance as sp
from xgpoc.stack import stats as ss

EXPERIMENT = ab.EXPERIMENT
ATTRIBUTES = EXPERIMENT / "results/xgdino_info_attributes"
LABELS = EXPERIMENT / "results/xgdino_info_labels"
FROZEN_OOF = EXPERIMENT / "results/xgdino_ablation/real/probabilities_primary.csv"
OFFSET_FRAMES = EXPERIMENT / "cache/xgc/offset_frames_v1/frames.csv"
INFO = Path("/mnt/storage/tactifoot_research/xg_dino/info")
REGION_MASKS = INFO / "region_masks.parquet"
ABLATION_PROTOCOL_SHA256 = "fd142d152a1bc546b023a5f3e3e199888b967e5b2296393c12ca40a4bf75b6b0"

Z = ab.Z
FLAGS = ("sn_direct_fk", "sn_indirect_fk_3s", "sn_corner_20s")
FLAGS_LE0 = ("sn_direct_fk_le0", "sn_indirect_fk_3s", "sn_corner_20s")
FAMILY = ("crowding", "shooter", "ball_t0", "ball_m400", "motion", "flags")
PRIMARY = "Xc"
VALUES = {b: tuple(at.BLOCKS[b]) for b in FAMILY if b != "flags"} | {"flags": FLAGS}
VALUES |= {"Xc": tuple(c for b in FAMILY for c in VALUES[b]), "flags_le0": FLAGS_LE0, "Xq": tuple(at.BLOCKS["quality"])}
BLOCKS = ("Xc", *FAMILY, "flags_le0", "Xq")
ROLES = {"Xc": "primary (single test)", **dict.fromkeys(FAMILY, "content family (Holm)"),
         "flags_le0": "sensitivity of flags (<= 0 ms direct free-kick window)",
         "Xq": "descriptive: quality-like, may carry broadcast/annotation artefacts",
         "Q": "known baseline (xg-dino BQ / CQ), descriptive"}
BASE_ARMS = ("B", "C", "BQ", "CQ")
ALPHA = 0.05
MOST, PART = 0.5, 0.25
VERDICTS = ("explains most", "explains part", "does not explain", "inconclusive")
WORDING = "predictive redundancy with DINOv2, never causal identification"
STATISTICS = ("G", "G_X", "R", "gain", "most", "part", "C-BX", "CX-C")
LEAK_STATISTICS = ("R", "gain", "C-BX")


def xa(column: str) -> str:
    """The design name of an attribute column: `xa_` + its name (ticket 09's ball_own@0 and shot_from_ball@<o> mean
    other things than the primary table's E3 columns of the same name)."""
    return f"xa_{column}"


def indicator(column: str) -> str:
    name, _, offset = column.partition("@")
    return f"{name}_missing@{offset}" if offset else f"{name}_missing"


XA_ALLOWLIST = frozenset(xa(c) for c in at.COLUMNS) | frozenset(indicator(xa(c)) for c in at.COLUMNS)


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class Inputs:
    """xg-dino's inputs, the attribute table aligned to the primary shots and the hashes of the info inputs."""

    ablation: ab.Inputs
    attributes: pd.DataFrame
    hashes: dict


def load_inputs(attributes: Path = ATTRIBUTES / "attributes.parquet") -> Inputs:
    """xg-dino's frozen inputs and the ticket 09 attribute table (label-free; StatsBomb and labels guarded)."""
    base = ab.load_inputs()
    with at.no_label_reads():
        table = pd.read_parquet(attributes)
    hashes = {"ablation": base.hashes, "attributes": {"path": str(attributes.relative_to(EXPERIMENT)),
                                                      "sha256": sha256_file(attributes)},
              "attributes_complete": sha256_file(attributes.parent / "complete.json")}
    return Inputs(base, table, hashes)


def design_table(table: pd.DataFrame, attributes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """The primary table with the Xa columns (`xa`) and their indicators, and every block's design columns (module
    docstring)."""
    if not attributes.index.equals(table.index):
        raise ValueError("the attribute table is not the primary population in its order")
    lb.assert_not_design(attributes.columns.drop("motion_status", errors="ignore"))
    added = {xa(c): attributes[c].astype(float) for c in at.COLUMNS}
    added |= {indicator(xa(c)): attributes[c].isna().astype(float) for c in at.COLUMNS if attributes[c].isna().any()}
    out = pd.concat([table, pd.DataFrame(added, index=table.index)], axis=1)
    columns = {}
    for block in BLOCKS:
        values = [xa(c) for c in VALUES[block]]
        found, seen = [], []
        for c in values:
            if indicator(c) in out.columns:
                v = out[indicator(c)].to_numpy()
                if not any(np.array_equal(v, s) for s in seen):
                    seen.append(v)
                    found.append(indicator(c))
        columns[block] = values + found
    return out, columns


def name_collisions(table: pd.DataFrame, attributes: pd.DataFrame) -> dict[str, float]:
    """Share of shots on which a column name of both tables holds the same value (both missing counts as same)."""
    out = {}
    for c in sorted(set(table.columns) & set(at.COLUMNS)):
        x, y = table[c].to_numpy(float), attributes[c].to_numpy(float)
        out[c] = float(np.mean((x == y) | (np.isnan(x) & np.isnan(y))))
    return out


# ------------------------------------------------------------------ arms
def check_inputs(arm: sf.Arm) -> None:
    """Every input of a new arm is z, a geometry column or on the Xa allowlist; none is a caption / GPT label."""
    lb.assert_not_design(arm.inputs)
    if bad := [c for c in arm.inputs if c != Z and c not in ab.GEOMETRY and c not in XA_ALLOWLIST]:
        raise ValueError(f"arm {arm.name} holds columns outside the geometry and Xa allowlists: {bad}")


def arm_set(table: pd.DataFrame, columns: dict[str, list[str]], blocks=BLOCKS) -> tuple[dict[str, sf.Arm], dict]:
    """xg-dino's B, C, BQ, CQ (`ablation.arm_set`, geometry assert) and B+X, C+X for every block of `blocks`, after
    the zero-variance rule; the dropped columns per arm."""
    arms, dropped = ab.arm_set(table, BASE_ARMS)
    geometry = [c for c in ab.GEOMETRY if c not in ab.SPLINE]
    for block in blocks:
        for base, head in (("B", ()), ("C", (Z,))):
            name = f"{base}+{block}"
            linear = [c for c in sf.unique((*head, *geometry, *columns[block])) if c not in ab.SPLINE]
            dropped[name] = sf.constant_columns(table, [*linear, *ab.SPLINE])
            arm = sf.Arm(name, ROLES[block], tuple(c for c in linear if c not in dropped[name]),
                         tuple(c for c in ab.SPLINE if c not in dropped[name]), ab.DF)
            sf.check_no_constant(table, arm)
            sp.check_names(arm.columns)
            check_inputs(arm)
            arms[name] = arm
    return arms, dropped


@dataclass(frozen=True)
class Prepared:
    shot_id: np.ndarray
    block: ab.Block
    masks: dict[str, np.ndarray]
    columns: dict[str, list[str]]


def prepare(table: pd.DataFrame, columns: dict[str, list[str]], subsets: dict[str, np.ndarray] | None = None,
            blocks=BLOCKS, draws: int = ss.N_BOOT, seed: int = ss.BOOT_SEED) -> Prepared:
    """Every arm's fold designs (label-free) and the bootstrap weights, as `ablation.prepare`'s primary block."""
    arms, dropped = arm_set(table, columns, blocks)
    models = {n: ab.model_for(a) for n, a in arms.items()}
    folds, games = table.fold.to_numpy(), table.game.to_numpy()
    designs = {n: sf.fold_designs(table, folds, a, models[n].penalised and len(a.columns) > 1) for n, a in arms.items()}
    block = ab.Block("info", np.arange(len(table)), games, folds, arms, models, designs, dropped,
                     ss.bootstrap_weights(games, draws, seed))
    return Prepared(table.index.to_numpy(), block, ab.masks(subsets) if subsets else {}, columns)


# ------------------------------------------------------------------ redundancy contract
def losses(y: np.ndarray, p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {n: shot_log_loss(y, q, EPS) for n, q in p.items()}


def vectors(ell: dict[str, np.ndarray], block: str) -> dict[str, np.ndarray]:
    """Per-shot vectors whose means are the contract's statistics (module docstring); block "Q" uses BQ / CQ."""
    bx, cx = ("BQ", "CQ") if block == "Q" else (f"B+{block}", f"C+{block}")
    g, gx = ell["B"] - ell["C"], ell[bx] - ell[cx]
    r = g - gx
    return {"G": g, "G_X": gx, "R": r, "gain": ell[bx] - ell["B"], "most": r - MOST * g, "part": r - PART * g,
            "C-BX": ell["C"] - ell[bx], "CX-C": ell[cx] - ell["C"]}


def stat(d: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None, y: np.ndarray | None = None,
         goal_rate: float | None = None) -> tuple[float, np.ndarray]:
    """(point, draws) of the mean of `d` over `mask`; with `goal_rate` standardised: π mean(goals) + (1 - π) mean(non)."""
    m = np.ones(len(d), bool) if mask is None else mask
    if goal_rate is None:
        return float(d[m].mean()), ss.boot_means(weights, d, m)
    goals, non = m & (y == 1), m & (y == 0)
    return (goal_rate * d[goals].mean() + (1 - goal_rate) * d[non].mean(),
            goal_rate * ss.boot_means(weights, d, goals) + (1 - goal_rate) * ss.boot_means(weights, d, non))


def verdict(gain_rejected: bool, gain_upper: float, most_lower: float, part_lower: float, part_upper: float) -> str:
    """The C3.2 verdict (module docstring)."""
    if gain_rejected and gain_upper < 0 and most_lower > 0:
        return "explains most"
    if gain_rejected and gain_upper < 0 and part_lower > 0:
        return "explains part"
    if part_upper < 0:
        return "does not explain"
    return "inconclusive"


def c1_wording(summary: dict, tost: dict) -> str:
    if summary["upper_one_sided_95"] < 0:
        return "DINOv2 adds"
    if tost["equivalent"]:
        return f"equivalent for this stacker at δ = {ab.DELTA}"
    return f"inconclusive at δ = {ab.DELTA}; smallest margin shown = {tost['smallest_margin']:.4f}"


def read_block(v: dict[str, np.ndarray], weights: np.ndarray, mask=None, y=None, goal_rate=None) -> dict:
    """Every statistic of one block (summaries), the ratio R / G, C1's wording of C - (B+X); no verdict (needs Holm)."""
    out, draws = {}, {}
    for name in STATISTICS:
        point, b = stat(v[name], weights, mask, y, goal_rate)
        out[name], draws[name] = ab._summary(point, b), b
    g, r = draws["G"], draws["R"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r / g
    finite = ratio[np.isfinite(ratio)]
    out["ratio_R_over_G"] = {"point": out["R"]["delta"] / out["G"]["delta"] if out["G"]["delta"] else None,
                             "lo95": float(np.percentile(finite, 2.5)), "hi95": float(np.percentile(finite, 97.5)),
                             "draws_G_le_0": int(np.sum(g <= 0)),
                             "reportable": bool(not out["G"]["lo95"] <= 0 <= out["G"]["hi95"])}
    tost = ab._equivalence(draws["C-BX"])[0]
    out["C-BX"]["tost"] = tost
    out["C-BX"]["wording"] = c1_wording(out["C-BX"], tost)
    return out


def decide(reads: dict[str, dict], not_explain_available: bool = True) -> dict[str, dict]:
    """Holm and verdicts over the blocks of `reads` (module docstring): {block: {rejected, adjustment, verdict, raw
    verdict}}. Without `not_explain_available` (`boundary_gate`) a "does not explain" reads "inconclusive"."""
    p = {b: reads[b]["gain"]["p_one_sided"] for b in reads}
    rejected = {}
    if set(FAMILY) <= set(p):
        rejected |= {b: (r, "Holm over the content family") for b, r in ss.holm({b: p[b] for b in FAMILY}, ALPHA).items()}
        if "flags_le0" in p:
            family = {("flags_le0" if b == "flags" else b): p["flags_le0" if b == "flags" else b] for b in FAMILY}
            rejected["flags_le0"] = (ss.holm(family, ALPHA)["flags_le0"], "Holm over the family with flags_le0 for flags")
    for b in p:
        if b not in rejected:
            rejected[b] = (p[b] <= ALPHA, "single test" if b == PRIMARY else "unadjusted, descriptive")
    out = {}
    for b, r in reads.items():
        hit, how = rejected[b]
        raw = verdict(bool(hit), r["gain"]["upper_one_sided_95"], r["most"]["lo90"], r["part"]["lo90"],
                      r["part"]["upper_one_sided_95"])
        gated = raw == "does not explain" and not not_explain_available
        out[b] = {"gain_rejected": bool(hit), "adjustment": how, "raw_verdict": raw,
                  "verdict": "inconclusive" if gated else raw,
                  "gate": "'does not explain' unavailable: the boundary control failed (boundary_gate)" if gated else None}
    return out


BOUNDARY_GATE = 0.05


def boundary_gate(controls: dict) -> dict:
    """The pre-registered gate of "does not explain" (C2's pattern for a boundary false-decision rate): available only
    if, at the planted R = 0.25 G (plant25, Xc: the widest block, the largest out-of-fold cost of B+X), the procedure's
    rate of "does not explain" is <= `BOUNDARY_GATE`; the Xc control stands for every block."""
    rate = controls["scenarios"]["plant25"]["blocks"][PRIMARY]["rates"]["does not explain"]["rate"]
    return {"rate": rate, "limit": BOUNDARY_GATE, "available": bool(rate <= BOUNDARY_GATE),
            "rule": "'does not explain' only if the false 'does not explain' rate at the plant R = 0.25 G (Xc) <= 0.05; "
                    "else every 'does not explain' reads 'inconclusive' (the raw verdict reported beside)"}


def leak(vs: dict[str, dict[str, np.ndarray]], weights: np.ndarray, masks: dict[str, np.ndarray],
         primary: dict[str, dict]) -> dict:
    """C1's leak table per block and statistic of `LEAK_STATISTICS` (module docstring)."""
    out = {}
    for block, v in vs.items():
        out[block] = {}
        for name in LEAK_STATISTICS:
            d = v[name]
            het = {}
            for key, (a, b) in ab.HETEROGENEITY.items():
                pa, da = stat(d, weights, masks[a])
                pb, db = stat(d, weights, masks[b])
                het[key] = ab._summary(pa - pb, da - db)
            no_v3 = ab._summary(*stat(d, weights, masks["no_v3"]))
            out[block][name] = ab.leak_verdict(primary[block][name]["delta"], no_v3, het) | {"heterogeneity": het}
    return out


def redundancy(y: np.ndarray, p: dict[str, np.ndarray], weights: np.ndarray, masks: dict[str, np.ndarray] | None = None,
               blocks=(*BLOCKS, "Q"), not_explain_available: bool = True) -> dict:
    """The redundancy read-out on outcomes `y` and out-of-fold probabilities `p` (module docstring)."""
    y = np.asarray(y, int)
    ell = losses(y, p)
    present = [b for b in blocks if (b == "Q" and "BQ" in p) or f"B+{b}" in p]
    vs = {b: vectors(ell, b) for b in present}
    reads = {b: read_block(vs[b], weights) for b in present}
    decided = decide(reads, not_explain_available)
    out = {"goal_rate": float(y.mean()), "blocks": {b: reads[b] | decided[b] | {"role": ROLES[b]} for b in present},
           "wording": WORDING}
    if masks:
        rate = float(y.mean())
        estimands = {}
        for name, kw in (("no_v3", {}), ("unseen", {}), ("unseen_standardised", {"goal_rate": rate})):
            m = masks[name.split("_standardised")[0]]
            r = {b: read_block(vs[b], weights, m, y, **kw) for b in present}
            gated = decide(r, not_explain_available)
            estimands[name] = {b: r[b] | gated[b] for b in present}
        out["estimands"] = estimands
        out["leak"] = leak(vs, weights, masks, reads)
    return out


def fit_arms(prepared: Prepared, y: np.ndarray, names=None) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    with no_statsbomb_reads():
        return ab.fit_block(prepared.block, np.asarray(y, int), names)


# ------------------------------------------------------------------ subgroups (C3.6)
MIN_SUPPORT = 100
TERCILE_KEYS = {"distance": "dist@0", "crowding": "box_players@0", "framing": "px_per_m@0"}
CONFIRMATORY_SUBGROUPS = ("header_gpt", "set_piece")
TIMING_BANDS_MS = (50.0, 200.0)


def key_values(table: pd.DataFrame, attributes: pd.DataFrame, column: str) -> np.ndarray:
    """A tercile key's values: geometry from the primary table, Xa columns from the attribute table."""
    source = table if column in ab.GEOMETRY else attributes
    return source.loc[table.index, column].to_numpy(float)


def tercile_boundaries(table: pd.DataFrame, attributes: pd.DataFrame) -> dict[str, list[float]]:
    """The 1/3 and 2/3 quantiles of each tercile key over its finite values (label-free; frozen in the protocol)."""
    out = {}
    for k, c in TERCILE_KEYS.items():
        x = key_values(table, attributes, c)
        out[k] = [float(q) for q in np.quantile(x[np.isfinite(x)], [1 / 3, 2 / 3])]
    return out


def side(inside: np.ndarray, defined: np.ndarray) -> np.ndarray:
    """"in" / "out" / None (undefined) per shot."""
    return np.where(defined, np.where(inside, "in", "out"), None)


def timing_band(abs_ms: np.ndarray) -> np.ndarray:
    """C2's bands (<= 50 / 50-200 / > 200 ms) merged into their lower neighbour below `MIN_SUPPORT`; NaN -> None."""
    lo, hi = TIMING_BANDS_MS
    band = np.where(np.isnan(abs_ms), "missing", np.where(abs_ms <= lo, "le50", np.where(abs_ms <= hi, "50to200", "gt200")))
    order = ["le50", "50to200", "gt200"]
    for i in (2, 1):
        if 0 < (band == order[i]).sum() < MIN_SUPPORT:
            band[band == order[i]] = order[i - 1]
    return band


def subgroup_keys(table: pd.DataFrame, attributes: pd.DataFrame, boundaries: dict, gt0: pd.Series, gpt: pd.Series,
                  caption: pd.Series, timing_ms: pd.Series, cut: pd.Series) -> pd.DataFrame:
    """Per shot and key: "in" / "out" / None (module docstring). `gt0`: gk_detected at t0; `gpt`: header_gpt (1 head, 0
    foot, NaN unclear or not sampled); `caption`: header_caption labels; `timing_ms`: |t0 timing error|; `cut`:
    cut_between_anchor at -400 ms as 1 / 0 (NaN unknown). Index: the population's shots."""
    idx = table.index
    keys = {}
    g = gpt.reindex(idx).to_numpy(float)
    keys["header_gpt"] = side(g == 1, np.isfinite(g))
    flags = attributes.loc[idx, list(FLAGS)].to_numpy(float)
    keys["set_piece"] = side((flags == 1).any(axis=1), np.isfinite(flags).all(axis=1))
    for k, c in TERCILE_KEYS.items():
        x = key_values(table, attributes, c)
        q1, q2 = boundaries[k]
        keys[f"{k}_tercile"] = side(x > q2, (x <= q1) | (x > q2))
    cap = caption.reindex(idx)
    keys["header_caption"] = side((cap == "header").to_numpy(), cap.notna().to_numpy())
    ball = attributes.loc[idx, "ball_own@0"].to_numpy(float)
    keys["ball_detected"] = side(ball == 1, np.isfinite(ball))
    gk = gt0.reindex(idx)
    keys["gk_detected"] = side(gk.eq(True).to_numpy(), gk.notna().to_numpy())
    band = timing_band(timing_ms.reindex(idx).to_numpy(float))
    keys["timing_le50"] = side(band == "le50", band != "missing")
    c = cut.reindex(idx).to_numpy(float)
    keys["cut_between_anchor"] = side(c == 1, np.isfinite(c))
    return pd.DataFrame(keys, index=idx)


def subgroups(y: np.ndarray, d: np.ndarray, weights: np.ndarray, keys: pd.DataFrame, sample_weight: pd.Series,
              goal_rate: float) -> dict:
    """Per key: both sides' mean d, share of the total gain, by outcome and standardised, and the contrast in - out
    (two-sided p); Holm over `CONFIRMATORY_SUBGROUPS`. header_gpt is weighted by `sample_weight` (stratum weights)."""
    y = np.asarray(y, int)
    total = float(d.sum())
    out = {}
    for key in keys.columns:
        k = keys[key].to_numpy()
        sw = sample_weight.reindex(keys.index).fillna(0.0).to_numpy(float) if key == "header_gpt" else np.ones(len(d))
        w = weights * sw[None, :]
        sides, draws = {}, {}
        for s in ("in", "out"):
            m = k == s
            n = int(m.sum())
            row = {"shots": n, "goals": int(y[m].sum()), "readable": n >= MIN_SUPPORT}
            if n >= MIN_SUPPORT:
                point = float(np.average(d[m], weights=sw[m]))
                draws[s] = ss.boot_means(w, d, m)
                row |= {"mean_d": ab._summary(point, draws[s]), "share_of_total_gain": float((d[m] * sw[m]).sum() / total) if key != "header_gpt" else None}
                parts = {}
                for o, label in ((1, "goals"), (0, "non_goals")):
                    mo = m & (y == o)
                    parts[label] = float(np.average(d[mo], weights=sw[mo])) if sw[mo].sum() > 0 else None
                row |= parts
                if None not in parts.values():
                    row["standardised"] = goal_rate * parts["goals"] + (1 - goal_rate) * parts["non_goals"]
            sides[s] = row
        entry = {"sides": sides, "status": "confirmatory" if key in CONFIRMATORY_SUBGROUPS else "descriptive",
                 "weighted": key == "header_gpt"}
        if len(draws) == 2:
            entry["contrast_in_minus_out"] = ab._summary(sides["in"]["mean_d"]["delta"] - sides["out"]["mean_d"]["delta"],
                                                         draws["in"] - draws["out"])
        out[key] = entry
    p = {k: out[k]["contrast_in_minus_out"]["p_two_sided"] for k in CONFIRMATORY_SUBGROUPS
         if "contrast_in_minus_out" in out.get(k, {})}
    holm = ss.holm(p, ALPHA) if p else {}
    for k in CONFIRMATORY_SUBGROUPS:
        if k in out:
            out[k]["holm_rejected"] = bool(holm.get(k, False))
    return {"keys": out, "note": "intervals conditional on the fitted predictions; d = ℓ_C - ℓ_B (negative: C better)"}


# ------------------------------------------------------------------ masking (C3.5 + notes after ticket 09)
MASK_FAMILY = ("shooter", "goalkeeper", "ball", "shooter+ball", "goal")
MASK_DESCRIPTIVE = ("players", "background")
KEEP_ONLY = {"keep_shooter_ball": ("shooter", "ball"), "keep_persons": ("players",), "keep_goal": ("goal",)}
FILLS = ("mean", "blur")
ESTIMANDS = ("refit", "frozen")
N_PLACEBO = rg.N_PLACEBO
MASKING = {
    "family": list(MASK_FAMILY), "descriptive": list(MASK_DESCRIPTIVE), "keep_only": {k: list(v) for k, v in KEEP_ONLY.items()},
    "fills": {"mean": "the frame's mean colour (primary)", "blur": "a blur fill (agreement check)",
              "disagreement": "the two fills differ in the Holm decision -> inconclusive"},
    "estimands": {"refit": "C refitted (same folds, inner CV, penalties) with z_g_combo replaced by the masked z, geometry "
                           "fixed: information left (primary)",
                  "frozen": "the fitted C fold models applied to the masked z: reliance of the deployed model"},
    "test": "per region: d1 = ΔLL(masked - unmasked), d2 = ΔLL(masked - placebo); one-sided p_i = (1 + #{draws <= 0}) / "
            "(B + 1) (degradation > 0); p = max(p1, p2); Holm over the family at 0.05; positive = 'sensitivity to the "
            "intervention', not proof of what DINOv2 uses",
    "placebo": f"the region's patch mask translated (ticket 09: y-band ±{rg.Y_BAND} patch row, on the pitch, off persons, "
               f"ball and goal), {N_PLACEBO} seeded draws; the per-shot log-loss is averaged over the draws",
    "shooter+ball": "the union of the stored shooter and ball masks, present when both are; its placebos: "
                    "regions.placebos(union, allowed, regions.rng_for(shot_id, 'shooter+ball')) with ticket 09's allowed "
                    "patches (replayed in ticket 14)",
    "population": "shots with the region present (primary; others keep the unmasked frame); all shots descriptive; goal "
                  "tested on its placebo-feasible subset only (labelled); a shot without a valid placebo is out of the "
                  "test population of its region",
    "gain_left": "(LL_B - LL_C_mask) / (LL_B - LL_C), point and the draws' 95% interval",
    "a_rows": "A_mask - A descriptive",
    "size_bins": "read-outs binned by region size in patches (terciles of the present shots, frozen at the read-out)",
    "overlap": "the pairwise region overlap matrix (`region_overlap`): ball vs shooter and goal vs goalkeeper are not "
               "separable by remove-masks alone",
}


def region_test(ell_unmasked: np.ndarray, ell_masked: np.ndarray, ell_placebo: np.ndarray, weights: np.ndarray,
                mask: np.ndarray) -> dict:
    """The per-region test (module docstring) over the shots of `mask`."""
    out = {}
    for name, d in (("vs_unmasked", ell_masked - ell_unmasked), ("vs_placebo", ell_masked - ell_placebo)):
        point, draws = stat(d, weights, mask)
        s = ab._summary(point, draws)
        finite = draws[np.isfinite(draws)]
        s["p_degradation"] = float((1 + np.sum(finite <= 0)) / (len(finite) + 1))
        out[name] = s
    out["p"] = max(out["vs_unmasked"]["p_degradation"], out["vs_placebo"]["p_degradation"])
    out["shots"] = int(mask.sum())
    return out


def masking_family(tests: dict[str, dict]) -> dict[str, bool]:
    return ss.holm({r: tests[r]["p"] for r in MASK_FAMILY if r in tests}, ALPHA)


def fill_agreement(by_fill: dict[str, dict[str, bool]]) -> dict[str, str]:
    """Per family region: "sensitive" / "not detected" when both fills agree, else "inconclusive (fills disagree)"."""
    out = {}
    for r in MASK_FAMILY:
        decisions = {by_fill[f].get(r) for f in FILLS if f in by_fill}
        if len(decisions) > 1:
            out[r] = "inconclusive (fills disagree)"
        else:
            out[r] = "sensitive to the intervention" if decisions.pop() else "no detectable sensitivity"
    return out


def gain_left(ell: dict[str, np.ndarray], masked: str, weights: np.ndarray) -> dict:
    """(LL_B - LL_{masked C}) / (LL_B - LL_C), point and the draws' 95% interval."""
    pn, dn = stat(ell["B"] - ell[masked], weights)
    pd_, dd = stat(ell["B"] - ell["C"], weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dn / dd
    r = r[np.isfinite(r)]
    return {"point": pn / pd_, "lo95": float(np.percentile(r, 2.5)), "hi95": float(np.percentile(r, 97.5))}


@dataclass(frozen=True)
class FoldFit:
    fold: int
    test: np.ndarray
    pre: sf.Preprocessor
    beta: np.ndarray
    c: float | None


def fold_fits(table: pd.DataFrame, arm: sf.Arm, model: sf.Model, y: np.ndarray) -> list[FoldFit]:
    """The arm's fitted fold models (C chosen as `fit.out_of_fold`), kept for `frozen_predict`."""
    folds, y = table.fold.to_numpy(), np.asarray(y, float)
    out = []
    for d in sf.fold_designs(table, folds, arm, model.penalised and len(arm.columns) > 1):
        c = sf.chosen_c(d, y, model)
        beta = sf.logistic_fit(d.x_train, y[d.train], model.penalties(d.x_train.shape[1], model.grid[-1] if c is None else c))
        out.append(FoldFit(d.fold, d.test, sf.Preprocessor.fit(arm, table.iloc[d.train]), beta, c))
    return out


def frozen_predict(fits: list[FoldFit], table: pd.DataFrame) -> np.ndarray:
    """Each fold's scored rows of `table` (e.g. with a masked z) through that fold's fitted model."""
    p = np.full(len(table), np.nan)
    for f in fits:
        p[f.test] = sf.predict(f.beta, f.pre(table.iloc[f.test]))
    return p


def refit(table: pd.DataFrame, arm: sf.Arm, model: sf.Model, y: np.ndarray) -> np.ndarray:
    """The arm refitted out of fold on `table` (e.g. with a masked z): same folds, inner CV and penalties."""
    return sf.out_of_fold(sf.fold_designs(table, table.fold.to_numpy(), arm, model.penalised and len(arm.columns) > 1),
                          np.asarray(y, float), model)


def with_z(table: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    out = table.copy()
    out[Z] = z
    return out


def region_overlap(masks: pd.DataFrame) -> pd.DataFrame:
    """Pairwise overlap of the stored t0 region masks (kind "region"): per ordered pair (a, b), shots with both present,
    the mean share of a's patches inside b and the mean Jaccard index; shooter+ball as the union."""
    regions = masks[masks.kind == "region"]
    by = {r: {s: rg.unpack(m) for s, m in zip(g.shot_id, g["mask"], strict=True)} for r, g in regions.groupby("region")}
    both = set(by.get("shooter", {})) & set(by.get("ball", {}))
    by["shooter+ball"] = {s: by["shooter"][s] | by["ball"][s] for s in both}
    rows = []
    names = [r for r in (*MASK_FAMILY, *MASK_DESCRIPTIVE) if r in by]
    for a in names:
        for b in names:
            shots = sorted(set(by[a]) & set(by[b]))
            if not shots:
                continue
            inter = np.array([(by[a][s] & by[b][s]).sum() for s in shots], float)
            size_a = np.array([by[a][s].sum() for s in shots], float)
            union = np.array([(by[a][s] | by[b][s]).sum() for s in shots], float)
            rows.append({"a": a, "b": b, "shots": len(shots), "share_of_a_in_b": float(np.mean(inter / np.maximum(size_a, 1))),
                         "jaccard": float(np.mean(inter / np.maximum(union, 1)))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ probes (C3.7)
PROBE_FAMILIES = {**{b: list(at.BLOCKS[b]) for b in ("quality", "crowding", "shooter", "ball_t0", "ball_m400", "motion")},
                  "flags": [*at.FLAG_WINDOWS, "set_piece"], "header": ["header_gpt", "header_caption"], "z_diff": ["z_C_minus_z_B"]}
PROBES = {
    "targets": PROBE_FAMILIES,
    "primary_label": "header_gpt (GPT-6.1 Sol low on the 1,000-shot sample; unclear missing); metrics weighted by the "
                     "stratum weights (primary), unweighted beside; reported by outcome",
    "secondary_label": "header_caption: header 1, foot 0, unknown missing; reported by outcome",
    "set_piece": "any of sn_direct_fk, sn_indirect_fk_3s, sn_corner_20s",
    "z_diff": "logit p_C - logit p_B of the frozen xg-dino OOF (results/xgdino_ablation/real), alone and beyond geometry "
              "(the geometry design of B vs geometry + embedding, incremental held-out R²)",
    "kind": "binary when every finite value is 0 or 1 (AUC), else continuous / count (R², MAE)",
    "encoders": "frozen DINOv2 and G-combo blocks 20-23 (five-fold-model routing, C1), offsets 0 and -400 ms",
    "rules": "C2 thresholds unchanged (xg-dino protocol fd142d15...: encodes beyond distance ΔR² >= 0.05 / ΔAUC >= 0.03 with "
             "95% lower bound > 0; readable R² >= 0.30 / AUC >= 0.70); nested layer selection; BH at q = 0.05 within "
             "each family (target x encoder x offset); the header probe's 'encoded beyond distance' is a readability "
             "statement only",
    "labels_barred": "header_gpt and header_caption are probe targets and subgroup keys only (labels.assert_not_design)",
}


def probe_targets(attributes: pd.DataFrame, gpt: pd.Series, caption: pd.Series, frozen: pd.DataFrame | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(targets per shot, one row per target: family, kind, present). `frozen`: shot_id-indexed p_B, p_C (optional)."""
    idx = attributes.index
    t = attributes[[c for f in ("quality", "crowding", "shooter", "ball_t0", "ball_m400", "motion") for c in at.BLOCKS[f]]
                   + list(at.FLAG_WINDOWS)].astype(float)
    flags = attributes[list(FLAGS)].to_numpy(float)
    t["set_piece"] = np.where(np.isfinite(flags).all(axis=1), (flags == 1).any(axis=1).astype(float), np.nan)
    t["header_gpt"] = gpt.reindex(idx).astype(float)
    cap = caption.reindex(idx)
    t["header_caption"] = np.where(cap == "header", 1.0, np.where(cap == "foot", 0.0, np.nan))
    if frozen is not None:
        p = frozen.reindex(idx)
        logit = lambda q: np.log(q) - np.log1p(-q)  # noqa: E731
        t["z_C_minus_z_B"] = logit(p.p_C.to_numpy(float)) - logit(p.p_B.to_numpy(float))
    rows = []
    for family, names in PROBE_FAMILIES.items():
        for name in names:
            if name not in t:
                continue
            v = t[name].dropna()
            rows.append({"target": name, "family": family, "kind": "binary" if set(v.unique()) <= {0.0, 1.0} else "continuous",
                         "present": int(len(v)), "source": {"header_gpt": "gpt", "header_caption": "caption"}.get(name, "automatic")})
    return t, pd.DataFrame(rows)


# ------------------------------------------------------------------ disclosures
DISCLOSURES = (
    "Development comparisons on the 328 training matches, out of fold; not an independent test (172 matches spent).",
    "C6 optimism (C1): G-combo's epoch and temperature were chosen on each scored fold, so C and C+X carry optimism that "
    "B and B+X do not. G = LL_B - LL_C and C - (B+X) are tilted toward 'DINOv2 adds'; R_X = G - G_X compares two "
    "differences that both carry it, so the tilt partly cancels but is not removed.",
    "Verdicts state predictive redundancy with DINOv2, never causal identification of what DINOv2 uses.",
    "The quality-like block Xq and xg-dino's Q may carry broadcast / annotation artefacts (C1 named Q an outcome-"
    "correlated channel); their read-outs are descriptive. BQ - B = -0.0065 (67% of C - B) is the known baseline.",
    "SoccerNet flags are retrospective context; sn_direct_fk reaches +300 ms; the <= 0 ms window is the sensitivity "
    "(C3.3). t0 is not certified pre-contact: if only ball_t0 / motion close the gap the wording is 'ball flight / "
    "post-contact', not 'body part'.",
    "Motion is present on 39% of shots (identity rule: mutual nearest within 4 m); missing values are mean-filled with an "
    "indicator, so the motion block also carries 'motion measurable'.",
    "The leak controls (no_v3 1,028 shots, unseen 3,249 shots with 5% goals) cannot resolve small attenuations.",
    "Header labels (caption, GPT) are noisy measurements; caption precision against GPT differs by outcome (goal 0.82, "
    "no goal 0.62), so caption-based reads are outcome-dependent; GPT is primary.",
    "The control truths are label-free (z's projection on the geometry and Xc designs), with the planted G set to "
    "xg-dino's observed C - B (0.0097); the null puts the whole image signal outside span(B, Xc), so every content block "
    "is null at once.",
    "Refitted-subset leak estimates are not computed (C1: descriptive only); the restricted estimands carry the verdict.",
    "The controls show the recovered R_X biased low out of fold (B+X recovers less of the image signal than its oracle, "
    "most for the wide Xc block); 'does not explain' is therefore gated by its boundary control (boundary_gate) and "
    "'explains part / most' are conservative.",
)


# ------------------------------------------------------------------ protocol
def protocol(inputs: Inputs, extras: dict) -> dict:
    """The frozen part (written by `scripts/xgdino_info_protocol.py --protocol`): blocks, arms with their exact design
    columns, the contract, Holm families, leak, subgroups with frozen boundaries, masking, probes, disclosures, and
    `extras` (validation record, region overlap digest, label-free counts)."""
    table, columns = design_table(inputs.ablation.primary.table, inputs.attributes)
    arms, dropped = arm_set(table, columns)
    return {
        "population": "xg-dino primary block (ablation.load_inputs; protocol " + ABLATION_PROTOCOL_SHA256 + ")",
        "blocks": {b: {"role": ROLES[b], "values": list(VALUES[b]), "design": columns[b]} for b in BLOCKS} |
                  {"Q": {"role": ROLES["Q"], "values": list(ab.QUALITY), "design": list(ab.QUALITY)}},
        "xa_allowlist": sorted(XA_ALLOWLIST),
        "arms": {n: {"columns": a.columns, "spline": list(a.spline), "df": a.df, "unpenalised": sorted(ab.model_for(a).free),
                     "dropped_zero_variance": dropped[n], "role": a.status} for n, a in arms.items()},
        "model": {"as": "xg-dino (ablation.model_for, C_GRID, inner CV over the other host folds, no extra Platt layer)",
                  "c_grid": list(ab.C_GRID), "bootstrap": {"draws": ss.N_BOOT, "seed": ss.BOOT_SEED, "unit": "match"}},
        "contract": {"statistics": {"G": "LL_B - LL_C", "G_X": "LL_{B+X} - LL_{C+X}", "R": "G - G_X", "gain": "LL_{B+X} - LL_B"},
                     "verdicts": {"explains most": f"gain Holm-rejected and its one-sided 95% upper bound < 0 and the 5th "
                                                   f"percentile of R - {MOST} G > 0",
                                  "explains part": f"the same with R - {PART} G",
                                  "does not explain": f"95th percentile of R - {PART} G < 0",
                                  "inconclusive": "otherwise; not evidence against"},
                     "holm": {"family": list(FAMILY), "alpha": ALPHA, "primary": f"{PRIMARY} single test at {ALPHA}",
                              "flags_le0": "the family's Holm with flags_le0 in place of flags",
                              "descriptive": "Xq and Q unadjusted"},
                     "boundary_gate": "'does not explain' only if the false 'does not explain' rate at the planted "
                                      "R = 0.25 G (Xc controls) <= 0.05; else it reads 'inconclusive' (raw verdict "
                                      "beside); evaluated from the validation record below",
                     "ratio": "R / G unclipped, the draws' 2.5 / 97.5 percentiles, reportable only when G's 95% interval "
                              "excludes 0; draws with G <= 0 counted",
                     "c_minus_bx": "C1 wording: 'DINOv2 adds' if the one-sided 95% upper bound < 0; else equivalent at δ = "
                                   f"{ab.DELTA} (90% interval inside ±δ); else inconclusive, smallest margin shown",
                     "wording": WORDING},
        "leak": {"estimands": ["no_v3 (restricted)", "unseen (restricted)", "unseen standardised to the primary goal rate"],
                 "statistics": list(LEAK_STATISTICS), "heterogeneity": {k: list(v) for k, v in ab.HETEROGENEITY.items()},
                 "verdict": "'not decidable' if a heterogeneity 95% interval excludes 0 or the no_v3 95% interval excludes "
                            "the primary point; else 'decidable' (ablation.leak_verdict)",
                 "limits": "these controls cannot resolve small attenuations"},
        "subgroups": {"confirmatory": list(CONFIRMATORY_SUBGROUPS), "holm_alpha": ALPHA, "test": "two-sided, in - out",
                      "terciles": {k: {"column": TERCILE_KEYS[k], "boundaries": v,
                                       "rule": "bottom x <= q1/3, top x > q2/3, contrast top - bottom"}
                                   for k, v in tercile_boundaries(table, inputs.attributes).items()},
                      "descriptive": ["distance_tercile", "crowding_tercile", "framing_tercile", "header_caption",
                                      "ball_detected", "gk_detected", "timing_le50", "cut_between_anchor"],
                      "min_support": MIN_SUPPORT, "header_gpt": "head vs foot on the GPT sample, stratum-weighted",
                      "timing_bands_ms": list(TIMING_BANDS_MS),
                      "d": "per-shot ℓ_C - ℓ_B of the run's B and C, which replay results/xgdino_ablation/real on the "
                           "real labels (max |Δp| reported)"},
        "masking": MASKING, "probes": PROBES, "disclosures": list(DISCLOSURES), **extras,
    }
