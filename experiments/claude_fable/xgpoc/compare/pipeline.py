"""One run of the xg-compare analysis (ticket 02): every output of `xgpoc.compare.analysis` and
`xgpoc.compare.describe` written to one folder, in an order that keeps the scan's freeze honest.

`run` is a pure function of the shot table, its label file, the table's label-free coverage and the shared draws,
apart from the files it writes. A dry run (`dry_run`) gets labels permuted within matches (`stats.permute_within_matches`)
and leaves out what reads no label: items 8 and 9, the raw and rank sensitivities, agreement and the disagreement
list (`analysis.family`). Order:
    1. the scan, first, so no shot of the validation half enters any fit before the freeze: the discovery half,
       candidates frozen to `CANDIDATES` with their SHA-256, then the validation half, which
       `analysis.scan_validation` builds only after it checked the frozen file
    2. the primary and combined populations (main scale refitted per draw); the paper's coefficients checked
    3. the family on both populations; items 1, 13, 6b and the negative control
    4. relative valuation of every level, the multivariable fit, agreement, who is right, disagreements, raw levels
Outputs are CSV and JSON files and carry no frame. Tables built from StatsBomb data carry `NOTICE` in the summary.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.compare import analysis as an
from xgpoc.compare import describe as ds
from xgpoc.compare import protocol as pr
from xgpoc.compare import stats as st
from xgpoc.compare.population import (
    Population,
    build_population,
    json_text,
    paper_coefficients_match,
)
from xgpoc.vendor.draws import TableDraws

DRY_RUN_SEED = 0
NOTICE = "Data: StatsBomb."
CANDIDATES = "scan_candidates.json"
READINGS = [   # where C4 and the protocol's readings leave a detail open (ticket 02), fixed before the dry run
    "Dry run: the label rows (goal, on target, outcome) are permuted jointly within each match, seed 0. Items 8 and 9, the "
    "raw and rank sensitivities, agreement and the disagreement list read no label, so a label permutation cannot null them; "
    "they are left out of the dry run so that no real result is seen before the run.",
    "xG points: 100 [sigmoid(logit(p) + effect) - p] with p the level's mean main-scale vendor probability; the interval "
    "bounds are transformed with p fixed.",
    "Reading against the threshold uses the 95% interval: 'interval inside the threshold' means it excludes the threshold in "
    "the tested direction (both directions for a two-sided item).",
    "Items 1 and 13: the H4 controls are on the gap's scale (main-scale z_V on the main scale, raw on raw); the body-part and "
    "shot-type reference levels are the most frequent in the item's rows, as in the paper; the one-sided p in the owner's "
    "direction (> 0) is reported outside BH. Item 1 runs on the primary population and, as robustness, on the combined one, "
    "where StatsBomb Blocked shots appear. Item 13's raw-scale F row re-expresses the paper's H4 coefficient.",
    "The negative control compares the first with the second half; extra-time shots, if any, are left out.",
    "Relative valuation covers 8 pairs: F-V, (F-400)-V, A1-V, P-V, E-V, D-V, F-D, F-P; raw and rank scales for F-V only; "
    "post-shot levels carry F-V and (F-400)-V only.",
    "The spec's Analysis 2 multivariable regression is reported, descriptive, as g ~ 1 + z_D + the indicators of the family's "
    "gap items over the shots where all are defined; C4 makes the single-indicator adjusted effect the tested estimand.",
    "Agreement rows exist for levels with at least 20 shots; the category floor is flagged. Spearman is recomputed on the "
    "level's own ranks in every draw.",
    "Who is right: every metric is computed for every level and flagged below the label floor (15 goals, 15 non-goals); the "
    "AUC only at the label floor; post-shot levels get the AUC only.",
    "Largest disagreements: round(0.05 n) shots each way by the observed main-scale F - V gap; the tail membership is held "
    "fixed when the goal rate is bootstrapped.",
    "Rank scale: the within-population percentile 100 (midrank - 1/2) / N, recomputed in every draw.",
    "Each scan half refits its main scale on its own match folds (ladder.match_folds of the half's matches, fold seed 0).",
    "The scan runs before anything else, so no fit on the whole primary population (which contains the validation half) "
    "exists before the candidates are frozen.",
    "BH and BY run over the family items that meet their floors (the power simulation of ticket 01 does the same); an "
    "item below its floor is descriptive and leaves m below 14.",
    "Every logit effect of the family, its references, items 1, 13, 6b, the negative control, the scan and the F - V "
    "level table is also given in xG points.",
    "The dry run is a weak null for the gap items: on permuted labels the recalibration slopes collapse toward 0, so it "
    "checks that refitting in every draw keeps the p-values honest on a degenerate scale; ticket 01's power simulation "
    "checks the bootstrap on the real scale.",
]


def scan_descriptive(levels: pd.DataFrame) -> pd.DataFrame:
    """The F - D and F - P adjusted effects of every scan level on the primary population (C4: appendix, no test)."""
    scan = set(pr.scan_levels())
    keep = levels.population.eq("primary") & levels.pair.isin(["F-D", "F-P"])
    keep &= np.array([(c, lv) in scan for c, lv in zip(levels.category, levels.level, strict=True)])
    return levels[keep][["pair", "category", "level", "n_in", "matches_in", "meets_category_floor", "main_adjusted_point",
                         "main_adjusted_lo", "main_adjusted_hi", "main_unadjusted_point"]].reset_index(drop=True)


def headline(family: pd.DataFrame, validation: pd.DataFrame, who: pd.DataFrame, agreement: pd.DataFrame | None) -> dict:
    """The numbers the ticket's summary quotes."""
    out = {"family": {r.item: {"point": r.effect_point, "lo": r.effect_lo, "hi": r.effect_hi, "p": r.effect_p, "q_bh": r.q_bh, "q_by": r.q_by,
                               "reading": r.reading} for r in family.itertuples()},
           "family_bh_rejections": int(family.reject_bh.sum()), "family_by_rejections": int(family.reject_by.sum()),
           "scan_candidates": int(len(validation)), "scan_replicated": int(validation.replicated.sum()) if len(validation) else 0}
    overall = who[who.category.eq("overall") & who.population.eq("primary")]
    out["who_is_right_overall"] = {r.method: {"log_loss": r.log_loss_point, "o_over_e": r.o_over_e_point, "auc": r.auc_point,
                                              "delta_ll_vs_V": getattr(r, "delta_ll_vs_V_point", np.nan)} for r in overall.itertuples()}
    if agreement is not None:
        rows = agreement[agreement.category.eq("overall") & agreement.population.eq("primary")]
        out["agreement_overall"] = {f"{r.kind}:{r.a}-{r.b}": {"point": r.point, "lo": r.lo, "hi": r.hi} for r in rows.itertuples()}
    return out


def run(table: pd.DataFrame, labels: pd.DataFrame, coverage: pd.DataFrame, draws: TableDraws, out: Path, *, dry_run: bool,
        paper_coefficients: dict | None, context: dict, log: Callable[[str], None] = lambda _: None) -> dict:
    """Run the analysis into `out` (module docstring); return the summary. `paper_coefficients` (the paper's primary
    recalibration) are checked on real labels; `context` (hashes of the inputs) goes into the candidate file; `log`
    receives progress lines."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    label_free = not dry_run
    if dry_run:
        labels = st.permute_within_matches(labels, table.match.to_numpy(), DRY_RUN_SEED)

    def population(mask: np.ndarray, name: str, base: str) -> Population:
        return build_population(table, labels, mask, draws, name, base)

    def half(name: str) -> Population:
        return population((table.in_primary & table.scan_half.eq(name)).to_numpy(), name, "primary")

    discovery = an.scan_discovery(half("discovery"), coverage)
    discovery.to_csv(out / "scan_discovery.csv", index=False)
    candidates_sha = an.freeze_candidates(discovery, out / CANDIDATES, {**context, "dry_run": dry_run})
    log(f"scan: {int(discovery.candidate.sum())} candidates frozen, sha256 {candidates_sha}")
    validation = an.scan_validation(lambda: half("validation"), out / CANDIDATES, candidates_sha)
    validation.to_csv(out / "scan_validation.csv", index=False)
    log(f"scan: {int(validation.replicated.sum()) if len(validation) else 0} replicated")

    primary = population(table.in_primary.to_numpy(), "primary", "primary")
    combined = population(np.ones(len(table), bool), "combined", "combined")
    log(f"populations: primary {len(primary.y)}, combined {len(combined.y)}")
    match_paper = paper_coefficients_match(primary, paper_coefficients) if paper_coefficients and not dry_run else None

    family = an.family(primary, label_free)
    family.to_csv(out / "family.csv", index=False)
    an.family(combined, label_free).to_csv(out / "family_combined.csv", index=False)
    outside = {"item_1": [an.item_1(p, label_free) for p in (primary, combined)], "item_13": an.item_13(primary, label_free),
               "item_6b": [an.level_item(p, *an.OPEN_GOAL, label_free=label_free) for p in (primary, combined)],
               "negative_control": [an.level_item(p, *an.NEGATIVE_CONTROL, label_free=label_free) for p in (primary, combined)]}
    (out / "outside_family.json").write_text(json_text(outside))
    log("family and items outside it written")

    levels = pd.concat([ds.level_table(p, label_free) for p in (primary, combined)], ignore_index=True)
    levels.to_csv(out / "levels.csv", index=False)
    scan_descriptive(levels).to_csv(out / "scan_descriptive.csv", index=False)
    ds.multivariable(primary).to_csv(out / "multivariable.csv", index=False)
    who = pd.concat([ds.who_is_right(p) for p in (primary, combined)], ignore_index=True)
    who.to_csv(out / "who_is_right.csv", index=False)
    pd.concat([ds.raw_levels(p) for p in (primary, combined)], ignore_index=True).to_csv(out / "raw_levels.csv", index=False)
    agreement = None
    if label_free:
        parts = [ds.agreement(p) for p in (primary, combined)]
        agreement = pd.concat([a for a, _ in parts], ignore_index=True)
        agreement.to_csv(out / "agreement.csv", index=False)
        pd.concat([d for _, d in parts], ignore_index=True).to_csv(out / "agreement_differences.csv", index=False)
        for name, frame in zip(("shots", "tails", "profile"), ds.disagreements(primary), strict=True):
            frame.to_csv(out / f"disagreement_{name}.csv", index=False)
    log("descriptive tables written")

    summary = {"notice": NOTICE, "dry_run": dry_run, "readings": READINGS, "dry_run_seed": DRY_RUN_SEED if dry_run else None,
               "candidates_sha256": candidates_sha, "paper_coefficients_max_abs_difference": match_paper,
               "n_boot": int(len(draws.counts)), **headline(family, validation, who, agreement)}
    (out / "summary.json").write_text(json_text(summary))
    return summary
