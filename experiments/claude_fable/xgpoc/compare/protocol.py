"""The frozen design of xg-compare (council C4, 2026-10-13) as data (ticket 01).

`design` transcribes the "Decisions -> C4" section of `.scratch/xg-compare/spec.md`: notation, scale, methods,
populations, floors, effect thresholds, the pre-declared family with its sides, the items outside the family, the
scan rules, the split seed and the bootstrap. `scripts/xgc_protocol.py` adds the category dictionary with its
label-free cuts, the shot table's hashes, the label-free coverage of the family items and the expected power. It
writes the result to `research/xgc_compare_protocol.json` before any category-level number exists. Ticket 02 opens
the file with `load`, which refuses a file whose SHA-256 is not the one recorded in ticket 01.

The constants below are what the power simulation (`xgpoc.compare.power`) runs, so the simulation and the protocol
cannot drift apart.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.compare import categories as cat

PROTOCOL = Path("research/xgc_compare_protocol.json")   # relative to experiments/claude_fable
SPLIT_SEED = 20261013
HALVES = ("discovery", "validation")
N_BOOT = 2000
BOOT_SEED = 0
N_FOLDS = 5
FOLD_SEED = 0
Q_BH = 0.05
SCAN_ALPHA = 0.05
HOLM_ALPHA = 0.05
FLOORS = {"category_in": 50, "category_out": 50, "category_matches_in": 20, "label_goals": 15, "label_non_goals": 15,
          "scan_half_in": 50, "scan_half_out": 50}
THRESHOLDS = {"family_logit": 0.15, "scan_logit": 0.20, "correlation": 0.10, "log_loss": 0.005}
SIDES = ("two-sided", "greater", "less")
THRESHOLD_OF = {"gap": "family_logit", "log_loss": "log_loss", "correlation_difference": "correlation",
                "correlation_contrast": "correlation"}
KINDS = {
    "gap": "coefficient of the level indicator in the OLS fit g ~ 1 + indicator + z_D over the shots where the category "
           "is defined; g = z_F - z_V on the main scale",
    "log_loss": "coefficient of the level indicator in the OLS fit (LL(F) - LL(V)) ~ 1 + indicator + z_D, with LL the "
                "per-shot log-loss of the main-scale probabilities",
    "correlation_difference": "Spearman(F, D) - Spearman(F, V) of the main-scale logits over the population",
    "correlation_contrast": "[Spearman(F, V) - Spearman(P, V)] among the level's shots minus the same among the "
                            "reference shots",
}
METHODS = {"V": "vendor: StatsBomb xG (pred_vendor_xg)", "P": "positions: event + freeze-frame rung (pred_freeze_frame)",
           "E": "events: event rung (pred_event)", "D": "distance + angle rung (pred_dist_angle)",
           "F": "frame: retrained G-combo at the anchor (frame_logit_0)", "F-400": "the same model at -400 ms (frame_logit_m400)",
           "A1": "G-combo with detector inputs (results/xgd_test/predictions.csv, primary shots only)"}


@dataclass(frozen=True)
class Item:
    """One pre-declared family item, tested on its adjusted effect against its kind's threshold (`THRESHOLD_OF`)."""

    id: str
    label: str
    kind: str
    category: str | None
    level: str | None
    side: str
    reference: str = "the other shots where the category is defined"
    sensitivity: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.side not in SIDES:
            raise ValueError(f"item {self.id}: unknown kind or side")
        if self.category is not None and (self.category not in cat.BY_NAME or self.level not in cat.BY_NAME[self.category].levels):
            raise ValueError(f"item {self.id}: {self.level!r} is not a level of {self.category}")

    @property
    def threshold(self) -> float:
        return THRESHOLDS[THRESHOLD_OF[self.kind]]

    @property
    def injection(self) -> float:
        """The effect the power simulation plants: the threshold, signed in the tested direction (positive when
        two-sided)."""
        return -self.threshold if self.side == "less" else self.threshold


FAMILY: tuple[Item, ...] = (
    Item("2", "rebound: same team, <= 5 s after the previous shot, same StatsBomb possession", "gap", "rebound_5s", "yes", "less",
         sensitivity="rebound_3s = yes (3 s window); subtypes after a save / after a block descriptive"),
    Item("3", "header (body = Head)", "gap", "body", "Head", "two-sided"),
    Item("4", "key pass is a cross (kp_cross)", "gap", "key_pass_cross", "yes", "two-sided"),
    Item("5", ">= 2 defenders in the shooting cone (ff_def_cone >= 2)", "gap", "defenders_in_cone", "2_plus", "two-sided"),
    Item("6a", "goalkeeper >= 4 off the goal line (gk_depth >= 4, StatsBomb units)", "gap", "gk_depth", "04_plus", "two-sided"),
    Item("7", "direct free kick (stype = Free Kick)", "gap", "shot_type", "Free Kick", "two-sided"),
    Item("8", "frame closer to distance + angle than to the vendor", "correlation_difference", None, None, "greater",
         reference="none (whole population)", sensitivity="Pearson of the main-scale logits"),
    Item("9", "frame-vendor agreement relative to positions-vendor agreement, headers vs foot shots", "correlation_contrast",
         "body", "Head", "less", reference="foot shots: body in (Right Foot, Left Foot); body Other left out"),
    Item("10", "tight view: lowest n_players tercile at offset 0", "log_loss", "players_visible", "low", "greater"),
    Item("11", "under pressure", "gap", "under_pressure", "yes", "two-sided"),
    Item("12", "one-on-one", "gap", "one_on_one", "yes", "two-sided"),
    Item("14", "outside the box (x < 102 or |y - 40| > 22)", "gap", "outside_box", "yes", "two-sided"),
    Item("15", "goalkeeper not detected at offset 0 (gk_detected false)", "gap", "gk_visible", "no", "two-sided"),
    Item("16", "|timing residual| in the top quartile", "log_loss", "timing_residual_top", "yes", "greater"),
)
BY_ID = {item.id: item for item in FAMILY}
ALL_ITEMS = tuple(BY_ID)


def scan_levels() -> list[tuple[str, str]]:
    """(category, level) of every level the open scan tests, in dictionary order."""
    return [(c.name, level) for c in cat.CATEGORIES if c.role == "category" for level in c.scan_levels]


def design() -> dict:
    """The C4 design (module docstring) as JSON-ready data."""
    return {
        "notation": {
            "z_X": "method X's logit on the main scale",
            "gap": "g = z_F - z_V (frame minus vendor) unless stated otherwise",
            "unadjusted_contrast": "mean(g | in) - mean(g | out)",
            "adjusted_effect": "coefficient of the indicator in the OLS fit g ~ 1 + indicator + z_D",
            "references": "beside every F - V effect, the same effects for D - V and P - V",
            "xg_points": "every logit effect is also given in xG points at the category mean",
        },
        "scale": {
            "main": "outcome-recalibrated: shared cross-fitted Platt recalibration as in the paper (xgpoc.vendor.ladder: "
                    f"{N_FOLDS} match-grouped folds, fold seed {FOLD_SEED}), refitted separately inside each population, each "
                    "split half and each bootstrap draw; never called label-free",
            "sensitivity": ["raw logits", "within-population ranks (percentile points)"],
            "raw_levels": "reported once, for context",
        },
        "methods": METHODS,
        "populations": {
            "primary": "main population: paired SoccerNet shots, non-penalty, with a scored frame (xgpoc.vendor.ladder."
                       "analysis_population on the confirmatory table)",
            "combined": "robustness only: primary plus StatsBomb-only shots and recovered goals with frames, without the label "
                        "exclusions and the pre-registered frame exclusion; it reuses the primary shots, so it is not a replication",
            "penalties": "out of every category analysis",
            "descriptors": "offset-0 detector descriptors from cache/xgd_full3/E3 (offset_ms == 0)",
        },
        "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED,
                      "draws_definition": "xgpoc.vendor.draws.table_draws over the matches of the shot table (match-clustered, "
                                          "count-weighted)",
                      "recalibration": "refitted inside every draw",
                      "p_values": {"greater": "share of draws <= 0", "less": "share of draws >= 0",
                                   "two-sided": "min(1, 2 * min(share of draws <= 0, share of draws >= 0))"},
                      "intervals": "95% percentile intervals"},
        "floors": {**FLOORS, "below_floor": "a level below a floor is shown as descriptive, with its n"},
        "thresholds": {**THRESHOLDS, "meaning": "a finding is called interesting only at or above its threshold"},
        "family": {
            "correction": f"Benjamini-Hochberg at q = {Q_BH} over the {len(FAMILY)} items; Benjamini-Yekutieli as a sensitivity check",
            "population": "primary",
            "test": "every item on its adjusted effect; directional items one-sided, the others two-sided",
            "kinds": KINDS,
            "items": [{**asdict(item), "threshold": item.threshold} for item in FAMILY],
        },
        "outside_family": {
            "1": "owner's statement (a): non-goal shots from outside the box (sb_outcome != Goal, outside_box = yes); on target = "
                 "sb_outcome in (Saved, Saved to Post), off target = sb_outcome in (Off T, Post, Wayward); Blocked and "
                 "Saved Off Target reported apart; F and F-400 side by side with the paper's H4 controls (vendor logit, StatsBomb "
                 "body-part and shot-type indicators); a re-expression of the paper's H4, post-contact",
            "13": "on target overall (on_target_sn): reported as the paper's known H4 result",
            "6b": "open goal (open_goal = yes): descriptive",
            "negative_control": "half: first against second half, expected to stay inside the threshold",
        },
        "scan": {
            "scope": "F - V only, every scan level of the dictionary (category role 'category'); post-shot, sensitivity and "
                     "negative-control categories are not scanned",
            "levels": [{"category": c, "level": level} for c, level in scan_levels()],
            "split": {"seed": SPLIT_SEED, "halves": list(HALVES), "unit": "match", "stratified_by": "league_season",
                      "rule": "for each league-season in sorted order, its sorted matches are permuted by one default_rng(seed); "
                              "the concatenated list is assigned alternately discovery, validation, discovery, ..."},
            "candidate": f"|adjusted effect| >= {THRESHOLDS['scan_logit']} and two-sided p < {SCAN_ALPHA} on the discovery half, "
                         "and the floors met in both halves",
            "freeze": "candidates, directions and estimands are written to a file and hashed before the validation half is touched",
            "validation": f"one-sided test in the discovered direction on the validation half, Holm at {HOLM_ALPHA} across the "
                          "candidates; pass = 'replicated exploratory'",
            "not_replicated": "'hypothesis-generating', appendix only, never in the summary",
            "descriptive_scans": "F - D and F - P scans in an appendix, without tests",
            "recalibration": "refitted inside each half",
        },
        "post_shot": {
            "categories": ["on_target_sn", "sb_outcome"],
            "rules": ["flagged as post-contact", "F and F-400 always side by side; a gap that vanishes at -400 ms reads as the "
                      "frame seeing the result after contact, not as a valuation", "gaps and discrimination (where both goal "
                      "classes exist) only; no calibration inside outcome-defined categories", "SoccerNet on target includes "
                      "blocked shots; StatsBomb Blocked appears mainly in the combined population"],
        },
        "frames": "no frame is looked at, by people or agents; the 5% largest F - V disagreements in each direction are listed "
                  "by shot ID with their category profile and goal rate, for the owner",
        "who_is_right": "descriptive, per category and method: O - E, O/E, log-loss, delta LL against V, AUC where both goal "
                        "classes meet the floor",
        "agreement": "descriptive except items 8 and 9: Spearman matrix of all methods, overall and per category, plus Pearson "
                     "on logits; differences between dependent correlations are bootstrapped jointly",
        "status": "exploratory, post hoc on frozen predictions whose labels the xg-vendor confirmatory run already joined",
    }


def load(path: Path, expected_sha256: str) -> dict:
    """The protocol, after checking that the file's SHA-256 is the recorded one."""
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {digest}, ticket 01 recorded {expected_sha256}")
    return json.loads(Path(path).read_text())
