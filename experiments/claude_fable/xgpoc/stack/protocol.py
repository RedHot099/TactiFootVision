"""The frozen design of xg-stack (council C5, amended by council C6, 2026-10-14) as data (ticket 01).

Two files, both written from the constants the code runs, so the files and the code cannot drift apart:
    PLAN      `research/xgs_stack_c6_plan.json`, written before any simulation of the selection: the candidates with
              their exact columns, the selection rule and seeds, (b)'s predicted recovery, the final controls' seeds,
              criteria and reports (power curve, clustered nulls), the stop rule (C6 Q4) and the corrected disclosures
              (`plan`).
    PROTOCOL  `research/xgs_stack_protocol_c6.json`, written after the selection: the chosen candidate's design
              (`design`), the selection's result and hashes, and the plan's SHA-256.
`scripts/xgs_protocol.py` adds the inputs' hashes, the population counts and the code hashes; `scripts/xgs_run.py`
opens the files with `load`, which refuses a file whose SHA-256 is not the pinned one. C5's protocol
(`research/xgs_stack_protocol.json`, sha256 4a00c90b...) and its controls (`results/xgs_stack/c5_controls`) are kept as
they were; the code no longer reproduces that design (C6 changed it).
"""

import json
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import selection as sl
from xgpoc.stack import stats as ss

PLAN = Path("research/xgs_stack_c6_plan.json")             # relative to experiments/claude_fable
PROTOCOL = Path("research/xgs_stack_protocol_c6.json")
C5_PROTOCOL = {"path": "research/xgs_stack_protocol.json",
               "sha256": "4a00c90b3841b757076c8bea7847901078cd0effe084fa040701c069fa81d03d",
               "controls": "results/xgs_stack/c5_controls (moved from results/xgs_stack/controls; complete.json sha256 "
                           "b785f883151704e84eb1a9541bfb4049ccebbe301d8275adb4beb3aa777de0c5)",
               "outcome": "control 4 failed (G recovered -0.00077 of a planted -0.002, power 0.105); no real run"}
FINAL_SEEDS = {"synthetic_nulls": 201, "permutation": 200, "positive_dist": 202, "positive_dist_angle": 203,
               "curve_-0.003": 204, "curve_-0.004": 205, "clustered_nulls": 206}

READINGS = [
    "Column names. Every E3 column is named <name>@<offset> (t0 = @0, -400 ms = @-400). The sanitised shooter_by_goal and "
    "gk_dist_goal enter as shooter_by_net and gk_dist_net, because control 1 bans the fragment 'goal' in a design column "
    "name; the values are unchanged.",
    "Missing indicators. dist, angle and shot_x are missing together, exactly where the sanitised shot_ok is 0, and share "
    "shot_missing; kp_instance_conf has kp_conf_missing; n_keypoints and n_players share frame_missing. Every indicator is "
    "checked to be 1 exactly where its columns are missing.",
    "Angle wrap (C6). After the exact E3 replay check, every angle column becomes min(θ, 2π - θ), so a shot location "
    "behind the goal line no longer has an angle above π; the counts of changed shots per offset are in the population "
    "counts (angles_wrapped).",
    "Zero-variance rule (C6). A linear input whose population variance over its non-missing values is 0 is dropped from "
    "the arm before any fit (label-free); the dropped columns are listed per arm and every remaining input is asserted "
    "non-constant (control 1). GS is G plus the S flags left by the rule.",
    "ball_source one-hot. The levels are none, own and one per fallback frame of E3's source order for that offset: at t0 "
    "own, m400, m800; at -400 ms own, m800. (b) keeps ball_none@0 as its no-ball indicator; the other levels are quality "
    "covariates.",
    "Spline. A natural cubic spline with df columns is the truncated-power basis of Hastie, Tibshirani & Friedman (eq. "
    "5.4-5.5) on df + 1 knots at the quantiles 0, 1/df, ..., 1 of the training rows' non-missing values (R's ns(x, df)), "
    "linear beyond the boundary knots; a missing value is imputed with the training mean before the basis.",
    "Standardisation is applied to every design column with the training rows' mean and population standard deviation; "
    "a constant column keeps scale 1.",
    "Model. Every logistic fit is Newton's method on the summed log-loss + 0.5 Σ λ_j w_j² with an unpenalised intercept "
    "(λ = 1 / C), which equals sklearn's LogisticRegression(l2, C) (tested). FIXED is C5's C = 1 on every column, z "
    "included. INNER_CV leaves z unpenalised by λ_z = 0 in every arm (rather than an offset or a x100 rescaling) and "
    "chooses C from the grid inside each outer training fold: the inner folds are the other four host folds (match-"
    "grouped), each inner design is preprocessed on its own inner training rows, the summed inner held-out log-loss "
    "picks C (ties: the smaller C), and the arm is refitted on the whole outer training fold. R under INNER_CV is the "
    "unpenalised fit on z.",
    "Q = G + the quality covariates (the C5 G linear columns that (b) leaves out: shot_x, the ball levels other than none, "
    "shot_from_ball, shooter_by_net, the keypoint instance confidence with its indicator, n_keypoints, n_players with "
    "frame_missing). Under (c) every quality covariate is already in G, so Q equals G. Q is reported with its interval and "
    "one-sided bound only.",
    "ALL = GS + the quality covariates + the remaining E3 columns at t0 (sanitised geometry not in G with a <column>_missing "
    "indicator each, the sanitised flags not encoded elsewhere, the sanitation indicators and counts) + every G input and "
    "extra at -400 ms (dist and angle linear), after the zero-variance rule.",
    "GBM inputs are the arm's raw inputs without the missing indicators (native NaN), z_g_combo first and monotone "
    "increasing; no spline, imputation or standardisation.",
    "z_g_combo in results/xgd_a1_ensemble/oof.csv is the mean of logit(stored p) of the two G-combo models that held each "
    "shot out, before the ensemble's Platt step, which acts on p_g_combo only. Every arm refits its own intercept and "
    "slope on it.",
    "Log-loss clips probabilities to [1e-6, 1 - 1e-6]; the calibration slope and intercept are the logistic regression of "
    "y on logit(p) with C = 1e6 (xgpoc.eval.calibration.PlattScaler).",
    "The one-sided 95% bound of every arm is the 5th percentile of the bootstrap ΔLL ('no improvement larger than "
    "|bound|', only where the clustered-null coverage allows the wording); the 95th percentile is reported beside it.",
    "Category ΔLL intervals restrict the same 2,000 match draws to the category's shots; a shot whose defining E3 input is "
    "missing is neither inside nor outside an E3 category.",
    "Control 2's Holm detectable rate is the share of replicates in which Holm over {G, GS} rejects at least one arm.",
    "Control 3 runs the full read-out of the real run on the permuted labels; its criterion is the Holm rejections only.",
    "Plants. dist: the E3 t0 distance standardised over the population, missing 0 (C5). dist_angle: (d - e) "
    "standardised, d as dist and e the wrapped E3 t0 angle standardised the same way, so farther and narrower shots score "
    "less through both inputs. β is the negative root of true ΔLL(β) = target (brentq on [-5, 0]); the β = 0 refit is the "
    "unpenalised logistic regression of p_β on [1, z] over the population.",
    "Selection truth: (a, b) of the unpenalised logistic regression of the real labels on [1, z] over every shot, one "
    "truth for every candidate; each plant's 200 outcome vectors come from that plant's seed and are shared by the "
    "candidates (common random numbers). Equal averaged powers means equal to the replicate count; the tie goes to the "
    "simpler design, in the order (b), (c), (b)+(c).",
    "Final truth: (a, b) is R fitted on the real labels of every shot under the chosen model (C5's rule).",
    "Power is the share of replicates in which G is a Holm 'detectable improvement', with Monte Carlo SE sqrt(p(1 - p)/n). "
    "The oracle's power is the share in which the oracle's own one-sided p (the same match draws) is <= 0.05, unadjusted: "
    "the ceiling for any arm.",
    "The power curve's -0.002 point is control 4's dist plant; -0.003 and -0.004 use their own seeds. The smallest "
    "detectable effect is the true ΔLL at which G's Holm power first reaches 0.8 along the grid, linearly interpolated; "
    "beyond -0.004 it is reported as not reached.",
    "Clustered nulls: a match random intercept u_m ~ N(0, 0.25²), drawn per replicate from the same generator before the "
    "outcomes; 0.25 logit is fixed here, before any fit, as a generous match-level variation in finishing beyond the shot. "
    "The coverage of the one-sided bound is the share of replicates whose bound is <= the replicates' mean ΔLL of the arm "
    "(the procedure's expected ΔLL); the wording 'no improvement larger than |bound|' needs coverage >= 0.95 - 2 Monte "
    "Carlo SE. The coverage of 0 is reported beside it. Control 2's replicates give the same coverage without clustering.",
]

DISCLOSURES = [
    "E3's configuration was chosen on 135 of the 172 held-out matches by StatsBomb position error, with no outcomes.",
    "Stacking provenance: the training-fold z values come from models that saw the scored fold.",
    "z_g_combo is the averaged pre-Platt logit of the G-combo ensemble; each member's epoch and temperature were chosen "
    "on its own scored fold, an outcome-dependent step (corrected by council C6; C5's wording named a Platt fit on all "
    "7,062 labels, which acts on p_g_combo only).",
    "The flag windows were shaped after the StatsBomb recall/precision check on the 135 matches.",
    "The flags are research-only: a product needs an action spotter.",
    "The A1 noise floor is not quoted: the stacker is deterministic.",
    "C5's frozen design failed its positive control; this design was chosen by council C6 among pre-listed candidates on "
    "simulated outcomes only.",
]


def arms_record(spec: sf.Spec) -> dict:
    return {name: {"status": arm.status, "linear": list(arm.linear), "spline": list(arm.spline), "df": arm.df,
                   "columns": arm.columns, "dropped_by_zero_variance": spec.dropped[name]} for name, arm in spec.arms.items()}


def model_record(model: sf.Model) -> dict:
    return {"name": model.name, "class": "L2 logistic regression, Newton's method (xgpoc.stack.fit.logistic_fit)",
            "c_grid": list(model.grid), "z_unpenalised": model.z_free,
            "choice": "inner match-grouped CV over the other four host folds, summed inner held-out log-loss, ties to the "
                      "smaller C" if model.inner_cv else "fixed",
            "design": "standardised; z_g_combo linear; no interactions"}


def common() -> dict:
    """The parts of the design every candidate shares (C5 as amended by C6)."""
    return {
        "population": {
            "shots": si.EXPECTED["shots"], "matches": si.EXPECTED["matches"], "folds": si.EXPECTED["folds"],
            "rule": "the non-penalty shots of results/xgd_a1_ensemble/oof.csv (the 328 training matches of the xg-vendor "
                    "split plan), with the host folds of cache/folds_xgv_home5.csv",
            "outcome": "the index goal column (v1.2 index, xgpoc.paths.INDEX), read inside the held-out label seal, kept "
                       "apart from the design; never an input",
            "never": ["StatsBomb in any form", "any shot or label of the 172 held-out matches (135 evaluation, 37 gate)"],
        },
        "frame_logit": {"column": sf.Z, "source": "results/xgd_a1_ensemble/oof.csv"},
        "e3": {
            "source": "cache/xgd_full3/E3 (frames.parquet, features.parquet)", "config": si.E3_CONFIG.name,
            "ball_source_order": {str(o): list(si.E3_CONFIG.order(o)) for o in si.OFFSETS},
            "replay": "the frames are replayed with E3's ball-source rule; the replayed raw features must equal "
                      "features.parquet exactly",
            "sanitation": "xgpoc.detector.sanitation.sanitised_features on the replayed frames",
            "angle_wrap": "min(θ, 2π - θ) on every angle column, after the replay check (C6)",
            "renamed": si.RENAMED, "ball_levels": {str(o): list(si.ball_levels(o)) for o in si.OFFSETS},
            "missing_groups": {str(o): si.missing_groups(o) for o in si.OFFSETS},
        },
        "soccernet_flags": {
            "source": "Labels-v2.json of each match (xgpoc.paths.LABELS_ROOT)",
            "never": ["the 'Goal' label (dropped at parse)", "this shot's own label type (both shot types become one kind "
                      "at parse)", "the index label, goal or visibility columns"],
            "team": "the team of the shot's own annotation; when it is missing, not home or away, or ambiguous, every flag "
                    "is 0 and sn_team_unknown is 1",
            "windows_ms": {flag: {"annotation": kind, "from": lo, "to": hi, "inclusive": True}
                           for flag, (kind, lo, hi) in si.WINDOWS.items()},
            "same_team_same_half": True, "hindsight_prone": ["sn_prev_shot_5s"], "columns": si.S_FLAGS,
        },
        "preprocessing": {
            "fitted_on": "the training rows only (outer training folds; inner training rows for the inner CV)",
            "imputation": "training mean, plus the missing indicator columns",
            "spline": "natural cubic, df columns, knots at the quantiles 0, 1/df, ..., 1 of the training rows",
            "standardisation": "every design column, training mean and standard deviation (ddof 0), scale 1 if constant",
            "zero_variance": f"a linear input with population variance <= {sf.ZERO_VARIANCE} is dropped before any fit",
        },
        "gbm": {"class": "sklearn.ensemble.HistGradientBoostingClassifier", **sf.GBM, "monotone": {sf.Z: "increasing"},
                "missing": "native NaN", "arms": list(sf.GBM_ARMS),
                "role": "sensitivity, reported either way, never changes a claim or the recommendation"},
        "read_out": {
            "delta_ll": "LL(arm) - LL(R), mean per-shot log-loss, out of fold, on the population", "clip": ss.CLIP,
            "bootstrap": {"draws": ss.N_BOOT, "seed": ss.BOOT_SEED, "unit": "match (the 328 training matches)",
                          "definition": "xgpoc.detector.testrule.match_weights", "interval": ss.INTERVAL,
                          "one_sided": ss.ONE_SIDED},
            "p_one_sided": "(1 + #{Δ_b >= 0}) / (B + 1)",
            "holm": {"arms": list(sf.CONFIRMATORY), "alpha": ss.ALPHA},
            "detectable_improvement": "a Holm-rejected confirmatory arm",
            "practical_candidate": f"a Holm-rejected arm with point ΔLL <= {ss.PRACTICAL}",
            "every_arm": "95% interval and one-sided 95% bound (5th percentile)",
            "q_arm": "descriptive, bound only",
            "also": ["AUC", "Brier", "calibration slope and intercept", "the chosen C per arm and fold"],
            "categories": {name: c.rule for name, c in ss.CATEGORIES.items()},
        },
        "final_controls": {
            "seeds": FINAL_SEEDS, "replicates": sc.N_REPLICATES,
            "1_provenance": {"pass": "every check passes (columns, constant inputs, banned names, held-out, flag code)"},
            "2_synthetic_nulls": {"truth": "y* ~ Bernoulli(p_R), R fitted on the real labels under the chosen model",
                                  "pass": f"Holm detectable rate <= {sc.NULL_MAX_DETECTABLE_RATE}"},
            "3_permutation": {"labels": "real labels permuted within matches", "pass": "no Holm rejection"},
            "4_positive": {"gate": "dist plant", "target_true_delta": sc.POSITIVE_TARGET,
                           "pass": f"mean recovered ΔLL of G in {list(sc.POSITIVE_WINDOW)}",
                           "reported": ["the dist_angle plant at the same target", "power of G and GS with Monte Carlo SE "
                                        "(not gated)", "the oracle ΔLL and power"]},
            "power_curve": {"plant": "dist", "targets": list(sc.CURVE), "ceiling": "the oracle",
                            "smallest_detectable": f"G's Holm power >= {sc.DETECTABLE_POWER}, linear interpolation"},
            "clustered_nulls": {"random_intercept_sd": sc.CLUSTER_SD, "unit": "match",
                                "coverage": f"one-sided bound <= the procedure's expected ΔLL; wording allowed at >= "
                                            f"{sc.COVERAGE_LEVEL} - 2 Monte Carlo SE"},
        },
        "stop_rule": "if the final control 4 fails: no real run and no further iteration; the report gives the power curve "
                     "and the smallest detectable effect; a real run after that needs the owner's protocol amendment (C6 Q4)",
        "gate": "the real run starts only if every final control passes",
        "disclosures": DISCLOSURES,
        "after_the_run": "a practical candidate goes to the owner, with the recomputed power on the 172 matches and the "
                         "caveat that E3 was tuned on 135 of them; a test is the owner's decision, nothing is tested otherwise",
        "c5": C5_PROTOCOL,
    }


def plan(specs: dict[str, sf.Spec]) -> dict:
    """The selection plan of C6 (module docstring), for the candidates' specs on the population."""
    return {
        "common": common(),
        "candidates": {name: {"description": s.candidate.description, "model": model_record(s.model),
                              "arms": arms_record(s), "gbm_columns": s.gbm_columns} for name, s in specs.items()},
        "selection": {
            "outcomes": "simulated only", "plants": list(sc.PLANTS), "target_true_delta": sl.TARGET,
            "replicates_per_plant": sc.N_REPLICATES, "seeds": sl.SEEDS, "common_random_numbers": True,
            "truth": "unpenalised logistic regression of the real labels on [1, z] over every shot",
            "criterion": "G's Holm power averaged over the two plants; the highest wins",
            "ties": f"equal averages go to the simpler design, in the order {list(sf.SIMPLICITY)}",
            "predicted_recovery_b": {"plant": "dist", "mean_recovered_delta_G": list(sl.PREDICTED_RECOVERY_B)},
            "seeds_distinct_from_final": sorted(set(sl.SEEDS.values()) & set(FINAL_SEEDS.values())) == [],
        },
        "readings": READINGS,
    }


def design(spec: sf.Spec) -> dict:
    """The final design of the chosen candidate (module docstring)."""
    return common() | {"chosen": spec.candidate.name, "description": spec.candidate.description,
                       "model": model_record(spec.model), "arms": arms_record(spec), "gbm_columns": spec.gbm_columns,
                       "confirmatory": list(sf.CONFIRMATORY), "descriptive": list(sf.DESCRIPTIVE), "readings": READINGS}


def load(path: Path, expected_sha256: str) -> dict:
    """The file, after checking that its SHA-256 is the recorded one."""
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {digest}, the ticket recorded {expected_sha256}")
    return json.loads(Path(path).read_text())
