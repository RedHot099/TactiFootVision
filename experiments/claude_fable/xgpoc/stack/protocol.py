"""The frozen design of xg-stack (council C5, 2026-10-14) as data (ticket 01).

`design` transcribes the "Decisions -> C5" section of `.scratch/xg-stack/spec.md` from the constants the code runs
(`inputs`, `fit`, `stats`, `controls`), so the protocol and the code cannot drift apart. `scripts/xgs_protocol.py`
adds the inputs' hashes, the population counts and the code hashes, and writes `research/xgs_stack_protocol.json`
before any arm is fitted on real labels. `scripts/xgs_run.py` opens it with `load`, which refuses a file whose SHA-256
is not the recorded one, and refuses to run when the design, the code or an input differs from the file.
"""

import json
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.stack import controls as sc
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import provenance as sp
from xgpoc.stack import stats as ss

PROTOCOL = Path("research/xgs_stack_protocol.json")   # relative to experiments/claude_fable

READINGS = [
    "Column names. Every E3 column is named <name>@<offset> (t0 = @0, -400 ms = @-400), as A1's vector. The sanitised "
    "shooter_by_goal and gk_dist_goal enter as shooter_by_net and gk_dist_net, because control 1 bans the fragment 'goal' "
    "in a design column name and C5 lists shooter_by_goal among the G inputs; the values are unchanged.",
    "Missing indicators. dist, angle and shot_x are missing together, exactly where the sanitised shot_ok is 0, so they "
    "share one indicator, shot_missing. kp_instance_conf has kp_conf_missing; n_keypoints and n_players are missing only "
    "for an unreadable frame and share frame_missing (0 on every population shot at t0, a constant column of zeros). "
    "Every indicator is checked to be 1 exactly where its columns are missing.",
    "ball_source one-hot. The levels are none, own (the row's own frame) and one per fallback frame of E3's source order "
    "for that offset: at t0 own, m400, m800 (E3 prefers the -400 ms ball, t0's own ball is the last fallback); at -400 ms "
    "own, m800. All levels are columns; the L2 penalty handles the redundancy with the intercept.",
    "Spline. A natural cubic spline with 4 df is the truncated-power basis of Hastie, Tibshirani & Friedman (eq. 5.4-5.5) "
    "on 5 knots at the 0/25/50/75/100% quantiles of the training fold's non-missing values (R's ns(x, df = 4)), linear "
    "beyond the boundary knots; a missing value is imputed with the training-fold mean before the basis.",
    "Standardisation is applied to every design column, z_g_combo, indicators and flags included, with the training "
    "fold's mean and population standard deviation; a constant column keeps scale 1.",
    "z_g_combo in results/xgd_a1_ensemble/oof.csv is the mean of logit(stored p) of the two G-combo models that held each "
    "shot out (temperature-scaled member outputs, before the ensemble's Platt step). R and every arm refit their own "
    "intercept and slope on it, so the ensemble Platt step, an affine map of z, does not change any arm. Each member's "
    "epoch and temperature were selected on its held-out fold (scripts/vendor_retrain_g.py), so z carries a little of "
    "its own fold's labels; every arm shares that z.",
    "ALL adds, at t0, the sanitised geometry not in G (each with a <column>_missing indicator), the sanitised flags not "
    "already encoded (shot_ok is 1 - shot_missing; ball_ok and ball_fallback are the ball one-hot), the sanitation "
    "indicators and counts; at -400 ms, the same plus every G input of that offset, dist and angle linear.",
    "GBM inputs are the raw G inputs without the missing indicators (native NaN), z_g_combo first and monotone "
    "increasing, the S flags for GS; no spline, imputation or standardisation.",
    "Log-loss clips probabilities to [1e-6, 1 - 1e-6] (xgpoc.eval.calibration.EPS); the calibration slope and intercept "
    "are the unpenalised logistic regression of y on logit(p).",
    "The one-sided 95% bound of every arm is the 5th percentile of the bootstrap ΔLL: a null reads 'no improvement "
    "larger than |bound|'. The 95th percentile is reported beside it.",
    "Category ΔLL intervals restrict the same 2,000 match draws to the category's shots. A shot whose defining E3 input "
    "is missing is neither inside nor outside an E3 category.",
    "Control 2's 'Holm detectable rate' is the share of replicates in which Holm over {G, GS} rejects at least one arm "
    "(the familywise rate); the per-arm rates are reported beside it.",
    "Control 4: β is the negative root of true ΔLL(β) = -0.002 (farther shots score less), solved by brentq on "
    "[-5, 0]. The β = 0 refit is the unpenalised logistic regression of the probabilities p_β on [1, z] over the "
    "population; (a, b) is R's L2 fit on every shot, mapped to the raw z scale. 'Empirical power of G' is the share of "
    "replicates in which G is a Holm 'detectable improvement'. The oracle ΔLL, LL(p_true) - LL(R out of fold) on each "
    "replicate, is a diagnostic only.",
    "Control 3 runs the full read-out of the real run (every arm, the GBM and the categories) on the permuted labels; "
    "its criterion is the Holm rejections over {G, GS} only.",
    "Seeds: control 2 default_rng(1), control 4 default_rng(2), control 3 default_rng(0) for the within-match "
    "permutation; each replicate draws y* = (rng.random(n) < p) in replicate order.",
]


def design() -> dict:
    """The C5 design (module docstring) as JSON-ready data."""
    return {
        "population": {
            "shots": si.EXPECTED["shots"], "matches": si.EXPECTED["matches"], "folds": si.EXPECTED["folds"],
            "rule": "the non-penalty shots of results/xgd_a1_ensemble/oof.csv (the 328 training matches of the xg-vendor "
                    "split plan), with the host folds of cache/folds_xgv_home5.csv",
            "outcome": "the index goal column (v1.2 index, xgpoc.paths.INDEX), read inside the held-out label seal, kept "
                       "apart from the design; never an input",
            "never": ["StatsBomb in any form", "any shot or label of the 172 held-out matches (135 evaluation, 37 gate)"],
        },
        "frame_logit": {"column": sf.Z, "source": "results/xgd_a1_ensemble/oof.csv",
                        "meaning": "out-of-fold 10-model G-combo ensemble logit; each fold's logits from models that never saw it"},
        "e3": {
            "source": "cache/xgd_full3/E3 (frames.parquet, features.parquet)", "config": si.E3_CONFIG.name,
            "ball_source_order": {str(o): list(si.E3_CONFIG.order(o)) for o in si.OFFSETS},
            "replay": "the frames are replayed with E3's ball-source rule; the replayed raw features must equal "
                      "features.parquet exactly",
            "sanitation": "xgpoc.detector.sanitation.sanitised_features on the replayed frames (the frozen label-free rule)",
            "renamed": si.RENAMED, "ball_levels": {str(o): list(si.ball_levels(o)) for o in si.OFFSETS},
            "missing_groups": {str(o): si.missing_groups(o) for o in si.OFFSETS},
            "columns_per_offset": {str(o): {"base": si.base_columns(o), "extra": si.extra_columns(o)} for o in si.OFFSETS},
        },
        "soccernet_flags": {
            "source": "Labels-v2.json of each match (xgpoc.paths.LABELS_ROOT)",
            "never": ["the 'Goal' label (dropped at parse)", "this shot's own label type (both shot types become one kind "
                      "at parse)", "the index label, goal or visibility columns"],
            "team": "the team of the shot's own annotation (a shot annotation at the index row's half and position); when "
                    "it is missing, not home or away, or ambiguous, every flag is 0 and sn_team_unknown is 1",
            "windows_ms": {flag: {"annotation": kind, "from": lo, "to": hi, "inclusive": True}
                           for flag, (kind, lo, hi) in si.WINDOWS.items()},
            "same_team_same_half": True, "hindsight_prone": ["sn_prev_shot_5s"], "columns": si.S_FLAGS,
        },
        "preprocessing": {
            "fitted_on": "the training folds only (the other four folds of each scored fold)",
            "imputation": "training-fold mean, plus the missing indicator columns",
            "spline": {"kind": "natural cubic", "df": sf.SPLINE_DF, "knot_quantiles": list(sf.KNOT_QUANTILES)},
            "standardisation": "every design column, training-fold mean and standard deviation (ddof 0), scale 1 if constant",
        },
        "arms": {name: {"status": arm.status, "linear": list(arm.linear), "spline": list(arm.spline), "columns": arm.columns}
                 for name, arm in sf.ARMS.items()},
        "confirmatory": list(sf.CONFIRMATORY), "descriptive": list(sf.DESCRIPTIVE),
        "models": {
            "logistic": {"class": "sklearn.linear_model.LogisticRegression", **sf.LOGISTIC,
                         "design": "standardised; z_g_combo linear; no interactions"},
            "gbm": {"class": "sklearn.ensemble.HistGradientBoostingClassifier", **sf.GBM, "monotone": {sf.Z: "increasing"},
                    "missing": "native NaN", "arms": list(sf.GBM_ARMS),
                    "columns": {k: v for k, v in sf.GBM_COLUMNS.items()},
                    "role": "sensitivity, reported either way, never changes a claim or the recommendation"},
            "scoring": "out of fold: each fold scored by the model fitted on the other four",
        },
        "read_out": {
            "delta_ll": "LL(arm) - LL(R), mean per-shot log-loss, out of fold, on the population",
            "clip": ss.CLIP,
            "bootstrap": {"draws": ss.N_BOOT, "seed": ss.BOOT_SEED, "unit": "match (the 328 training matches)",
                          "definition": "xgpoc.detector.testrule.match_weights (readability.match_draw_counts over the "
                                        "sorted matches): each draw the match-count-weighted mean of the per-shot differences",
                          "interval": ss.INTERVAL, "one_sided": ss.ONE_SIDED},
            "p_one_sided": "(1 + #{Δ_b >= 0}) / (B + 1)",
            "holm": {"arms": list(sf.CONFIRMATORY), "alpha": ss.ALPHA},
            "detectable_improvement": "a Holm-rejected confirmatory arm",
            "practical_candidate": f"a Holm-rejected arm with point ΔLL <= {ss.PRACTICAL}",
            "every_arm": "95% interval and one-sided 95% bound (5th percentile; a null reads 'no improvement larger than "
                         "|bound|')",
            "also": ["AUC", "Brier", "calibration slope and intercept"],
            "categories": {name: rule for name, (_, rule) in ss.CATEGORIES.items()},
            "categories_role": "descriptive: out-of-fold O/E and ΔLL against R inside each category",
        },
        "controls": {
            "1_provenance": {"checks": ["every arm's design columns equal the frozen list and the matrix width",
                                        f"no design column name contains {list(sp.BANNED)}",
                                        "no population match or shot is in the 172 held-out matches of the plan",
                                        "the flags do not change when Goal annotations are added and every shot type swapped"],
                             "pass": "every check passes"},
            "2_synthetic_nulls": {"replicates": sc.N_NULL, "seed": sc.NULL_SEED, "truth": "y* ~ Bernoulli(p_R), R fitted "
                                  "on the real labels of every shot", "pass": f"Holm detectable rate <= {sc.NULL_MAX_DETECTABLE_RATE}"},
            "3_permutation": {"seed": sc.PERMUTATION_SEED, "labels": "real labels permuted within matches",
                              "pass": "no Holm rejection"},
            "4_positive": {"replicates": sc.N_POSITIVE, "seed": sc.POSITIVE_SEED, "signal": f"{sc.SIGNAL} standardised over "
                           "the population, missing set to 0", "target_true_delta": sc.POSITIVE_TARGET,
                           "beta": f"negative root, brentq on {list(sc.BETA_BRACKET)}",
                           "report": ["beta", "empirical power of G", "mean recovered ΔLL of G"],
                           "pass": f"mean recovered ΔLL of G in {list(sc.POSITIVE_WINDOW)}"},
            "gate": "if any control fails, the real run does not start; the report states it and the code is fixed under review",
        },
        "disclosures": [
            "E3's configuration was chosen on 135 of the 172 held-out matches by StatsBomb position error, with no outcomes.",
            "Stacking provenance: the training-fold z values come from models that saw the scored fold.",
            "z_g_combo's ensemble carries a Platt fit on all 7,062 labels (2 parameters, shared by every arm).",
            "The flag windows were shaped after the StatsBomb recall/precision check on the 135 matches.",
            "The flags are research-only: a product needs an action spotter.",
            "The A1 noise floor is not quoted: the stacker is deterministic.",
        ],
        "after_the_run": "a practical candidate goes to the owner, with the recomputed power on the 172 matches and the "
                         "caveat that E3 was tuned on 135 of them; a test is the owner's decision, nothing is tested otherwise",
    }


def load(path: Path, expected_sha256: str) -> dict:
    """The protocol, after checking that the file's SHA-256 is the recorded one."""
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {digest}, ticket 01 recorded {expected_sha256}")
    return json.loads(Path(path).read_text())
