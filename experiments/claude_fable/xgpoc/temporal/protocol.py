"""The frozen protocol of xg-temporal (ticket 03; spec, "Dane i arms", "Kontrola przecieku", "Ewaluacja"; owner
decisions 2026-10-24 and 2026-10-25; council C8) as data.

`research/xgt_protocol.json` (`PROTOCOL`) is written by `scripts/xgt_protocol.py` before any label is read, from two
parts:
    frozen()   every rule the code runs, built from the constants of `xgpoc.temporal` (window, k, columns' naming,
               model and penalty grid, preprocessing, bootstrap, readings, controls' settings, the ticket 07 grid, the
               descriptive arms, the disclosures). `scripts/xgt_run.py` refuses a protocol whose `frozen` part differs
               from what the current code builds, so the file and the code cannot drift apart.
    data       written by the script from the label-free inputs: input hashes (the g_xgv completion record included),
               population counts and shot-list hashes of the main population and of every grid cell, the arms'
               columns, the provenance checks, the ticket 02 result with its hash, the label-free controls' record
               with its hash, and the package code digest (`records.package_hashes`).
`load` opens the file only at the pinned SHA-256.
"""

import json
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss
from xgpoc.temporal import categories as tc
from xgpoc.temporal import controls as tco
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp
from xgpoc.temporal import readout as tr

PROTOCOL = Path("research/xgt_protocol.json")              # relative to experiments/claude_fable
SAFE_MARGIN = Path("results/xgt_safe_margin/result.json")
CONTROLS = Path("results/xgt/controls")                    # the permutation's record (complete.json) and the gate of --real
LABEL_FREE = CONTROLS / "label_free"                       # the label-free controls, before the protocol
MAIN_SOURCE = "g_xgv"
DESCRIPTIVE_SOURCES = ("g_xgc", "dinov2")

GRID_O_MAX = (-1200, -1000, -800, -600, -400)
GRID_START = (-1600, -1200, -800)
LEAK_ROWS = (-200, 0, 400)                                 # start -1600, leakage-control rows
LEAK_CURVE = (-1000, -800, -600, -400, -200, 0, 400)       # start -1600, computed by ticket 04
K_SENSITIVITY = (16, 64)
SAFE_MARGIN_RESULT_SHA256 = "6a1ca0e145ad0cd9aae5ba1fd6e7a6a9ca1e5bda8099d4e546f8b73c387f5551"


# ------------------------------------------------------------------ the ticket 07 grid
def cell_role(start: int, o_max: int) -> str:
    if (start, o_max) == (ti.START, ti.O_MAX):
        return "main"
    return "leak_control" if o_max > tp.LEAK_LIMIT else "descriptive"


def grid_cells() -> list[dict]:
    """Every window of the ticket 07 grid and the leakage-control rows (spec, leak control 6; C8 Q2): role, control
    flag, the ticket that computes it, the motion features it has."""
    windows = [(s, o) for o in GRID_O_MAX for s in GRID_START if s < o] + [(-1600, o) for o in LEAK_ROWS]
    cells = []
    for start, o_max in windows:
        window = ti.Window(start, o_max, o_max > tp.LEAK_LIMIT)
        cells.append({"start": start, "o_max": o_max, "role": cell_role(start, o_max), "control_flag": window.control,
                      "offsets": list(window.offsets), "middle": window.middle,
                      "motion_features": tf.motion_kinds(window),
                      "ticket": "04" if start == -1600 and o_max in LEAK_CURVE else "07",
                      "candidate": cell_role(start, o_max) == "main"})
    return cells


def cell_window(cell: dict) -> ti.Window:
    return ti.Window(cell["start"], cell["o_max"], cell["control_flag"])


# ------------------------------------------------------------------ the frozen part
def model_record() -> dict:
    return {"class": "L2 logistic regression, Newton's method on the standardised design (xgpoc.stack.fit.logistic_fit; "
                     "summed log-loss + 0.5 Σ λ_j w_j², λ = 1 / C, intercept unpenalised)",
            "name": tr.MODEL.name, "c_grid": list(tr.C_GRID), "z_unpenalised": tr.MODEL.z_free,
            "c_grid_source": "council C8 Q1 (b): (1e-5, 3e-5, 1e-4, 3e-4) + xgpoc.stack.fit.C_GRID, no fallback to R",
            "choice": "per outer fold, inner match-grouped CV over the other four host folds, each inner design "
                      "preprocessed on its own inner training rows, summed inner held-out log-loss, ties to the "
                      "smaller C; refit on the whole outer training fold",
            "newton": {"iterations": sf.NEWTON_ITERATIONS, "tolerance": sf.NEWTON_TOLERANCE},
            "applies_to": ["R", "S", "T", "L", "S_xgc", "T_xgc", "F_S", "F_T", "k sensitivity"]}


def preprocessing_record() -> dict:
    return {"window": {"grid_ms": list(ti.GRID), "step_ms": ti.STEP, "start": ti.START, "o_max": ti.O_MAX,
                       "middle": "the grid offset nearest (start + o_max) / 2, the earlier one on a tie"},
            "usable_frame": "frame_status 'ok', no cut between the frame and the anchor, an embedding",
            "imputation": "an unusable window frame takes the nearest earlier usable window frame, else the nearest "
                          "later one (the leading run a cut leaves); one missing indicator per window offset; "
                          "row-wise, no fitted state",
            "pca": {"k": tf.K, "solver": "randomized", "seed": tf.PCA_SEED,
                    "fitted_on": "every usable window frame of the training shots pooled (outer training fold; inner "
                                 "training rows for the inner CV), applied to every frame of the scored shots"},
            "standardisation": "every design column with the training mean and SD (ddof 0), scale 1 if constant",
            "motion_features": {"vs_start": "score at o_max minus at the window start",
                                "vs_mid": "score at o_max minus at the middle frame; omitted when the middle frame is "
                                          "the start",
                                "slope": "least-squares slope per second over every window frame; omitted for <= 3 "
                                         "frames"},
            "missing_indicators": "all in S, so T - S holds only the motion scores",
            "column_names": "<feature>@<offset>, the offset being the latest frame the column uses; z_g_combo bare"}


def arms_record() -> dict:
    return {"R": {"status": "reference", "inputs": "z_g_combo (results/xgd_a1_ensemble/oof.csv)"},
            "S": {"status": "static control", "inputs": "R + pc01..pc32@o_max + missing@<offset> for every window offset"},
            "T": {"status": "confirmatory (H1, H2)", "inputs": "S + pcJ_vs_start@o_max + pcJ_vs_mid@o_max + "
                                                                "pcJ_slope@o_max, J = 1..32"},
            "source": MAIN_SOURCE, "k": tf.K,
            "k_rule": "k = 32 fixed without labels (ticket 01: explained variance 16/32/64 on the training folds); "
                      "never chosen on a positive control"}


def readings_record() -> dict:
    return {
        "delta_ll": "LL(arm) - LL(reference), mean per-shot log-loss (clip [1e-6, 1 - 1e-6]), out of fold on P_G",
        "bootstrap": {"draws": ss.N_BOOT, "seed": ss.BOOT_SEED, "unit": "match (the population's matches)",
                      "definition": "xgpoc.detector.testrule.match_weights", "interval": ss.INTERVAL},
        "p_one_sided": "(1 + #{Δ_b >= 0}) / (B + 1)",
        "holm": {"contrasts": list(tr.CONFIRMATORY), "alpha": ss.ALPHA},
        "detectable_improvement": "a Holm-rejected contrast",
        "practical_candidate": f"a Holm-rejected contrast with point ΔLL <= {ss.PRACTICAL}, unless the readability veto "
                               "fires",
        "motion_helps": "only when T - S is Holm-rejected",
        "descriptive": list(tr.DESCRIPTIVE),
        "readability_veto": {"population": "non-goal shots of P_G", "score": "logit(p_T) - logit(p_S), out of fold",
                             "target": "on_target (SoccerNet class 'Shots on target')",
                             "statistic": "AUC; lower bound = 5th percentile of the match bootstrap (one-sided 95%, "
                                          "2,000 draws, seed 0, matches of the non-goal shots, "
                                          "xgpoc.probe.readability.weighted_auc)",
                             "rule": f"lower bound > {tr.VETO_AUC} blocks every 'practical candidate' claim (both "
                                     "contrasts); reported either way"},
        "arm_near_reference": f"an arm whose inner CV chose the grid's smallest C ({tr.C_GRID[0]}) in >= "
                              f"{tr.NEAR_REFERENCE_FOLDS} of the 5 outer folds is reported as 'arm ≈ R'; the grid is "
                              "not extended again (C8)",
        "no_improvement_wording": "'no improvement larger than |bound|' (bound = 5th percentile of the bootstrap ΔLL) "
                                  "only for a contrast whose clustered-null coverage is >= 0.95 - 2 Monte Carlo SE",
        "also_reported": ["AUC", "Brier", "calibration slope and intercept", "the chosen C per arm and outer fold",
                          "the power curve of T - S from the controls"],
        "categories": tc.definitions(),
        "categories_source": "xgpoc.stack.inputs.e3_table on cache/xgd_full3/E3 (xg-stack's frozen E3 table), t0 columns",
        "no_test": "the 172 held-out matches are spent; every result is development (D7)",
    }


def leak_record() -> dict:
    return {
        "curve": {"start": -1600, "o_max": list(LEAK_CURVE), "contrasts": ["S-R", "T-R", "T-S"], "ticket": "04"},
        "cell_population": "each cell's window population (>= 1 usable frame in [start, o_max]) intersected with the "
                           "main population P_G, so every cell is read on a subset of the same shots (equal to P_G "
                           "for start -1600 and o_max >= -600); R refitted on the cell's shots",
        "method_positive_control": {"contrast": "S-R", "o_max": tr.METHOD_POSITIVE_O_MAX, "start": -1600,
                                    "pass": "hi95 < 0 (the 95% interval below zero)",
                                    "if_failed": "the leak curve cannot detect leakage; the report says so and makes no "
                                                 "'no leakage' statement"},
        "jump_rule": {"contrasts": ["T-R", "T-S"],
                      "trend": "g = (Δ(-600) - Δ(-1000)) / 2, the change per 200 ms step over the pre-margin points",
                      "excess": "e(o) = Δ(o) - (Δ(-600) + g (o + 600) / 200) for o in {-400, -200}",
                      "signal": f"e(-400) <= {tr.LEAK_EXCESS} or e(-200) <= {tr.LEAK_EXCESS} for either contrast",
                      "consequence": "a leak signal is disclosed beside every claim of the main read-out; it changes "
                                     "no reading (descriptive)"},
        "control_rows": "o_max -400 (every start; C8 Q2), -200, 0, +400 are leakage-control rows with the control flag, "
                        "never a candidate; 'control' does not mean confirmed contamination",
        "grid_rules": "every cell: arms S and T with the model and penalty rule of the main read-out; S - R, T - R and "
                      "T - S with the match bootstrap; no correction, no choice of a best cell, the grid never selects "
                      "a design",
        "safe_margin_checks": "the -600 (5/120 worst case) and -400 (6/120) checks of ticket 02 are reported side by side",
    }


def descriptive_record() -> dict:
    same = "model, penalty grid, preprocessing and bootstrap of the main read-out; descriptive: no Holm, no claim"
    return {
        "L": {"ticket": "09", "source": "g_xgv predictions.parquet: xg_logit (mean over the two seeds of logit / T) and "
                                        "on_target_logit (mean over the two seeds) of the xgv fold models",
              "inputs": "R + xg_logit@<offset> and on_target_logit@<offset> for every window offset (-1600..-600), "
                        "imputed as the embeddings, + missing@<offset>",
              "population": "P_G", "contrasts": ["L-R", "L-T"], "rules": same,
              "comparison": "results/preshot_frames_2026_10/summary.json (2281fc1): differences of population, folds and "
                            "logit source described, no tuning"},
        "S_xgc/T_xgc": {"ticket": "09", "source": "g_xgc (results/xgc_offset_scores, v1.2 fold models)",
                        "population": "g_xgc window population (6,320 shots, 297 matches at o_max -600)",
                        "contrasts": ["S_xgc-R", "T_xgc-R", "T_xgc-S_xgc"], "rules": same,
                        "provenance": "the v1.2 fold models saw labels of xgv evaluation folds; the count of xgv "
                                      "evaluation matches with v1.2 train/valid shots is computed label-free by ticket 09"},
        "F_S/F_T": {"ticket": "09", "source": "dinov2 (frozen DINOv2-L, results/xgc_offset_scores)",
                    "population": "dinov2 window population (7,010 shots, 328 matches)",
                    "contrasts": ["F_S-R", "F_T-R", "F_T-F_S"], "rules": same},
        "k_sensitivity": {"ticket": "09", "k": list(K_SENSITIVITY), "arms": ["S", "T"], "source": MAIN_SOURCE,
                          "population": "P_G", "rules": same},
        "seed_sensitivity": {"ticket": "09", "arms": ["S", "T"],
                             "source": "g_xgv seed-1 embeddings (embeddings_g_combo_seed1.npy, same embedding_ref rows)",
                             "population": "P_G", "rules": same},
        "N": {
            "ticket": "08", "status": "descriptive", "source": MAIN_SOURCE, "population": "P_G",
            "sequence": "the 6 window frames -1600..-600 in time order; per step the PCA-32 scores of the outer fold's PCA "
                        "(the T design's PCA, imputed as T) standardised per component with the training frames' mean "
                        "and SD, plus the step's missing indicator (33 inputs per step)",
            "architecture": "GRU, 1 layer, hidden size 8, input dropout 0.1; head linear(8 -> 1) initialised to 0; logit "
                            "= α + β z_g_combo + head(last hidden state), α and β initialised at R's fit on the training "
                            "rows and unpenalised (z as an offset with its own slope, as in R)",
            "training": "binary cross-entropy, AdamW lr 1e-3, weight decay 1e-2 on the GRU and the head only, batch 256, "
                        "at most 60 epochs, torch.set_num_threads(1), CPU",
            "early_stopping": "inner match-grouped CV per outer fold: each of the 4 inner splits (other host folds) "
                              "trains 60 epochs and records its held-out log-loss after every epoch; E* = argmin over "
                              "epochs 0..60 of the summed inner held-out log-loss (ties to fewer epochs; E* = 0 is R); "
                              "the outer model is retrained on the whole outer training fold for E* epochs",
            "seeds": [0, 1, 2, 3, 4],
            "reported": "ΔLL N - R and N - T per seed with the match bootstrap, their mean and SD over seeds, and the "
                        "ΔLL of the seed-averaged probability; E* per outer fold and seed",
            "rules": "no Holm, no claim; hyperparameters are not tuned on ticket 04's results",
        },
    }


def disclosures() -> list[str]:
    return [
        "Ticket 02 STOP: the frozen safe-margin rule (research/xgt_safe_margin_rule.md, sha256 5522847a...) found 5 "
        "worst-case violations of 120 at -600 ms (one-sided 95% Clopper-Pearson bound 8.56% > 5%); every violation is one "
        "run's 'unknown' against the other's 'no', no run answered 'yes'. Owner decision 2026-10-25: o_max stays -600 "
        "with the stop disclosed; boundary sensitivity from the ticket 07 grid. -400 had 6 of 120 (9.6%).",
        "The rich null's u uses one PCA pooled over the population's window frames; the arms use a PCA per training "
        "fold (C8).",
        "The motion plant is the pooled PCA's first-component o_max - start score residualised on S's pooled inputs; T's "
        "motion columns come from the fold PCA, so the plant lies in T's span only approximately.",
        "Every truth of the label-free controls uses z_g_combo with intercept 0 and slope 1; no real label sets it.",
        "z_g_combo is the averaged pre-Platt logit of the two G-combo models that held each shot out; each member's "
        "epoch and temperature were chosen on its own scored fold (xg-stack C6).",
        "The g_xgv embeddings are seed 0 of the two models of the fold that held out the shot's match.",
        "The 31 training matches of the official v1.2 test split are scored by the xgv fold models like every other "
        "training match.",
        "Imputation from the nearest later usable frame where no earlier one exists (a cut's leading run).",
        "The g_xgc replication has the v1.2-fold provenance caveat (spec, 'Proweniencja embeddingów G xgc').",
    ]


def frozen() -> dict:
    """Every rule the code runs (module docstring), from the constants of `xgpoc.temporal`."""
    return {
        "question": "does a small temporal head on the frame embeddings before contact lower the out-of-fold "
                    "log-loss of G-combo on the 328 training matches?",
        "hypotheses": {"H1": "T has a lower out-of-fold LL than R", "H2": "T has a lower out-of-fold LL than S"},
        "o_max": {"value": ti.O_MAX, "decision": "D1 (owner, 2026-10-22/24): fixed; may move only earlier and only by "
                                                 "the owner",
                  "ticket_02": "STOP, kept by the owner 2026-10-25 with disclosure (data.safe_margin)"},
        "population": {"source": MAIN_SOURCE, "role": tp.ROLE, "rule": "non-penalty shots of "
                       "results/xgd_a1_ensemble/oof.csv in the plan's training role with >= 1 usable g_xgv frame in "
                       "[-1600, -600]; folds = the plan's xgv host folds", "expected_base": ti.EXPECTED_BASE,
                       "never": ["StatsBomb in any form", "any shot of the 172 held-out matches"]},
        "outcome": {"goal": "index column goal", "on_target": "index column label == 'Shots on target' (veto only)",
                    "read": "only inside xgpoc.detector.seal.development_labels_only (xgpoc.temporal.labels), after "
                            "this protocol is committed"},
        "arms": arms_record(), "model": model_record(), "preprocessing": preprocessing_record(),
        "leak_assertions": {"banned_fragments": list(tp.BANNED), "leak_limit": tp.LEAK_LIMIT, "anchor": tp.ANCHOR,
                            "held_out_roles": list(tp.HELD_OUT_ROLES)},
        "readings": readings_record(), "controls": tco.settings(), "grid": grid_cells(), "leak": leak_record(),
        "descriptive": descriptive_record(), "disclosures": disclosures(),
        "after_freeze": ["the permutation control inside the label seal (results/xgt/controls/complete.json)",
                         "ticket 04: the single real run (results/xgt/real)", "tickets 07-09: the descriptive runs"],
        "on_failure": "a failed control is fixed in the code, never by loosening a criterion; a re-freeze only after "
                      "a code change, with the reason; a positive control failing twice stops for the council (D9); a "
                      "positive-control miss caused by shrinkage goes to the council as a grid question (C8)",
    }


def load(path: Path, expected_sha256: str | None) -> dict:
    """The protocol, refused unless a SHA-256 is pinned and the file has it."""
    if expected_sha256 is None:
        raise ValueError(f"no SHA-256 is pinned for {path}: commit the protocol, then pin it")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path} has SHA-256 {digest}, the code pins {expected_sha256}")
    return json.loads(Path(path).read_text())
