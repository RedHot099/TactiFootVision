"""The descriptive StatsBomb agreement of the xg-stack test (ticket 04, council C7 Q4). Data: StatsBomb.

It runs once, after the test result exists (`scripts/xgs_test_statsbomb.py` refuses otherwise); nothing here feeds a
model, the headline or the success rule, and every number is labelled "hypothesis formed on these matches". This
module only computes; it reads no file.

Population: the primary shots of xg-compare's table (`results/xgc_compare/table/shots.parquet`, `in_primary`): the
2,721 paired non-penalty shots of the 135 evaluation matches, with the vendor's xG `p_vendor` and the StatsBomb
location (`sb_x`, `sb_y` in the 120 x 80 yard frame; `sb_dist` to the goal centre in yards; `sb_angle` the goal-mouth
angle in radians), joined one to one with the test predictions of R, G and the deployed G-combo.

    spearman     Spearman's ρ of each model's probability with the vendor's xG over the primary shots
    categories   xg-compare's gap items in StatsBomb coordinates, kept apart from the E3 < 6 m category of the test:
                     sb_outside_box     x < 102 or |y - 40| > 22 (`compare.categories.outside_box`, item 14)
                     sb_angle_lt_15deg  goal-mouth angle < 15° (level 00_15 of xg-compare's angle category)
                     sb_dist_lt_6yd     distance < 6 yd (level 00_06 of xg-compare's distance category)
                 per category and model, over the shots where the category is defined: the shots in and out, the mean
                 probability of the model and of the vendor in, the gap in xG points 100 (mean p_model - mean p_vendor)
                 in and out, and the unadjusted contrast of the logit gap, mean(logit p_model - logit p_vendor | in) -
                 mean(... | out). The probabilities are the models' own (no recalibration), unlike xg-compare's
                 cross-fitted main scale, so the numbers are not xg-compare's effects.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xgpoc.compare.categories import outside_box
from xgpoc.eval.calibration import logit

ANGLE_DEG = 15.0
DIST_YD = 6.0
RULES = {"sb_outside_box": "sb_x < 102 or |sb_y - 40| > 22 (StatsBomb yards)",
         "sb_angle_lt_15deg": "degrees(sb_angle) < 15 (goal-mouth angle)",
         "sb_dist_lt_6yd": "sb_dist < 6 yd (to the goal centre)"}
LABEL = "hypothesis formed on these matches; Data: StatsBomb."


def category_masks(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (defined, inside) over the rows of `frame` (sb_x, sb_y, sb_dist, sb_angle); a NaN input is undefined."""
    x, y = frame.sb_x.to_numpy(float), frame.sb_y.to_numpy(float)
    dist, angle = frame.sb_dist.to_numpy(float), frame.sb_angle.to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)
    return {"sb_outside_box": (located, located & (outside_box(frame.sb_x, frame.sb_y) == "yes").to_numpy()),
            "sb_angle_lt_15deg": (np.isfinite(angle), np.isfinite(angle) & (np.degrees(np.nan_to_num(angle, nan=np.inf)) < ANGLE_DEG)),
            "sb_dist_lt_6yd": (np.isfinite(dist), np.isfinite(dist) & (np.nan_to_num(dist, nan=np.inf) < DIST_YD))}


def agreement(frame: pd.DataFrame, models: tuple[str, ...]) -> dict:
    """{"shots", "spearman": model -> ρ, "categories": [rows], "label"} for the columns p_<model> of `frame` against
    `p_vendor` (module docstring)."""
    vendor = frame.p_vendor.to_numpy(float)
    out = {"shots": int(len(frame)), "label": LABEL, "rules": RULES,
           "spearman": {m: float(spearmanr(frame[f"p_{m}"].to_numpy(float), vendor).statistic) for m in models}}
    rows = []
    for name, (defined, inside) in category_masks(frame).items():
        outside = defined & ~inside
        for m in models:
            p = frame[f"p_{m}"].to_numpy(float)
            gap = logit(p) - logit(vendor)
            rows.append({"category": name, "rule": RULES[name], "model": m, "n_defined": int(defined.sum()),
                         "n_in": int(inside.sum()), "n_out": int(outside.sum()),
                         "mean_p_model_in": float(p[inside].mean()), "mean_p_vendor_in": float(vendor[inside].mean()),
                         "gap_points_in": float(100 * (p[inside].mean() - vendor[inside].mean())),
                         "gap_points_out": float(100 * (p[outside].mean() - vendor[outside].mean())),
                         "logit_gap_contrast": float(gap[inside].mean() - gap[outside].mean())})
    out["categories"] = rows
    return out
