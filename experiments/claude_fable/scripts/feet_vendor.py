"""Feet-variant sweep vs the StatsBomb vendor on the 2,721 paired primary shots (descriptive; Data: StatsBomb)."""

import numpy as np
import pandas as pd
from xgs_test import read_test_labels

f = pd.read_csv("results/sweep_feet_2026_10/predictions.csv").set_index("shot_id")
b = pd.read_csv("results/sweep_2026_10/predictions.csv").set_index("shot_id")
sb = pd.read_parquet("results/xgc_compare/table/shots.parquet")
sb = sb[sb.population == "primary"].set_index("soccernet_shot_id")
ids = sb.index[sb.index.isin(f.index)]
y = read_test_labels(pd.Series(ids)).astype(float)


def ll(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(q) + (1 - y) * np.log(1 - q))))


g = ll(b.loc[ids, "p_g_combo"].to_numpy())
for name, p in [
    ("vendor", 1 / (1 + np.exp(-sb.loc[ids, "z_vendor"].to_numpy(float)))),
    ("dd2c ALL", b.loc[ids, "p_stack_ALL_new"].to_numpy()),
    ("dd2c Q", b.loc[ids, "p_stack_Q_new"].to_numpy()),
    ("feet ALL", f.loc[ids, "p_stack_ALL_new"].to_numpy()),
    ("feet Q", f.loc[ids, "p_stack_Q_new"].to_numpy()),
]:
    print(name, round(ll(p) - g, 4))
print(len(ids))
