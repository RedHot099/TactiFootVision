"""Permutation importance of current tabular features (GBM, valid split, neg log-loss)."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from xgpoc.features.tabular import FEATURE_NAMES

idx = pd.read_parquet("/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet")
tab = pd.read_parquet("cache/full_v2_fallback/positional/tabular.parquet").drop_duplicates("shot_id")
df = idx.merge(tab, on="shot_id")
tr, va = df[df.split == "train"], df[df.split == "valid"]
m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0)
m.fit(tr[FEATURE_NAMES], tr.goal)
r = permutation_importance(m, va[FEATURE_NAMES], va.goal, scoring="neg_log_loss", n_repeats=5, random_state=0)
print(pd.Series(r.importances_mean, index=FEATURE_NAMES).sort_values(ascending=False).round(5).to_string())
