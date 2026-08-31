"""Positional GBM as an ensemble member, with validation predictions.

    ./run.sh python scripts/build_positional_member.py --out results/premerge/gbm/seed0

The image models are strong overall but badly calibrated on penalties (mean xG ~0.26
against a real conversion of ~0.77), which is exactly where the positional model with
`is_penalty` is good. That makes it worth carrying into the stack rather than dropping
it for its weak global AUC.

Unlike earlier runs this fits on the *training* split only. Ensemble weights are fitted
on validation, so a member that had already seen validation would be scored optimistically
there and would grab weight it has not earned.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from xgpoc.data.index import load_index
from xgpoc.eval.metrics import evaluate
from xgpoc.features.tabular import FEATURE_NAMES
from xgpoc.models.common import write_result

INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"
CACHE = "cache/full_v2_fallback/positional"
ACCEPTED = ["is_penalty", "ball_in_air", "ball_air_valid", "def_in_cone_fixed", "nearest_def_fixed", "teams_known"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", type=Path, default=Path("results/premerge/gbm/seed0"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = load_index(Path(a.index))[["shot_id", "split", "goal"]]
    tab = pd.read_parquet(Path(a.cache) / "tabular.parquet").drop_duplicates("shot_id")
    aux = pd.read_parquet(Path(a.cache) / "aux.parquet").drop_duplicates("shot_id")
    data = df.merge(tab, on="shot_id").merge(aux[["shot_id", *ACCEPTED]], on="shot_id")
    feats = FEATURE_NAMES + ACCEPTED
    tr = data[data.split == "train"]
    g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15, random_state=a.seed)
    g.fit(tr[feats], tr.goal)
    va, te = data[data.split == "valid"], data[data.split == "test"]
    write_result(a.out, "a_tabular_gbm_sel", te, g.predict_proba(te[feats])[:, 1],
                 {"n_train": len(tr), "features": feats, "fit_on": "train only"},
                 valid_df=va, p_valid=g.predict_proba(va[feats])[:, 1])
    print("valid", {k: round(v, 4) for k, v in evaluate(va.goal.to_numpy(),
                                                        g.predict_proba(va[feats])[:, 1]).items()})


if __name__ == "__main__":
    main()
