"""Consolidated before/after-features comparison.

Saves GBM prediction CSVs (base and +accepted features), then prints/saves a table:
before (D base, GBM base) vs after (D+pen, GBM+features), global metrics with CI
plus the penalty pocket. Outputs into results/before_after/.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from compare import bootstrap  # noqa: E402

from xgpoc.data.index import load_index
from xgpoc.eval.metrics import evaluate
from xgpoc.features.tabular import FEATURE_NAMES

from xgpoc.paths import INDEX as _INDEX  # noqa: E402
INDEX = str(_INDEX)
ACCEPTED = ["is_penalty", "ball_in_air", "ball_air_valid", "def_in_cone_fixed", "nearest_def_fixed", "teams_known"]
OUT = Path("results/before_after")


def gbm_preds(df, tab, aux, cols):
    data = df[["shot_id", "split", "goal"]].merge(tab, on="shot_id")
    if cols:
        data = data.merge(aux[["shot_id", *cols]], on="shot_id")
    feats = FEATURE_NAMES + cols
    tr, te = data[data.split != "test"], data[data.split == "test"]
    g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15, random_state=0)
    g.fit(tr[feats], tr.goal)
    return pd.DataFrame({"shot_id": te.shot_id, "goal": te.goal, "p": g.predict_proba(te[feats])[:, 1]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_index(Path(INDEX))
    tab = pd.read_parquet("cache/full_v2_fallback/positional/tabular.parquet").drop_duplicates("shot_id")
    aux = pd.read_parquet("cache/full_v2_fallback/positional/aux.parquet")
    idx_games = df[["shot_id", "game"]]
    pen_ids = set(df[df.is_penalty & (df.split == "test")].shot_id)

    variants = {
        "GBM przed": gbm_preds(df, tab, aux, []),
        "GBM po poprawkach": gbm_preds(df, tab, aux, ACCEPTED),
        "D przed": pd.read_csv("results/feature_ablation/D_base_test_predictions.csv"),
        "D po poprawkach (+pen)": pd.read_csv("results/feature_ablation/D_pen_test_predictions.csv"),
    }
    rows = []
    for name, p in variants.items():
        p = p.drop_duplicates("shot_id")
        p.to_csv(OUT / f"{name.replace(' ', '_')}_test_predictions.csv", index=False)
        m = evaluate(p.goal.to_numpy(), p.p.to_numpy())
        ci = bootstrap(p.merge(idx_games, on="shot_id"))
        rows.append({"model": name, "auc": m["auc"], "auc_ci_lo": ci["auc"][0], "auc_ci_hi": ci["auc"][1],
                     "bss": m["brier_skill_score"], "log_loss": m["log_loss"], "ece": m["ece"],
                     "penalty_xg": float(p[p.shot_id.isin(pen_ids)].p.mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "before_after.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
