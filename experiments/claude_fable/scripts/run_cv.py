"""Home-team-grouped 5-fold cross-validation over train+valid (400 games).

    ./run.sh python scripts/run_cv.py --stages gbm dl_mask g_l4 --seeds 1

For every stage of run_v12.STAGES and every fold k: train on the other folds, write the
held-out fold's predictions (out-of-fold) as `*_valid_predictions.csv` and the test set's
as `*_test_predictions.csv` under results/v12_cv/<stage>/fold<k>/seed<s>/. The ensemble
builder then concatenates the OOF predictions (400 games, pooled base rate) to fit
calibration and stack weights, and averages the five fold models on the test set.

Folds are grouped by the home team, not by game: 85 of the 100 test games share a home
team (stadium, broadcaster) with training games and the fine-tuned model lost 5 pp AUC on
the other 15, so a game-level CV would overstate what transfers to new venues. Epoch
selection and the temperature still use the held-out fold, which makes the OOF numbers
slightly optimistic; the test set is never touched before the final report.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import run_poc as rp
from run_v12 import CACHE, STAGES, done
from xgpoc.data.index import load_index
from xgpoc.features.tabular import FEATURE_NAMES
from xgpoc.models.common import TrainConfig, write_result
from xgpoc.paths import INDEX

OUT = Path("results/v12_cv")
FOLDS = Path("cache/folds_v12_home5.csv")
HOME_RE = r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ "
POS_CACHE = Path("cache/full_v2_fallback/positional")
GBM_AUX = ["ball_in_air", "ball_air_valid", "def_in_cone_fixed", "nearest_def_fixed", "teams_known"]


def fold_table(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """game -> fold, grouped by home team; cached so every stage uses identical folds."""
    if FOLDS.exists():
        return pd.read_csv(FOLDS)
    pool = df[df.split != "test"].drop_duplicates("game")[["game"]].sort_values("game").reset_index(drop=True)
    pool["home"] = pool.game.str.extract(HOME_RE)[0].fillna(pool.game)
    pool["fold"] = -1
    for k, (_, ho) in enumerate(GroupKFold(n_folds).split(pool, groups=pool.home)):
        pool.loc[ho, "fold"] = k
    FOLDS.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(FOLDS, index=False)
    return pool


def run_gbm(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, out: Path, seed: int) -> None:
    tab = pd.read_parquet(POS_CACHE / "tabular.parquet").drop_duplicates("shot_id")
    aux = pd.read_parquet(POS_CACHE / "aux.parquet").drop_duplicates("shot_id")[["shot_id", *GBM_AUX]]
    feats = FEATURE_NAMES + ["is_penalty", *GBM_AUX]

    def prep(d):
        d = d[["shot_id", "goal", "is_penalty"]].assign(is_penalty=lambda x: x.is_penalty.astype(float))
        return d.merge(tab, on="shot_id").merge(aux, on="shot_id")

    a, b, c = prep(tr), prep(va), prep(te)
    g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15, random_state=seed)
    g.fit(a[feats], a.goal)
    write_result(out, "a_tabular_gbm_sel", c, g.predict_proba(c[feats])[:, 1], {"n_train": len(a), "features": feats},
                 valid_df=b, p_valid=g.predict_proba(b[feats])[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["gbm", "dl_mask", "c_mask", "g_l4"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="subset of folds to run")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args()
    df = rp.add_targets(load_index(Path(a.index)))
    folds = fold_table(df)
    pool = df[df.split != "test"].merge(folds[["game", "fold"]], on="game")
    te = df[df.split == "test"].reset_index(drop=True)
    summary = pool.groupby("fold").agg(games=("game", "nunique"), shots=("shot_id", "size"), goal_rate=("goal", "mean"))
    summary["homes"] = folds.groupby("fold").home.nunique()
    print(f"index {a.index}; folds by home team:\n{summary.round(4)}", flush=True)
    which = a.folds if a.folds is not None else sorted(pool.fold.unique())

    for st in a.stages:
        for k in which:
            tr, va = (pool[pool.fold != k].reset_index(drop=True), pool[pool.fold == k].reset_index(drop=True))
            for s in range(a.seeds):
                out = a.out / st / f"fold{k}" / f"seed{s}"
                if st == "gbm":
                    if not done(out, "a_tabular_gbm_sel"):
                        run_gbm(tr, va, te, out, s)
                    continue
                runner, name, cfg_over, kw = STAGES[st]
                if done(out, name):
                    print(f"skip {out}/{name}")
                    continue
                print(f"=== {out}/{name} ({time.strftime('%H:%M')}) train {len(tr)} oof {len(va)} ===", flush=True)
                t0 = time.time()
                rp.RUNNERS[runner](tr, va, te, CACHE, out, TrainConfig(seed=s, **cfg_over), **kw)
                print(f"--- {name} fold {k} seed {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = []
    for f in sorted(a.out.glob("**/*_metrics.json")):
        m = pd.read_json(f, typ="series")
        rows.append({"run": str(f.parent.relative_to(a.out)), "model": m["model"], "test_auc": m["auc"],
                     "test_log_loss": m["log_loss"], "best_epoch": m.get("best_epoch", np.nan)})
    if rows:
        pd.set_option("display.width", 250)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"CV QUEUE DONE ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
