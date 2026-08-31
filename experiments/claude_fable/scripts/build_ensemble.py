"""Seed-ensemble every variant, then combine variants with weights fitted on validation.

    ./run.sh python scripts/build_ensemble.py --results results/premerge --out results/ensemble

Two levels, both honest about where the test set is allowed to be used:
  * within a variant, seeds are averaged in logit space (a plain mean, no fitting);
  * across variants, a non-negative logistic stack is fitted on the *validation*
    predictions only, and the test set is touched once, at the end, to report.

Previous rounds used an unweighted average of whatever looked good on the test table,
which quietly turns model selection into a test-set fit. Any variant lacking
validation predictions is therefore refused rather than silently included.

Selection between candidates uses a game-grouped cross-validated log-loss computed
inside the validation set, so a bigger stack cannot win by fitting validation harder.
Single models are scored on plain validation log-loss; the only thing they fitted
there is one temperature scalar, so the optimism is negligible next to a k-weight stack.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from compare import bootstrap  # noqa: E402

from xgpoc.eval.metrics import evaluate  # noqa: E402

INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def collect(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """variant -> {'valid': df, 'test': df} with seeds averaged in logit space.

    Grouping is by (directory, model name), not by directory alone: one directory can
    hold several different models - the control probes do - and averaging those
    together as if they were seeds of one model would be meaningless."""
    found: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    models_per_dir: dict[str, set[str]] = {}
    for f in sorted(root.glob("**/*_predictions.csv")):
        split = "valid" if f.name.endswith("_valid_predictions.csv") else "test"
        model = f.name.replace(f"_{split}_predictions.csv", "")
        parent = f.parent
        while parent.name.startswith("seed"):
            parent = parent.parent
        variant = str(parent.relative_to(root))
        models_per_dir.setdefault(variant, set()).add(model)
        found.setdefault((variant, model, split), []).append(pd.read_csv(f).drop_duplicates("shot_id"))
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for (variant, model, split), dfs in found.items():
        label = variant if len(models_per_dir[variant]) == 1 else f"{variant}:{model}"
        base = dfs[0][["shot_id", "goal"]]
        z = np.mean([logit(d.set_index("shot_id").loc[base.shot_id, "p"].to_numpy()) for d in dfs], axis=0)
        out.setdefault(label, {})[split] = base.assign(p=1 / (1 + np.exp(-z)), seeds=len(dfs))
    return out


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Logistic stack on logits, weights clipped to non-negative: a negative weight
    would mean "invert this model", which is never what an ensemble should learn from
    2 000 validation rows."""
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
    w, b = lr.coef_[0], float(lr.intercept_[0])
    if (w < 0).any():
        w = np.clip(w, 0, None)
        w = w if w.sum() else np.ones_like(w)
        w = w / w.sum() * max(lr.coef_[0].sum(), 1e-3)
    return w, b


def stack(variants: list[str], data: dict, games: pd.DataFrame, n_folds: int = 5) -> dict:
    va = [data[v]["valid"].set_index("shot_id") for v in variants]
    te = [data[v]["test"].set_index("shot_id") for v in variants]
    va_ids = sorted(set.intersection(*[set(d.index) for d in va]))
    te_ids = sorted(set.intersection(*[set(d.index) for d in te]))
    xv = np.column_stack([logit(d.loc[va_ids, "p"].to_numpy()) for d in va])
    xt = np.column_stack([logit(d.loc[te_ids, "p"].to_numpy()) for d in te])
    yv = va[0].loc[va_ids, "goal"].to_numpy().astype(int)
    yt = te[0].loc[te_ids, "goal"].to_numpy().astype(int)

    # selection score: game-grouped CV inside validation, so that adding members cannot
    # buy a better number through in-sample fit alone
    g = games.set_index("shot_id").loc[va_ids, "game"].to_numpy()
    uniq = np.unique(g)
    fold_of = {game: i % n_folds for i, game in enumerate(uniq)}
    folds = np.array([fold_of[x] for x in g])
    oof = np.zeros(len(yv))
    for f in range(n_folds):
        tr, ho = folds != f, folds == f
        if yv[tr].min() == yv[tr].max() or not ho.any():
            oof[ho] = yv.mean()
            continue
        w, b = _fit(xv[tr], yv[tr])
        oof[ho] = 1 / (1 + np.exp(-(xv[ho] @ w + b)))
    cv = evaluate(yv, oof)

    w, b = _fit(xv, yv)
    p = 1 / (1 + np.exp(-(xt @ w + b)))
    m = evaluate(yt, p)
    pred = pd.DataFrame({"shot_id": te_ids, "goal": yt, "p": p})
    ci = bootstrap(pred.merge(games, on="shot_id"))
    return {"members": "+".join(variants), "k": len(variants), "n": m["n"],
            "weights": [round(float(x), 3) for x in w], "valid_cv_log_loss": cv["log_loss"],
            "valid_cv_auc": cv["auc"], "auc": m["auc"], "auc_ci": ci["auc"],
            "log_loss": m["log_loss"], "log_loss_ci": ci["log_loss"], "brier": m["brier"],
            "bss": m["brier_skill_score"], "ece": m["ece"], "_pred": pred}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results/premerge"])
    ap.add_argument("--out", type=Path, default=Path("results/ensemble"))
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="restrict candidate variants")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(a.index)[["shot_id", "game"]].drop_duplicates("shot_id")

    data: dict[str, dict[str, pd.DataFrame]] = {}
    for root in a.results:
        for k, v in collect(Path(root)).items():
            data[k] = {**data.get(k, {}), **v}
    usable = [v for v, d in data.items() if {"valid", "test"} <= set(d)]
    skipped = sorted(set(data) - set(usable))
    if a.only:
        usable = [v for v in usable if v in a.only]
    usable = sorted(usable)
    print(f"candidates: {usable}")
    if skipped:
        print(f"skipped (no validation predictions): {skipped}")

    singles = []
    for v in usable:
        te, va = data[v]["test"], data[v]["valid"]
        m = evaluate(te.goal.to_numpy(), te.p.to_numpy())
        mv = evaluate(va.goal.to_numpy(), va.p.to_numpy())
        ci = bootstrap(te.merge(games, on="shot_id"))
        singles.append({"members": v, "k": 1, "n": m["n"], "weights": [1.0],
                        "valid_cv_log_loss": mv["log_loss"], "valid_cv_auc": mv["auc"],
                        "auc": m["auc"], "auc_ci": ci["auc"], "log_loss": m["log_loss"],
                        "log_loss_ci": ci["log_loss"], "brier": m["brier"],
                        "bss": m["brier_skill_score"], "ece": m["ece"], "seeds": int(te.seeds.iloc[0])})
        te.to_csv(a.out / f"{v.replace('/', '__').replace(':', '__')}_test_predictions.csv", index=False)

    combos = []
    for k in range(2, a.max_k + 1):
        for c in combinations(usable, k):
            combos.append(stack(list(c), data, games))
    # the reported ensemble is chosen on the validation CV score, never on the test one
    best = min(combos + singles, key=lambda r: r["valid_cv_log_loss"]) if combos else None
    if best is not None and "_pred" in best:
        best["_pred"].to_csv(a.out / "best_ensemble_test_predictions.csv", index=False)
    if best is not None:
        (a.out / "best_ensemble.json").write_text(json.dumps(
            {k: v for k, v in best.items() if k != "_pred"}, indent=2, default=float))
        print(f"\nchosen on validation CV: {best['members']} "
              f"(valid CV log-loss {best['valid_cv_log_loss']:.4f}) -> test log-loss "
              f"{best['log_loss']:.4f}, AUC {best['auc']:.4f}")
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_pred"} for r in singles + combos])
    table = table.sort_values("valid_cv_log_loss")
    table.to_csv(a.out / "ensemble_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.head(25).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
