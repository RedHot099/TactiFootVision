"""Seed-ensemble every variant, then combine variants with weights fitted on validation.

    ./run.sh python scripts/build_ensemble.py --results results/v12 --out results/v12_ensemble \
        --prior-shift --penalty-route

Two levels, both honest about where the test set is allowed to be used:
  * within a variant, seeds are averaged in logit space (a plain mean, no fitting);
  * across variants, a non-negative logistic stack is fitted on the *validation*
    predictions only, and the test set is touched once, at the end, to report.

Selection between candidates uses a game-grouped cross-validated log-loss computed
inside the validation set, so a bigger stack cannot win by fitting validation harder.
Single models are scored on plain validation log-loss; the only thing they fitted
there is one temperature scalar, so the optimism is negligible next to a k-weight stack.

Post-processing of the reported test predictions (off by default, both recommended):
  --prior-shift    the validation split has 14.9% goals against 13.1% in train and 13.3%
                   in test; anything calibrated on it inherits that prior. Predictions are
                   shifted in logit space to the pooled train+valid rate.
  --penalty-route  penalties are a known input condition; they get the Beta-posterior
                   conversion rate from the training penalties instead of the model's guess.
Selection is done *before* post-processing (on validation), so the choice of members
cannot depend on it.
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

from xgpoc.eval.calibration import PlattScaler, logit, penalty_rate, prior_shift, route_penalties  # noqa: E402
from xgpoc.eval.metrics import evaluate  # noqa: E402
from xgpoc.paths import INDEX as _INDEX  # noqa: E402

INDEX = str(_INDEX)


def collect(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """variant -> {'valid': df, 'test': df}, averaged in logit space per shot.

    Directory layout: <variant>/[fold<k>/][seed<s>/]<model>_{valid,test}_predictions.csv.
    Averaging is per shot_id, which covers both layouts at once: with seeds only, every
    shot appears once per seed; with cross-validation folds the validation files are
    disjoint (each shot is out-of-fold exactly once per seed) and are concatenated into one
    OOF frame, while the test files (one per fold model) are averaged.

    Grouping is by (directory, model name), not by directory alone: one directory can
    hold several different models - the control probes do - and averaging those
    together as if they were seeds of one model would be meaningless."""
    found: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    models_per_dir: dict[str, set[str]] = {}
    for f in sorted(root.glob("**/*_predictions.csv")):
        split = "valid" if f.name.endswith("_valid_predictions.csv") else "test"
        model = f.name.replace(f"_{split}_predictions.csv", "")
        parent = f.parent
        while parent.name.startswith(("seed", "fold")):
            parent = parent.parent
        variant = str(parent.relative_to(root))
        models_per_dir.setdefault(variant, set()).add(model)
        found.setdefault((variant, model, split), []).append(pd.read_csv(f).drop_duplicates("shot_id")[["shot_id", "p"]])
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for (variant, model, split), dfs in found.items():
        label = variant if len(models_per_dir[variant]) == 1 else f"{variant}:{model}"
        allp = pd.concat(dfs, ignore_index=True)
        allp["z"] = logit(allp.p.to_numpy())
        agg = allp.groupby("shot_id", sort=True).z.agg(["mean", "size"]).reset_index()
        out.setdefault(label, {})[split] = pd.DataFrame({"shot_id": agg.shot_id, "p": 1 / (1 + np.exp(-agg["mean"])),
                                                         "seeds": agg["size"]})
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


def stack(variants: list[str], data: dict, labels: pd.Series, games: pd.DataFrame, n_folds: int = 5) -> dict:
    """Fit on validation, score the selection metric by game-grouped CV inside
    validation, return *raw* test predictions (post-processing happens in main)."""
    va = [data[v]["valid"].set_index("shot_id") for v in variants]
    te = [data[v]["test"].set_index("shot_id") for v in variants]
    va_ids = sorted(set.intersection(*[set(d.index) for d in va]))
    te_ids = sorted(set.intersection(*[set(d.index) for d in te]))
    xv = np.column_stack([logit(d.loc[va_ids, "p"].to_numpy()) for d in va])
    xt = np.column_stack([logit(d.loc[te_ids, "p"].to_numpy()) for d in te])
    yv = labels.loc[va_ids].to_numpy().astype(int)
    yt = labels.loc[te_ids].to_numpy().astype(int)

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
    return {"members": "+".join(variants), "k": len(variants), "weights": [round(float(x), 3) for x in w],
            "valid_cv_log_loss": cv["log_loss"], "valid_cv_auc": cv["auc"], "_from_rate": float(yv.mean()),
            "_pred": pd.DataFrame({"shot_id": te_ids, "goal": yt, "p": p})}


def score(pred: pd.DataFrame, games: pd.DataFrame) -> dict:
    m = evaluate(pred.goal.to_numpy(), pred.p.to_numpy())
    ci = bootstrap(pred.merge(games, on="shot_id"))
    return {"n": m["n"], "auc": m["auc"], "auc_ci": ci["auc"], "log_loss": m["log_loss"], "log_loss_ci": ci["log_loss"],
            "brier": m["brier"], "bss": m["brier_skill_score"], "ece": m["ece"], "mean_xg": float(pred.p.mean()),
            "goal_rate": m["goal_rate"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results/premerge"])
    ap.add_argument("--out", type=Path, default=Path("results/ensemble"))
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="restrict candidate variants")
    ap.add_argument("--prior-shift", action="store_true")
    ap.add_argument("--penalty-route", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(a.index).drop_duplicates("shot_id")
    idx["goal"] = idx.goal.astype(bool)
    games = idx[["shot_id", "game"]]
    labels = idx.set_index("shot_id")["goal"]          # labels come from the index, never from the CSVs
    pooled_rate = float(idx[idx.split != "test"].goal.mean())
    valid_rate = float(idx[idx.split == "valid"].goal.mean())
    pen_train = idx[(idx.split == "train") & idx.is_penalty]
    p_pen = penalty_rate(int(pen_train.goal.sum()), len(pen_train))
    is_pen = idx.set_index("shot_id")["is_penalty"].astype(bool)

    def post(pred: pd.DataFrame, from_rate: float) -> pd.DataFrame:
        p = pred.p.to_numpy()
        if a.prior_shift:
            p = prior_shift(p, from_rate, pooled_rate)
        if a.penalty_route:
            p = route_penalties(p, is_pen.loc[pred.shot_id].to_numpy(), p_pen)
        return pred.assign(p=p)

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
    print(f"pooled train+valid goal rate {pooled_rate:.4f}, official valid {valid_rate:.4f}, penalty rate {p_pen:.3f} "
          f"(prior_shift={a.prior_shift}, penalty_route={a.penalty_route})")

    rows = []
    for v in usable:
        te, va = data[v]["test"].copy(), data[v]["valid"].copy()
        yv = labels.loc[va.shot_id].to_numpy()
        from_rate = float(yv.mean())                    # base rate of the frame the calibration is fitted on
        mv = evaluate(yv, va.p.to_numpy())
        # Platt on validation logits (scale + offset), then the same post-processing as the stacks
        platt = PlattScaler().fit(logit(va.p.to_numpy()), yv)
        pred = post(pd.DataFrame({"shot_id": te.shot_id, "goal": labels.loc[te.shot_id].to_numpy().astype(int),
                                  "p": platt(logit(te.p.to_numpy()))}), from_rate)
        raw = score(pd.DataFrame({"shot_id": te.shot_id, "goal": pred.goal, "p": te.p}), games)
        rows.append({"members": v, "k": 1, "weights": [1.0], "valid_cv_log_loss": mv["log_loss"], "valid_cv_auc": mv["auc"],
                     **score(pred, games), "auc_raw": raw["auc"], "log_loss_raw": raw["log_loss"], "seeds": int(te.seeds.iloc[0])})
        pred.to_csv(a.out / f"{v.replace('/', '__').replace(':', '__')}_test_predictions.csv", index=False)

    combos = []
    for k in range(2, a.max_k + 1):
        for c in combinations(usable, k):
            s = stack(list(c), data, labels, games)
            pred = post(s.pop("_pred"), s.pop("_from_rate"))
            combos.append({**s, **score(pred, games), "_pred": pred})
    best = min(combos + rows, key=lambda r: r["valid_cv_log_loss"]) if combos else None
    if best is not None:
        if "_pred" in best:
            best["_pred"].to_csv(a.out / "best_ensemble_test_predictions.csv", index=False)
        (a.out / "best_ensemble.json").write_text(json.dumps(
            {k: v for k, v in best.items() if k != "_pred"} | {"prior_shift": a.prior_shift, "penalty_route": a.penalty_route,
                                                              "pooled_rate": pooled_rate, "valid_rate": valid_rate, "p_pen": p_pen},
            indent=2, default=float))
        print(f"\nchosen on validation CV: {best['members']} (valid CV log-loss {best['valid_cv_log_loss']:.4f}) -> "
              f"test log-loss {best['log_loss']:.4f}, AUC {best['auc']:.4f}, mean xG {best['mean_xg']:.4f}")
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_pred"} for r in rows + combos]).sort_values("valid_cv_log_loss")
    table.to_csv(a.out / "ensemble_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.head(25).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
