"""Recompute the September audit from saved predictions, without training or selection.

Run with ./run.sh python scripts/audit_results.py. Inputs stay untouched.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from compare import bootstrap
from significance import paired_bootstrap
from xgpoc.eval.metrics import evaluate
from xgpoc.paths import INDEX

HOME_RE = r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ "
MEMBERS = ["c_combo", "g_combo", "g_combo_mix", "dl_mask", "gbm"]


def read_predictions(path, expected_ids):
    df = pd.read_csv(path)
    assert not df.shot_id.duplicated().any(), path
    assert set(df.shot_id) == set(expected_ids), path
    assert np.isfinite(df.p).all() and df.p.between(0, 1).all(), path
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("results_reports/audit_2026-09-07"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(a.index)
    assert not idx.shot_id.duplicated().any()
    assert idx.groupby("game").split.nunique().max() == 1
    idx["home"] = idx.game.str.extract(HOME_RE)[0]
    assert idx.home.notna().all()
    test = idx[idx.split == "test"]
    pool = idx[idx.split != "test"]
    labels = idx.set_index("shot_id").goal.astype(int)
    folds = pd.read_csv("cache/folds_v12_home5.csv")
    assert set(folds.game) == set(pool.game)
    assert not folds.game.duplicated().any()
    assert folds.groupby("home").fold.nunique().max() == 1
    coverage = []
    for member in MEMBERS:
        files = sorted(Path("results/v12_cv", member).glob("fold*/seed*/*_valid_predictions.csv"))
        counts = []
        for f in files:
            fold = int(f.parent.parent.name.removeprefix("fold"))
            ids = pool[pool.game.isin(folds[folds.fold == fold].game)].shot_id
            va = read_predictions(f, ids)
            te = read_predictions(Path(str(f).replace("_valid_", "_test_")), test.shot_id)
            for df in (va, te):
                assert np.array_equal(df.goal.astype(int), labels.loc[df.shot_id]), f
            counts.append(va.shot_id)
        count = pd.concat(counts).value_counts()
        assert set(count.index) == set(pool.shot_id) and count.nunique() == 1, member
        coverage.append({"member": member, "models": len(files), "oof_predictions_per_shot": int(count.iloc[0]),
                         "oof_shots": len(count), "test_shots_per_model": len(test)})
    pd.DataFrame(coverage).to_csv(a.out / "coverage.csv", index=False)

    root = Path("results/v12_cv_ensemble")
    paths = {"FINAL": root / "best_ensemble_test_predictions.csv",
             "G_combo": root / "g_combo_test_predictions.csv",
             "DL_cv": root / "dl_mask_test_predictions.csv",
             "GBM": root / "gbm_test_predictions.csv",
             "OLD": Path("results/ensemble/best_ensemble_test_predictions.csv")}
    preds, metrics = {}, []
    for name, path in paths.items():
        p = read_predictions(path, test.shot_id)
        stale = int((p.goal.astype(int).to_numpy() != labels.loc[p.shot_id].to_numpy()).sum())
        p["goal"] = labels.loc[p.shot_id].to_numpy()
        preds[name] = p
        for group, ids in [("all", test.shot_id), ("non_penalty", test[~test.is_penalty].shot_id)]:
            d = p[p.shot_id.isin(ids)]
            metrics.append({"model": name, "group": group, **evaluate(d.goal, d.p),
                            "mean_xg": float(d.p.mean()), "goals": int(d.goal.sum()),
                            "sum_xg": float(d.p.sum()), "csv_labels_replaced": stale})
    pd.DataFrame(metrics).to_csv(a.out / "metrics.csv", index=False)
    final = preds["FINAL"].drop(columns="goal").merge(test, on="shot_id", validate="one_to_one")
    absolute_ci = bootstrap(final, n_boot=a.n_boot)
    pairs = []
    for first, second in [("FINAL", "OLD"), ("FINAL", "G_combo"), ("G_combo", "DL_cv")]:
        result = paired_bootstrap(preds[first], preds[second], idx[["shot_id", "game", "goal"]], a.n_boot)
        pairs.append({"first": first, "second": second, **result})
        print(first, second, "paired AUC CI", result["d_auc_lo"], result["d_auc_hi"], flush=True)
    pd.DataFrame(pairs).to_csv(a.out / "paired.csv", index=False)

    subgroups = []
    for scope, seen in [("official_train", set(idx[idx.split == "train"].home)), ("train_valid", set(pool.home))]:
        for known in (True, False):
            d = final[final.home.isin(seen) == known]
            subgroups.append({"scope": scope, "known_home": known, "games": d.game.nunique(),
                              "goals": int(d.goal.sum()), **evaluate(d.goal, d.p)})
    pd.DataFrame(subgroups).to_csv(a.out / "venues.csv", index=False)
    tab = pd.read_parquet("cache/full_v2_fallback/positional/tabular.parquet")
    aux = pd.read_parquet("cache/full_v2_fallback/positional/aux.parquet")
    feature_quality = {c: float(tab[c].mean()) for c in ("has_ball", "homography_ok", "gk_present", "attack_team_known")}
    feature_quality.update({c: float(aux[c].mean()) for c in ("gkp_valid", "occl_valid", "teams_known")})

    curves = []
    for stage in ("dl_mask", "g_l4"):
        for f in sorted(Path("results/v12_lc", stage).glob("frac*/seed*/*_metrics.json")):
            m = json.loads(f.read_text())
            curves.append({"stage": stage, "games": int(f.parent.parent.name.removeprefix("frac")) * 3,
                           "seed": f.parent.name, "auc": m["auc"], "log_loss": m["log_loss"],
                           "brier": m["brier"]})
    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(a.out / "learning_curve.csv", index=False)
    temporal = []
    for path in sorted(Path("results/night_v1/f").glob("seed*/*_test_predictions.csv")):
        p = read_predictions(path, test.shot_id)
        temporal.append({"seed": path.parent.name, **evaluate(labels.loc[p.shot_id], p.p)})
    pd.DataFrame(temporal).to_csv(a.out / "temporal_on_v12.csv", index=False)
    summary = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "index": str(a.index), "index_sha256": hashlib.sha256(a.index.read_bytes()).hexdigest(),
               "prediction_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
               "split_counts": idx.groupby("split").agg(shots=("shot_id", "size"), games=("game", "nunique"),
                                                         goals=("goal", "sum"), penalties=("is_penalty", "sum")).to_dict("index"),
               "final_ci": absolute_ci, "bootstrap_resamples": a.n_boot, "bootstrap_seed": 0,
               "feature_quality": feature_quality,
               "checkpoints_in_experiment": [str(p) for p in Path(".").rglob("*") if p.is_file() and p.suffix in {".pt", ".pth", ".ckpt", ".joblib"}],
               "note": "Saved-prediction audit. No retraining, model selection or inference benchmark."}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    plot(curve_df, pd.DataFrame(pairs), a.out / "audit.png")
    print(json.dumps(summary, indent=2, default=float))


def plot(curves, pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for stage, rows in curves.groupby("stage"):
        means = rows.groupby("games")[["auc", "log_loss"]].mean()
        for ax, metric in zip(axes[:2], ("auc", "log_loss")):
            ax.plot(means.index, means[metric], "o-", label=stage)
            ax.set(xlabel="Mecze treningowe", ylabel=metric)
            ax.grid(alpha=0.2)
    axes[0].set_title("Ranking rośnie wraz z liczbą meczów")
    axes[1].set_title("Log-loss G poprawia się wolniej")
    axes[0].legend()
    for i, row in pairs.iterrows():
        axes[2].errorbar(row.d_auc, i, xerr=[[row.d_auc - row.d_auc_lo], [row.d_auc_hi - row.d_auc]], fmt="o", capsize=4)
    axes[2].axvline(0, color="gray", linestyle="--")
    axes[2].set(yticks=range(len(pairs)), yticklabels=pairs["first"] + " vs " + pairs["second"],
                xlabel="Różnica AUC, 95% CI po meczach", title="Istotność dotyczy konkretnej pary")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
