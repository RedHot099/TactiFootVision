"""Per-feature ablation against the D (frozen DINOv2-L) head and the positional GBM.

    ./run.sh python scripts/run_feature_ablation.py --out results/feature_ablation

Each config trains 3 seeds; report = seed-ensemble metrics + mean xG on penalties.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare import bootstrap  # noqa: E402

from xgpoc.data.datasets import FeatureDataset
from xgpoc.data.index import load_index
from xgpoc.eval.metrics import evaluate
from xgpoc.features.aux import AUX_NAMES
from xgpoc.features.tabular import FEATURE_NAMES
from xgpoc.models.common import TrainConfig, predict_torch, train_torch
from xgpoc.models.d_foundation import AttentionPoolHead, extract_patch_features

CONFIGS = {
    "base": [],
    "pen": ["is_penalty"],
    "occl": ["occlusion", "occlusion_gk", "n_shadow_players", "occl_valid"],
    "gk": ["gkp_depth", "gkp_lateral", "gkp_dist_ball", "gkp_valid"],
    "air": ["ball_in_air", "ball_air_valid"],
    "fixdef": ["def_in_cone_fixed", "nearest_def_fixed", "teams_known"],
    "all": [c for c in AUX_NAMES],
    "sel": ["is_penalty", "ball_in_air", "ball_air_valid", "def_in_cone_fixed", "nearest_def_fixed", "teams_known"],
}
INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/feature_ablation"))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = load_index(Path(INDEX))
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    aux_df = pd.read_parquet("cache/full_v2_fallback/positional/aux.parquet").set_index("shot_id")
    tab = pd.read_parquet("cache/full_v2_fallback/positional/tabular.parquet").drop_duplicates("shot_id")
    feats = extract_patch_features(pd.concat([tr, va, te]), Path("cache/full_v1/features_dinov2_l.npz"), "dinov2_l")
    idx_games = df[["shot_id", "game"]]
    pen_ids = set(df[df.is_penalty & (df.split == "test")].shot_id)

    # standardise continuous aux columns on train stats
    cont = [c for c in AUX_NAMES if not c.endswith("_valid") and c not in ("is_penalty", "teams_known", "ball_in_air")]
    mu = aux_df.loc[aux_df.index.isin(tr.shot_id), cont].mean()
    sd = aux_df.loc[aux_df.index.isin(tr.shot_id), cont].std().replace(0, 1)
    aux_std = aux_df.copy()
    aux_std[cont] = (aux_df[cont] - mu) / sd

    rows = []
    for name, cols in CONFIGS.items():
        # --- D head with aux ---
        preds = []
        for seed in range(a.seeds):
            cfg = TrainConfig(epochs=100, max_minutes=10, lr=1e-4, seed=seed)
            aux_map = None
            if cols:
                aux_map = {sid: aux_std.loc[sid, cols].to_numpy(np.float32) for sid in aux_std.index}
            dim = next(iter(feats.values())).shape[-1]
            model = AttentionPoolHead(dim, aux_dim=len(cols))
            fit = train_torch(model, FeatureDataset(tr, feats, aux=aux_map), FeatureDataset(va, feats, aux=aux_map), cfg)
            ds = FeatureDataset(te, feats, aux=aux_map)
            preds.append(pd.DataFrame({"shot_id": ds.df.shot_id, "goal": ds.df.goal,
                                       "p": predict_torch(model, ds, cfg, fit["temperature"])}))
        base = preds[0][["shot_id", "goal"]]
        p = np.mean([d.set_index("shot_id").loc[base.shot_id, "p"].to_numpy() for d in preds], axis=0)
        ens = base.assign(p=p)
        m = evaluate(ens.goal.to_numpy(), ens.p.to_numpy())
        ci = bootstrap(ens.merge(idx_games, on="shot_id"))
        pen_xg = float(ens[ens.shot_id.isin(pen_ids)].p.mean()) if pen_ids else float("nan")
        ens.to_csv(a.out / f"D_{name}_test_predictions.csv", index=False)
        rows.append({"model": f"D+{name}", "auc": m["auc"], "auc_ci": ci["auc"], "bss": m["brier_skill_score"],
                     "log_loss": m["log_loss"], "ece": m["ece"], "penalty_xg": pen_xg})
        print(rows[-1])

        # --- GBM with aux ---
        from sklearn.ensemble import HistGradientBoostingClassifier
        base_cols = df[["shot_id", "split", "goal"]]
        gtab = base_cols.merge(tab, on="shot_id")
        if cols:
            gtab = gtab.merge(aux_df.reset_index()[["shot_id", *cols]], on="shot_id")
        cols_g = FEATURE_NAMES + cols
        gtr = gtab[gtab.split != "test"]
        gte = gtab[gtab.split == "test"]
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.15, random_state=0)
        g.fit(gtr[cols_g], gtr.goal)
        gp = g.predict_proba(gte[cols_g])[:, 1]
        gens = pd.DataFrame({"shot_id": gte.shot_id, "goal": gte.goal, "p": gp})
        gm = evaluate(gens.goal.to_numpy(), gens.p.to_numpy())
        gpen = float(gens[gens.shot_id.isin(pen_ids)].p.mean()) if pen_ids else float("nan")
        rows.append({"model": f"GBM+{name}", "auc": gm["auc"], "auc_ci": None, "bss": gm["brier_skill_score"],
                     "log_loss": gm["log_loss"], "ece": gm["ece"], "penalty_xg": gpen})
        print(rows[-1])

    out = pd.DataFrame(rows)
    out.to_csv(a.out / "ablation.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
