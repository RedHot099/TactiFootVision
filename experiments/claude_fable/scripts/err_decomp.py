"""Error decomposition of the sweep (2026-10-22, owner-ordered, descriptive): where does stack_ALL_new still lose log-loss,
split by the retrained detector's goalkeeper detection and its shot-location / goalkeeper-position error against
StatsBomb (validation only; no model sees StatsBomb). Reads results/sweep_2026_10 and cache/xgd_full4/E3 (t0 rows).
Data: StatsBomb."""

import numpy as np
import pandas as pd
from xgpoc.detector.conventions import YARD, statsbomb_to_metres
from xgs_test import read_test_labels

OUT = "results/err_decomp"
CLIP = 1e-6


def ll(y, p):
    q = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def boot_mean(d, games, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.Series(d).groupby(games)
    s, c = g.sum().to_numpy(), g.size().to_numpy()
    idx = rng.integers(0, len(s), (n, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(b, [2.5, 97.5])


def main():
    pred = pd.read_csv("results/sweep_2026_10/predictions.csv").set_index("shot_id")
    y = read_test_labels(pd.Series(pred.index)).astype(float)
    e3 = pd.read_parquet("cache/xgd_full4/E3/features.parquet")
    e3 = e3[e3.offset_ms == 0].set_index("shot_id").reindex(pred.index)
    sb = pd.read_parquet("results/xgc_compare/table/shots.parquet")
    sb = (
        sb[sb.population == "primary"]
        .set_index("soccernet_shot_id")
        .reindex(pred.index)
    )

    d = pd.DataFrame(index=pred.index)
    d["game"] = pred.game
    d["y"] = y
    for m in ("g_combo", "stack_ALL_new", "pos_ALL_new"):
        d[f"ll_{m}"] = ll(y, pred[f"p_{m}"].to_numpy())
    d["ll_vendor"] = ll(y, 1 / (1 + np.exp(-sb.z_vendor.to_numpy(float))))
    d["gk_detected"] = e3.gk_detected
    d["gk_ok"] = e3.gk_ok
    d["shot_ok"] = e3.shot_ok
    d["ball_ok"] = e3.ball_ok
    sbm = statsbomb_to_metres(sb[["sb_x", "sb_y"]].to_numpy(float))
    d["shot_err_m"] = np.hypot(e3.shot_x - sbm[:, 0], e3.shot_y - sbm[:, 1])
    d["dist_err_m"] = (e3.dist - sb.sb_dist * YARD).abs()
    d["gk_depth_err_m"] = (e3.gk_depth - sb.sb_gk_depth * YARD).abs()
    d["sb_paired"] = sb.sb_x.notna()
    d.to_parquet(f"{OUT}/shots.parquet")

    def strata(name, key, mask=None):
        m = d if mask is None else d[mask]
        rows = []
        for lvl, s in m.groupby(key, dropna=False, observed=True):
            delta = s.ll_stack_ALL_new - s.ll_g_combo
            lo, hi = boot_mean(delta.to_numpy(), s.game.to_numpy())
            row = {
                "split": name,
                "level": str(lvl),
                "shots": len(s),
                "goal_rate": s.y.mean(),
                "LL_g_combo": s.ll_g_combo.mean(),
                "LL_stack_ALL_new": s.ll_stack_ALL_new.mean(),
                "LL_pos_ALL_new": s.ll_pos_ALL_new.mean(),
                "d_stack_vs_gcombo": delta.mean(),
                "lo": lo,
                "hi": hi,
                "share_of_total_LL_stack": s.ll_stack_ALL_new.sum()
                / m.ll_stack_ALL_new.sum(),
            }
            if s.sb_paired.all():
                row["LL_vendor"] = s.ll_vendor.mean()
                row["gap_stack_minus_vendor"] = (
                    s.ll_stack_ALL_new - s.ll_vendor
                ).mean()
            rows.append(row)
        return rows

    paired = d.sb_paired
    d["shot_err_bin"] = pd.cut(
        d.shot_err_m,
        [0, 1, 2, 3, 5, np.inf],
        right=False,
        labels=["<1m", "1-2m", "2-3m", "3-5m", ">=5m"],
    ).cat.add_categories("missing")
    d.loc[paired & d.shot_err_m.isna(), "shot_err_bin"] = "missing"
    d["gk_err_bin"] = pd.cut(
        d.gk_depth_err_m,
        [0, 1, 2, 5, np.inf],
        right=False,
        labels=["<1m", "1-2m", "2-5m", ">=5m"],
    ).cat.add_categories("missing")
    d.loc[paired & d.gk_depth_err_m.isna(), "gk_err_bin"] = "missing"
    rows = []
    rows += strata("all172:gk_detected", "gk_detected")
    rows += strata("all172:shot_ok", "shot_ok")
    rows += strata("sb_paired:gk_detected", "gk_detected", paired)
    rows += strata("sb_paired:shot_err", "shot_err_bin", paired)
    rows += strata("sb_paired:gk_depth_err", "gk_err_bin", paired)
    rows += strata("sb_paired:all", pd.Series("all", index=d.index), paired)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/strata.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    p = d[paired]
    print(
        "\nSB-paired shots:",
        len(p),
        "| median shot err m:",
        p.shot_err_m.median(),
        "| median gk depth err m:",
        p.gk_depth_err_m.median(),
        "| gk detected:",
        p.gk_detected.mean(),
    )


if __name__ == "__main__":
    main()
