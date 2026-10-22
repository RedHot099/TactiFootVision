"""Pre-shot frames test (2026-10-15 chat result, persisted 2026-10-22): does a single-frame G-combo gain from the frames
before the shot? Development result, CPU, no model run.

    ./run.sh --no-sync python scripts/preshot_frames.py [--out results/preshot_frames_2026_10]

Input: the frozen per-frame G-combo logits of xg-credibility ticket 08 (`results/xgc_offset_scores/predictions.parquet`,
checked against its completion record): 30 frames per shot from -1600 to +1000 ms, each shot scored by G-combo models
that never saw its match (out-of-fold for the training matches, the 10-model ensemble for the held-out ones; the
held-out 172 matches are spent). Labels: the index `goal` (v1.2), every shot of the table.

Population A: non-penalty shots with a G-combo logit at every pre-contact frame from -400 to 0 ms (11 frames). Each
window's score is the mean of the available logits of its frames (no frame after 0 ms is ever used), read as a
probability (raw LL) and after a Platt refit by 5-fold match-grouped CV (recal-CV LL; logistic regression, C = 1e6).
Population B (as the 2026-10-15 run): non-penalty shots with logits at 0, -200, -400 and -800 ms; data-fitted weights
of the t0 logit plus earlier logits (logistic regression on the logits, the same CV) minus the t0 logit alone (the
same CV), mean shot log-loss difference with a 95% match bootstrap (2,000 draws, seed 0, ratio of match sums).

Writes <out>/windows.csv, weights.csv, summary.json and complete.json (input and output hashes), last.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results/xgc_offset_scores/predictions.parquet"
OUT = ROOT / "results/preshot_frames_2026_10"
PRE = [-400, -360, -320, -280, -240, -200, -160, -120, -80, -40, 0]
WINDOWS = {"t0": [0], "-40": [-40], "-200..0 mean": [-200, -160, -120, -80, -40, 0], "-400..0 mean": PRE,
           "-800..0 mean": [-800, -600, *PRE], "-1600..0 mean": [-1600, -1400, -1200, -1000, -800, -600, *PRE]}
WEIGHTS = {"z0 + z(-200)": [0, -200], "z0 + z(-400)": [0, -400], "z0 + z(-400) + z(-800)": [0, -400, -800]}
B_FRAMES = [0, -400, -800, -200]
CLIP, FOLDS, DRAWS, SEED = 1e-6, 5, 2000, 0


def shot_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def cv_probability(x: np.ndarray, y: np.ndarray, games: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Out-of-fold probability of a logistic regression on `x` (5-fold, grouped by match)."""
    out = np.zeros(len(y))
    for train, test in GroupKFold(FOLDS).split(x, y, games):
        out[test] = LogisticRegression(C=1e6, max_iter=max_iter).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    return out


def logit_table() -> pd.DataFrame:
    """Shot x offset G-combo logits with goal, is_penalty and game (every shot of the scored table)."""
    record = json.loads((SCORES.parent / "complete.json").read_text())
    if record["files"]["predictions.parquet"] != sha256_file(SCORES):
        raise SystemExit(f"{SCORES} does not match its completion record")
    p = pd.read_parquet(SCORES, columns=["shot_id", "game", "offset_ms", "encoder", "oof", "xg_logit"])
    g = p[(p.encoder == "g_combo") & p.oof]
    index = pd.read_parquet(INDEX, columns=["shot_id", "goal", "is_penalty"]).set_index("shot_id")
    w = g.pivot_table(index="shot_id", columns="offset_ms", values="xg_logit")
    return w.join(index, how="left").join(g.groupby("shot_id").game.first())


def windows(w: pd.DataFrame) -> pd.DataFrame:
    y, games = w.goal.astype(int).to_numpy(), w.game.to_numpy()
    rows = []
    for name, frames in WINDOWS.items():
        z = w[frames].mean(axis=1).to_numpy()
        rows.append({"window": name, "frames": len(frames), "shots": len(y), "goals": int(y.sum()),
                     "raw_ll": float(shot_ll(y, 1 / (1 + np.exp(-z))).mean()),
                     "recal_cv_ll": float(shot_ll(y, cv_probability(z[:, None], y, games, max_iter=1000)).mean())})
    return pd.DataFrame(rows)


def weights(w: pd.DataFrame) -> pd.DataFrame:
    y, games = w.goal.astype(int).to_numpy(), w.game.to_numpy()
    base = shot_ll(y, cv_probability(w[[0]].to_numpy(), y, games))
    uniq, codes = np.unique(games, return_inverse=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, frames in WEIGHTS.items():
        d = shot_ll(y, cv_probability(w[frames].to_numpy(), y, games)) - base
        per_match, n_match = np.bincount(codes, d, len(uniq)), np.bincount(codes, minlength=len(uniq))
        boot = []
        for _ in range(DRAWS):
            s = rng.integers(0, len(uniq), len(uniq))
            boot.append(per_match[s].sum() / n_match[s].sum())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": name, "base": "z0 (same CV)", "shots": len(y), "matches": len(uniq),
                     "delta_ll": float(d.mean()), "lo95": float(lo), "hi95": float(hi), "draws": DRAWS})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    w = logit_table()
    if w.goal.isna().any():
        raise SystemExit("a scored shot has no index label")
    w = w[~w.is_penalty.astype(bool)]
    a, b = w.dropna(subset=PRE), w.dropna(subset=B_FRAMES)
    win, wts = windows(a), weights(b)
    args.out.mkdir(parents=True)
    win.to_csv(args.out / "windows.csv", index=False, float_format="%.6f")
    wts.to_csv(args.out / "weights.csv", index=False, float_format="%.6f")
    summary = {"label": "development result: every match was seen before; the 172 held-out matches are spent",
               "population_a": {"rule": "non-penalty, G-combo logit at every frame -400..0 ms", "shots": len(a),
                                "goals": int(a.goal.sum()), "matches": int(a.game.nunique())},
               "population_b": {"rule": "non-penalty, G-combo logit at 0, -200, -400, -800 ms", "shots": len(b),
                                "goals": int(b.goal.sum()), "matches": int(b.game.nunique())},
               "windows": win.set_index("window").to_dict(orient="index"),
               "weights": wts.set_index("model").to_dict(orient="index"),
               "chat_2026_10_15": {"t0": 0.3401, "-200..0 mean": 0.3404, "-400..0 mean": 0.3415, "-1600..0 mean": 0.3461,
                                   "z0 + z(-400)": [-0.0004, -0.0010, 0.0002], "shots": 10680}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    complete = {"inputs": {"offset_scores": {"path": str(SCORES.relative_to(ROOT)), "sha256": sha256_file(SCORES)},
                           "index": {"path": str(INDEX), "sha256": sha256_file(INDEX)}},
                "code": {str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__).resolve())},
                "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"),
                        "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=1) + "\n")
    print(win.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(wts.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(json.dumps({k: summary[k] for k in ("population_a", "population_b")}))


if __name__ == "__main__":
    main()
