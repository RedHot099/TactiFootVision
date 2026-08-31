"""Acceptance gate: is the game-level split actually leak-free at pixel level?

    ./run.sh python scripts/audit_leakage.py --out results/audit

Checks, all reported into leakage.json:
  1. structural: duplicate shot_id, split/game disjointness, one fixture under two names
  2. degenerate frames (unreadable / near-constant luminance) and byte-identical dHash
  3. near-duplicate frames across splits, using DINOv2-S CLS embeddings

Why embeddings and not a perceptual hash: a 64-bit dHash cannot separate broadcast
football frames at all. Measured on this dataset, same-shot pairs (t0 vs -400 ms,
i.e. certainly the same moment) sit at median Hamming 13 while random pairs of
unrelated shots sit at median 31 with a 1st percentile of 21 - the distributions
overlap, so every threshold either misses real twins or flags thousands of unrelated
wide shots. Both calibrations are kept in the report; the verdict uses cosine
similarity of frozen DINOv2 CLS features, which is what a model actually "sees".

The per-frame check is complemented by a per-game-pair check: a duplicated broadcast
stored under two names would show up as an abnormally high mean top-1 similarity
between one test game and one train game, even if no single frame pair is extreme.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xgpoc.data.datasets import ImageDataset
from xgpoc.data.index import load_index

INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str, size: int = 8) -> tuple[np.ndarray, float]:
    """Returns (8 bytes of dHash, luminance std). Zero hash if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(8, np.uint8), 0.0
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).ravel()), float(img.std())


def hash_frames(paths: list[str], workers: int = 16) -> tuple[np.ndarray, np.ndarray]:
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(dhash, paths))
    return np.stack([r[0] for r in res]), np.array([r[1] for r in res])


@torch.no_grad()
def cls_embeddings(df: pd.DataFrame, cache: Path, path_col: str = "frame_path",
                   device: str = "cuda") -> np.ndarray:
    """L2-normalised DINOv2-S CLS vector per row (float16 cache, ~8 MB for 11k frames)."""
    if cache.exists():
        z = np.load(cache)
        if list(z["shot_id"]) == df.shot_id.tolist():
            return z["emb"].astype(np.float32)
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              img_size=(252, 448), dynamic_img_size=True).to(device).eval()
    loader = DataLoader(ImageDataset(df, size_wh=(448, 252), path_col=path_col), 32, num_workers=8)
    out = []
    for batch in tqdm(loader, desc=f"cls:{path_col}"):
        with torch.autocast("cuda", enabled=device == "cuda"):
            out.append(model.forward_features(batch["image"].to(device))[:, 0].float().cpu().numpy())
    emb = np.concatenate(out)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, shot_id=df.shot_id.to_numpy(dtype=str), emb=emb.astype(np.float16))
    return emb


def collage(pairs: list[dict], out: Path, n: int = 20) -> None:
    rows = []
    for p in pairs[:n]:
        ims = [cv2.imread(p[k]) for k in ("test_path", "ref_path")]
        if any(i is None for i in ims):
            continue
        row = np.hstack([cv2.resize(i, (320, 180)) for i in ims])
        cv2.putText(row, f"cos={p['similarity']:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    if rows:
        cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("results/audit"))
    ap.add_argument("--cache", type=Path, default=Path("cache/audit"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = load_index(Path(a.index)).reset_index(drop=True)
    rep: dict = {"n_shots": len(df)}

    # --- 1. structural --------------------------------------------------------
    rep["duplicate_shot_ids"] = int(df.shot_id.duplicated().sum())
    games = {s: set(df[df.split == s].game) for s in ("train", "valid", "test")}
    rep["games"] = {s: len(g) for s, g in games.items()}
    rep["game_overlap"] = {f"{x}|{y}": sorted(games[x] & games[y])
                           for x, y in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    fixture = df.game.str.replace(r"^[^/]+/[^/]+/", "", regex=True)
    n_splits = df.assign(fx=fixture).groupby("fx").split.nunique()
    rep["fixtures_in_multiple_splits"] = sorted(n_splits[n_splits > 1].index.tolist())

    # --- 2. degenerate frames + dHash calibration -----------------------------
    print(f"hashing {len(df)} frames...")
    h, std = hash_frames(df.frame_path.tolist())
    rep["unreadable_frames"] = int((std == 0).sum())
    rep["degenerate_frames"] = int((std < 5).sum())
    rep["frames_with_identical_dhash"] = int(pd.Series([bytes(x) for x in h]).duplicated().sum())
    rng = np.random.default_rng(0)
    sub = df.sample(600, random_state=0)
    h_t0, _ = hash_frames(sub.frame_path.tolist())
    h_m4, _ = hash_frames(sub["frame_path_-400"].tolist())
    same = np.array([POPCOUNT[np.bitwise_xor(x, y)].sum() for x, y in zip(h_t0, h_m4)])
    i, j = rng.integers(0, len(h), (2, 20000))
    rand = POPCOUNT[np.bitwise_xor(h[i], h[j])].sum(1)
    rep["dhash_calibration"] = {
        "same_shot_p50": float(np.median(same)), "same_shot_p90": float(np.percentile(same, 90)),
        "random_p1": float(np.percentile(rand, 1)), "random_p50": float(np.median(rand)),
        "separable": bool(np.percentile(same, 90) < np.percentile(rand, 1)),
    }

    # --- 3. embedding near-duplicates ----------------------------------------
    emb = cls_embeddings(df, a.cache / "cls_dinov2_s.npz")
    emb_m4 = cls_embeddings(sub.reset_index(drop=True), a.cache / "cls_dinov2_s_m400.npz",
                            path_col="frame_path_-400")
    pos = emb[sub.index.to_numpy()] @ emb_m4.T
    same_cos = np.diag(pos)
    rand_cos = (emb[i] * emb[j]).sum(1)
    # two thresholds, on purpose: "similar scene" is common and benign (same stadium,
    # same camera), "indistinguishable from the same moment" is the leak we care about
    thr_scene = float(np.percentile(same_cos, 5))
    thr_twin = float(np.median(same_cos))
    rep["embedding_calibration"] = {
        "same_shot_cos_p5": thr_scene, "same_shot_cos_p50": thr_twin,
        "random_cos_p99": float(np.percentile(rand_cos, 99)), "random_cos_p50": float(np.median(rand_cos)),
        "separable": bool(thr_scene > np.percentile(rand_cos, 99)),
        "thr_scene_rule": "cos >= 5th pct of same-shot (t0 vs -400 ms) pairs: similar scene",
        "thr_twin_rule": "cos >= median of same-shot pairs: as similar as the same moment 400 ms apart",
    }

    te = (df.split == "test").to_numpy()
    te_df, ref_df = df[te].reset_index(drop=True), df[~te].reset_index(drop=True)
    sim = emb[te] @ emb[~te].T
    best = sim.argmax(1)
    best_sim = sim[np.arange(len(best)), best]
    pairs = [{"test_shot": te_df.shot_id[r], "ref_shot": ref_df.shot_id[c], "similarity": float(sim[r, c]),
              "same_game": bool(te_df.game[r] == ref_df.game[c]), "ref_split": ref_df.split[c],
              "test_path": te_df.frame_path[r], "ref_path": ref_df.frame_path[c]}
             for r, c in np.argwhere(sim >= thr_scene)]
    pairs.sort(key=lambda p: -p["similarity"])
    scene = {p["test_shot"] for p in pairs}
    twins = {p["test_shot"] for p in pairs if p["similarity"] >= thr_twin}
    rep["cross_split"] = {
        "test_frames_with_similar_scene": len(scene),
        "test_frames_with_similar_scene_pct": round(100 * len(scene) / len(te_df), 3),
        "test_frames_with_twin": len(twins),
        "test_frames_with_twin_pct": round(100 * len(twins) / len(te_df), 3),
        "pairs_similar_scene": len(pairs),
        "pairs_same_game": sum(p["same_game"] for p in pairs),
        "max_similarity": float(best_sim.max()),
        "nearest_similarity_p50": float(np.median(best_sim)),
        "nearest_similarity_p99": float(np.percentile(best_sim, 99)),
    }
    pd.DataFrame(pairs[:2000]).to_csv(a.out / "cross_split_pairs.csv", index=False)
    collage(pairs, a.out / "cross_split_pairs.jpg")

    # --- 4. duplicated-broadcast check at game-pair level ---------------------
    gp = pd.DataFrame({"test_game": te_df.game, "ref_game": ref_df.game.to_numpy()[best], "sim": best_sim})
    agg = (gp.groupby(["test_game", "ref_game"]).sim.agg(["mean", "count"])
             .reset_index().sort_values(["count", "mean"], ascending=False))
    agg.to_csv(a.out / "game_pair_similarity.csv", index=False)
    top = agg[agg["count"] >= 5].head(5)
    rep["top_game_pairs"] = top.to_dict("records")
    # a duplicated broadcast would concentrate one test game's nearest neighbours on
    # one train game AND at high similarity; both conditions together are the alarm
    share = gp.groupby("test_game").ref_game.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    conc = share.sort_values(ascending=False)
    rep["max_nn_concentration"] = {"game": conc.index[0], "share": float(conc.iloc[0]),
                                   "mean_sim": float(gp[gp.test_game == conc.index[0]].sim.mean()),
                                   "duplicate_broadcast_suspected": bool(conc.iloc[0] > 0.8 and
                                       gp[gp.test_game == conc.index[0]].sim.mean() >= thr_twin)}

    # --- 5. venue overlap: not leakage, but a generalisation caveat -----------
    home = df.game.str.extract(r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ ")[0]
    dfh = df.assign(home=home)
    tr_home = set(dfh[dfh.split == "train"].home.dropna())
    te_games = dfh[dfh.split == "test"].groupby("game").home.first()
    rep["venue_overlap"] = {
        "test_games_with_home_team_seen_in_train": int(te_games.isin(tr_home).sum()),
        "test_games": int(len(te_games)),
        "unparsed_game_names": int(home.isna().sum()),
        "note": "same home team implies same stadium and often the same broadcast style; "
                "measured effect on metrics is reported in eval_subgroups.py (seen vs unseen venue)",
    }
    dfh[["shot_id", "game", "split", "home"]].to_csv(a.out / "venues.csv", index=False)

    rep["pass"] = bool(rep["duplicate_shot_ids"] == 0
                       and not any(rep["game_overlap"].values())
                       and not rep["fixtures_in_multiple_splits"]
                       and rep["unreadable_frames"] == 0
                       and rep["cross_split"]["pairs_same_game"] == 0
                       and rep["cross_split"]["test_frames_with_twin_pct"] < 0.5
                       and not rep["max_nn_concentration"]["duplicate_broadcast_suspected"])
    (a.out / "leakage.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float)[:5000])


if __name__ == "__main__":
    main()
