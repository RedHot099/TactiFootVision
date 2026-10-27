"""Probes of xg-dino on stored per-layer embeddings (ticket 04; `xgpoc.dino.targets`, `xgpoc.dino.probes`).

    ./run.sh --no-sync python scripts/xgdino_probes.py --layers dino_final g_final g_pool      # ticket 04 (resumable)
    ./run.sh --no-sync python scripts/xgdino_probes.py --layers dino_blocks:23 g_blocks:23 --out results/...

Layers: `dino_final`, `g_final`, `g_pool`, `dino_blocks:<b>` (b = 0-23), `g_blocks:<b>` (b = 20-23); token
`both` (CLS and mean patch concatenated, default), `cls` or `mean` (not for g_pool). `dino_*` is the frozen DINOv2
(the same array for every outer fold), `g_*` the G-combo fold model k for outer fold k. Offset --offset (0).

Population: the 7,062 training shots of the 328 matches (`stack.inputs.training_shots`, host folds) with a stored frame
at the offset; automatic targets at t0 from `cache/xgd_full4_feet_t0/E3` (feet_t0 rule). Everything up to the probe
predictions runs inside `no_statsbomb_reads()`; the paired predictions (each of the five fold probes on its own fold
model's embedding, averaged) are written before StatsBomb is read. Then, outside the block, `targets.statsbomb_targets`
builds the paired StatsBomb targets and the frozen predictions are scored against them (descriptive; nothing fitted).

Each layer's outer-fold run is stored in `<out>/runs/<layer>__fold<k>.pkl` and skipped when present (resume;
--folds / --fit-only split the work); the read-out (metrics,
decisions, encoder differences, GK strata, paired evaluation, timing) is rebuilt from the stored runs. complete.json last.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg
from xgpoc.paths import INDEX
from xgpoc.stack import inputs as si

ROOT = Path(__file__).resolve().parents[1]
LAYERS = Path("/mnt/storage/tactifoot_research/xg_dino/layers")
FEET = ROOT / "cache/xgd_full4_feet_t0/E3"
OUT = ROOT / "results/xgdino_probes_last"
N_FOLDS = 5
G_FIRST_BLOCK = 20
EXPECTED = {"shots": 7062, "matches": 328, "paired": 2721}
CONSISTENCY = {"distance": ("dist", False), "angle": ("angle", False), "def_cone": ("ff_def_cone", False),
               "near_opp_m": ("ff_near_opp", False), "gk_depth_m": ("gk_depth", True)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ inputs
def population(index: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and paired rows of the layers index at `offset` with a stored frame, by shot id; the training shots'
    match and fold checked against `stack.inputs.training_shots` (host folds)."""
    at = index[index.offset_ms == offset].set_index("shot_id")
    shots = si.training_shots(INDEX, si.OOF, a1.verify_plan(INDEX))
    train = at[at.set == "training"]
    if not train.index.sort_values().equals(shots.index.sort_values()):
        raise ValueError("the layers index's training shots are not the population's")
    if (train.fold != shots.fold.loc[train.index]).any() or (train.game != shots.game.loc[train.index]).any():
        raise ValueError("a training shot's fold or match differs from the population's")
    paired = at[at.set == "paired"]
    return train[train.row >= 0].sort_index(), paired[paired.row >= 0].sort_index()


def run_path(out: Path, layer: str, fold: int, targets: list[str] | None = None) -> Path:
    tag = "" if targets is None else "__" + "+".join(targets)
    return out / "runs" / f"{layer.replace(':', '_')}__fold{fold}{tag}.pkl"


def stored_targets(out: Path, layer: str, fold: int) -> set[str]:
    names = set()
    for path in (out / "runs").glob(f"{layer.replace(':', '_')}__fold{fold}*.pkl"):
        with open(path, "rb") as f:
            names |= set(pickle.load(f).targets)
    return names


def load_fold(out: Path, layer: str, fold: int) -> pr.Run:
    """The stored run of one layer and fold, joined over its target subsets; every target must be present once."""
    parts = []
    for path in sorted((out / "runs").glob(f"{layer.replace(':', '_')}__fold{fold}*.pkl")):
        with open(path, "rb") as f:
            parts.append(pickle.load(f))
    names = [t for p in parts for t in p.targets]
    if sorted(names) != sorted(tg.TARGETS):
        raise ValueError(f"{layer} fold {fold}: stored targets {sorted(names)} are not every target once")
    return pr.Run(layer, [fold], {t: p.targets[t] for t in tg.TARGETS for p in parts if t in p.targets},
                  [x for p in parts for x in p.timing])


def parse_layer(name: str) -> tuple[str, int | None]:
    kind, _, block = name.partition(":")
    if kind not in ("dino_final", "g_final", "g_pool", "dino_blocks", "g_blocks"):
        raise ValueError(f"unknown layer {name}")
    if (block == "") != (kind in ("dino_final", "g_final", "g_pool")):
        raise ValueError(f"{name}: blocks need :<b>, final kinds take none")
    return kind, (int(block) if block else None)


def layer_arrays(layer: str, token: str, rows_train: np.ndarray, rows_paired: np.ndarray, root: Path = LAYERS):
    """(train, paired) float16 arrays [n, F or 1, d] of one layer (F = 5 fold models for G, 1 for DINOv2)."""
    kind, block = parse_layer(layer)
    arr = np.load(root / f"{kind}.npy", mmap_mode="r")
    tok = {"both": slice(None), "cls": slice(0, 1), "mean": slice(1, 2)}[token]

    def take(rows):
        a = np.asarray(arr[np.sort(rows)])[np.argsort(np.argsort(rows))]
        if kind == "dino_blocks":
            a = a[:, block, tok][:, None]
        elif kind == "dino_final":
            a = a[:, tok][:, None]
        elif kind == "g_blocks":
            a = a[:, :, block - G_FIRST_BLOCK, tok]
        elif kind == "g_final":
            a = a[:, :, tok]
        elif token != "both":
            raise ValueError("g_pool has no tokens")
        return a.reshape(a.shape[0], a.shape[1], -1)
    return take(rows_train), take(rows_paired)


def encoder(layer: str, token: str, train_idx: pd.DataFrame, paired_idx: pd.DataFrame) -> pr.Encoder:
    xt, xp = layer_arrays(layer, token, train_idx.row.to_numpy(), paired_idx.row.to_numpy())
    pick = (lambda a, k: a[:, k]) if xt.shape[1] == N_FOLDS else (lambda a, k: a[:, 0])
    return pr.Encoder(layer, lambda k: pick(xt, k).astype(np.float32), lambda k: pick(xp, k).astype(np.float32))


def consistency(auto: pd.DataFrame, features: pd.DataFrame) -> dict:
    """max |Δ| of the automatic targets against the cache's own feature table (same frames, same rule)."""
    f = features[features.offset_ms == 0].set_index("shot_id").loc[auto.index]
    out = {}
    for target, (column, signed) in CONSISTENCY.items():
        ref = f[column].abs() if signed else f[column]
        both = auto[target].notna() & ref.notna()
        out[target] = {"nan_mismatch": int((auto[target].isna() != ref.isna()).sum()),
                       "max_abs_diff": float((auto[target][both] - ref[both]).abs().max())}
    if any(v["nan_mismatch"] or v["max_abs_diff"] > 1e-9 for v in out.values()):
        raise ValueError(f"automatic targets disagree with the cache's features: {out}")
    return out


# ------------------------------------------------------------------ read-out
def read_out(runs: dict[str, pr.Run], data: pr.ProbeData, rules: pr.Rules) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    d = pr.draws(data.games)
    metrics, decisions, diffs = [], [], []
    per = {}
    for layer, run in runs.items():
        for target, t in run.targets.items():
            md = pr.target_metrics(t, d)
            per[layer, target] = md
            dec = pr.decide(target, md, rules)
            decisions.append({"layer": layer, "target": target} | dec | {"penalty": t.penalty["embedding"]})
            metrics += pr.rows_table(layer, target, md, dec, layer)
            if target in tg.GOALKEEPER:
                for stratum in ("gk_detected", "gk_proxy"):
                    mask = data.targets[stratum].to_numpy()[t.rows] == 1
                    if mask.sum() > 20:
                        metrics += pr.rows_table(layer, target, pr.target_metrics(t, d, mask=mask), {}, layer, stratum)
    dino = [layer for layer in runs if layer.startswith("dino")]
    for g in (layer for layer in runs if layer.startswith("g_")):
        for ref in dino:
            for target in runs[g].targets:
                a, b = per[g, target], per[ref, target]
                name = pr.metric_name(target)
                for key in ("embedding", "distance_embedding"):
                    diffs.append({"a": g, "b": ref, "target": target, "probe": key, "metric": name}
                                 | pr.difference(a, b, key, name))
    return pd.DataFrame(metrics), decisions, pd.DataFrame(diffs)


def selection(runs: dict[str, pr.Run], data: pr.ProbeData, rules: pr.Rules) -> list[dict]:
    """Best layer per encoder and target (all development OOF) and the nested "encodes" read-out, when an encoder has
    more than one block layer (`<kind>:<b>`, in block order) in this run; the final kinds are reported separately."""
    d = pr.draws(data.games)
    out = []
    for enc in ("dino", "g_"):
        group = {k: v for k, v in runs.items() if k.startswith(enc) and ":" in k}
        if len(group) < 2:
            continue
        for target in next(iter(group.values())).targets:
            composite, chosen = pr.nested(group, data, target, rules)
            out.append({"encoder": enc.rstrip("_"), "target": target, "best_layer": pr.chosen_layer(group, data, target, rules),
                        "nested_layers": chosen} | pr.decide(target, pr.target_metrics(composite, d), rules))
    return out


def paired_read_out(runs: dict[str, pr.Run], sb: pd.DataFrame, auto: pd.DataFrame, games: np.ndarray) -> pd.DataFrame:
    d = pr.draws(games)
    rows = []
    for layer, run in runs.items():
        for target, t in run.targets.items():
            for key, md in pr.paired_metrics(t, sb[target], auto[target], d).items():
                n = md.pop("n")
                for metric, (point, boot) in md.items():
                    rows.append({"layer": layer, "target": target, "probe": key, "metric": metric, "n": n}
                                | pr.summary(point, boot))
    return pd.DataFrame(rows)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["dino_final", "g_final", "g_pool"])
    ap.add_argument("--token", default="both", choices=["both", "cls", "mean"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--protocol", type=Path, help="JSON with a `probe_rules` object (C2 defaults otherwise)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="outer folds to fit in this call")
    ap.add_argument("--targets", nargs="+", choices=tg.TARGETS, help="fit only these targets (finer resume units)")
    ap.add_argument("--threads", type=int, help="torch threads (several fold processes at once)")
    ap.add_argument("--fit-only", action="store_true", help="fit and store the given layers and folds, no read-out")
    args = ap.parse_args()
    started = time.time()
    if args.threads:
        torch.set_num_threads(args.threads)
    rules = pr.Rules.from_protocol(json.loads(args.protocol.read_text()).get("probe_rules") if args.protocol else None)
    out = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)

    with no_statsbomb_reads():
        index = pd.read_parquet(LAYERS / "index.parquet")
        train_idx, paired_idx = population(index, args.offset)
        ids = [*train_idx.index, *paired_idx.index]
        targets_file = out / "targets_automatic.parquet"
        if targets_file.exists():
            auto = pd.read_parquet(targets_file)
            auto.attrs["source"] = "automatic"
        else:
            log("automatic targets (feet_t0 replay)")
            frames = pd.read_parquet(FEET / "frames.parquet", filters=[("shot_id", "in", ids)])
            auto = tg.automatic_targets(frames, ids)
            features = pd.read_parquet(FEET / "features.parquet", filters=[("shot_id", "in", ids)])
            (out / "targets_check.json").write_text(json.dumps(consistency(auto, features), indent=1))
            auto.to_parquet(targets_file)
        auto_train = tg.tagged(auto.loc[train_idx.index].copy(), "automatic")
        auto_paired = tg.tagged(auto.loc[paired_idx.index].copy(), "automatic")
        data = pr.ProbeData(auto_train, train_idx.game.to_numpy(), train_idx.fold.to_numpy(), auto_paired)
        counts = {"shots": len(train_idx), "matches": int(train_idx.game.nunique()), "paired": len(paired_idx),
                  "paired_matches": int(paired_idx.game.nunique())}
        log(f"population {counts}")
        enc = {}
        for layer in args.layers:
            for k in args.folds:
                path = run_path(out, layer, k, args.targets)
                if path.exists() or set(args.targets or tg.TARGETS) <= stored_targets(out, layer, k):
                    continue
                if layer not in enc:
                    enc[layer] = encoder(layer, args.token, train_idx, paired_idx)
                log(f"layer {layer} fold {k}")
                part = pr.run(data, enc[layer], args.targets or tg.TARGETS, folds=[k], log=log)
                tmp = path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(part, f)
                tmp.rename(path)
            enc.pop(layer, None)
        if args.fit_only:
            return 0
        runs = {}
        for layer in args.layers:
            runs[layer] = pr.merge([load_fold(out, layer, k) for k in range(N_FOLDS)])
        metrics, decisions, diffs = read_out(runs, data, rules)
        chosen = selection(runs, data, rules)
        timing = pd.DataFrame([t for r in runs.values() for t in r.timing])
        coverage = tg.coverage(auto_train)
        for layer, run in runs.items():
            oof = pd.DataFrame({"shot_id": auto_train.index})
            pairs = pd.DataFrame({"shot_id": auto_paired.index})
            for target, t in run.targets.items():
                for key, pred in t.oof.items():
                    col = np.full(len(oof), np.nan)
                    col[t.rows] = pred
                    oof[f"{target}__{key}"] = col
                for key, pred in t.paired.items():
                    pairs[f"{target}__{key}"] = pred
            oof.to_parquet(out / f"oof_{layer.replace(':', '_')}.parquet")
            pairs.to_parquet(out / f"paired_pred_{layer.replace(':', '_')}.parquet")
    # outside the guard: the one StatsBomb read, evaluation of frozen predictions only
    sb = tg.statsbomb_targets(paired_idx.index)
    sb = sb.loc[paired_idx.index]
    paired = paired_read_out(runs, sb, auto_paired, paired_idx.game.to_numpy())

    metrics.to_csv(out / "metrics.csv", index=False)
    diffs.to_csv(out / "encoder_differences.csv", index=False)
    timing.to_csv(out / "timing.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    tg.coverage(sb).to_csv(out / "coverage_statsbomb.csv", index=False)
    paired.to_csv(out / "paired_vs_statsbomb.csv", index=False)
    (out / "decisions.json").write_text(json.dumps(jsonable({"rules": rules.__dict__, "per_layer": decisions,
                                                             "selection": chosen}), indent=1))
    fits = timing.groupby("model")[["wall_s", "cpu_s"]].mean()
    complete = {"ticket": "xg-dino 04", "layers": args.layers, "token": args.token, "offset_ms": args.offset,
                "counts": counts, "rules": rules.__dict__, "alphas": pr.ALPHAS, "lambdas": pr.LAMBDAS,
                "inputs": {"layers_index": sha256_file(LAYERS / "index.parquet"),
                           "layers_complete": sha256_file(LAYERS / "complete.json"),
                           "feet_frames": records.verified_file(FEET / "frames.parquet"),
                           "feet_features": records.verified_file(FEET / "features.parquet")},
                "timing_per_fold_fit": {m: {"wall_s": float(r.wall_s), "cpu_s": float(r.cpu_s)} for m, r in fits.iterrows()},
                "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                "files": {p.name: sha256_file(p) for p in sorted(out.glob("*")) if p.is_file() and p.name != "complete.json"},
                "wall_s": time.time() - started}
    (out / "complete.json").write_text(json.dumps(jsonable(complete), indent=1))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
