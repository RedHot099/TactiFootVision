"""Per-layer readability curves of xg-dino (ticket 06; `xgpoc.dino.curves`, councils C1 and C2).

    ./run.sh --no-sync python scripts/xgdino_layer_curves.py --verify      # fast ridge vs ticket 04 on the final kinds
    ./run.sh --no-sync python scripts/xgdino_layer_curves.py --fit         # resumable, stops after --max-seconds
    ./run.sh --no-sync python scripts/xgdino_layer_curves.py --select      # curves, best_layers.json (no paired read)
    ./run.sh --no-sync python scripts/xgdino_layer_curves.py --paired      # resumable: paired reads, logistic refits
    ./run.sh --no-sync python scripts/xgdino_layer_curves.py --finish      # StatsBomb evaluation, figures, complete.json

Layers: the 24 frozen DINOv2 blocks (`dino_blocks:<b>`, CLS and mean patch before the final norm, concatenated) and the
G-combo blocks 20-23 (`g_blocks:<b>`, fold model k for outer fold k, C1). Blocks 0-19 of every G-combo fold model are
the frozen DINOv2 blocks (verified in ticket 05), so the G-combo curve is `dino_blocks` below block 20. Offsets 0 and
-400 ms (the population at an offset: the training shots with a stored frame there); targets: the automatic t0
targets of ticket 04 (`results/xgdino_probes_last/targets_automatic.parquet`, sha256 recorded).

Probes: the eigendecomposition ridge of `curves.run` for every target (binary targets scored by the AUC of the ridge
output), on ticket 04's splits, references, shuffled control and match bootstrap. --verify fits `dino_final` and
`g_final` at offset 0 and compares them with ticket 04's stored OOF predictions.

Selection (C2): best layer per encoder x target x offset by the mean over outer folds of the embedding-only OOF metric
(R² continuous / counts, AUC binary), ties within 0.01 R² / 0.005 AUC -> later block; "encodes beyond distance" from
the nested choice (layer picked per outer fold inside its training folds); "readable" at the best layer.
`best_layers.json` is written by --select, before any paired-shot prediction exists; --paired refuses to run without
it and reads only offset-0 best layers: the ridge probes refitted with the paired reads (each of the five fold probes on
its own fold model's embedding, averaged) and, for binary targets, ticket 04's logistic probe (on the GPU). --finish
reads StatsBomb once, outside the guard, and scores the frozen predictions against it (R², MAE, Spearman, mean signed
error, AUC; descriptive, nothing fitted to StatsBomb).
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgdino_probes as xp
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import curves as cv
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_probes_layers"
TICKET04 = ROOT / "results/xgdino_probes_last"
OFFSETS = (0, -400)
N_BLOCKS, G_BLOCKS = 24, (20, 21, 22, 23)
LAYERS = [f"dino_blocks:{b}" for b in range(N_BLOCKS)] + [f"g_blocks:{b}" for b in G_BLOCKS]
KEYS = ("embedding", "distance_embedding", "embedding_shuffled")
REF_KEYS = ("distance", "constant")
GROUPS = {"location": ["distance", "angle"], "opponents": ["def_cone", "opp_3m", "near_opp_m"],
          "opponents_binary": ["def_cone_ge1", "opp_3m_ge1"],
          "goalkeeper": ["gk_depth_m", "gk_lateral_m", "gk_lateral_signed_m"]}
log = xp.log


def block_layer(encoder: str, block: int) -> str:
    """The stored layer an encoder reads at a block (G-combo below 20 = frozen DINOv2)."""
    return f"g_blocks:{block}" if encoder == "g_combo" and block >= G_BLOCKS[0] else f"dino_blocks:{block}"


def encoder_layers(encoder: str) -> dict[str, str]:
    """{block name: stored layer} in block order."""
    return {str(b): block_layer(encoder, b) for b in range(N_BLOCKS)}


def tag(layer: str) -> str:
    return layer.replace(":", "_")


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.rename(path)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Inputs:
    """Population, automatic targets and the block arrays of one offset (training rows; paired on demand)."""

    def __init__(self, offset: int):
        index = pd.read_parquet(xp.LAYERS / "index.parquet")
        self.train_idx, self.paired_idx = xp.population(index, offset)
        auto = pd.read_parquet(TICKET04 / "targets_automatic.parquet")
        missing = set(self.train_idx.index) - set(auto.index)
        if missing:
            raise ValueError(f"{len(missing)} training shots lack automatic targets")
        self.auto_train = tg.tagged(auto.loc[self.train_idx.index].copy(), "automatic")
        self.auto_paired = tg.tagged(auto.loc[self.paired_idx.index].copy(), "automatic")
        self.data = pr.ProbeData(self.auto_train, self.train_idx.game.to_numpy(), self.train_idx.fold.to_numpy(),
                                 self.auto_paired)
        self._arrays: dict[str, np.ndarray] = {}

    def array(self, kind: str) -> np.ndarray:
        """float16 [n, ..., 2, d] of the training rows (whole kind, cached)."""
        if kind not in self._arrays:
            rows = self.train_idx.row.to_numpy()
            arr = np.load(xp.LAYERS / f"{kind}.npy", mmap_mode="r")
            order = np.argsort(rows)
            out = np.empty((len(rows), *arr.shape[1:]), arr.dtype)
            out[order] = arr[rows[order]]
            self._arrays[kind] = out
        return self._arrays[kind]

    def encoder(self, layer: str) -> pr.Encoder:
        kind, block = xp.parse_layer(layer)
        a = self.array(kind)
        n = len(a)
        if kind == "dino_blocks":
            return pr.Encoder(layer, lambda k: a[:, block].reshape(n, -1).astype(np.float32))
        return pr.Encoder(layer, lambda k: a[:, k, block - G_BLOCKS[0]].reshape(n, -1).astype(np.float32))


def run_file(out: Path, offset: int, layer: str) -> Path:
    return out / "runs" / f"o{offset}" / f"{tag(layer)}.pkl"


# ------------------------------------------------------------------ stages
def verify(out: Path, device: str) -> dict:
    """Fast ridge on dino_final / g_final at offset 0 against ticket 04's stored OOF predictions."""
    inp = Inputs(0)
    report = {}
    for layer in ("dino_final", "g_final"):
        path = out / "runs" / "verify" / f"{layer}.pkl"
        if not path.exists():
            t0 = time.time()
            enc = xp.encoder(layer, "both", inp.train_idx, inp.paired_idx)
            save(cv.run(inp.data, pr.Encoder(layer, enc.train), device=device, log=log), path)
            log(f"verify {layer}: {time.time() - t0:.0f} s")
        fast = load(path)
        old = pd.read_parquet(TICKET04 / f"oof_{layer}.parquet").set_index("shot_id").loc[inp.auto_train.index]
        rows = []
        for target, t in fast.targets.items():
            binary = tg.kind(target) == "binary"
            keys = ["distance", "constant"] if binary else ["embedding", "distance_embedding", "distance", "constant",
                                                            "embedding_shuffled"]
            for key in keys:
                ref = old[f"{target}__{key}"].to_numpy()[t.rows]
                y = t.oof["shuffled_target"] if key == "embedding_shuffled" else t.y
                rows.append({"target": target, "probe": key, "max_abs_pred_diff": float(np.nanmax(np.abs(t.oof[key] - ref))),
                             "metric_fast": pr.point_metric(y, t.oof[key], binary),
                             "metric_ticket04": pr.point_metric(y, ref, binary)})
                rows[-1]["metric_diff"] = rows[-1]["metric_fast"] - rows[-1]["metric_ticket04"]
            ticket04 = json.loads((TICKET04 / "decisions.json").read_text())["per_layer"]
            pen = next(d["penalty"] for d in ticket04 if d["layer"] == layer and d["target"] == target)
            if not binary:
                rows.append({"target": target, "probe": "penalty_embedding_equal", "max_abs_pred_diff": float(t.penalty["embedding"] != pen)})
        report[layer] = rows
    flat = pd.DataFrame([r | {"layer": layer} for layer, rs in report.items() for r in rs])
    flat.to_csv(out / "verify_ticket04.csv", index=False)
    cont = flat[flat.probe.isin(["embedding", "distance_embedding", "embedding_shuffled"])]
    cont = cont[~((cont.target == "distance") & (cont.probe == "distance_embedding"))]
    summary = {"max_abs_metric_diff_continuous": float(cont.metric_diff.abs().max()),
               "max_abs_pred_diff_continuous": float(cont.max_abs_pred_diff.max()),
               "max_abs_metric_diff_references": float(flat[flat.probe.isin(REF_KEYS)].metric_diff.abs().max()),
               "penalties_differ": int(flat[flat.probe == "penalty_embedding_equal"].max_abs_pred_diff.sum())}
    (out / "verify_ticket04.json").write_text(json.dumps(summary, indent=1))
    log(f"verify: {summary}")
    return summary


def fit(out: Path, device: str, max_seconds: float) -> bool:
    """Every (offset, layer) run; stops cleanly after `max_seconds`. True when all are stored."""
    started = time.time()
    for offset in OFFSETS:
        todo = [layer for layer in LAYERS if not run_file(out, offset, layer).exists()]
        if not todo:
            continue
        inp = Inputs(offset)
        for layer in todo:
            if time.time() - started > max_seconds:
                log("time budget reached; rerun to resume")
                return False
            t0 = time.time()
            r = cv.run(inp.data, inp.encoder(layer), device=device)
            r.timing.append({"encoder": layer, "fold": -1, "targets": "all", "model": "layer_total", "rows": len(inp.data.targets),
                             "dim": 2048, "wall_s": time.time() - t0, "cpu_s": float("nan")})
            save(r, run_file(out, offset, layer))
            log(f"offset {offset} {layer}: {time.time() - t0:.1f} s")
    return True


def curves_table(runs: dict[str, pr.Run], data: pr.ProbeData, offset: int, rules: pr.Rules) -> tuple[list[dict], dict]:
    """Per-layer metric rows (with 95% match-bootstrap intervals) and the per-layer metric draws."""
    d = pr.draws(data.games)
    rows, per = [], {}
    first = next(iter(runs.values()))
    refs = {t: pr.target_metrics(first.targets[t], d, REF_KEYS) for t in tg.TARGETS}
    for t in tg.TARGETS:
        for key, md in refs[t].items():
            for metric, (point, boot) in md.items():
                rows.append({"offset_ms": offset, "layer": "reference", "target": t, "probe": key, "metric": metric}
                            | pr.summary(point, boot))
    for layer, r in runs.items():
        for t in tg.TARGETS:
            if not np.allclose(r.targets[t].oof["distance"], first.targets[t].oof["distance"], equal_nan=True):
                raise ValueError(f"{layer} {t}: the distance reference differs between layers")
            md = pr.target_metrics(r.targets[t], d, KEYS) | refs[t]
            per[layer, t] = md
            dec = pr.decide(t, md, rules)
            for key in (k for k in KEYS if k in md):    # binary targets have no shuffled control
                for metric, (point, boot) in md[key].items():
                    rows.append({"offset_ms": offset, "layer": layer, "target": t, "probe": key, "metric": metric}
                                | pr.summary(point, boot))
            if "gain" in dec:
                rows.append({"offset_ms": offset, "layer": layer, "target": t, "probe": "gain_over_distance",
                             "metric": dec["metric"]} | dec["gain"])
    return rows, per


def select(out: Path, rules: pr.Rules) -> dict:
    """Curves, encoder differences at blocks 20-23, best layers and nested "encodes" (no paired read)."""
    curve_rows, diff_rows, best = [], [], []
    for offset in OFFSETS:
        inp = Inputs(offset)
        data = inp.data
        runs = {layer: load(run_file(out, offset, layer)) for layer in LAYERS}
        rows, per = curves_table(runs, data, offset, rules)
        curve_rows += rows
        d = pr.draws(data.games)
        for b in G_BLOCKS:
            for t in tg.TARGETS:
                name = pr.metric_name(t)
                for key in ("embedding", "distance_embedding"):
                    diff_rows.append({"offset_ms": offset, "block": b, "target": t, "probe": key, "metric": name}
                                     | pr.difference(per[f"g_blocks:{b}", t], per[f"dino_blocks:{b}", t], key, name))
        for encoder in ("dino", "g_combo"):
            group = {block: runs[layer] for block, layer in encoder_layers(encoder).items()}
            for t in tg.TARGETS:
                binary = tg.kind(t) == "binary"
                kind = "binary" if binary else "continuous"
                scores = {block: float(np.mean(pr.fold_scores(r, data, t))) for block, r in group.items()}
                chosen = pr.chosen_layer(group, data, t, rules)
                composite, nested_layers = pr.nested(group, data, t, rules)
                nested_dec = pr.decide(t, pr.target_metrics(composite, d), rules)
                at = per[block_layer(encoder, int(chosen)), t]
                dec = pr.decide(t, at, rules)
                name = pr.metric_name(t)
                best.append({
                    "offset_ms": offset, "encoder": encoder, "target": t, "metric": name, "best_block": int(chosen),
                    "best_layer": block_layer(encoder, int(chosen)), "mean_fold_metric": scores[chosen],
                    "oof_metric": pr.summary(*at["embedding"][name]),
                    "distance_only": pr.summary(*at["distance"][name]),
                    "distance_embedding": pr.summary(*at["distance_embedding"][name]),
                    "readable": dec["readable"],
                    "first_readable_block": cv.first_readable(scores, rules.readable_min[kind]),
                    "max_block": max(scores, key=scores.get), "mean_fold_metric_by_block": scores,
                    "nested_layers": [int(x) for x in nested_layers],
                    "nested_gain": nested_dec.get("gain"),
                    "encodes_beyond_distance": nested_dec.get("encodes_beyond_distance"),
                    "encodes": nested_dec.get("encodes", dec.get("encodes")),
                    "gain_at_best": dec.get("gain"),
                })
                log(f"offset {offset} {encoder} {t}: block {chosen} {name} {scores[chosen]:.3f}")
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(out / "curves.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out / "encoder_differences.csv", index=False)
    payload = {"ticket": "xg-dino 06", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rules": rules.__dict__,
               "rule": "best block per encoder x target x offset: mean over outer folds of the embedding-only OOF "
                       "metric (ridge; AUC of the ridge output for binary targets); ties within the margin -> later "
                       "block; G-combo blocks 0-19 = frozen DINOv2",
               "paired_reads_exist": any((out / "runs_paired").glob("*.pkl")) if (out / "runs_paired").exists() else False,
               "curves_sha256": sha256_file(out / "curves.csv"), "best": best}
    if payload["paired_reads_exist"]:
        raise RuntimeError("paired reads exist before the best-layer table; refusing to (re)write it")
    (out / "best_layers.json").write_text(json.dumps(xp.jsonable(payload), indent=1))
    return payload


def best_table(out: Path) -> list[dict]:
    path = out / "best_layers.json"
    if not path.exists():
        raise RuntimeError("best_layers.json must exist before any paired-shot reading (run --select)")
    return [b for b in json.loads(path.read_text())["best"] if b["offset_ms"] == 0]


def paired(out: Path, device: str, max_seconds: float) -> bool:
    """Offset-0 best layers: ridge with paired reads; logistic refits of the binary targets (resumable)."""
    started = time.time()
    best = best_table(out)
    inp = Inputs(0)
    for layer in sorted({b["best_layer"] for b in best}):
        path = out / "runs_paired" / f"{tag(layer)}.pkl"
        if path.exists():
            continue
        if time.time() - started > max_seconds:
            return False
        t0 = time.time()
        save(cv.run(inp.data, xp.encoder(layer, "both", inp.train_idx, inp.paired_idx), paired=True, device=device), path)
        log(f"paired ridge {layer}: {time.time() - t0:.1f} s")
    for b in best:
        if tg.kind(b["target"]) != "binary":
            continue
        enc = None
        for k in range(xp.N_FOLDS):
            path = out / "runs_logistic" / f"{tag(b['best_layer'])}__{b['target']}__fold{k}.pkl"
            if path.exists():
                continue
            if time.time() - started > max_seconds:
                return False
            enc = enc or xp.encoder(b["best_layer"], "both", inp.train_idx, inp.paired_idx)
            t0 = time.time()
            save(pr.run(inp.data, enc, [b["target"]], folds=[k], device=device), path)
            log(f"logistic {b['best_layer']} {b['target']} fold {k}: {time.time() - t0:.1f} s")
    return True


def spearman_summary(y, pred, rows, d: pr.Draws) -> dict:
    return pr.summary(float(cv.weighted_spearman(y, pred)), cv.weighted_spearman(y, pred, d.weights[:, rows]))


def finish(out: Path, rules: pr.Rules, started: float) -> None:
    best = best_table(out)
    inp = Inputs(0)
    d_train = pr.draws(inp.data.games)
    logistic, ridge_runs = {}, {}
    with no_statsbomb_reads():
        for b in best:
            ridge_runs[b["best_layer"]] = ridge_runs.get(b["best_layer"]) or load(out / "runs_paired" / f"{tag(b['best_layer'])}.pkl")
            if tg.kind(b["target"]) == "binary":
                key = (b["best_layer"], b["target"])
                if key not in logistic:
                    logistic[key] = pr.merge([load(out / "runs_logistic" / f"{tag(key[0])}__{key[1]}__fold{k}.pkl")
                                              for k in range(xp.N_FOLDS)]).targets[key[1]]
        # the paired ridge refit equals the curve run on the training rows (same probes): checked
        for layer, r in ridge_runs.items():
            curve = load(run_file(out, 0, layer))
            for t in tg.TARGETS:
                if not np.allclose(r.targets[t].oof["embedding"], curve.targets[t].oof["embedding"], rtol=1e-9, atol=1e-9):
                    raise ValueError(f"{layer} {t}: the paired refit differs from the curve run")
        logistic_rows = []
        for b in best:
            if tg.kind(b["target"]) != "binary":
                continue
            lt = logistic[b["best_layer"], b["target"]]
            md = pr.target_metrics(lt, d_train, ("embedding", "distance_embedding", "distance"))
            dec = pr.decide(b["target"], md, rules)
            logistic_rows.append({"encoder": b["encoder"], "target": b["target"], "best_layer": b["best_layer"],
                                  "ridge_auc": b["oof_metric"], "logistic_auc": pr.summary(*md["embedding"]["AUC"]),
                                  "logistic_distance_embedding_auc": pr.summary(*md["distance_embedding"]["AUC"]),
                                  "logistic_gain": dec["gain"], "logistic_encodes_beyond_distance": dec["encodes_beyond_distance"],
                                  "logistic_readable": dec["readable"], "lambda": lt.penalty["embedding"]})
    # outside the guard: the one StatsBomb read; frozen predictions scored, nothing fitted
    sb = tg.statsbomb_targets(inp.paired_idx.index).loc[inp.paired_idx.index]
    d = pr.draws(inp.paired_idx.game.to_numpy())
    rows = []
    for b in best:
        t = b["target"]
        y = sb[t].to_numpy(float)
        tr = ridge_runs[b["best_layer"]].targets[t]
        preds = {f"ridge_{k}": v for k, v in tr.paired.items()} | {"detector": inp.auto_paired[t].to_numpy(float)}
        if tg.kind(t) == "binary":
            preds |= {f"logistic_{k}": v for k, v in logistic[b["best_layer"], t].paired.items()}
        for key, pred in preds.items():
            m = np.isfinite(y) & np.isfinite(pred)
            rws = np.flatnonzero(m)
            base = {"encoder": b["encoder"], "target": t, "best_layer": b["best_layer"], "probe": key, "n": int(m.sum())}
            for metric, (point, boot) in pr.metric_draws(t, y[m], pred[m], rws, d).items():
                rows.append(base | {"metric": metric} | pr.summary(point, boot))
            if tg.kind(t) != "binary" and m.sum() > 2:
                rows.append(base | {"metric": "Spearman"} | spearman_summary(y[m], pred[m], rws, d))
                err = pred[m] - y[m]
                w = d.weights[:, rws]
                rows.append(base | {"metric": "mean_signed_error"} | pr.summary(float(err.mean()), (w @ err) / w.sum(1)))
    paired_df = pd.DataFrame(rows)
    paired_df.to_csv(out / "paired_vs_statsbomb.csv", index=False)
    (out / "logistic_best.json").write_text(json.dumps(xp.jsonable(logistic_rows), indent=1))
    figures(out)
    timing = pd.DataFrame([x for o in OFFSETS for layer in LAYERS for x in load(run_file(out, o, layer)).timing])
    timing.to_csv(out / "timing.csv", index=False)
    complete = {
        "ticket": "xg-dino 06", "layers": LAYERS, "offsets_ms": list(OFFSETS), "token": "both (CLS + mean patch)",
        "probe": "ridge by one eigendecomposition per split and variant (GPU float64), all targets and penalties",
        "alphas": pr.ALPHAS, "lambdas_logistic": pr.LAMBDAS, "rules": rules.__dict__,
        "counts": {str(o): int(len(Inputs(o).train_idx)) for o in OFFSETS} | {"paired": int(len(inp.paired_idx))},
        "inputs": {"layers_index": sha256_file(xp.LAYERS / "index.parquet"),
                   "layers_complete": sha256_file(xp.LAYERS / "complete.json"),
                   "targets_automatic": sha256_file(TICKET04 / "targets_automatic.parquet")},
        "best_layers_sha256": sha256_file(out / "best_layers.json"),
        "timing": {"layer_total_s": timing[timing.model == "layer_total"].wall_s.describe().to_dict(),
                   "ridge_unit_s": timing[timing.model == "ridge_eigh"].wall_s.describe().to_dict()},
        "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
        "files": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.glob("*")) if p.is_file() and p.name != "complete.json"},
        "wall_s_finish": time.time() - started}
    (out / "complete.json").write_text(json.dumps(xp.jsonable(complete), indent=1))
    log("done")


def figures(out: Path) -> None:
    """One small figure per target group: embedding-only metric vs block (DINOv2 and G-combo, both offsets), the
    distance-only reference and the readable threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = pd.read_csv(out / "curves.csv")
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    rules = pr.DEFAULT_RULES
    for group, names in GROUPS.items():
        fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 2.8), squeeze=False)
        for ax, t in zip(axes[0], names, strict=True):
            metric = pr.metric_name(t)
            c = curves[(curves.target == t) & (curves.metric == metric)]
            for offset, style in ((0, "-"), (-400, "--")):
                e = c[(c.offset_ms == offset) & (c.probe == "embedding")]
                dino = e[e.layer.str.startswith("dino_blocks")].assign(b=lambda x: x.layer.str.split(":").str[1].astype(int)).sort_values("b")
                g = e[e.layer.str.startswith("g_blocks")].assign(b=lambda x: x.layer.str.split(":").str[1].astype(int)).sort_values("b")
                ax.plot(dino.b, dino.point, style, color="#1f77b4", lw=1.4, label=f"DINOv2 {offset} ms")
                ax.plot(g.b, g.point, style, color="#d62728", lw=1.4, marker="o", ms=3, label=f"G-combo {offset} ms")
                if offset == 0:
                    ax.fill_between(dino.b, dino.lo, dino.hi, color="#1f77b4", alpha=0.15, lw=0)
                    ax.fill_between(g.b, g.lo, g.hi, color="#d62728", alpha=0.15, lw=0)
                    ref = c[(c.offset_ms == 0) & (c.layer == "reference") & (c.probe == "distance")]
                    if len(ref) and t != "distance":
                        ax.axhline(float(ref.point.iloc[0]), color="0.3", lw=1, ls=":", label="distance only")
            ax.axhline(rules.readable_min["binary" if metric == "AUC" else "continuous"], color="0.6", lw=0.8, ls="-.",
                       label="readable")
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("block", fontsize=8)
            ax.set_ylabel(f"OOF {metric} (embedding only)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0][0].legend(fontsize=6, loc="best")
        fig.tight_layout()
        path = figdir / f"{group}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        if path.stat().st_size > 200_000:
            raise ValueError(f"{path} exceeds 200 KB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = ap.add_mutually_exclusive_group(required=True)
    for s in ("verify", "fit", "select", "paired", "finish"):
        stage.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seconds", type=float, default=480)
    ap.add_argument("--protocol", type=Path, help="JSON with a `probe_rules` object (C2 defaults otherwise)")
    args = ap.parse_args()
    started = time.time()
    rules = pr.Rules.from_protocol(json.loads(args.protocol.read_text()).get("probe_rules") if args.protocol else None)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.finish:
        finish(args.out, rules, started)
        return 0
    with no_statsbomb_reads():
        if args.verify:
            verify(args.out, args.device)
        elif args.fit:
            done = fit(args.out, args.device, args.max_seconds)
            log("all layers stored" if done else "partial")
        elif args.select:
            select(args.out, rules)
        elif args.paired:
            done = paired(args.out, args.device, args.max_seconds)
            log("paired reads complete" if done else "partial")
    log(f"wall {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
