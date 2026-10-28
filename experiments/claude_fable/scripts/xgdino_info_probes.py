"""Probes of xg-dino-info on the automatic attributes Xa, the header labels and `z_C - z_B` (ticket 12).

    ./run.sh --no-sync python scripts/xgdino_info_probes.py --check                       # protocol and input hashes
    ./run.sh --no-sync python scripts/xgdino_info_probes.py --prep                        # geometry, outcome, z_C - z_B
    ./run.sh --no-sync python scripts/xgdino_info_probes.py --fit --offset 0 --max-seconds 540     # resumable, CPU
    ./run.sh --no-sync python scripts/xgdino_info_probes.py --finish                      # read-out, complete.json last

The pinned protocol (`research/xgdino_info_protocol.json`, sha256 in `PROTOCOL_SHA256`; never edited) binds the probe
families: every Xa column, `set_piece`, the header labels and `z_C - z_B`, frozen DINOv2 and G-combo blocks 20-23
(five-fold-model routing, C1), offsets 0 and -400 ms, C2 thresholds, nested layer selection, BH within each family.
Fitting is `xgpoc.dino.curves.run` (CPU float64 eigendecomposition ridge; binary targets are scored by the AUC of the
ridge output); `xgpoc.dino.info_probes` holds the targets, weighting and tests. Everything runs inside
`no_statsbomb_reads()`. The goal outcome is read once, in --prep, only to report the header probes by outcome.
Per (offset, layer, row group) runs are stored in `<STORE>/runs/` and skipped when present.
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
import xgdino_probes as xp
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import ablation as ab
from xgpoc.dino import curves as cv
from xgpoc.dino import info_probes as ip
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg
from xgpoc.stack import inputs as si

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_info_probes"
STORE = Path("/mnt/storage/tactifoot_research/xg_dino/info/probes")
PROTOCOL = ROOT / "research/xgdino_info_protocol.json"
PROTOCOL_SHA256 = "39a547b5b7b7642e5327a1a0de362faa46b3043dc79189736757b2a37bba3298"
OFFSETS = (0, -400)
BLOCKS = (20, 21, 22, 23)
ENCODERS = {"dino": "dino_blocks", "g_combo": "g_blocks"}
LAYERS = [f"{kind}:{b}" for kind in ENCODERS.values() for b in BLOCKS]
KEYS = ("embedding", "distance_embedding", "embedding_shuffled", "distance", "constant")
GEO_KEYS = ("geometry_embedding", "geometry")
log = xp.log


def protocol() -> dict:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise SystemExit("the protocol file differs from the pinned sha256; not editing it, not running")
    return json.loads(PROTOCOL.read_text())


def families(p: dict) -> dict[str, list[str]]:
    return p["frozen"]["probes"]["targets"]


def check_inputs(p: dict) -> dict:
    """sha256 of every protocol input this ticket reads, against the protocol's record."""
    ins, out = p["inputs"], {}
    for name in ("attributes", "header_gpt", "gpt_sample", "header_caption", "frozen_oof"):
        path = Path(ins[name]["path"])
        path = path if path.is_absolute() else ROOT / path
        out[name] = {"path": str(path), "sha256": sha256_file(path), "protocol": ins[name]["sha256"]}
        if out[name]["sha256"] != ins[name]["sha256"]:
            raise SystemExit(f"{name} differs from the protocol's record")
    return out


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.rename(path)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ inputs
def prep(p: dict) -> Path:
    """Geometry design of B, the goal outcome (header-by-outcome only) and `z_C - z_B`, by shot id, to STORE."""
    ins = check_inputs(p)
    with no_statsbomb_reads():
        inputs = ab.load_inputs()
        table = inputs.primary.table
        arm = ab.arm_set(table, ("B",))[0]["B"]
        lin = [c for c in arm.columns if "_ns" not in c]
        out = table[[*lin, "dist@0"]].add_prefix("geo_")
        out["goal"] = si.load_outcomes(inputs.primary).aligned(inputs.primary)
    probs = pd.read_csv(ins["frozen_oof"]["path"])
    z = ip.z_difference(probs)
    out = out.join(z)
    if out[ip.ZDIFF].isna().any():
        raise ValueError("a population shot has no frozen OOF")
    STORE.mkdir(parents=True, exist_ok=True)
    path = STORE / "prep.parquet"
    out.to_parquet(path)
    log(f"prep: {out.shape} -> {path}; goals {int(out.goal.sum())}; geometry columns {len(lin) + 1}")
    return path


class Inputs:
    """Population, probe tables and layer matrices of one offset."""

    def __init__(self, offset: int, fam: dict[str, list[str]]):
        index = pd.read_parquet(xp.LAYERS / "index.parquet")
        self.offset = offset
        self.train_idx, _ = xp.population(index, offset)
        ids = self.train_idx.index
        prepared = pd.read_parquet(STORE / "prep.parquet")
        attributes = pd.read_parquet(ROOT / "results/xgdino_info_attributes/attributes.parquet")
        auto = pd.read_parquet(ROOT / "results/xgdino_probes_last/targets_automatic.parquet")
        gpt = pd.read_csv(ROOT / "results/xgdino_info_labels/header_gpt.csv")
        caption = pd.read_csv(ROOT / "results/xgdino_info_labels/header_caption.csv")
        sample = pd.read_csv(ROOT / "results/xgdino_info_labels/gpt_sample.csv").set_index("shot_id")
        labels = ip.header_targets(gpt, caption, prepared.index)
        wanted = [t for ts in fam.values() for t in ts]
        xa = [t for t in wanted if t not in ip.HEADER and t != ip.ZDIFF]
        self.tables = {
            "automatic": ip.probe_table(attributes, labels, prepared[ip.ZDIFF], auto.distance, [*xa, ip.ZDIFF], ids),
            "gpt": ip.probe_table(attributes, labels, prepared[ip.ZDIFF], auto.distance, ["header_gpt"], ids),
            "caption": ip.probe_table(attributes, labels, prepared[ip.ZDIFF], auto.distance, ["header_caption"], ids)}
        # set_piece belongs to the flags family of the protocol; the table holds it once
        self.data = {s: pr.ProbeData(t, self.train_idx.game.to_numpy(), self.train_idx.fold.to_numpy())
                     for s, t in self.tables.items()}
        self.geometry = prepared.filter(like="geo_").reindex(ids).to_numpy(float)
        self.goal = prepared.goal.reindex(ids).to_numpy(float)
        self.header_weight = sample.weight.reindex(ids).fillna(1.0).to_numpy(float)
        self.groups = {s: [(f"{s}__{names[0]}", rows, names) for rows, names in
                           cv.mask_groups(d, [c for c in d.targets.columns if c != "distance"])]
                       for s, d in self.data.items()}
        self._matrix: dict[str, np.ndarray] = {}

    def matrix(self, layer: str) -> np.ndarray:
        """float16 [n, d] (dino) or [n, 5, d] (g_combo) of the training rows."""
        if layer not in self._matrix:
            kind, block = xp.parse_layer(layer)
            rows = self.train_idx.row.to_numpy()
            arr = np.load(xp.LAYERS / f"{kind}.npy", mmap_mode="r")
            order = np.argsort(rows)
            if kind == "dino_blocks":
                got = np.asarray(arr[rows[order], block]).reshape(len(rows), -1)
            else:
                got = np.asarray(arr[rows[order]][:, :, block - xp.G_FIRST_BLOCK]).reshape(len(rows), xp.N_FOLDS, -1)
            out = np.empty_like(got)
            out[order] = got
            self._matrix[layer] = out
        return self._matrix[layer]

    def encoder(self, layer: str) -> pr.Encoder:
        a = self.matrix(layer)
        if a.ndim == 2:
            return pr.Encoder(layer, lambda k: a.astype(np.float32))
        return pr.Encoder(layer, lambda k: a[:, k].astype(np.float32))


def run_file(offset: int, layer: str, group: str) -> Path:
    return STORE / "runs" / f"o{offset}" / layer.replace(":", "_") / f"{group}.pkl"


# ------------------------------------------------------------------ fit
def fit(offsets: list[int], max_seconds: float, fam: dict) -> bool:
    started = time.time()
    torch.set_num_threads(int(__import__("os").environ.get("XGD_THREADS", "8")))
    for offset in offsets:
        inp = Inputs(offset, fam)
        (STORE / f"groups_o{offset}.json").write_text(json.dumps(
            {s: [{"group": g, "rows": int(len(r)), "targets": n} for g, r, n in gs] for s, gs in inp.groups.items()}, indent=1))
        for layer in LAYERS:
            todo = [(s, g, names) for s, gs in inp.groups.items() for g, _, names in gs if not run_file(offset, layer, g).exists()]
            if not todo:
                continue
            enc = inp.encoder(layer)
            for s, g, names in todo:
                if time.time() - started > max_seconds:
                    log("time budget reached; rerun to resume")
                    return False
                t0 = time.time()
                geo = inp.geometry if names == [ip.ZDIFF] else None
                r = cv.run(inp.data[s], enc, names, geometry=geo)
                save({"run": r, "targets": names, "source": s}, run_file(offset, layer, g))
                log(f"offset {offset} {layer} {g} ({len(names)} targets): {time.time() - t0:.0f} s")
    return True


# ------------------------------------------------------------------ read-out
def merged(offset: int, layer: str, inp: Inputs) -> dict[str, pr.Run]:
    """One `Run` per probe table (source) holding all its targets, from the stored group runs."""
    out = {}
    for s, gs in inp.groups.items():
        parts = [load(run_file(offset, layer, g))["run"] for g, _, _ in gs]
        out[s] = pr.Run(parts[0].encoder, parts[0].folds, {t: v for r in parts for t, v in r.targets.items()},
                        [x for r in parts for x in r.timing])
    return out


def flat(prefix: str, s: dict) -> dict:
    return {f"{prefix}_point": s["point"], f"{prefix}_lo": s["lo"], f"{prefix}_hi": s["hi"]}


def summarise(offset: int, encoder: str, family: str, target: str, runs: dict[str, pr.Run], data: pr.ProbeData,
              d: pr.Draws, rules: pr.Rules, rw: np.ndarray | None, label: str | None = None) -> tuple[dict, dict]:
    """One summary row (best layer, metrics, nested decision) and the per-block curve metrics of one target."""
    binary = tg.kind(target) == "binary"
    chosen = pr.chosen_layer(runs, data, target, rules)
    composite, nested_blocks = pr.nested(runs, data, target, rules)
    keys = (*KEYS, *GEO_KEYS) if ip.ZDIFF == target else KEYS
    at = ip.target_metrics_w(runs[chosen].targets[target], d, keys, rw)
    nested_md = ip.target_metrics_w(composite, d, keys, rw)
    dec, nested_dec = pr.decide(target, at, rules), pr.decide(target, nested_md, rules)
    name = pr.metric_name(target)
    t = runs[chosen].targets[target]
    row = {"offset_ms": offset, "encoder": encoder, "family": family, "target": label or target,
           "role": "descriptive Xq (may carry broadcast / annotation artefacts)" if family == "quality" else "content",
           "kind": "binary" if binary else "continuous", "metric": name, "n_rows": int(len(t.rows)),
           "positives": int(t.y.sum()) if binary else None, "best_block": int(chosen),
           "best_layer": f"{ENCODERS[encoder]}:{chosen}", **flat("embedding", pr.summary(*at["embedding"][name])),
           **flat("distance_embedding", pr.summary(*at["distance_embedding"][name])),
           **flat("distance_only", pr.summary(*at["distance"][name])), "constant_point": at["constant"][name][0],
           **flat("gain_nested", nested_dec["gain"]), **flat("gain_at_best", dec["gain"]),
           "p_gain": ip.one_sided_p(nested_md["distance_embedding"][name][1] - nested_md["distance"][name][1]),
           "readable": dec["readable"], "encodes_c2": nested_dec["encodes_beyond_distance"],
           "nested_blocks": ",".join(map(str, nested_blocks)),
           "shuffled_point": at["embedding_shuffled"][name][0] if "embedding_shuffled" in at else None}
    if ip.ZDIFF == target:
        geo = ip.geometry_decision(nested_md, rules)
        row |= {**flat("geometry_embedding", pr.summary(*nested_md["geometry_embedding"]["R2"])),
                **flat("geometry_only", pr.summary(*nested_md["geometry"]["R2"])), **flat("gain_beyond_geometry", geo["gain"]),
                "p_beyond_geometry": geo["p"], "encodes_beyond_geometry": geo["encodes_beyond_geometry"]}
    curve = {}
    for block, r in runs.items():
        md = ip.target_metrics_w(r.targets[target], d, ("embedding", "distance_embedding"), rw)
        curve[block] = {k: pr.summary(*md[k][name]) for k in md}
    return row, curve


def by_outcome(offset: int, encoder: str, target: str, run: pr.Run, data: pr.ProbeData, d: pr.Draws, goal: np.ndarray,
               rw: np.ndarray | None, layer: str, weighted: bool) -> list[dict]:
    """The header probe on all shots, goals and non-goals: AUC (embedding, distance + embedding, distance only),
    prevalence and mean score."""
    t = run.targets[target]
    rows = []
    for stratum, value in ip.OUTCOMES.items():
        mask = np.ones(len(t.rows), bool) if value is None else goal[t.rows] == value
        md = ip.target_metrics_w(t, d, ("embedding", "distance_embedding", "distance", "constant"), rw, mask)
        base = {"offset_ms": offset, "encoder": encoder, "target": target, "weighted": weighted, "layer": layer,
                "outcome": stratum, "n": int(mask.sum()), "positives": int(t.y[mask].sum())}
        prev = ip.prevalence(t, d, rw, mask)
        rows.append(base | {"probe": "prevalence", "metric": "share", **prev})
        for key, metrics in md.items():
            for metric, (point, boot) in metrics.items():
                if key != "constant":
                    rows.append(base | {"probe": key, "metric": metric} | pr.summary(point, boot))
        gain = (md["distance_embedding"]["AUC"][0] - md["distance"]["AUC"][0],
                md["distance_embedding"]["AUC"][1] - md["distance"]["AUC"][1])
        rows.append(base | {"probe": "gain_over_distance", "metric": "AUC"} | pr.summary(*gain))
        if value is None:
            continue
        # mean ridge score of the true heads and the true feet within the outcome (calibration of the direction)
        for label, lv in (("score_head", 1.0), ("score_foot", 0.0)):
            m = mask & (t.y == lv)
            rows.append(base | {"probe": "embedding", "metric": label, "point": float(t.oof["embedding"][m].mean()) if m.any() else np.nan,
                                "lo": np.nan, "hi": np.nan})
    return rows


def finish(fam: dict, rules: pr.Rules, started: float, ins: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, curves, outcome_rows, timing = [], [], [], []
    for offset in OFFSETS:
        inp = Inputs(offset, fam)
        by_layer = {layer: merged(offset, layer, inp) for layer in LAYERS}
        timing += [x | {"offset_ms": offset} for ms in by_layer.values() for r in ms.values() for x in r.timing]
        draws = {s: pr.draws(inp.data[s].games) for s in inp.data}
        for encoder, kind in ENCODERS.items():
            runs_by_source = {src: {str(b): by_layer[f"{kind}:{b}"][src] for b in BLOCKS} for src in inp.data}
            for family, targets in fam.items():
                rows_family = []
                for target in targets:
                    source = "gpt" if target == "header_gpt" else "caption" if target == "header_caption" else "automatic"
                    data, d = inp.data[source], draws[source]
                    runs = runs_by_source[source]
                    rw = inp.header_weight if target == "header_gpt" else None
                    row, curve = summarise(offset, encoder, family, target, runs, data, d, rules, rw)
                    rows_family.append(row)
                    for block, c in curve.items():
                        for key, s in c.items():
                            curves.append({"offset_ms": offset, "encoder": encoder, "family": family, "target": target,
                                           "block": int(block), "probe": key} | s)
                    if target in ip.HEADER:
                        best = runs[str(row["best_block"])]
                        outcome_rows += by_outcome(offset, encoder, target, best, data, d, inp.goal, rw, row["best_layer"], rw is not None)
                        if rw is not None:      # the unweighted read beside the weighted primary
                            unw, _ = summarise(offset, encoder, family, target, runs, data, d, rules, None, "header_gpt_unweighted")
                            summary.append(unw | {"bh_reject": None, "encodes": None})
                            outcome_rows += by_outcome(offset, encoder, target, best, data, d, inp.goal, None, row["best_layer"], False)
                reject = ip.bh([r["p_gain"] for r in rows_family])
                for r, rej in zip(rows_family, reject, strict=True):
                    r["bh_reject"], r["encodes"] = rej, bool(rej and r["encodes_c2"])
                summary += rows_family
                log(f"offset {offset} {encoder} {family}: {len(targets)} targets")
    summary_df, curves_df, outcome_df = pd.DataFrame(summary), pd.DataFrame(curves), pd.DataFrame(outcome_rows)
    summary_df.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    curves_df.to_csv(OUT / "curves.csv", index=False, float_format="%.6g")
    outcome_df.to_csv(OUT / "header_by_outcome.csv", index=False, float_format="%.6g")
    tdf = pd.DataFrame(timing)
    tdf.to_csv(OUT / "timing.csv", index=False)
    groups = {o: json.loads((STORE / f"groups_o{o}.json").read_text()) for o in OFFSETS}
    (OUT / "row_groups.json").write_text(json.dumps(groups, indent=1))
    complete = {
        "ticket": "xg-dino 12", "protocol_sha256": PROTOCOL_SHA256, "inputs": ins, "layers": LAYERS, "offsets_ms": list(OFFSETS),
        "families": fam, "rules": rules.__dict__, "alphas": pr.ALPHAS, "bh_q": ip.Q, "main_share": ip.MAIN_SHARE,
        "probe": "ridge by one eigendecomposition per split and variant (CPU float64), all targets and penalties; binary "
                 "targets scored by the AUC of the ridge output; header probes weighted in the metric by the GPT stratum "
                 "weights (GPT primary), fit unweighted",
        "population": {"train_rows": {str(o): int(len(Inputs(o, fam).train_idx)) for o in OFFSETS}},
        "outputs": {p.name: sha256_file(p) for p in sorted(OUT.glob("*")) if p.is_file() and p.name != "complete.json"},
        "wall_s_fit": float(tdf.wall_s.sum()), "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
        "wall_s_finish": time.time() - started}
    (OUT / "complete.json").write_text(json.dumps(xp.jsonable(complete), indent=1))
    log("done")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = ap.add_mutually_exclusive_group(required=True)
    for s in ("check", "prep", "fit", "finish"):
        stage.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--offset", type=int, choices=OFFSETS)
    ap.add_argument("--max-seconds", type=float, default=540)
    args = ap.parse_args()
    started = time.time()
    p = protocol()
    ins = check_inputs(p)
    fam = families(p)
    if args.check:
        print(json.dumps({"protocol_sha256": PROTOCOL_SHA256, "inputs": ins, "families": {k: len(v) for k, v in fam.items()}}, indent=1))
    elif args.prep:
        prep(p)
    elif args.fit:
        with no_statsbomb_reads():
            done = fit([args.offset] if args.offset is not None else list(OFFSETS), args.max_seconds, fam)
        log("all runs stored" if done else "partial")
    else:
        with no_statsbomb_reads():
            finish(fam, pr.Rules.from_protocol(None), started, ins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
