"""Agreement of DINO probe, detector and StatsBomb geometry, and frame synchronisation (ticket 07; `xgpoc.dino.agreement`,
councils C1 and C2). Descriptive, on the 2,721 paired shots.

    ./run.sh --no-sync python scripts/xgdino_agreement.py --predict     # resumable: -400 ms probe reads, detector
    ./run.sh --no-sync python scripts/xgdino_agreement.py --finish      # StatsBomb evaluation, gallery, complete.json

--predict (inside `no_statsbomb_reads()`): for every offset-0 best layer of ticket 06 that an agreement feature uses, the
ridge probes are refitted exactly as ticket 06 (same splits, penalties; the training OOF must equal the stored curve
run) and read on the paired shots' -400 ms embeddings: the t0-trained probe applied to the earlier frame, so a change
between offsets is motion seen by one fixed instrument. The detector's automatic targets at -400 ms come from the same
feet_t0 replay (`targets.automatic_targets`), the t0 ones are recomputed and checked against ticket 04's table.

--finish reads StatsBomb once (`targets.statsbomb_targets`) and the pairing residual column once, outside the guard,
and scores the frozen sources: errors and their agreement, the C2 cells with the permutation null, the fixed
combination rules, synchronisation (t0 vs -400, own ball at both offsets, cut frames dropped) and timing bands.
Tolerances are read from the pinned ablation protocol and checked against C2. Nothing is fitted to StatsBomb.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgdino_layer_curves as lc
import xgdino_probes as xp
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import agreement as ag
from xgpoc.dino import curves as cv
from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_agreement"
WORK = OUT / "work"
TICKET06 = lc.OUT
PROTOCOL = ROOT / "research/xgdino_ablation_protocol.json"
PAIRS = ROOT / "results/xgv_pairing/pairs.parquet"
FRAMES = ROOT / "cache/xgc/offset_frames_v1"
ENCODERS = ("g_combo", "dino")                       # G-combo primary
GALLERY_MAX, GALLERY_PER_FEATURE = 24, 4
log = xp.log


def best_layers() -> dict[tuple[str, str], str]:
    """{(encoder, feature): offset-0 best layer} of ticket 06 for the agreement features."""
    return {(b["encoder"], b["target"]): b["best_layer"] for b in lc.best_table(TICKET06) if b["target"] in ag.TARGET_COLUMNS}


def layer_targets() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for (_, t), layer in sorted(best_layers().items()):
        out.setdefault(layer, [])
        if t not in out[layer]:
            out[layer].append(t)
    return {k: [t for t in ag.TARGET_COLUMNS if t in v] for k, v in out.items()}


def paired_index(offset: int) -> pd.DataFrame:
    index = pd.read_parquet(xp.LAYERS / "index.parquet")
    return xp.population(index, offset)[1]


# ------------------------------------------------------------------ --predict (no StatsBomb)
def predict(device: str, max_seconds: float) -> bool:
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    inp = lc.Inputs(0)
    p400 = paired_index(-400)
    if not p400.index.equals(inp.paired_idx.index):
        raise ValueError("the paired shots differ between offsets")
    for layer, targets in layer_targets().items():
        path = WORK / f"probe_m400_{lc.tag(layer)}.parquet"
        if path.exists():
            continue
        if time.time() - started > max_seconds:
            log("time budget reached; rerun to resume")
            return False
        t0 = time.time()
        enc = xp.encoder(layer, "both", inp.train_idx, p400)
        run = cv.run(inp.data, enc, targets, paired=True, device=device)
        stored = lc.load(lc.run_file(TICKET06, 0, layer))
        check = {}
        for t in targets:
            a, b = run.targets[t].oof["embedding"], stored.targets[t].oof["embedding"]
            if not np.allclose(a, b, rtol=1e-7, atol=1e-7, equal_nan=True) \
                    or run.targets[t].penalty["embedding"] != stored.targets[t].penalty["embedding"]:
                raise ValueError(f"{layer} {t}: the refitted probe differs from ticket 06's")
            check[t] = float(np.nanmax(np.abs(a - b)))
        pd.DataFrame({t: run.targets[t].paired["embedding"] for t in targets}, index=p400.index).to_parquet(path)
        log(f"-400 probe {layer} {targets}: {time.time() - t0:.0f} s, max |OOF - ticket 06| {max(check.values()):.2e}")
    path = WORK / "detector.parquet"
    if not path.exists():
        ids = list(inp.paired_idx.index)
        frames = pd.read_parquet(xp.FEET / "frames.parquet", filters=[("shot_id", "in", ids)])
        t0 = tg.automatic_targets(frames, ids, 0)
        stored = pd.read_parquet(lc.TICKET04 / "targets_automatic.parquet").loc[t0.index]
        num = [c for c in tg.COLUMNS if c != "near_opp_status"]
        if not np.allclose(t0[num].to_numpy(float), stored[num].to_numpy(float), equal_nan=True):
            raise ValueError("t0 automatic targets differ from ticket 04's")
        m400 = tg.automatic_targets(frames, ids, -400)
        feats = pd.read_parquet(xp.FEET / "features.parquet", columns=["shot_id", "offset_ms", "ball_source"],
                                filters=[("shot_id", "in", ids)]).pivot(index="shot_id", columns="offset_ms",
                                                                        values="ball_source")
        out = pd.concat({"t0": t0[num], "m400": m400[num]}, axis=1)
        out.columns = [f"{c}@{o}" for o, c in out.columns]
        out["ball_source@t0"], out["ball_source@m400"] = feats[0].loc[out.index], feats[-400].loc[out.index]
        out.to_parquet(path)
        log("detector targets at 0 and -400 ms written (t0 equal to ticket 04)")
    return True


# ------------------------------------------------------------------ --finish
class Sources:
    """Frozen per-shot values (index = paired shot ids, sorted) of every source, feature and offset."""

    def __init__(self):
        inp = lc.Inputs(0)
        self.idx = inp.paired_idx
        self.games = self.idx.game.to_numpy()
        det = pd.read_parquet(WORK / "detector.parquet").loc[self.idx.index]
        self.det = {}
        for o in ("t0", "m400"):
            for f in ag.FEATURES:
                v = det[f"{f.name}@{o}"].to_numpy(float).copy()
                if f.goalkeeper:
                    v[det[f"gk_detected@{o}"].to_numpy() != 1] = np.nan     # proxy goalkeeper -> missing (C2)
                self.det[f.name, o] = v
        self.ball = det[["ball_source@t0", "ball_source@m400"]]
        self.ball_own_both = (det["ball_source@t0"] == 0).to_numpy() & (det["ball_source@m400"] == 0).to_numpy()
        self.layers = best_layers()
        self.probe, self.dist = {}, {}
        runs = {}
        for (enc, t), layer in self.layers.items():
            runs[layer] = runs.get(layer) or lc.load(TICKET06 / "runs_paired" / f"{lc.tag(layer)}.pkl")
            tr = runs[layer].targets[t]
            self.probe[enc, t, "t0"] = tr.paired["embedding"]
            self.dist[enc, t] = tr.paired["distance"]
            m400 = pd.read_parquet(WORK / f"probe_m400_{lc.tag(layer)}.parquet").loc[self.idx.index]
            self.probe[enc, t, "m400"] = m400[t].to_numpy(float)
        ag.check_sources(det)


def finish(started: float) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    tau = ag.tolerances(protocol["frozen"]["agreement"])
    pnull = protocol["frozen"]["cell_permutation_null"]
    if pnull["permutations"] != ag.N_PERMUTATIONS or pnull["seed"] != ag.PERMUTATION_SEED:
        raise ValueError("the permutation null differs from the protocol")
    if protocol["frozen"]["timing_bands"]["min_shots"] != ag.MIN_BAND:
        raise ValueError("the timing-band minimum differs from the protocol")
    with no_statsbomb_reads():
        src = Sources()
        index = pd.read_parquet(xp.LAYERS / "index.parquet")
    ids = src.idx.index
    # outside the guard: the one StatsBomb read and the pairing residual column, evaluation only
    sb = tg.statsbomb_targets(ids).loc[ids]
    resid = pq.read_table(PAIRS, columns=["shot_id", "residual_ms"]).to_pandas().set_index("shot_id").residual_ms
    resid = resid.reindex(ids).to_numpy(float)
    d = pr.draws(src.games)
    sb_dist = sb.distance.to_numpy(float)
    strata = ag.distance_strata(sb_dist)

    errors, cells, perms, combos, cell_rows = [], [], [], [], []
    for f in ag.FEATURES:
        y = sb[f.name].to_numpy(float)
        det = src.det[f.name, "t0"]
        for enc in ENCODERS:
            probe, dist = src.probe[enc, f.name, "t0"], src.dist[enc, f.name]
            base = {"encoder": enc, "layer": src.layers[enc, f.name], "tau": tau[f.name]}
            errors.append(base | ag.error_table(f, y, det, probe, d)
                          | {"probe_mae_all_sb": ag.mean_ci(np.abs(probe - y)[np.isfinite(y)] * f.scale, np.flatnonzero(np.isfinite(y)), d),
                             "n_sb": int(np.isfinite(y).sum()), "n_detector_missing": int((np.isfinite(y) & ~np.isfinite(det)).sum())})
            m = np.isfinite(y)
            ry = ag.reading(f, y[m])
            rows, n = ag.cells(np.abs(ag.reading(f, det[m]) - ry), np.abs(ag.reading(f, probe[m]) - ry),
                               np.abs(ag.reading(f, dist[m]) - ry), tau[f.name])
            p = ag.permutation_null((rows.detector == "wrong").to_numpy(),
                                    (rows.probe_right & rows.probe_beats_distance).to_numpy(), strata[m])
            # diagnostic: the probe's right flag alone (the distance-only reference is a spline of the detector's own
            # distance, so "beats the reference" is tied to the detector being wrong for distance and angle)
            p_right = ag.permutation_null((rows.detector == "wrong").to_numpy(), rows.probe_right.to_numpy(), strata[m])
            p |= {"diagnostic_probe_right_only_observed": p_right["observed"],
                  "diagnostic_probe_right_only_null_mean": p_right["null_mean"],
                  "diagnostic_probe_right_only_p": p_right["p_one_sided"]}
            cells.append(base | {"feature": f.name} | n)
            perms.append(base | {"feature": f.name} | p)
            rows.insert(0, "shot_id", ids[m])
            cell_rows.append(rows.assign(encoder=enc, feature=f.name))
            combos += [base | r for r in ag.combination_table(f, y, det, probe, dist, d)]
    errors_df, combos_df = ag.flatten(errors), ag.flatten(combos)
    cells_df = pd.DataFrame(cells).merge(pd.DataFrame(perms), on=["encoder", "layer", "tau", "feature"])
    OUT.mkdir(parents=True, exist_ok=True)
    errors_df.to_csv(OUT / "errors.csv", index=False)
    cells_df.to_csv(OUT / "cells.csv", index=False)
    combos_df.to_csv(OUT / "combinations.csv", index=False)
    cell_rows = pd.concat(cell_rows)
    cell_rows.to_parquet(OUT / "cells_per_shot.parquet")

    sync_df, bands_df, sync_info = sync(src, sb, resid, index)
    sync_df.to_csv(OUT / "sync.csv", index=False)
    bands_df.to_csv(OUT / "timing_bands.csv", index=False)
    (OUT / "sync_summary.json").write_text(json.dumps(xp.jsonable(sync_info), indent=1))
    figures(errors_df, bands_df, src, sb)
    gallery = make_gallery(cell_rows, src, sb)
    complete = {
        "ticket": "xg-dino 07", "descriptive": True, "paired_shots": len(ids), "matches": int(src.idx.game.nunique()),
        "tolerances": tau, "wrong_quantile": ag.WRONG_QUANTILE, "permutation_null": pnull,
        "probe": "embedding-only ridge at ticket 06's offset-0 best layer (fitted on automatic targets of the 328 "
                 "matches), mean of the five fold probes; counts rounded for the cells",
        "best_layers": {f"{e}:{t}": layer for (e, t), layer in src.layers.items()},
        "statsbomb": "targets.statsbomb_targets, read once outside the guard, evaluation only; nothing fitted",
        "gallery": gallery,
        "inputs": {"protocol": sha256_file(PROTOCOL), "best_layers": sha256_file(TICKET06 / "best_layers.json"),
                   "layers_index": sha256_file(xp.LAYERS / "index.parquet"),
                   "targets_automatic": sha256_file(lc.TICKET04 / "targets_automatic.parquet"),
                   "feet_features": sha256_file(xp.FEET / "features.parquet"),
                   **{f"work/{p.name}": sha256_file(p) for p in sorted(WORK.glob("*.parquet"))}},
        "code": records.code_hashes(Path(__file__)) | {"agreement": sha256_file(Path(ag.__file__))},
        "git": records.git_state(),
        "files": {str(p.relative_to(OUT)): sha256_file(p) for p in sorted(OUT.rglob("*"))
                  if p.is_file() and p.name != "complete.json" and "gallery" not in p.parts and "work" not in p.parts},
        "wall_s_finish": time.time() - started}
    (OUT / "complete.json").write_text(json.dumps(xp.jsonable(complete), indent=1))
    log("done")


def sync(src: Sources, sb: pd.DataFrame, resid: np.ndarray, index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """t0 agreement with StatsBomb, the -400 -> t0 change of each source (motion), and errors by timing band."""
    pi = index[index.set == "paired"].set_index(["offset_ms", "shot_id"])
    cut = pi.loc[-400].cut_between_anchor.reindex(src.idx.index).fillna(False).to_numpy(bool)
    keep = src.ball_own_both & ~cut
    rows = []
    for f in ag.FEATURES:
        y = sb[f.name].to_numpy(float) * f.scale
        series = {"detector": (src.det[f.name, "t0"], src.det[f.name, "m400"])}
        series |= {f"probe_{e}": (src.probe[e, f.name, "t0"], src.probe[e, f.name, "m400"]) for e in ENCODERS}
        changes = {}
        for name, (a0, a4) in series.items():
            a0, a4 = np.asarray(a0) * f.scale, np.asarray(a4) * f.scale
            m = keep & np.isfinite(y) & np.isfinite(a0) & np.isfinite(a4) & np.isfinite(src.det[f.name, "t0"]) \
                & np.isfinite(src.det[f.name, "m400"])
            r = np.flatnonzero(m)
            ch = a0[m] - a4[m]
            changes[name] = np.where(m, a0 - a4, np.nan)
            rows.append({"feature": f.name, "source": name, "n": int(m.sum()),
                         "mae_t0": ag.mean_ci(np.abs(a0[m] - y[m]), r, pr.draws(src.games)),
                         "mae_m400": ag.mean_ci(np.abs(a4[m] - y[m]), r, pr.draws(src.games)),
                         "mean_change_t0_minus_m400": ag.mean_ci(ch, r, pr.draws(src.games)),
                         "median_abs_change": float(np.median(np.abs(ch))) if len(ch) else np.nan})
        for e in ENCODERS:
            a, b = changes["detector"], changes[f"probe_{e}"]
            m = np.isfinite(a) & np.isfinite(b)
            rows.append({"feature": f.name, "source": f"spearman_change_detector_vs_probe_{e}", "n": int(m.sum()),
                         "spearman": ag.spearman_ci(a[m], b[m], np.flatnonzero(m), pr.draws(src.games))})
    bands, band_counts = ag.timing_bands(np.abs(resid))
    band_rows = []
    for f in ag.FEATURES:
        det = src.det[f.name, "t0"]                    # identical rows for every source: where the detector exists
        sources = {"detector": det} | {f"probe_{e}": np.where(np.isfinite(det), src.probe[e, f.name, "t0"], np.nan)
                                       for e in ENCODERS}
        band_rows += ag.band_errors(f, bands, sb[f.name].to_numpy(float), sources, src.games)
    pi0, pi4 = pi.loc[0].reindex(src.idx.index), pi.loc[-400].reindex(src.idx.index)
    info = {
        "rows": len(src.idx), "own_ball_both_offsets": int(src.ball_own_both.sum()),
        "own_ball_both_share": float(src.ball_own_both.mean()), "cut_between_anchor_m400": int(cut.sum()),
        "sync_rows": int(keep.sum()), "sync_share": float(keep.mean()),
        "ball_source": {o: {str(k): int(v) for k, v in src.ball[f"ball_source@{o}"].value_counts(dropna=False).items()}
                        for o in ("t0", "m400")},
        "decoded": {str(o): {"frame_status": p.frame_status.value_counts().to_dict(),
                             "decoded_relative_ms": p.decoded_relative_ms.describe().to_dict(),
                             "timing_error_ms_abs_max": float(p.timing_error_ms.abs().max()),
                             "cut_between_anchor": int(p.cut_between_anchor.fillna(False).sum())}
                    for o, p in ((0, pi0), (-400, pi4))},
        "timing_bands": band_counts, "residual_abs_ms": pd.Series(np.abs(resid)).describe().to_dict(),
        "residual_missing": int(np.isnan(resid).sum()),
        "caveat": "nominal t0 and -400 ms are not certified pre-contact frames (temporal study)"}
    return ag.flatten(rows), ag.flatten(band_rows), info


# ------------------------------------------------------------------ figures and gallery
def figures(errors: pd.DataFrame, bands: pd.DataFrame, src: Sources, sb: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    names = [f.name for f in ag.FEATURES]
    units = {"distance": "m", "angle": "deg", "near_opp_m": "m", "opp_3m": "count", "def_cone": "count",
             "gk_depth_m": "m", "gk_lateral_m": "m"}

    fig, axes = plt.subplots(1, len(names), figsize=(15, 2.6))
    for ax, t in zip(axes, names, strict=True):
        e = errors[errors.feature == t].set_index("encoder")
        vals = [("detector", e.loc["g_combo", "detector_mae"], e.loc["g_combo", "detector_mae_lo"], e.loc["g_combo", "detector_mae_hi"], "#444444"),
                ("G-combo", e.loc["g_combo", "probe_mae"], e.loc["g_combo", "probe_mae_lo"], e.loc["g_combo", "probe_mae_hi"], "#d62728"),
                ("DINOv2", e.loc["dino", "probe_mae"], e.loc["dino", "probe_mae_lo"], e.loc["dino", "probe_mae_hi"], "#1f77b4")]
        for i, (_, p, lo, hi, c) in enumerate(vals):
            ax.bar(i, p, color=c, width=0.7)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="k", lw=1, capsize=2)
        ax.axhline(float(e.tau.iloc[0]), color="0.5", ls=":", lw=1)
        ax.set_xticks(range(3), [v[0] for v in vals], fontsize=7, rotation=30)
        ax.set_title(f"{t} ({units[t]})", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("MAE vs StatsBomb", fontsize=8)
    fig.tight_layout()
    save(fig, figdir / "errors.png")

    fig, axes = plt.subplots(1, len(names), figsize=(15, 2.6))
    for ax, t in zip(axes, names, strict=True):
        b = bands[(bands.feature == t) & (bands.band != "missing")]
        order = list(dict.fromkeys(b.band))
        for s, c, dx in (("detector", "#444444", -0.2), ("probe_g_combo", "#d62728", 0.0), ("probe_dino", "#1f77b4", 0.2)):
            bs = b[b.source == s].set_index("band").reindex(order)
            x = np.arange(len(order)) + dx
            ax.errorbar(x, bs["mae"], yerr=[bs["mae"] - bs["mae_lo"], bs["mae_hi"] - bs["mae"]], fmt="o", ms=3, color=c,
                        lw=1, label=s)
        ax.set_xticks(range(len(order)), [f"{o}\n(n={int(b[b.band == o].n.max())})" for o in order], fontsize=6)
        ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("MAE by |timing residual| (ms)", fontsize=8)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, figdir / "timing_bands.png")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, t in zip(axes, ("distance", "gk_depth_m", "near_opp_m"), strict=True):
        f = ag.BY_NAME[t]
        a = (src.det[t, "t0"] - src.det[t, "m400"]) * f.scale
        b = (src.probe["g_combo", t, "t0"] - src.probe["g_combo", t, "m400"]) * f.scale
        m = src.ball_own_both & np.isfinite(a) & np.isfinite(b)
        ax.scatter(a[m], b[m], s=3, alpha=0.3, color="#d62728", lw=0)
        lim = np.nanpercentile(np.abs(np.r_[a[m], b[m]]), 99)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("detector change t0 - (-400 ms)", fontsize=7)
        ax.set_ylabel("G-combo probe change", fontsize=7)
        ax.set_title(f"{t} (n={int(m.sum())})", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, figdir / "sync_change.png")


def save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=80)
    plt.close(fig)
    if path.stat().st_size > 200_000:
        raise ValueError(f"{path} exceeds 200 KB")


def make_gallery(cell_rows: pd.DataFrame, src: Sources, sb: pd.DataFrame) -> dict:
    """<= 24 thumbnails of the counted "detector wrong, DINO right" cell (G-combo), the largest detector errors per
    feature first; offset-0 frames, values annotated. Not committed."""
    from PIL import Image, ImageDraw
    gdir = OUT / "gallery"
    gdir.mkdir(exist_ok=True)
    for old in gdir.glob("*.jpg"):
        old.unlink()
    frames = pd.read_csv(FRAMES / "frames.csv", usecols=["shot_id", "offset_ms", "frame_path"])
    frames = frames[frames.offset_ms == 0].set_index("shot_id").frame_path
    pos = {s: i for i, s in enumerate(src.idx.index)}
    picks = []
    counted = cell_rows[(cell_rows.encoder == "g_combo") & cell_rows.counted]
    for f in ag.FEATURES:
        c = counted[counted.feature == f.name]
        if not len(c):
            continue
        i = np.array([pos[s] for s in c.shot_id])
        y = sb[f.name].to_numpy(float)[i]
        err = np.abs(src.det[f.name, "t0"][i] - y)
        picks += [(f, c.shot_id.iloc[j]) for j in np.argsort(-err, kind="stable")[:GALLERY_PER_FEATURE]]
    picks = picks[:GALLERY_MAX]
    written = []
    for n, (f, shot) in enumerate(picks):
        i = pos[shot]
        img = Image.open(FRAMES / frames.loc[shot]).convert("RGB")
        img.thumbnail((448, 252))
        canvas = Image.new("RGB", (img.width, img.height + 30), "white")
        canvas.paste(img, (0, 30))
        y, dv = sb[f.name].iloc[i] * f.scale, src.det[f.name, "t0"][i] * f.scale
        pv = src.probe["g_combo", f.name, "t0"][i] * f.scale
        ImageDraw.Draw(canvas).text((4, 2), f"{f.name}: SB {y:.1f}  detector {dv:.1f}  G-probe {pv:.1f}\n{shot[-40:]}",
                                    fill="black")
        path = gdir / f"{n:02d}_{f.name}.jpg"
        canvas.save(path, quality=70)
        written.append({"file": path.name, "shot_id": shot, "feature": f.name})
    return {"thumbnails": len(written), "items": written}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = ap.add_mutually_exclusive_group(required=True)
    stage.add_argument("--predict", action="store_true")
    stage.add_argument("--finish", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seconds", type=float, default=420)
    args = ap.parse_args()
    started = time.time()
    if args.finish:
        finish(started)
    else:
        with no_statsbomb_reads():
            done = predict(args.device, args.max_seconds)
        log("predictions complete" if done else "partial")
    log(f"wall {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
