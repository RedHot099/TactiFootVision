"""Readability curves, linear probes, xG event study and controls of the direction probe (xg-credibility 09).

    CUDA_VISIBLE_DEVICES="" ./run.sh python scripts/report_readability.py

CPU only. Reads only the per-shot prediction table of ticket 08 (results/xgc_offset_scores: predictions.parquet,
embeddings_<encoder>.npy, complete.json; every file hash is verified first) and the index it references, plus the
Labels-v2 source annotations for the rebound-chain stratum (ticket 03 chains). Implements the approved protocol
research/xgc_direction_probe_protocol.md (hash asserted) through `xgpoc.probe.readability`:

  - population: non-penalty shots without a goal, complete cases (stop when > 5% excluded);
  - readings F, Gp, G0p (fixed-in-time linear probes, C per encoder by grouped CV), G, X, X0 (table logits);
  - D(o), the PRE trend P(o) and D - P for every reading, overall and in the strata (rebound chain, official test
    split, no cut), and the on-target AUC by offset for F, G, Gp, G0p; 2,000 match draws shared by everything;
  - xG event study: D - P for X and X0 in logits;
  - controls: placebo (full gate), negative control with labels shuffled within matches (full gate, F pipeline
    retrained with the selected C), and the first positive-control condition (lower bound of Delta > 0) for F, G, X.

Not here (ticket 10): the estimator S, U, t50, the second positive-control condition (half-width of the U interval),
the known-answer test, the verdict and the attribution sensitivity. Ticket 10 reads the work files below.

Writes the result folder results_reports/readability_2026-10 (CSVs, figures, summary.json, complete.json with
source, input and protocol hashes) and the work folder results/xgc_readability (git-ignored):
  logits.parquet   one row per complete-case non-penalty shot (goals included) and offset: shot_id, game, half,
                   position_ms, split, offset_ms, goal, in_population, y (1 = on target), y_perm (negative-control
                   label, population only, else -1), rebound_chain, no_cut, fold (-1 test), then one float64 column
                   per reading F, G, X, X0, Gp, G0p, F_neg (NaN where the reading does not exist: Gp/G0p on test shots,
                   F_neg outside the population). Probes are trained on the population; goal shots are scored by the
                   probe of their match's fold (test: mean of the five), for ticket 10's attribution rows.
  curves.npz       keys (M,) bootstrap match keys; counts (2000, M) draw counts; offsets (30,); for every reading r
                   and stratum s: D/<r>/<s> (30,) point, Dd/<r>/<s> (2000, 30) draws (NaN = skipped draw), and the
                   per-match sums on/<r>/<s>, off/<r>/<s> (M, 30), n_on/<r>/<s>, n_off/<r>/<s> (M,), so that any
                   statistic of D (P, Delta, S, U, t50, paired differences) can be recomputed on the same draws.
  probe_<name>.npz cached probe fits (logits for every C, cv losses, iterations, identity digest).
"""

import argparse
import inspect
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from report_complexity import GRID as GRID_COLOR
from report_complexity import INK, INK2, MUTED, SURFACE, _axes_style
from xgpoc.bundle import canonical_digest, code_version, sha256_file
from xgpoc.eval.attribution import attribute, chain_shot_keys, source_events
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.probe import readability as rd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
SCORES = EXP / "results" / "xgc_offset_scores"
WORK = EXP / "results" / "xgc_readability"
OUT = EXP / "results_reports" / "readability_2026-10"
PROTOCOL = EXP / "research" / "xgc_direction_probe_protocol.md"
FOLDS = EXP / "cache" / "folds_v12_home5.csv"
SCORE_FILES = ("predictions.parquet", "embeddings_g_combo.npy", "embeddings_g_combo_no_head.npy",
               "embeddings_dinov2_frozen.npy")
SOURCES = ("scripts/report_readability.py", "xgpoc/probe/readability.py", "xgpoc/probe/table.py",
           "xgpoc/data/offset_frames.py", "xgpoc/eval/attribution.py", "scripts/report_complexity.py", "xgpoc/bundle.py")
ALL_READINGS = ("F", "G", "X", "X0", "Gp", "G0p")
PANEL_ORDER = ("F", "G", "X", "Gp", "G0p", "X0")       # 2 x 3 small multiples: frozen / head / xG over the probes
SERIES = {"F": "#2a78d6", "G": "#eb6834", "Gp": "#1baf7a", "G0p": "#eda100", "X": "#2a78d6", "X0": "#eb6834"}
STRATUM_COLOR = {"all": "#2a78d6", "rebound_chain": "#eb6834", "test_split": "#1baf7a", "no_cut": "#eda100"}
LABEL = {"F": "F: sonda na DINOv2-L (zamrożony)", "G": "G: głowa celności G-combo", "X": "X: logit xG G-combo",
         "X0": "X0: logit xG kontroli bez głowy", "Gp": "Gp: sonda na embeddingu G-combo",
         "G0p": "G0p: sonda na embeddingu kontroli", "F_neg": "F, etykiety permutowane w meczu"}
STRATUM_LABEL = {"all": "cała populacja", "rebound_chain": "dobitki", "test_split": "podział test",
                 "no_cut": "bez cięć"}


# ---------------------------------------------------------------- inputs

def verify_inputs(scores: Path, index_path: Path, folds_path: Path, protocol: Path) -> dict[str, str]:
    """Protocol hash, ticket-08 file hashes and identity (index, folds, protocol). Returns the input hashes."""
    if sha256_file(protocol) != rd.PROTOCOL_SHA256:
        raise SystemExit(f"{protocol} is not the approved protocol")
    record = json.loads((scores / "complete.json").read_text())
    for name in SCORE_FILES:
        if sha256_file(scores / name) != record["files"].get(name):
            raise SystemExit(f"{scores / name} does not match its completion record")
    identity = record["identity"]
    checks = {"protocol": (identity["protocol"]["sha256"], rd.PROTOCOL_SHA256),
              "index": (identity["index_sha256"], sha256_file(index_path)),
              "folds": (identity["folds_sha256"], sha256_file(folds_path))}
    for what, (recorded, actual) in checks.items():
        if recorded != actual:
            raise SystemExit(f"the prediction table was built on another {what} ({recorded} != {actual})")
    return {"complete.json": sha256_file(scores / "complete.json"), **record["files"],
            "index": checks["index"][1], "folds": checks["folds"][1]}


def protocol_approval(scores: Path) -> dict:
    """The approval record the prediction table was built under (ticket 06 Comments, carried through 07 and 08)."""
    return json.loads((scores / "complete.json").read_text())["identity"]["protocol"]["approval"]


def rebound_chain(shots: pd.DataFrame, labels_root: Path) -> np.ndarray:
    """Shots of a ticket-03 rebound chain of two or more shots (GLOSSARY definition, from Labels-v2)."""
    keys = chain_shot_keys(attribute(source_events(shots.game.unique(), labels_root)))
    return np.array([k in keys for k in zip(shots.game, shots.half, shots.position_ms, strict=True)])


def probe_code_digest(nested: bool) -> str:
    """Cache key of a probe fit: the source of the functions it runs and the protocol constants it reads. Edits
    elsewhere keep the cache."""
    functions = [rd.fit_probe, rd.probe_readout, rd.outer_folds, rd.embedding_loader, rd.permute_within_match,
                 rd.probe_shots, rd.pivot_offsets] + ([rd.nested_probe_readout, rd.inner_folds] if nested else [])
    constants = {"sklearn": sklearn.__version__, "on_target": rd.ON_TARGET, "eps_ms": rd.EPS_MS,
                 "frame_statuses": rd.FRAME_STATUSES_WITH_FRAME, "grid": rd.GRID, "probe_train": rd.PROBE_TRAIN, "c_grid": rd.C_GRID, "max_iter": rd.MAX_ITER,
                 "n_inner": rd.N_INNER, "neg_seed": rd.NEG_SEED, "time_tolerance_ms": rd.TIME_TOLERANCE_MS}
    return canonical_digest([[inspect.getsource(f) for f in functions], constants])


def cached_probe(path: Path, identity: dict, fit) -> rd.ProbeResult:
    digest = canonical_digest(identity)
    if path.exists():
        with np.load(path) as f:
            if str(f["identity"]) == digest:
                return rd.ProbeResult(f["logits"], f["cv_loss"], f["n_iter"], tuple(f["c_grid"].tolist()))
    res = fit()
    np.savez(path, logits=res.logits, cv_loss=res.cv_loss, n_iter=res.n_iter, c_grid=np.array(res.c_grid),
             identity=np.array(digest))
    return res


def selection_rows(name: str, res: rd.ProbeResult) -> list[dict]:
    chosen = res.selected()
    return [{"reading": name, "C": c, "cv_logloss": float(res.cv_loss[:, i].mean()),
             "cv_logloss_min_fold": float(res.cv_loss[:, i].min()), "cv_logloss_max_fold": float(res.cv_loss[:, i].max()),
             "folds": len(res.cv_loss), "max_iter_used": int(res.n_iter[:, i].max()),
             "converged": bool((res.n_iter[:, i] < rd.MAX_ITER).all()), "selected": i == chosen}
            for i, c in enumerate(res.c_grid)]


# ---------------------------------------------------------------- figures

def _offset_axis(ax) -> None:
    ax.axvline(0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks([-1600, -1200, -800, -400, 0, 400, 800])
    _axes_style(ax)


def plot_curves(curves: pd.DataFrame, out: Path) -> None:
    """Small multiples: D(o) with its interval and the PRE trend per reading, main population."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURFACE)
    for ax, r in zip(axes.ravel(), PANEL_ORDER, strict=True):
        c = curves[(curves.reading == r) & (curves.stratum == "all")]
        ax.axvspan(min(rd.PRE), max(rd.PRE), color=GRID_COLOR, alpha=0.5, lw=0, zorder=0)
        ax.axvspan(min(rd.POST), max(rd.POST), color=GRID_COLOR, alpha=0.5, lw=0, zorder=0)
        ax.fill_between(c.offset_ms, c.D_lo, c.D_hi, color=SERIES[r], alpha=0.15, lw=0)
        ax.plot(c.offset_ms, c.P, "--", color=INK2, lw=1.2, label="trend PRE")
        ax.plot(c.offset_ms, c.D, color=SERIES[r], lw=2, marker="o", ms=3, label="D(o)")
        ax.set_title(LABEL[r], fontsize=9.5, loc="left", color=INK)
        _offset_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    for ax in axes[1]:
        ax.set_xlabel("offset względem kotwicy [ms]")
    for ax in axes[:, 0]:
        ax.set_ylabel("D: różnica centrowanych logitów")
    fig.suptitle("Czytelność celności po offsetach (strzały bez gola, przedział 95% z 2 000 losowań meczów; "
                 "szare pasma: okna PRE i POST)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_auc(auc: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.axhline(0.5, color=MUTED, lw=1)
    ends = {}
    for r in rd.AUC_READINGS:
        c = auc[(auc.reading == r) & (auc.stratum == "all")]
        ax.fill_between(c.offset_ms, c.auc_lo, c.auc_hi, color=SERIES[r], alpha=0.12, lw=0)
        ax.plot(c.offset_ms, c.auc, color=SERIES[r], lw=2, marker="o", ms=3, label=LABEL[r])
        ends[r] = c.auc.iloc[-1]
    placed = -np.inf
    for r, end in sorted(ends.items(), key=lambda kv: kv[1]):      # direct labels, at least 0.007 apart
        placed = max(end, placed + 0.007)
        ax.annotate(r, (max(rd.GRID), placed), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK2)
    _offset_axis(ax)
    ax.set_xlim(-1700, 1150)
    ax.set_xlabel("offset względem kotwicy [ms]")
    ax.set_ylabel("AUC celności (logity bez centrowania)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    ax.set_title("AUC celności po offsetach, strzały bez gola (opisowo; Gp i G0p bez strzałów test)",
                 fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_event_study(curves: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.axhline(0, color=MUTED, lw=1)
    for r in rd.EVENT_READINGS:
        c = curves[(curves.reading == r) & (curves.stratum == "all")]
        ax.fill_between(c.offset_ms, c.D_minus_P_lo, c.D_minus_P_hi, color=SERIES[r], alpha=0.15, lw=0)
        ax.plot(c.offset_ms, c.D_minus_P, color=SERIES[r], lw=2, marker="o", ms=3, label=LABEL[r])
    _offset_axis(ax)
    ax.set_xlabel("offset względem kotwicy [ms]")
    ax.set_ylabel("D(o) − P(o) [logit xG]")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    ax.set_title("Studium zdarzeń xG: luka celne − niecelne ponad trendem z okna PRE (przedział 95%)",
                 fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_strata(curves: pd.DataFrame, out: Path) -> None:
    """D - P per stratum for every reading (small multiples, one stratum colour each)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURFACE)
    for ax, r in zip(axes.ravel(), PANEL_ORDER, strict=True):
        ax.axhline(0, color=MUTED, lw=1)
        for s in rd.STRATA:
            c = curves[(curves.reading == r) & (curves.stratum == s)]
            if c.empty or c.D.isna().all():
                continue
            ax.fill_between(c.offset_ms, c.D_minus_P_lo, c.D_minus_P_hi, color=STRATUM_COLOR[s], alpha=0.08, lw=0)
            ax.plot(c.offset_ms, c.D_minus_P, color=STRATUM_COLOR[s], lw=1.8, label=STRATUM_LABEL[s])
        ax.set_title(LABEL[r], fontsize=9.5, loc="left", color=INK)
        _offset_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    for ax in axes[1]:
        ax.set_xlabel("offset względem kotwicy [ms]")
    for ax in axes[:, 0]:
        ax.set_ylabel("D(o) − P(o)")
    fig.suptitle("Warstwy: D − P po offsetach z przedziałami 95% (Gp i G0p nie mają strzałów test)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


FIGURES = {"readability_curves.png": plot_curves, "auc_by_offset.png": plot_auc, "event_study_xg.png": plot_event_study,
           "strata.png": plot_strata}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--folds", type=Path, default=FOLDS)
    ap.add_argument("--labels-root", type=Path, default=LABELS_ROOT)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    for d in (a.work, a.out):
        d.mkdir(parents=True, exist_ok=True)

    inputs = verify_inputs(a.scores, a.index, a.folds, PROTOCOL)
    table = pd.read_parquet(a.scores / "predictions.parquet")
    index = pd.read_parquet(a.index)
    shots, exclusions = rd.probe_shots(table, index)
    exclusions.to_csv(a.out / "exclusions.csv", index=False)
    if set(shots.shot_id) != set(table.shot_id[table.complete_case]):
        raise SystemExit("the 20 ms complete-case rule disagrees with the table's complete_case column")
    if rd.stop_rule_triggered(exclusions):
        raise SystemExit("more than 5% of the population excluded: the protocol stops here (see exclusions.csv)")
    shots["rebound_chain"] = rebound_chain(shots, a.labels_root)
    folds = pd.read_csv(a.folds)
    shots["fold"] = rd.outer_folds(shots, folds)
    fold_matches = {int(k): sorted(g) for k, g in folds.groupby("fold").game}
    fold = shots.fold.to_numpy()
    y = shots.y.to_numpy().astype(int)
    pop = shots.in_population.to_numpy()
    print(f"shots {len(shots)} (population {pop.sum()}, matches {shots.game[pop].nunique()})")

    base_identity = {"inputs": inputs, "probe_code": probe_code_digest(nested=False)}
    logits = {r: rd.table_reading(table, shots, r) for r in ("G", "X", "X0")}
    probes, selection = {}, []
    refs = {e: rd.pivot_offsets(table, shots.shot_id, e, "embedding_ref").astype(np.int64)
            for e in ("dinov2_frozen", "g_combo", "g_combo_no_head")}

    emb = np.load(a.scores / "embeddings_dinov2_frozen.npy", mmap_mode="r")
    load_f = rd.embedding_loader(emb, refs["dinov2_frozen"])
    probes["F"] = cached_probe(a.work / "probe_F.npz", base_identity | {"reading": "F"},
                               lambda: rd.probe_readout(load_f, y, pop, fold))
    c_f = probes["F"].c_grid[probes["F"].selected()]
    y_perm = np.full(len(shots), -1)
    y_perm[pop] = rd.permute_within_match(shots[pop])
    pop_rows = np.flatnonzero(pop)

    def load_pop(rows: np.ndarray, j: int) -> np.ndarray:
        return load_f(pop_rows[rows], j)

    neg = cached_probe(a.work / "probe_F_neg.npz", base_identity | {"reading": "F_neg", "C": c_f},
                       lambda: rd.probe_readout(load_pop, y_perm[pop], np.ones(len(pop_rows), bool), fold[pop], (c_f,)))
    for name, encoder in (("Gp", "g_combo"), ("G0p", "g_combo_no_head")):
        trval = fold >= 0
        if (refs[encoder][trval] < 0).any():
            raise SystemExit(f"{encoder}: a train/valid complete-case shot lacks an embedding")
        emb = np.load(a.scores / f"embeddings_{encoder}.npy", mmap_mode="r")
        load_g = rd.embedding_loader(emb, refs[encoder])
        probes[name] = cached_probe(a.work / f"probe_{name}.npz",
                                    base_identity | {"reading": name, "probe_code": probe_code_digest(nested=True)},
                                    lambda load_g=load_g: rd.nested_probe_readout(load_g, shots, fold, fold_matches))
    for name, res in probes.items():
        if not (res.n_iter < rd.MAX_ITER).all(axis=0).any():
            raise SystemExit(f"{name}: the probe converges at no C (protocol: stop and report)")
        logits[name] = res.logits[res.selected()]
        selection += selection_rows(name, res)
    for name, res in (*probes.items(), ("F_neg", neg)):
        if (res.n_iter[:, res.selected()] >= rd.MAX_ITER).any():
            print(f"WARNING {name}: the selected C did not converge in every fold (probe_selection.csv)")
    selection += selection_rows("F_neg", neg)
    f_neg = np.full((len(shots), len(rd.GRID)), np.nan)
    f_neg[pop] = neg.logits[0]
    logits["F_neg"] = f_neg

    keys, counts = rd.match_draw_counts(shots.game[pop].to_numpy())
    readings = {r: (logits[r], y) for r in ALL_READINGS} | {"F_neg": (logits["F_neg"], np.where(pop, y_perm, 0))}
    only_all = {"F_neg": ("all",)}
    curves = rd.readability_curves(readings, shots, keys, counts, only_all)
    auc = rd.auc_table({r: readings[r] for r in (*rd.AUC_READINGS, "F_neg")}, shots, keys, counts, only_all)
    curve_tab = rd.curve_table(curves)
    jumps = rd.jump_table(curves)
    controls = rd.control_record(curves)
    masks = rd.strata_masks(shots)
    strata = pd.DataFrame([{"reading": r, "stratum": s, "shots": int(m.sum()), "matches": int(shots.game[m].nunique()),
                            "on_target": int(y[m].sum()), "off_target": int((1 - y[m]).sum()),
                            "skipped_draws": int(np.isnan(curves.draws[r, s]).any(1).sum())}
                           for (r, s) in curves.point
                           for m in [masks[s] & np.isfinite(readings[r][0]).all(1)]])

    # work files for ticket 10
    long = shots.loc[shots.index.repeat(len(rd.GRID)), ["shot_id", "game", "half", "position_ms", "split", "goal",
                                                         "in_population", "y", "rebound_chain", "no_cut", "fold"]]
    long.insert(5, "offset_ms", np.tile(rd.GRID, len(shots)))
    long.insert(9, "y_perm", np.repeat(y_perm, len(rd.GRID)))
    for r in (*ALL_READINGS, "F_neg"):
        long[r] = logits[r].reshape(-1)
    long.reset_index(drop=True).to_parquet(a.work / "logits.parquet", index=False)
    arrays = {"keys": keys, "counts": counts, "offsets": np.array(rd.GRID)}
    for (r, s), sums in curves.sums.items():
        arrays |= {f"D/{r}/{s}": curves.point[r, s], f"Dd/{r}/{s}": curves.draws[r, s], f"on/{r}/{s}": sums.on,
                   f"off/{r}/{s}": sums.off, f"n_on/{r}/{s}": sums.n_on, f"n_off/{r}/{s}": sums.n_off}
    np.savez_compressed(a.work / "curves.npz", **arrays)

    tables = {"curves.csv": curve_tab, "auc.csv": auc, "jumps.csv": jumps, "strata.csv": strata,
              "probe_selection.csv": pd.DataFrame(selection)}
    for f, t in tables.items():
        t.to_csv(a.out / f, index=False)
    gates = {"placebo_passed": controls["placebo"]["passed"], "negative_passed": controls["negative"]["passed"],
             "positive_delta_lower_positive": {r: v["delta_lower_positive"] for r, v in controls["positive_jump"].items()}}
    summary = {"development_result": True, "protocol_sha256": rd.PROTOCOL_SHA256, "n_boot": rd.N_BOOT,
               "bootstrap_seed": rd.BOOT_SEED, "negative_control_seed": rd.NEG_SEED,
               "population": {"shots": int(pop.sum()), "matches": int(len(keys)), "on_target": int(y[pop].sum()),
                              "excluded": int(exclusions.excluded.iloc[-1]),
                              "excluded_fraction": float(exclusions.fraction.iloc[-1])},
               "scored_goal_shots": int((~pop).sum()),
               "selected_C": {n: r.c_grid[r.selected()] for n, r in probes.items()} | {"F_neg": c_f},
               "selected_C_at_grid_edge": {n: r.selected() in (0, len(r.c_grid) - 1) for n, r in probes.items()},
               "xg_logit_column": "xg_logit_raw", "controls": controls, "gates": gates,
               "not_here": "U, S, t50, the U half-width condition of the positive control, the known-answer test, "
                           "the verdict and the attribution sensitivity are ticket 10"}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    for f, plot in FIGURES.items():
        plot(auc if plot is plot_auc else curve_tab, a.out / f)

    work_files = {f: sha256_file(a.work / f) for f in ("logits.parquet", "curves.npz")}
    outputs = {f: sha256_file(a.out / f) for f in sorted([*tables, "exclusions.csv", "summary.json", *FIGURES])}
    record = {"ticket": "xg-credibility 09", "protocol_sha256": rd.PROTOCOL_SHA256,
              "protocol_approval": protocol_approval(a.scores),
              "sources": {s: sha256_file(EXP / s) for s in SOURCES}, "commit": code_version(EXP),
              "inputs": inputs, "inputs_digest": canonical_digest(inputs),
              "labels_digest": canonical_digest({g: sha256_file(a.labels_root / g / "Labels-v2.json")
                                                 for g in sorted(shots.game.unique())}),
              "work_files": work_files, "files": outputs}
    (a.out / "complete.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
