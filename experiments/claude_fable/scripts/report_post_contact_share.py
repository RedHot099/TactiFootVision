"""Post-contact share, known-answer test and verdict of the direction probe (xg-credibility 10).

    CUDA_VISIBLE_DEVICES="" ./run.sh python scripts/report_post_contact_share.py

CPU only. Reads ticket 09's work files (results/xgc_readability: logits.parquet, curves.npz; schema in
scripts/report_readability.py) after checking them against ticket 09's completion record, re-verifies ticket 08's
prediction table (the known-answer test refits the F probe on its frozen embeddings), and implements the rest of the
approved protocol research/xgc_direction_probe_protocol.md (hash asserted) through `xgpoc.probe.share`:

  - S(o), U, t50 and Delta for every reading and stratum, on ticket 09's 2,000 match draws (the D curves are
    recomputed from logits.parquet and must equal curves.npz);
  - the gates on F: positive control (both conditions), known answer (+200 ms, F probe refitted with the selected C
    on the frames of the moved anchors; G and X descriptive on the logits at o + 200), placebo and negative control;
  - the verdict, rule by rule, with every input; paired differences U_G - U_F, U_X - U_F, U_Gp - U_G0p, U_X - U_X0
    and U_stratum - U_all;
  - attribution sensitivity: U_F, Delta_F, t50_F and U_X for the non-goal population under every ticket-03 rule and
    both bounds (probes stay those of the main population; only the scored set changes). The bounds give each chain
    goal to the chain shot with the lowest ("worst") or highest ("best") G-combo xG at the anchor (`xg_logit` at
    offset 0, i.e. the shot that makes G-combo's log-loss worst or best).

Writes results_reports/post_contact_share_2026-10 (CSVs, figures, summary.json, complete.json with source, input and
protocol hashes) and caches the refitted probes in results/xgc_post_contact (git-ignored).
"""

import argparse
import inspect
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from report_complexity import GRID as GRID_COLOR
from report_complexity import INK, INK2, MUTED, SURFACE, _axes_style
from report_readability import (
    ALL_READINGS,
    LABEL,
    SERIES,
    STRATUM_LABEL,
    cached_probe,
    probe_code_digest,
    verify_inputs,
)
from xgpoc.bundle import canonical_digest, code_version, sha256_file
from xgpoc.eval.attribution import (
    RULES,
    attribute,
    bound_labels,
    chain_shot_keys,
    relabel,
    source_events,
)
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.probe import readability as rd
from xgpoc.probe import share as sh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
SCORES = EXP / "results" / "xgc_offset_scores"
READABILITY_WORK = EXP / "results" / "xgc_readability"
READABILITY_OUT = EXP / "results_reports" / "readability_2026-10"
WORK = EXP / "results" / "xgc_post_contact"
OUT = EXP / "results_reports" / "post_contact_share_2026-10"
PROTOCOL = EXP / "research" / "xgc_direction_probe_protocol.md"
FOLDS = EXP / "cache" / "folds_v12_home5.csv"
SOURCES = ("scripts/report_post_contact_share.py", "xgpoc/probe/share.py", "xgpoc/probe/readability.py",
           "scripts/report_readability.py", "xgpoc/eval/attribution.py", "scripts/report_complexity.py",
           "xgpoc/bundle.py")
TICKET09_SOURCES = ("scripts/report_readability.py", "xgpoc/probe/readability.py", "xgpoc/eval/attribution.py")
READINGS = ALL_READINGS
GATE_READINGS = rd.POSITIVE_CONTROL_READINGS     # F, G, X: the readings the gates and rule 3 read
SHOT_COLUMNS = ["shot_id", "game", "half", "position_ms", "split", "goal", "in_population", "y", "y_perm",
                "rebound_chain", "no_cut", "fold"]
BOUNDS = ("worst", "best")
U_AXIS = (-0.8, 0.8)            # x range of the U overview; wider intervals are clipped and marked


# ---------------------------------------------------------------- inputs

def verify_ticket09(readability_out: Path, readability_work: Path, inputs: dict) -> dict:
    """Ticket 09's work files and sources must be the ones its completion record holds, built on the same ticket-08
    inputs. Returns the record."""
    record = json.loads((readability_out / "complete.json").read_text())
    if record["protocol_sha256"] != rd.PROTOCOL_SHA256:
        raise SystemExit("ticket 09 ran under another protocol")
    for name, digest in record["work_files"].items():
        if sha256_file(readability_work / name) != digest:
            raise SystemExit(f"{readability_work / name} does not match ticket 09's completion record")
    for name in TICKET09_SOURCES:
        if sha256_file(EXP / name) != record["sources"][name]:
            raise SystemExit(f"{name} changed after ticket 09's run: re-run scripts/report_readability.py first")
    if record["inputs"] != inputs:
        raise SystemExit("ticket 09 read other ticket-08 inputs than the ones on disk")
    return record


def load_logits(path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """One row per shot (sorted by shot_id, as written by ticket 09) and an (n, 30) matrix per reading."""
    long = pd.read_parquet(path)
    n = len(long) // len(rd.GRID)
    if len(long) != n * len(rd.GRID) or not (long.offset_ms.to_numpy().reshape(n, -1) == np.array(rd.GRID)).all():
        raise SystemExit("logits.parquet is not shot-major over the protocol grid")
    shots = long.iloc[::len(rd.GRID)][SHOT_COLUMNS].reset_index(drop=True)
    if not shots.shot_id.is_monotonic_increasing or shots.shot_id.duplicated().any():
        raise SystemExit("logits.parquet shots are not unique and sorted by shot_id")
    return shots, {r: long[r].to_numpy(float).reshape(n, -1) for r in (*READINGS, "F_neg")}


def check_curves(curves: rd.Curves, npz) -> None:
    """The D curves recomputed here equal ticket 09's (point and every draw)."""
    for (r, s), point in curves.point.items():
        np.testing.assert_allclose(point, npz[f"D/{r}/{s}"], rtol=0, atol=1e-12, err_msg=f"D/{r}/{s}")
        np.testing.assert_allclose(curves.draws[r, s], npz[f"Dd/{r}/{s}"], rtol=0, atol=1e-12, err_msg=f"Dd/{r}/{s}")


def ka_probe_digest() -> str:
    return canonical_digest([probe_code_digest(nested=False),
                             [inspect.getsource(f) for f in (sh.move_anchor, sh.ka_half)],
                             {"move_ms": sh.KA_MOVE_MS, "seed": sh.KA_SEED}])


# ---------------------------------------------------------------- analysis pieces

def summaries(curves: rd.Curves) -> dict[tuple[str, str], dict]:
    """Delta, U, t50 per reading and stratum; not for F_neg (Delta_neg ~ 0 makes its S meaningless; its gate reads
    Delta_neg and D_neg(0) only) nor for empty strata (Gp, G0p have no test shots)."""
    return {k: sh.reading_summary(p, curves.draws[k]) for k, p in curves.point.items()
            if k[0] in READINGS and np.isfinite(p).all()}


def share_rows(stats: dict[tuple[str, str], dict], curves: rd.Curves, shots: pd.DataFrame,
               readings: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    masks = rd.strata_masks(shots)
    rows = []
    for (r, s), v in stats.items():
        m = masks[s] & np.isfinite(readings[r][0]).all(1)
        pc = sh.positive_control(v)
        rows.append({"reading": r, "stratum": s, "shots": int(m.sum()), "matches": int(shots.game[m].nunique()),
                     **sh.flat("delta", v["delta"]), **sh.flat("U", v["u"]), "U_half_width": v["u_half_width"],
                     "positive_control_passed": pc["passed"], **sh.flat("t50", v["t50"]), "t50_label": v["t50"]["label"],
                     "skipped_draws": v["u"]["skipped_draws"],
                     "role": "gate_input" if s == "all" and r in GATE_READINGS else "descriptive"})
    return pd.DataFrame(rows)


def s_curve_rows(curves: rd.Curves) -> pd.DataFrame:
    frames = []
    for (r, s), point in curves.point.items():
        if r not in READINGS or not np.isfinite(point).all():
            continue
        lo, hi, skipped = rd.interval(sh.normalised(curves.draws[r, s]))
        frames.append(pd.DataFrame({"reading": r, "stratum": s, "offset_ms": rd.GRID, "S": sh.normalised(point),
                                    "S_lo": lo, "S_hi": hi, "skipped_draws": skipped}))
    return pd.concat(frames, ignore_index=True)


def u_pair(curves: rd.Curves, key: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    return sh.share(curves.point[key]), sh.share(curves.draws[key])


def difference_rows(curves: rd.Curves) -> pd.DataFrame:
    """Paired U differences: the rule-3 inputs, the head-free comparison and every stratum against the whole."""
    pairs = [("U_G - U_F", ("G", "all"), ("F", "all"), "rule_3_input"),
             ("U_X - U_F", ("X", "all"), ("F", "all"), "rule_3_input"),
             ("U_Gp - U_G0p", ("Gp", "all"), ("G0p", "all"), "descriptive"),
             ("U_X - U_X0", ("X", "all"), ("X0", "all"), "descriptive")]
    pairs += [(f"U_{r}[{s}] - U_{r}[all]", (r, s), (r, "all"), "descriptive")
              for r in READINGS for s in rd.STRATA[1:] if np.isfinite(curves.point.get((r, s), np.nan)).all()]
    return pd.DataFrame([{"difference": name, "role": role, **sh.paired_difference(u_pair(curves, a), u_pair(curves, b))}
                         for name, a, b, role in pairs])


def ka_rows(curves: rd.Curves, ka: rd.Curves) -> tuple[dict[str, dict], pd.DataFrame]:
    gates, rows = {}, []
    for r in GATE_READINGS:
        point = sh.known_answer(curves.point[r, "all"], ka.point[r, "all"])
        draws = sh.known_answer(curves.draws[r, "all"], ka.draws[r, "all"])
        g = sh.known_answer_gate(point, draws)
        gates[r] = g
        row = {"reading": r, "role": "gate" if r == "F" else "descriptive (logits at o + 200, no refit)"}
        for k in ("A", "A_pred", "anchor_change", "anchor_change_pred"):
            row |= sh.flat(k, g[k])
        row |= {"ratio": g["ratio"], "ratio_lo": g["ratio_interval"]["lo"], "ratio_hi": g["ratio_interval"]["hi"],
                "passed": g["passed"]}
        rows.append(row)
    return gates, pd.DataFrame(rows)


def sensitivity(shots: pd.DataFrame, logits: dict[str, np.ndarray], index: pd.DataFrame, labels_root: Path,
                g_xg0: pd.Series, keys: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    """U_F, Delta_F, t50_F and U_X of the non-goal complete-case population under every attribution rule and bound."""
    ix = index.drop_duplicates("shot_id")
    ix = ix[~ix.is_penalty.astype(bool)].reset_index(drop=True)
    attr = attribute(source_events(ix.game.unique(), labels_root))
    p = 1 / (1 + np.exp(-ix.shot_id.map(g_xg0).to_numpy(float)))
    chain = chain_shot_keys(attr)
    in_chain = np.array([k in chain for k in zip(ix.game, ix.half.astype(int), ix.position_ms.astype(int), strict=True)])
    if not np.isfinite(p[in_chain]).all():
        raise SystemExit("a rebound-chain shot has no G-combo xG at the anchor; the bounds are undefined")
    labels = {r: relabel(ix, attr, r) for r in RULES}
    labels |= {b: (bound_labels(ix, attr, p, b), np.ones(len(ix), bool)) for b in BOUNDS}
    if (labels["nearest"][0] != ix.goal.to_numpy(bool)).any():
        raise SystemExit("the nearest rule does not reproduce the index goal column")
    row_of = pd.Series(np.arange(len(ix)), index=ix.shot_id)
    at = row_of.reindex(shots.shot_id).to_numpy(float)
    if np.isnan(at).any():
        raise SystemExit("a probe shot is not a non-penalty index shot")
    at = at.astype(int)
    rows = []
    for name, (goal, keep) in labels.items():
        pop = ~goal[at] & keep[at]           # complete cases (every row of `shots`) without a goal, minus left-out shots
        outside = pop & ~np.isin(shots.game.to_numpy(), keys)
        pop &= ~outside
        alt = shots.assign(in_population=pop)
        y = alt.y.to_numpy().astype(int)
        cur = rd.readability_curves({r: (logits[r], y) for r in ("F", "X")}, alt, keys, counts,
                                    {"F": ("all",), "X": ("all",)})
        f = sh.reading_summary(cur.point["F", "all"], cur.draws["F", "all"])
        x = sh.reading_summary(cur.point["X", "all"], cur.draws["X", "all"])
        rows.append({"rule": name, "kind": "bound" if name in BOUNDS else "rule", "shots": int(pop.sum()),
                     "matches": int(alt.game[pop].nunique()), "on_target": int(y[pop].sum()),
                     "vs_main_added": int((pop & ~shots.in_population).sum()),
                     "vs_main_removed": int((~pop & shots.in_population).sum()), "outside_bootstrap_keys": int(outside.sum()),
                     **sh.flat("U_F", f["u"]), **sh.flat("delta_F", f["delta"]), **sh.flat("t50_F", f["t50"]),
                     "t50_F_label": f["t50"]["label"], **sh.flat("U_X", x["u"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def _offset_axis(ax, xlim=(-1700, 1100)) -> None:
    ax.axvline(0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.set_xticks([-1600, -1200, -800, -400, 0, 400, 800])
    ax.set_xlim(*xlim)
    _axes_style(ax)


def _windows(ax) -> None:
    for o in (rd.PRE, rd.POST):
        ax.axvspan(min(o), max(o), color=GRID_COLOR, alpha=0.5, lw=0, zorder=0)


def plot_share_curves(s_curves: pd.DataFrame, shares: pd.DataFrame, out: Path) -> None:
    """S(o) of the gate readings F, G, X with the 95% band; U marked at the anchor."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    for ax, r in zip(axes, GATE_READINGS, strict=True):
        c = s_curves[(s_curves.reading == r) & (s_curves.stratum == "all")]
        u = shares[(shares.reading == r) & (shares.stratum == "all")].iloc[0]
        _windows(ax)
        for level in (0, 1):
            ax.axhline(level, color=MUTED, lw=1)
        ax.axhline(sh.U_POST_CONTACT_LEAKAGE, color=INK2, lw=1, ls="--")
        ax.fill_between(c.offset_ms, c.S_lo, c.S_hi, color=SERIES[r], alpha=0.15, lw=0)
        ax.plot(c.offset_ms, c.S, color=SERIES[r], lw=2, marker="o", ms=3)
        ax.errorbar([0], [u.U], yerr=[[u.U - u.U_lo], [u.U_hi - u.U]], fmt="o", ms=8, color=INK, mfc=SURFACE, mew=2,
                    capsize=4, zorder=5)
        ax.text(0.03, 0.9, f"U = {u.U:.2f} [{u.U_lo:.2f}; {u.U_hi:.2f}]".replace(".", ","), transform=ax.transAxes,
                fontsize=9, color=INK)
        ax.set_title(LABEL[r], fontsize=9.5, loc="left", color=INK)
        _offset_axis(ax)
        ax.set_xlabel("offset względem kotwicy [ms]")
    axes[0].set_ylabel("S(o) = (D − P) / Δ")
    axes[0].annotate("próg 0,10", (-1650, sh.U_POST_CONTACT_LEAKAGE), xytext=(0, 3), textcoords="offset points", fontsize=8, color=INK2)
    lo = max(-1.0, float(s_curves[s_curves.reading.isin(GATE_READINGS) & (s_curves.stratum == "all")].S_lo.min()))
    axes[0].set_ylim(lo - 0.1, None)
    fig.suptitle("Znormalizowana czytelność S(o) i udział klatek po kontakcie U = S(0) (przedział 95% z 2 000 losowań "
                 "meczów; szare pasma: PRE i POST)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_head_free(curves: pd.DataFrame, s_curves: pd.DataFrame, out: Path) -> None:
    """G-combo against the head-free control on common axes: D and S for Gp / G0p and for X / X0."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), facecolor=SURFACE)
    for row, (a, b) in enumerate((("Gp", "G0p"), ("X", "X0"))):
        for col, (frame, value, ylabel) in enumerate(((curves, "D", "D(o)"), (s_curves, "S", "S(o)"))):
            ax = axes[row, col]
            _windows(ax)
            ax.axhline(0, color=MUTED, lw=1)
            colours = {a: SERIES["F"], b: SERIES["G"]}           # slot 1 = G-combo, slot 2 = head-free control
            for r in (a, b):
                c = frame[(frame.reading == r) & (frame.stratum == "all")]
                ax.fill_between(c.offset_ms, c[f"{value}_lo"], c[f"{value}_hi"], color=colours[r], alpha=0.12, lw=0)
                ax.plot(c.offset_ms, c[value], color=colours[r], lw=2, label=LABEL[r])
            _offset_axis(ax)
            ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("offset względem kotwicy [ms]")
            ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    fig.suptitle("G-combo a kontrola bez głowy celności: sondy na embeddingach (góra, tylko train i valid) "
                 "i logity xG (dół), przedział 95%", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_known_answer(s_curves: pd.DataFrame, s_ka: pd.DataFrame, gate: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    ax.axvspan(min(sh.KA), max(sh.KA), color=GRID_COLOR, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=1)
    for frame, colour, name in ((s_curves, SERIES["F"], "S_F: dane bez przesunięcia"),
                                (s_ka, SERIES["G"], "S_KA: połowa strzałów przesunięta o +200 ms")):
        c = frame[(frame.reading == "F") & (frame.stratum == "all") & ~frame.offset_ms.isin(sh.NO_PARTNER)]
        ax.fill_between(c.offset_ms, c.S_lo, c.S_hi, color=colour, alpha=0.12, lw=0)
        ax.plot(c.offset_ms, c.S, color=colour, lw=2, marker="o", ms=3, label=name)
    a, ap = gate["A"], gate["A_pred"]
    text = (f"A = {a['value']:.3f} [{a['lo']:.3f}; {a['hi']:.3f}]\nA_pred = {ap['value']:.3f}\n"
            f"A / A_pred = {gate['ratio']:.2f} (wymagane 0,5–1,5)").replace(".", ",")
    ax.text(0.02, 0.62, text, transform=ax.transAxes, fontsize=9, color=INK, va="top")
    _offset_axis(ax)
    ax.set_xlabel("offset względem kotwicy [ms] (bez offsetów +240…+360 i +1000, które nie mają partnera)")
    ax.set_ylabel("S(o), odczyt F")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    ax.set_title("Test znanej odpowiedzi: szare pasmo to okno KA (−400…+200 ms)", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_u_overview(shares: pd.DataFrame, sens: pd.DataFrame, out: Path) -> None:
    """U with its interval per reading (main population), per stratum (F, X) and per attribution rule (F)."""
    groups = [("Odczyty, cała populacja", [(LABEL[r].split(":")[0], shares[(shares.reading == r) & (shares.stratum == "all")])
                                          for r in READINGS]),
              ("Warstwy, odczyt F", [(STRATUM_LABEL[s], shares[(shares.reading == "F") & (shares.stratum == s)])
                                     for s in rd.STRATA]),
              ("Przypisanie goli, odczyt F", [(n, sens[sens.rule == n].rename(columns={"U_F": "U", "U_F_lo": "U_lo",
                                                                                     "U_F_hi": "U_hi"}))
                                              for n in sens.rule])]
    lim = U_AXIS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE, sharex=True)
    for ax, (title, items) in zip(axes, groups, strict=True):
        items = [(n, f.iloc[0]) for n, f in items if len(f)]
        ypos = np.arange(len(items))[::-1]
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(sh.U_POST_CONTACT_LEAKAGE, color=INK2, lw=1, ls="--")
        for yv, (_, r) in zip(ypos, items, strict=True):
            lo, hi = max(r.U_lo, lim[0]), min(r.U_hi, lim[1])            # clipped ends get an arrow head
            ax.plot([lo, hi], [yv, yv], color=SERIES["F"], lw=2, solid_capstyle="round")
            for end, edge, marker in ((r.U_lo, lim[0], "<"), (r.U_hi, lim[1], ">")):
                if (end < edge) if marker == "<" else (end > edge):
                    ax.plot([edge], [yv], marker, ms=7, color=SERIES["F"])
            ax.plot([np.clip(r.U, *lim)], [yv], "o", ms=8, color=SERIES["F"], mec=SURFACE, mew=2)
            text = f"{r.U:.2f} [{r.U_lo:.2f}; {r.U_hi:.2f}]".replace(".", ",")
            ax.annotate(text, (lim[1], yv), xytext=(-2, 7), textcoords="offset points", ha="right", fontsize=7.5,
                        color=INK2)
        ax.set_yticks(ypos, [n for n, _ in items])
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        _axes_style(ax)
        ax.set_xlabel("U (przedział 95%)")
    fig.suptitle("Udział klatek kotwicy po kontakcie U (przedział 95%; strzałka: przedział poza osią); linia "
                 "przerywana: próg 0,10 reguły przecieku", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:  # noqa: PLR0915 (one linear pipeline, as report_readability.main)
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--folds", type=Path, default=FOLDS)
    ap.add_argument("--labels-root", type=Path, default=LABELS_ROOT)
    ap.add_argument("--readability-work", type=Path, default=READABILITY_WORK)
    ap.add_argument("--readability-out", type=Path, default=READABILITY_OUT)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    for d in (a.work, a.out):
        d.mkdir(parents=True, exist_ok=True)

    inputs = verify_inputs(a.scores, a.index, a.folds, PROTOCOL)          # protocol hash + ticket-08 files
    record09 = verify_ticket09(a.readability_out, a.readability_work, inputs)
    summary09 = json.loads((a.readability_out / "summary.json").read_text())
    shots, logits = load_logits(a.readability_work / "logits.parquet")
    pop = shots.in_population.to_numpy(bool)
    y = shots.y.to_numpy().astype(int)
    y_perm = shots.y_perm.to_numpy().astype(int)

    # the curves of ticket 09, recomputed on its draws
    with np.load(a.readability_work / "curves.npz", allow_pickle=True) as npz:     # keys: object array; file hash-checked
        keys, counts = npz["keys"], npz["counts"]
        k2, c2 = rd.match_draw_counts(shots.game[pop].to_numpy())
        if not (np.array_equal(keys, k2) and np.array_equal(counts, c2)):
            raise SystemExit("curves.npz does not hold the protocol's draws of this population")
        readings = {r: (logits[r], y) for r in READINGS} | {"F_neg": (logits["F_neg"], np.where(pop, y_perm, 0))}
        curves = rd.readability_curves(readings, shots, keys, counts, {"F_neg": ("all",)})
        check_curves(curves, npz)
    print(f"population {pop.sum()} shots, {len(keys)} matches; ticket-09 curves reproduced")

    # known-answer test: F refitted on the frames of the moved anchors with the selected C; G and X read at o + 200
    c_f = summary09["selected_C"]["F"]
    pop_shots = shots[pop].reset_index(drop=True)
    half = sh.ka_half(len(pop_shots))
    table = pd.read_parquet(a.scores / "predictions.parquet",
                            columns=["shot_id", "offset_ms", "encoder", "embedding_ref", "xg_logit"])
    refs = rd.pivot_offsets(table, pop_shots.shot_id, "dinov2_frozen", "embedding_ref").astype(np.int64)
    emb = np.load(a.scores / "embeddings_dinov2_frozen.npy", mmap_mode="r")
    fold_pop = pop_shots.fold.to_numpy()
    y_pop = pop_shots.y.to_numpy().astype(int)
    identity = {"inputs": inputs, "probe_code": ka_probe_digest(), "C": c_f}
    every = np.ones(len(pop_shots), bool)
    refit = cached_probe(a.work / "probe_F_refit.npz", identity | {"reading": "F_refit"},
                         lambda: rd.probe_readout(rd.embedding_loader(emb, refs), y_pop, every, fold_pop, (c_f,)))
    refit_gap = float(np.abs(refit.logits[0] - logits["F"][pop]).max())
    if refit_gap > 1e-6:
        raise SystemExit(f"the population-only F pipeline does not reproduce ticket 09's F logits ({refit_gap:.2e})")
    ka_fit = cached_probe(a.work / "probe_F_ka.npz", identity | {"reading": "F_ka"},
                          lambda: rd.probe_readout(rd.embedding_loader(emb, sh.move_anchor(refs, half)), y_pop, every,
                                                   fold_pop, (c_f,)))
    f_ka = np.full_like(logits["F"], np.nan)
    f_ka[pop] = ka_fit.logits[0]
    moved_all = np.zeros(len(shots), bool)
    moved_all[np.flatnonzero(pop)[half]] = True
    ka_readings = {"F": (f_ka, y), "G": (sh.move_anchor(logits["G"], moved_all), y),
                   "X": (sh.move_anchor(logits["X"], moved_all), y)}
    ka_curves = rd.readability_curves(ka_readings, shots, keys, counts, dict.fromkeys(ka_readings, ("all",)))
    ka_gates, ka_tab = ka_rows(curves, ka_curves)

    # estimator, gates and verdict
    stats = summaries(curves)
    shares = share_rows(stats, curves, shots, readings)
    s_tab = s_curve_rows(curves)
    diffs = difference_rows(curves)
    controls = rd.control_record(curves)
    for gate in ("placebo", "negative"):
        if controls[gate]["passed"] != summary09["controls"][gate]["passed"]:
            raise SystemExit(f"the {gate} gate differs from ticket 09's record")
    positive = {r: sh.positive_control(stats[r, "all"]) for r in GATE_READINGS}
    f_gates = {"positive_control": positive["F"]["passed"], "known_answer": ka_gates["F"]["passed"],
               "placebo": controls["placebo"]["passed"], "negative_control": controls["negative"]["passed"]}
    diff_of = diffs.set_index("difference")
    g_rule = {r: {"positive_passed": positive[r]["passed"], "u": stats[r, "all"]["u"],
                  "diff": diff_of.loc[f"U_{r} - U_F", ["value", "lo", "hi", "skipped_draws"]].to_dict()}
              for r in sh.G_RULE_READINGS}
    result = sh.verdict(f_gates, stats["F", "all"]["u"], g_rule)
    print(json.dumps({"verdict": result["verdict"], "rule": result["rule"], "failed_gates": result["failed_gates"]}))

    # attribution sensitivity
    index = pd.read_parquet(a.index)
    g_xg0 = table[(table.encoder == "g_combo") & (table.offset_ms == 0)].set_index("shot_id").xg_logit
    sens = sensitivity(shots, logits, index, a.labels_root, g_xg0, keys, counts)
    main_row = sens[sens.rule == "nearest"].iloc[0]
    if not np.isclose(main_row.U_F, stats["F", "all"]["u"]["value"], rtol=0, atol=1e-12) or main_row.vs_main_added \
            or main_row.vs_main_removed:
        raise SystemExit("the nearest-rule sensitivity row does not reproduce the main population")

    # outputs
    # offsets without a +200 partner hold a half-moved mixture and enter no statistic; flag them
    s_ka = s_curve_rows(ka_curves).assign(partnered=lambda d: ~d.offset_ms.isin(sh.NO_PARTNER))
    tables = {"shares.csv": shares, "s_curves.csv": s_tab, "s_curves_known_answer.csv": s_ka,
              "differences.csv": diffs, "known_answer.csv": ka_tab, "sensitivity.csv": sens}
    for f, t in tables.items():
        t.to_csv(a.out / f, index=False)
    curves09 = pd.read_csv(a.readability_out / "curves.csv")
    figures = {"share_curves.png": lambda p: plot_share_curves(s_tab, shares, p),
               "head_free_comparison.png": lambda p: plot_head_free(curves09, s_tab, p),
               "known_answer.png": lambda p: plot_known_answer(s_tab, s_ka, ka_gates["F"], p),
               "u_overview.png": lambda p: plot_u_overview(shares, sens, p)}
    for f, plot in figures.items():
        plot(a.out / f)

    f_main = stats["F", "all"]
    rule_inputs = {"pi_-800": controls["placebo"]["pi_-800"], "pi_-600": controls["placebo"]["pi_-600"],
                   "delta_neg": controls["negative"]["delta_neg"], "d_neg_0": controls["negative"]["d_neg_0"],
                   "delta_F": f_main["delta"], "U_F": f_main["u"], "U_F_half_width": f_main["u_half_width"],
                   "A": ka_gates["F"]["A"], "A_pred": ka_gates["F"]["A_pred"], "A_over_A_pred": ka_gates["F"]["ratio"],
                   "U_G": stats["G", "all"]["u"], "U_X": stats["X", "all"]["u"],
                   "U_G_minus_U_F": g_rule["G"]["diff"], "U_X_minus_U_F": g_rule["X"]["diff"],
                   "plateau_slope_F": float(rd.plateau_slope(curves.point["F", "all"]))}
    summary = {"development_result": True, "protocol_sha256": rd.PROTOCOL_SHA256, "n_boot": rd.N_BOOT,
               "bootstrap_seed": rd.BOOT_SEED, "ka_seed": sh.KA_SEED, "selected_C_F": c_f,
               "population": summary09["population"], "ka_moved_shots": int(half.sum()),
               "ka_offsets_without_partner": list(sh.NO_PARTNER), "f_refit_max_abs_logit_gap": refit_gap,
               "verdict": result, "f_gates": f_gates, "positive_control": positive,
               "known_answer": ka_gates, "placebo": controls["placebo"], "negative_control": controls["negative"],
               "rule_inputs": rule_inputs, "readings": {f"{r}/{s}": v for (r, s), v in stats.items()}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    outputs = {f: sha256_file(a.out / f) for f in sorted([*tables, "summary.json", *figures])}
    work_files = {f: sha256_file(a.work / f) for f in ("probe_F_refit.npz", "probe_F_ka.npz")}
    record = {"ticket": "xg-credibility 10", "protocol_sha256": rd.PROTOCOL_SHA256,
              "protocol_approval": record09["protocol_approval"],
              "sources": {s: sha256_file(EXP / s) for s in SOURCES}, "commit": code_version(EXP),
              "inputs": inputs, "inputs_digest": canonical_digest(inputs),
              "ticket09": {"complete.json": sha256_file(a.readability_out / "complete.json"),
                           "summary.json": sha256_file(a.readability_out / "summary.json"),
                           "curves.csv": sha256_file(a.readability_out / "curves.csv"),
                           "work_files": record09["work_files"]},
              "labels_digest": canonical_digest({g: sha256_file(a.labels_root / g / "Labels-v2.json")
                                                 for g in sorted(index.game.unique())}), "work_files": work_files, "files": outputs}
    (a.out / "complete.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"f_gates": f_gates, "U_F": f_main["u"], "A": ka_gates["F"]["A"], "ratio": ka_gates["F"]["ratio"]},
                     indent=2))


if __name__ == "__main__":
    main()
