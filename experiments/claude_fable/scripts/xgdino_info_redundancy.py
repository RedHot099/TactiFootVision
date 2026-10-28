"""Read-out of the xg-dino-info redundancy and subgroup run (ticket 13) and its replay checks. CPU only.

The run itself is the pinned protocol script, unchanged (it refuses unless the protocol sha256 39a547b5... is pinned and
the code, inputs, controls and dry run match):

    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --real --out results/xgdino_info_redundancy/real
    ./run.sh --no-sync python scripts/xgdino_info_redundancy.py

This script refuses unless `real/complete.json` matches its files and the storage probabilities and names the pinned
protocol. Then:
    replay   refits one arm pair (B+Xc, C+Xc: the primary test) from the same inputs, folds and seeds and compares it with
             the stored out-of-fold probabilities (max |Δ|); re-derives the whole redundancy read-out (every block, the
             estimands and the leak table) from the stored probabilities and compares it leaf by leaf with summary.json;
             the run's refitted B, C, BQ, CQ against the frozen results/xgdino_ablation/real (from summary.json).
    tables   verdicts.csv (per block: LL_{B+X} - LL_B, R_X, R_X / G unclipped, raw and gated verdict, C - (B+X) with
             C1's wording, C+X - C), estimands.csv (no_v3, unseen, unseen standardised), leak.csv (C1's table),
             subgroups.csv (both sides, contrast, Holm) and tables.md.
complete.json is written last (hashes of every file in the folder, the run's included).

Pinned-code issue found in the real run (no decision rule involved): `info_protocol.subgroups` writes a side's goal count
under "goals" and then overwrites it with the mean d among goals (same key). subgroups.csv takes the counts from the
protocol's own `subgroup_keys` here (`goal_count`) and the stored "goals" value as `mean_d_goals`.
"""

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import info_protocol as ip

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xgdino_info_protocol as run  # noqa: E402

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "results/xgdino_info_redundancy"
REPLAY_ARMS = ("B+Xc", "C+Xc")
TOL = 1e-6


# ------------------------------------------------------------------ checks
def checked_run(folder: Path) -> dict:
    """The run's completion record after checking its files, its storage probabilities and the pin."""
    record = json.loads((folder / "complete.json").read_text())
    if record["mode"] != "real" or record["protocol_sha256"] != run.PROTOCOL_SHA256 or record["pinned"] != run.PROTOCOL_SHA256:
        raise SystemExit(f"{folder} is not a real run under the pinned protocol")
    for name, sha in record["files"].items():
        if sha256_file(folder / name) != sha:
            raise SystemExit(f"{folder / name} does not match the run's completion record")
    storage = Path(record["storage"]["root"]) / "probabilities.csv"
    if sha256_file(storage) != record["storage"]["probabilities.csv"]:
        raise SystemExit(f"{storage} does not match the run's completion record")
    return record


def leaves(a, b, path: str = "") -> tuple[int, float, list[str]]:
    """(leaves compared, max |Δ| over numeric leaves, non-numeric mismatches) of two JSON trees."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return 0, 0.0, [f"{path}: keys differ"]
        parts = [leaves(a[k], b[k], f"{path}/{k}") for k in a]
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        parts = [leaves(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b, strict=True))]
    elif isinstance(a, bool | str) or isinstance(b, bool | str) or a is None or b is None:
        return 1, 0.0, [] if a == b else [f"{path}: {a!r} != {b!r}"]
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return 1, 0.0, []
    else:
        return 1, abs(float(a) - float(b)), []
    return (sum(p[0] for p in parts), max((p[1] for p in parts), default=0.0), [m for p in parts for m in p[2]])


def replay(inputs: ip.Inputs, y: np.ndarray, stored: pd.DataFrame, summary: dict, protocol: dict) -> dict:
    """The replay checks of the module docstring (StatsBomb guarded)."""
    table, columns = run.population(inputs)
    if not stored.index.equals(table.index):
        raise SystemExit("the stored probabilities are not the population in its order")
    with no_statsbomb_reads():
        prepared = ip.prepare(table, columns, inputs.ablation.subsets, blocks=(ip.PRIMARY,))
        p, chosen = ip.fit_arms(prepared, y, REPLAY_ARMS)
        weights = prepared.block.weights
        p_stored = {c.removeprefix("p_"): stored[c].to_numpy(float) for c in stored.columns}
        gate = protocol["frozen"]["boundary_gate"]["available"]
        again = ip.redundancy(y, p_stored, weights, prepared.masks, not_explain_available=gate)
    arm = {a: {"max_abs_delta_p": float(np.abs(p[a] - p_stored[a]).max()),
               "chosen_c_equal": chosen[a] == summary["chosen_c"][a]} for a in REPLAY_ARMS}
    n, worst, mismatches = leaves(summary["redundancy"], json.loads(json.dumps(again, default=float)))
    return {"arm_refit": arm,
            "read_out_from_stored_probabilities": {"leaves": n, "max_abs_delta": worst, "mismatches": mismatches},
            "refit_vs_frozen_xgdino": {k: v for k, v in summary["replay"].items() if k.startswith("max_abs")},
            "tolerance": TOL,
            "note": "stored probabilities are written with 9 significant digits; differences of ~1e-9 in p are rounding",
            "passed": bool(all(v["max_abs_delta_p"] <= TOL and v["chosen_c_equal"] for v in arm.values())
                           and worst <= TOL and not mismatches
                           and all(v <= TOL for k, v in summary["replay"].items() if k.startswith("max_abs")))}


def goal_counts(inputs: ip.Inputs, y: np.ndarray, protocol: dict) -> pd.DataFrame:
    """Shots and goals per subgroup key and side from the protocol's own keys (the stored counts are overwritten)."""
    table, _ = run.population(inputs)
    side = run.side_inputs(table)
    boundaries = {k: v["boundaries"] for k, v in protocol["frozen"]["subgroups"]["terciles"].items()}
    keys = ip.subgroup_keys(table, inputs.attributes, boundaries, side["gk"], side["gpt"], side["caption"],
                            side["timing_ms"], side["cut"])
    rows = [{"key": k, "side": s, "shots": int((keys[k] == s).sum()), "goal_count": int(y[(keys[k] == s).to_numpy()].sum())}
            for k in keys.columns for s in ("in", "out")]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tables
def interval(s: dict) -> dict:
    return {"delta": s["delta"], "lo95": s["lo95"], "hi95": s["hi95"]}


def flat(prefix: str, s: dict, keys=("delta", "lo95", "hi95")) -> dict:
    return {f"{prefix}_{k}": s[k] for k in keys}


def block_row(block: str, r: dict) -> dict:
    ratio = r["ratio_R_over_G"]
    return ({"block": block, "role": r.get("role")} | flat("gain", r["gain"], ("delta", "lo95", "hi95", "upper_one_sided_95", "p_one_sided"))
            | {"gain_rejected": r["gain_rejected"], "adjustment": r["adjustment"]}
            | flat("R", r["R"]) | flat("G", r["G"])
            | {"R_over_G": ratio["point"], "R_over_G_lo95": ratio["lo95"], "R_over_G_hi95": ratio["hi95"],
               "R_over_G_reportable": ratio["reportable"], "draws_G_le_0": ratio["draws_G_le_0"],
               "part_lo90": r["part"]["lo90"], "part_upper_one_sided_95": r["part"]["upper_one_sided_95"],
               "most_lo90": r["most"]["lo90"], "raw_verdict": r["raw_verdict"], "verdict": r["verdict"]}
            | flat("C_minus_BX", r["C-BX"]) | {"C_minus_BX_wording": r["C-BX"]["wording"]} | flat("CX_minus_C", r["CX-C"]))


def verdict_table(redundancy: dict) -> pd.DataFrame:
    return pd.DataFrame([block_row(b, r) for b, r in redundancy["blocks"].items()])


def estimand_table(redundancy: dict) -> pd.DataFrame:
    return pd.DataFrame([{"estimand": e} | block_row(b, r) for e, blocks in redundancy["estimands"].items()
                         for b, r in blocks.items()])


def leak_table(redundancy: dict) -> pd.DataFrame:
    rows = []
    for block, stats in redundancy["leak"].items():
        for name, v in stats.items():
            row = {"block": block, "statistic": name, "primary": redundancy["blocks"][block][name]["delta"]}
            no_v3 = redundancy["estimands"]["no_v3"][block][name]
            row |= flat("no_v3", no_v3) | flat("unseen", redundancy["estimands"]["unseen"][block][name])
            for key, h in v["heterogeneity"].items():
                row |= flat(key, h)
            rows.append(row | {"leak_verdict": v["verdict"], "heterogeneity_excluding_0": ";".join(v["heterogeneity_excluding_0"]),
                               "no_v3_excludes_primary_point": v["no_v3_interval_excludes_primary_point"]})
    return pd.DataFrame(rows)


def subgroup_table(subgroups: dict, counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counts = counts.set_index(["key", "side"])
    for key, e in subgroups["keys"].items():
        contrast = e.get("contrast_in_minus_out")
        for s, v in e["sides"].items():
            if counts.loc[(key, s), "shots"] != v["shots"]:
                raise SystemExit(f"subgroup {key}/{s}: shots differ from the run's")
            row = {"key": key, "status": e["status"], "weighted": e["weighted"], "side": s, "shots": v["shots"],
                   "goal_count": int(counts.loc[(key, s), "goal_count"]), "readable": v["readable"]}
            if v["readable"]:
                row |= flat("mean_d", v["mean_d"]) | {"share_of_total_gain": v["share_of_total_gain"],
                                                      "mean_d_goals": v["goals"], "mean_d_non_goals": v["non_goals"],
                                                      "standardised": v.get("standardised")}
            if contrast:
                row |= flat("contrast", contrast, ("delta", "lo95", "hi95", "p_two_sided"))
            rows.append(row | {"holm_rejected": e.get("holm_rejected")})
    return pd.DataFrame(rows)


def ci(d, lo, hi, digits=4) -> str:
    return f"{d:+.{digits}f} [{lo:+.{digits}f}; {hi:+.{digits}f}]"


def markdown(verdicts: pd.DataFrame, estimands: pd.DataFrame, leak: pd.DataFrame, subgroups: pd.DataFrame,
             checks: dict, gate: dict) -> str:
    out = ["# xg-dino-info: redundancy arms and subgroups (ticket 13)", "",
           "Out of fold on 7,062 shots / 328 matches, 2,000 match draws; protocol sha256 " + run.PROTOCOL_SHA256[:8] + ". "
           "gain = LL_{B+X} - LL_B, R = G - G_X, G = LL_B - LL_C. Wording: " + ip.WORDING + ".",
           f"Boundary gate: 'does not explain' false rate at plant R = 0.25 G = {gate['rate']:.3f} > {gate['limit']}, so "
           "every raw 'does not explain' reads 'inconclusive'.", "",
           "| block | role | gain [95% CI] | Holm/test rejects | R [95% CI] | R/G [95% CI] | raw | verdict | C - (B+X) | C1 wording |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in verdicts.iterrows():
        out.append(f"| {r.block} | {r.role} | {ci(r.gain_delta, r.gain_lo95, r.gain_hi95, 5)} | {r.gain_rejected} | "
                   f"{ci(r.R_delta, r.R_lo95, r.R_hi95, 5)} | {r.R_over_G:.3f} [{r.R_over_G_lo95:.3f}; {r.R_over_G_hi95:.3f}] | "
                   f"{r.raw_verdict} | {r.verdict} | {ci(r.C_minus_BX_delta, r.C_minus_BX_lo95, r.C_minus_BX_hi95)} | "
                   f"{r.C_minus_BX_wording} |")
    g = verdicts.iloc[0]
    out += ["", f"G = {ci(g.G_delta, g.G_lo95, g.G_hi95)}; draws with G <= 0: {int(g.draws_G_le_0)}.", "",
            "## Estimands (no_v3, unseen, unseen standardised)", "",
            "| estimand | block | gain | R | R/G | raw | verdict |", "|---|---|---|---|---|---|---|"]
    for _, r in estimands.iterrows():
        out.append(f"| {r.estimand} | {r.block} | {ci(r.gain_delta, r.gain_lo95, r.gain_hi95, 5)} | "
                   f"{ci(r.R_delta, r.R_lo95, r.R_hi95, 5)} | {r.R_over_G:.3f} | {r.raw_verdict} | {r.verdict} |")
    out += ["", "## Leak table (C1)", "", "| block | statistic | primary | no_v3 [95% CI] | seen - unseen | v3 - no_v3 | verdict |",
            "|---|---|---|---|---|---|---|"]
    for _, r in leak.iterrows():
        out.append(f"| {r.block} | {r.statistic} | {r.primary:+.5f} | {ci(r.no_v3_delta, r.no_v3_lo95, r.no_v3_hi95, 5)} | "
                   f"{ci(r['seen-unseen_delta'], r['seen-unseen_lo95'], r['seen-unseen_hi95'], 5)} | "
                   f"{ci(r['v3-no_v3_delta'], r['v3-no_v3_lo95'], r['v3-no_v3_hi95'], 5)} | {r.leak_verdict} |")
    out += ["", "## Subgroups (d = ℓ_C - ℓ_B; intervals conditional on the fitted predictions)", "",
            "| key | status | side | shots | goals | mean d [95% CI] | share of gain | contrast in - out [95% CI] | p | Holm |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in subgroups.iterrows():
        md = ci(r.mean_d_delta, r.mean_d_lo95, r.mean_d_hi95) if r.readable else "not read (< 100 shots)"
        share = "" if pd.isna(r.get("share_of_total_gain")) else f"{r.share_of_total_gain:.3f}"
        con = "" if pd.isna(r.get("contrast_delta")) else ci(r.contrast_delta, r.contrast_lo95, r.contrast_hi95)
        p = "" if pd.isna(r.get("contrast_p_two_sided")) else f"{r.contrast_p_two_sided:.3f}"
        holm = "" if r.status != "confirmatory" else str(r.holm_rejected)
        out.append(f"| {r.key} | {r.status} | {r.side} | {r.shots} | {r.goal_count} | {md} | {share} | {con} | {p} | {holm} |")
    a = checks["arm_refit"]
    out += ["", "## Checks", "",
            "- refit B/C/BQ/CQ vs frozen xg-dino OOF, max |Δp|: " + ", ".join(
                f"{k.removeprefix('max_abs_delta_p_')} {v:.1e}" for k, v in checks["refit_vs_frozen_xgdino"].items()),
            "- replay of " + ", ".join(f"{k} max |Δp| {v['max_abs_delta_p']:.1e} (C per fold equal: {v['chosen_c_equal']})"
                                       for k, v in a.items()),
            f"- redundancy read-out re-derived from the stored probabilities: {checks['read_out_from_stored_probabilities']['leaves']} "
            f"leaves, max |Δ| {checks['read_out_from_stored_probabilities']['max_abs_delta']:.1e}, "
            f"mismatches {len(checks['read_out_from_stored_probabilities']['mismatches'])}",
            "", "## Disclosures", "", *[f"- {d}" for d in ip.DISCLOSURES], ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = datetime.now(UTC)
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete")
    record = checked_run(args.out / "real")
    protocol = json.loads(run.PROTOCOL.read_text())
    if sha256_file(run.PROTOCOL) != run.PROTOCOL_SHA256:
        raise SystemExit("the protocol file is not the pinned one")
    summary = json.loads((args.out / "real/summary.json").read_text())
    stored = pd.read_csv(Path(record["storage"]["root"]) / "probabilities.csv", index_col="shot_id")
    inputs = ip.load_inputs()
    y = run.outcomes(inputs)
    checks = replay(inputs, y, stored, summary, protocol)
    print(json.dumps(checks, indent=2, default=float))
    redundancy = summary["redundancy"]
    tables = {"verdicts": verdict_table(redundancy), "estimands": estimand_table(redundancy),
              "leak": leak_table(redundancy), "subgroups": subgroup_table(summary["subgroups"], goal_counts(inputs, y, protocol))}
    for name, frame in tables.items():
        frame.to_csv(args.out / f"{name}.csv", index=False, float_format="%.9g")
    run.write_json(args.out / "checks.json", checks)
    (args.out / "tables.md").write_text(markdown(*tables.values(), checks, summary["boundary_gate"]))
    run.write_json(args.out / "complete.json", {
        "ticket": "xg-dino 13", "protocol_sha256": run.PROTOCOL_SHA256, "run": {"complete_sha256": sha256_file(args.out / "real/complete.json"),
                                                                                "storage": record["storage"]},
        "checks_passed": checks["passed"], "git": detector_records.git_state(),
        "started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": run.files(args.out)})
    print((args.out / "tables.md").read_text())


if __name__ == "__main__":
    main()
