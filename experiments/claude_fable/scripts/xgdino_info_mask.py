"""Masking read-out (d) of xg-dino-info (ticket 15). CPU only, no StatsBomb, labels = the index goal.

    ./run.sh --no-sync python scripts/xgdino_info_mask.py

Runs the pinned protocol's masking family (`research/xgdino_info_protocol.json`, sha256 39a547b5..., unchanged) on the
real masked scores of ticket 14 with the pinned library functions (`info_protocol.region_test`, `masking_family`,
`fill_agreement`, `gain_left`, `fold_fits`, `frozen_predict`, `refit`, `region_overlap`). The pinned script runs the
masking read-out only on stand-in z in its dry run (`masking_dry_run`); this script is that function with the real z.
It refuses unless the protocol is pinned and matches (`xgdino_info_protocol.checked_protocol`), the dry run ran under it
and ticket 14's `complete.json` matches its files.

Unmasked reference (spec-info, notes after the ticket 14 gate): the replayed unmasked z of `masked/unmasked.parquet`
(not the stored OOF z), so masked-vs-unmasked contrasts carry no replay noise. A shot whose region is absent keeps
its unmasked z. Estimands: `refit` (the arm refitted out of fold on the masked z, geometry fixed; primary) and `frozen`
(the arm's fold models fitted on the stored OOF z, applied to the masked z). Arms: C (primary), A (z alone;
descriptive). Tests per family region on the shots with the region present and a valid placebo; the placebo loss is the
mean of the 3 draws.

Outputs (`results/xgdino_info_mask/`, complete.json last): summary.json, regions.csv (family test + descriptive
regions, both arms), size_bins.csv, gain_left.csv, overlap.csv, checks.json, tables.md. The per-shot log-losses of every
variant go to Storage (`.../info/mask/losses.parquet`).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import ablation as ab
from xgpoc.dino import info_protocol as ip
from xgpoc.stack import stats as ss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xgdino_info_protocol as run  # noqa: E402

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "results/xgdino_info_mask"
MASKED = ip.INFO / "masked"
STORAGE = ip.INFO / "mask"
ARMS = ("C", "A")                       # C primary; A descriptive
FAMILY = ip.MASK_FAMILY
DESCRIPTIVE = (*ip.MASK_DESCRIPTIVE, *ip.KEEP_ONLY)
REGIONS = (*FAMILY, *DESCRIPTIVE)
BINS = ("small", "middle", "large")


# ------------------------------------------------------------------ inputs
def load_scores(index: pd.Index) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], dict]:
    """(replayed unmasked z, {(variant, fill): z with absent shots at the unmasked value}, per-region facts), aligned to
    `index`. Facts: present / placebo-feasible masks and n_patches per region (`REGIONS` + shooter+ball)."""
    scores = pd.read_parquet(MASKED / "scores.parquet", columns=["shot_id", "variant", "kind", "region", "draw", "fill",
                                                                "n_patches", "z"])
    ref = pd.read_parquet(MASKED / "unmasked.parquet").set_index("shot_id")
    if set(ref.index) != set(index) or len(ref) != len(index):
        raise SystemExit("unmasked.parquet is not the primary population")
    z0 = ref.z.reindex(index).to_numpy(float)
    unmasked = scores[scores.kind == "unmasked"].set_index("shot_id").z.reindex(index).to_numpy(float)
    if not np.array_equal(unmasked, z0):
        raise SystemExit("scores.parquet's unmasked rows differ from unmasked.parquet")
    z: dict[tuple[str, str], np.ndarray] = {}
    for (variant, fill), g in scores[scores.kind != "unmasked"].groupby(["variant", "fill"]):
        v = z0.copy()
        v[index.get_indexer(g.shot_id)] = g.z.to_numpy()
        z[(variant, fill)] = v
    mean_regions = scores[(scores.kind == "region") & (scores.fill == "mean")]
    placebo = scores[(scores.kind == "placebo") & (scores.fill == "mean")]
    facts = {}
    for r in (*REGIONS, "shooter+ball"):
        rows = mean_regions[mean_regions.region == r] if r in ip.MASK_FAMILY or r in ip.MASK_DESCRIPTIVE else \
            scores[(scores.kind == "keep") & (scores.region == r) & (scores.fill == "mean")]
        present = np.zeros(len(index), bool)
        present[index.get_indexer(rows.shot_id)] = True
        patches = np.full(len(index), np.nan)
        patches[index.get_indexer(rows.shot_id)] = rows.n_patches.to_numpy(float)
        draws = placebo[placebo.region == r].groupby("shot_id").size().reindex(index, fill_value=0).to_numpy()
        facts[r] = {"present": present, "n_patches": patches, "feasible": draws == 3, "draws": draws}
    return z0, z, facts


# ------------------------------------------------------------------ estimators
class Estimator:
    """An arm's per-shot out-of-fold log-loss on a replaced z: `refit` (refitted on it) or `frozen` (fold models fitted
    on the stored OOF z). Results are kept by (estimand, key)."""

    def __init__(self, table: pd.DataFrame, arm, model, y: np.ndarray):
        self.table, self.arm, self.model, self.y = table, arm, model, y
        self.fits = ip.fold_fits(table, arm, model, y)

    def probability(self, estimand: str, z: np.ndarray) -> np.ndarray:
        t = ip.with_z(self.table, z)
        return ip.frozen_predict(self.fits, t) if estimand == "frozen" else ip.refit(t, self.arm, self.model, self.y)

    def loss(self, estimand: str, z: np.ndarray) -> np.ndarray:
        return ip.losses(self.y, {"q": self.probability(estimand, z)})["q"]


def all_losses(estimators: dict[str, Estimator], z0: np.ndarray, z: dict, log) -> tuple[dict, dict]:
    """({(arm, estimand, fill, variant): loss}, {(arm, estimand): reference loss on z0}) for every variant."""
    ref = {(a, e): est.loss(e, z0) for a, est in estimators.items() for e in ip.ESTIMANDS}
    out = {}
    for n, ((variant, fill), v) in enumerate(sorted(z.items())):
        for a, est in estimators.items():
            for e in ip.ESTIMANDS:
                out[(a, e, fill, variant)] = est.loss(e, v)
        if n % 20 == 0:
            log(f"variant {n + 1}/{len(z)}: {variant} ({fill})")
    return out, ref


# ------------------------------------------------------------------ read-outs
def flat(prefix: str, s: dict) -> dict:
    return {f"{prefix}_delta": s["delta"], f"{prefix}_lo95": s["lo95"], f"{prefix}_hi95": s["hi95"]}


def placebo_loss(losses: dict, arm: str, estimand: str, fill: str, region: str) -> np.ndarray:
    return np.mean([losses[(arm, estimand, fill, f"{region}|placebo{k}")] for k in range(ip.N_PLACEBO)], axis=0)


def region_rows(losses, ref, facts, weights, arm: str, estimand: str, fill: str) -> tuple[list[dict], dict]:
    """(rows, tests) of one (arm, estimand, fill): the family regions with their pinned test on the present and
    placebo-feasible shots, then all descriptive regions."""
    base = ref[(arm, estimand)]
    rows, tests = [], {}
    for region in REGIONS:
        masked = losses[(arm, estimand, fill, region)]
        f = facts[region]
        row = {"arm": arm, "estimand": estimand, "fill": fill, "region": region, "present": int(f["present"].sum())}
        point, draws = ip.stat(masked - base, weights, f["present"])
        row |= flat("present_vs_unmasked", ab._summary(point, draws))
        point, draws = ip.stat(masked - base, weights)
        row |= flat("all_vs_unmasked", ab._summary(point, draws))
        if region in FAMILY:
            test_pop = f["present"] & f["feasible"]
            t = ip.region_test(base, masked, placebo_loss(losses, arm, estimand, fill, region), weights, test_pop)
            tests[region] = t
            row |= {"test_shots": t["shots"]} | flat("vs_unmasked", t["vs_unmasked"]) | flat("vs_placebo", t["vs_placebo"])
            row |= {"p_vs_unmasked": t["vs_unmasked"]["p_degradation"], "p_vs_placebo": t["vs_placebo"]["p_degradation"],
                    "p": t["p"]}
        rows.append(row)
    holm = ip.masking_family(tests)
    for row in rows:
        row["holm_rejected"] = holm.get(row["region"])
    return rows, {"tests": tests, "holm": holm}


def size_bins(losses, ref, facts, weights, arm: str, estimand: str, fill: str) -> list[dict]:
    """Read-outs of the family regions binned by region size in patches: terciles of n_patches over the test shots
    (bottom <= q1/3, top > q2/3), frozen here."""
    base = ref[(arm, estimand)]
    rows = []
    for region in FAMILY:
        f = facts[region]
        pop = f["present"] & f["feasible"]
        n = f["n_patches"]
        q1, q2 = np.percentile(n[pop], [100 / 3, 200 / 3])
        bins = {"small": pop & (n <= q1), "middle": pop & (n > q1) & (n <= q2), "large": pop & (n > q2)}
        masked, placebo = losses[(arm, estimand, fill, region)], placebo_loss(losses, arm, estimand, fill, region)
        for name, m in bins.items():
            row = {"arm": arm, "estimand": estimand, "fill": fill, "region": region, "bin": name, "shots": int(m.sum()),
                   "patches_min": float(n[m].min()) if m.any() else np.nan, "patches_max": float(n[m].max()) if m.any() else np.nan}
            if m.sum() >= 30:
                row |= flat("vs_unmasked", ab._summary(*ip.stat(masked - base, weights, m)))
                row |= flat("vs_placebo", ab._summary(*ip.stat(masked - placebo, weights, m)))
            rows.append(row)
    return rows


def gain_rows(losses, ref, facts, weights, ell_b: dict, estimand: str, fill: str) -> list[dict]:
    """(LL_B - LL_C_mask) / (LL_B - LL_C): over all shots (the pinned `gain_left`) and over the shots with the region
    present (same ratio restricted)."""
    rows = []
    b = ell_b["B"]
    for region in REGIONS:
        ell = {"B": b, "C": ref[("C", estimand)], "Cm": losses[("C", estimand, fill, region)]}
        row = {"estimand": estimand, "fill": fill, "region": region} | {f"all_{k}": v for k, v in ip.gain_left(ell, "Cm", weights).items()}
        m = facts[region]["present"]
        pn, dn = ip.stat(ell["B"] - ell["Cm"], weights, m)
        pd_, dd = ip.stat(ell["B"] - ell["C"], weights, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = dn / dd
        r = r[np.isfinite(r)]
        row |= {"present_point": pn / pd_, "present_lo95": float(np.percentile(r, 2.5)), "present_hi95": float(np.percentile(r, 97.5)),
                "present_LL_B_minus_LL_C": pd_}
        rows.append(row)
    return rows


def decisions(by_key: dict) -> dict:
    """Per estimand: the Holm decision per fill and the fill agreement (pinned `fill_agreement`)."""
    return {e: {"holm": {f: by_key[(e, f)]["holm"] for f in ip.FILLS},
                "agreement": ip.fill_agreement({f: by_key[(e, f)]["holm"] for f in ip.FILLS})} for e in ip.ESTIMANDS}


# ------------------------------------------------------------------ checks
def continuity(estimators: dict, z0: np.ndarray, z_stored: np.ndarray, frozen: pd.DataFrame) -> dict:
    """The replayed-unmasked arm against the frozen xg-dino one (results/xgdino_ablation/real) and the stored OOF z."""
    out = {"max_abs_dz_replay_vs_stored": float(np.abs(z0 - z_stored).max()), "mean_abs_dz": float(np.abs(z0 - z_stored).mean())}
    p_c = frozen.p_C.to_numpy(float)
    for e in ip.ESTIMANDS:
        out[f"C_{e}_replay_vs_frozen_xgdino_max_abs_dp"] = float(np.abs(estimators["C"].probability(e, z0) - p_c).max())
        out[f"C_{e}_replay_vs_frozen_xgdino_mean_abs_dp"] = float(np.abs(estimators["C"].probability(e, z0) - p_c).mean())
    out["C_refit_on_stored_z_vs_frozen_xgdino_max_abs_dp"] = float(np.abs(estimators["C"].probability("refit", z_stored) - p_c).max())
    return out


def markdown(regions: pd.DataFrame, bins: pd.DataFrame, gains: pd.DataFrame, overlap: pd.DataFrame, summary: dict) -> str:
    def ci(r, p, d=4):
        return f"{r[p + '_delta']:+.{d}f} [{r[p + '_lo95']:+.{d}f}; {r[p + '_hi95']:+.{d}f}]"
    out = ["# xg-dino-info: masking read-out (ticket 15)", "",
           "Out of fold, 7,062 shots / 328 matches, 2,000 match draws, protocol sha256 " + run.PROTOCOL_SHA256[:8] +
           ". Positive = degradation = \"sensitivity to the intervention\", not proof of what DINOv2 uses. Unmasked reference "
           "= the replayed unmasked z (ticket 14). Test population = region present and placebo-feasible.", ""]
    for estimand in ip.ESTIMANDS:
        d = summary["decisions"][estimand]
        out += [f"## Family regions, C, {estimand} estimand", "",
                "| region | n present | n test | mean fill: vs unmasked [95% CI] | vs placebo [95% CI] | p | Holm | blur fill: vs unmasked | "
                "vs placebo | Holm | fills | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        sel = regions[(regions.arm == "C") & (regions.estimand == estimand) & regions.region.isin(FAMILY)]
        for region in FAMILY:
            m = sel[(sel.region == region) & (sel.fill == "mean")].iloc[0]
            b = sel[(sel.region == region) & (sel.fill == "blur")].iloc[0]
            out.append(f"| {region} | {m.present} | {int(m.test_shots)} | {ci(m, 'vs_unmasked')} | {ci(m, 'vs_placebo')} | {m.p:.4f} | "
                       f"{m.holm_rejected} | {ci(b, 'vs_unmasked')} | {ci(b, 'vs_placebo')} | {b.holm_rejected} | "
                       f"{'agree' if d['agreement'][region].startswith(('sensitive', 'no ')) else 'disagree'} | {d['agreement'][region]} |")
        out.append("")
    out += ["## Descriptive regions (C, mean fill, present shots, vs unmasked)", "",
            "| region | estimand | present | refit/frozen delta [95% CI] | all shots |", "|---|---|---|---|---|"]
    for _, r in regions[(regions.arm == "C") & (regions.fill == "mean") & regions.region.isin(DESCRIPTIVE)].iterrows():
        out.append(f"| {r.region} | {r.estimand} | {r.present} | {ci(r, 'present_vs_unmasked')} | {ci(r, 'all_vs_unmasked')} |")
    out += ["", "## A_mask - A (descriptive, mean fill, present shots vs unmasked; family also vs placebo)", "",
            "| region | estimand | vs unmasked | vs placebo |", "|---|---|---|---|"]
    for _, r in regions[(regions.arm == "A") & (regions.fill == "mean")].iterrows():
        vp = ci(r, "vs_placebo") if r.region in FAMILY else ""
        out.append(f"| {r.region} | {r.estimand} | {ci(r, 'present_vs_unmasked')} | {vp} |")
    out += ["", "## C-B gain left, (LL_B - LL_C_mask) / (LL_B - LL_C), mean fill", "",
            "| region | estimand | all shots [95% CI] | present shots [95% CI] |", "|---|---|---|---|"]
    for _, r in gains[gains.fill == "mean"].iterrows():
        out.append(f"| {r.region} | {r.estimand} | {r.all_point:.3f} [{r.all_lo95:.3f}; {r.all_hi95:.3f}] | "
                   f"{r.present_point:.3f} [{r.present_lo95:.3f}; {r.present_hi95:.3f}] |")
    out += ["", "## By region size (patches; C, refit, mean fill)", "",
            "| region | bin | shots | patches | vs unmasked | vs placebo |", "|---|---|---|---|---|---|"]
    for _, r in bins[(bins.arm == "C") & (bins.estimand == "refit") & (bins.fill == "mean")].iterrows():
        if "vs_unmasked_delta" in r and not pd.isna(r.vs_unmasked_delta):
            out.append(f"| {r.region} | {r.bin} | {r.shots} | {r.patches_min:.0f}-{r.patches_max:.0f} | {ci(r, 'vs_unmasked')} | {ci(r, 'vs_placebo')} |")
    out += ["", "## Region overlap (share of a's patches inside b; Jaccard)", "",
            "| a | b | shots | share of a in b | Jaccard |", "|---|---|---|---|---|"]
    for _, r in overlap.iterrows():
        out.append(f"| {r.a} | {r.b} | {r.shots} | {r.share_of_a_in_b:.3f} | {r.jaccard:.3f} |")
    out += ["", "## Checks", "", "```", json.dumps(summary["checks"], indent=2), "```", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = datetime.now(UTC)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    if not (MASKED / "complete.json").exists():
        raise SystemExit("ticket 14's masked/complete.json is missing")
    inputs = ip.load_inputs()
    table, _ = run.population(inputs)
    _, protocol_sha = run.checked_protocol(run.PROTOCOL, inputs, table, need_pin=True)
    dry = run.OUT / "dryrun" / "complete.json"
    if not dry.exists() or json.loads(dry.read_text())["protocol_sha256"] != protocol_sha:
        raise SystemExit("the dry run did not run under the pinned protocol")
    y = run.outcomes(inputs)
    z0, z, facts = load_scores(table.index)
    z_stored = table[ab.Z].to_numpy(float)
    frozen = run.frozen_probabilities(table)
    weights = ss.bootstrap_weights(table.game.to_numpy(), ss.N_BOOT, ss.BOOT_SEED)
    arms, _ = ab.arm_set(table, ("A", "B", "C"))
    models = {n: ab.model_for(a) for n, a in arms.items()}
    log = lambda s: print(s, flush=True)  # noqa: E731
    with no_statsbomb_reads():
        estimators = {a: Estimator(table, arms[a], models[a], y) for a in ARMS}
        losses, ref = all_losses(estimators, z0, z, log)
        ell_b = {"B": ip.losses(y, {"B": ip.refit(table, arms["B"], models["B"], y)})["B"]}
        checks = continuity(estimators, z0, z_stored, frozen)
    checks["B_refit_vs_frozen_xgdino_max_abs_dp"] = float(np.abs(
        ip.refit(table, arms["B"], models["B"], y) - frozen.p_B.to_numpy(float)).max())
    # populations
    summary = {"protocol_sha256": protocol_sha, "population": {r: {"present": int(f["present"].sum()),
                                                                  "placebo_feasible": int((f["present"] & f["feasible"]).sum())}
                                                               for r, f in facts.items()}}
    summary["population_notes"] = {"goal": "tested on its placebo-feasible subset only", "players": "descriptive (14 placebo-feasible shots)",
                                   "background": "descriptive, no placebo", "keep_*": "descriptive, no placebo"}
    region_table, by_key = [], {}
    bins, gains = [], []
    for arm in ARMS:
        for estimand in ip.ESTIMANDS:
            for fill in ip.FILLS:
                rows, tests = region_rows(losses, ref, facts, weights, arm, estimand, fill)
                region_table += rows
                if arm == "C":
                    by_key[(estimand, fill)] = tests
                bins += size_bins(losses, ref, facts, weights, arm, estimand, fill)
    for estimand in ip.ESTIMANDS:
        for fill in ip.FILLS:
            gains += gain_rows(losses, ref, facts, weights, ell_b, estimand, fill)
    summary["decisions"] = decisions(by_key)
    summary["tests"] = {f"{e}:{f}": {r: t for r, t in by_key[(e, f)]["tests"].items()} for e in ip.ESTIMANDS for f in ip.FILLS}
    summary["checks"] = checks
    regions_df, bins_df, gains_df = pd.DataFrame(region_table), pd.DataFrame(bins), pd.DataFrame(gains)
    overlap = ip.region_overlap(pd.read_parquet(ip.REGION_MASKS))
    summary["overlap_matches_protocol"] = bool(
        overlap.round(6).to_dict(orient="records") == json.loads(run.PROTOCOL.read_text())["frozen"]["region_overlap"])
    args.out.mkdir(parents=True)
    for name, frame in (("regions", regions_df), ("size_bins", bins_df), ("gain_left", gains_df), ("overlap", overlap)):
        frame.to_csv(args.out / f"{name}.csv", index=False, float_format="%.9g")
    run.write_json(args.out / "summary.json", summary)
    run.write_json(args.out / "checks.json", checks)
    (args.out / "tables.md").write_text(markdown(regions_df, bins_df, gains_df, overlap, summary))
    STORAGE.mkdir(parents=True, exist_ok=True)
    lw = pd.DataFrame({"|".join(k): v for k, v in losses.items()}, index=table.index)
    lw.to_parquet(STORAGE / "losses.parquet")
    run.write_json(args.out / "complete.json", {
        "ticket": "xg-dino 15", "protocol_sha256": protocol_sha, "masked_complete_sha256": sha256_file(MASKED / "complete.json"),
        "storage": {"losses.parquet": sha256_file(STORAGE / "losses.parquet")}, "git": detector_records.git_state(),
        "started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": run.files(args.out)})
    print((args.out / "tables.md").read_text())


if __name__ == "__main__":
    main()
