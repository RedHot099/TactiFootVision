"""The xg-dino-info protocol (ticket 11): label-free controls of the redundancy contract, the protocol file, the
permuted-label dry run, and the gates of the real redundancy / subgroup run (ticket 13). CPU only.

    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --controls --replicates 200 --jobs 15
    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --controls --replicates 2000 --start 0 --stop 150 --jobs 15
        ... (chunks; the chunk that completes [0, replicates) writes controls.json and complete.json last)
    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --protocol      # research/xgdino_info_protocol.json
    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --dryrun        # results/xgdino_info_protocol/dryrun
    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --real --check-only
    ./run.sh --no-sync python scripts/xgdino_info_protocol.py --finish        # results/xgdino_info_protocol/complete.json

--controls  label-free (`xgpoc.dino.info_controls`): the scenarios' truths are solved once per folder (truth.json),
            replicates [start, stop) of every scenario go to parts/rows_<start>_<stop>.csv; the chunk that completes
            the run writes controls.json, replicates.csv and complete.json last. A folder refuses other code.
--protocol  refuses unless the controls' complete.json matches its files and the code and the gate passed; computes the
            region overlap matrix (label-free) and writes the protocol (the frozen part `info_protocol.protocol`, input
            hashes, the import closure's hashes, git state). Refuses an existing file. Prints its sha256.
--dryrun    refuses unless the protocol's frozen part, inputs and code are the current ones; reads the index `goal` only
            to permute it within matches (seed 0); runs every arm and the redundancy read-out with the leak estimands,
            the subgroups on the frozen xg-dino OOF probabilities, the masking read-out on label-free stand-in masked z
            (z + N(0, 0.5²) noise on the region's shots, both fills, both estimands) and builds the probe target table.
            The within-match permutation diagnostic of C3.8.
--real      refuses unless PROTOCOL_SHA256 is pinned and matches, the frozen part, inputs and code match and the dry run
            ran under the same protocol; `--check-only` stops there. Then the redundancy and subgroup read-outs on the
            real labels (ticket 13).
--finish    writes results/xgdino_info_protocol/complete.json last (hashes of every file in the folder).
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.compare.stats import permute_within_matches
from xgpoc.detector import records as detector_records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import ablation as ab
from xgpoc.dino import info_controls as ic
from xgpoc.dino import info_protocol as ip
from xgpoc.dino import records
from xgpoc.dino import regions as rg
from xgpoc.stack import inputs as si

ROOT = Path(__file__).absolute().parents[1]
PROTOCOL = ROOT / "research/xgdino_info_protocol.json"
OUT = ROOT / "results/xgdino_info_protocol"
CONTROLS = OUT / "controls_r2000"
STORAGE = ip.INFO / "protocol"
PERMUTATION_SEED = 0
STANDIN_SD = 0.5
STANDIN_SEED = 11
CODE_ROOTS = ("xgpoc/dino/info_protocol.py", "xgpoc/dino/info_controls.py")
PROTOCOL_SHA256: str | None = "39a547b5b7b7642e5327a1a0de362faa46b3043dc79189736757b2a37bba3298"


# ------------------------------------------------------------------ records
def code_hashes() -> dict:
    """sha256 of the import closure of `CODE_ROOTS` and this script (pin-normalised), and their digest."""
    script = Path(__file__).resolve()
    files = records.import_closure([*(ROOT / r for r in CODE_ROOTS), script])
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in files if p != script}
    hashes[f"{script.relative_to(ROOT)} (pin normalised)"] = records.run_script_sha256(script)
    return {"digest": canonical_digest(hashes), "files": dict(sorted(hashes.items()))}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float, allow_nan=True) + "\n")


def json_equal(a, b) -> bool:
    return json.loads(json.dumps(a, default=float)) == json.loads(json.dumps(b, default=float))


def files(folder: Path) -> dict:
    return {str(p.relative_to(folder)): sha256_file(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "complete.json"}


def input_hashes(inputs: ip.Inputs) -> dict:
    extra = {"labels_complete": ip.LABELS / "complete.json", "header_gpt": ip.LABELS / "header_gpt.csv",
             "gpt_sample": ip.LABELS / "gpt_sample.csv", "header_caption": ip.LABELS / "header_caption.csv",
             "roles": ip.ATTRIBUTES / "roles.parquet", "region_summary": ip.ATTRIBUTES / "region_summary.parquet",
             "region_masks": ip.REGION_MASKS, "frozen_oof": ip.FROZEN_OOF, "offset_frames": ip.OFFSET_FRAMES,
             "ablation_protocol": ROOT / "research/xgdino_ablation_protocol.json"}
    return inputs.hashes | {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in extra.items()}


# ------------------------------------------------------------------ inputs
def population(inputs: ip.Inputs):
    table, columns = ip.design_table(inputs.ablation.primary.table, inputs.attributes)
    return table, columns


def side_inputs(table: pd.DataFrame) -> dict:
    """Label files and label-free side tables, aligned to the population's shots."""
    idx = table.index
    gpt = pd.read_csv(ip.LABELS / "header_gpt.csv", index_col="shot_id")
    sample = pd.read_csv(ip.LABELS / "gpt_sample.csv", index_col="shot_id")
    caption = pd.read_csv(ip.LABELS / "header_caption.csv", index_col="shot_id")
    for name, frame, source in (("header_gpt", gpt, "gpt"), ("header_caption", caption, "caption")):
        if not (frame.source == source).all():
            raise SystemExit(f"{name}: every row must carry source={source}")
    roles = pd.read_parquet(ip.ATTRIBUTES / "roles.parquet")
    gk = roles[roles.offset_ms == 0].set_index("shot_id").gk_detected.reindex(idx)
    frames = pd.read_csv(ip.OFFSET_FRAMES, usecols=["shot_id", "offset_ms", "timing_error_ms", "cut_between_anchor"],
                         dtype={"cut_between_anchor": "object"})
    t0 = frames[frames.offset_ms == 0].set_index("shot_id").timing_error_ms.abs().reindex(idx)
    cut = frames[frames.offset_ms == -400].set_index("shot_id").cut_between_anchor.reindex(idx)
    cut = cut.map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}).astype(float)          # NaN: unknown
    return {"gpt": gpt.header_gpt, "weight": sample.weight, "caption": caption.header_caption, "gk": gk,
            "timing_ms": t0, "cut": cut}


# ------------------------------------------------------------------ controls
def run_controls(out: Path, replicates: int, start: int, stop: int, jobs: int) -> None:
    digest = code_hashes()["digest"]
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} is complete")
    marker = out / "package_digest.txt"
    if marker.exists() and marker.read_text().strip() != digest:
        raise SystemExit(f"{out} was started under other code")
    parts = out / "parts"
    for p in sorted(parts.glob("rows_*.csv")) if parts.exists() else []:
        a, b = map(int, p.stem.split("_")[1:])
        if a < stop and start < b:
            raise SystemExit(f"chunk [{start}, {stop}) overlaps {p.name}")
    if not 0 <= start < stop <= replicates:
        raise SystemExit("need 0 <= start < stop <= replicates")
    clock = time.perf_counter()
    inputs = ip.load_inputs()
    table, columns = population(inputs)
    with no_statsbomb_reads():
        prepared = ip.prepare(table, columns, blocks=(ip.PRIMARY, *ip.FAMILY))
        truths = ic.Truths.build(prepared, table)
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(digest + "\n")
        if (out / "truth.json").exists():
            solved = json.loads((out / "truth.json").read_text())
        else:
            solved = ic.solve_all(truths)
            write_json(out / "truth.json", solved)
        frame = ic.chunk(prepared, truths, solved, start, stop, jobs, log=print)
    seconds = time.perf_counter() - clock
    parts.mkdir(exist_ok=True)
    frame.to_csv(parts / f"rows_{start}_{stop}.csv", index=False, float_format="%.9g")
    write_json(parts / f"timing_{start}_{stop}.json", {"seconds": round(seconds, 1), "jobs": jobs})
    print(f"chunk [{start}, {stop}): {seconds:.0f} s with {jobs} workers")
    rows = read_rows(parts)
    if len(rows) == len(ic.SCENARIOS) * replicates:
        summary = ic.finish(solved, rows.sort_values(["scenario", "replicate"]).reset_index(drop=True), replicates)
        write_json(out / "controls.json", summary)
        rows.to_csv(out / "replicates.csv", index=False, float_format="%.9g")
        chunks = [json.loads(p.read_text()) for p in sorted(parts.glob("timing_*.json"))]
        record = {"mode": f"controls_r{replicates}", "controls": summary, "inputs": input_hashes(inputs),
                  "code": {"package_digest": digest}, "git": detector_records.git_state(),
                  "timings_s": {"chunks": chunks, "total": round(sum(c["seconds"] for c in chunks), 1)},
                  "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": files(out)}
        write_json(out / "complete.json", record)
        print(json.dumps({"gates": summary["gates"], "timings_s": record["timings_s"]["total"]}, indent=2, default=float))


def read_rows(parts: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, keep_default_na=False, na_values=["", "nan", "NaN"]) for p in sorted(parts.glob("rows_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def controls_record(folder: Path, digest: str) -> tuple[dict, str]:
    path = folder / "complete.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --controls first")
    record = json.loads(path.read_text())
    for name, sha in record["files"].items():
        if sha256_file(folder / name) != sha:
            raise SystemExit(f"{folder / name} does not match the controls' completion record")
    if record["code"]["package_digest"] != digest:
        raise SystemExit("the controls ran under other code")
    if not record["controls"]["gates"]["passed"]:
        raise SystemExit(f"a control gate failed: {record['controls']['gates']}; stop for the council")
    return record, sha256_file(path)


# ------------------------------------------------------------------ protocol
def extras(inputs: ip.Inputs, table: pd.DataFrame, controls: Path, digest: str) -> dict:
    controls = controls.absolute()
    record, sha = controls_record(controls, digest)
    c = record["controls"]
    overlap = ip.region_overlap(pd.read_parquet(ip.REGION_MASKS))
    summary = pd.read_parquet(ip.ATTRIBUTES / "region_summary.parquet")
    present = {r: int(g.present.sum()) for r, g in summary.groupby("region")}
    feasible = {r: int((g.placebo_status == "ok").sum()) for r, g in summary.groupby("region")}
    return {"validation": {"controls": {"path": str(controls.relative_to(ROOT)), "complete_sha256": sha,
                                        "replicates": c["replicates"], "seeds": c["seeds"], "truth": c["truth"],
                                        "scenarios": c["scenarios"], "gates": c["gates"]}},
            "boundary_gate": ip.boundary_gate(c),
            "region_overlap": overlap.round(6).to_dict(orient="records"),
            "region_counts": {"present": present, "placebo_feasible": feasible},
            "name_collisions": {"share_equal": ip.name_collisions(inputs.ablation.primary.table, inputs.attributes),
                                "rule": "Xa design columns are namespaced xa_<name>"},
            "dry_run": {"labels": f"permuted within matches (seed {PERMUTATION_SEED})", "role": "diagnostic only (C3.8)",
                        "masking": f"label-free stand-in masked z = z + N(0, {STANDIN_SD}²) on the region's shots"},
            "counts": inputs.ablation.counts}


def frozen_part(inputs: ip.Inputs, table: pd.DataFrame, controls: Path, digest: str) -> dict:
    return ip.protocol(inputs, extras(inputs, table, controls, digest))


def checked_protocol(path: Path, inputs: ip.Inputs, table: pd.DataFrame, need_pin: bool) -> tuple[dict, str]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist: run --protocol first")
    sha = sha256_file(path)
    if need_pin and (PROTOCOL_SHA256 is None or sha != PROTOCOL_SHA256):
        raise SystemExit("PROTOCOL_SHA256 is not pinned or does not match the protocol file: --real refuses")
    record = json.loads(path.read_text())
    digest = code_hashes()["digest"]
    if record["code"]["digest"] != digest:
        raise SystemExit("the code differs from the code the protocol pins")
    controls = ROOT / record["frozen"]["validation"]["controls"]["path"]
    if not json_equal(record["frozen"], frozen_part(inputs, table, controls, digest)):
        raise SystemExit("the protocol's frozen part differs from info_protocol.protocol()")
    if not json_equal(record["inputs"], input_hashes(inputs)):
        raise SystemExit("the inputs differ from the protocol's")
    return record, sha


# ------------------------------------------------------------------ read-outs
def outcomes(inputs: ip.Inputs) -> np.ndarray:
    with no_statsbomb_reads():
        y = si.load_outcomes(inputs.ablation.primary).aligned(inputs.ablation.primary)
    goals = {"primary": int(y.sum()), "no_v3": int(y[inputs.ablation.subsets["no_v3"]].sum())}
    if goals != ab.EXPECTED_GOALS:
        raise SystemExit(f"goal counts {goals} are not the spec's {ab.EXPECTED_GOALS}")
    return y


def frozen_probabilities(table: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_csv(ip.FROZEN_OOF, index_col="shot_id")
    if not frozen.index.equals(table.index):
        raise SystemExit("the frozen OOF probabilities are not the population in its order")
    return frozen


def subgroup_read(table, inputs, side, y, p: dict, weights, boundaries: dict) -> dict:
    """The subgroups on the run's own out-of-fold B and C (on the real labels they replay the frozen xg-dino OOF:
    `replay`); frozen predictions scored against permuted labels would not be a null."""
    ell = ip.losses(y, {"B": p["B"], "C": p["C"]})
    keys = ip.subgroup_keys(table, inputs.attributes, boundaries, side["gk"], side["gpt"], side["caption"],
                            side["timing_ms"], side["cut"])
    return ip.subgroups(y, ell["C"] - ell["B"], weights, keys, side["weight"], float(y.mean()))


def standin_loss(table, arm, model, fits, estimand: str, y: np.ndarray, rng: np.random.Generator,
                 on: np.ndarray) -> np.ndarray:
    """Per-shot log-loss of C on a stand-in masked z (z + N(0, STANDIN_SD²) on the shots `on`), refitted or frozen."""
    z = table[ab.Z].to_numpy(float)
    zm = np.where(on, z + rng.normal(0.0, STANDIN_SD, len(z)), z)
    if estimand == "frozen":
        q = ip.frozen_predict(fits, ip.with_z(table, zm))
    else:
        q = ip.refit(ip.with_z(table, zm), arm, model, y)
    return ip.losses(y, {"q": q})["q"]


def masking_dry_run(table: pd.DataFrame, prepared: ip.Prepared, y: np.ndarray, p: dict, log) -> dict:
    """The masking read-out on label-free stand-in masked z (module docstring)."""
    block = prepared.block
    arm, model = block.arms["C"], block.models["C"]
    summary = pd.read_parquet(ip.ATTRIBUTES / "region_summary.parquet").set_index(["region", "shot_id"])
    present = {r: summary.loc[r].present.reindex(table.index, fill_value=False).to_numpy(bool) for r in rg.REGIONS}
    feasible = {r: (summary.loc[r].placebo_status == "ok").reindex(table.index, fill_value=False).to_numpy(bool)
                for r in rg.PLACEBO_REGIONS}
    present["shooter+ball"] = present["shooter"] & present["ball"]
    feasible["shooter+ball"] = feasible["shooter"] & feasible["ball"]
    for name, parts in ip.KEEP_ONLY.items():
        present[name] = np.logical_or.reduce([present[r] for r in parts])
    fits = ip.fold_fits(table, arm, model, y)
    ell = ip.losses(y, {"B": p["B"], "C": p["C"]})
    rng = np.random.default_rng(STANDIN_SEED)
    out = {}
    for fill in ip.FILLS:
        for estimand in ip.ESTIMANDS:
            tests, gains = {}, {}
            for region in (*ip.MASK_FAMILY, *ip.MASK_DESCRIPTIVE, *ip.KEEP_ONLY):
                on = present[region]
                n_draws = ip.N_PLACEBO if region in ip.MASK_FAMILY else 0

                masked = standin_loss(table, arm, model, fits, estimand, y, rng, on)
                if n_draws:
                    placebo = np.mean([standin_loss(table, arm, model, fits, estimand, y, rng, on) for _ in range(n_draws)], axis=0)
                    population = on & feasible[region]
                    tests[region] = ip.region_test(ell["C"], masked, placebo, block.weights, population)
                    tests[region]["population"] = "present and placebo-feasible"
                else:
                    point, draws = ip.stat(masked - ell["C"], block.weights, on)
                    tests[region] = {"vs_unmasked": ab._summary(point, draws), "shots": int(on.sum()),
                                     "population": "present (descriptive, no placebo)"}
                if estimand == "refit":
                    gains[region] = ip.gain_left(ell | {"Cm": masked}, "Cm", block.weights)
            holm = ip.masking_family(tests)
            out[f"{fill}:{estimand}"] = {"tests": tests, "holm": holm, "gain_left": gains}
            log(f"masking {fill}:{estimand}: Holm {holm}")
    out["fill_agreement"] = {e: ip.fill_agreement({f: out[f"{f}:{e}"]["holm"] for f in ip.FILLS}) for e in ip.ESTIMANDS}
    return out


def read_all(inputs: ip.Inputs, table, columns, y, protocol: dict, log, masking: bool) -> tuple[dict, dict]:
    clock = time.perf_counter()
    prepared = ip.prepare(table, columns, inputs.ablation.subsets)
    p, chosen = ip.fit_arms(prepared, y)
    log(f"arms fitted in {time.perf_counter() - clock:.0f} s")
    weights = prepared.block.weights
    gate = protocol["frozen"]["boundary_gate"]
    summary = {"boundary_gate": gate,
               "redundancy": ip.redundancy(y, p, weights, prepared.masks, not_explain_available=gate["available"]),
               "chosen_c": chosen, "grid_edge": {n: ab.grid_edges(c) for n, c in chosen.items()}}
    side = side_inputs(table)
    boundaries = {k: v["boundaries"] for k, v in protocol["frozen"]["subgroups"]["terciles"].items()}
    summary["subgroups"] = subgroup_read(table, inputs, side, y, p, weights, boundaries)
    frozen = frozen_probabilities(table)
    summary["replay"] = {f"max_abs_delta_p_{a}": float(np.abs(p[a] - frozen[f"p_{a}"].to_numpy()).max())
                         for a in ("B", "C", "BQ", "CQ")} | {
        "meaning": "against results/xgdino_ablation/real: ~0 on the real labels (ticket 13 gate); not 0 on permuted labels"}
    if masking:
        summary["masking"] = masking_dry_run(table, prepared, y, p, log)
        frozen = frozen_probabilities(table)
        targets, meta = ip.probe_targets(inputs.attributes, side["gpt"], side["caption"], frozen)
        summary["probe_targets"] = meta.to_dict(orient="records")
    return summary, p


def dry_run_check(summary: dict) -> dict:
    blocks = summary["redundancy"]["blocks"]
    hits = {b: r["verdict"] for b, r in blocks.items() if r["verdict"] in ("explains most", "explains part")}
    rejected = [b for b, r in blocks.items() if r["gain_rejected"]]
    sub = summary["subgroups"]["keys"]
    confirm = [k for k in ip.CONFIRMATORY_SUBGROUPS if sub.get(k, {}).get("holm_rejected")]
    mask = [f"{k}:{r}" for k, v in summary.get("masking", {}).items() if ":" in k for r, h in v["holm"].items() if h]
    return {"explains_verdicts": hits, "gain_rejected": rejected, "subgroup_holm_rejected": confirm,
            "masking_holm_rejected": mask,
            "passed": not hits and not confirm and not mask,
            "verdicts": {b: r["verdict"] for b, r in blocks.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    for m in ("--controls", "--protocol", "--dryrun", "--real", "--finish"):
        mode.add_argument(m, action="store_true")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=15)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--controls-dir", type=Path, default=CONTROLS)
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL, help="the protocol --dryrun / --real check")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    started = datetime.now(UTC)
    if args.controls:
        run_controls(args.out or OUT / f"controls_r{args.replicates}", args.replicates, args.start,
                     args.stop or args.replicates, args.jobs)
        return
    if args.finish:
        write_json(OUT / "complete.json", {"ticket": "xg-dino 11", "protocol_sha256": sha256_file(PROTOCOL),
                                           "pinned": PROTOCOL_SHA256, "code": code_hashes(), "git": detector_records.git_state(),
                                           "written": started.isoformat(timespec="seconds"), "files": files(OUT)})
        return
    inputs = ip.load_inputs()
    table, columns = population(inputs)
    if args.protocol:
        out = args.protocol_file
        if out.exists():
            raise SystemExit(f"{out} exists; the protocol is written once")
        code = code_hashes()
        frozen = frozen_part(inputs, table, args.controls_dir, code["digest"])
        OUT.mkdir(parents=True, exist_ok=True)
        if out == PROTOCOL:
            pd.DataFrame(frozen["region_overlap"]).to_csv(OUT / "region_overlap.csv", index=False)
        record = {"purpose": "xg-dino ticket 11: the frozen xg-dino-info protocol, written before any real fit",
                  "frozen": frozen, "inputs": input_hashes(inputs), "code": code, "git": detector_records.git_state(),
                  "written": started.isoformat(timespec="seconds")}
        out.write_text(json.dumps(record, indent=2, allow_nan=False, default=float) + "\n")
        print(f"protocol sha256 {sha256_file(out)}")
        return
    name = "dryrun" if args.dryrun else "real"
    out = args.out or OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists; every run goes to a new folder")
    protocol, protocol_sha = checked_protocol(args.protocol_file, inputs, table, need_pin=args.real)
    if args.real:
        dry = OUT / "dryrun" / "complete.json"
        if not dry.exists() or json.loads(dry.read_text())["protocol_sha256"] != protocol_sha:
            raise SystemExit("the dry run did not run under the pinned protocol")
        if args.check_only:
            print(f"--real accepts: protocol {protocol_sha} pinned, code, inputs, controls and dry run match; no label read")
            return
    y = outcomes(inputs)
    if args.dryrun:
        y = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), PERMUTATION_SEED).goal.to_numpy()
    clock = time.perf_counter()
    with no_statsbomb_reads():
        summary, p = read_all(inputs, table, columns, y, protocol, print, masking=args.dryrun)
    seconds = round(time.perf_counter() - clock, 1)
    out.mkdir(parents=True)
    storage = STORAGE / out.name
    storage.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"shot_id": table.index} | {f"p_{k}": v for k, v in p.items()}).to_csv(
        storage / "probabilities.csv", index=False, float_format="%.9g")
    write_json(out / "summary.json", summary)
    extra = {"dry_run": dry_run_check(summary), "permutation_seed": PERMUTATION_SEED} if args.dryrun else {}
    record = {"protocol_sha256": protocol_sha, "pinned": PROTOCOL_SHA256, "mode": name, **extra,
              "storage": {"probabilities.csv": sha256_file(storage / "probabilities.csv"), "root": str(storage)},
              "code": {"digest": protocol["code"]["digest"]}, "git": detector_records.git_state(),
              "timings_s": {"read_out": seconds}, "started": started.isoformat(timespec="seconds"),
              "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": files(out)}
    write_json(out / "complete.json", record)
    print(json.dumps({"mode": name, "timings_s": seconds} | extra, indent=2, default=float))


if __name__ == "__main__":
    main()
