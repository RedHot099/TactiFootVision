"""xg-temporal read-out of arms R, S and T: the synthetic tracer (ticket 01), the label-free controls (ticket 03), the
permutation control and the single real run (tickets 03 and 04, after the frozen protocol). CPU only.

    ./run.sh --no-sync python scripts/xgt_run.py --synthetic --source dinov2 [--start -1600 --o-max -600] [--k 32]
        [--replicates 1] [--seed 0] [--control] [--out results/xgt_synthetic_<source>_<start>_<o_max>]
    ./run.sh --no-sync python scripts/xgt_run.py --controls --remedy [--jobs 8]   # -> results/xgt/controls/label_free_remedy
    ./run.sh --no-sync python scripts/xgt_run.py --controls [--remedy] [--source ...] [--replicates ...]
        [--simulations ...] --out <trial folder>
    ./run.sh --no-sync python scripts/xgt_run.py --permutation       # after the protocol commit -> results/xgt/controls
    ./run.sh --no-sync python scripts/xgt_run.py --real              # ticket 04 -> results/xgt/real

Common steps: verify the plan and the source (`xgpoc.temporal.inputs`), build the population and the window, run the
leak assertions (`xgpoc.temporal.provenance`; the run stops when one fails), read the window embeddings through a
memmap and build the fold-wise designs of R, S and T once (`xgpoc.temporal.features`). Every mode refuses an existing
output, so each is made once, and runs inside `no_statsbomb_reads()`.

--synthetic    per replicate y* ~ Bernoulli(sigmoid(z_g_combo)), every arm out of fold, ΔLL T - R, T - S (Holm) and
               S - R with the match bootstrap (`xgpoc.temporal.readout`). Writes population.csv, replicates.csv,
               summary.json, complete.json.
--controls     the label-free gates and reports of `xgpoc.temporal.controls` (fits in `--jobs` worker processes, one
               BLAS thread each; T under `readout.T_MODEL`, the C9 Q2 remedy). With `--remedy` the simulations draw
               from `controls.REMEDY_SEEDS` (the remedy's fresh seeds), without it from the first attempt's `SEEDS`.
               A run is the protocol's controls (`protocol_controls` in its record) only with `--remedy`, the main
               source g_xgv, the main window, k = 32, 200 replicates and every simulation (the descriptive
               rich_nulls_-0.002 may be left out, C8, and the record names it in `omitted_simulations`); it goes to
               results/xgt/controls/label_free_remedy (`protocol.LABEL_FREE`), next to the failed first attempt
               (results/xgt/controls/label_free, kept). Anything else is a trial run and is refused there. Writes
               population.csv, <simulation>.csv, controls.json, complete.json (`label_free_passed`, `remedy`, `seeds`).
--permutation  refuses unless the protocol has the SHA-256 pinned here (`PROTOCOL_SHA256`) and is committed as it is
               (`git show HEAD:research/xgt_protocol.json`, `check_committed`), its frozen part equals
               `xgpoc.temporal.protocol.frozen()`, its code digest equals the current package's, its population equals
               the rebuilt one, the xgv extraction's complete.json is the one it pins, and the label-free controls'
               record is the one it pins. Then reads goal inside the label seal (`xgpoc.temporal.labels`), runs the
               read-out on the labels permuted within matches and writes
               results/xgt/controls/permutation.json and complete.json with every gate and `all_passed`.
--real         the same protocol checks, then refuses unless results/xgt/controls/complete.json matches its files, was
               written under this protocol and code and has `all_passed: true` (`check_controls`). Then reads the
               categories' E3 columns, refused unless their files have the protocol's SHA-256 (`check_e3`), reads goal
               and on_target inside the label seal and writes the read-out (`real_run`): results/xgt/real/summary.json
               (scores, ΔLL, Holm, claims, readability veto, chosen C, the leak curve's rules on the shots common to
               every leak-curve window and, descriptive, on each window's own population, the controls' power
               curve), categories.csv, leak_curve.csv (both populations) and complete.json; the out-of-fold probabilities (no
               label) go to outputs/xgt_real/probabilities.csv, named in complete.json by path and SHA-256.
"""

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX
from xgpoc.stack import stats as ss
from xgpoc.temporal import categories as tc
from xgpoc.temporal import controls as tco
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import labels as tl
from xgpoc.temporal import protocol as tpr
from xgpoc.temporal import provenance as tp
from xgpoc.temporal import readout as tr
from xgpoc.temporal import records as tr_records

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256: str | None = None          # pinned after research/xgt_protocol.json is committed (ticket 03)
REAL_OUT = Path("results/xgt/real")          # relative to ROOT
REAL_LARGE = Path("outputs/xgt_real")
MODES = ("synthetic", "controls", "permutation", "real")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    for name in MODES:
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--source", choices=sorted(ti.SOURCES), default=None)
    parser.add_argument("--start", type=int, default=ti.START)
    parser.add_argument("--o-max", type=int, default=ti.O_MAX)
    parser.add_argument("--control", action="store_true", help="a leakage-control row (o_max > -600), never a candidate")
    parser.add_argument("--k", type=int, default=tf.K)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="--synthetic only")
    parser.add_argument("--jobs", type=int, default=tco.default_jobs(), help="--controls: worker processes")
    parser.add_argument("--simulations", nargs="+", choices=tco.SIMULATIONS, default=list(tco.SIMULATIONS))
    parser.add_argument("--remedy", action="store_true", help="--controls: the C9 Q2 remedy's run on REMEDY_SEEDS")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    args.mode = next((name for name in MODES if getattr(args, name)), None)
    if args.mode is None:
        parser.error(f"choose one of {['--' + m for m in MODES]}")
    if args.remedy and args.mode != "controls":
        parser.error("--remedy belongs to --controls")
    if args.mode in ("permutation", "real"):
        if args.source not in (None, tpr.MAIN_SOURCE) or (args.start, args.o_max, args.k) != (ti.START, ti.O_MAX, tf.K) \
                or args.control or args.out is not None:
            parser.error(f"--{args.mode} runs the frozen protocol only: no --source, window, --k or --out")
        args.source = tpr.MAIN_SOURCE
        return args
    if args.source is None:
        if args.mode == "synthetic":
            parser.error("--synthetic needs --source")
        args.source = tpr.MAIN_SOURCE
    if args.replicates is None:
        args.replicates = 1 if args.mode == "synthetic" else tco.N_REPLICATES
    if args.k < 1 or args.replicates < 1:
        parser.error("--k and --replicates must be at least 1")
    if args.mode == "synthetic" and args.out is None:
        suffix = "_control" if args.control else ""
        args.out = ROOT / f"results/xgt_synthetic_{args.source}_{args.start}_{args.o_max}{suffix}"
    if args.mode == "controls":
        args.protocol_controls = protocol_controls(args)
        if args.out is None:
            args.out = ROOT / tpr.LABEL_FREE
        if Path(args.out).resolve() == (ROOT / tpr.LABEL_FREE).resolve() and not args.protocol_controls:
            parser.error(f"a trial run of the controls (without --remedy, or source, window, k, replicates or "
                         f"simulations other than the protocol's) needs its own --out, not {tpr.LABEL_FREE}")
        if Path(args.out).resolve() == (ROOT / tpr.FIRST_ATTEMPT).resolve():
            parser.error(f"{tpr.FIRST_ATTEMPT} holds the first attempt; it is kept as it is")
    return args


def omitted_simulations(simulations) -> list[str] | None:
    """The simulations left out of `tco.SIMULATIONS` (in run order), or None when `simulations` is not
    `tco.SIMULATIONS` in its order with only `tco.DROPPABLE` ones left out."""
    omitted = [s for s in tco.SIMULATIONS if s not in simulations]
    if list(simulations) != [s for s in tco.SIMULATIONS if s not in omitted] or not set(omitted) <= set(tco.DROPPABLE):
        return None
    return omitted


def protocol_controls(args: argparse.Namespace) -> bool:
    """The run is the protocol's label-free controls (module docstring): the remedy's run, every simulation, or all but
    the descriptive `tco.DROPPABLE` (C8: the first to drop when CPU time binds), recorded as omitted."""
    return (args.remedy and args.source == tpr.MAIN_SOURCE and (args.start, args.o_max, args.k) == (ti.START, ti.O_MAX, tf.K)
            and not args.control and args.replicates == tco.N_REPLICATES
            and omitted_simulations(args.simulations) is not None)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=tr_records.json_default, allow_nan=False) + "\n")


class Stages:
    def __init__(self):
        self.clock = self.last = time.time()
        self.seconds = {}

    def __call__(self, name: str) -> None:
        self.seconds[name] = round(time.time() - self.last, 1)
        self.last = time.time()

    def record(self) -> dict:
        return {"stages_seconds": self.seconds, "seconds": round(time.time() - self.clock, 1),
                "peak_rss_gib": round(tr_records.peak_rss_gib(), 2), "device": "cpu"}


def prepared(source: str, window: ti.Window, k: int, plan: pd.DataFrame, stages: Stages,
             restrict_to=None, files: ti.SourceFiles | None = None) -> tuple[ti.Population, dict, dict, tf.DesignData, dict]:
    """(population, its arms' columns, the provenance record, the design data, {designs, pca, files}), the run
    stopped when a leak assertion fails; with `restrict_to` the population is restricted to those shot ids first;
    `files` (the verified source) is verified again when not given."""
    population, files = ti.build_population(ti.SOURCES[source], window, plan, files=files)
    if restrict_to is not None:
        population = ti.restrict(population, restrict_to)
    columns = tf.arm_columns(window, k)
    provenance = tp.control(population, columns, plan)
    if not provenance["passed"]:
        raise SystemExit(f"provenance failed: {[c for c in provenance['checks'] if not c['passed']]}")
    stages(f"population_{window.label()}")
    embeddings, position = ti.load_embeddings(files, population.refs)
    data = tf.DesignData(window, population.z, population.folds, position, embeddings, k)
    designs, pca = tf.fold_designs(data)
    stages(f"designs_{window.label()}")
    return population, columns, provenance, data, {"designs": designs, "pca": pca, "files": files}


def population_csv(population: ti.Population, path: Path) -> None:
    population.table.reset_index()[["shot_id", "game", "fold", "frames", "complete"]].to_csv(path, index=False)


def completion(out: Path, base: dict, started: datetime, extra_files: dict | None = None) -> dict:
    """complete.json of `out`: `base`, the code (package digest and the run's closure), git, time and the hashes of
    every file under `out` (recursively) plus `extra_files`."""
    files = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p != out / "complete.json"}
    record = base | {"code": {"package": tr_records.package_hashes(), "run": tr_records.code_hashes(Path(__file__))},
                     "git": records.git_state(),
                     "run_time": {"started": started.isoformat(timespec="seconds"),
                                  "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                     "files": files | (extra_files or {})}
    write_json(out / "complete.json", record)
    return record


def verified_record(folder: Path) -> dict:
    """A folder's complete.json, refused unless every file it names (relative to the folder) matches."""
    path = folder / "complete.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    record = json.loads(path.read_text())
    for name, digest in record["files"].items():
        file = Path(name) if Path(name).is_absolute() else folder / name
        if not file.exists() or sha256_file(file) != digest:
            raise SystemExit(f"{file} does not match the completion record {path}")
    return record


# ------------------------------------------------------------------ --synthetic
def replicate_row(i: int, y: np.ndarray, result: dict) -> dict:
    row = {"replicate": i, "events": int(y.sum())}
    for name, d in result["deltas"].items():
        row |= {f"delta_{name}": d["delta_ll"], f"lo95_{name}": d["lo95"], f"hi95_{name}": d["hi95"],
                f"p_{name}": d["p_one_sided"]}
    for name, r in result["readings"].items():
        row |= {f"holm_{name}": r["holm_rejected"]}
    return row


def run_synthetic(args: argparse.Namespace, plan: pd.DataFrame, stages: Stages, started: datetime) -> None:
    window = ti.Window(args.start, args.o_max, args.control)
    population, columns, provenance, _, built = prepared(args.source, window, args.k, plan, stages)
    weights = ss.bootstrap_weights(population.games)
    rng = np.random.default_rng(args.seed)
    rows, results, chosen = [], [], {}
    for i in range(args.replicates):
        y = tr.synthetic_outcomes(population.z, rng)
        probabilities, chosen[str(i)] = tr.out_of_fold(built["designs"], y)
        result = tr.contrasts(y, probabilities, weights)
        rows.append(replicate_row(i, y, result))
        results.append(result)
        print(f"replicate {i}: " + ", ".join(f"{n} {d['delta_ll']:+.5f} [{d['lo95']:+.5f}; {d['hi95']:+.5f}]"
                                              for n, d in result["deltas"].items()), flush=True)
    stages("fits")
    args.out.mkdir(parents=True)
    population_csv(population, args.out / "population.csv")
    pd.DataFrame(rows).to_csv(args.out / "replicates.csv", index=False, float_format="%.6g")
    summary = {"label": "synthetic tracer: y* ~ Bernoulli(sigmoid(z_g_combo)); no label read",
               "source": args.source, "window": window_record(window), "k": args.k, "seed": args.seed,
               "replicates": args.replicates, "model": {"name": tr.MODEL.name, "c_grid": list(tr.C_GRID)},
               "population": population.counts, "population_csv_sha256": sha256_file(args.out / "population.csv"),
               "columns": {arm: {"n": len(c), "names": c} for arm, c in columns.items()},
               "provenance": provenance, "pca": built["pca"], "chosen_c": chosen,
               "first_replicate": {k: results[0][k] for k in ("deltas", "readings", "scores")},
               "mean_delta": {n: float(np.mean([r[f"delta_{n}"] for r in rows])) for n in results[0]["deltas"]},
               "holm_rejection_rate": {n: float(np.mean([r[f"holm_{n}"] for r in rows])) for n in tr.CONFIRMATORY},
               "run": stages.record()}
    write_json(args.out / "summary.json", summary)
    completion(args.out, {"mode": "synthetic", "inputs": population.inputs}, started)
    print(json.dumps({"population": {k: population.counts[k] for k in ("shots", "matches", "complete_shots")},
                      "mean_delta": summary["mean_delta"], "holm_rejection_rate": summary["holm_rejection_rate"],
                      "run": summary["run"]}, indent=1))


def window_record(window: ti.Window) -> dict:
    return {"start": window.start, "o_max": window.o_max, "control": window.control, "offsets": list(window.offsets),
            "middle": window.middle}


# ------------------------------------------------------------------ --controls
def run_controls(args: argparse.Namespace, plan: pd.DataFrame, stages: Stages, started: datetime) -> None:
    window = ti.Window(args.start, args.o_max, args.control)
    population, columns, provenance, data, built = prepared(args.source, window, args.k, plan, stages)
    truths = tco.Truths.build(data, population.games)
    stages("truths")
    weights = ss.bootstrap_weights(population.games)
    print(f"controls: {population.counts['shots']} shots, {population.counts['matches']} matches, "
          f"{args.replicates} replicates, {args.jobs} worker processes", flush=True)
    seeds = tco.REMEDY_SEEDS if args.remedy else tco.SEEDS
    with tco.Fitter(built["designs"], args.jobs) as fitter:
        frames, summary = tco.run_label_free(fitter, weights, truths, provenance, args.replicates, seeds,
                                             simulations=tuple(args.simulations), log=lambda m: print(m, flush=True))
    stages("simulations")
    args.out.mkdir(parents=True)
    population_csv(population, args.out / "population.csv")
    for name, frame in frames.items():
        frame.to_csv(args.out / f"{name}.csv", index=False, float_format="%.9g")
    summary |= {"source": args.source, "window": window_record(window), "k": args.k,
                "protocol_controls": args.protocol_controls, "remedy": args.remedy, "jobs": args.jobs,
                "model": {"name": tr.MODEL.name, "t_model": tr.T_MODEL, "c_grid": list(tr.C_GRID)},
                "pca": built["pca"], "run": stages.record()}
    write_json(args.out / "controls.json", summary)
    gates = {g: {"passed": summary["gates"][g]["passed"], "criterion": summary["gates"][g]["criterion"]}
             for g in tco.LABEL_FREE_GATES}
    completion(args.out, {"mode": "controls", "source": args.source, "protocol_controls": args.protocol_controls,
                          "remedy": args.remedy, "seeds": dict(seeds), "t_model": tr.T_MODEL,
                          "simulations": list(args.simulations),
                          "omitted_simulations": omitted_simulations(args.simulations),
                          "settings": tco.settings(), "label_free_passed": summary["label_free_passed"], "gates": gates,
                          "columns": columns, **population.record()}, started)
    print(json.dumps({"gates": gates, "label_free_passed": summary["label_free_passed"],
                      "power_curve": summary["reports"]["power_curve"], "timing_seconds": summary["timing_seconds"],
                      "run": summary["run"]}, indent=1, default=tr_records.json_default))


# ------------------------------------------------------------------ the frozen protocol's gate
def opened_protocol() -> dict:
    """The protocol at the pinned SHA-256 (module docstring), refused unless its frozen part and code digest are the
    current code's."""
    path = ROOT / tpr.PROTOCOL
    try:
        protocol = tpr.load(path, PROTOCOL_SHA256)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from None
    if protocol.get("draft", True):
        raise SystemExit(f"{path} is a draft, not the frozen protocol")
    if not json_equal(protocol["frozen"], tpr.frozen()):
        raise SystemExit("the protocol's frozen part differs from xgpoc.temporal.protocol.frozen()")
    if protocol["data"]["code"]["digest"] != tr_records.package_hashes()["digest"]:
        raise SystemExit("the code differs from the code the protocol pins; re-freeze after a code change")
    check_committed(path)
    return protocol


def committed(path: Path) -> bytes | None:
    """`path` as the checkout's HEAD commit holds it (None when HEAD does not hold it or there is no repository)."""
    run = subprocess.run(["git", "-C", str(path.parent), "show", f"HEAD:./{path.name}"], capture_output=True)
    return run.stdout if run.returncode == 0 else None


def check_committed(path: Path) -> None:
    """The protocol file is the one the HEAD commit holds, byte for byte (so its pinned SHA-256 is a committed one)."""
    if committed(path) != path.read_bytes():
        raise SystemExit(f"{path} is not committed as it is (git show HEAD:{tpr.PROTOCOL} differs or is missing)")


def json_equal(a, b) -> bool:
    return json.loads(json.dumps(a, default=tr_records.json_default)) == json.loads(json.dumps(b, default=tr_records.json_default))


def check_population(population: ti.Population, frozen: dict, what: str) -> None:
    if not json_equal(population.record(), {k: frozen[k] for k in ("inputs", "counts", "shot_list_sha256")}):
        raise SystemExit(f"the {what} population's inputs, counts or shot list differ from the protocol's")


def check_label_free(protocol: dict) -> dict:
    """The label-free controls' record, refused unless it is the one the protocol pins."""
    folder = ROOT / tpr.LABEL_FREE
    record = verified_record(folder)
    if sha256_file(folder / "complete.json") != protocol["data"]["label_free_controls"]["complete_sha256"]:
        raise SystemExit("the label-free controls' record is not the one the protocol pins")
    return record


def check_controls(folder: Path, protocol_sha256: str, package_digest: str) -> dict:
    """The controls' completion record, refused unless every file matches it, it was written under this protocol and
    this package digest, it holds every gate and every gate passed."""
    record = verified_record(folder)
    if record.get("protocol_sha256") != protocol_sha256 or record["code"]["package"]["digest"] != package_digest:
        raise SystemExit("the controls ran under another protocol or other code")
    if set(record.get("controls", {})) != set(tco.GATES) or record.get("all_passed") is not True \
            or not all(c["passed"] for c in record["controls"].values()):
        raise SystemExit("not every control passed: the real run does not start")
    return record


def main_population(protocol: dict, plan: pd.DataFrame, stages: Stages):
    built = prepared(tpr.MAIN_SOURCE, ti.Window(), tf.K, plan, stages)
    check_population(built[0], protocol["data"]["main"], "main")
    if not json_equal(built[1], protocol["data"]["main"]["columns"]):
        raise SystemExit("the arms' columns differ from the protocol's")
    record = built[4]["files"].source.folder / ti.RECORD
    if sha256_file(record) != protocol["data"]["main"]["source_files"]["complete_json"]["sha256"]:
        raise SystemExit(f"{record} differs from the extraction record the protocol pins")
    return built


# ------------------------------------------------------------------ --permutation
def run_permutation(protocol: dict, plan: pd.DataFrame, stages: Stages, started: datetime) -> None:
    out = ROOT / tpr.CONTROLS
    if (out / "complete.json").exists() or (out / "permutation.json").exists():
        raise SystemExit(f"{out} already holds the permutation; every run is made once")
    label_free = check_label_free(protocol)
    population, _, _, _, built = main_population(protocol, plan, stages)
    labels = tl.load_labels(population, INDEX, ti.PLAN_CSV)
    stages("labels")
    result = tco.permutation(built["designs"], ss.bootstrap_weights(population.games), labels.goal, population.games)
    stages("permutation")
    write_json(out / "permutation.json", result | {"run": stages.record()})
    controls = {g: {"passed": bool(label_free["gates"][g]["passed"]), "criterion": label_free["gates"][g]["criterion"]}
                for g in tco.LABEL_FREE_GATES}
    controls["permutation"] = {"passed": result["passed"], "criterion": result["criterion"]}
    all_passed = all(c["passed"] for c in controls.values())
    lf = ROOT / tpr.LABEL_FREE / "complete.json"
    completion(out, {"mode": "permutation", "protocol_sha256": PROTOCOL_SHA256, "controls": controls,
                     "all_passed": all_passed, "label_free": {"path": str(tpr.LABEL_FREE), "complete_sha256": sha256_file(lf)}},
               started)
    print(json.dumps({"controls": controls, "all_passed": all_passed}, indent=1))


# ------------------------------------------------------------------ --real (ticket 04)
def check_e3(hashes: dict, recorded: dict) -> None:
    """The categories' E3 files, refused unless their SHA-256 are the ones the protocol pins (paths may differ)."""
    if {k: v["sha256"] for k, v in hashes.items()} != {k: v["sha256"] for k, v in recorded.items()}:
        raise SystemExit("the E3 files differ from the ones the protocol pins")


def read_window(designs: dict, y: np.ndarray, weights: np.ndarray) -> tuple[dict, dict, dict]:
    probabilities, chosen = tr.out_of_fold(designs, y)
    return probabilities, chosen, tr.contrasts(y, probabilities, weights)


def leak_populations(protocol: dict, plan: pd.DataFrame, main: ti.Population, files: ti.SourceFiles) -> tuple[dict, dict]:
    """(o_max -> the leak-curve window's population ∩ P_G, o_max -> its population on the common shots), each checked
    against the protocol (data.cells, data.leak_common), label-free."""
    cells = {(c["start"], c["o_max"]): c for c in protocol["data"]["cells"]}
    windows = tpr.leak_populations(tpr.MAIN_SOURCE, plan, main, files)
    for o_max, population in windows.items():
        check_population(population, cells[(-1600, o_max)], f"cell -1600/{o_max}")
    common = tpr.common_populations(windows)
    frozen = protocol["data"]["leak_common"]
    for o_max, population in common.items():
        check_population(population, frozen["cells"][str(o_max)], f"common leak population -1600/{o_max}")
    return windows, common


def curve_entry(population: ti.Population, plan: pd.DataFrame, stages: Stages, files: ti.SourceFiles, labels: tl.Labels,
                weights: np.ndarray, main_ids: pd.Index) -> tuple[dict, dict]:
    """One leak-curve point on `population` (a checked subset of P_G): the arms refitted on its shots, `curve_point`
    with the main population's match draws restricted to them, and every arm's chosen C per outer fold."""
    window = population.window
    built, _, _, _, cell = prepared(tpr.MAIN_SOURCE, window, tf.K, plan, stages, population.table.index, files)
    if not json_equal(built.record(), population.record()):
        raise SystemExit(f"the leak-curve population at {window.label()} changed while the run read it")
    position = main_ids.get_indexer(built.table.index)
    if (position < 0).any():
        raise SystemExit("a leak-curve shot is not in the main population")
    cell_labels = labels.aligned(built)
    probabilities, chosen = tr.out_of_fold(cell["designs"], cell_labels.goal)
    return tr.curve_point(cell_labels.goal, probabilities, weights[:, position]), chosen


def real_run(protocol: dict, plan: pd.DataFrame, stages: Stages, out: Path, large: Path) -> dict:
    """The read-out of ticket 04 (module docstring); returns the summary. Every leak-curve point uses the main
    population's match draws restricted to its shots, so the jump rule's excess has a paired interval; the jump rule
    and the method's positive control read the curve on the shots common to every window (`leak`), the curve on each
    window's own population is descriptive (`leak_by_window`)."""
    population, columns, provenance, _, built = main_population(protocol, plan, stages)
    e3, e3_hashes = tc.e3_inputs(population.table.index, ROOT / protocol["data"]["e3"]["source"])     # label-free first
    check_e3(e3_hashes, protocol["data"]["e3"]["hashes"])
    windows, common = leak_populations(protocol, plan, population, built["files"])
    stages("label_free_checks")
    labels = tl.load_labels(population, INDEX, ti.PLAN_CSV)
    stages("labels")
    weights = ss.bootstrap_weights(population.games)
    probabilities, chosen, result = read_window(built["designs"], labels.goal, weights)
    veto = tr.readability_veto(labels.goal, labels.on_target, probabilities["T"], probabilities["S"], population.games)
    categories = tc.category_table(e3, labels.goal, probabilities, weights)
    stages("main_read_out")
    main_ids, curves, rows = population.table.index, {"window": {}, "common": {}}, []
    for scope, populations in (("window", windows), ("common", common)):
        for o_max in tpr.LEAK_CURVE:
            cell = populations[o_max]
            if scope == "window" and o_max == ti.O_MAX:
                curves[scope][o_max], cell_chosen = tr.curve_point(labels.goal, probabilities, weights), chosen
            else:
                curves[scope][o_max], cell_chosen = curve_entry(cell, plan, stages, built["files"], labels, weights, main_ids)
            for name, d in curves[scope][o_max].items():
                rows.append({"population": scope, "start": -1600, "o_max": o_max, "control": o_max > tp.LEAK_LIMIT,
                             "contrast": name, "shots": cell.counts["shots"],
                             **{k: d[k] for k in ("delta_ll", "lo95", "hi95", "bound_one_sided_95", "p_one_sided")},
                             "near_reference_T": tr.near_reference(cell_chosen, "T")})
    leak = tr.leak_rules(curves["common"]) | {"population": "common", "shots": common[ti.O_MAX].counts["shots"]}
    leak_by_window = tr.leak_rules(curves["window"]) | {"population": "window", "descriptive": True}
    claims = tr.claims(result, veto, chosen, leak)
    stages("leak_curve")
    large.mkdir(parents=True)
    pd.DataFrame({"shot_id": population.table.index, "game": population.games, "fold": population.folds}
                 | {f"p_{arm}": p for arm, p in probabilities.items()}).to_csv(large / "probabilities.csv", index=False,
                                                                                  float_format="%.9g")
    categories.to_csv(out / "categories.csv", index=False, float_format="%.9g")
    pd.DataFrame(rows).to_csv(out / "leak_curve.csv", index=False, float_format="%.9g")
    label_free = json.loads((ROOT / tpr.LABEL_FREE / "controls.json").read_text())
    summary = {"scores": result["scores"], "deltas": result["deltas"], "readings": result["readings"], "claims": claims,
               "readability_veto": veto, "chosen_c": chosen, "population": population.counts, "provenance": provenance,
               "leak": leak, "leak_by_window": leak_by_window, "e3": e3_hashes,
               "power_curve": label_free["reports"]["power_curve"], "pca": built["pca"], "run": stages.record()}
    write_json(out / "summary.json", summary)
    return summary


def run_real(protocol: dict, plan: pd.DataFrame, stages: Stages, started: datetime) -> None:
    out, large = ROOT / REAL_OUT, ROOT / REAL_LARGE
    if out.exists() or large.exists():
        raise SystemExit(f"{out} or {large} exists; the real run is made once")
    controls = check_controls(ROOT / tpr.CONTROLS, PROTOCOL_SHA256, protocol["data"]["code"]["digest"])
    out.mkdir(parents=True)
    summary = real_run(protocol, plan, stages, out, large)
    probabilities = large / "probabilities.csv"
    completion(out, {"mode": "real", "protocol_sha256": PROTOCOL_SHA256,
                     "controls_record_sha256": sha256_file(ROOT / tpr.CONTROLS / "complete.json"),
                     "controls_passed": controls["all_passed"], "claims": summary["claims"],
                     "readability_veto": summary["readability_veto"]["vetoed"]},
               started, {str(probabilities): sha256_file(probabilities)})
    print(json.dumps({"deltas": {n: d["delta_ll"] for n, d in summary["deltas"].items()}, "claims": summary["claims"],
                      "veto": summary["readability_veto"], "leak": summary["leak"]}, indent=1,
                     default=tr_records.json_default))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in ("synthetic", "controls") and args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    if args.mode == "synthetic":
        tp.check_window(args.o_max, args.control)     # before any input is read
    protocol = opened_protocol() if args.mode in ("permutation", "real") else None
    started, stages = datetime.now(UTC), Stages()
    with no_statsbomb_reads():
        plan = a1.verify_plan(INDEX)
        if args.mode == "synthetic":
            run_synthetic(args, plan, stages, started)
        elif args.mode == "controls":
            run_controls(args, plan, stages, started)
        elif args.mode == "permutation":
            run_permutation(protocol, plan, stages, started)
        else:
            run_real(protocol, plan, stages, started)


if __name__ == "__main__":
    main()
