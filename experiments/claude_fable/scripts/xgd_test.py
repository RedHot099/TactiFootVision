"""The single test of the xg-detector study, frozen in ticket 06 and run once in ticket 07: the 10-model A1 ensemble
against the 10-model G-combo on the non-penalty shots of the 172 held-out matches (research/xg_detector_test_rule.md).

    ./run.sh python scripts/xgd_test.py --freeze                       # ticket 06 only: write the hash list
    ./run.sh python scripts/xgd_test.py --check --expect-hashlist SHA  # verify everything, score nothing (GPU path check)
    ./run.sh python scripts/xgd_test.py --expect-hashlist SHA          # the test (ticket 07), once

SHA is the sha256 of research/xg_detector_test_hashes.json quoted in ticket 06's Comments. Before anything is scored the
script refuses unless:
* the hash list hashes to SHA and is the committed, unmodified file;
* every pinned file hashes as listed: the rule (text and parameters), the power record, the A1 and G-combo bundles,
  completion records and ensemble calibrations, the A1 replay, G-combo's test predictions, the sanitised D3 table, the
  split plan and folds, the index, the overlay mask and the pretrained backbone;
* the code this process loaded from the checkout is exactly the pinned code closure, byte for byte;
* the t0 frames of the test population have the pinned fingerprint, and the sanitised D3 table has both offsets of
  every population shot;
* every variant has complete records: the ten A1 runs (folds 0-4 x seeds 0, 1) with their files and a passed replay of
  all ten, A1's ensemble.json naming exactly those bundles and temperatures, G-combo's ensemble.json and bundles, and
  G-combo's test predictions covering the test population;
* the path check passes: A1 fold 0 seed 0 rescores the first 16 shots of its held-out (training-pool) fold in the stored
  order within 1e-4 of its stored logits;
* results/xgd_test does not exist yet (the test runs once; a rerun needs the owner's decision).

The run, in this order:
1. Inside `no_statsbomb_reads()` and the held-out label seal: the test population (label-free index columns), the ten
   A1 bundles scored on it (t0 frame, each bundle's training-fold standardiser on the sanitised D3 table, batch 8),
   A1's ensemble probability (temperature-scaled logits averaged, its Platt, prior correction, penalty routing),
   G-combo's stored test probabilities. Both arms' predictions are written with a record; the loaded code is checked
   against the closure again.
2. Only then the labels of the test shots are read (index `goal`), and the frozen rule is applied (`testrule`):
   shot-weighted ΔLL = A1 - G-combo with p clipped at the rule's clip, the paired match bootstrap (the rule's draws and
   seed), the 95% interval (success / worse), the 90% interval from the same draws (descriptive TOST at δ), the reading
   label and the smallest equivalence margin. Writes results/xgd_test/{predictions.csv, a1_members.csv,
   predictions_record.json, result.json, complete.json}.
"""

import argparse
import gc
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgd_approach1_report as dev
from run_xgc_g import OVERLAY, frame_fingerprint
from xgd_a1_ensemble import MEMBERS, POWER, member_records
from xgd_approach1 import SANITISED, check_gpu_free
from xgd_gcombo_test import held_out_shots, read_ensemble
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal, testrule
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.models.g_bundle import ensemble_probability, pretrained_weight_file
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = records.CHECKOUT
HASHLIST = ROOT / "research/xg_detector_test_hashes.json"
OUT = ROOT / "results/xgd_test"
A1_ENSEMBLE = ROOT / "results/xgd_a1_ensemble"
G_TEST = ROOT / "results/xgd_gcombo_test"
PATH_SHOTS, PATH_TOLERANCE = 16, 1e-4


class Refused(SystemExit):
    """The frozen test refuses to run."""


def pinned_files(index: Path = INDEX) -> list[Path]:
    """Every data file the test depends on, in a fixed order."""
    files = [testrule.RULE_MD, testrule.RULE_JSON, POWER / "power.json", POWER / "complete.json"]
    a1_root = a1.results_dir("A1")
    for k, s in MEMBERS:
        run = a1_root / f"fold{k}" / f"seed{s}"
        files += [run / "bundle.pt", run / "complete.json", run / "training.json", run / f"{a1.run_name('A1')}_valid_predictions.csv"]
    files += [a1_root / "replay_ensemble.json"] + [A1_ENSEMBLE / n for n in ("ensemble.json", "oof.csv", "comparison.json", "complete.json")]
    for k, s in MEMBERS:
        files += [dev.G_COMBO / f"fold{k}" / f"seed{s}" / n for n in ("bundle.pt", "complete.json")]
    files += [dev.G_COMBO / "ensemble.json"] + [G_TEST / n for n in ("predictions.csv", "unpaired_members.csv", "checks.json", "complete.json")]
    files += [SANITISED, SANITISED.parent / "complete.json"]
    files += [a1.PLAN_DIR / n for n in ("plan.csv", "plan.json", "complete.json")] + [a1.FOLDS, index, ROOT / OVERLAY]
    files.append(pretrained_weight_file("dinov2_l"))
    return files


def name_of(path: Path) -> str:
    """Checkout-relative name of a file inside the checkout (symlinked folders kept as written), else the absolute path."""
    path = Path(path)
    return str(path.relative_to(CHECKOUT)) if path.is_relative_to(CHECKOUT) else str(path)


def path_of(name: str) -> Path:
    return Path(name) if Path(name).is_absolute() else CHECKOUT / name


def loaded_code() -> dict[str, str]:
    """sha256 of every Python file of the checkout loaded in this process."""
    files = {Path(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    files = {f for f in files if f.is_absolute() and f.suffix == ".py" and f.is_relative_to(CHECKOUT)}
    return {name_of(f): sha256_file(f) for f in sorted(files)}


def test_population(index: Path, plan: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """shot_id, game, role, is_penalty, frame_path of the rule's population, sorted by shot_id; label-free columns only.
    Refused unless its size is the rule's."""
    shots = held_out_shots(index, plan)
    if sorted(shots.role.unique()) != sorted(rule["population"]["roles"]):
        raise Refused(f"held-out roles {sorted(shots.role.unique())} differ from the rule's")
    population = shots[~shots.is_penalty].reset_index(drop=True)
    if len(population) != rule["population"]["expected_shots"] or population.game.nunique() != rule["population"]["matches"]:
        raise Refused(f"test population {len(population)} shots / {population.game.nunique()} matches differs from the rule's")
    return population


def variant_problems(rule: dict, population: pd.DataFrame | None) -> list[str]:
    """Why the variants' records are not complete (an empty list when they are)."""
    problems = []
    try:
        runs = member_records(a1.results_dir("A1"))
        ensemble = json.loads((A1_ENSEMBLE / "ensemble.json").read_text())
        listed = {(m["fold"], m["seed"]): (m["sha256"], m["temperature"]) for m in ensemble["members"]}
        for key, record in runs.items():
            training = json.loads((a1.results_dir("A1") / f"fold{key[0]}" / f"seed{key[1]}" / "training.json").read_text())
            if listed.get(key) != (record["bundle"]["sha256"], training["temperature"]):
                problems.append(f"A1 ensemble.json does not list run {key} with its bundle and temperature")
        if set(listed) != set(runs):
            problems.append("A1 ensemble.json lists other members than the ten runs")
        records.verified_file(A1_ENSEMBLE / "ensemble.json")
    except (OSError, KeyError, ValueError) as e:
        problems.append(f"A1 records incomplete: {e}")
    try:
        g = read_ensemble(dev.G_COMBO)
        for m in g["members"]:
            if sha256_file(dev.G_COMBO / f"fold{m['fold']}" / f"seed{m['seed']}" / "bundle.pt") != m["sha256"]:
                problems.append(f"G-combo bundle f{m['fold']}s{m['seed']} differs from ensemble.json")
        if population is not None:
            g_combo_predictions(population)
    except (OSError, KeyError, ValueError) as e:
        problems.append(f"G-combo records incomplete: {e}")
    if rule.get("power") is None:
        problems.append("the rule states no power")
    return problems


def g_combo_predictions(population: pd.DataFrame) -> pd.DataFrame:
    """The population's G-combo test probabilities (shot_id, p), checked against their completion record."""
    records.verified_file(G_TEST / "predictions.csv")
    stored = pd.read_csv(G_TEST / "predictions.csv", usecols=["shot_id", "game", "is_penalty", "p"])
    out = population[["shot_id", "game"]].merge(stored, on=["shot_id", "game"], how="left", validate="one_to_one")
    if out.p.isna().any():
        raise ValueError(f"G-combo's test predictions miss {int(out.p.isna().sum())} population shots")
    if out.is_penalty.astype(bool).any():
        raise ValueError("a population shot is a penalty in G-combo's test predictions")
    return out[["shot_id", "p"]]


def sanitised_problems(population: pd.DataFrame, sanitised: Path = SANITISED) -> list[str]:
    """Why the sanitised D3 table cannot give A1 its input for every population shot at both offsets."""
    rows = pd.read_parquet(sanitised, columns=["shot_id", "offset_ms"])
    rows = rows[rows.shot_id.isin(population.shot_id)]
    have = set(zip(rows.shot_id, rows.offset_ms.astype(int), strict=True))
    missing = len(population) * len(a1.OFFSETS) - len(have & {(s, o) for s in population.shot_id for o in a1.OFFSETS})
    return [f"the sanitised D3 table misses {missing} (shot, offset) rows of the population"] if missing else []


def git_clean(path: Path) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=CHECKOUT, capture_output=True).returncode == 0
    unchanged = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=CHECKOUT).returncode == 0
    return tracked and unchanged


def verify(hashlist: Path, expected: str, code: dict[str, str] | None, population: pd.DataFrame | None, *,
           check_git: bool = True) -> dict:
    """The frozen state, refused (`Refused`, every problem listed) unless the hash list hashes to `expected` (and is
    committed and clean), every pinned file and the loaded `code` match it, the frames match their fingerprint and the
    variants' records are complete."""
    digest = sha256_file(hashlist)
    if digest != expected:
        raise Refused(f"{hashlist} hashes to {digest}, not the committed rule's {expected}")
    if check_git and not git_clean(hashlist):
        raise Refused(f"{hashlist} is not the committed, unmodified file")
    frozen = json.loads(hashlist.read_text())
    problems = [f"{name} changed (pinned {sha})" for name, sha in frozen["files"].items()
                if not path_of(name).is_file() or sha256_file(path_of(name)) != sha]
    if code is not None and code != frozen["code"]:
        changed = sorted(set(code.items()) ^ set(frozen["code"].items()))
        problems.append(f"the loaded code is not the pinned closure: {[n for n, _ in changed][:10]}")
    rule = testrule.load_rule()
    if population is not None:
        if frame_fingerprint(population) != frozen["frames"]:
            problems.append("the test population's t0 frames differ from their fingerprint")
        problems += sanitised_problems(population)
    problems += variant_problems(rule, population)
    if problems:
        raise Refused("the frozen test refuses to run:\n  " + "\n  ".join(problems))
    return {"hashlist_sha256": digest, "rule": rule, "frozen": frozen}


def path_check() -> dict:
    """A1 fold 0 seed 0 rescores the first `PATH_SHOTS` shots of its held-out fold (training pool) in the stored order."""
    run = a1.results_dir("A1") / "fold0" / "seed0"
    stored = pd.read_csv(run / f"{a1.run_name('A1')}_valid_predictions.csv", usecols=["shot_id", "logit"]).head(PATH_SHOTS)
    frames = pd.read_parquet(INDEX, columns=["shot_id", "frame_path"]).set_index("shot_id").loc[stored.shot_id].reset_index()
    bundle = a1.load(run / "bundle.pt")
    logits = score(bundle, frames)
    gap = float(np.max(np.abs(logits - stored.logit.to_numpy())))
    if gap > PATH_TOLERANCE:
        raise Refused(f"the path check failed: max |logit difference| {gap:.2e} on {PATH_SHOTS} training-pool shots")
    return {"shots": PATH_SHOTS, "max_abs_logit": gap, "tolerance": PATH_TOLERANCE}


def score(bundle, frames: pd.DataFrame) -> np.ndarray:
    """xG logits of one A1 bundle on `frames` (shot_id, frame_path) in row order: its training-fold standardiser on the
    sanitised D3 table, batch 8, eval mode. The dataset's `goal` column holds a constant (no label is read)."""
    sanitised = pd.read_parquet(SANITISED)
    rows = sanitised[sanitised.shot_id.isin(frames.shot_id)]
    table = a1.a1_table(rows).loc[frames.shot_id]
    geometry = a1.geometry_inputs(table, a1.standardiser_of(bundle.spec))
    cfg = bundle.recipe["cfg"]
    return a1.score(bundle.model, frames.assign(goal=0.0), a1.dataset_args(bundle.spec, geometry, None), cfg["batch_size"],
                    cfg["device"], cfg["num_workers"])


def a1_predictions(population: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(shot_id x member logits, A1's ensemble probability) on the population."""
    ensemble = json.loads((A1_ENSEMBLE / "ensemble.json").read_text())
    members = sorted(ensemble["members"], key=lambda m: (m["fold"], m["seed"]))
    logits = {}
    for m in members:
        path = ROOT / m["bundle"]
        if sha256_file(path) != m["sha256"]:
            raise Refused(f"{path} is not the bundle A1's ensemble.json lists")
        bundle = a1.load(path)
        if bundle.calibration["temperature"] != m["temperature"]:
            raise Refused(f"{path}: its temperature differs from ensemble.json")
        logits[f"f{m['fold']}s{m['seed']}"] = score(bundle, population[["shot_id", "frame_path"]])
        print(f"A1 f{m['fold']}s{m['seed']} scored ({datetime.now(UTC):%H:%M:%S})", flush=True)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    p = ensemble_probability(list(logits.values()), [m["temperature"] for m in members], ensemble["calibration"],
                                      population.is_penalty.to_numpy(bool))
    return pd.DataFrame(logits, index=pd.Index(population.shot_id, name="shot_id")), p


def read_test_labels(index: Path, population: pd.DataFrame) -> np.ndarray:
    """The index `goal` of the population's shots, in population order: the only label read, after the predictions exist."""
    labels = pd.read_parquet(index, columns=["shot_id", "goal"])
    return population[["shot_id"]].merge(labels, on="shot_id", how="left", validate="one_to_one").goal.astype(int).to_numpy()


def freeze(index: Path, code: dict[str, str], population: pd.DataFrame, path: dict) -> None:
    if HASHLIST.exists():
        raise Refused(f"{HASHLIST} exists; the rule is frozen once")
    rule = testrule.load_rule()
    if problems := variant_problems(rule, population):
        raise Refused("cannot freeze:\n  " + "\n  ".join(problems))
    frozen = {"study": "xg-detector", "ticket": "06", "rule": name_of(testrule.RULE_MD),
              "files": {name_of(f): sha256_file(f) for f in pinned_files(index)}, "code": code,
              "frames": frame_fingerprint(population), "path_check": path, "git": records.git_state(),
              "frozen": datetime.now(UTC).isoformat(timespec="seconds")}
    HASHLIST.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"wrote {HASHLIST}: sha256 {sha256_file(HASHLIST)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="ticket 06: write the hash list (refused if it exists)")
    mode.add_argument("--check", action="store_true", help="verify the frozen state and the path check, score nothing")
    parser.add_argument("--expect-hashlist", help="sha256 of the committed hash list (ticket 06 Comments)")
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    if not args.freeze and not args.expect_hashlist:
        raise Refused("--expect-hashlist is required")
    if not args.freeze and OUT.exists():
        raise Refused(f"{OUT} exists: the test runs once (a rerun needs the owner's decision)")
    check_gpu_free()

    with no_statsbomb_reads():
        with seal.development_labels_only(a1.PLAN_DIR / "plan.csv", args.index):
            plan = a1.verify_plan(args.index)
            population = test_population(args.index, plan, testrule.load_rule())
            path = path_check()
            code = loaded_code()
            if args.freeze:
                freeze(args.index, code, population, path)
                return
            state = verify(HASHLIST, args.expect_hashlist, code, population)
            rule = state["rule"]
            print(f"frozen state verified (hash list {state['hashlist_sha256']}), path check {path}", flush=True)
            if args.check:
                return
            OUT.mkdir(parents=True)
            logits, p_a1 = a1_predictions(population)
            g = g_combo_predictions(population)
            predictions = population[["shot_id", "game", "role"]].assign(p_a1=p_a1, p_g_combo=g.p.to_numpy())
            predictions.to_csv(OUT / "predictions.csv", index=False, float_format="%.17g")
            logits.reset_index().to_csv(OUT / "a1_members.csv", index=False, float_format="%.9g")
            if loaded_code() != code:
                raise Refused("the code loaded while scoring is not the pinned closure")
            written = {"hashlist_sha256": state["hashlist_sha256"], "files": records.output_hashes(OUT),
                       "written": datetime.now(UTC).isoformat(timespec="seconds"), "labels_read": False}
            (OUT / "predictions_record.json").write_text(json.dumps(written, indent=2) + "\n")
        y = read_test_labels(args.index, population)
        delta = testrule.paired_delta(y, predictions.p_a1.to_numpy(), predictions.p_g_combo.to_numpy(),
                                      predictions.game.to_numpy(), rule)
        verdict = testrule.reading(delta, rule)
        result = {"rule": state["frozen"]["rule"], "hashlist_sha256": state["hashlist_sha256"],
                  "population": {"shots": len(y), "matches": int(predictions.game.nunique()), "goals": int(y.sum()),
                                 "by_role": predictions.role.value_counts().to_dict()},
                  "delta": delta, "reading": verdict, "path_check": path,
                  "descriptive": {"a1": dev.scores(y, predictions.p_a1.to_numpy()),
                                  "g_combo": dev.scores(y, predictions.p_g_combo.to_numpy())}}
        (OUT / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        record = {"hashlist_sha256": state["hashlist_sha256"], "code": code, "git": records.git_state(),
                  "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": records.output_hashes(OUT)}
        (OUT / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
