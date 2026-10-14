"""The owner-ordered test of xg-stack on the 172 held-out matches (ticket 04, council C7; CPU, no GPU), under the
frozen rule `research/xg_stack_test_rule.md` (parameters `research/xg_stack_test_rule.json`).

    ./run.sh --no-sync python scripts/xgs_test.py --prepare                    # final fit, test inputs, checks, gate
    ./run.sh --no-sync python scripts/xgs_test.py --freeze                     # write the hash list (once)
    ./run.sh --no-sync python scripts/xgs_test.py --check --expect-hashlist SHA   # verify and replay, read no label
    ./run.sh --no-sync python scripts/xgs_test.py --test --expect-hashlist SHA    # the test, once

SHA is the sha256 of research/xg_stack_test_hashes.json quoted in ticket 04's Comments.

Every mode first runs `prepare` inside `testset.no_vendor_reads()` (no StatsBomb or vendor path) and the held-out
label seal (`seal.development_labels_only`: only training labels can be read):
* the training population of the C6 protocol (`inputs.build_population`, checked against the protocol's input hashes
  and counts), its outcomes, the arms R, G and GS of the C6 design (columns checked against the protocol), and their
  final fit on all 7,062 training shots (`final.final_fit`: C by match-grouped CV over the 5 host folds);
* the test population (3,819 non-penalty shots of the 135 evaluation and 37 gate matches, label-free index columns),
  its z recovered from G-combo's deployed test probability by inverting the frozen Platt, the E3 inputs (exact replay
  check, sanitation, angle wrap) and the SoccerNet flags, and the test predictions of R, G and GS;
* the label-free checks of C7 Q5 (`testset`): z inversion against the member logits, the test z's mean and SD against
  the training z's, coverage, ball-source levels and provenance.

--prepare   refuses if results/xgs_test_prep exists. Writes the final fit, the test inputs and predictions, the checks,
            the power scenarios (`rule.power_scenarios`: development SD of G, A1's test SE, C6's recovery) and the
            synthetic-null gate (`rule.null_gate`, labels unread), then complete.json with `all_passed`. When every
            check and the gate pass, it writes the rule's `before_scoring` block (power, gate, success level in force).
--freeze    refuses if the hash list exists, unless the preparation passed, the replay reproduces its files byte for
            byte, the rule's `before_scoring` block is the preparation's, and every pinned file in the checkout is
            committed and unmodified. Writes the hash list: the rule, the code closure loaded by this process, every
            input, the preparation's files, the StatsBomb descriptive script (pinned, never loaded here).
--check     refuses unless the hash list hashes to SHA and is committed, every pinned file and the loaded code match
            it, and the replay reproduces the preparation's files byte for byte. Reads and scores no test label.
--test      everything --check does, and refuses if results/xgs_test exists. Then, outside the seal (still with no
            StatsBomb path), the index `goal` of the 3,819 test shots is read once and the rule is applied: ΔLL(G - R)
            with the paired match bootstrap (5,000 draws, seed 0), the reading at the success level in force, the TOST
            at δ = 0.002 and 0.005, the pre-written report sentence; then the descriptive readings of C7 Q4 without
            StatsBomb (GS - R, G - deployed G-combo, evaluation and gate matches, the E3 categories). Writes
            results/xgs_test/{result.json, categories.csv, complete.json}. A rerun only after a purely technical
            failure with the same inputs, written down in ticket 04.
StatsBomb is read only by `scripts/xgs_test_statsbomb.py`, after results/xgs_test/complete.json exists.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal
from xgpoc.paths import INDEX
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import protocol as pr
from xgpoc.stack import stats as ss
from xgpoc.stack_heldout import final
from xgpoc.stack_heldout import rule as tr
from xgpoc.stack_heldout import testset as ts
from xgs_run import PROTOCOL_SHA256, check_population

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = records.CHECKOUT
PREP = ROOT / "results/xgs_test_prep"
OUT = ROOT / "results/xgs_test"
HASHLIST = ROOT / "research/xg_stack_test_hashes.json"
G_COMBO = ROOT / "results/xgv_g_combo"
G_NAME = "g_finetune_dinov2_l_b4_overlay_combo"
G_TEST = ROOT / "results/xgd_gcombo_test"
DEV = ROOT / "results/xgs_stack/real"
CONTROLS = ROOT / "results/xgs_stack/controls"
STATSBOMB_CODE = (ROOT / "scripts/xgs_test_statsbomb.py", ROOT / "xgpoc/stack_heldout/vendor.py")
ARMS = ("R", "G", "GS")
A1_TEST_SE = 0.001082           # the bootstrap SD of the A1 test's ΔLL (results/xgd_test, ticket 07), quoted by C7
REPLAYED = ("final_fit.json", "test_inputs.csv", "predictions.csv", "checks.json")


class Refused(SystemExit):
    """The xg-stack test refuses to run."""


@dataclass(frozen=True)
class Prepared:
    """Everything `prepare` builds: the training population, the final fits, the test table and predictions, the
    checks, and the files `REPLAYED` as text."""

    population: si.Population
    fits: dict[str, final.FinalFit]
    table: pd.DataFrame
    predictions: pd.DataFrame
    checks: dict
    files: dict[str, str]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def as_json(obj) -> str:
    return json.dumps(obj, indent=2, default=float) + "\n"


# ------------------------------------------------------------------ inputs of the test side
def g_combo_ensemble() -> tuple[dict, str]:
    """G-combo's ensemble.json, refused unless it is the one its test predictions were made with."""
    path = G_COMBO / "ensemble.json"
    digest = sha256_file(path)
    if json.loads((G_TEST / "complete.json").read_text())["inputs"]["ensemble_json"] != digest:
        raise Refused("results/xgv_g_combo/ensemble.json is not the one G-combo's test predictions were made with")
    return json.loads(path.read_text()), digest


def deployed_probability(shots: pd.DataFrame) -> pd.Series:
    """G-combo's deployed test probability of the population's shots (index shot_id, population order), checked
    against its completion record; no label column is read."""
    records.verified_file(G_TEST / "predictions.csv")
    stored = pd.read_csv(G_TEST / "predictions.csv", usecols=["shot_id", "game", "is_penalty", "p"]).set_index("shot_id")
    if not stored.index.is_unique:
        raise Refused("G-combo's test predictions repeat a shot id")
    if missing := int((~shots.index.isin(stored.index)).sum()):
        raise Refused(f"G-combo's test predictions miss {missing} population shots")
    rows = stored.loc[shots.index]
    if (rows.game != shots.game).any() or rows.is_penalty.astype(bool).any():
        raise Refused("G-combo's test predictions disagree with the index on a shot's match or penalty flag")
    return rows.p.astype(float)


def member_logits(ensemble: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    """(shot_id x member logits where they are stored: the gate shots' stored predictions of each member and the
    unpaired evaluation shots' unpaired_members.csv; the members' temperatures). No goal column is read."""
    temperatures = {f"f{m['fold']}s{m['seed']}": float(m["temperature"])
                    for m in sorted(ensemble["members"], key=lambda m: (m["fold"], m["seed"]))}
    gate = {}
    for name in temperatures:
        path = G_COMBO / f"fold{name[1]}" / f"seed{name[3]}" / f"{G_NAME}_gate_predictions.csv"
        records.verified_file(path)
        gate[name] = pd.read_csv(path, usecols=["shot_id", "logit"]).set_index("shot_id").logit
    records.verified_file(G_TEST / "unpaired_members.csv")
    unpaired = pd.read_csv(G_TEST / "unpaired_members.csv").set_index("shot_id")[list(temperatures)]
    logits = pd.concat([pd.DataFrame(gate), unpaired])
    if not logits.index.is_unique:
        raise Refused("a shot has member logits in two sources")
    return logits, temperatures


# ------------------------------------------------------------------ prepare (label-free on the test side)
def prepare() -> Prepared:
    """The training-only final fit and the label-free test side (module docstring). Must run inside the seal."""
    plan = a1.verify_plan(INDEX)
    protocol = pr.load(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
    population = si.build_population(plan=plan)
    check_population(population, protocol)
    spec = sf.spec(protocol["design"]["chosen"], population.table)
    frozen = {name: protocol["design"]["arms"][name]["columns"] for name in ARMS}
    y = si.load_outcomes(population).aligned(population)
    fits = {name: final.final_fit(population.table, population.folds, y, spec.arms[name], spec.model) for name in ARMS}

    shots = ts.heldout_shots(INDEX, plan)
    ensemble, ensemble_sha = g_combo_ensemble()
    p_deployed = deployed_probability(shots)
    z = pd.Series(ts.invert_platt(p_deployed.to_numpy(), ensemble["calibration"]), index=shots.index)
    logits, temperatures = member_logits(ensemble)
    annotations = {g: si.read_annotations(g) for g in sorted(shots.game.unique())}
    table, wrapped = ts.heldout_table(shots, z, si.E3, annotations)
    predictions = pd.DataFrame({"shot_id": table.index, "game": table.game.to_numpy(), "role": table.role.to_numpy(),
                                sf.Z: table[sf.Z].to_numpy(), "p_g_combo": p_deployed.to_numpy()}
                               | {f"p_{name}": fits[name].predict(table) for name in ARMS})
    training_z = population.table[sf.Z].to_numpy(float)
    knots = fits["G"].preprocessor.knots[si.col("dist", 0)]
    dist = table[si.col("dist", 0)].to_numpy(float)
    checks = {
        "z_inversion": ts.z_inversion(p_deployed, z, ts.member_z(logits, temperatures), ensemble["calibration"])
        | {"paired_shots": "no member logits outside the vendor tables; their p was checked against the members in ticket 06 "
                           "(results/xgd_gcombo_test/checks.json paired.max_abs_dp_vs_member_logits)",
           "paired_max_abs_dp_ticket_06": json.loads((G_TEST / "checks.json").read_text())["paired"]["max_abs_dp_vs_member_logits"]},
        "z_distribution": ts.z_distribution(table[sf.Z].to_numpy(float), training_z),
        "coverage": ts.coverage(table, predictions),
        "ball_levels": ts.ball_levels(table, population.table),
        "provenance": ts.provenance(table, population.table, spec, frozen, annotations),
        "e3": {"replay": "the replayed raw E3 features equal features.parquet exactly (inputs.e3_table)",
               "angle_wrapped": wrapped, "shot_missing_t0": int(table[si.col("shot_missing", 0)].sum()),
               "dist_t0_outside_training_knots": int(((dist < knots[0]) | (dist > knots[-1])).sum()),
               "flags": {f: int(table[f].sum()) for f in si.S_FLAGS}, "passed": True},
        "inputs": {"ensemble_json": ensemble_sha, "g_combo_predictions": sha256_file(G_TEST / "predictions.csv"),
                   "labels_v2_games": len(annotations), "training": population.inputs, "passed": True},
    }
    files = {"final_fit.json": as_json({name: fit.record() for name, fit in fits.items()}),
             "test_inputs.csv": table.reset_index().to_csv(index=False, float_format="%.17g"),
             "predictions.csv": predictions.to_csv(index=False, float_format="%.17g"),
             "checks.json": as_json(checks)}
    return Prepared(population, fits, table, predictions, checks, files)


def all_passed(checks: dict) -> bool:
    return all(c["passed"] for c in checks.values())


def power(population: si.Population) -> dict:
    """The power scenarios (`rule.power_scenarios`) from the development run and controls, with no test label."""
    summary = json.loads((DEV / "summary.json").read_text())
    controls = json.loads((CONTROLS / "controls.json").read_text())
    for path in (DEV / "summary.json", CONTROLS / "controls.json"):
        records.verified_file(path)
    g = summary["deltas"]["G"]
    dev = {"shots": population.counts["shots"], "matches": population.counts["matches"], "delta_ll": g["delta_ll"],
           "boot_sd": g["boot_sd"], "draws": g["draws"]}
    positive = controls["4_positive"]
    recovery = positive["mean_recovered_delta_G"] / positive["target"]
    test = {"shots": sum(ts.SHOTS_BY_ROLE.values()), "matches": sum(ts.ROLES.values())}
    return {"development": dev, "test": test, "a1_test_se": A1_TEST_SE,
            **tr.power_scenarios(dev, test, A1_TEST_SE, recovery)}


def before_scoring(power_record: dict, gate: dict) -> dict:
    return {"power": power_record, "gate": gate, "level_in_force": gate["level_in_force"]}


def run_prepare(rule: dict) -> None:
    if PREP.exists():
        raise Refused(f"{PREP} exists; the preparation is written once")
    with ts.no_vendor_reads(), seal.development_labels_only(a1.PLAN_DIR / "plan.csv", INDEX):
        prepared = prepare()
        code = loaded_code()
        power_record = power(prepared.population)
        p = prepared.predictions
        gate = tr.null_gate(p.p_R.to_numpy(), p.p_G.to_numpy(), p.game.to_numpy(),
                            tr.bootstrap_weights(p.game.to_numpy(), rule), rule) if all_passed(prepared.checks) else None
    PREP.mkdir(parents=True)
    for name, text in prepared.files.items():
        (PREP / name).write_text(text)
    write_json(PREP / "power.json", power_record)
    if gate is not None:
        write_json(PREP / "gate.json", gate)
    passed = all_passed(prepared.checks) and gate is not None and gate["passed"]
    write_json(PREP / "complete.json", {
        "rule_parameters_sha256": tr.parameters_digest(rule), "protocol_sha256": PROTOCOL_SHA256, "all_passed": passed,
        "checks": {k: c["passed"] for k, c in prepared.checks.items()}, "gate_passed": None if gate is None else gate["passed"],
        "level_in_force": None if gate is None else gate["level_in_force"], "labels_read": "training only (inside the seal)",
        "code": code, "git": records.git_state(), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": records.output_hashes(PREP)})
    print(json.dumps({k: c["passed"] for k, c in prepared.checks.items()}
                     | {"gate": None if gate is None else gate["false_success_rate"], "all_passed": passed}, indent=2))
    if not passed:
        raise Refused("a label-free check or the synthetic-null gate failed: the test does not run (see results/xgs_test_prep)")
    block = before_scoring(power_record, gate)
    if tr.BEFORE_SCORING in rule:
        if json.loads(json.dumps(rule[tr.BEFORE_SCORING], default=float)) != json.loads(json.dumps(block, default=float)):
            raise Refused(f"{tr.RULE_JSON} already holds another before_scoring block")
        return
    tr.RULE_JSON.write_text(json.dumps(rule | {tr.BEFORE_SCORING: block}, indent=2, ensure_ascii=False, default=float) + "\n")
    print(f"wrote the before_scoring block into {tr.RULE_JSON} (level in force {gate['level_in_force']})")


# ------------------------------------------------------------------ the frozen state
def name_of(path: Path) -> str:
    """Checkout-relative name of a file inside the checkout, else the absolute path (as `xgd_test.name_of`)."""
    path = Path(path)
    return str(path.relative_to(CHECKOUT)) if path.is_relative_to(CHECKOUT) else str(path)


def path_of(name: str) -> Path:
    return Path(name) if Path(name).is_absolute() else CHECKOUT / name


def loaded_code() -> dict[str, str]:
    """sha256 of every Python file of the checkout loaded in this process (as `xgd_test.loaded_code`)."""
    files = {Path(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    files = {f for f in files if f.is_absolute() and f.suffix == ".py" and f.is_relative_to(CHECKOUT)}
    return {name_of(f): sha256_file(f) for f in sorted(files)}


def git_clean(path: Path) -> bool:
    """Tracked and unchanged against HEAD."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=CHECKOUT, capture_output=True).returncode == 0
    unchanged = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=CHECKOUT).returncode == 0
    return tracked and unchanged


def pinned_files(population: si.Population, games: list[str]) -> list[Path]:
    """Every file the test depends on, in a fixed order: the rule, the protocol, the preparation, the development
    record the power reads, the training and test inputs, G-combo's files and the StatsBomb descriptive code."""
    files = [tr.RULE_MD, tr.RULE_JSON, ROOT / pr.PROTOCOL]
    files += sorted(p for p in PREP.iterdir() if p.is_file())
    files += [DEV / "summary.json", DEV / "complete.json", CONTROLS / "controls.json", CONTROLS / "complete.json"]
    files += [INDEX, a1.FOLDS, *[a1.PLAN_DIR / n for n in ("plan.csv", "plan.json", "complete.json")]]
    files += [si.OOF, si.OOF.parent / "complete.json", *[si.E3 / n for n in ("frames.parquet", "features.parquet", "complete.json")]]
    files += [G_COMBO / "ensemble.json"] + [G_TEST / n for n in ("predictions.csv", "unpaired_members.csv", "checks.json", "complete.json")]
    for k in range(5):
        for s in (0, 1):
            run = G_COMBO / f"fold{k}" / f"seed{s}"
            files += [run / f"{G_NAME}_gate_predictions.csv", run / "complete.json"]
    files += [si.labels_file(g) for g in sorted(set(games) | set(population.table.game))]
    return files + list(STATSBOMB_CODE)


def replay_problems(prepared: Prepared) -> list[str]:
    """The replayed files that differ from the preparation's, byte for byte."""
    return [f"{name} differs from the preparation's" for name, text in prepared.files.items()
            if not (PREP / name).is_file() or (PREP / name).read_text() != text]


def prep_problems(rule: dict, code: dict[str, str]) -> list[str]:
    """Why the preparation is not a passed, complete record of this rule written by the loaded `code` (an empty list
    when it is)."""
    problems = []
    try:
        record = json.loads((PREP / "complete.json").read_text())
        for name, digest in record["files"].items():
            if sha256_file(PREP / name) != digest:
                problems.append(f"{PREP / name} does not match the preparation's completion record")
        if not record["all_passed"]:
            problems.append("the preparation did not pass")
        if record["rule_parameters_sha256"] != tr.parameters_digest(rule):
            problems.append("the preparation ran under other rule parameters")
        if record["code"] != code:
            problems.append("the preparation ran under other code than the loaded closure")
        block = before_scoring(json.loads((PREP / "power.json").read_text()), json.loads((PREP / "gate.json").read_text()))
        if rule.get(tr.BEFORE_SCORING) != block:
            problems.append("the rule's before_scoring block is not the preparation's power and gate")
        if tr.level_in_force(rule) != record["level_in_force"]:
            problems.append("the rule's success level in force is not the gate's")
    except (OSError, KeyError, ValueError) as e:
        problems.append(f"the preparation is incomplete: {e}")
    return problems


def freeze(rule: dict, prepared: Prepared, code: dict[str, str]) -> None:
    if HASHLIST.exists():
        raise Refused(f"{HASHLIST} exists; the test is frozen once")
    problems = prep_problems(rule, code) + replay_problems(prepared)
    files = pinned_files(prepared.population, sorted(prepared.table.game.unique()))
    tracked = [tr.RULE_MD, tr.RULE_JSON, ROOT / pr.PROTOCOL, *STATSBOMB_CODE, *map(path_of, code)]
    problems += [f"{name_of(f)} is not committed and unmodified" for f in tracked if not git_clean(f)]
    if problems:
        raise Refused("cannot freeze:\n  " + "\n  ".join(problems))
    frozen = {"study": "xg-stack", "ticket": "04", "council": "C7", "rule": name_of(tr.RULE_MD),
              "level_in_force": tr.level_in_force(rule), "files": {name_of(f): sha256_file(f) for f in files},
              "code": code, "git": records.git_state(), "frozen": datetime.now(UTC).isoformat(timespec="seconds")}
    HASHLIST.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"wrote {HASHLIST}: sha256 {sha256_file(HASHLIST)}")


def verify(expected: str, prepared: Prepared, code: dict[str, str], *, check_git: bool = True) -> dict:
    """The frozen state, refused (`Refused`, every problem listed) unless the hash list hashes to `expected` (and is
    committed and clean), every pinned file and the loaded `code` match it, the preparation passed under this rule and
    the replay reproduces its files."""
    digest = sha256_file(HASHLIST)
    if digest != expected:
        raise Refused(f"{HASHLIST} hashes to {digest}, not {expected}")
    if check_git and not git_clean(HASHLIST):
        raise Refused(f"{HASHLIST} is not the committed, unmodified file")
    frozen = json.loads(HASHLIST.read_text())
    problems = [f"{name} changed (pinned {sha})" for name, sha in frozen["files"].items()
                if not path_of(name).is_file() or sha256_file(path_of(name)) != sha]
    if code != frozen["code"]:
        changed = sorted({n for n, _ in set(code.items()) ^ set(frozen["code"].items())})
        problems.append(f"the loaded code is not the pinned closure: {changed[:10]}")
    rule = tr.load_rule()
    problems += prep_problems(rule, code) + replay_problems(prepared)
    if problems:
        raise Refused("the frozen test refuses to run:\n  " + "\n  ".join(problems))
    return {"hashlist_sha256": digest, "rule": rule, "frozen": frozen}


# ------------------------------------------------------------------ the test
def read_test_labels(shot_ids: pd.Series) -> np.ndarray:
    """The index `goal` of the test shots in `shot_ids` order: the only test label read, after every check."""
    labels = pd.read_parquet(INDEX, columns=["shot_id", "goal"])
    y = pd.DataFrame({"shot_id": shot_ids}).merge(labels, on="shot_id", how="left", validate="one_to_one").goal
    if y.isna().any():
        raise Refused("a test shot has no label")
    return y.astype(int).to_numpy()


def apply_rule(rule: dict, y: np.ndarray, predictions: pd.DataFrame, table: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """(result.json's readings, the category table) of the test (module docstring, --test)."""
    level, clip = tr.level_in_force(rule), rule["metric"]["clip"]
    games = predictions.game.to_numpy()
    weights = tr.bootstrap_weights(games, rule)
    p = {name: predictions[f"p_{name}"].to_numpy() for name in (*ARMS, "g_combo")}
    primary = tr.delta(y, p["G"], p["R"], weights, clip)
    read = tr.reading(primary, level)
    roles = predictions.role.to_numpy()
    descriptive = {
        "label": "descriptive (C7 Q4): never a success, cannot rescue the primary reading; nothing feeds the headline",
        "scores": {name: ss.scores(y, q) for name, q in p.items()},
        "GS_vs_R": tr.delta(y, p["GS"], p["R"], weights, clip),
        "G_vs_g_combo": tr.delta(y, p["G"], p["g_combo"], weights, clip),
        "R_vs_g_combo": tr.delta(y, p["R"], p["g_combo"], weights, clip),
        "G_vs_R_by_role": {"label": "hypothesis formed on these matches",
                           **{role: tr.delta(y, p["G"], p["R"], weights, clip, roles == role) for role in ts.ROLES}},
    }
    result = {"level_in_force": level, "population": {"shots": len(y), "matches": int(predictions.game.nunique()),
                                                      "goals": int(y.sum()), "by_role": predictions.role.value_counts().to_dict()},
              "primary": {"variant": "G", "base": "R", "delta": primary, "reading": read,
                          "sentence": tr.sentence(rule, primary, read)},
              "descriptive": descriptive}
    categories = ss.category_table(table, y, {name: p[name] for name in ARMS}, weights)
    return result, categories.assign(label="hypothesis formed on these matches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "freeze", "check", "test"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--expect-hashlist", help="sha256 of the committed hash list (ticket 04 Comments)")
    args = parser.parse_args()
    if (args.check or args.test) and not args.expect_hashlist:
        raise Refused("--expect-hashlist is required")
    if args.test and OUT.exists():
        raise Refused(f"{OUT} exists: the test runs once (a rerun only after a purely technical failure, written down)")
    rule = tr.load_rule()
    if args.prepare:
        run_prepare(rule)
        return

    with ts.no_vendor_reads():
        with seal.development_labels_only(a1.PLAN_DIR / "plan.csv", INDEX):
            prepared = prepare()
            code = loaded_code()
            if args.freeze:
                freeze(rule, prepared, code)
                return
            state = verify(args.expect_hashlist, prepared, code)
            print(f"frozen state verified (hash list {state['hashlist_sha256']}, success level "
                  f"{tr.level_in_force(state['rule'])}); replay identical; no test label read", flush=True)
            if args.check:
                return
        y = read_test_labels(prepared.predictions.shot_id)
        result, categories = apply_rule(state["rule"], y, prepared.predictions, prepared.table)
        if loaded_code() != code:
            raise Refused("the code loaded while scoring is not the pinned closure")
        OUT.mkdir(parents=True)
        write_json(OUT / "result.json", {"rule": state["frozen"]["rule"], "hashlist_sha256": state["hashlist_sha256"]} | result)
        categories.to_csv(OUT / "categories.csv", index=False, float_format="%.9g")
        write_json(OUT / "complete.json", {"hashlist_sha256": state["hashlist_sha256"], "code": code,
                                           "git": records.git_state(), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                                           "files": records.output_hashes(OUT)})
    print(json.dumps({k: result[k] for k in ("level_in_force", "population", "primary")}, indent=2, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
