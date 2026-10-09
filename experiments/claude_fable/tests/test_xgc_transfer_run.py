"""Transfer arms (ticket 12): plan assertion, role sets, resume and completeness; CPU only, no model."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from xgpoc.bundle import sha256_file
from xgpoc.paths import INDEX
from xgpoc.transfer import arms
from xgpoc.transfer.arms import (
    ARM_SIZES,
    PLAN_SHA256,
    PROTOCOL_SHA256,
    FitKey,
    arm_constants,
    arm_sets,
    completeness,
    fit_grid,
    verify_plan,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_xgc_transfer import (  # noqa: E402
    FitContext,
    check,
    fit_identity,
    fit_queue,
    foreign_gpu_processes,
    load_shots,
    main,
    read_failures,
    validate_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "research" / "xgc_transfer_plan"
PROTOCOL = ROOT / "research" / "xgc_transfer_protocol.md"
needs_index = pytest.mark.skipif(not INDEX.exists(), reason="shot index not available")


@pytest.fixture
def plan():
    return pd.read_csv(PLAN_DIR / "plan.csv")


@pytest.fixture
def plan_shots(plan):
    """Two shots per planned match; the second is a penalty and a goal."""
    games = sorted(plan.game.unique())
    return pd.DataFrame({"shot_id": [f"{g}#{i}" for g in games for i in range(2)], "game": [g for g in games for _ in range(2)],
                         "goal": [i == 1 for _ in games for i in range(2)], "is_penalty": [i == 1 for _ in games for i in range(2)]})


def test_shipped_plan_and_protocol_carry_the_approved_hashes():
    assert sha256_file(PLAN_DIR / "plan.csv") == PLAN_SHA256
    assert sha256_file(PROTOCOL) == PROTOCOL_SHA256


@needs_index
def test_approved_plan_is_accepted_on_the_real_index():
    plan, hashes = verify_plan(pd.read_parquet(INDEX), PLAN_DIR, PROTOCOL)
    assert len(plan) == 6798 and hashes["plan_sha256"] == PLAN_SHA256 and hashes["protocol_sha256"] == PROTOCOL_SHA256
    assert set(hashes["split_plan_code"]) == {"xgpoc/transfer/plan.py", "scripts/make_transfer_plan.py"}


def test_a_plan_that_is_not_the_approved_one_is_refused(monkeypatch, plan):
    monkeypatch.setattr(arms.split_plan, "build_plan", lambda index, seed: plan.iloc[1:])  # one planned row lost
    with pytest.raises(ValueError, match="not the approved one"):
        verify_plan(pd.DataFrame(), PLAN_DIR, PROTOCOL)


@needs_index
def test_an_edited_plan_file_manifest_or_protocol_is_refused(tmp_path):
    index = pd.read_parquet(INDEX)
    edited = tmp_path / "plan"
    edited.mkdir()
    (edited / "manifest.json").write_text((PLAN_DIR / "manifest.json").read_text())
    (edited / "plan.csv").write_text((PLAN_DIR / "plan.csv").read_text().replace("train", "select", 1))
    with pytest.raises(ValueError, match="plan.csv"):
        verify_plan(index, edited, PROTOCOL)
    (edited / "plan.csv").write_text((PLAN_DIR / "plan.csv").read_text())
    manifest = json.loads((PLAN_DIR / "manifest.json").read_text())
    (edited / "manifest.json").write_text(json.dumps({**manifest, "plan_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="manifest"):
        verify_plan(index, edited, PROTOCOL)
    (edited / "manifest.json").write_text(json.dumps({**manifest, "sources": {"xgpoc/transfer/plan.py": "0" * 64}}))
    with pytest.raises(ValueError, match="code changed"):
        verify_plan(index, edited, PROTOCOL)
    protocol = tmp_path / "protocol.md"
    protocol.write_text(PROTOCOL.read_text() + "\nedit\n")
    with pytest.raises(ValueError, match="protocol"):
        verify_plan(index, PLAN_DIR, protocol)


def test_grid_is_40_runs_seed_major():
    grid = fit_grid()
    assert len(grid) == len(set(grid)) == 40
    assert {k.seed for k in grid[:20]} == {0} and {k.seed for k in grid[20:]} == {1}


def test_sets_follow_the_roles_of_the_plan(plan, plan_shots):
    for key in fit_grid((0,)):
        sets = arm_sets(plan_shots, plan, key)
        rows = plan[(plan.league == key.league) & (plan.half_set == key.half_set) & (plan.arm == key.arm)]
        for role, df in sets.items():
            assert set(df.game) == set(rows.game[rows.role == role])
            assert len(df) == 2 * df.game.nunique()
        assert {r: df.game.nunique() for r, df in sets.items() if r != "test"} == ARM_SIZES
        trained = set().union(*(set(sets[r].game) for r in ARM_SIZES))
        assert not trained & set(sets["test"].game)
        assert arm_constants(sets)["arm_penalty_goals"] == 300


def test_arms_of_one_half_set_share_the_test_matches_and_differ_in_training(plan, plan_shots):
    a = arm_sets(plan_shots, plan, FitKey("england_epl", "A", "without", 0))
    b = arm_sets(plan_shots, plan, FitKey("england_epl", "A", "with", 0))
    assert set(a["test"].game) == set(b["test"].game)
    assert set(a["train"].game) != set(b["train"].game)


def test_missing_matches_and_wrong_role_sizes_are_refused(plan, plan_shots):
    key = FitKey("england_epl", "A", "with", 0)
    with pytest.raises(ValueError, match="no shots"):
        arm_sets(plan_shots.iloc[2:], plan, key)
    one_select = plan[(plan.arm == "with") & (plan.role == "select") & (plan.league == key.league) & (plan.half_set == "A")].index[0]
    with pytest.raises(ValueError, match="role sizes"):
        arm_sets(plan_shots, plan.drop(one_select), key)


def fake_queue(tmp_path, plan, plan_shots, keys, identity_of=lambda key: {"key": str(key.path)}):
    trained = []

    def train(key, sets, out, identity):
        out.mkdir(parents=True)
        pd.DataFrame({"shot_id": sets["test"].shot_id}).to_csv(out / "test_predictions.csv", index=False)
        (out / "complete.json").write_text(json.dumps({"identity": identity, "files": {
            "test_predictions.csv": sha256_file(out / "test_predictions.csv")}}))
        trained.append(key)

    run = lambda max_runs=None: fit_queue(keys, tmp_path, plan_shots, plan, lambda k, s: identity_of(k), train, max_runs)  # noqa: E731
    return run, trained


def test_resume_skips_finished_runs_and_redoes_unfinished_ones(tmp_path, plan, plan_shots):
    keys = fit_grid((0,))[:4]
    run, trained = fake_queue(tmp_path, plan, plan_shots, keys)
    assert run(max_runs=2) == 2 and trained == keys[:2]
    (tmp_path / keys[1].path / "complete.json").unlink()  # an interrupted run has no record
    for leftover in (tmp_path / keys[1].path).iterdir():
        leftover.unlink()
    (tmp_path / keys[1].path).rmdir()
    assert run() == 3 and trained[2:] == keys[1:]
    assert run() == 0


def test_resume_refuses_a_run_trained_under_another_identity(tmp_path, plan, plan_shots):
    keys = fit_grid((0,))[:1]
    run, _ = fake_queue(tmp_path, plan, plan_shots, keys)
    run()
    changed, _ = fake_queue(tmp_path, plan, plan_shots, keys, identity_of=lambda key: {"key": "other"})
    with pytest.raises(ValueError, match="different run"):
        changed()


def write_run(out, key, shots, plan, *, roles=("select", "calibration", "test"), limit_hit=False, record_files=True):
    """A finished run with predictions for the planned shots of each role in `roles`."""
    run = out / key.path
    run.mkdir(parents=True)
    arm = plan[(plan.league == key.league) & (plan.half_set == key.half_set) & (plan.arm == key.arm)]
    names = []
    for role in roles:
        ids = shots.shot_id[shots.game.isin(arm.game[arm.role == role])]
        pd.DataFrame({"shot_id": ids, "goal": 0, "logit": 0.0, "p": 0.5}).to_csv(run / f"{role}_predictions.csv", index=False)
        names.append(f"{role}_predictions.csv")
    best = {"valid_log_loss_cal": 0.3, "valid_log_loss": 0.31, "valid_auc": 0.75, "temperature": 1.1}
    (run / "training.json").write_text(json.dumps({"best_epoch": 2, "best": best, "time_limit_hit": limit_hit,
                                                   "compute": {"epochs_run": 5, "train_minutes": 7.0}}))
    names.append("training.json")
    (run / "complete.json").write_text(json.dumps({"plan_sha256": PLAN_SHA256, "protocol_sha256": PROTOCOL_SHA256,
                                                   "files": {n: sha256_file(run / n) for n in names} if record_files else {}}))


def write_all(out, shots, plan, skip=(), **kw):
    for k in fit_grid():
        if k not in skip:
            write_run(out, k, shots, plan, **kw)


def test_check_passes_on_a_complete_grid_and_writes_table_and_deviations(tmp_path, plan, plan_shots):
    write_all(tmp_path, plan_shots, plan)
    table, problems = check(tmp_path, plan_shots, plan)
    assert problems == [] and len(table) == 40 and not table.time_limit_hit.any()
    assert {"select_ll_cal", "select_auc"} <= set(table) and not any("test" in c for c in table)
    assert json.loads((tmp_path / "deviations.json").read_text())["time_limit_hit"] == []
    assert (tmp_path / "training_table.csv").exists()


def test_check_reports_missing_cells_time_limit_runs_and_failures(tmp_path, plan, plan_shots):
    grid = fit_grid()
    write_all(tmp_path, plan_shots, plan, skip=(grid[0], grid[1]))
    (tmp_path / grid[2].path / "training.json").write_text(
        json.dumps({"best_epoch": 0, "best": {"valid_log_loss_cal": 0.3, "valid_log_loss": 0.3, "valid_auc": 0.7, "temperature": 1.0},
                    "time_limit_hit": True, "compute": {"epochs_run": 3, "train_minutes": 45.1}}))
    (tmp_path / "failures.json").write_text(json.dumps({str(grid[0].path): {"attempts": 2, "error": "boom"}}))
    table, problems = check(tmp_path, plan_shots, plan)
    deviations = json.loads((tmp_path / "deviations.json").read_text())
    assert len(table) == 38 and sum("no complete.json" in p for p in problems) == 2
    assert deviations["time_limit_hit"] == [str(grid[2].path)] and str(grid[0].path) in deviations["failed_twice"]
    assert any("changed since" in p and grid[2].league in p for p in problems)  # training.json no longer matches its hash


def test_completeness_needs_select_and_calibration_predictions_too(tmp_path, plan, plan_shots):
    grid = fit_grid()
    write_all(tmp_path, plan_shots, plan, skip=(grid[0], grid[1]))
    write_run(tmp_path, grid[0], plan_shots, plan, roles=("test", "calibration"))
    write_run(tmp_path, grid[1], plan_shots, plan)
    problems = completeness(tmp_path, plan_shots, plan)
    assert any("no select_predictions.csv" in p for p in problems)
    short = tmp_path / grid[1].path / "calibration_predictions.csv"
    short.write_text("".join(short.read_text().splitlines(keepends=True)[:-1]))  # drop a shot; the record's hash no longer fits
    messages = " ".join(completeness(tmp_path, plan_shots, plan))
    assert "calibration_predictions.csv missing or changed" in messages and "calibration predictions cover" in messages


def test_completeness_refuses_an_unhashed_or_foreign_record(tmp_path, plan, plan_shots):
    write_all(tmp_path, plan_shots, plan)
    first = fit_grid()[0]
    record = json.loads((tmp_path / first.path / "complete.json").read_text())
    (tmp_path / first.path / "complete.json").write_text(json.dumps({**record, "plan_sha256": "0" * 64}))
    assert completeness(tmp_path, plan_shots, plan) == [f"{first.path}: completion record names another plan or protocol"]


def test_seeds_outside_the_protocol_grid_are_refused_unless_smoke():
    assert validate_seeds([0, 1], smoke=False) == [0, 1] and validate_seeds([1], smoke=False) == [1]
    with pytest.raises(ValueError, match="subset"):
        validate_seeds([0, 2], smoke=False)
    assert validate_seeds([7], smoke=True) == [7]


def test_only_foreign_python_gpu_processes_block_training():
    listing = "2578, /opt/1Password/1password --type=gpu-process\n3591, /usr/bin/chrome\n"
    assert foreign_gpu_processes(listing, own_pid=1) == []
    listing += "2222748, /home/kuba/.venv/bin/python3\n99, /usr/bin/python3 train.py\n"
    assert len(foreign_gpu_processes(listing, own_pid=99)) == 1
    assert foreign_gpu_processes("", own_pid=1) == []


def flaky_queue(tmp_path, plan, plan_shots, keys, fail_times):
    attempts = {k: 0 for k in keys}

    def train(key, sets, out, identity):
        attempts[key] += 1
        if attempts[key] <= fail_times.get(key, 0):
            raise RuntimeError("CUDA out of memory")
        out.mkdir(parents=True)
        (out / "complete.json").write_text(json.dumps({"identity": identity, "files": {}}))

    run = lambda: fit_queue(keys, tmp_path, plan_shots, plan, lambda k, s: {"key": str(k.path)}, train)  # noqa: E731
    return run, attempts


def test_a_failed_run_is_retried_once_then_recorded_missing_without_stopping_the_queue(tmp_path, plan, plan_shots):
    keys = fit_grid((0,))[:3]
    run, attempts = flaky_queue(tmp_path, plan, plan_shots, keys, {keys[0]: 1, keys[1]: 5})
    assert run() == 2  # keys[0] succeeds on the retry, keys[1] never, keys[2] first time
    assert attempts == {keys[0]: 2, keys[1]: 2, keys[2]: 1}
    assert list(read_failures(tmp_path)) == [str(keys[1].path)]
    assert run() == 0 and attempts[keys[1]] == 2  # a relaunch does not try the failed cell a third time


def test_no_test_set_metric_is_computed_by_the_runner():
    source = (ROOT / "scripts" / "run_xgc_transfer.py").read_text()
    assert "evaluate" not in source and "log_loss(" not in source and "roc_auc" not in source


@pytest.fixture
def ctx():
    from run_xgc_g import recipe
    from run_xgc_transfer import AUX_WEIGHT
    cfg, recipe_kw = recipe(AUX_WEIGHT, smoke=False)
    return FitContext(cfg, recipe_kw, {"a.py": "0" * 64}, {"plan_sha256": PLAN_SHA256, "protocol_sha256": PROTOCOL_SHA256})


def test_fit_identity_changes_with_anything_that_makes_a_fit_different(plan, plan_shots, ctx):
    key = FitKey("england_epl", "A", "with", 0)
    sets = arm_sets(plan_shots, plan, key)
    identity = fit_identity(key, ctx, sets)
    assert fit_identity(key, ctx, sets) == identity and json.loads(json.dumps(identity)) == identity  # survives the record
    assert identity["cfg"]["seed"] == 0 and identity["cfg"]["aux_weight"] == 0.5 and identity["matches"]["train"] == 225
    other_seed = fit_identity(FitKey(*key.path.parts[:3], 1), ctx, sets)
    other_arm = fit_identity(FitKey("england_epl", "A", "without", 0), ctx, arm_sets(plan_shots, plan, FitKey("england_epl", "A", "without", 0)))
    moved = {**sets, "train": sets["train"][~sets["train"].game.eq(sets["train"].game.iloc[0])],
             "select": pd.concat([sets["select"], sets["train"][sets["train"].game.eq(sets["train"].game.iloc[0])]])}
    other_members = fit_identity(key, ctx, moved)
    other_plan = fit_identity(key, FitContext(ctx.cfg, ctx.recipe_kw, ctx.sources, {**ctx.shared_identity, "plan_sha256": "0" * 64}), sets)
    other_recipe = fit_identity(key, FitContext(ctx.cfg, {**ctx.recipe_kw, "backbone_lr_mult": 0.5}, ctx.sources, ctx.shared_identity), sets)
    for changed in (other_seed, other_arm, other_members, other_plan, other_recipe):
        assert changed != identity
    assert other_members["matches_sha256"] != identity["matches_sha256"]


def test_load_shots_keeps_every_shot_and_refuses_missing_frames(tmp_path):
    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for f in frames:
        f.write_bytes(b"x")
    index = pd.DataFrame({"shot_id": ["s1", "s2"], "game": "g", "frame_path": [str(f) for f in frames], "goal": [True, False],
                          "label": ["Shots on target", "Shots off target"], "split": "train", "source": "soccernet"})
    file = tmp_path / "index.parquet"
    index.to_parquet(file)
    shots = load_shots(file)
    assert len(shots) == 2 and shots.on_target.tolist() == [1.0, 0.0]
    frames[1].unlink()
    with pytest.raises(FileNotFoundError, match="1 frames"):
        load_shots(file)


@needs_index
def test_main_check_on_an_empty_output_directory_reports_all_cells_missing(monkeypatch, tmp_path, capsys):
    out = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(sys, "argv", ["run_xgc_transfer.py", "--check", "--out", str(out)])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 1
    assert (out / "training_table.csv").exists() and len(json.loads((out / "deviations.json").read_text())["problems"]) == 40
    assert "verified" in capsys.readouterr().out
