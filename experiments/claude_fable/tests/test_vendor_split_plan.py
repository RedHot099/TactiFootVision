"""Retraining split plan (xg-vendor ticket 05, test seam 1) on a fixture index whose answers are known by construction.

The fixture world has two leagues over two seasons. Matches are named so that every home club plays several home matches.
Some matches are overlap matches (paired in the match map); the official split is train, valid or test.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_split_plan as script  # noqa: E402
from xgpoc.vendor import split_check as sc  # noqa: E402
from xgpoc.vendor import split_plan as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {"england_epl": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"], "spain_laliga": ["Eta", "Theta", "Iota", "Kappa", "Lambda"]}
SEASONS = ("2014-2015", "2015-2016")


def fixture_world(n_per_pair: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(index, match map). Every ordered pair of clubs of a league plays in each season; the index holds 2 shots a match."""
    rng = np.random.default_rng(7)
    rows, mapped = [], []
    for league, clubs in LEAGUES.items():
        for season in SEASONS:
            for i, home in enumerate(clubs):
                for away in clubs:
                    if home == away:
                        continue
                    game = f"{league}/{season}/2015-{1 + i:02d}-{1 + clubs.index(away):02d} - 18-00 {home} 1 - 0 {away}"
                    split = rng.choice(["train", "valid", "test"], p=[0.5, 0.2, 0.3])
                    rows += [{"shot_id": f"{game}#{k}", "game": game, "split": split, "goal": int(k == 0)} for k in range(2)]
                    mapped.append({"game": game, "reason": "paired" if rng.random() < 0.3 else "no StatsBomb match within 1 day"})
    return pd.DataFrame(rows), pd.DataFrame(mapped)


@pytest.fixture(scope="module")
def world():
    return fixture_world()


@pytest.fixture(scope="module")
def plan(world):
    return sp.build_plan(*world, seed=3, n_gate=10)


def test_roles_are_disjoint_and_complete(world, plan):
    index, match_map = world
    assert sorted(plan.game) == sorted(index.game.unique())
    assert plan.game.is_unique
    assert set(plan.role) == set(sp.ROLES)
    overlap = set(match_map.game[match_map.reason == "paired"])
    assert set(plan.game[plan.role == "evaluation"]) == overlap
    assert len(plan[plan.role == "gate"]) == 10
    assert len(plan) == plan.role.value_counts().sum()
    # an evaluation match is neither in training nor in the gate: the three role sets partition the matches
    sets = [set(plan.game[plan.role == r]) for r in sp.ROLES]
    assert sum(map(len, sets)) == len(set().union(*sets)) == len(plan)


def test_gate_comes_from_non_overlap_official_test_matches(world, plan):
    overlap = set(world[1].game[world[1].reason == "paired"])
    gate = plan[plan.role == "gate"]
    assert (gate.official_split == "test").all()
    assert not set(gate.game) & overlap
    # the training matches hold every other match, including official-test matches the gate did not draw
    assert set(plan.game[plan.role == "training"]) == set(plan.game) - overlap - set(gate.game)
    assert (plan[plan.role == "training"].official_split == "test").any()


def test_gate_is_stratified_by_league_and_season(world, plan):
    pool = plan[(plan.official_split == "test") & (plan.role != "evaluation")]
    strata = sp.stratum_of(pool)
    sizes = strata.value_counts()
    drawn = sp.stratum_of(plan[plan.role == "gate"]).value_counts().reindex(sizes.index, fill_value=0)
    exact = 10 * sizes / sizes.sum()
    assert (drawn - exact).abs().max() < 1   # largest remainder: floor or ceil of the proportional share
    assert (drawn >= np.floor(exact)).all()
    assert (drawn <= sizes).all()
    assert drawn.sum() == 10 and len(sizes) == 4


def test_largest_remainder_known_answers():
    assert sp.largest_remainder({"a": 10, "b": 6, "c": 4}, 5) == {"a": 3, "b": 1, "c": 1}   # tie on the remainder: larger stratum
    assert sp.largest_remainder({"a": 3, "b": 3}, 3) == {"a": 2, "b": 1}                     # tie, equal sizes: the name
    assert sp.largest_remainder({"a": 1, "b": 9}, 5) == {"a": 0, "b": 5}                     # shares 0.5 and 4.5: tie, larger stratum
    assert sp.largest_remainder({"a": 3, "b": 7}, 4) == {"a": 1, "b": 3}                     # shares 1.2 and 2.8: b has the larger remainder
    assert sp.largest_remainder({"a": 7, "b": 3}, 10) == {"a": 7, "b": 3}                    # whole pool
    assert sp.largest_remainder({"a": 4, "b": 4}, 0) == {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="Cannot draw"):
        sp.largest_remainder({"a": 2}, 3)


def test_allocation_does_not_exceed_strata_and_sums_to_total():
    rng = np.random.default_rng(0)
    for _ in range(200):
        sizes = {f"s{i}": int(n) for i, n in enumerate(rng.integers(1, 9, size=rng.integers(1, 12)))}
        total = int(rng.integers(0, sum(sizes.values()) + 1))
        alloc = sp.largest_remainder(sizes, total)
        assert sum(alloc.values()) == total
        assert all(0 <= alloc[s] <= sizes[s] for s in sizes)


def test_plan_is_identical_for_the_same_seed_and_changes_with_it(world, plan):
    again = sp.build_plan(*world, seed=3, n_gate=10)
    assert sp.plan_text(plan) == sp.plan_text(again)
    shuffled = sp.build_plan(world[0].sample(frac=1, random_state=1), world[1].sample(frac=1, random_state=2), seed=3, n_gate=10)
    assert sp.plan_text(plan) == sp.plan_text(shuffled)   # a function of the content, not the row order
    other = sp.build_plan(*world, seed=4, n_gate=10)
    assert set(other.game[other.role == "gate"]) != set(plan.game[plan.role == "gate"])


def test_folds_are_grouped_by_home_team(plan):
    training = plan[plan.role == "training"]
    assert set(training.fold) == set(range(sp.N_FOLDS))
    assert (training.groupby("home").fold.nunique() == 1).all()          # a home club is in one fold
    assert (plan[plan.role != "training"].fold == sp.NO_FOLD).all()
    sizes = training.fold.value_counts()
    assert sizes.max() - sizes.min() <= 2 * training.groupby("home").size().max()   # GroupKFold balances the groups
    assert (training.home == training.game.str.extract(sp.HOME_RE)[0]).all()


def test_home_folds_do_not_depend_on_input_order():
    games = pd.Series([f"l/s/2015-01-0{d} - 18-00 {h} 1 - 0 X{d}" for d, h in zip(range(1, 9), "ABCDEFGH", strict=True)])
    a, b = sp.home_folds(games), sp.home_folds(games.iloc[::-1])
    assert a.equals(b)
    with pytest.raises(ValueError, match="home teams cannot fill"):
        sp.home_folds(games.iloc[:3])


def test_match_table_refuses_inconsistent_inputs(world):
    index, match_map = world
    with pytest.raises(ValueError, match="different matches"):
        sp.match_table(index, match_map.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate matches"):
        sp.match_table(index, pd.concat([match_map, match_map.iloc[:1]]))
    two_splits = index.copy()
    two_splits.loc[two_splits.index[0], "split"] = "other"
    with pytest.raises(ValueError, match="more than one official split"):
        sp.match_table(two_splits, match_map)
    with pytest.raises(ValueError, match="Cannot draw"):
        sp.build_plan(index, match_map, seed=0, n_gate=10_000)
    bad = {"game": "league/2015 - no season level", "reason": "paired"}
    with pytest.raises(ValueError, match="league/season/match"):
        sp.match_table(pd.DataFrame([{"game": bad["game"], "split": "test", "shot_id": "s"}]), pd.DataFrame([bad]))


@pytest.mark.skipif(not (ROOT / "cache/folds_v12_home5.csv").exists(), reason="archived folds file absent")
def test_folds_reproduce_the_archived_construction_on_the_archived_pool():
    archived = pd.read_csv(ROOT / "cache/folds_v12_home5.csv")
    rebuilt = sp.home_folds(archived.game)
    assert rebuilt.equals(archived[["game", "home", "fold"]].sort_values("game", ignore_index=True))


@pytest.mark.parametrize(("d_ll", "d_auc", "ll", "auc"), [
    (0.010, 0.0, True, True), (0.0101, 0.0, False, True), (-0.05, -0.020, True, True),
    (0.0, -0.0201, True, False), (0.02, -0.03, False, False), (0.0, 0.05, True, True)])
def test_gate_tolerance_is_applied_to_point_estimates(d_ll, d_auc, ll, auc):
    verdict = sc.gate_passes(d_ll, d_auc)
    assert verdict == {"log_loss": ll, "auc": auc, "passed": ll and auc}


def test_tolerance_is_fixed_in_the_plan():
    assert sp.GATE_TOLERANCE == {"log_loss": 0.010, "auc": 0.020}


# ---- the written plan and the check the retrain runs --------------------------------------------------------------------------

@pytest.fixture
def written(tmp_path, monkeypatch, world):
    """The plan of the fixture world written by the script into tmp_path, with the check's paths pointing there."""
    index, match_map = world
    index = index.assign(goal=index.goal.astype(int))
    index.to_parquet(tmp_path / "index.parquet")
    match_map.to_csv(tmp_path / "match_map.csv", index=False)
    archived = index[["shot_id", "goal"]].assign(p=np.linspace(0.05, 0.6, len(index)))
    archived.to_csv(tmp_path / "archived.csv", index=False)
    pairing = {"files": {"match_map.csv": script.file_sha256(tmp_path / "match_map.csv")}}
    (tmp_path / "pairing.json").write_text(json.dumps(pairing))
    monkeypatch.setattr(script, "INDEX", tmp_path / "index.parquet")
    for name, value in {"MATCH_MAP": tmp_path / "match_map.csv", "PAIRING_COMPLETE": tmp_path / "pairing.json",
                        "ARCHIVED_G_COMBO": tmp_path / "archived.csv", "FOLDS": tmp_path / "cache/folds.csv",
                        "PLAN_DIR": tmp_path / "plan", "EXPECTED_COUNTS": {r: int(n) for r, n in
                                                                            sp.build_plan(index, match_map, 0, 10).role.value_counts().items()}}.items():
        monkeypatch.setattr(sc, name, value)
    monkeypatch.setattr(sp, "N_GATE", 10)
    script.main(["--out", str(tmp_path / "plan")])
    sha = hashlib.sha256((tmp_path / "plan/plan.csv").read_bytes()).hexdigest()
    kwargs = {"index_path": tmp_path / "index.parquet", "match_map_path": tmp_path / "match_map.csv", "plan_dir": tmp_path / "plan",
              "folds_path": tmp_path / "cache/folds.csv", "archived_path": tmp_path / "archived.csv", "expected_plan": sha,
              "expected_counts": sc.EXPECTED_COUNTS}
    return tmp_path, kwargs, sha


def test_verify_plan_returns_the_approved_plan(written, world):
    tmp, kwargs, sha = written
    plan, record = sc.verify_plan(**kwargs)
    assert plan.equals(pd.read_csv(tmp / "plan/plan.csv"))
    assert record["plan_csv_sha256"] == sha
    assert record["gate_tolerance"]["log_loss"] == 0.010 and record["gate_tolerance"]["auc"] == 0.020
    assert record["archived_g_combo"]["gate"]["shots"] == 20
    assert sum(v["gate"] for k, v in record["gate_allocation"].items() if k != "_total") == 10
    assert pd.read_csv(kwargs["folds_path"]).equals(sp.training_folds(plan))
    script.main(["--out", str(tmp / "plan")])   # a finished folder is verified, not rebuilt


def test_verify_plan_refuses_any_other_plan_hash(written):
    _, kwargs, sha = written
    with pytest.raises(ValueError, match="not the approved one"):
        sc.verify_plan(**{**kwargs, "expected_plan": "0" * 64})
    assert sc.verify_plan(**kwargs)[1]["plan_csv_sha256"] == sha


@pytest.mark.parametrize("target", ["plan/plan.csv", "plan/plan.json", "cache/folds.csv", "archived.csv", "match_map.csv", "index.parquet"])
def test_verify_plan_refuses_a_changed_file(written, target):
    tmp, kwargs, _ = written
    with (tmp / target).open("ab") as f:
        f.write(b"\n")
    with pytest.raises(ValueError, match="changed since|hashes to|not in the completion"):
        sc.verify_plan(**kwargs)


def test_verify_plan_refuses_wrong_counts_and_unlisted_files(written):
    tmp, kwargs, _ = written
    with pytest.raises(ValueError, match="role counts"):
        sc.verify_plan(**{**kwargs, "expected_counts": {"evaluation": 1, "gate": 1, "training": 1}})
    (tmp / "plan/extra.txt").write_text("x")
    with pytest.raises(ValueError, match="not in the completion record"):
        sc.verify_plan(**kwargs)


def test_archived_predictions_must_cover_the_gate(written, world):
    index, _ = world
    plan, _ = sc.verify_plan(**written[1])
    archived = pd.read_csv(written[0] / "archived.csv")
    assert sc.archived_on_gate(index, plan, archived)["shots"] == 20
    with pytest.raises(ValueError, match="no archived prediction"):
        sc.archived_on_gate(index, plan, archived.iloc[:-30])
    with pytest.raises(ValueError, match="labels differ"):
        sc.archived_on_gate(index, plan, archived.assign(goal=1 - archived.goal))


def test_script_refuses_a_replacement_and_unexpected_counts(written, monkeypatch):
    tmp, _, _ = written
    (tmp / "plan/complete.json").unlink()
    with pytest.raises(SystemExit, match="refusing to replace"):
        script.main(["--out", str(tmp / "plan")])
    monkeypatch.setattr(sc, "EXPECTED_COUNTS", {"evaluation": 1, "gate": 1, "training": 1})
    with pytest.raises(SystemExit, match="STOPPED"):
        script.main(["--out", str(tmp / "other")])
    assert not (tmp / "other").exists()


@pytest.mark.skipif(not (sc.PLAN_DIR / "complete.json").exists(), reason="split plan not written")
def test_written_plan_is_the_approved_one_with_the_expected_roles():
    plan, record = sc.verify_plan()
    assert plan.role.value_counts().to_dict() == sc.EXPECTED_COUNTS
    by_split = plan.groupby(["role", "official_split"]).size().to_dict()
    assert by_split == {("evaluation", "test"): 32, ("evaluation", "train"): 72, ("evaluation", "valid"): 31, ("gate", "test"): 37,
                        ("training", "test"): 31, ("training", "train"): 228, ("training", "valid"): 69}
    assert record["archived_g_combo"]["gate"]["shots"] == 825
