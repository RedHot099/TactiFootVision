"""Control 1 of xg-stack (ticket 01, council C5): every provenance assertion fires."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import provenance as sp
from xgs_fixtures import synthetic_table


@pytest.mark.parametrize("name", ["shooter_by_goal@0", "gk_dist_goal@-400", "label_x", "OUTCOME", "visibility", "statsbomb_xg",
                                  "sb_dist", "is_goal"])
def test_a_banned_fragment_in_a_design_column_raises(name):
    with pytest.raises(sp.ProvenanceError, match="banned"):
        sp.check_names(["z_g_combo", name])


@pytest.mark.parametrize("candidate", ["b", "c", "bc"])
def test_every_candidate_s_arms_pass_the_name_check(candidate):
    spec = sf.spec(candidate, synthetic_table())
    for arm in spec.arms.values():
        sp.check_names(arm.columns)
    for columns in spec.gbm_columns.values():
        sp.check_names(columns)


def test_a_design_that_differs_from_the_frozen_list_raises():
    arm = sf.spec("c", synthetic_table()).arms["G"]
    sp.check_columns(arm, arm.columns, np.zeros((3, len(arm.columns))))
    with pytest.raises(sp.ProvenanceError, match="differ"):
        sp.check_columns(arm, arm.columns[:-1])
    with pytest.raises(sp.ProvenanceError, match="width|columns, the frozen"):
        sp.check_columns(arm, arm.columns, np.zeros((3, len(arm.columns) + 1)))


def plan(extra_evaluation: list[str] = ()) -> pd.DataFrame:
    games = [f"e{i}" for i in range(135)] + [f"g{i}" for i in range(37)] + ["t0", "t1"]
    roles = ["evaluation"] * 135 + ["gate"] * 37 + ["training"] * 2
    return pd.DataFrame({"game": games, "role": roles})


def test_held_out_matches_and_shots_raise():
    index_shots = pd.DataFrame({"shot_id": ["t0_1", "t1_1", "e3_1", "g5_1"], "game": ["t0", "t1", "e3", "g5"]})
    sp.check_held_out(["t0", "t1"], ["t0_1", "t1_1"], plan(), index_shots)
    with pytest.raises(sp.ProvenanceError, match="matches are held out"):
        sp.check_held_out(["t0", "e3"], ["t0_1"], plan(), index_shots)
    with pytest.raises(sp.ProvenanceError, match="shots are from held-out"):
        sp.check_held_out(["t0"], ["t0_1", "g5_1"], plan(), index_shots)


def test_a_plan_without_the_172_held_out_matches_raises():
    short = plan().iloc[1:]
    with pytest.raises(sp.ProvenanceError, match="held-out roles"):
        sp.held_out(short)
    assert len(sp.held_out(plan())) == 172


def labels(team: str = "home") -> list[dict]:
    return [{"gameTime": "1 - 10:00", "label": "Shots off target", "position": "600000", "team": team, "visibility": "visible"},
            {"gameTime": "1 - 09:50", "label": "Corner", "position": "590000", "team": team, "visibility": "visible"},
            {"gameTime": "1 - 09:57", "label": "Shots on target", "position": "597000", "team": team, "visibility": "visible"}]


SHOTS = pd.DataFrame({"game": ["A"], "half": [1], "position_ms": [600000]}, index=["s1"])


def test_the_flag_code_check_passes_on_the_parser_and_fires_on_a_type_dependent_one(monkeypatch):
    sp.check_flag_code(SHOTS, {"A": labels()})
    monkeypatch.setattr(si, "SHOT_TYPES", frozenset({"Shots off target"}))        # a parser that reads the shot's type
    with pytest.raises(sp.ProvenanceError, match="label type"):
        sp.check_flag_code(SHOTS, {"A": labels()})


def test_the_flag_code_check_fires_when_goal_is_read(monkeypatch):
    original = si.parse_annotations

    def keeps_goal(raw):
        return original([dict(a, label="Direct free-kick") if a["label"] == "Goal" else a for a in raw])
    monkeypatch.setattr(si, "parse_annotations", keeps_goal)
    with pytest.raises(sp.ProvenanceError):
        sp.check_flag_code(SHOTS, {"A": labels()})


def test_control_records_every_check_and_fails_on_any(monkeypatch):
    table = synthetic_table(n_matches=4, per_match=5).assign(game=lambda t: np.where(t.fold == 0, "t0", "t1"))
    table["position_ms"] = 600000
    population = si.Population(table, {}, {})
    index_shots = pd.DataFrame({"shot_id": table.index, "game": table.game})
    spec = sf.spec("b", table)
    frozen = {name: arm.columns for name, arm in spec.arms.items()}
    annotations = {"t0": labels(), "t1": labels("away")}
    record = sp.control(population, spec, frozen, plan(), index_shots, annotations)
    assert record["passed"] and {c["check"] for c in record["checks"]} >= {"held_out", "flag_code", "columns:G", "constant:G",
                                                                             "names:gbm_GS"}
    frozen["GS"] = frozen["GS"][:-1]
    record = sp.control(population, spec, frozen, plan(), index_shots, annotations)
    assert not record["passed"] and [c["check"] for c in record["checks"] if not c["passed"]] == ["columns:GS"]
    table["ball_none@0"] = 0.0                                                # a constant input the frozen spec still holds
    record = sp.control(si.Population(table, {}, {}), spec, {n: a.columns for n, a in spec.arms.items()}, plan(), index_shots,
                        annotations)
    assert {c["check"] for c in record["checks"] if not c["passed"]} == {"constant:G", "constant:GS", "constant:Q", "constant:ALL"}
