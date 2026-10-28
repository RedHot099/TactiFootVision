"""xg-temporal leak assertions (ticket 01; spec, "Kontrola przecieku" 1, 3, 4): the offset ceiling, the anchor frame,
cut and undecoded frames, banned names, the control flag, the training role, and `control` over a population."""

import pandas as pd
import pytest
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal import provenance as tp
from xgt_fixtures import base_table, source_frames


def test_the_offset_assertion_fires_on_a_column_at_minus_200_when_o_max_is_minus_600():
    columns = tf.arm_columns(ti.Window(), 4, 4)["T"]
    tp.check_columns(columns, -600)
    for late in ("pc01@-200", "pc03_vs_start@-200", "missing@-400"):
        with pytest.raises(tp.ProvenanceError, match="> o_max -600"):
            tp.check_columns([*columns, late], -600)


def test_the_anchor_frame_and_late_windows_need_the_control_flag():
    with pytest.raises(tp.ProvenanceError, match="control flag"):
        tp.check_columns(["pc01@-400"], -400)
    tp.check_columns(["pc01@-400"], -400, control=True)
    with pytest.raises(tp.ProvenanceError, match="anchor"):
        tp.check_offset(0, 400, False, "column")
    tp.check_columns(["pc01@0", "pc01@400"], 400, control=True)
    with pytest.raises(tp.ProvenanceError, match="> o_max"):
        tp.check_columns(["pc01@400"], 0, control=True)        # the ceiling holds in a control row too


@pytest.mark.parametrize("name", ["goal_dist@-600", "Label@-800", "outcome@-600", "StatsBomb_xg@-600", "sb_x@-600"])
def test_banned_fragments_are_refused_case_insensitively(name):
    with pytest.raises(tp.ProvenanceError, match="banned"):
        tp.check_columns(["z_g_combo", name], -600)


def test_every_column_but_the_reference_carries_an_offset():
    for name in ("pc01", "pc01@", "@-600", "pc01@minus600"):
        with pytest.raises(tp.ProvenanceError, match="<feature>@<offset>"):
            tp.check_columns([name], -600)
    tp.check_columns(["z_g_combo"], -600)


def test_used_frames_are_decoded_uncut_embedded_and_not_after_o_max():
    ok = source_frames([("s", "A", -800, "ok", False, 1, 0), ("s", "A", -1600, "ok", True, 2, 0)])
    ok["used"] = [True, False]                                    # an unused cut frame is fine
    tp.check_frames(ok, -600)
    for row, message in [((-800, "ok", True, 1, 0), "cut"), ((-800, "ok", pd.NA, 1, 0), "cut"),
                         ((-800, "missing", False, 1, 0), "decoded"), ((-800, "ok", False, -1, 0), "embedding"),
                         ((-400, "ok", False, 1, 0), "> o_max"), ((0, "ok", False, 1, 0), "> o_max")]:
        bad = source_frames([("s", "A", *row)])
        bad["used"] = True
        with pytest.raises(tp.ProvenanceError, match=message):
            tp.check_frames(bad, -600)


def test_roles_refuse_a_held_out_or_unknown_match():
    plan = pd.DataFrame({"game": ["A", "B", "E", "G"], "role": ["training", "training", "evaluation", "gate"]})
    tp.check_roles(["A", "B", "A"], plan)
    plan = pd.concat([plan, pd.DataFrame({"game": ["O"], "role": ["other"]})])
    for games, message in ((["A", "E"], "held out"), (["G"], "held out"), (["A", "X"], "not in the plan"),
                           (["O"], "outside the training role")):
        with pytest.raises(tp.ProvenanceError, match=message):
            tp.check_roles(games, plan)


def test_control_records_every_check_and_fails_on_a_held_out_match():
    base = base_table([("a1", "A", 0, -1.0), ("e1", "E", 1, -1.0)])
    rows = [(s, g, o, "ok", False, i * 20 + j, f) for i, (s, g, f) in enumerate([("a1", "A", 0), ("e1", "E", 1)])
            for j, o in enumerate(ti.GRID)]
    population = ti.population_from(base, source_frames(rows), ti.Window(), ti.SOURCES["g_xgv"])
    columns = tf.arm_columns(population.window, 4, 4)
    training = pd.DataFrame({"game": ["A", "E"], "role": ["training", "training"]})
    record = tp.control(population, columns, training)
    assert record["passed"] and [c["check"] for c in record["checks"]] == ["window", "columns:R", "columns:S", "columns:T",
                                                                           "frames", "roles"]
    held = pd.DataFrame({"game": ["A", "E"], "role": ["training", "evaluation"]})
    record = tp.control(population, columns | {"X": ["pc01@-200"]}, held)
    failed = {c["check"] for c in record["checks"] if not c["passed"]}
    assert not record["passed"] and failed == {"columns:X", "roles"}
