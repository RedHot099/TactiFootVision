"""Pre-registration of the vendor comparison (xg-vendor ticket 14): parameter block, hash list, records, table shells,
the dry run's permutations and labels."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgpoc.vendor import dryrun, labels, shells
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import prereg as pr
from xgpoc.vendor.prereg import PreRegistrationError

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "research/xg_vendor_preregistration.md"
sys.path.insert(0, str(ROOT / "scripts"))
from vendor_dryrun import dryrun_table  # noqa: E402
from vendor_probes import probe_subsets  # noqa: E402

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
THRESHOLDS = {"dist": 0.5, "angle": 0.25, "ff_def_cone": 0.25, "def_cone_ge1": 0.7, "def_cone_ge2": 0.7, "gk_depth": 0.25,
              "gk_off_line_abs": 0.25, "gk_dist_goal": 0.25, "ff_opp_3m": 0.25, "head": 0.7}


def parameters(**changes) -> dict:
    pins = {name: {"path": None, "sha256": None} for name in pr.REQUIRED_PINS}
    return {"margin": None, "dry_run_margin": 0.005, "n_boot": 5000, "fold_seed": 0, "boot_seed": 0, "permutation_seed": 0,
            "offset_logits": "calibrated", "h3_thresholds": THRESHOLDS, "h3_confirmatory_targets": ["dist", "def_cone_ge1"], "pins": pins} | changes


def document(params: dict, prose: str = "") -> str:
    return (f"# doc\n\n{prose}\n\n{pr.PARAMETERS_BEGIN}\n```json\n{json.dumps(params, indent=2)}\n```\n{pr.PARAMETERS_END}\n\n"
            f"{pr.HASHES_BEGIN}\n{pr.HASHES_END}\n")


# ------------------------------------------------------------------ parameter block
def test_the_document_parameter_block_parses_and_lists_its_placeholders():
    text = DOC.read_text()
    params = pr.read_parameters(text)
    assert params["n_boot"] >= 5000 and params["offset_logits"] == "calibrated"
    assert params["pins"]["primary_label_free_table"]["sha256"].startswith("829aa7ed")
    assert set(params["h3_confirmatory_targets"]) == {"dist", "def_cone_ge1"}
    open_items = pr.placeholders(text, params)
    if params["margin"] is None:   # the draft round: delta and the combined table are still open
        assert "margin" in open_items and "pin combined_label_free_table" in open_items


@pytest.mark.parametrize(("change", "message"), [
    ({"margin": 0.02}, "margin must be a log-loss margin"),
    ({"margin": 5}, "margin must be a log-loss margin"),
    ({"n_boot": 1000}, "at least 5,000"),
    ({"offset_logits": "both"}, "offset_logits"),
    ({"h3_thresholds": {"dist": 0.5}}, "h3_thresholds"),
    ({"h3_confirmatory_targets": ["xg"]}, "unknown confirmatory"),
])
def test_the_parameter_block_refuses_bad_values(change, message):
    with pytest.raises(PreRegistrationError, match=message):
        pr.read_parameters(document(parameters(**change)))


def test_a_pin_needs_both_path_and_hash_and_the_block_needs_its_markers():
    params = parameters()
    params["pins"]["split_plan"] = {"path": "plan.csv", "sha256": None}
    with pytest.raises(PreRegistrationError, match="both set or both null"):
        pr.read_parameters(document(params))
    with pytest.raises(PreRegistrationError, match="exactly one"):
        pr.read_parameters(document(parameters()).replace(pr.PARAMETERS_END, ""))


def test_placeholder_marks_in_the_generated_block_do_not_count():
    params = parameters(margin=0.003)
    text = document(params, prose=f"tekst {pr.PLACEHOLDER_TOKEN}")
    assert pr.placeholders(text, params)[-1] == f"1 '{pr.PLACEHOLDER_TOKEN}' marks in the document"
    inside = pr.replace_hash_block(document(params), f"\n{pr.PLACEHOLDER_TOKEN}\n")
    assert not any("marks" in item for item in pr.placeholders(inside, params))


# ------------------------------------------------------------------ code closure and hash list
def make_tree(root: Path) -> None:
    files = {"xgpoc/__init__.py": "", "xgpoc/vendor/__init__.py": "", "xgpoc/vendor/a.py": "from xgpoc.vendor import b\nimport numpy\n",
             "xgpoc/vendor/b.py": "from xgpoc.other import helper\n", "xgpoc/other.py": "def helper():\n    return 1\n",
             "xgpoc/vendor/unused.py": "", "scripts/entry.py": "from xgpoc.vendor import a\nfrom xgpoc.vendor.b import thing\n"}
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body)


def test_the_code_closure_follows_xgpoc_imports_transitively(tmp_path):
    make_tree(tmp_path)
    found = [str(p.relative_to(tmp_path)) for p in pr.code_closure(tmp_path, ("scripts/entry.py",))]
    assert found == ["scripts/entry.py", "xgpoc/__init__.py", "xgpoc/other.py", "xgpoc/vendor/__init__.py", "xgpoc/vendor/a.py", "xgpoc/vendor/b.py"]


def test_the_real_closure_covers_the_analysis_modules():
    found = {str(p.relative_to(ROOT)) for p in pr.code_closure(ROOT)}
    assert {"xgpoc/vendor/ladder.py", "xgpoc/vendor/probes.py", "xgpoc/vendor/leakage.py", "xgpoc/vendor/evaltable.py",
            "xgpoc/vendor/draws.py", "xgpoc/probe/readability.py", "xgpoc/vendor/prereg.py", "scripts/vendor_dryrun.py"} <= found


def hashed_tree(tmp_path: Path, margin=None) -> tuple[Path, dict]:
    make_tree(tmp_path)
    (tmp_path / "plan.csv").write_text("match,role\n")
    (tmp_path / "rule.md").write_text("rule\n")
    params = parameters(margin=margin)
    params["pins"]["split_plan"] = {"path": "plan.csv", "sha256": hashlib.sha256(b"match,role\n").hexdigest()}
    params["pins"]["timing_rule"] = {"path": "rule.md", "sha256": hashlib.sha256(b"rule\n").hexdigest()}
    doc = tmp_path / "doc.md"
    doc.write_text(document(params))
    return doc, params


def test_the_hash_list_verifies_pins_and_is_deterministic(tmp_path):
    doc, params = hashed_tree(tmp_path)
    hashes = pr.hash_list(tmp_path, doc, ("scripts/entry.py",))
    assert hashes["status"] == "draft" and "margin" in hashes["placeholders"] and "evidence dry_run_primary" in hashes["placeholders"]
    assert hashes["pins"]["split_plan"]["sha256"] == params["pins"]["split_plan"]["sha256"] and len(hashes["code"]) == 6
    assert pr.hash_list_bytes(hashes) == pr.hash_list_bytes(pr.hash_list(tmp_path, doc, ("scripts/entry.py",)))
    (tmp_path / "plan.csv").write_text("changed\n")
    with pytest.raises(PreRegistrationError, match="pin split_plan"):
        pr.hash_list(tmp_path, doc, ("scripts/entry.py",))


def test_the_hash_block_is_regenerated_in_place(tmp_path):
    doc, _ = hashed_tree(tmp_path)
    hashes = pr.hash_list(tmp_path, doc, ("scripts/entry.py",))
    block = pr.render_hash_block(hashes, "hashes.json", "f" * 64)
    once = pr.replace_hash_block(doc.read_text(), block)
    assert pr.replace_hash_block(once, block) == once and "`" + "f" * 64 + "`" in once and "SZKIC" in once
    assert pr.read_parameters(once) == pr.read_parameters(doc.read_text())


# ------------------------------------------------------------------ records
def test_a_draft_record_uses_the_dry_run_margin_while_delta_is_open(tmp_path):
    doc, params = hashed_tree(tmp_path)
    with pytest.raises(PreRegistrationError, match="still a placeholder"):
        pr.draft_record(doc, "primary_label_free_table", NOW)
    params["pins"]["primary_label_free_table"] = {"path": "t.parquet", "sha256": "c" * 64}
    doc.write_text(document(params))
    record = pr.draft_record(doc, "primary_label_free_table", NOW)
    assert record.draft and record.margin == 0.005 and record.label_free_table_sha256 == "c" * 64
    assert record.sha256 == hashlib.sha256(doc.read_bytes()).hexdigest()
    params["margin"] = 0.002
    doc.write_text(document(params))
    assert pr.draft_record(doc, "primary_label_free_table", NOW).margin == 0.002


def test_a_confirmatory_record_needs_a_complete_hash_list(tmp_path):
    doc, params = hashed_tree(tmp_path, margin=0.003)
    hashes = pr.hash_list(tmp_path, doc, ("scripts/entry.py",))
    with pytest.raises(PreRegistrationError, match="draft"):
        pr.confirmatory_record(doc, hashes, NOW, "abc")
    params["pins"]["combined_label_free_table"] = {"path": "c.parquet", "sha256": "d" * 64}
    complete = hashes | {"status": "complete", "placeholders": [], "parameters": params}
    record = pr.confirmatory_record(doc, complete, NOW, "abc")
    assert not record.draft and record.margin == 0.003 and record.label_free_table_sha256 == "d" * 64 and record.commit == "abc"


# ------------------------------------------------------------------ shells
def test_the_empty_shells_are_printed_verbatim_in_the_document():
    text = DOC.read_text()
    params = pr.read_parameters(text)
    for (shell, population), body in shells.shell_bodies(params).items():
        assert body in text, f"shell {shell} ({population}) differs from the document"


@pytest.mark.parametrize(("p_tost", "p_sup", "lo95", "reading"), [
    (0.01, 0.01, -0.003, "poprawa istotna, mniejsza niż δ"),
    (0.20, 0.01, -0.006, "klatka dodaje; równoważności przy δ brak"),
    (0.01, 0.90, 0.0005, "równoważne przy δ; stos istotnie gorszy"),
    (0.01, 0.40, -0.002, "równoważne przy δ: klatka nie dodaje więcej niż δ"),
    (0.30, 0.99, 0.001, "stos istotnie gorszy; równoważności przy δ brak"),
    (0.30, 0.30, -0.004, "nierozstrzygnięty"),
])
def test_h2_reading_follows_the_decision_table(p_tost, p_sup, lo95, reading):
    assert shells.h2_reading({"p_tost": p_tost, "p_superiority": p_sup, "lo95": lo95}) == reading


def test_h3_reading_is_decided_by_the_lower_end():
    assert shells.h3_reading(0.70, 0.52, 0.80, 0.50) == "koduje"
    assert shells.h3_reading(0.55, 0.45, 0.62, 0.50) == "nierozstrzygnięty"   # the point estimate alone does not decide
    assert shells.h3_reading(0.40, 0.30, 0.49, 0.50) == "nie koduje (poniżej progu)"
    assert shells.h3_reading(-0.001, -0.01, 0.0, None, control=True) == "kontrola przechodzi"
    assert shells.h3_reading(0.004, -0.01, 0.02, None, control=True) == "kontrola NIE przechodzi"
    assert [shells.difference_reading(*ci) for ci in ((0.01, 0.1), (-0.1, -0.01), (-0.01, 0.02))] == [
        "G > DINOv2", "G < DINOv2", "brak rozróżnialnej różnicy"]


def test_a_failed_control_invalidates_every_h3_reading_of_its_encoder():
    """The hard gate of the shuffled-target control: no "koduje" and no difference reading once a control decodes."""
    assert shells.h3_reading(0.70, 0.52, 0.80, 0.50, gate_passed=False) == shells.INVALID_READING
    assert shells.h3_reading(0.20, 0.10, 0.30, None, control=True, gate_passed=False) == "kontrola NIE przechodzi"
    assert shells.difference_reading(0.01, 0.1, gate_passed=False) == shells.INVALID_READING


# ------------------------------------------------------------------ labels and permutations
def eval_rows(n_matches=4, per_match=12) -> pd.DataFrame:
    n = n_matches * per_match
    rng = np.random.default_rng(5)
    population = np.where(np.arange(n) % 6 == 5, "statsbomb_only", "primary")
    return pd.DataFrame({"key": [f"k{i:03d}" for i in range(n)], "statsbomb_event_id": [f"e{i:03d}" for i in range(n)],
                         "match": np.repeat([f"m{k}" for k in range(n_matches)], per_match), "population": population,
                         "sb_shot_type": np.where(np.arange(n) % 11 == 0, "Penalty", "Open Play"), "goal": rng.random(n) < 0.3})


def label_rows(rows: pd.DataFrame) -> pd.DataFrame:
    primary = rows.population.eq("primary").to_numpy()
    return pd.DataFrame({"key": rows.key, "goal_sn": rows.goal.to_numpy(),
                         "on_target_sn": pd.array(np.where(primary, np.arange(len(rows)) % 2 == 0, None), dtype="boolean"),
                         "sb_outcome": np.where(rows.goal, "Goal", "Saved"), "is_penalty": rows.sb_shot_type.eq("Penalty").to_numpy(),
                         "label_exclusion": None})


def test_labels_are_permuted_within_match_and_penalty_strata():
    rows = eval_rows()
    original = label_rows(rows)
    permuted = dryrun.permute_labels(original, rows, seed=0)
    assert permuted.key.equals(original.key) and permuted.is_penalty.equals(original.is_penalty)
    assert permuted.on_target_sn.isna().equals(original.on_target_sn.isna())
    strata = [rows.match, original.is_penalty]
    for column in ("goal_sn", "sb_outcome", "on_target_sn"):
        assert (permuted.groupby(strata)[column].apply(lambda s: sorted(s.dropna())).equals(
            original.groupby(strata)[column].apply(lambda s: sorted(s.dropna()))))
    assert (permuted.goal_sn == (permuted.sb_outcome == "Goal")).all()   # goal and outcome move together
    assert not permuted.goal_sn.equals(original.goal_sn)
    assert dryrun.permute_labels(original, rows, seed=0).equals(permuted)
    assert not dryrun.permute_labels(original, rows, seed=1).equals(permuted)


def test_shot_payloads_are_permuted_within_matches():
    rows = eval_rows()
    shots = pd.DataFrame({"event_id": [*rows.statsbomb_event_id, "other"], "x": np.arange(len(rows) + 1, dtype=float),
                          "y": 40.0, "body_part": "Right Foot", "freeze_frame_json": "[]"})
    out = dryrun.permute_shots(shots, rows, seed=0)
    assert out.event_id.tolist() == rows.statsbomb_event_id.tolist()
    penalty = rows.sb_shot_type.eq("Penalty").to_numpy()
    for match in rows.match.unique():
        for flag in (True, False):
            pick = (rows.match.eq(match) & (penalty == flag)).to_numpy()
            assert sorted(out.x[pick]) == sorted(shots.x[:len(rows)][pick])
    assert not np.array_equal(out.x.to_numpy(), shots.x[:len(rows)].to_numpy())


def test_primary_labels_follow_decision_d4():
    rows = eval_rows(n_matches=1, per_match=4).assign(population="primary")
    pairs = pd.DataFrame({"shot_id": [*rows.key, "unpaired"], "paired": [True, True, True, True, False],
                          "label": ["Shots on target", "Shots off target", "Shots on target", "Shots on target", "Shots on target"],
                          "goal": [True, False, False, True, False], "is_penalty": [False, False, False, True, False],
                          "sb_event_id": [*rows.statsbomb_event_id, None], "sb_outcome": ["Goal", "Off T", "Blocked", "Goal", "Saved"]})
    out = labels.primary_labels(rows, pairs)
    assert out.goal_sn.tolist() == [True, False, False, True]
    assert out.on_target_sn.tolist() == [True, False, True, True]   # a blocked shot is on target in SoccerNet
    assert out.is_penalty.tolist() == [False, False, False, True]
    with pytest.raises(ValueError, match="another StatsBomb event"):
        labels.primary_labels(rows, pairs.assign(sb_event_id=["x", *pairs.sb_event_id[1:]]))
    with pytest.raises(ValueError, match="primary rows only"):
        labels.primary_labels(rows.assign(population="statsbomb_only"), pairs)


def secondary_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four secondary rows (a recovered goal that is a penalty, a StatsBomb-only save, a StatsBomb-only goal without any
    SoccerNet goal, a StatsBomb-only block) and the pairing's population table for their events, plus an unrelated event."""
    rows = pd.DataFrame({"key": ["e1", "e2", "e3", "e4"], "statsbomb_event_id": ["e1", "e2", "e3", "e4"], "match": "m0",
                         "population": ["recovered_goal", "statsbomb_only", "statsbomb_only", "statsbomb_only"]})
    unpaired = pd.DataFrame({"event_id": ["e3", "e1", "e2", "e4", "e9"],
                             "population": ["secondary", "recovered_goal", "secondary", "secondary", "possible_mispair"],
                             "outcome": ["Goal", "Goal", "Saved", "Blocked", "Goal"], "penalty": [False, True, False, False, False],
                             "sb_goal_without_sn_goal": [True, False, False, False, False]})
    return rows, unpaired


def test_secondary_labels_follow_decision_d4_and_exclude_statsbomb_goals_without_a_soccernet_goal():
    rows, unpaired = secondary_rows()
    out = labels.secondary_labels(rows, unpaired)
    assert out.key.tolist() == ["e1", "e2", "e3", "e4"] and list(out.columns) == ["key", *et.LABEL_COLUMNS]
    assert out.goal_sn.tolist() == [True, False, False, False]     # only the recovery attributes a SoccerNet goal
    assert out.sb_outcome.tolist() == ["Goal", "Saved", "Goal", "Blocked"]
    assert out.is_penalty.tolist() == [True, False, False, False]  # the StatsBomb penalty flag, a pre-shot condition
    assert out.on_target_sn.isna().all()
    assert out.label_exclusion.tolist() == [None, None, "sb_goal_without_sn_goal", None]
    with pytest.raises(ValueError, match="population"):
        labels.secondary_labels(rows.assign(population="recovered_goal"), unpaired)
    with pytest.raises(ValueError, match="not in the pairing"):
        labels.secondary_labels(rows, unpaired[unpaired.event_id.ne("e2")])
    with pytest.raises(ValueError, match="flag disagrees"):
        labels.secondary_labels(rows, unpaired.assign(sb_goal_without_sn_goal=False))
    with pytest.raises(ValueError, match="secondary rows only"):
        labels.secondary_labels(rows.assign(population="primary"), unpaired)


def test_table_labels_join_both_rules_in_table_order():
    primary = eval_rows(n_matches=1, per_match=2).assign(population="primary")
    pairs = pd.DataFrame({"shot_id": primary.key, "paired": True, "label": "Shots on target", "goal": [True, False],
                          "is_penalty": False, "sb_event_id": primary.statsbomb_event_id, "sb_outcome": ["Goal", "Saved"]})
    secondary, unpaired = secondary_rows()
    table = pd.concat([secondary.iloc[:2], primary[["key", "statsbomb_event_id", "match", "population"]], secondary.iloc[2:]],
                      ignore_index=True)
    out = labels.table_labels(table, pairs, unpaired)
    assert out.key.tolist() == table.key.tolist()
    assert out.goal_sn.tolist() == [True, False, True, False, False, False]
    assert out.on_target_sn.isna().tolist() == [True, True, False, False, True, True]
    assert out.label_exclusion.notna().tolist() == [False, False, False, False, True, False]
    assert labels.table_labels(primary, pairs, None).goal_sn.tolist() == [True, False]
    with pytest.raises(ValueError, match="needs the pairing's population table"):
        labels.table_labels(table, pairs, None)


def test_probes_leave_out_penalties_and_secondary_rows():
    rows = eval_rows().assign(split="train")
    subsets, left_out = probe_subsets(rows, by_split=False)
    keep = rows.population.eq("primary") & rows.sb_shot_type.ne("Penalty")
    assert np.array_equal(subsets["all"], keep.to_numpy())
    assert left_out == {"penalties": int((rows.population.eq("primary") & rows.sb_shot_type.eq("Penalty")).sum()),
                        "secondary_rows": int(rows.population.ne("primary").sum())}


def test_the_dryrun_table_changes_only_its_kind():
    rows = eval_rows().assign(table_kind="confirmatory")
    assert dryrun_table(rows).table_kind.eq("dryrun").all()


def probe_outputs(control_r2: float) -> dict:
    targets = ["dist_shuffled", *THRESHOLDS]
    metrics = pd.DataFrame([{"encoder": e, "subset": "all", "coverage": "own", "target": t, "n": 100, "value": control_r2 if t == "dist_shuffled" and
                             e == "emb_g_fold0" else (-0.001 if t == "dist_shuffled" else 0.8), "lo": 0.75, "hi": 0.85,
                             "negative_control": t == "dist_shuffled"} for e in ("emb_g_fold0", "emb_dinov2") for t in targets])
    differences = pd.DataFrame({"encoder": "emb_g_fold0", "subset": "all", "target": targets, "difference": 0.05, "lo": 0.01, "hi": 0.09})
    return {"metrics": metrics, "differences": differences}


def test_filled_probe_shells_obey_the_control_gate():
    params = parameters()
    clean = {row[0]: row for row in shells._probe_rows(probe_outputs(-0.002), "emb_g_fold0", params)}
    assert clean["dist"][8] == "koduje" and clean["dist"][9] == "G > DINOv2" and clean["dist"][7] == "konfirmacyjny"
    leaky = {row[0]: row for row in shells._probe_rows(probe_outputs(0.3), "emb_g_fold0", params)}
    assert leaky["dist_shuffled"][8] == "kontrola NIE przechodzi"
    assert all(leaky[t][8] == leaky[t][9] == shells.INVALID_READING for t in THRESHOLDS)


def test_filled_penalty_shell_reports_the_penalties_apart():
    rung = {"n": 3, "mean_p": 0.76, "log_loss": 0.61}
    ladder = {"h2": {"sets": {"primary": {"penalties": {"penalties": 3, "goals": 2, "conversion": 2 / 3, "vendor": rung, "frame": rung,
                                                        "freeze_frame": rung}}}}}
    primary, combined = shells._penalty_rows(ladder)
    assert primary == ["pierwotna", "3", "2", "0.667", "0.760", "0.6100", "0.760", "0.6100", "0.760", "0.6100"]
    assert combined[:2] == ["łączona", "pominięta"]
