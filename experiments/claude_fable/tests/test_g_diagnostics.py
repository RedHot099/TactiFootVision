import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_g  # noqa: E402
from diagnose_g import reconcile, report, safe_rows  # noqa: E402
from diagnostic_review import CHOICES  # noqa: E402
from prepare_external_review import readiness  # noqa: E402
from xgpoc.v13.protocol import digest  # noqa: E402


def review(tmp_path, reviewer, lower, upper):
    row = dict.fromkeys(CHOICES, "unknown") | {"review_id": "r1", "contact_status": "bounded",
                                             "contact_lower_ms": lower, "contact_upper_ms": upper}
    path = tmp_path / f"{reviewer}.json"
    path.write_text(json.dumps({"packet_id": "p", "source": "human", "reviewer": reviewer, "annotations": [row]}))
    return path


def test_single_reviewer_and_disagreement_never_produce_agreed_contact(tmp_path):
    packet = {"packet_id": "p", "items": [{"review_id": "r1", "frames": [{"time_ms": t} for t in (-80, -40, 0, 40)]}]}
    a = review(tmp_path, "a", -80, -40)
    b = review(tmp_path, "b", 0, 40)
    assert reconcile([a], packet).contact_quality.tolist() == ["single"]
    assert reconcile([a, b], packet).contact_quality.tolist() == ["unresolved"]
    assert reconcile([a, b], packet).lower_ms.isna().all()
    with pytest.raises(ValueError, match="Repeated reviewer"):
        reconcile([a, a], packet)


def test_agreeing_reviewers_use_conservative_union_and_strict_measured_cutoff(tmp_path):
    packet = {"packet_id": "p", "items": [{"review_id": "r1", "frames": [{"time_ms": t} for t in (-80, -40, 0, 40)]}]}
    a = review(tmp_path, "a", -80, 0)
    b = review(tmp_path, "b", -40, 40)
    contacts = reconcile([a, b], packet).assign(shot_id="s1")
    assert contacts.lower_ms.iloc[0] == -80
    assert contacts.upper_ms.iloc[0] == 40
    frames = pd.DataFrame({"shot_id": ["s1"] * 4, "nominal_ms": [-120, -80, -160, 0],
                           "measured_ms": [-120, -80, None, 0], "p": [.1] * 4})
    assert safe_rows(frames, contacts).nominal_ms.tolist() == [-120]


def complete_external():
    halves = pd.DataFrame([{"match_id": "m1", "half": h, "duration_ms": 1000, "reviewed_from_ms": 0,
                            "reviewed_until_ms": 1000, "reviewer": "a", "all_attempts_checked": "yes",
                            "clock_alignment_checked": "yes", "overlap_status": "distinct",
                            "overlap_evidence": "independent fixture identities checked", "goal_count_verified": 1}
                           for h in (1, 2)])
    attempts = pd.DataFrame([{"review_id": f"s{h}", "match_id": "m1", "half": h, "is_attempt": "yes",
                              "outcome": "goal", "contact_lower_ms": 100, "contact_upper_ms": 120, "reviewer": "a"}
                             for h in (1, 2)])
    return halves, attempts


def test_external_requires_full_review_goal_accounting_and_identity_evidence():
    halves, attempts = complete_external()
    assert readiness(halves, attempts)[0]["ready_for_protocol_review"]
    for field, value in [("reviewed_until_ms", 900), ("overlap_evidence", ""), ("goal_count_verified", 2),
                         ("reviewer", ""), ("all_attempts_checked", "unknown")]:
        changed = halves.copy()
        changed.loc[0, field] = value
        assert not readiness(changed, attempts)[0]["ready_for_protocol_review"]
    attempts.loc[0, "contact_upper_ms"] = 2000
    assert not readiness(halves, attempts)[0]["ready_for_protocol_review"]
    attempts.loc[0, "match_id"] = "foreign"
    with pytest.raises(ValueError, match="unknown match"):
        readiness(halves, attempts)


def test_report_preserves_unreviewed_shots_and_later_uses_agreed_human_labels(tmp_path, monkeypatch):
    packet = {"packet_id": "p", "items": [{"review_id": "r1", "frames": [{"time_ms": t} for t in (-80, -40, 0, 40)]}]}
    monkeypatch.setattr(diagnose_g, "verify_packet", lambda root: ({"packet_id": "p"}, packet))
    pd.DataFrame({"shot_id": ["s1"], "review_id": ["r1"], "sampling_group": ["random"], "role": ["select"],
                  "goal": [False], "is_penalty": [False]}).to_csv(tmp_path / "sample.csv", index=False)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pd.DataFrame({"shot_id": ["s1", "s1"], "nominal_ms": [-120, 0], "measured_ms": [-120, 0],
                  "p": [.1, .2]}).to_csv(predictions / "predictions.csv", index=False)
    (predictions / "manifest.json").write_text(json.dumps({"identity": {"packet_id": "p"},
        "files": {"predictions.csv": digest(predictions / "predictions.csv")}}))
    before = tmp_path / "before"
    report(tmp_path, predictions, [], before)
    summary = json.loads((before / "summary.json").read_text())
    assert summary["remaining_without_verified_safe_frame"] == 1
    assert summary["training_gate"] == "blocked_pending_human_review"
    a, b = review(tmp_path, "a", -80, 0), review(tmp_path, "b", -40, 40)
    after = tmp_path / "after"
    report(tmp_path, predictions, [a, b], after)
    assert json.loads((after / "summary.json").read_text())["shots_with_verified_safe_frame"] == 1
    paired = pd.read_csv(after / "paired_safe_comparison.csv")
    assert paired.verified_goal.isna().all()  # uncertain outcomes are never silently treated as negatives
    assert paired.loss_safe_original_label.iloc[0] < paired.loss_t0_original_label.iloc[0]
