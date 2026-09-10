import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_v13 import mean_logits, report, runs  # noqa: E402


def test_run_from_another_protocol_is_rejected(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({"version": 1}))
    (tmp_path / "suite_identity.json").write_text("{}")
    run = tmp_path / "full/seed0"
    run.mkdir(parents=True)
    (run / "complete.json").write_text(json.dumps({"identity": {
        "protocol": {"version": 2}, "spec": {"variant": "full"}}}))
    with pytest.raises(ValueError, match="another protocol"):
        runs(tmp_path, ["full"])


def test_ensemble_logits_follow_requested_shot_order(tmp_path):
    pd.DataFrame({"shot_id": ["b", "a"], "logit": [2., 1.]}).to_csv(
        tmp_path / "evaluate_predictions.csv", index=False)
    assert mean_logits([tmp_path / "complete.json"], "evaluate", ["a", "b"]).tolist() == [1., 2.]


def test_evaluation_labels_cannot_enter_blind_prediction_file(tmp_path):
    pd.DataFrame({"shot_id": ["a"], "logit": [1.], "goal": [1]}).to_csv(
        tmp_path / "evaluate_predictions.csv", index=False)
    with pytest.raises(ValueError, match="blind"):
        mean_logits([tmp_path / "complete.json"], "evaluate", ["a"])


def test_changed_analysis_rejected_before_labels_are_accessed(tmp_path):
    (tmp_path / "frozen.json").write_text(json.dumps({"analysis": {"n_boot": 99}}))
    with pytest.raises(ValueError, match="specification frozen"):
        report(tmp_path, None, 2000)
