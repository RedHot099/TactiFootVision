"""The confirmatory label-join entry (xg-vendor ticket 14 fix round, finding 6): every recorded hash is verified before
any label is joined, and any mismatch refuses the join."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from test_vendor_evaltable import make_table
from test_vendor_prereg import document, dryrun_record, make_tree, parameters
from xgpoc.vendor import confirm
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import prereg as pr

ENTRY = ("scripts/entry.py",)
COMMITTED = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_file(root: Path, name: str, text: str) -> dict:
    (root / name).write_text(text)
    return {"path": name, "sha256": sha(root / name)}


def frozen_tree(root: Path) -> dict:
    """A complete pre-registration in `root`: code, every pinned input, the label-free combined table, the dry-run
    evidence, the document with its hash block, the hash list and the confirmatory record."""
    make_tree(root)
    table = make_table("confirmatory")
    sets = {c: {"file": f"{c}.npy", "sha256": "0" * 64, "dim": 4, "rows": 12, "model": "fixture"} for c in et.EMBEDDING_COLUMNS}
    meta = et.write_table(table, root / "table", "t", kind="confirmatory", embedding_sets=sets)
    pairs = pd.DataFrame({"shot_id": ["x"], "paired": [True]})
    pairs.to_parquet(root / "pairs.parquet")
    pd.DataFrame({"event_id": ["e"]}).to_parquet(root / "sb_unpaired.parquet")
    pairing = {"files": {"pairs.parquet": sha(root / "pairs.parquet"), "sb_unpaired.parquet": sha(root / "sb_unpaired.parquet")}}
    params = parameters(margin=0.005)
    params["pins"] |= {"primary_label_free_table": {"path": "table/t.parquet", "sha256": meta["table_sha256"]},
                       "combined_label_free_table": {"path": "table/t.parquet", "sha256": meta["table_sha256"]},
                       "split_plan": pinned_file(root, "plan.csv", "match,role\n"),
                       "comparator_predictions": pinned_file(root, "comparator.csv", "shot,p\n"),
                       "timing_verdict": pinned_file(root, "verdict.json", "{}\n"),
                       "timing_rule": pinned_file(root, "rule.md", "rule\n"),
                       "delta_rule": pinned_file(root, "delta.md", "delta\n"),
                       "pairs": {"path": "pairs.parquet", "sha256": sha(root / "pairs.parquet")},
                       "pairing_record": pinned_file(root, "pairing.json", json.dumps(pairing))}
    doc, hashes_path = root / "doc.md", root / "hashes.json"
    doc.write_text(document(params))
    code = pr.hash_list(root, doc, ENTRY)["code"]
    for name, path in pr.EVIDENCE.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps(dryrun_record(code, params, params["pins"][pr.EVIDENCE_TABLE_PINS[name]]["sha256"])))
    hashes = pr.hash_list(root, doc, ENTRY)
    payload = pr.hash_list_bytes(hashes)
    doc.write_text(pr.replace_hash_block(doc.read_text(), pr.render_hash_block(hashes, "hashes.json", hashlib.sha256(payload).hexdigest())))
    hashes = pr.hash_list(root, doc, ENTRY)
    hashes_path.write_bytes(pr.hash_list_bytes(hashes))
    record = root / "record.json"
    record.write_text(json.dumps(et.preregistration_record(pr.confirmatory_record(doc, hashes, COMMITTED, "c0ffee"))))
    return {"root": root, "document": doc, "hashes_path": hashes_path, "record_path": record, "table_dir": root / "table",
            "table_name": "t", "entries": ENTRY}


def test_an_intact_preregistration_is_verified(tmp_path):
    tree = frozen_tree(tmp_path)
    verified = confirm.verify(**tree)
    assert verified.prereg.margin == 0.005 and not verified.prereg.draft and verified.hashes["status"] == "complete"
    pairs, unpaired = confirm.label_sources(tmp_path, verified.hashes, tmp_path / "sb_unpaired.parquet")
    assert len(pairs) == len(unpaired) == 1


@pytest.mark.parametrize(("tamper", "message"), [
    (lambda t: (t["root"] / "xgpoc/other.py").write_text("def helper():\n    return 2\n"), "hash list differs"),
    (lambda t: (t["root"] / "plan.csv").write_text("match,role\nm,gate\n"), "pin split_plan"),
    (lambda t: (t["root"] / "comparator.csv").write_text("shot,p\ns,0.1\n"), "pin comparator_predictions"),
    (lambda t: (t["root"] / "verdict.json").write_text('{"verdict": "shared"}\n'), "pin timing_verdict"),
    (lambda t: (t["root"] / "results/xgv_dryrun_primary/complete.json").write_text("[]\n"), "hash list differs"),
    (lambda t: t["record_path"].write_text(json.dumps(json.loads(t["record_path"].read_text()) | {"recalibrated_bootstrap": "fixed"})),
     "recalibrated_bootstrap"),
    (lambda t: t["record_path"].write_text(json.dumps(json.loads(t["record_path"].read_text()) | {"frame_exclusions": {"k": "decode_corrupted_frame"}})),
     "frame_exclusions"),
    (lambda t: t["document"].write_text(t["document"].read_text() + "\nedited after the commit\n"), "document"),
    (lambda t: t["record_path"].write_text(json.dumps(json.loads(t["record_path"].read_text()) | {"margin": 0.002})), "margin"),
    (lambda t: t["record_path"].write_text(json.dumps(json.loads(t["record_path"].read_text()) | {"draft": True})), "draft"),
])
def test_any_mismatch_refuses_before_a_label_is_joined(tmp_path, tamper, message):
    tree = frozen_tree(tmp_path)
    tamper(tree)
    with pytest.raises(confirm.ConfirmationRefused, match=message):
        confirm.verify(**tree)


def test_a_table_other_than_the_pinned_one_is_refused(tmp_path):
    tree = frozen_tree(tmp_path)
    other = make_table("confirmatory").assign(pred_event=0.2)
    sets = json.loads((tmp_path / "table/t.json").read_text())["embedding_sets"]
    et.write_table(other, tmp_path / "other", "t", kind="confirmatory", embedding_sets=sets)
    with pytest.raises(confirm.ConfirmationRefused, match="not the label-free table the pre-registration pins"):
        confirm.verify(**(tree | {"table_dir": tmp_path / "other"}))


def test_label_sources_must_be_the_pinned_pairing_files(tmp_path):
    tree = frozen_tree(tmp_path)
    verified = confirm.verify(**tree)
    pd.DataFrame({"event_id": ["e", "f"]}).to_parquet(tmp_path / "sb_unpaired.parquet")
    with pytest.raises(confirm.ConfirmationRefused, match="sb_unpaired.parquet"):
        confirm.label_sources(tmp_path, verified.hashes, tmp_path / "sb_unpaired.parquet")


def test_uncommitted_or_changed_files_refuse(tmp_path):
    tree = frozen_tree(tmp_path)
    verified = confirm.verify(**tree)
    text = tree["document"].read_text()

    def clean_git(*args):
        return text if args[0] == "show" else ""

    confirm.committed_unchanged(tmp_path, verified, tree["document"], tree["hashes_path"], clean_git)
    with pytest.raises(confirm.ConfirmationRefused, match="not committed"):
        confirm.committed_unchanged(tmp_path, verified, tree["document"], tree["hashes_path"],
                                    lambda *a: " M scripts/entry.py" if a[0] == "status" else text)
    with pytest.raises(confirm.ConfirmationRefused, match="commit c0ffee"):
        confirm.committed_unchanged(tmp_path, verified, tree["document"], tree["hashes_path"],
                                    lambda *a: "other text" if a[0] == "show" else "")


def test_the_join_happens_only_after_verification(tmp_path):
    tree = frozen_tree(tmp_path)
    (tmp_path / "plan.csv").write_text("changed\n")

    def labels_must_not_be_built(*_):
        raise AssertionError("labels were built before the verification refused")

    with pytest.raises(confirm.ConfirmationRefused):
        confirm.verified_join(tree, labels_must_not_be_built, tmp_path / "out", now=COMMITTED)
    assert not (tmp_path / "out").exists()


def test_a_verified_join_writes_the_confirmatory_table_with_its_log(tmp_path):
    tree = frozen_tree(tmp_path)
    table, _ = et.read_table(tree["table_dir"], "t")
    labels = pd.DataFrame({"key": table.key, "goal_sn": False,
                           "on_target_sn": pd.array([True if p == "primary" else None for p in table.population], dtype="boolean"),
                           "sb_outcome": "Saved", "is_penalty": False, "label_exclusion": None})
    verified, meta = confirm.verified_join(tree, lambda *_: labels, tmp_path / "out", now=COMMITTED)
    joined, joined_meta = et.load_table(tmp_path / "out/table", confirm.JOINED, verified.prereg)
    assert joined_meta["label_join"]["label_free_sha256"] == verified.prereg.label_free_table_sha256 == meta["label_join"]["label_free_sha256"]
    assert (tmp_path / "out" / confirm.JOIN_LOG).read_text().count("\n") == 1
    with pytest.raises(et.EvalTableError, match="before the pre-registration record"):
        confirm.verified_join(tree, lambda *_: labels, tmp_path / "again", now=COMMITTED.replace(year=2025))


def test_the_analysis_commands_take_every_parameter_from_the_record(tmp_path):
    commands = confirm.analysis_commands(tmp_path / "t", "joined", tmp_path / "record.json", tmp_path / "shots.parquet", tmp_path / "v.json", tmp_path)
    assert set(commands) == set(confirm.ACCEPTED_EXITS)
    for step, args in commands.items():
        assert "--prereg" in args and not {"--n-boot", "--seed", "--margin", "--thresholds", "--offset-logits", "--expected-rule-sha256"} & set(args), step
    assert commands["vendor_leakage.py"][commands["vendor_leakage.py"].index("--timing-verdict") + 1] == str(tmp_path / "v.json")
