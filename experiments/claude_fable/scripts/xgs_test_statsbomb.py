"""The descriptive StatsBomb agreement of the xg-stack test (ticket 04, council C7 Q4; CPU, seconds). Data: StatsBomb.

    ./run.sh --no-sync python scripts/xgs_test_statsbomb.py --expect-hashlist SHA
        # -> results/xgs_test_statsbomb/{agreement.json, categories.csv, complete.json}

It runs only after the test: it refuses unless research/xg_stack_test_hashes.json hashes to SHA, the test's
results/xgs_test/complete.json exists, names that hash list and matches its files, and the preparation's predictions
are the ones the hash list pins. It refuses an existing output folder, so the look happens once.

The one StatsBomb input is xg-compare's shot table (`results/xgc_compare/table`, read with `compare.table.read`, which
checks it against its sidecar; the folder's completion record is checked too). Its primary rows (the 2,721 paired
non-penalty shots of the 135 evaluation matches) are joined one to one with the test predictions of R, G and the
deployed G-combo, and `stack_heldout.vendor.agreement` gives Spearman's ρ with the vendor's xG and the gap items in
StatsBomb coordinates. No label is read. Every number is labelled "hypothesis formed on these matches" and "Data:
StatsBomb."; nothing feeds a model, the headline or the success rule.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.compare import table as tb
from xgpoc.detector import records
from xgpoc.stack_heldout import vendor

ROOT = Path(__file__).resolve().parents[1]
HASHLIST = ROOT / "research/xg_stack_test_hashes.json"
TEST = ROOT / "results/xgs_test"
PREDICTIONS = ROOT / "results/xgs_test_prep/predictions.csv"
TABLE = ROOT / "results/xgc_compare/table"
OUT = ROOT / "results/xgs_test_statsbomb"
MODELS = ("R", "G", "g_combo")
PRIMARY_SHOTS = 2721


class Refused(SystemExit):
    """The StatsBomb look refuses to run."""


def require_test_done(expected: str, hashlist: Path = HASHLIST, test: Path = TEST, predictions: Path = PREDICTIONS) -> None:
    """Refuse unless the test has been run under the hash list of sha256 `expected` (module docstring)."""
    if not hashlist.is_file() or sha256_file(hashlist) != expected:
        raise Refused(f"{hashlist} does not hash to {expected}")
    record = test / "complete.json"
    if not record.is_file():
        raise Refused(f"{record} does not exist: StatsBomb is read only after the test result is written")
    complete = json.loads(record.read_text())
    if complete["hashlist_sha256"] != expected or "result.json" not in complete["files"]:
        raise Refused("the test's completion record names another hash list or no result")
    for name, digest in complete["files"].items():
        if not (test / name).is_file() or sha256_file(test / name) != digest:
            raise Refused(f"{test / name} does not match the test's completion record")
    pinned = json.loads(hashlist.read_text())["files"]
    if not any(name.endswith("results/xgs_test_prep/predictions.csv") and digest == sha256_file(predictions)
               for name, digest in pinned.items()):
        raise Refused(f"{predictions} is not the file the hash list pins")


def primary_frame(shots: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """xg-compare's primary rows joined one to one with the evaluation-match test predictions (module docstring)."""
    primary = shots[shots.in_primary.astype(bool)][["soccernet_shot_id", "p_vendor", "sb_x", "sb_y", "sb_dist", "sb_angle"]]
    frame = primary.merge(predictions, left_on="soccernet_shot_id", right_on="shot_id", how="left", validate="one_to_one")
    if frame.shot_id.isna().any() or (frame.role != "evaluation").any():
        raise Refused("a primary shot has no evaluation-match test prediction")
    if len(frame) != PRIMARY_SHOTS:
        raise Refused(f"{len(frame)} primary shots, not {PRIMARY_SHOTS}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--expect-hashlist", required=True)
    args = parser.parse_args()
    require_test_done(args.expect_hashlist)
    if OUT.exists():
        raise Refused(f"{OUT} exists: the StatsBomb look happens once")
    records.verified_file(TABLE / tb.TABLE_FILE)
    shots, _ = tb.read(TABLE)
    predictions = pd.read_csv(PREDICTIONS, usecols=["shot_id", "role", *[f"p_{m}" for m in MODELS]])
    result = vendor.agreement(primary_frame(shots, predictions), MODELS)
    OUT.mkdir(parents=True)
    (OUT / "agreement.json").write_text(json.dumps({k: v for k, v in result.items() if k != "categories"}, indent=2) + "\n")
    pd.DataFrame(result["categories"]).assign(label=vendor.LABEL).to_csv(OUT / "categories.csv", index=False, float_format="%.9g")
    record = {"hashlist_sha256": args.expect_hashlist, "test_complete_sha256": sha256_file(TEST / "complete.json"),
              "inputs": {"table": sha256_file(TABLE / tb.TABLE_FILE), "predictions": sha256_file(PREDICTIONS)},
              "code": {name: sha256_file(ROOT / name) for name in ("scripts/xgs_test_statsbomb.py", "xgpoc/stack_heldout/vendor.py")},
              "git": records.git_state(), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
              "files": records.output_hashes(OUT)}
    (OUT / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("shots", "spearman", "label")}, indent=2))


if __name__ == "__main__":
    main()
