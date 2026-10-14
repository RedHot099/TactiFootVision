"""The descriptive StatsBomb agreement of the xg-stack test (ticket 04, council C7 Q4; CPU, seconds). Data: StatsBomb.

    ./run.sh --no-sync python scripts/xgs_test_statsbomb.py --expect-hashlist SHA
        # -> results/xgs_test_statsbomb/{agreement.json, categories.csv, complete.json}

It runs only after the test: it refuses unless research/xg_stack_test_hashes.json hashes to SHA and is committed
unmodified, the test's results/xgs_test/complete.json exists, names that hash list and matches its files, and the
preparation's predictions and this look's own code (`CODE`) are the files the hash list pins. It refuses an existing output folder, so the look happens once.

The one StatsBomb input is xg-compare's shot table (`results/xgc_compare/table`, read with `compare.table.read`, which
checks it against its sidecar; the folder's completion record is checked too). Its primary rows (the 2,721 paired
non-penalty shots of the 135 evaluation matches) are joined one to one with the test predictions of R, G and the
deployed G-combo, and `stack_heldout.vendor.agreement` gives Spearman's ρ with the vendor's xG and the gap items in
StatsBomb coordinates. No label is read. Every number is labelled "hypothesis formed on these matches" and "Data:
StatsBomb."; nothing feeds a model, the headline or the success rule.
"""

import argparse
import json
import subprocess
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
CODE = (Path(__file__).resolve(), ROOT / "xgpoc/stack_heldout/vendor.py")    # pinned by the hash list (xgs_test.STATSBOMB_CODE)
MODELS = ("R", "G", "g_combo")
PRIMARY_SHOTS = 2721


class Refused(SystemExit):
    """The StatsBomb look refuses to run."""


def committed(path: Path) -> bool:
    """Tracked and unchanged against HEAD."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=path.parent, capture_output=True)
    unchanged = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=path.parent)
    return tracked.returncode == 0 and unchanged.returncode == 0


def pinned(pins: dict[str, str], path: Path) -> bool:
    """`path` is pinned by its checkout-relative name (or its absolute path) with its current sha256."""
    path = Path(path).absolute()                  # not resolved: pinned names keep symlinked folders as written
    names = [n for n in pins if n == str(path) or str(path).endswith("/" + n)]
    return len(names) == 1 and pins[names[0]] == sha256_file(path)


def require_test_done(expected: str, hashlist: Path = HASHLIST, test: Path = TEST, predictions: Path = PREDICTIONS,
                      code: tuple[Path, ...] = CODE, check_git: bool = True) -> None:
    """Refuse unless the test has been run under the hash list of sha256 `expected` (module docstring)."""
    if not hashlist.is_file() or sha256_file(hashlist) != expected:
        raise Refused(f"{hashlist} does not hash to {expected}")
    if check_git and not committed(hashlist):
        raise Refused(f"{hashlist} is not the committed, unmodified file")
    record = test / "complete.json"
    if not record.is_file():
        raise Refused(f"{record} does not exist: StatsBomb is read only after the test result is written")
    complete = json.loads(record.read_text())
    if complete["hashlist_sha256"] != expected or "result.json" not in complete["files"]:
        raise Refused("the test's completion record names another hash list or no result")
    for name, digest in complete["files"].items():
        if not (test / name).is_file() or sha256_file(test / name) != digest:
            raise Refused(f"{test / name} does not match the test's completion record")
    pins = json.loads(hashlist.read_text())["files"]
    for path in (predictions, *code):
        if not pinned(pins, path):
            raise Refused(f"{path} is not the file the hash list pins")


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
              "code": {str(p.relative_to(ROOT)): sha256_file(p) for p in CODE},
              "git": records.git_state(), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
              "files": records.output_hashes(OUT)}
    (OUT / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("shots", "spearman", "label")}, indent=2))


if __name__ == "__main__":
    main()
