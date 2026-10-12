"""The confirmatory label-join entry of the vendor comparison (xg-vendor tickets 14, 15; `scripts/vendor_confirm.py`).

Nothing joins a label to the new predictions before `verify` has passed. It checks, in this order, and raises
`ConfirmationRefused` on the first mismatch:
1. the pre-registration record (written by `scripts/vendor_prereg_hashes.py --record`) is not a draft and names its commit;
2. a fresh hash list of the document (`prereg.hash_list`) re-hashes every pinned file against its pin: the combined and
   primary label-free tables, the split plan, the comparator predictions, the timing verdict and rule, the delta rule,
   the pairing and the rest of the parameter block's pins;
3. that list equals the committed hash list byte for byte, so every source file of the analysis code (the code closure
   of `prereg.ENTRY_SCRIPTS`) and every dry-run completion record still has its recorded sha256, and its status is
   complete (every dry run ran the current code, parameters and pinned table); the document's hash block is the one the
   list generates;
4. the record equals the record the document and the list imply (document sha256, pinned label-free table, margin,
   draws, seeds, offset logits, H3 thresholds and targets, timing rule hash, bootstrap of the recalibrated rungs, frame
   exclusions), the commit and its time taken as recorded;
5. the label-free table to be joined is the pinned one (its parquet matches its sidecar, which carries the pinned hash).
`committed_unchanged` adds the git side (the document at the recorded commit is the working file; the document, the
hash list and every code file are committed and clean); `label_sources` admits the label inputs only when they are the
files of the pinned pairing record. `verified_join` runs `verify`, copies the label-free table into the run folder,
builds the labels and joins them with the logged step (`evaltable.join_labels`, which refuses a join before the record).
`analysis_commands` are the three analysis runs, shared with the dry run, so both take every parameter from the record.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import pandas as pd

from xgpoc.bundle import sha256_file
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import prereg as pr
from xgpoc.vendor.probes import CONTROL_FAILED_EXIT

LABEL_FREE, JOINED, JOIN_LOG = "confirmatory_labels_absent", "confirmatory", "label_join_log.jsonl"
ACCEPTED_EXITS = {"vendor_ladder.py": (0,), "vendor_probes.py": (0, CONTROL_FAILED_EXIT), "vendor_leakage.py": (0,)}


class ConfirmationRefused(RuntimeError):
    """A recorded hash, the record or the table does not match; no label may be joined."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmationRefused(message)


@dataclass(frozen=True)
class Verified:
    """What `verify` established: the record, the recomputed hash list and the sidecar of the pinned label-free table."""

    prereg: et.PreRegistration
    hashes: dict
    table_meta: dict


def verify(root: Path, document: Path, hashes_path: Path, record_path: Path, table_dir: Path, table_name: str,
           entries: tuple[str, ...] = pr.ENTRY_SCRIPTS) -> Verified:
    """Every check of the module docstring but the git side; raises `ConfirmationRefused`."""
    root, document, hashes_path = Path(root), Path(document), Path(hashes_path)
    try:
        prereg = et.load_preregistration(record_path)
        _require(not prereg.draft and prereg.commit, "the pre-registration record is a draft or names no commit")
        hashes = pr.hash_list(root, document, entries)
        payload = pr.hash_list_bytes(hashes)
        _require(payload == hashes_path.read_bytes(), f"the committed hash list differs from a recomputation ({hashes_path})")
        _require(hashes["status"] == "complete", f"the hash list is a draft: {hashes['placeholders']}")
        block = pr.render_hash_block(hashes, str(hashes_path.resolve().relative_to(root.resolve())), hashlib.sha256(payload).hexdigest())
        _require(pr.replace_hash_block(document.read_text(), block) == document.read_text(), "the document's hash block is stale")
        expected = pr.confirmatory_record(document, hashes, prereg.recorded_at, prereg.commit)
        differ = [f.name for f in fields(expected) if getattr(expected, f.name) != getattr(prereg, f.name)]
        _require(not differ, f"the pre-registration record disagrees with the committed document and hash list: {differ}")
        _, meta = et.read_table(table_dir, table_name)
        _require(meta["table_kind"] == "confirmatory" and meta["labels_state"] == "absent", "the table to join is not a confirmatory label-free table")
        _require(meta["table_sha256"] == prereg.label_free_table_sha256, "this is not the label-free table the pre-registration pins")
    except (pr.PreRegistrationError, et.EvalTableError, OSError, KeyError) as refusal:
        raise ConfirmationRefused(str(refusal)) from refusal
    return Verified(prereg, hashes, meta)


def committed_unchanged(root: Path, verified: Verified, document: Path, hashes_path: Path, git: Callable[..., str]) -> None:
    """The git side: `git("show", "<commit>:<document>")` must be the working document and `git("status", "--porcelain",
    "--", files...)` empty for the document, the hash list and every code file of the list."""
    root = Path(root)
    files = [Path(document), Path(hashes_path), *(root / p for p in verified.hashes["code"])]
    dirty = git("status", "--porcelain", "--", *map(str, files))
    _require(not dirty, f"not committed or changed since the commit:\n{dirty}")
    relative = Path(document).resolve().relative_to(root.resolve())
    committed = git("show", f"{verified.prereg.commit}:./{relative}")
    _require(committed == Path(document).read_text(), f"the document differs from the one at commit {verified.prereg.commit}")


def label_sources(root: Path, hashes: dict, unpaired_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pair table, the pairing's population table of the unpaired StatsBomb shots) after checking both against the
    pinned pairing record (the hash list has verified the record and the pair table against their pins)."""
    pins = hashes["pins"]
    record = json.loads((Path(root) / pins["pairing_record"]["path"]).read_text())
    _require(record["files"]["pairs.parquet"] == pins["pairs"]["sha256"], "the pinned pair table is not the pinned pairing's")
    found = sha256_file(Path(unpaired_path))
    _require(found == record["files"]["sb_unpaired.parquet"], f"{unpaired_path} is not the sb_unpaired.parquet of the pinned pairing")
    return pd.read_parquet(Path(root) / pins["pairs"]["path"]), pd.read_parquet(unpaired_path)


def verified_join(inputs: dict, build_labels: Callable[[pd.DataFrame, Verified], pd.DataFrame], out: Path, now: datetime) -> tuple[Verified, dict]:
    """`verify(**inputs)`; then the label-free table is written again into `<out>/table` (the same bytes, checked; links
    to its embedding files), `build_labels(table, verified)` gives the labels and `evaltable.join_labels` joins them with
    the record at `now` into `JOINED`, logging to `<out>/JOIN_LOG`. Returns (verified, the joined table's sidecar)."""
    verified = verify(**inputs)
    table, meta = et.read_table(inputs["table_dir"], inputs["table_name"])
    table_dir = Path(out) / "table"
    table_dir.mkdir(parents=True)
    for entry in meta["embedding_sets"].values():
        if not Path(entry["file"]).is_absolute() and (Path(inputs["table_dir"]) / entry["file"]).exists():
            (table_dir / entry["file"]).symlink_to(Path(inputs["table_dir"]).resolve() / entry["file"])
    copy = et.write_table(table, table_dir, LABEL_FREE, kind="confirmatory", embedding_sets=meta["embedding_sets"],
                          unavailable_columns=tuple(meta["unavailable_columns"]), notes=tuple(meta["notes"]), sources=meta["sources"])
    _require(copy["table_sha256"] == verified.prereg.label_free_table_sha256, "the copied label-free table changed its bytes")
    labels = build_labels(table, verified)
    joined = et.join_labels(table_dir, LABEL_FREE, labels, JOINED, Path(out) / JOIN_LOG, prereg=verified.prereg, now=now)
    return verified, joined


def analysis_commands(table_dir: Path, table_name: str, prereg_path: Path, shots_path: Path, timing_verdict: Path,
                      out: Path) -> dict[str, list[str]]:
    """Arguments of the three analysis scripts of a dry-run or confirmatory run: every parameter comes from the record."""
    table, common = [str(table_dir), table_name], ["--prereg", str(prereg_path)]
    return {"vendor_ladder.py": ["--table-dir", table[0], "--table-name", table[1], *common, "--out", str(Path(out) / "ladder")],
            "vendor_probes.py": ["--table", table[0], "--name", table[1], *common, "--shots", str(shots_path), "--out", str(Path(out) / "probes")],
            "vendor_leakage.py": ["--table", table[0], "--name", table[1], *common, "--timing-verdict", str(timing_verdict),
                                  "--out", str(Path(out) / "leakage")]}
