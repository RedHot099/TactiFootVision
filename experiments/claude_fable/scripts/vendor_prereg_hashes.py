"""Hash list and pre-registration record of the xg-vendor pre-registration (ticket 14; `xgpoc.vendor.prereg`).

    ./run.sh python scripts/vendor_prereg_hashes.py [--doc research/xg_vendor_preregistration.md]
        [--hashes research/xg_vendor_prereg_hashes.json] [--check]
Computes the hash list (pins verified against their files, code closure of the confirmatory run, dry-run evidence,
open placeholders), writes it and regenerates the document's hash block, which prints the list's sha256. `--check`
writes nothing and fails when either file would change. A draft list is allowed; its status says so.

    ./run.sh python scripts/vendor_prereg_hashes.py --record results/xgv_prereg/preregistration_record.json
After the document and the hash list are committed: refuses unless the list is complete, the document, the list and
every code file are committed and unchanged, and a recomputation gives the committed bytes; then writes the
pre-registration record (`evaltable.PreRegistration`, not a draft) with the document's sha256, the time of the last
commit that touched the document, the pinned combined label-free table and the parameters. Ticket 15 passes this
record to the label join and to every analysis.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from xgpoc.vendor import evaltable as et
from xgpoc.vendor import prereg as pr

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
DOC = ROOT / "research/xg_vendor_preregistration.md"
HASHES = ROOT / "research/xg_vendor_prereg_hashes.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build(doc: Path, hashes_path: Path) -> tuple[dict, bytes, str]:
    """(hash list, its bytes, the document text with the regenerated hash block)."""
    hashes = pr.hash_list(ROOT, doc)
    payload = pr.hash_list_bytes(hashes)
    block = pr.render_hash_block(hashes, str(hashes_path.relative_to(ROOT)), hashlib.sha256(payload).hexdigest())
    return hashes, payload, pr.replace_hash_block(doc.read_text(), block)


def record(doc: Path, hashes_path: Path, out: Path) -> None:
    hashes, payload, text = build(doc, hashes_path)
    if hashes["status"] != "complete":
        sys.exit(f"the pre-registration is a draft: {hashes['placeholders']}")
    if payload != hashes_path.read_bytes() or text != doc.read_text():
        sys.exit("the hash list or the document's hash block is stale; rerun without --record and commit")
    files = [doc, hashes_path, *(ROOT / p for p in hashes["code"])]
    dirty = git("status", "--porcelain", "--", *map(str, files))
    if dirty:
        sys.exit(f"not committed or changed since the commit:\n{dirty}")
    last = git("log", "-1", "--format=%H%n%cI", "--", str(doc)).splitlines()
    if len(last) != 2:
        sys.exit(f"{doc} has no commit")
    commit, committed_at = last
    prereg = pr.confirmatory_record(doc, hashes, et.parse_time(committed_at), commit)
    if out.exists():
        sys.exit(f"{out} exists; a pre-registration record is written once")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(et.preregistration_record(prereg), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}: document {prereg.sha256[:12]}, commit {commit[:12]} at {committed_at}, margin {prereg.margin}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--hashes", type=Path, default=HASHES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="write nothing; fail when the hash list or the hash block would change")
    mode.add_argument("--record", type=Path, help="write the pre-registration record of the committed document here")
    args = parser.parse_args()
    doc, hashes_path = args.doc.resolve(), args.hashes.resolve()
    if args.record:
        record(doc, hashes_path, args.record)
        return
    hashes, payload, text = build(doc, hashes_path)
    stale = [p.name for p, new in ((hashes_path, payload), (doc, text.encode())) if not p.exists() or p.read_bytes() != new]
    if args.check:
        sys.exit(f"stale: {stale}" if stale else 0)
    hashes_path.write_bytes(payload)
    doc.write_text(text)
    print(f"hash list {hashes['status']}: {len(hashes['pins'])} pins, {len(hashes['code'])} code files, "
          f"sha256 {hashlib.sha256(payload).hexdigest()}; open: {hashes['placeholders'] or 'none'}")


if __name__ == "__main__":
    main()
