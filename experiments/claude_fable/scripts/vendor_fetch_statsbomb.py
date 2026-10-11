"""Build the StatsBomb open-data version `statsbomb_open_v1` (xg-vendor ticket 01).

Fetches competitions, match lists and the full event JSON of every match in the covered competition
seasons (`xgpoc.vendor.statsbomb_fetch.COVERED`) from github.com/statsbomb/open-data at ONE pinned commit,
stores them compressed, derives the parquet tables (`xgpoc.vendor.statsbomb_tables`) and publishes them
with a completion record: the commit, source hashes of this script and the two library modules, the terms
files' hashes and a sha256 of every output file.

The fetch is resumable (a match whose stored file verifies is skipped), retries network errors with
backoff, uses at most 4 threads, and fails with the list of missing matches when any expected match could
not be stored: then no table is published. A finished store is verified instead of rebuilt; changed
sources, a changed file or a file the record does not list is an error. Data from StatsBomb: attribution
and the user agreement are in research/xg_vendor_statsbomb_terms.md.

    ./run.sh python scripts/vendor_fetch_statsbomb.py [--out DIR] [--sha SHA] [--prototype DIR]
"""

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow
from xgpoc.paths import STATSBOMB
from xgpoc.vendor import statsbomb_fetch as fetching
from xgpoc.vendor.statsbomb_fetch import file_sha256
from xgpoc.vendor.statsbomb_tables import build_tables

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
VERSION = "statsbomb_open_v1"
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/statsbomb_fetch.py", ROOT / "xgpoc/vendor/statsbomb_tables.py")
TABLES = ("matches", "shots", "context", "timing")
TABLE_DIR = "tables"
HASH_WORKERS = 4
PROTOTYPE_DIR = Path("/home/kuba/projects/football_data/statsbomb_open")
CROSSCHECK_OUT = ROOT / "results/xgv_statsbomb_data/prototype_crosscheck.json"


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def parallel_digests(paths: list[Path]) -> list[str]:
    with ThreadPoolExecutor(HASH_WORKERS) as pool:
        return list(pool.map(file_sha256, paths))


def build_identity(root: Path, sha: str, covered) -> dict:
    return {"version": VERSION, "open_data": {"repo": fetching.REPO, "sha": sha}, "covered": [list(pair) for pair in covered],
            "terms": {n: file_sha256(fetching.terms_path(root, n)) for n in fetching.TERMS_FILES},
            "sources": {str(p): h for p, h in zip(SOURCE_FILES, parallel_digests(list(SOURCE_FILES)), strict=True)}}


def output_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != "complete.json" and not p.name.endswith(".tmp"))


def write_tables(root: Path, sha: str, covered) -> dict[str, pd.DataFrame]:
    """Derive the tables from the verified raw files and publish them with one directory rename."""
    records = fetching.load_match_lists(root, sha, covered)

    def events():
        for match in records:
            mid = match["match_id"]
            yield mid, json.loads(fetching.load_raw(fetching.events_path(root, mid), sha, f"data/events/{mid}.json"))

    tables = build_tables(records, events())
    stage = root / f".{TABLE_DIR}.partial"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    for name in TABLES:
        tables[name].to_parquet(stage / f"{name}.parquet", index=False, compression="zstd")
    shutil.rmtree(root / TABLE_DIR, ignore_errors=True)   # tables of an earlier run that never got a completion record
    stage.rename(root / TABLE_DIR)
    return tables


def summarize(tables: dict[str, pd.DataFrame], expected: list[int]) -> dict:
    shots, matches = tables["shots"], tables["matches"]
    if sorted(matches.match_id) != expected:
        raise ValueError("Match table and expected match ids differ")
    return {"matches": len(matches), "matches_per_competition_season": {f"{k[0]} {k[1]}": int(v) for k, v in
                                                                         matches.groupby(["competition", "season"]).size().items()},
            "shots": len(shots), "matches_with_shots": int(shots.match_id.nunique()), "goals": int(shots.outcome.eq("Goal").sum()),
            "shots_with_freeze_frame": int(shots.freeze_frame_json.notna().sum()),
            "context_events": {k: int(v) for k, v in tables["context"].type.value_counts().sort_index().items()},
            "timing_events": {str(k): int(v) for k, v in tables["timing"].soccernet_label.fillna("(none)").value_counts().sort_index().items()}}


def complete(root: Path, identity: dict, summary: dict) -> None:
    files = output_files(root)
    hashes = dict(zip((str(p.relative_to(root)) for p in files), parallel_digests(files), strict=True))
    if build_identity(root, identity["open_data"]["sha"], [tuple(c) for c in identity["covered"]]) != identity:
        raise ValueError("A source changed during the build")
    runtime = {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    temporary = root / ".complete.json.tmp"
    temporary.write_text(json_text({"identity": identity, "summary": summary, "runtime": runtime, "files": hashes}))
    temporary.rename(root / "complete.json")


def verify_complete(root: Path, covered) -> dict:
    """Identity (so sources and commit) must equal the freshly hashed one, every listed file must match
    its hash, and every sidecar must carry the pinned commit. Files the record does not list are an error."""
    record = json.loads((root / "complete.json").read_text())
    if record["identity"] != build_identity(root, fetching.read_pin(root), covered):
        raise ValueError("Sources, commit or terms changed since the completion record; use a new output folder, or delete complete.json and tables/ to re-derive from the raw files")
    present = {str(p.relative_to(root)) for p in output_files(root)}
    if unlisted := sorted(present - set(record["files"])):
        raise ValueError(f"{len(unlisted)} files are not in the completion record, e.g. {unlisted[0]}")
    names = list(record["files"])
    for name, found in zip(names, parallel_digests([root / n for n in names]), strict=True):
        if found != record["files"][name]:
            raise ValueError(f"Output changed or missing: {name}")
    sha = record["identity"]["open_data"]["sha"]
    for sidecar in (root / "raw").rglob("*.meta.json"):
        if json.loads(sidecar.read_text())["sha"] != sha:
            raise ValueError(f"{sidecar} was fetched at another commit than {sha}")
    return record


def build(out: Path, fetch: fetching.Fetch = fetching.http_fetch, sha: str | None = None, covered=fetching.COVERED,
          workers: int = fetching.MAX_WORKERS, attempts: int = fetching.ATTEMPTS, backoff_s: float = fetching.BACKOFF_S,
          sleep: Callable[[float], None] | None = None, progress: Callable[[str], None] | None = print) -> dict:
    """Build or verify the data version in `out`; returns its completion record."""
    out = Path(out)
    say = progress or (lambda _: None)
    pinned = fetching.read_pin(out)
    if sha is not None and pinned is not None and sha != pinned:
        raise ValueError(f"{out} is pinned to {pinned}; refusing to mix commits (asked for {sha})")
    if (out / "complete.json").exists():
        record = verify_complete(out, covered)
        say("Verified existing StatsBomb data version; nothing rebuilt")
        return record
    sha = sha or pinned or fetching.resolve_sha(fetch)
    say(f"Open-data commit {sha}")
    kwargs = {"sleep": sleep} if sleep is not None else {}
    expected = fetching.fetch_all(fetch, out, sha, covered, workers, attempts, backoff_s, progress=progress, **kwargs)
    tables = write_tables(out, sha, covered)
    summary = summarize(tables, expected)
    complete(out, build_identity(out, sha, covered), summary)
    return json.loads((out / "complete.json").read_text())


def prototype_crosscheck(out: Path, prototype: Path) -> dict:
    """Compare the new tables with the prototype's ad-hoc build: counts, and the shared columns of matches,
    shots (by event id, event JSON included) and context events (by event id)."""
    result = {}
    for name, key, new_name in (("matches", "match_id", "matches"), ("shots", "event_id", "shots"), ("context", "event_id", "context")):
        old = pd.read_parquet(prototype / f"{name}.parquet").sort_values(key).reset_index(drop=True)
        new = pd.read_parquet(out / TABLE_DIR / f"{new_name}.parquet").sort_values(key).reset_index(drop=True)
        shared = [c for c in old.columns if c in new.columns]
        same_ids = old[key].tolist() == new[key].tolist()
        differing = {} if not same_ids else {c: int((~(old[c].eq(new[c]) | (old[c].isna() & new[c].isna()))).sum()) for c in shared}
        result[name] = {"prototype_rows": len(old), "new_rows": len(new), "same_keys": same_ids,
                        "only_in_prototype": len(set(old[key]) - set(new[key])), "only_in_new": len(set(new[key]) - set(old[key])),
                        "shared_columns": shared, "columns_with_differences": {c: n for c, n in differing.items() if n}}
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=STATSBOMB)
    parser.add_argument("--sha", help="open-data commit to pin; default: the store's pin, else master now")
    parser.add_argument("--workers", type=int, default=fetching.MAX_WORKERS, help=f"download threads, at most {fetching.MAX_WORKERS}")
    parser.add_argument("--prototype", type=Path, nargs="?", const=PROTOTYPE_DIR,
                        help=f"compare with the prototype's build afterwards (default folder {PROTOTYPE_DIR})")
    args = parser.parse_args(argv)
    try:
        record = build(args.out, sha=args.sha, workers=args.workers)
    except fetching.MissingMatchesError as exc:
        sys.exit(f"FAILED, no tables published. {exc}")
    print(json_text(record["summary"]))
    if args.prototype is not None:
        result = prototype_crosscheck(args.out, args.prototype)
        CROSSCHECK_OUT.parent.mkdir(parents=True, exist_ok=True)
        CROSSCHECK_OUT.write_text(json_text(result))
        print(json_text(result))


if __name__ == "__main__":
    main()
