"""Build the release package of the xg-vendor study (ticket 16) into `results/xgv_release_v1/`.

The package ships no StatsBomb data (owner, 2026-10-11): ids, our pairing fields, our G-combo predictions, the
split plan, our code and the README/ATTRIBUTION/manifest; the rule and every column are in
`xgpoc.vendor.release`. Inputs are read only after their hashes match the pre-registration's pins
(`research/xg_vendor_preregistration.md`, parameter block) or the completion record of the folder they come
from; labels are never read. The code archive is `git archive` of HEAD, so the tracked tree must be clean.

    data/match_map.csv, shots.csv, recovered_goals.csv, predictions.csv, split_plan.csv
    code/xg_vendor_code.tar.gz   our code at HEAD (experiment package, scripts, tests, study documents, report)
    README.md, ATTRIBUTION.md, MANIFEST.md, manifest.json, complete.json

A finished folder is verified, never rebuilt (remove it on purpose).

    ./run.sh python scripts/vendor_release.py [--out DIR]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from xgpoc.bundle import code_version
from xgpoc.vendor import prereg
from xgpoc.vendor import release as rel
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
OUT = ROOT / "results/xgv_release_v1"
PREREG = ROOT / "research/xg_vendor_preregistration.md"
OPEN_DATA_COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
SCORES = {"primary": ROOT / "results/xgv_eval_primary", "secondary": ROOT / "results/xgv_eval_secondary"}
# Paths of the code archive, relative to the repository root (`ARCHIVE_EXPERIMENT` is prefixed to the first group).
ARCHIVE_EXPERIMENT = ["xgpoc", "scripts", "tests", "run.sh", "pytest.ini", "research/xg_vendor_preregistration.md",
                      "research/xg_vendor_prereg_hashes.json", "research/xg_vendor_timing_rule.md",
                      "research/xg_vendor_delta_rule.md", "research/xg_vendor_statsbomb_terms.md",
                      "research/quality_v13_decisions.tsv", "results_reports/xg_vendor_2026-10"]
ARCHIVE_REPO = ["pyproject.toml", "uv.lock", "LICENSE"]


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def need(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"STOPPED, nothing written. {message}")


def checked(path: Path, sha256: str) -> Path:
    need(path.is_file(), f"{path} is missing")
    need(file_sha256(path) == sha256, f"{path} is not the file the study used (sha256 differs)")
    return path


def recorded(folder: Path, name: str) -> Path:
    """`folder/name` after its hash matches the folder's completion record."""
    files = json.loads((folder / "complete.json").read_text())["files"]
    need(name in files, f"{folder / 'complete.json'} does not list {name}")
    return checked(folder / name, files[name])


def read_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict, dict, dict]:
    """Source tables, member scores, ensemble, pre-registration parameters and the input hashes, each checked."""
    params = prereg.read_parameters(PREREG.read_text())
    pins = params["pins"]

    def pinned(name: str) -> Path:
        return checked(ROOT / pins[name]["path"], pins[name]["sha256"])

    checked(ROOT / pins["pairing_record"]["path"], pins["pairing_record"]["sha256"])
    pairing = ROOT / "results/xgv_pairing"
    paths = {
        "match_map": recorded(pairing, "match_map.csv"),
        "pairs": pinned("pairs"),
        "recovered_goals": recorded(pairing, "recovered_goals.csv"),
        "eval_table": pinned("combined_label_free_table"),
        "split_plan": pinned("split_plan"),
    }
    need({k: str(p.relative_to(ROOT)) for k, p in paths.items()} == rel.SOURCES, "source paths differ from release.SOURCES")
    sources = {k: pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for k, p in paths.items()}
    scores_paths = [recorded(folder, "scores.parquet") for folder in SCORES.values()]
    scores = pd.concat([pd.read_parquet(p) for p in scores_paths], ignore_index=True)
    scores = scores[scores.role != "gate"].reset_index(drop=True)
    ensemble_path = pinned("g_combo_ensemble")
    inputs = {str(p.relative_to(ROOT)): file_sha256(p) for p in [*paths.values(), *scores_paths, ensemble_path, PREREG]}
    return sources, scores, json.loads(ensemble_path.read_text()), params, inputs


def readme(tables: dict[str, pd.DataFrame], comparator_sha: str, penalty_rate: float) -> str:
    rows = {f: len(t) for f, t in tables.items()}
    return f"""# xg-vendor release package

Broadcast-frame xG (G-combo) against StatsBomb vendor xG and a positional comparator, on SoccerNet shots paired with
StatsBomb open data. This package holds the identifiers that join the two sources, the outcome of our pairing rules,
our G-combo predictions and our code. It holds **no StatsBomb data** and **no SoccerNet video, frames, clips,
embeddings or model weights**. See `ATTRIBUTION.md` for the terms and `MANIFEST.md` for every column with its
provenance and for every source column left out, with the reason.

## Contents

| File | Rows | What |
|---|---|---|
| `data/match_map.csv` | {rows['match_map.csv']} | every SoccerNet game with its StatsBomb match id, or why it has none |
| `data/shots.csv` | {rows['shots.csv']} | the pair table: every index shot of the paired games with its StatsBomb shot event id |
| `data/recovered_goals.csv` | {rows['recovered_goals.csv']} | SoccerNet goals without a shot annotation and the StatsBomb shot each was recovered to |
| `data/predictions.csv` | {rows['predictions.csv']} | G-combo logits and probabilities at offsets 0 and -400 ms for every evaluation row |
| `data/split_plan.csv` | {rows['split_plan.csv']} | role (evaluation, gate, training) and fold of every SoccerNet game in the retrain |
| `code/xg_vendor_code.tar.gz` | | our code at the commit in `complete.json`, with the pre-registration and the report |

The frame predictions are not penalty-routed: in the analyses a penalty gets the pool penalty rate ({penalty_rate:.4f}) instead of
the model's value, and the penalty flag is a SoccerNet annotation or a StatsBomb value, so it is not shipped.

## Rebuilding what is not shipped

Everything StatsBomb-derived is rebuilt from StatsBomb open data with the code in `code/`:

1. Register with StatsBomb and accept their user agreement (statsbomb.com, open data), then fetch the open data at
   commit `{OPEN_DATA_COMMIT}`: `./run.sh python scripts/vendor_fetch_statsbomb.py`. It pins that commit and writes
   the full events, the shots (with freeze frames) and the derived tables.
2. Get SoccerNet: the Labels-v2 annotations and, for frames, the 720p videos (SoccerNet NDA). The shot index v1.2
   comes from `scripts/collect_soccernet.py` (clips and frames) and `scripts/relabel_index.py` (v1.2 label rules).
3. Pair: `./run.sh python scripts/vendor_pair.py`. The match map and pair table it writes must give the ids of
   `data/match_map.csv`, `data/shots.csv` and `data/recovered_goals.csv`; with them come the vendor xG, StatsBomb
   outcomes, half offsets and timing residuals.
4. Positional comparator: `./run.sh python scripts/vendor_comparator.py`. Its `predictions.parquet` should have
   sha256 `{comparator_sha}` (deterministic: fixed seeds, single-thread fits).
5. Labels and analyses: `xgpoc/vendor/labels.py` builds the labels (SoccerNet goal and on-target, D4); join them to
   `data/predictions.csv` by `key`, route penalties, and run `scripts/vendor_ladder.py`, `vendor_probes.py` and
   `vendor_leakage.py` as described in the pre-registration (`research/xg_vendor_preregistration.md`, section 18).
   H3 needs the frame embeddings, which only the SoccerNet video gives.

## Attribution

Data: StatsBomb open data (StatsBomb Services Ltd). Video and annotations: SoccerNet. See `ATTRIBUTION.md`.
"""


def attribution() -> str:
    return f"""# Attribution and terms

## StatsBomb

The ids in this package (`statsbomb_match_id`, `statsbomb_event_id`) refer to StatsBomb open data
(https://github.com/statsbomb/open-data), commit `{OPEN_DATA_COMMIT}`. The package contains no StatsBomb data:
no event, shot, freeze frame, location, outcome, timestamp, xG or value computed from them. The ids are join keys;
the data behind them must be fetched from StatsBomb by each user under StatsBomb's terms.

Anyone who publishes analysis built on that data must follow StatsBomb's terms: name StatsBomb as the data source
and show the StatsBomb logo (README "Terms & Conditions"; Public Data User Agreement 1.4), no distribution or
transfer of the data to third parties without StatsBomb's written consent (1.2.1, 7), no commercial exploitation
(1.2.2). Every table and figure of our report built from StatsBomb data carries "Data: StatsBomb".

Data: StatsBomb.

## SoccerNet

SoccerNet game ids, shot ids, halves and times in the half video come from SoccerNet-v2 (Deliège et al., "SoccerNet-v2:
A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos", CVPR Workshops 2021). The videos
are under the SoccerNet NDA and are not redistributed; neither are frames, clips, embeddings or model weights
derived from them. SoccerNet annotations (labels, teams, splits) are not in the package.

## Our code and predictions

Code: Apache License 2.0 (`LICENSE` in the code archive). Predictions and pairing tables: produced by this study;
cite the study when you use them.
"""


def write_package(out: Path, tables: dict[str, pd.DataFrame], texts: dict[str, str], archive: bytes | None,
                  record: dict) -> dict:
    """Stage every file, hash it, write the manifest and the completion record, publish by one rename."""
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "data").mkdir(parents=True)
    hashes = {}
    for name, frame in tables.items():
        path = stage / "data" / name
        frame.to_csv(path, index=False, lineterminator="\n")
        hashes[name] = file_sha256(path)
    record_manifest = rel.manifest(tables, hashes)
    (stage / "manifest.json").write_text(json_text(record_manifest))
    (stage / "MANIFEST.md").write_text(rel.manifest_markdown(record_manifest))
    for name, text in texts.items():
        (stage / name).write_text(text)
    if archive is not None:
        (stage / "code").mkdir()
        (stage / "code" / "xg_vendor_code.tar.gz").write_bytes(archive)
    files = {str(p.relative_to(stage)): file_sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    (stage / "complete.json").write_text(json_text({**record, "files": files}))
    out.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(out)
    return files


def verify(out: Path) -> None:
    files = json.loads((out / "complete.json").read_text())["files"]
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} - {"complete.json"}
    need(on_disk == set(files), f"{out}: files differ from its completion record")
    bad = [name for name, sha in files.items() if file_sha256(out / name) != sha]
    need(not bad, f"{out}: changed files {bad}")
    print(f"Verified {out}: {len(files)} files; nothing rebuilt")


def code_archive() -> bytes:
    prefix = subprocess.check_output(["git", "rev-parse", "--show-prefix"], cwd=ROOT, text=True).strip()
    paths = [prefix + p for p in ARCHIVE_EXPERIMENT] + ARCHIVE_REPO
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    return subprocess.check_output(["git", "archive", "--format=tar.gz", "--prefix=xg_vendor_code/", "HEAD", "--", *paths],
                                   cwd=top)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if (args.out / "complete.json").exists():
        verify(args.out)
        return
    need(not args.out.exists() or not any(args.out.iterdir()), f"{args.out} exists without a completion record; remove it")
    code = code_version(ROOT)
    need(not code["dirty"], "the tracked tree has uncommitted changes; the code archive is HEAD, so commit first")

    sources, scores, ensemble, params, inputs = read_inputs()
    try:
        frames = rel.unrouted_frame_logits(scores, ensemble, sources["eval_table"])
        tables = rel.build_tables(sources, frames, params["frame_exclusions"])
    except rel.ReleaseError as e:
        need(False, str(e))
    texts = {"README.md": readme(tables, params["pins"]["comparator_predictions"]["sha256"], ensemble["calibration"]["penalty_rate"]),
             "ATTRIBUTION.md": attribution()}
    record = {"code": code, "inputs": inputs, "open_data_commit": OPEN_DATA_COMMIT,
              "rows": {name: len(frame) for name, frame in tables.items()}}
    files = write_package(args.out, tables, texts, code_archive(), record)
    print(json_text({"out": str(args.out), "rows": record["rows"], "files": files}))


if __name__ == "__main__":
    main()
