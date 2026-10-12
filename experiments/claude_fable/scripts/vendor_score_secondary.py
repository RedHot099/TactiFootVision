"""Score the secondary population (recovered goals and StatsBomb-only shots) with the retrained G-combo and write the
combined label-free evaluation table (xg-vendor ticket 13; the scoring stage is approved GPU work, decision D2).

    ./run.sh python scripts/vendor_score_secondary.py score --gpu-approved   # GPU: ten fold models x two offsets + DINOv2
    ./run.sh python scripts/vendor_score_secondary.py replay                 # GPU: a sample again in identical batch order
    ./run.sh python scripts/vendor_score_secondary.py build                  # CPU: ensemble, join, combined table, checks

It reuses ticket 11's code path (`scripts/vendor_score_eval.py`: `score_member`, `score_dino`, `stage_replay`, the ensemble
and its calibration; `xgpoc.vendor.eval_scoring`), with the rows of the re-download's frame index instead of the paired
shots. The frames are the JPEGs of `xgv_secondary_v1` (offsets 0 and -400 ms at StatsBomb time plus the half offset); the
clips of that folder are an archive and are not read here.

Stage `score`. Checks that no other job holds the GPU, the re-download's completion record (no refused half, the frame
index hash), the population table's hash against the pairing record and that every bundle is the one
`results/xgv_g_combo/ensemble.json` lists; every frame the index calls ok must exist with its recorded sha256. A model that is
done is kept in `members/`, so an interrupted run resumes. Pooled embeddings of the fold 0 and fold 1 seed-0 models and the
frozen DINOv2 embeddings are written for the shots with an offset-0 frame.
Stage `replay`. As ticket 11's: the first 64 scored rows per offset again in a fresh process, same batches.
Stage `build`. Averages the ten models with the single calibration, joins the secondary rows with vendor xG and the
frozen comparator rungs, appends them to ticket 11's primary table (rows unchanged, first) and writes the combined table,
the combined embedding matrices (primary rows first), the list of targets without frames and the completion record, which
proves that the primary rows are byte-identical to ticket 11's table. No label is read or joined.
"""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import torch
import vendor_score_eval as ve
from xgpoc.bundle import code_version, sha256_file
from xgpoc.paths import INDEX
from xgpoc.vendor import eval_scoring as es
from xgpoc.vendor import eval_secondary as esec
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import split_check
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
OUT = ROOT / "results/xgv_eval_secondary"
COMBINED = ROOT / "results/xgv_eval_combined"
PRIMARY_OUT = ROOT / "results/xgv_eval_primary"
REDOWNLOAD = Path("/mnt/storage/tactifoot_research/xgv_secondary_v1")
REDOWNLOAD_VERSION = "xgv_secondary_frames_v1"
VERSION = "xgv_eval_combined_v1"
SECONDARY_VERSION = "xgv_eval_secondary_v1"
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "scripts/vendor_score_eval.py", ROOT / "xgpoc/vendor/eval_secondary.py",
                ROOT / "xgpoc/vendor/eval_scoring.py", ROOT / "xgpoc/vendor/evaltable.py")


# ------------------------------------------------------------------ inputs
def read_frames(root: Path) -> tuple[pd.DataFrame, dict]:
    """The re-download's frame index after checking its completion record (version, no refused half, index hash)."""
    record = json.loads((root / "complete.json").read_text())
    if record["identity"]["version"] != REDOWNLOAD_VERSION:
        raise SystemExit(f"{root} is not the {REDOWNLOAD_VERSION} data version")
    if record["summary"]["halves_refused"] or (root / "refused_halves.json").read_text().strip() not in ("[]", "{}"):
        raise SystemExit("the re-download refused a half; resolve it before scoring")
    if record["files"]["frames_index.parquet"] != file_sha256(root / "frames_index.parquet"):
        raise SystemExit("frames_index.parquet differs from the re-download's completion record")
    return pd.read_parquet(root / "frames_index.parquet"), {"complete_json_sha256": file_sha256(root / "complete.json"),
                                                            "frames_index_sha256": record["files"]["frames_index.parquet"],
                                                            "summary": record["summary"]}


def read_unpaired(pairing: Path) -> tuple[pd.DataFrame, str]:
    """The population table of the unpaired StatsBomb shots, after checking it against the pairing's completion record."""
    record = json.loads((pairing / "complete.json").read_text())
    if file_sha256(pairing / "sb_unpaired.parquet") != record["files"]["sb_unpaired.parquet"]:
        raise SystemExit("sb_unpaired.parquet differs from the pairing's completion record")
    return pd.read_parquet(pairing / "sb_unpaired.parquet"), record["files"]["sb_unpaired.parquet"]


def secondary_inputs(args) -> dict:
    frames, redownload = read_frames(args.redownload)
    unpaired, unpaired_sha = read_unpaired(args.pairing)
    index = pd.read_parquet(args.index, columns=["game", "split"]).drop_duplicates()
    if index.game.duplicated().any():
        raise SystemExit("a game has two splits in the index")
    shots = esec.secondary_shots(frames, unpaired, dict(zip(index.game, index.split, strict=True)))
    return {"shots": shots, "redownload": redownload, "unpaired_sha256": unpaired_sha, "ensemble": ve.read_ensemble(args.g_root)}


# ------------------------------------------------------------------ stage score
def stage_score(args) -> None:
    ve.run_score(args, secondary_inputs)


# ------------------------------------------------------------------ stage build
def verified_primary(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Ticket 11's label-free table and sidecar and its completion record, after re-hashing every file against the record."""
    record = json.loads((root / "complete.json").read_text())
    for name, expected in record["files"].items():
        if sha256_file(root / name) != expected:
            raise SystemExit(f"{root / name} differs from ticket 11's completion record")
    table, meta = et.load_table(root, ve.TABLE_NAME)
    if meta["table_sha256"] != record["summary"]["label_free_table_sha256"] or meta["labels_state"] != "absent":
        raise SystemExit("ticket 11's table is not the recorded label-free table")
    return table, meta, record


def combined_embeddings(args, primary_meta: dict, secondary_rows: int) -> dict[str, np.ndarray]:
    """Per embedding set, the primary matrix followed by the secondary one (checked for width, dtype and row count)."""
    arrays = {}
    for column, entry in primary_meta["embedding_sets"].items():
        primary = np.load(args.primary / entry["file"])
        extra = np.load(args.out / entry["file"])
        if extra.shape != (secondary_rows, primary.shape[1]) or extra.dtype != primary.dtype or primary.shape[0] != entry["rows"]:
            raise SystemExit(f"{entry['file']}: secondary {extra.shape} {extra.dtype} does not extend primary {primary.shape} {primary.dtype}")
        arrays[column] = np.concatenate([primary, extra])
    return arrays


def stage_build(args) -> None:
    if (args.combined / "complete.json").exists():
        raise SystemExit(f"{args.combined} is complete; delete it to rebuild")
    shots = pd.read_parquet(args.out / "shots.parquet")
    state = secondary_inputs(args)
    if not state["shots"].equals(shots):
        raise SystemExit("the rows and frames found now differ from those scored")
    replay = json.loads((args.out / "replay.json").read_text())
    if not replay["passed"]:
        raise SystemExit("the replay did not pass")
    primary, primary_meta, primary_record = verified_primary(args.primary)
    ensemble = state["ensemble"]
    members, temperatures = ve.load_members(args, ensemble)
    calibration = json.loads((args.g_root / "ensemble.json").read_text())["calibration"]
    if primary_record["identity"]["ensemble_json_sha256"] != sha256_file(args.g_root / "ensemble.json"):
        raise SystemExit("ticket 11 scored with another ensemble.json")
    scores = es.assemble_scores(shots, members, temperatures, calibration)

    _, pairing = ve.read_pairing(args.pairing)
    comparator_record = ve.wait_for_comparator(args.comparator, 0)
    if comparator_record["identity"]["inputs"]["pairing"]["complete_json_sha256"] != pairing["complete_json_sha256"]:
        raise SystemExit("the comparator was built on another pairing")
    if comparator_record["files"]["predictions.parquet"] != primary_record["identity"]["comparator_predictions_sha256"]:
        raise SystemExit("the comparator predictions are not the ones ticket 11 joined")
    comparator = pd.read_parquet(args.comparator / "predictions.parquet")

    counts = {entry["rows"] for entry in primary_meta["embedding_sets"].values()}
    if len(counts) != 1:
        raise SystemExit(f"ticket 11's three embedding sets differ in their row counts: {sorted(counts)}")
    n_primary_embedded = counts.pop()
    n_embedded = int(es.scored_mask(shots, "0").sum())
    secondary = esec.build_secondary_table(shots, scores, comparator, first_embedding_row=n_primary_embedded)
    combined = esec.combine_tables(primary, secondary)
    arrays = combined_embeddings(args, primary_meta, n_embedded)

    args.combined.mkdir(parents=True, exist_ok=True)
    sets = {}
    for column, entry in primary_meta["embedding_sets"].items():
        np.save(args.combined / entry["file"], arrays[column])
        sets[column] = {**entry, "sha256": sha256_file(args.combined / entry["file"]), "rows": int(arrays[column].shape[0]),
                        "model": f"{entry['model']}; rows 0-{entry['rows'] - 1} are the primary shots (ticket 11, unchanged), the rest the "
                                 "secondary shots with a scored offset-0 frame in table order"}
    fingerprint = es.shots_fingerprint(shots)
    sources = {**primary_meta["sources"], "primary_label_free_table": primary_meta["table_sha256"], "redownload_complete.json": state["redownload"]["complete_json_sha256"],
               "redownload_frames_index": state["redownload"]["frames_index_sha256"], "sb_unpaired.parquet": state["unpaired_sha256"],
               "secondary_frames": fingerprint["sha256"]}
    notes = (f"combined confirmatory table, labels absent: the {len(primary):,} primary rows of ticket 11 (unchanged, first) and the secondary rows "
             "(population recovered_goal or statsbomb_only), one row per target of the pairing, whether or not it has frames",
             "secondary rows: key and statsbomb_event_id are the StatsBomb event id, anchor_convention statsbomb_time (frame time = StatsBomb time + "
             "the half's frozen offset), no SoccerNet shot id and no timing residual; is_penalty (a label column, joined later) is the StatsBomb "
             "penalty (shot type Penalty) and routes the frame logit of a penalty to the penalty rate",
             "a secondary target whose frame lies beyond the end of the half video keeps its row with frame_status naming the reason and NaN "
             "frame logits; the analyses leave such rows out",
             "primary notes: " + " | ".join(primary_meta["notes"]))
    meta = et.write_table(combined, args.combined, ve.TABLE_NAME, kind="confirmatory", embedding_sets=sets, notes=notes, sources=sources)
    loaded, loaded_meta = et.load_table(args.combined, ve.TABLE_NAME)   # reads and validates

    identity = primary_identity(loaded, primary, primary_meta, args, arrays)
    no_frames = esec.no_frame_report(shots)
    no_frames.to_csv(args.combined / "no_frame_targets.csv", index=False)
    disagreements = esec.frame_clip_disagreements(shots)
    disagreements.to_csv(args.combined / "frame_clip_disagreements.csv", index=False)
    scores.to_parquet(args.out / "scores.parquet", index=False, compression="zstd")

    secondary_dir = {str(p.relative_to(args.out)): sha256_file(p) for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "complete.json"}
    files = {str(p.relative_to(args.combined)): sha256_file(p) for p in sorted(args.combined.rglob("*")) if p.is_file() and p.name != "complete.json"}
    sec = loaded[loaded.population.ne("primary")]
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "identity": {"version": VERSION, "table_kind": "confirmatory", "schema_version": et.SCHEMA_VERSION, "bundles": {
                  es.member_name(m["fold"], m["seed"]): m["sha256"] for m in ensemble["members"]},
                           "plan_sha256": split_check.PLAN_SHA256, "pairing": pairing, "comparator_complete_json_sha256": file_sha256(args.comparator / "complete.json"),
                           "comparator_predictions_sha256": comparator_record["files"]["predictions.parquet"],
                           "ensemble_json_sha256": sha256_file(args.g_root / "ensemble.json"), "index_sha256": sha256_file(args.index),
                           "primary_complete_json_sha256": file_sha256(args.primary / "complete.json"),
                           "redownload": {k: v for k, v in state["redownload"].items() if k != "summary"}, "sb_unpaired_sha256": state["unpaired_sha256"],
                           "secondary_frame_fingerprint": fingerprint, "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES},
                           "code": code_version(ROOT)},
              "summary": {"rows": meta["rows"], "by_population_split": meta["by_population_split"], "labels_state": loaded_meta["labels_state"],
                          "combined_label_free_table_sha256": meta["table_sha256"], "frame_status": loaded.frame_status.value_counts().to_dict(),
                          "secondary_frame_status": sec.frame_status.value_counts().to_dict(),
                          "secondary_scored_both_offsets": int(sec.frame_status.eq(et.OK).sum()), "secondary_without_frames": int(len(no_frames)),
                          "secondary_without_frames_by_population": no_frames.population.value_counts().to_dict(),
                          "no_frame_reasons": no_frames.frame_status_0.value_counts().to_dict(),
                          "frame_clip_disagreements": disagreements.to_dict("records"), "redownload_summary": state["redownload"]["summary"],
                          "replay": {k: v for k, v in replay.items() if k != "members"},
                          "embedding_rows": {c: s["rows"] for c, s in sets.items()}, "primary_identity": identity},
              "secondary_files": secondary_dir, "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
                                                           "torch": torch.__version__},
              "files": files}
    (args.combined / "complete.json").write_text(ve.json_text(record))
    secondary_record = {"completed_at": record["completed_at"], "identity": {**record["identity"], "version": SECONDARY_VERSION,
                        "combined_label_free_table_sha256": meta["table_sha256"]}, "files": secondary_dir}
    (args.out / "complete.json").write_text(ve.json_text(secondary_record))
    print(f"wrote {args.combined}; combined label-free table sha256 {meta['table_sha256']}; primary rows identical: {identity['identical']}", flush=True)


def primary_identity(combined: pd.DataFrame, primary: pd.DataFrame, primary_meta: dict, args, arrays: dict) -> dict:
    """The proof that the primary rows are ticket 11's: the combined table's primary rows, written alone by the table writer,
    hash to the sha256 of ticket 11's table file; the rows are equal frame for frame; and the embedding matrices start with
    ticket 11's, unchanged. Raises SystemExit when any of it fails."""
    n_primary = len(primary)
    file_sha = sha256_file(args.primary / f"{ve.TABLE_NAME}.parquet")
    identity = {"ticket_11_table_file_sha256": file_sha, "ticket_11_sidecar_table_sha256": primary_meta["table_sha256"],
                "combined_primary_rows_sha256": esec.primary_rows_sha256(combined),
                "rows_equal": bool(combined.iloc[:n_primary].equals(primary)), "primary_first": bool(combined.population.iloc[:n_primary].eq("primary").all()
                                                                                                    and combined.population.iloc[n_primary:].ne("primary").all()),
                "embedding_prefix_equal": {c: bool(np.array_equal(arrays[c][: e["rows"]], np.load(args.primary / e["file"])))
                                           for c, e in primary_meta["embedding_sets"].items()}}
    identity["identical"] = bool(identity["combined_primary_rows_sha256"] == file_sha == identity["ticket_11_sidecar_table_sha256"] and identity["rows_equal"]
                                 and identity["primary_first"] and all(identity["embedding_prefix_equal"].values()))
    if not identity["identical"]:
        raise SystemExit(f"the primary rows are not byte-identical to ticket 11's: {identity}")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("score", "replay", "build"))
    parser.add_argument("--gpu-approved", action="store_true", help="required by the scoring stage")
    parser.add_argument("--out", type=Path, default=OUT, help="scores, members and secondary embeddings")
    parser.add_argument("--combined", type=Path, default=COMBINED, help="the combined table, its embeddings and the completion record")
    parser.add_argument("--primary", type=Path, default=PRIMARY_OUT, help="ticket 11's output (read only)")
    parser.add_argument("--redownload", type=Path, default=REDOWNLOAD, help="the re-download (read only)")
    parser.add_argument("--g-root", type=Path, default=ve.G_COMBO)
    parser.add_argument("--pairing", type=Path, default=ve.PAIRING)
    parser.add_argument("--comparator", type=Path, default=ve.COMPARATOR)
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    started = time.time()
    {"score": stage_score, "replay": ve.stage_replay, "build": stage_build}[args.stage](args)
    print(f"{args.stage} done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
