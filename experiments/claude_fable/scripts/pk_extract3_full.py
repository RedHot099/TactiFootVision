"""Run audit 3's chosen configuration on every index shot at t0 and -400 ms (pitch-keypoints 04), resumably.

    ./run.sh python scripts/pk_extract3_full.py [--out cache/xgd_full3] [--report results/xgd_full3] [--chunk 500]
        [--limit N] [--device cuda] [--detector dd2c_1280]

Which configuration: `chosen` in audit 3's `results/xgd_audit3/choice.json`, read first, here and nowhere else (a name,
no StatsBomb value). Everything after it runs inside `no_statsbomb_reads()`; the index is read for the frame paths and
the game, no label column; the pairing is never read.

Each shot's t0 and -400 ms frames are analysed with teams, each giving a feature row; the ball comes from the
configuration's source order without any frame after the row's own (`extract3.extract` with `STUDY_OFFSETS`); other
frames are analysed without teams and only when the order reaches them.

Resumable: shots go in chunks of `--chunk` in shot-id order; each finished chunk is written to
<out>/<config>/parts/<n>.frames.parquet and <n>.features.parquet (renamed into place when complete) and logged with its
wall time in parts/sessions.jsonl. A re-run with the same arguments skips finished chunks; the inputs of the parts
(configuration, models, shots, code) are pinned in parts/identity.json and a run with other inputs is refused. When
every chunk is done the parts are joined into frames.parquet and features.parquet and complete.json is written, last;
the t0 rows of the shots in the audit cache (`--reference`/<config>) are compared with it. <report>/coverage_by_role.csv
(with its complete.json) gives the coverage by split role (xg-vendor split plan) and offset.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import pandas as pd
from tqdm import tqdm
from xgd_extract_full import PLAN, chosen_config, reference_differences
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as e3
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
CHOICE = ROOT / "results/xgd_audit3/choice.json"


def chunks(shot_ids: list[str], size: int) -> list[list[str]]:
    """Consecutive chunks of at most `size` shots."""
    return [shot_ids[i:i + size] for i in range(0, len(shot_ids), size)]


def pinned_identity(parts: Path, identity: dict) -> None:
    """Write the parts' identity on the first run; refuse a later run whose identity differs."""
    path = parts / "identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise SystemExit(f"{parts} holds parts of other inputs (configuration, models, shots or code); use a new --out")
        return
    parts.mkdir(parents=True, exist_ok=True)
    path.write_text(ex.json_text(identity))


def write_part(parts: Path, n: int, frames: pd.DataFrame, features: pd.DataFrame) -> None:
    """Both tables of chunk `n`, each renamed into place only when written."""
    for kind, table in (("frames", frames), ("features", features)):
        tmp = parts / f"{n:03d}.{kind}.parquet.tmp"
        table.to_parquet(tmp, index=False)
        tmp.rename(parts / f"{n:03d}.{kind}.parquet")


def done(parts: Path, n: int) -> bool:
    """Whether chunk `n` has both of its tables in place."""
    return all((parts / f"{n:03d}.{kind}.parquet").exists() for kind in ("frames", "features"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "cache/xgd_full3")
    parser.add_argument("--report", type=Path, default=ROOT / "results/xgd_full3")
    parser.add_argument("--choice", type=Path, default=CHOICE)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--reference", type=Path, default=ROOT / "cache/xgd_extract3")
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--limit", type=int, help="first N shots only (smoke runs)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--detector", choices=sorted(e3.DETECTORS), default=e3.DEFAULT_DETECTOR.name)
    args = parser.parse_args()
    detector = e3.DETECTORS[args.detector]

    name = chosen_config(args.choice)
    choice_input = {"path": str(args.choice), "sha256": sha256_file(args.choice), "chosen": name}
    config = e3.CONFIGS[name]
    folder = args.out / name
    parts = folder / "parts"
    for finished in (folder / "complete.json", args.report / "complete.json"):
        if finished.exists():
            raise SystemExit(f"{finished} exists; use a new --out / --report")
    if sha256_file(e3.KEYPOINT_WEIGHTS) != e3.KEYPOINT_SHA256:
        raise SystemExit(f"{e3.KEYPOINT_WEIGHTS} is not the declared pitch model")

    with no_statsbomb_reads():
        import tactifoot_vision as tv

        shot_games = pd.read_parquet(args.index, columns=["shot_id", "game"]).sort_values("shot_id", kind="stable")
        shot_ids = shot_games.shot_id.tolist()[: args.limit]
        shot_games = shot_games[shot_games.shot_id.isin(shot_ids)]
        code = records.code_hashes(Path(__file__))
        identity = {"config": e3.settings(config, detector), "chunk": args.chunk,
                    "models": records.model_hashes(detector.weights, e3.KEYPOINT_WEIGHTS, ex.SIGLIP),
                    "code_digest": code["digest"], "libraries": records.library_versions(),
                    "inputs": {"index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                         "columns": ["shot_id", "game", *ex.FRAME_COLUMNS.values()]},
                               "shots": {"rule": "every index shot", "count": len(shot_ids),
                                         "digest": canonical_digest(shot_ids), "limit": args.limit},
                               "choice": choice_input}}
        pinned_identity(parts, identity)
        todo = [(n, ids) for n, ids in enumerate(chunks(shot_ids, args.chunk)) if not done(parts, n)]
        if todo:
            embedder = tv.teams.SigLIPEmbedder(model_name=ex.SIGLIP, device=args.device)
            analyzer, ball_only = e3.build_analyzers(embedder, device=args.device, detector=detector)
            for n, ids in todo:
                started, clock = datetime.now(UTC), time.perf_counter()
                out = e3.extract(ex.frame_table(args.index, ids), [config], analyzer, ball_only, cv2.imread,
                                 offsets=e3.STUDY_OFFSETS,
                                 progress=lambda rows, n=n: tqdm(rows, desc=f"{name} part {n}", mininterval=60))
                write_part(parts, n, ex.frames_table(out.frames), pd.DataFrame(out.features[name]))
                session = {"part": n, "shots": len(ids), "started": started.isoformat(timespec="seconds"),
                           "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                           "seconds": round(time.perf_counter() - clock, 1), "git": records.git_state(),
                           "device": args.device}
                with (parts / "sessions.jsonl").open("a") as log:
                    log.write(json.dumps(session, sort_keys=True) + "\n")
                print(json.dumps(session), flush=True)

        n_parts = len(chunks(shot_ids, args.chunk))
        frames = pd.concat([pd.read_parquet(parts / f"{n:03d}.frames.parquet") for n in range(n_parts)], ignore_index=True)
        features = pd.concat([pd.read_parquet(parts / f"{n:03d}.features.parquet") for n in range(n_parts)], ignore_index=True)
        if sorted(features.shot_id.unique()) != shot_ids:
            raise SystemExit("the parts do not hold every shot exactly once")
        frames.to_parquet(folder / "frames.parquet", index=False)
        features.to_parquet(folder / "features.parquet", index=False)
        sessions = [json.loads(line) for line in (parts / "sessions.jsonl").read_text().splitlines()]
        seconds = sum(s["seconds"] for s in sessions)
        summary = {str(offset): ex.coverage(rows) for offset, rows in features.groupby("offset_ms", sort=False)}
        summary["frames_analysed"] = len(frames)
        summary["frames_by_offset"] = {str(k): int(v) for k, v in frames.offset_ms.value_counts().items()}
        reference = reference_differences(features, args.reference / name / "features.parquet")
        record = {"config": e3.settings(config, detector), "identity": identity | {"code": code, "git": records.git_state()},
                  "run": {"offsets_ms": list(e3.STUDY_OFFSETS), "sessions": sessions, "seconds": round(seconds, 1),
                          "ms_per_shot": round(1000 * seconds / max(len(shot_ids), 1), 1)},
                  "coverage": summary, "audit_cache_t0_differences": reference,
                  "files": records.output_hashes(folder)}
        (folder / "complete.json").write_text(ex.json_text(record))

        plan = pd.read_csv(args.plan, usecols=["game", "role"])
        table = ex.coverage_by_role(features, shot_games, plan)
        args.report.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.report / "coverage_by_role.csv", index=False, float_format="%.4f")
        report_record = {"config": name, "cache": {"path": str(folder), "complete_json_sha256": sha256_file(folder / "complete.json")},
                         "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan), "columns": ["game", "role"]},
                         "code": code, "git": records.git_state(), "files": records.output_hashes(args.report)}
        (args.report / "complete.json").write_text(ex.json_text(report_record))
    print(table.to_string(index=False))
    print(f"done: {len(shot_ids)} shots, {name}, {seconds / 60:.1f} min ({1000 * seconds / max(len(shot_ids), 1):.0f} ms per shot); "
          f"audit-cache t0 differences: {reference and reference['shots_differing']}")


if __name__ == "__main__":
    main()
