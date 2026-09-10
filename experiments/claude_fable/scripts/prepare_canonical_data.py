"""Build versioned clip inputs and quality records without changing historical data or labels."""

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd
import torch

from compare_ai_contact import verify_evidence
from diagnostic_review import json_text, validate_annotations, verify_packet
from xgpoc.data.canonical_frames import (
    array_digest, extract_inputs, image_quality, model_input_digest, runtime_identity, write_png,
)
from xgpoc.paths import SOCCERNET
from xgpoc.v13.protocol import digest


def verify_result(root: Path) -> dict:
    manifest = json.loads((root / "complete.json").read_text())
    for name, checksum in manifest["identity"]["sources"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Source changed: {name}")
    for name, checksum in manifest["files"].items():
        if digest(root / name) != checksum:
            raise ValueError(f"Output changed: {name}")
    return manifest


def complete(stage: Path, out: Path, identity: dict) -> None:
    for name, checksum in identity["sources"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Source changed during build: {name}")
    files = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    (stage / "complete.json").write_text(json_text({"identity": identity, "files": files}))
    stage.rename(out)


def source_inventory(sample: pd.DataFrame, assets_path: Path, raw_root: Path) -> pd.DataFrame:
    assets = [json.loads(line) for line in assets_path.read_text().splitlines() if line.strip()]
    records = []
    for (game, half), shots in sample.groupby(["game", "half"]):
        matches = [r for r in assets if r.get("game") == game and r.get("half") == half and
                   r.get("component") == "720p" and r.get("sha256") and not r.get("error")]
        hashes = {r["sha256"] for r in matches}
        if len(hashes) != 1:
            raise ValueError(f"Missing or conflicting raw source provenance: {game}, half {half}")
        asset = matches[-1]
        path = raw_root / game / f"{half}_720p.mkv"
        present = path.is_file()
        if present and digest(path) != asset["sha256"]:
            raise ValueError(f"Raw source checksum differs: {path}")
        records.append({"game": game, "half": half, "review_ids": "|".join(shots.review_id),
                        "expected_path": str(path.resolve()), "expected_sha256": asset["sha256"],
                        "expected_bytes": asset["bytes"], "source_available": present,
                        "action": "available_for_separate_source_comparison" if present else "restore_source_if_needed"})
    return pd.DataFrame(records)


def build_tables(sample: pd.DataFrame, annotations: pd.DataFrame, packet: dict, recovery: pd.DataFrame, stage: Path):
    items = {row["review_id"]: row for row in packet["items"]}
    raw = recovery.set_index(["game", "half"])
    shots, frames = [], []
    for shot in sample.itertuples():
        annotation = annotations.loc[shot.review_id]
        usable = annotation.contact_status == "bounded" and not annotation.prior_annotation_exposure
        reason = "available" if usable else "prior_annotation_exposure" if annotation.prior_annotation_exposure else annotation.contact_status
        source = raw.loc[(shot.game, shot.half)]
        metadata = {"review_id": shot.review_id, "shot_id": shot.shot_id, "game": shot.game, "half": shot.half,
                    "role": shot.role, "sampling_group": shot.sampling_group, "source_quality": "lossy_clip",
                    "clip_path": str(Path(shot.clip_path).resolve()), "clip_sha256": digest(Path(shot.clip_path)),
                    "raw_source_sha256": source.expected_sha256, "raw_source_available": bool(source.source_available),
                    "source_goal": bool(shot.goal), "is_penalty": bool(shot.is_penalty),
                    "source_label": shot.label, "annotation_source": annotation.source,
                    "reviewer": annotation.reviewer, "prior_annotation_exposure": bool(annotation.prior_annotation_exposure),
                    "contact_status": annotation.contact_status, "contact_quality": "ai_provisional" if usable else "unresolved",
                    "contact_lower_ms": annotation.contact_lower_ms, "contact_upper_ms": annotation.contact_upper_ms,
                    "contact_width_ms": annotation.contact_upper_ms - annotation.contact_lower_ms if usable else None,
                    "ai_outcome": annotation.outcome, "ai_goal_attribution": annotation.goal_attribution,
                    "ball_visible": annotation.ball_visible, "keeper_visible": annotation.keeper_visible,
                    "body_part": annotation.body_part, "situation": annotation.situation,
                    "label_review_status": "unresolved" if annotation.outcome == "unknown" or annotation.goal_attribution == "unknown" else
                        "ai_agrees_with_source" if (annotation.outcome == "goal") == bool(shot.goal) else "ai_source_disagreement",
                    "extraction_status": "ok", "extraction_error": ""}
        selected, timeline = {}, None
        try:
            if shot.position_ms < 4000:
                raise ValueError("Early-half clip requires an annotation packet with an explicit corrected anchor")
            timeline, selected = extract_inputs(Path(shot.clip_path), float(annotation.contact_lower_ms) if usable else None)
            known = {timeline.relative_ms(i, 4000) for i in range(len(timeline.pts))}
            if any(f["time_ms"] not in known for f in items[shot.review_id]["frames"]):
                raise ValueError("Packet timestamps do not match the normalized source PTS")
        except (ValueError, OSError, subprocess.SubprocessError, cv2.error) as exc:
            metadata.update(extraction_status="failed", extraction_error=f"{type(exc).__name__}: {exc}")
            selected = {}
        shots.append(metadata)
        if timeline is not None and metadata["extraction_status"] == "ok":
            folder = stage / "timelines"
            folder.mkdir(exist_ok=True)
            (folder / f"{shot.review_id}.json").write_text(json_text(asdict(timeline)))
        for variant in ("t0", "precontact"):
            row = {"review_id": shot.review_id, "shot_id": shot.shot_id, "variant": variant,
                   "frame_path": "", "source_frame_index": None, "source_pts": None, "time_base": None,
                   "source_pts_ms": None, "anchor_relative_ms": None, "gap_ms": None,
                   "width": None, "height": None, "file_sha256": None, "pixel_sha256": None, "tensor_sha256": None,
                   "availability": "decode_failed" if metadata["extraction_status"] == "failed" else
                       reason if variant == "precontact" and not usable else "no_eligible_frame"}
            if variant in selected:
                frame = selected[variant]
                path = stage / "frames" / f"{shot.review_id}_{variant}.png"
                write_png(path, frame.bgr)
                row.update(frame_path=str(path.relative_to(stage)), source_frame_index=frame.index,
                           source_pts=frame.pts, time_base=timeline.time_base, source_pts_ms=timeline.milliseconds(frame.index),
                           anchor_relative_ms=frame.relative_ms, gap_ms=float(annotation.contact_lower_ms) - frame.relative_ms if usable else None,
                           width=timeline.width, height=timeline.height, file_sha256=digest(path),
                           pixel_sha256=array_digest(frame.bgr), tensor_sha256=model_input_digest(path),
                           availability="available", **image_quality(frame))
            frames.append(row)
        print(f"{shot.review_id}: {metadata['extraction_status']}, {len(selected)} inputs", flush=True)
    return pd.DataFrame(shots), pd.DataFrame(frames)


def duplicate_frames(frames: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    present = frames[frames.availability.eq("available")].merge(shots[["shot_id", "role"]], on="shot_id", validate="many_to_one")
    rows = []
    for column in ("pixel_sha256", "tensor_sha256"):
        for checksum, group in present.groupby(column):
            if group.shot_id.nunique() > 1:
                rows.append({"kind": column, "sha256": checksum, "shots": "|".join(sorted(group.shot_id.unique())),
                             "review_ids": "|".join(sorted(group.review_id.unique())),
                             "roles": "|".join(sorted(group.role.unique())), "cross_role": group.role.nunique() > 1})
    return pd.DataFrame(rows, columns=["kind", "sha256", "shots", "review_ids", "roles", "cross_role"])


def build(packet_root: Path, annotations_path: Path, assets_path: Path, raw_root: Path, out: Path) -> None:
    packet_manifest, packet = verify_packet(packet_root)
    payload = json.loads(annotations_path.read_text())
    if payload.get("source") != "model_review":
        raise ValueError("This data version requires explicitly provisional AI annotations")
    annotations = validate_annotations(payload, packet, allow_model_review=True).set_index("review_id")
    if payload["run"].get("public_packet_sha256") != digest(packet_root / "public/packet.json"):
        raise ValueError("AI export refers to different packet bytes")
    inspected = verify_evidence(payload, packet, packet_root, annotations_path.parent)
    sample = pd.read_csv(packet_root / "sample.csv").set_index("review_id").loc[annotations.index].reset_index()
    if sample.shot_id.duplicated().any() or not sample.role.isin(["fit", "select"]).all():
        raise ValueError("Require unique development fit/select shots")
    recovery = source_inventory(sample, assets_path, raw_root)
    sources = [Path(__file__), Path("xgpoc/data/canonical_frames.py"), Path("xgpoc/data/datasets.py"),
               Path("scripts/diagnostic_review.py"), Path("scripts/compare_ai_contact.py"), Path("xgpoc/v13/protocol.py"),
               Path("scripts/collect_soccernet.py"), Path("xgpoc/data/frames.py"), Path("configs/overlay_mask.png"),
               annotations_path, packet_root / "manifest.json", packet_root / "sample.csv", assets_path]
    for shot in sample.itertuples():
        clip = Path(shot.clip_path)
        if digest(clip) != packet_manifest["identity"]["clips"][shot.review_id]["sha256"]:
            raise ValueError("Original clip differs from annotation packet")
        sources.append(clip)
    identity = {"version": "canonical_clip_v1", "packet_id": packet["packet_id"], "sample_ids": sample.shot_id.tolist(),
                "source_quality": "lossy_clip", "anchor_ms": 4000, "rule": "latest time <= 0 and strictly below AI lower bound",
                "storage": "native BGR decoded by OpenCV FFmpeg backend; PNG compression 3; no resize",
                "model_input": "existing ImageDataset, 448x252, INTER_AREA, overlay mask, ImageNet normalization",
                "runtime": runtime_identity() | {"torch": str(torch.__version__)},
                "raw_availability": recovery[["expected_path", "source_available"]].to_dict("records"),
                "sources": {str(p.resolve()): digest(p) for p in sources}}
    if out.exists():
        saved = verify_result(out)
        if saved["identity"] != identity:
            raise ValueError("Data inputs or runtime changed; use a new result directory")
        print("Verified existing canonical data; no extraction repeated")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as temporary:
        stage = Path(temporary)
        shots, frames = build_tables(sample, annotations, packet, recovery, stage)
        duplicates = duplicate_frames(frames, shots)
        shots.to_csv(stage / "shots.csv", index=False)
        frames.to_csv(stage / "frames.csv", index=False)
        duplicates.to_csv(stage / "duplicates.csv", index=False)
        recovery.to_csv(stage / "source_recovery.csv", index=False)
        frames.groupby(["variant", "availability"]).size().rename("shots").reset_index().to_csv(stage / "coverage.csv", index=False)
        shutil.copyfile(annotations_path, stage / "annotations.source.json")
        present = frames[frames.availability.eq("available")]
        summary = {"version": identity["version"], "status": "provisional_development_data", "shots": len(shots),
                   "source_goals": int(shots.source_goal.sum()), "available_frames": present.variant.value_counts().to_dict(),
                   "extraction_failures": int(shots.extraction_status.eq("failed").sum()),
                   "unresolved_ai_outcomes": int(shots.ai_outcome.eq("unknown").sum()),
                   "ai_source_label_disagreements": int(shots.label_review_status.eq("ai_source_disagreement").sum()),
                   "checked_evidence_frames": inspected, "duplicate_groups_within_reviewed_shots": len(duplicates),
                   "missing_raw_sources": int((~recovery.source_available).sum()),
                   "missing_raw_bytes": int(recovery.loc[~recovery.source_available, "expected_bytes"].sum()),
                   "near_black_review_frames": int(present.get("near_black_review", pd.Series(dtype=bool)).sum()),
                   "note": "Exact duplicate check covers these reviewed shots only. Blur and adjacent-frame change are measurements, "
                           "not validated replay/cut detectors. AI/source agreement is not independent label verification."}
        (stage / "summary.json").write_text(json_text(summary))
        complete(stage, out, identity)
        print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=SOCCERNET / "raw_assets.jsonl")
    parser.add_argument("--raw-root", type=Path, default=SOCCERNET / "raw")
    parser.add_argument("--out", type=Path, default=Path("cache/v15/canonical_data_v1"))
    args = parser.parse_args()
    build(args.packet, args.annotations, args.assets, args.raw_root, args.out)


if __name__ == "__main__":
    main()
