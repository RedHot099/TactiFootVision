"""Compare original, coarse and near-contact inputs to frozen G using provisional AI intervals."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from diagnostic_review import json_text, validate_annotations, verify_packet
from xgpoc.data.datasets import ImageDataset
from xgpoc.eval.calibration import PlattScaler, route_penalties
from xgpoc.models.common import _predict_logits
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.v13.protocol import digest

VARIANTS = ("original_t0", "coarse_precontact", "clip_t0", "dense_precontact")
PAIRS = (("dense_precontact", "original_t0"), ("dense_precontact", "coarse_precontact"),
         ("dense_precontact", "clip_t0"), ("clip_t0", "original_t0"))


def verify_evidence(payload, packet, packet_root, run_root):
    items = {item["review_id"]: item for item in packet["items"]}
    total = 0
    for row in payload["annotations"]:
        rid = row["review_id"]
        item = items[rid]
        available = {frame["image"]: frame["time_ms"] for frame in item["frames"]}
        seen = set()
        inspected = row.get("inspected_frames", [])
        if not inspected:
            raise ValueError(f"No inspected evidence for {rid}")
        clip_times = None
        for frame in inspected:
            basis = frame.get("time_basis", "anchor")
            path = Path(frame["image"])
            if basis == "anchor":
                if frame["image"] not in available or available[frame["image"]] != frame["time_ms"]:
                    raise ValueError(f"Evidence frame not in packet for {rid}")
                path = packet_root / "public" / path
            elif basis == "clip_pts":
                clip = (packet_root / "public" / item["clip"]).resolve()
                if not path.resolve().is_relative_to((run_root / "audit" / rid).resolve()) or \
                        Path(frame["source_clip"]).resolve() != clip:
                    raise ValueError(f"Foreign evidence source for {rid}")
                if clip_times is None:
                    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(clip)]))
                    clip_times = [float(f["best_effort_timestamp_time"]) * 1000 for f in probe["frames"]]
                index = frame["source_frame_index"]
                if type(index) is not int or not 0 <= index < len(clip_times) or \
                        abs(clip_times[index] - frame["time_ms"]) >= .001:
                    raise ValueError(f"Evidence timestamp differs from clip for {rid}")
            else:
                raise ValueError(f"Unknown evidence time basis for {rid}")
            if digest(path) != frame["sha256"]:
                raise ValueError(f"Evidence image changed: {path}")
            seen.add((frame["image"], frame["time_ms"], basis))
            total += 1
        evidence = row.get("evidence", [])
        if not evidence:
            raise ValueError(f"No visual observations for {rid}")
        for entry in evidence:
            if (entry["image"], entry["time_ms"], entry.get("time_basis", "anchor")) not in seen or not entry.get("observation"):
                raise ValueError(f"Evidence was not inspected for {rid}")
        if row["contact_status"] == "bounded":
            if row["contact_lower_ms"] >= row["contact_upper_ms"]:
                raise ValueError("AI contact bounds must describe distinct frames")
            for side in ("lower", "upper"):
                if not any(e["supports"] == f"{side}_bound" and e["time_ms"] == row[f"contact_{side}_ms"] and
                           e.get("time_basis", "anchor") == "anchor" for e in evidence):
                    raise ValueError(f"Missing {side} bound evidence for {rid}")
    return total


def decode_inputs(clip, lower_ms):
    """Keep native decoded pixels; the same decoder supplies both the t0 control and dense input."""
    capture = cv2.VideoCapture(str(clip))
    selected = {}
    previous = None
    index = -1
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {clip}")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            index += 1
            time_ms = round(float(capture.get(cv2.CAP_PROP_POS_MSEC)) - 4000, 3)
            if previous is not None and time_ms <= previous:
                raise ValueError("Decoded timestamps must increase")
            previous = time_ms
            if time_ms > 0:
                break
            selected["clip_t0"] = (image, time_ms, index)
            if lower_ms is not None and time_ms < lower_ms:
                selected["dense_precontact"] = (image, time_ms, index)
    finally:
        capture.release()
    return selected


def choose_coarse(frames, lower_ms):
    if lower_ms is None:
        return None
    candidates = frames[frames.measured_ms.notna() & frames.measured_ms.lt(lower_ms) &
                        frames.measured_ms.le(0) & frames.nominal_ms.le(0)]
    return candidates.sort_values("measured_ms").iloc[-1] if len(candidates) else None


def build_inputs(sample, annotations, cached, staging):
    rows = []
    for shot in sample.itertuples():
        annotation = annotations.loc[shot.review_id]
        usable = annotation.contact_status == "bounded" and not annotation.prior_annotation_exposure
        lower = float(annotation.contact_lower_ms) if usable else None
        decoded = decode_inputs(Path(shot.clip_path), lower)
        old = cached[cached.shot_id.eq(shot.shot_id)]
        original = old[old.nominal_ms.eq(0)]
        if len(original) != 1:
            raise ValueError("Cached t0 must exist exactly once per shot")
        coarse = choose_coarse(old, lower)
        for variant in VARIANTS:
            row = {"review_id": shot.review_id, "shot_id": shot.shot_id, "role": shot.role,
                   "sampling_group": shot.sampling_group, "variant": variant, "source": annotation.source,
                   "contact_quality": "ai_provisional" if usable else "unresolved",
                   "contact_lower_ms": lower, "contact_upper_ms": float(annotation.contact_upper_ms) if usable else None,
                   "image": "", "time_ms": None, "source_frame_index": None, "image_sha256": None,
                   "image_origin": "", "gap_ms": None, "availability": "unknown_contact" if lower is None else "no_eligible_frame"}
            old_row = original.iloc[0] if variant == "original_t0" else coarse if variant == "coarse_precontact" else None
            if old_row is not None:
                row.update(image=str(Path(old_row.frame_path).resolve()), time_ms=old_row.measured_ms,
                           image_sha256=old_row.frame_sha256, image_origin="legacy_cache", availability="available")
            elif variant in decoded:
                image, time_ms, frame_index = decoded[variant]
                target = staging / "frames" / f"{shot.review_id}_{variant}.png"
                target.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(target), image):
                    raise OSError(target)
                row.update(image=str(target.relative_to(staging)), time_ms=time_ms, source_frame_index=frame_index,
                           image_sha256=digest(target), image_origin="original_clip", availability="available")
            if lower is not None and pd.notna(row["time_ms"]):
                row["gap_ms"] = lower - row["time_ms"]
            rows.append(row)
    return pd.DataFrame(rows)


def loss(goal, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(goal, float)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def comparisons(predictions, sample):
    wide = predictions.pivot(index="shot_id", columns="variant", values="p")
    pairs, details = [], []
    for first, second in PAIRS:
        joined = sample[["shot_id", "review_id", "role", "sampling_group", "goal"]].merge(
            wide[[first, second]], on="shot_id", validate="one_to_one")
        joined["first"], joined["second"] = first, second
        joined["p_first"], joined["p_second"] = joined[first], joined[second]
        joined["delta_p"] = joined[first] - joined[second]
        joined["delta_loss_original_label"] = loss(joined.goal, joined[first]) - loss(joined.goal, joined[second])
        joined["paired"] = joined[[first, second]].notna().all(axis=1)
        details.append(joined.drop(columns=[first, second]))
        for (role, group), rows in joined.groupby(["role", "sampling_group"]):
            valid = rows[rows.paired]
            pairs.append({"first": first, "second": second, "role": role, "sampling_group": group,
                          "total_shots": len(rows), "paired_shots": len(valid), "unpaired_shots": int((~rows.paired).sum()),
                          "source_goals_paired": int(valid.goal.sum()),
                          "mean_delta_p": valid.delta_p.mean(), "mean_absolute_delta_p": valid.delta_p.abs().mean(),
                          "mean_delta_loss_original_label": valid.delta_loss_original_label.mean()})
    return pd.DataFrame(pairs), pd.concat(details, ignore_index=True)


def review_sheet(details, inputs, staging):
    ranking = details[details["first"].eq("dense_precontact") & details["second"].eq("clip_t0") & details.paired].copy()
    ranking["magnitude"] = ranking.delta_p.abs()
    tiles = []
    chosen = ranking.sort_values("magnitude", ascending=False).head(5)
    for item in chosen.itertuples():
        for variant in ("original_t0", "coarse_precontact", "clip_t0", "dense_precontact"):
            row = inputs[inputs.shot_id.eq(item.shot_id) & inputs.variant.eq(variant)].iloc[0]
            tile = np.zeros((240, 384, 3), np.uint8)
            if row.availability == "available":
                path = Path(row.image) if row.image_origin == "legacy_cache" else staging / row.image
                original = cv2.imread(str(path))
                if original is None:
                    raise ValueError(f"Cannot render comparison image {path}")
                tile[24:] = cv2.resize(original, (384, 216))
            caption = f"{item.review_id} {variant} {row.time_ms:g}ms" if pd.notna(row.time_ms) else f"{item.review_id} {variant} missing"
            cv2.putText(tile, caption, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1)
            tiles.append(tile)
    if tiles:
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        if not cv2.imwrite(str(staging / "largest_changes.jpg"), sheet):
            raise OSError("Cannot write comparison sheet")
    chosen.to_csv(staging / "largest_changes.csv", index=False)


def run(packet_root, annotations_path, baseline, frozen_path, out, device):
    packet_manifest, packet = verify_packet(packet_root)
    payload = json.loads(annotations_path.read_text())
    if payload.get("source") != "model_review":
        raise ValueError("This diagnostic is explicitly for provisional AI annotations")
    annotations = validate_annotations(payload, packet, allow_model_review=True).set_index("review_id")
    if payload["run"].get("public_packet_sha256") != digest(packet_root / "public/packet.json"):
        raise ValueError("AI review packet checksum changed")
    sample = pd.read_csv(packet_root / "sample.csv").set_index("review_id").loc[annotations.index].reset_index()
    if not sample.role.isin(["fit", "select"]).all():
        raise ValueError("Only development fit/select shots may enter this diagnostic")
    cached_manifest = json.loads((baseline / "manifest.json").read_text())
    if cached_manifest["identity"]["packet_id"] != packet["packet_id"] or \
            cached_manifest["identity"]["frozen_sha256"] != digest(frozen_path) or \
            digest(baseline / "predictions.csv") != cached_manifest["files"]["predictions.csv"]:
        raise ValueError("Baseline prediction provenance differs")
    cached = pd.read_csv(baseline / "predictions.csv")
    cached = cached[cached.shot_id.isin(sample.shot_id)]
    if cached.duplicated(["shot_id", "nominal_ms"]).any():
        raise ValueError("Duplicate cached frames")
    sources = [annotations_path, packet_root / "manifest.json", baseline / "manifest.json", frozen_path,
               Path(__file__), Path("scripts/diagnostic_review.py"), Path("xgpoc/data/datasets.py"),
               Path("xgpoc/models/d_foundation.py"), Path("xgpoc/models/common.py"),
               Path("xgpoc/eval/calibration.py"), Path("configs/overlay_mask.png")]
    frozen = json.loads(frozen_path.read_text())["variants"]["g_base"]
    sources.extend(Path(p) for p in frozen["members"])
    for name, checksum in frozen["members"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Frozen model changed: {name}")
    for shot in sample.itertuples():
        clip = Path(shot.clip_path)
        if digest(clip) != packet_manifest["identity"]["clips"][shot.review_id]["sha256"]:
            raise ValueError("Native clip changed since annotation packet generation")
        sources.append(clip)
    for row in cached.itertuples():
        if row.frame_path and digest(Path(row.frame_path)) != row.frame_sha256:
            raise ValueError("Legacy cached frame changed")
    identity = {"source": "model_review", "packet_id": packet["packet_id"], "variants": list(VARIANTS),
                "pairs": [list(p) for p in PAIRS], "source_sha256": {str(p): digest(p) for p in sources},
                "rule": "latest decoded frame with anchor-relative time <= 0 and strictly below AI lower bound",
                "sample_ids": sample.shot_id.tolist(), "torch_version": str(torch.__version__), "device": device}
    inspected = verify_evidence(payload, packet, packet_root, annotations_path.parent)
    if out.exists():
        complete = json.loads((out / "complete.json").read_text())
        if complete["identity"] != identity:
            raise ValueError("Diagnostic inputs changed; use a new result directory")
        for name, checksum in complete["files"].items():
            if digest(out / name) != checksum:
                raise ValueError(f"Diagnostic output changed: {name}")
        print("Verified existing AI-contact comparison; no inference or extraction repeated")
        return
    staging = out.with_name(out.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    inputs = build_inputs(sample, annotations, cached, staging)
    present = inputs[inputs.availability.eq("available")].copy()
    present["frame_path"] = present.apply(lambda r: r.image if r.image_origin == "legacy_cache" else str(staging / r.image), axis=1)
    dataset = ImageDataset(present.assign(goal=False), size_wh=(448, 252), mask="overlay")
    all_logits = []
    for name in frozen["members"]:
        print(f"Frozen G: {len(present)} inputs with {name}", flush=True)
        bundle = torch.load(name, map_location="cpu", weights_only=False)
        if bundle["spec"]["variant"] != "g_base":
            raise ValueError("Expected G baseline bundle")
        model = FinetuneViTXG(**bundle["spec"]["kwargs"], pretrained=False).to(device)
        model.load_state_dict(bundle["state_dict"], strict=True)
        model.eval()
        all_logits.append(_predict_logits(model, DataLoader(dataset, batch_size=8, num_workers=2), device))
        del model, bundle
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    penalties = present.shot_id.map(sample.set_index("shot_id").is_penalty).to_numpy(bool)
    inputs["p"] = np.nan
    inputs.loc[present.index, "p"] = route_penalties(PlattScaler(**frozen["platt"])(np.mean(all_logits, axis=0)),
                                                   penalties, frozen["penalty_rate"])
    reference = cached[cached.nominal_ms.eq(0)].set_index("shot_id").p
    observed = inputs[inputs.variant.eq("original_t0")].set_index("shot_id").p
    replay_error = float((observed - reference.loc[observed.index]).abs().max())
    if not inputs.loc[present.index, "p"].between(0, 1).all() or replay_error > 1e-5:
        raise ValueError(f"Original input replay failed: {replay_error}")
    pair_summary, details = comparisons(inputs, sample)
    pair_summary.to_csv(staging / "paired_summary.csv", index=False)
    details.to_csv(staging / "paired_details.csv", index=False)
    inputs.to_csv(staging / "predictions.csv", index=False)
    review_sheet(details, inputs, staging)
    shutil.copyfile(annotations_path, staging / "annotations.source.json")
    summary = {"source": "model_review", "status": "exploratory_only", "shots": len(sample),
               "source_goals": int(sample.goal.sum()), "roles": sample.role.value_counts().to_dict(),
               "contact_status": annotations.contact_status.value_counts().to_dict(),
               "prior_exposure": int(annotations.prior_annotation_exposure.sum()), "checked_evidence_frames": inspected,
               "available_inputs": present.variant.value_counts().to_dict(),
               "replay_max_probability_error": replay_error,
               "note": "AI intervals are provisional. Keep unknown contacts in coverage. Original goal labels remain unchanged. "
                       "Fit is in-sample, select was used for selection. No model ranking, retraining or calibration fitting."}
    (staging / "summary.json").write_text(json_text(summary))
    for name, checksum in identity["source_sha256"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Source changed during comparison: {name}")
    for row in present.itertuples():
        if digest(Path(row.frame_path)) != row.image_sha256:
            raise ValueError("Input pixels changed during comparison")
    files = {str(p.relative_to(staging)): digest(p) for p in staging.rglob("*") if p.is_file()}
    (staging / "complete.json").write_text(json_text({"identity": identity, "files": files}))
    staging.rename(out)
    print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("cache/v15/g_sensitivity_verified"))
    parser.add_argument("--frozen", type=Path, default=Path("results/v14/frozen.json"))
    parser.add_argument("--out", type=Path, default=Path("cache/v15/ai_contact_comparison"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.packet, args.annotations, args.baseline, args.frozen, args.out, args.device)


if __name__ == "__main__":
    main()
