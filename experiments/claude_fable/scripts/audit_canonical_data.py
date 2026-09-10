"""Isolate decoder, JPEG and timing effects on canonical inputs with frozen G."""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from diagnostic_review import json_text
from prepare_canonical_data import complete, verify_result
from xgpoc.data.canonical_frames import (
    array_digest, decode_ffmpeg_frame, encode_jpeg_control, model_input_digest, runtime_identity, write_png,
)
from xgpoc.data.datasets import ImageDataset
from xgpoc.eval.calibration import PlattScaler, route_penalties
from xgpoc.models.common import _predict_logits
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.v13.protocol import digest

PAIRS = (("decoder_t0", "canonical_t0"), ("jpeg_t0", "canonical_t0"),
         ("canonical_t0", "legacy_t0"), ("canonical_precontact", "canonical_t0"))
# Same image, different batch neighbours: frozen G logits move by up to ~5e-4 (cuDNN TF32 kernels chosen per batch
# shape), i.e. ~1e-4 in probability. Effects of interest are ~1e-2. Measured 2026-09-10 on the 20 legacy t0 inputs.
NOISE_BOUND = 1e-3


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in details.groupby(["first", "second", "role", "sampling_group"]):
        valid = group[group.paired]
        rows.append(dict(zip(["first", "second", "role", "sampling_group"], keys)) |
                    {"total_shots": len(group), "paired_shots": len(valid), "unpaired_shots": int((~group.paired).sum()),
                     "mean_delta_p": valid.delta_p.mean(), "mean_absolute_delta_p": valid.delta_p.abs().mean(),
                     "mean_native_pixel_mae": valid.native_pixel_mae.mean(),
                     "mean_tensor_mae": valid.tensor_mae.mean()})
    return pd.DataFrame(rows)


def controls(data_root: Path, frames: pd.DataFrame, shots: pd.DataFrame, legacy: pd.DataFrame, stage: Path) -> pd.DataFrame:
    rows = []
    for row in frames.itertuples():
        path = str((data_root / row.frame_path).resolve()) if row.availability == "available" else ""
        rows.append({"shot_id": row.shot_id, "variant": f"canonical_{row.variant}", "frame_path": path,
                     "file_sha256": row.file_sha256, "tensor_sha256": row.tensor_sha256, "availability": row.availability})
    folder = stage / "controls"
    folder.mkdir()
    for shot in shots.itertuples():
        t0 = frames[frames.shot_id.eq(shot.shot_id) & frames.variant.eq("t0")].iloc[0]
        decoded = decode_ffmpeg_frame(Path(shot.clip_path), int(t0.source_frame_index), int(t0.width), int(t0.height))
        ffmpeg_png = folder / f"{shot.review_id}_ffmpeg.png"
        jpeg = folder / f"{shot.review_id}_q3.jpg"
        write_png(ffmpeg_png, decoded)
        encode_jpeg_control(data_root / t0.frame_path, jpeg)
        original = legacy[legacy.shot_id.eq(shot.shot_id) & legacy.variant.eq("original_t0")].iloc[0]
        original_path = Path(original.image)
        if digest(original_path) != original.image_sha256:
            raise ValueError("Historical JPEG changed")
        for variant, path in (("decoder_t0", ffmpeg_png), ("jpeg_t0", jpeg), ("legacy_t0", original_path)):
            rows.append({"shot_id": shot.shot_id, "variant": variant,
                         "frame_path": str(path.relative_to(stage)) if variant != "legacy_t0" else str(path.resolve()),
                         "file_sha256": digest(path), "tensor_sha256": model_input_digest(path), "availability": "available"})
    return pd.DataFrame(rows)


def paired_details(inputs: pd.DataFrame, shots: pd.DataFrame, stage: Path) -> pd.DataFrame:
    indexed = inputs.set_index(["shot_id", "variant"])
    rows = []
    for first, second in PAIRS:
        for shot in shots.itertuples():
            a, b = indexed.loc[(shot.shot_id, first)], indexed.loc[(shot.shot_id, second)]
            paired = a.availability == b.availability == "available"
            row = {"review_id": shot.review_id, "shot_id": shot.shot_id, "role": shot.role, "sampling_group": shot.sampling_group,
                   "source_goal": shot.source_goal, "first": first, "second": second, "paired": paired,
                   "p_first": a.p, "p_second": b.p, "delta_p": a.p - b.p,
                   "native_pixel_mae": np.nan, "tensor_mae": np.nan, "same_pixels": False, "same_tensor": False}
            if paired:
                pa, pb = stage / a.frame_path, stage / b.frame_path
                ia, ib = cv2.imread(str(pa)), cv2.imread(str(pb))
                if ia.shape == ib.shape:
                    row["native_pixel_mae"] = float(np.abs(ia.astype(float) - ib).mean())
                    row["same_pixels"] = bool(np.array_equal(ia, ib))
                dataset = ImageDataset(pd.DataFrame({"frame_path": [str(pa), str(pb)], "goal": False}),
                                       size_wh=(448, 252), mask="overlay")
                ta, tb = dataset[0]["image"].numpy(), dataset[1]["image"].numpy()
                row.update(tensor_mae=float(np.abs(ta - tb).mean()), same_tensor=bool(np.array_equal(ta, tb)))
            rows.append(row)
    return pd.DataFrame(rows)


def run(data_root: Path, contact_root: Path, frozen_path: Path, out: Path, device: str):
    data_manifest = verify_result(data_root)
    contact_manifest = json.loads((contact_root / "complete.json").read_text())
    for name, checksum in contact_manifest["files"].items():
        if digest(contact_root / name) != checksum:
            raise ValueError(f"Previous contact result changed: {name}")
    if contact_manifest["identity"]["packet_id"] != data_manifest["identity"]["packet_id"]:
        raise ValueError("Different annotation packets")
    shots = pd.read_csv(data_root / "shots.csv")
    frames = pd.read_csv(data_root / "frames.csv").fillna({"frame_path": ""})
    if not shots.extraction_status.eq("ok").all():
        raise ValueError("Resolve reported extraction failures before the controlled inference audit")
    legacy = pd.read_csv(contact_root / "predictions.csv")
    if set(shots.shot_id) != set(legacy.shot_id):
        raise ValueError("The replay reference must cover exactly the reviewed shots")
    frozen = json.loads(frozen_path.read_text())["variants"]["g_base"]
    sources = [data_root / "complete.json", contact_root / "complete.json", contact_root / "predictions.csv", frozen_path,
               Path(__file__), Path("scripts/prepare_canonical_data.py"), Path("xgpoc/data/canonical_frames.py"),
               Path("xgpoc/data/datasets.py"), Path("xgpoc/models/d_foundation.py"), Path("xgpoc/models/common.py"),
               Path("xgpoc/eval/calibration.py"), Path("xgpoc/v13/protocol.py"), Path("configs/overlay_mask.png")]
    sources.extend(data_root / r.frame_path for r in frames[frames.availability.eq("available")].itertuples())
    sources.extend(Path(r.image) for r in legacy[legacy.variant.eq("original_t0")].itertuples())
    for name, checksum in frozen["members"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Frozen model changed: {name}")
        sources.append(Path(name))
    identity = {"data_version": data_manifest["identity"]["version"], "device": device,
                "runtime": runtime_identity() | {"torch": str(torch.__version__)},
                "pairs": [list(p) for p in PAIRS], "jpeg_control": "canonical PNG -> FFmpeg MJPEG q=3 yuvj420p",
                "sources": {str(p.resolve()): digest(p) for p in sources}}
    if out.exists():
        saved = verify_result(out)
        if saved["identity"] != identity:
            raise ValueError("Audit inputs or runtime changed; use a new result directory")
        print("Verified existing extraction audit; no inference repeated")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as temporary:
        stage = Path(temporary)
        inputs = controls(data_root, frames, shots, legacy, stage)
        present = inputs[inputs.availability.eq("available")].copy()
        present.frame_path = present.frame_path.map(lambda p: str(stage / p))
        dataset = ImageDataset(present.assign(goal=False), size_wh=(448, 252), mask="overlay")
        for i, row in enumerate(present.itertuples()):
            if array_digest(dataset[i]["image"].numpy()) != row.tensor_sha256:
                raise ValueError("Model tensor differs from canonical manifest")
        # Replaying the previous comparison in its own input order proves model and pipeline are unchanged.
        # The audit batches differ, and cuDNN picks different kernels per batch shape, so the inputs shared
        # with that comparison measure the batch-composition noise floor rather than a replay.
        reference = legacy[legacy.availability.eq("available")].copy()
        reference["frame_path"] = [str(Path(r.image) if r.image_origin == "legacy_cache" else (contact_root / r.image).resolve())
                                   for r in reference.itertuples()]
        reference_set = ImageDataset(reference.assign(goal=False), size_wh=(448, 252), mask="overlay")
        logits, reference_logits = [], []
        for name in frozen["members"]:
            print(f"Frozen G: {len(present)} extraction-control inputs + {len(reference)} replay inputs, {name}", flush=True)
            bundle = torch.load(name, map_location="cpu", weights_only=False)
            if bundle["spec"]["variant"] != "g_base":
                raise ValueError("Expected frozen G baseline")
            model = FinetuneViTXG(**bundle["spec"]["kwargs"], pretrained=False).to(device)
            model.load_state_dict(bundle["state_dict"], strict=True)
            model.eval()
            logits.append(_predict_logits(model, DataLoader(dataset, batch_size=8, num_workers=2), device))
            reference_logits.append(_predict_logits(model, DataLoader(reference_set, batch_size=8, num_workers=2), device))
            del bundle, model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        is_penalty = shots.set_index("shot_id").is_penalty

        def probabilities(rows, member_logits):
            return route_penalties(PlattScaler(**frozen["platt"])(np.mean(member_logits, axis=0)),
                                   rows.shot_id.map(is_penalty).to_numpy(bool), frozen["penalty_rate"])

        inputs["p"] = np.nan
        inputs.loc[present.index, "p"] = probabilities(present, logits)
        exact = float((probabilities(reference, reference_logits) - reference.p).abs().max())
        if exact > 1e-5:
            raise ValueError(f"Frozen replay in the original batch composition differs: {exact}")
        replay = {"identical_composition": {"inputs": len(reference), "max_probability_error": exact},
                  "batch_composition_noise": {"bound": NOISE_BOUND}}
        for current, previous in (("legacy_t0", "original_t0"), ("canonical_t0", "clip_t0"),
                                  ("canonical_precontact", "dense_precontact")):
            now = inputs[inputs.variant.eq(current)].set_index("shot_id").p
            before = legacy[legacy.variant.eq(previous)].set_index("shot_id").p.loc[now.index]
            if not now.isna().equals(before.isna()):
                raise ValueError("Canonical replay availability differs")
            error = float((now - before).abs().max()) if now.notna().any() else None
            replay["batch_composition_noise"][current] = {"shots": int(now.notna().sum()), "max_probability_difference": error}
            if error is not None and error > NOISE_BOUND:
                raise ValueError(f"Shared inputs differ beyond batch-composition noise: {current}, {error}")
        if not inputs.loc[present.index, "p"].between(0, 1).all():
            raise ValueError("Invalid model probabilities")
        details = paired_details(inputs, shots, stage)
        inputs.to_csv(stage / "predictions.csv", index=False)
        details.to_csv(stage / "paired_details.csv", index=False)
        summarize(details).to_csv(stage / "paired_summary.csv", index=False)
        aggregate = details[details.paired].groupby(["first", "second"]).agg(
            shots=("shot_id", "size"), mean_delta_p=("delta_p", "mean"),
            mean_absolute_delta_p=("delta_p", lambda x: x.abs().mean()),
            mean_native_pixel_mae=("native_pixel_mae", "mean"), mean_tensor_mae=("tensor_mae", "mean"),
            identical_pixels=("same_pixels", "sum"), identical_tensors=("same_tensor", "sum")).reset_index()
        aggregate.to_csv(stage / "aggregate.csv", index=False)
        summary = {"status": "exploratory_only", "source_quality": "lossy_clip", "shots": len(shots),
                   "source_goals": int(shots.source_goal.sum()), "model_inputs": len(present), "replay": replay,
                   "note": "Controls isolate decoder and additional JPEG encoding on the same clip frame. "
                           "Historical JPEG vs clip also includes earlier clip transcoding and potentially alignment differences. "
                           "Missing original halves prevent attribution of that full difference to CRF 26. No training or calibration fitting."}
        (stage / "summary.json").write_text(json_text(summary))
        for row in present.itertuples():
            if digest(Path(row.frame_path)) != row.file_sha256:
                raise ValueError("Input image changed during audit")
        complete(stage, out, identity)
        print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cache/v15/canonical_data_v1"))
    parser.add_argument("--contact-reference", type=Path, default=Path("cache/v15/ai_contact_comparison"))
    parser.add_argument("--frozen", type=Path, default=Path("results/v14/frozen.json"))
    parser.add_argument("--out", type=Path, default=Path("cache/v15/canonical_extraction_audit_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.data, args.contact_reference, args.frozen, args.out, args.device)


if __name__ == "__main__":
    main()
