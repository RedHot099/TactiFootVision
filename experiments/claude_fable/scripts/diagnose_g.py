"""Frozen G timing sensitivity and human-contact diagnostics; never train or recalibrate."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from diagnostic_review import json_text, validate_annotations, verify_packet
from xgpoc.data.datasets import ImageDataset
from xgpoc.eval.calibration import PlattScaler, route_penalties
from xgpoc.models.common import _predict_logits
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.v13.contact import locate_frames_in_clip
from xgpoc.v13.inputs import OFFSETS
from xgpoc.v13.protocol import digest


def measure(row):
    records, paths = [], []
    for offset in OFFSETS:
        path = row.get("frame_path" if offset == 0 else f"frame_path_{offset}")
        exists = isinstance(path, str) and Path(path).is_file()
        records.append({"shot_id": row.shot_id, "nominal_ms": offset, "frame_path": path if exists else "",
                        "measured_ms": None, "match_margin": None, "frame_sha256": digest(Path(path)) if exists else None})
        if exists:
            paths.append(path)
    matches = iter(locate_frames_in_clip(paths, row.clip_path)) if paths else iter(())
    for record in records:
        if record["frame_path"]:
            match = next(matches)
            record.update(measured_ms=match.measured_offset(), match_margin=match.margin)
    return records


def infer(packet_root, frozen_path, out, device):
    manifest, _ = verify_packet(packet_root)
    frozen = json.loads(frozen_path.read_text())["variants"]["g_base"]
    sample = pd.read_csv(packet_root / "sample.csv")
    for row in sample.itertuples():
        if digest(Path(row.clip_path)) != manifest["identity"]["clips"][row.review_id]["sha256"]:
            raise ValueError("Original clip differs from the human-review packet")
    identity = {"packet_id": manifest["packet_id"], "frozen_sha256": digest(frozen_path),
                "source_sha256": {str(p): digest(p) for p in [Path(__file__), Path("xgpoc/data/datasets.py"),
                    Path("xgpoc/models/d_foundation.py"), Path("xgpoc/models/common.py"),
                    Path("xgpoc/v13/contact.py"), Path("xgpoc/eval/calibration.py"),
                    Path("scripts/diagnostic_review.py"), Path("configs/overlay_mask.png")]},
                "offsets": list(OFFSETS), "members": frozen["members"]}
    for name, checksum in frozen["members"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Frozen model changed: {name}")
    with ThreadPoolExecutor(4) as executor:
        records = [r for group in executor.map(measure, (r for _, r in sample.iterrows())) for r in group]
    frames = pd.DataFrame(records)
    identity["frames"] = frames[["shot_id", "nominal_ms", "frame_sha256"]].fillna("").to_dict("records")
    if out.exists():
        saved = json.loads((out / "manifest.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Diagnostic inputs changed; use a new output directory")
        for name, checksum in saved["files"].items():
            if digest(out / name) != checksum:
                raise ValueError(f"Diagnostic output changed: {name}")
        print("Verified existing frozen inference")
        return
    present = frames[frames.frame_path.ne("")].copy().assign(goal=False)
    dataset = ImageDataset(present, size_wh=(448, 252), mask="overlay")
    logits = []
    for name in frozen["members"]:
        print(f"Predicting {len(present)} frames with {name}", flush=True)
        bundle = torch.load(name, map_location="cpu", weights_only=False)
        if bundle["spec"]["variant"] != "g_base":
            raise ValueError("Only the frozen global G baseline is supported")
        model = FinetuneViTXG(**bundle["spec"]["kwargs"], pretrained=False).to(device)
        model.load_state_dict(bundle["state_dict"], strict=True)
        model.eval()
        logits.append(_predict_logits(model, DataLoader(dataset, batch_size=8, num_workers=2), device))
        del model, bundle
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    mean = np.mean(logits, axis=0)
    penalties = present.shot_id.map(sample.set_index("shot_id").is_penalty).to_numpy(bool)
    frames["p"] = np.nan
    frames.loc[present.index, "p"] = route_penalties(PlattScaler(**frozen["platt"])(mean), penalties, frozen["penalty_rate"])
    frames["prediction_status"] = np.where(frames.p.notna(), "predicted_nominal_not_proven_preshot", "missing_frame")
    references = []
    for name in frozen["members"]:
        reference = pd.read_csv(Path(name).parent / "select_predictions.csv")
        references.append(reference[reference.admission.eq("nominal")].set_index("shot_id").logit)
    common = pd.concat(references, axis=1).dropna().mean(axis=1)
    ids = sample[sample.role.eq("select") & sample.shot_id.isin(common.index)].shot_id
    observed = frames[frames.nominal_ms.eq(0)].set_index("shot_id").loc[ids, "p"].to_numpy()
    expected = route_penalties(PlattScaler(**frozen["platt"])(common.loc[ids].to_numpy()),
                               ids.map(sample.set_index("shot_id").is_penalty).to_numpy(bool), frozen["penalty_rate"])
    replay_error = float(np.max(np.abs(observed - expected))) if len(ids) else None
    if replay_error is not None and replay_error > 1e-5:
        raise ValueError(f"Frozen G replay failed on unchanged select inputs: {replay_error}")
    for name, checksum in identity["source_sha256"].items():
        if digest(Path(name)) != checksum:
            raise ValueError(f"Source changed during inference: {name}")
    for row in frames[frames.frame_path.ne("")].itertuples():
        if digest(Path(row.frame_path)) != row.frame_sha256:
            raise ValueError("Input frame changed during inference")
    if digest(frozen_path) != identity["frozen_sha256"]:
        raise ValueError("Frozen ensemble changed during inference")
    out.mkdir(parents=True)
    frames.to_csv(out / "predictions.csv", index=False)
    (out / "manifest.json").write_text(json_text({"identity": identity,
        "replay": {"unchanged_select_shots": len(ids), "max_probability_error": replay_error},
        "files": {"predictions.csv": digest(out / "predictions.csv")}}))
    print(f"Saved {len(frames)} rows, {frames.p.isna().sum()} missing, without opening evaluation labels")


def reconcile(exports, packet):
    if not exports:
        return pd.DataFrame(columns=["review_id", "review_count", "contact_quality", "lower_ms", "upper_ms"])
    tables = [validate_annotations(json.loads(p.read_text()), packet) for p in exports]
    rows = pd.concat(tables, ignore_index=True)
    if rows.duplicated(["reviewer", "review_id"]).any():
        raise ValueError("Repeated reviewer for a shot cannot count as independent review")
    result = []
    for rid, group in rows.groupby("review_id"):
        bounded = group[group.contact_status.eq("bounded")]
        overlap = len(bounded) == len(group) and bounded.contact_lower_ms.max() <= bounded.contact_upper_ms.min()
        quality = "agreed" if overlap and len(group) >= 2 else "single" if overlap else "unresolved"
        row = {"review_id": rid, "review_count": len(group), "contact_quality": quality,
               "lower_ms": float(bounded.contact_lower_ms.min()) if quality == "agreed" else None,
               "upper_ms": float(bounded.contact_upper_ms.max()) if quality == "agreed" else None}
        for field in ("body_part", "situation", "ball_visible", "keeper_visible", "outcome", "goal_attribution"):
            row[field] = group[field].iloc[0] if group[field].nunique() == 1 else "disagreement"
        result.append(row)
    return pd.DataFrame(result)


def safe_rows(frames, contacts):
    merged = frames.merge(contacts, on="shot_id", how="left", validate="many_to_one")
    # A nominal t0 is not a measured pre-contact frame. Never infer timing safety from the filename.
    return merged[merged.contact_quality.eq("agreed") & merged.measured_ms.notna() &
                  merged.measured_ms.lt(merged.lower_ms) & merged.nominal_ms.le(0) & merged.p.notna()].copy()


def binary_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def report(packet_root, predictions_root, exports, out):
    manifest, packet = verify_packet(packet_root)
    inference = json.loads((predictions_root / "manifest.json").read_text())
    if inference["identity"]["packet_id"] != manifest["packet_id"]:
        raise ValueError("Predictions belong to another review packet")
    if digest(predictions_root / "predictions.csv") != inference["files"]["predictions.csv"]:
        raise ValueError("Predictions changed")
    if out.exists():
        raise ValueError("Use a new report directory for each review revision")
    sample = pd.read_csv(packet_root / "sample.csv")
    frames = pd.read_csv(predictions_root / "predictions.csv")
    contacts = reconcile(exports, packet).merge(sample[["review_id", "shot_id"]], how="right", validate="one_to_one")
    contacts["contact_quality"] = contacts.contact_quality.fillna("unreviewed")
    joined = frames.merge(sample[["shot_id", "sampling_group", "role", "goal", "is_penalty"]], validate="many_to_one")
    joined["loss"] = binary_loss(joined.goal, joined.p)
    baseline = joined[joined.nominal_ms.eq(0)][["shot_id", "p", "loss"]].rename(columns={"p": "p_t0", "loss": "loss_t0"})
    joined = joined.merge(baseline, on="shot_id", validate="many_to_one")
    summaries = []
    for (group, role, offset), rows in joined.groupby(["sampling_group", "role", "nominal_ms"]):
        comparable = rows.dropna(subset=["p", "p_t0"])
        summaries.append({"sampling_group": group, "role": role, "nominal_ms": offset, "shots": len(rows),
                          "paired_shots": len(comparable), "mean_abs_probability_change": float((comparable.p - comparable.p_t0).abs().mean()),
                          "mean_loss_change": float((comparable.loss - comparable.loss_t0).mean())})
    safe = safe_rows(frames, contacts)
    chosen = safe.sort_values("measured_ms").groupby("shot_id", as_index=False).tail(1)
    coverage = contacts.merge(sample[["shot_id", "sampling_group", "role"]], on="shot_id", validate="one_to_one")
    agreement = contacts[contacts.contact_quality.eq("agreed")].merge(baseline, on="shot_id")
    measured_t0 = frames[frames.nominal_ms.eq(0)][["shot_id", "measured_ms"]]
    agreement = agreement.merge(measured_t0, on="shot_id")
    after = agreement.measured_ms.notna() & agreement.measured_ms.gt(agreement.upper_ms)
    agreement["t0_definitely_after_contact"] = after
    groups = []
    for (sampling_group, role), subset in coverage.groupby(["sampling_group", "role"]):
        ids = set(subset.shot_id)
        audited = agreement[agreement.shot_id.isin(ids)]
        groups.append({"sampling_group": sampling_group, "role": role, "sampled": len(subset),
                       "agreed_contacts": len(audited), "measured_t0": int(audited.measured_ms.notna().sum()),
                       "t0_definitely_after_contact": int(audited.t0_definitely_after_contact.sum()),
                       "safe_inputs": int(chosen.shot_id.isin(ids).sum())})
    paired = chosen.merge(sample[["shot_id", "sampling_group", "role", "goal"]], on="shot_id", validate="one_to_one")
    paired = paired.merge(baseline, on="shot_id", validate="one_to_one")
    if len(paired):
        paired["loss_safe_original_label"] = binary_loss(paired.goal, paired.p)
        paired["loss_t0_original_label"] = binary_loss(paired.goal, paired.p_t0)
        paired["verified_goal"] = paired.outcome.map({"goal": 1, "no_goal": 0})
        paired["loss_safe_human_label"] = binary_loss(paired.verified_goal, paired.p)
        paired["loss_t0_human_label"] = binary_loss(paired.verified_goal, paired.p_t0)
    strata = []
    if len(paired):
        for field in ("body_part", "situation", "ball_visible", "keeper_visible"):
            for (sampling_group, role, category), rows in paired.groupby(["sampling_group", "role", field]):
                strata.append({"sampling_group": sampling_group, "role": role, "field": field, "category": category,
                    "shots": len(rows), "verified_outcomes": int(rows.verified_goal.notna().sum()),
                    "original_label_mean_loss_change": float((rows.loss_safe_original_label - rows.loss_t0_original_label).mean()),
                    "human_label_mean_loss_change": (rows.loss_safe_human_label - rows.loss_t0_human_label).mean()})
    out.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(out / "nominal_sensitivity.csv", index=False)
    coverage.to_csv(out / "human_coverage.csv", index=False)
    chosen.to_csv(out / "human_safe_frames.csv", index=False)
    agreement.to_csv(out / "contact_comparison.csv", index=False)
    pd.DataFrame(groups).to_csv(out / "coverage_by_sample.csv", index=False)
    paired.to_csv(out / "paired_safe_comparison.csv", index=False)
    pd.DataFrame(strata, columns=["sampling_group", "role", "field", "category", "shots", "verified_outcomes",
                                 "original_label_mean_loss_change", "human_label_mean_loss_change"]).to_csv(
        out / "situation_comparison.csv", index=False)
    summary = {"shots": len(sample), "human_contact_status": contacts.contact_quality.value_counts().to_dict(),
               "agreed_contact_with_measured_t0": int(agreement.measured_ms.notna().sum()),
               "t0_definitely_after_contact": int(after.sum()), "shots_with_verified_safe_frame": len(chosen),
               "remaining_without_verified_safe_frame": len(sample) - len(chosen),
               "training_gate": "blocked_pending_human_review" if not len(chosen) else "design_required_no_automatic_training",
               "interpretation": "Exploratory sensitivity only. Fit is in-sample; select was used for epoch selection. "
                   "Nominal frames may be post-contact. Human-reviewed subsets are not population estimates. "
                   "No new generalization or causal claim; no calibration was refitted."}
    (out / "summary.json").write_text(json_text(summary))
    (out / "manifest.json").write_text(json_text({"packet_id": manifest["packet_id"],
        "prediction_manifest_sha256": digest(predictions_root / "manifest.json"), "script_sha256": digest(Path(__file__)),
        "reviews": {str(p): digest(p) for p in exports}}))
    print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inf = commands.add_parser("infer")
    inf.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    inf.add_argument("--frozen", type=Path, default=Path("results/v14/frozen.json"))
    inf.add_argument("--out", type=Path, default=Path("cache/v15/g_sensitivity"))
    inf.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    rep = commands.add_parser("report")
    rep.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    rep.add_argument("--predictions", type=Path, default=Path("cache/v15/g_sensitivity"))
    rep.add_argument("--review-exports", type=Path, nargs="*", default=[])
    rep.add_argument("--out", type=Path, default=Path("research/v15_diagnostics/before_human_review"))
    args = parser.parse_args()
    if args.command == "infer":
        infer(args.packet, args.frozen, args.out, args.device)
    else:
        report(args.packet, args.predictions, args.review_exports, args.out)


if __name__ == "__main__":
    main()
