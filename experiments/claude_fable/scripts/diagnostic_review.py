"""Prepare a blind review packet and import human or explicitly marked AI annotations."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from xgpoc.paths import INDEX
from xgpoc.v13.protocol import assign_roles, digest

HERE = Path(__file__).parent
WINDOW = (-1600, 1200)
CHOICES = {
    "contact_status": {"bounded", "uncertain", "out_of_window"},
    "body_part": {"foot", "head", "other", "unknown"},
    "situation": {"ordinary", "rebound", "first_time", "unknown"},
    "ball_visible": {"yes", "no", "unknown"},
    "keeper_visible": {"yes", "no", "unknown"},
    "outcome": {"goal", "no_goal", "unknown"},
    "goal_attribution": {"this_attempt", "other_attempt", "no_goal", "unknown"},
}


def json_text(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def select_sample(pool, review, random_n=200, targeted_n=100, seed=1500):
    eligible = pool[pool.role.isin(["fit", "select"])].sort_values("shot_id")
    if eligible.shot_id.duplicated().any() or review.shot_id.duplicated().any():
        raise ValueError("Duplicate shot identifiers")
    if random_n <= 0 or targeted_n < 0 or random_n > len(eligible):
        raise ValueError("Invalid sample sizes")
    random = eligible.sample(random_n, random_state=seed).assign(sampling_group="random", sampling_reason="random")
    remaining = eligible[~eligible.shot_id.isin(random.shot_id)].copy()
    status = review.set_index("shot_id").interval_status
    remaining["sampling_reason"] = remaining.shot_id.map(status).fillna("unreviewed")
    # Oversample unresolved contacts, but include previously confident reviews for falsification.
    parts = []
    for label, count in (("uncertain", (targeted_n + 1) // 2), ("reviewed_known", targeted_n // 2)):
        mask = remaining.sampling_reason.eq("uncertain") if label == "uncertain" else ~remaining.sampling_reason.isin(
            ["uncertain", "unreviewed"])
        candidates = remaining[mask]
        parts.append(candidates.sample(min(count, len(candidates)), random_state=seed).assign(sampling_reason=label))
    targeted = pd.concat(parts).assign(sampling_group="targeted")
    sample = pd.concat([random, targeted]).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["review_id"] = [f"r{i:04d}" for i in range(len(sample))]
    return sample


def render(row, public):
    folder = public / row.review_id
    folder.mkdir()
    video = cv2.VideoCapture(row.clip_path)
    frames = []
    previous = None
    try:
        if not video.isOpened():
            raise ValueError(f"Cannot open clip for {row.review_id}")
        while True:
            ok, frame = video.read()
            if not ok:
                break
            timestamp = round(float(video.get(cv2.CAP_PROP_POS_MSEC)) - 4000, 3)
            if previous is not None and timestamp <= previous:
                raise ValueError(f"Non-increasing decoded timestamps for {row.review_id}")
            previous = timestamp
            if timestamp > WINDOW[1]:
                break
            if timestamp < WINDOW[0]:
                continue
            path = folder / f"{len(frames):03d}.jpg"
            if not cv2.imwrite(str(path), cv2.resize(frame, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError(path)
            frames.append({"time_ms": timestamp, "image": f"{row.review_id}/{path.name}"})
    finally:
        video.release()
    if len(frames) < 2:
        raise ValueError(f"Not enough decoded frames for {row.review_id}")
    shutil.copyfile(row.clip_path, folder / "clip.mp4")
    return {"review_id": row.review_id, "frames": frames, "clip": f"{row.review_id}/clip.mp4"}


def verify_packet(root):
    manifest = json.loads((root / "manifest.json").read_text())
    identity = manifest["identity"]
    if hashlib.sha256(json_text(identity).encode()).hexdigest() != manifest["packet_id"]:
        raise ValueError("Packet identity changed")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Packet file changed: {name}")
    packet = json.loads((root / "public/packet.json").read_text())
    if packet["packet_id"] != manifest["packet_id"]:
        raise ValueError("Public packet identity differs")
    return manifest, packet


def prepare(index, review_path, out, random_n, targeted_n):
    pool = assign_roles(pd.read_parquet(index))
    sample = select_sample(pool, pd.read_csv(review_path), random_n, targeted_n)
    identity = {"version": 1, "index_sha256": digest(index), "review_sha256": digest(review_path),
                "seed": 1500, "random_n": random_n, "targeted_n": targeted_n, "window_ms": list(WINDOW),
                "script_sha256": digest(Path(__file__)), "html_sha256": digest(HERE / "review.html"),
                "clips": {r.review_id: {"shot_id": r.shot_id, "sha256": digest(Path(r.clip_path))}
                          for r in sample.itertuples()}}
    packet_id = hashlib.sha256(json_text(identity).encode()).hexdigest()
    if out.exists():
        manifest, _ = verify_packet(out)
        if manifest["identity"] != identity:
            raise ValueError("Existing packet differs; use a new directory")
        print(f"Verified existing packet: {len(sample)} shots; {packet_id}")
        return
    staging = out.with_name(out.name + ".building")
    if staging.exists():
        raise ValueError(f"Incomplete build at {staging}; inspect it and use a new output directory")
    public = staging / "public"
    public.mkdir(parents=True)
    with ThreadPoolExecutor(4) as executor:
        items = list(executor.map(lambda row: render(row, public), sample.itertuples()))
    for row in sample.itertuples():
        if digest(public / row.review_id / "clip.mp4") != identity["clips"][row.review_id]["sha256"]:
            raise ValueError("Clip changed while building packet")
    packet = {"packet_id": packet_id, "items": items}
    (public / "packet.json").write_text(json_text(packet))
    shutil.copyfile(HERE / "review.html", public / "index.html")
    sample.to_csv(staging / "sample.csv", index=False)
    summary = {"count": len(sample), "groups": sample.sampling_group.value_counts().to_dict(),
               "reasons": sample.sampling_reason.value_counts().to_dict(),
               "roles": sample.role.value_counts().to_dict(), "games": sample.game.nunique(),
               "human_annotations": 0, "warning": "AUDIT ONLY: public frames include post-shot information. "
                    "Serve only public/, never the packet root. Random and targeted samples are distinct."}
    (staging / "summary.json").write_text(json_text(summary))
    files = {str(p.relative_to(staging)): digest(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    (staging / "manifest.json").write_text(json_text({"packet_id": packet_id, "identity": identity, "files": files}))
    staging.rename(out)
    print(json_text(summary))


def validate_annotations(payload, packet, *, allow_model_review=False):
    sources = {"human", "model_review"} if allow_model_review else {"human"}
    source = payload.get("source")
    if payload.get("packet_id") != packet["packet_id"] or source not in sources:
        raise ValueError("Wrong packet or unsupported annotation source")
    if source == "model_review" and not isinstance(payload.get("run", {}).get("prior_annotation_exposure"), bool):
        raise ValueError("AI review must declare prior annotation exposure")
    reviewer = payload.get("reviewer", "")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 80:
        raise ValueError("Use a nonempty reviewer alias (maximum 80 characters)")
    items = {item["review_id"]: item for item in packet["items"]}
    seen, output = set(), []
    for row in payload.get("annotations", []):
        rid = row.get("review_id")
        if rid not in items or rid in seen:
            raise ValueError("Unknown or duplicate review_id")
        seen.add(rid)
        if source == "model_review" and not isinstance(row.get("prior_annotation_exposure"), bool):
            raise ValueError("AI annotation must declare prior annotation exposure")
        for key, options in CHOICES.items():
            if row.get(key) not in options:
                raise ValueError(f"Invalid {key} for {rid}")
        times = {f["time_ms"] for f in items[rid]["frames"]}
        lower, upper = row.get("contact_lower_ms"), row.get("contact_upper_ms")
        if row["contact_status"] == "bounded":
            if isinstance(lower, bool) or isinstance(upper, bool) or lower not in times or upper not in times or lower > upper:
                raise ValueError(f"Contact bounds must be ordered decoded timestamps for {rid}")
        elif lower is not None or upper is not None:
            raise ValueError("Unknown contact must not have bounds")
        shooter = [row.get(k) for k in ("shooter_x", "shooter_y", "shooter_frame_ms")]
        if any(x is not None for x in shooter):
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in shooter) \
                    or not all(0 <= x <= 1 for x in shooter[:2]) or shooter[2] not in times:
                raise ValueError("Shooter point must have normalized coordinates and a decoded timestamp")
        attribution = row["goal_attribution"]
        if attribution == "this_attempt" and row["outcome"] != "goal" \
                or attribution in {"other_attempt", "no_goal"} and row["outcome"] != "no_goal":
            raise ValueError("Outcome and goal attribution disagree")
        output.append({k: row.get(k) for k in ["review_id", *CHOICES, "contact_lower_ms", "contact_upper_ms",
                                              "shooter_x", "shooter_y", "shooter_frame_ms", "note"]} |
                      {"reviewer": reviewer.strip(), "source": source, "packet_id": packet["packet_id"],
                       "prior_annotation_exposure": bool(row.get("prior_annotation_exposure", False) or
                                                         payload.get("run", {}).get("prior_annotation_exposure", False))})
    if not output:
        raise ValueError("No completed annotations; do not fabricate them")
    return pd.DataFrame(output)


def import_annotations(root, export, out):
    _, packet = verify_packet(root)
    payload = json.loads(export.read_text())
    rows = validate_annotations(payload, packet, allow_model_review=True)
    if payload["source"] == "model_review" and payload["run"].get("public_packet_sha256") != digest(root / "public/packet.json"):
        raise ValueError("AI review public packet checksum differs")
    sample = pd.read_csv(root / "sample.csv")
    result = rows.merge(sample[["review_id", "shot_id", "role", "sampling_group"]], validate="one_to_one")
    text = result.to_csv(index=False)
    if out.exists() and out.read_text() != text:
        raise ValueError("Import destination already contains different annotations")
    provenance = out.with_suffix(".source.json")
    if provenance.exists() and provenance.read_bytes() != export.read_bytes():
        raise ValueError("Import destination has different source evidence")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not provenance.exists():
        provenance.write_bytes(export.read_bytes())
    if not out.exists():
        out.write_text(text)
    print(f"Validated {len(result)} {payload['source']} annotations; source evidence saved to {provenance}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--index", type=Path, default=INDEX)
    prep.add_argument("--review", type=Path, default=Path("cache/v13/contact_review/combined_review.csv"))
    prep.add_argument("--out", type=Path, default=Path("cache/v15/human_review"))
    prep.add_argument("--random-n", type=int, default=200)
    prep.add_argument("--targeted-n", type=int, default=100)
    imp = commands.add_parser("import")
    imp.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    imp.add_argument("--export", type=Path, required=True)
    imp.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.index, args.review, args.out, args.random_n, args.targeted_n)
    elif args.command == "import":
        import_annotations(args.packet, args.export, args.out)
    else:
        _, packet = verify_packet(args.packet)
        print(f"Verified {len(packet['items'])} packet items without regeneration")


if __name__ == "__main__":
    main()
