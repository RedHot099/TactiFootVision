"""Sample training shots for visual contact review; future frames are audit-only."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from xgpoc.paths import INDEX
from xgpoc.v13.protocol import digest


def sheet(row, out, step_ms=200):
    if step_ms <= 0 or 1600 % step_ms:
        raise ValueError("step_ms must be a positive divisor of 1600")
    target = out / f"{row.review_id:03d}.jpg"
    receipt = target.with_suffix(".json")
    identity = {"shot_id": row.shot_id, "clip_sha256": digest(Path(row.clip_path)),
                "step_ms": step_ms, "anchor_ms": 4000, "source_sha256": digest(Path(__file__))}
    if target.exists():
        if not receipt.exists():
            raise ValueError(f"Unverified existing sheet: {target}; use a new output directory")
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity or saved["image_sha256"] != digest(target):
            raise ValueError(f"Contact sheet identity changed: {target}")
        return str(target)
    video = cv2.VideoCapture(row.clip_path)
    tiles = []
    for offset in range(-800, 801, step_ms):
        video.set(cv2.CAP_PROP_POS_MSEC, 4000 + offset)
        ok, frame = video.read()
        if not ok:
            video.release()
            raise ValueError(f"Cannot decode {row.shot_id} at {offset}")
        frame = cv2.resize(frame, (480, 270))
        cv2.rectangle(frame, (0, 0), (480, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"Review {row.review_id:03d} | {offset:+d}ms | AUDIT ONLY", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(frame)
    video.release()
    columns = 3 if step_ms == 200 else 5
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    image = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    temporary = target.with_suffix(".tmp.jpg")
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(target)
    if digest(Path(row.clip_path)) != identity["clip_sha256"]:
        temporary.unlink()
        raise ValueError("Source clip changed during rendering")
    receipt.write_text(json.dumps({"identity": identity, "image_sha256": digest(temporary)}, indent=2) + "\n")
    temporary.replace(target)
    return str(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("cache/v13/contact_review"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--uncertain-review", type=Path, help="Existing combined review; select uncertain rows only")
    ap.add_argument("--step-ms", type=int, default=200)
    a = ap.parse_args()
    if a.n <= 0 or a.step_ms <= 0 or 1600 % a.step_ms:
        ap.error("n must be positive; step-ms must be a positive divisor of 1600")
    idx = pd.read_parquet(a.index)
    pool = idx[idx.split == "train"]
    selected = pool.sample(min(a.n, len(pool)), random_state=1300).reset_index(drop=True)
    selected["review_id"] = np.arange(len(selected))
    if a.uncertain_review:
        reviewed = pd.read_csv(a.uncertain_review)
        selected = reviewed[reviewed.interval_status == "uncertain"].copy()
        if selected.review_id.duplicated().any() or selected.shot_id.duplicated().any():
            raise ValueError("Duplicate review or shot IDs")
        if not set(selected.shot_id).issubset(set(pool.shot_id)):
            raise ValueError("Contact review must contain official training shots only")
        canonical = pool.set_index("shot_id").clip_path
        if not selected.clip_path.eq(selected.shot_id.map(canonical)).all():
            raise ValueError("Review clip differs from the training index")
    a.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as executor:
        selected["sheet_path"] = list(executor.map(lambda r: sheet(r, a.out, a.step_ms), selected.itertuples()))
    selected[["review_id", "shot_id", "game", "goal", "label", "is_penalty", "sheet_path", "clip_path"]].to_csv(
        a.out / "sample.csv", index=False)
    summary = {"sampling": "uncertain subset of previous review" if a.uncertain_review else "simple random sample from official train only",
               "seed": 1300, "n": len(selected), "step_ms": a.step_ms,
               "review_sha256": digest(a.uncertain_review) if a.uncertain_review else None,
               "games": selected.game.nunique(), "goals": int(selected.goal.sum()),
               "warning": "Sheets contain post-shot information and MUST NOT be model inputs.",
               "annotation": "contact_lower_ms,contact_upper_ms,t0_status,body_part,confidence,note; uncertain is valid"}
    (a.out / "sampling.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
