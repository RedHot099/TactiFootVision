"""SoccerNet GSR (+ SoccerNet-Tracking 2023 images) into the detector-data pool (detector-data 03, GSR part).

    uv run python scripts/detector_pool_gsr.py

Owner rule 2026-10-14 ("collect everything, tag, split later"): every labelled GSR frame of train, valid and test
(all 25 fps frames, 750 per clip) goes into one YOLO set, `sets/soccernet_gsr/`, images symlinked; each image gets
its tags in `pool_index_gsr.parquet` (concatenated into `pool_index.parquet` by the pool script of the sources
branch). Splits are views over the tags; the default training view excludes `gsr_split` valid/test.

SoccerNet-Tracking 2023 train/test are the GSR train/test clips (SNMOT-xxx = SNGS-xxx, same 750 frames, the same
images byte for byte and the same boxes: 1,297,548 GSR objects vs 1,297,549 Tracking rows, one Tracking box not in
GSR). It adds no label, so no `soccernet_tracking` rows; it supplies the images of the GSR train/test frames that the
local GSR download skipped (it kept every 5th), tagged `image_source = tracking`. The script re-checks the byte
identity on every frame present in both.

Classes: GSR categories ball, goalkeeper, player, referee; category "other" (staff, ball boys: people without a
role) is dropped and counted per image (`n_other_dropped`); all four classes are annotated exhaustively.
`stride_group` = (frame - 1) // 5 per clip, so a view can keep one frame per group (group k's first frame,
1 + 5k, is the benchmark's frame). `near_dup_of` is the benchmark frame (GSR valid/test on disk, football_yolo valid)
that a GSR train image duplicates (dHash <= 4 bits and thumbnail correlation >= 0.95, the sources branch's rule);
GSR valid/test rows are the benchmark games themselves and are marked by `gsr_split`.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import tactifoot_vision as tv
from tactifoot_vision.data import Dataset, Sample

CHECKOUT = Path("/home/kuba/projects/tactifoot_vision")
GSR = CHECKOUT / "data/SoccerNetGS"
TRACKING = CHECKOUT / "data/soccernet/tracking/extracted"
ROOT = Path("/mnt/storage/tactifoot_research/detector_data")
OUT = ROOT / "sets/soccernet_gsr"
INDEX = ROOT / "pool_index_gsr.parquet"
SOURCES = CHECKOUT / ".claude/worktrees/detector-data-sources"
GROUP = 5
CLASSES = ["ball", "goalkeeper", "player", "referee"]
LICENCE = (
    "SoccerNet GSR / Tracking 2023: broadcast footage under the SoccerNet NDA (research use, weights for research "
    "only); GSR annotations as released by the SoccerNet team (sn-gamestate)"
)
MANIFEST = {
    "source": "soccernet_gsr",
    "content": "SoccerNet Game State Reconstruction v1.3 (sn-gamestate), train + valid + test, every labelled frame; "
    "12 games of the 2019 Swiss Super League (info.game_id 2-11), 164 clips of 30 s at 25 fps, 1920x1080",
    "licence": LICENCE,
    "access_route": "GSR: SoccerNet downloader share (data/SoccerNetGS/fetch_log.jsonl; train/test zips kept every "
    "5th image); Tracking 2023 train/test zips (data/soccernet/tracking-2023, extracted to "
    "data/soccernet/tracking/extracted) for the other train/test images",
    "mapping": {
        "player": "player",
        "goalkeeper": "goalkeeper",
        "referee": "referee",
        "ball": "ball",
        "other": "dropped (no class; n_other_dropped per image; a detection on such a person is not penalised by "
        "the loss only if a trainer masks it, so they stay a small source of label noise)",
    },
    "annotated_classes": CLASSES,
    "game_ids": "info.game_id is an index of GSR's 12 games, not a SoccerNet game path; train 4/6/9, valid 2/3/5, "
    "test 7/8/11. None is an xG plan game (plan.csv has 2014-2017 SoccerNet-v2 games, no Swiss league)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources_common():
    """The sources branch's `scripts/detector_data/common.py` (near-duplicate rule and benchmark fingerprints)."""
    path = SOURCES / "scripts/detector_data/common.py"
    spec = importlib.util.spec_from_file_location("detector_data_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = (
        module  # its process pool pickles its functions by module name
    )
    spec.loader.exec_module(module)
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(SOURCES),
            "log",
            "-1",
            "--format=%H",
            "--",
            "scripts/detector_data/common.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return module, {
        "file": str(path),
        "last_commit": commit,
        "sha256": sha256_file(path),
    }


def tracking_identity() -> dict:
    """Byte identity of every frame that both the GSR download and Tracking hold."""
    checked, different = 0, []
    for split in ("train", "test"):
        for image in sorted((GSR / split).glob("SNGS-*/img1/*.jpg")):
            twin = (
                TRACKING
                / split
                / image.parts[-3].replace("SNGS-", "SNMOT-")
                / "img1"
                / image.name
            )
            checked += 1
            if image.stat().st_size != twin.stat().st_size or sha256_file(
                image
            ) != sha256_file(twin):
                different.append(str(image))
    if different:
        raise SystemExit(
            f"{len(different)} GSR images differ from Tracking, e.g. {different[:3]}"
        )
    return {"frames_compared": checked, "different": 0, "rule": "file size and sha256"}


def main() -> None:
    started = datetime.now(UTC)
    common, common_ref = load_sources_common()
    identity = tracking_identity()
    dataset = Dataset.from_gsr(
        GSR, splits=("train", "valid", "test"), tracking_root=TRACKING
    )

    links = OUT / "_links"
    renamed = {}
    for split in dataset.split_names:
        folder = links / split
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for s in dataset[split]:
            link = folder / f"{s.meta['sequence']}_{int(s.meta['frame']):06d}.jpg"
            if not link.is_symlink():
                link.symlink_to(s.image_path)
            elif link.resolve() != s.image_path.resolve():
                raise SystemExit(f"{link} points elsewhere; remove {links}")
            out.append(Sample(link, s.width, s.height, s.annotations, s.meta))
        renamed[split] = out
    linked = Dataset(dataset.task, dataset.class_names, renamed, name="soccernet_gsr")
    data_yaml = linked.to_yolo(OUT / "yolo")

    references = common.benchmark_fingerprints()
    train_paths = [s.image_path for s in dataset["train"]]
    train_prints = common.Fingerprints.of(train_paths)
    hits = [
        hit
        for ref in references.values()
        for hit in common.near_duplicates(train_prints, ref)
    ]
    near = {}
    for hit in sorted(hits, key=lambda h: -h["corr"]):
        near.setdefault(hit["image"], hit["reference"])

    rows = []
    for split in linked.split_names:
        for original, s in zip(dataset[split], linked[split], strict=True):
            frame = int(s.meta["frame"])
            ids = s.annotations.class_ids
            rows.append(
                {
                    "image": str(
                        data_yaml.parent / split / "images" / s.image_path.name
                    ),
                    "label": str(
                        data_yaml.parent / split / "labels" / f"{s.image_path.stem}.txt"
                    ),
                    "image_target": str(original.image_path),
                    "source": "soccernet_gsr",
                    "image_source": s.meta["image_source"],
                    "game_id": f"gsr:{s.meta['game']}",
                    "game_path": None,
                    "clip": s.meta["sequence"],
                    "frame": frame,
                    "stride_group": (frame - 1) // GROUP,
                    "gsr_split": split,
                    "xg_role": None,
                    "near_dup_of": near.get(str(original.image_path)),
                    "licence": LICENCE,
                    "annotated_classes": ",".join(CLASSES),
                    "width": s.width,
                    "height": s.height,
                    **{
                        f"n_{name}": int((ids == i).sum())
                        for i, name in enumerate(CLASSES)
                    },
                    "n_other_dropped": int(s.meta["other_dropped"]),
                }
            )
    index = pd.DataFrame(rows)
    missing = [p for p in index.image.head(1000) if not Path(p).exists()]
    if missing:
        raise SystemExit(f"exported image paths not found, e.g. {missing[:3]}")
    index.to_parquet(INDEX, index=False)

    labels = sorted(OUT.glob("yolo/*/labels/*.txt"))
    sha_file = OUT / "label_sha256.txt"
    sha_file.write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(OUT).as_posix()}\n" for p in labels)
    )
    summary = linked.summary()
    manifest = MANIFEST | {
        "created": started.isoformat(timespec="seconds"),
        "data_yaml": str(data_yaml),
        "images": len(linked),
        "boxes_per_split": {
            split: {k: int(v) for k, v in row.items()}
            for split, row in summary.iterrows()
        },
        "other_dropped": int(index.n_other_dropped.sum()),
        "image_source_counts": index.groupby(["gsr_split", "image_source"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict(orient="records"),
        "tracking_identity": identity,
        "soccernet_tracking": "no rows: same clips, frames, images and boxes as GSR train/test (one extra Tracking "
        "box in 1.3M); used only as the image source of the frames the GSR download skipped",
        "stride_group": f"(frame - 1) // {GROUP} within a clip",
        "near_duplicates": {
            "rule": "dHash <= 4 bits and 64x36 thumbnail Pearson >= 0.95 against the benchmark frames "
            "(GSR valid/test on disk, football_yolo valid); checked for GSR train",
            "train_images_flagged": int(index.near_dup_of.notna().sum()),
            "code": common_ref,
        },
        "index": {"file": str(INDEX), "sha256": sha256_file(INDEX), "rows": len(index)},
        "label_files": len(labels),
        "label_sha256_of_list": sha256_file(sha_file),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "images",
                    "boxes_per_split",
                    "other_dropped",
                    "image_source_counts",
                    "tracking_identity",
                    "near_duplicates",
                )
            },
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    tv.setup_logging("INFO")
    sys.exit(main())
