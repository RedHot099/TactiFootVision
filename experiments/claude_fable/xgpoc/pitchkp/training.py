"""Training of pitch keypoint candidates on ticket 01's dataset (pitch-keypoints 03).

A run trains on ticket 01's `train` split and validates on a fixed subset of its `val` split (every `val_step`-th image
in name order, so GSR clips and Calibration games are both sampled); the test splits (`test_gsr`, `test_calib`) are
never listed. The run's own validation only steers Ultralytics' `best.pt`; candidates are chosen on GSR valid in metres
and Calibration valid in px (`scripts/pitchkp_gsr_eval.py`).

Augmentation (`YOLO_AUGMENT`) keeps one whole pitch per image: no mosaic, mixup, cutmix or copy-paste (they paste
several pitches or parts of them into one image); horizontal flips with the dataset's `flip_idx`; no vertical flips (a
broadcast frame shows the far touchline at the top); scale, translation and colour jitter, which map a pitch view to
another pitch view.

Ultralytics letterboxes every training image onto a square canvas (grey bands above and below a 16:9 frame), and the
model relies on that canvas: run it on the square canvas at inference (the package's `resize="square"`, evaluator input
`square`), not on the stride-padded letterbox, where y11m_960 found the pitch instance on only 20% of GSR valid images.
"""

import random
from pathlib import Path

import yaml

DATASET = Path("/mnt/storage/tactifoot_research/pitch_keypoints_sn")
RUNS = Path("/mnt/storage/tactifoot_research/pk_runs")
TRAIN_SPLIT, VAL_SPLIT = "train", "val"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

YOLO_AUGMENT = {
    "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "fliplr": 0.5, "flipud": 0.0, "degrees": 0.0, "shear": 0.0, "perspective": 0.0,
    "translate": 0.1, "scale": 0.3,
}


def split_images(root: Path, split: str) -> list[Path]:
    """The images of one split of the dataset, sorted by name."""
    folder = Path(root) / "images" / split
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no images in {folder}")
    return images


def training_yaml(root: Path, out: Path, *, val_step: int = 1, train_limit: int | None = None, seed: int = 0) -> Path:
    """Write `out/data.yaml` (and its image lists) for one run and return it.

    `train`: every training image, or `train_limit` of them drawn with `seed` (smoke runs). `val`: every `val_step`-th
    validation image. `kpt_shape`, `flip_idx`, `nc` and `names` are copied from the dataset's `data.yaml`.
    """
    if val_step < 1:
        raise ValueError(f"val_step must be >= 1, got {val_step}")
    source = yaml.safe_load((Path(root) / "data.yaml").read_text())
    train = split_images(root, TRAIN_SPLIT)
    if train_limit is not None:
        train = sorted(random.Random(seed).sample(train, min(train_limit, len(train))))
    val = split_images(root, VAL_SPLIT)[::val_step]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, images in (("train", train), ("val", val)):
        (out / f"{name}.txt").write_text("".join(f"{p}\n" for p in images))
    config = {"path": str(out.resolve()), "train": "train.txt", "val": "val.txt"}
    config |= {key: source[key] for key in ("kpt_shape", "flip_idx", "nc", "names")}
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return data_yaml

