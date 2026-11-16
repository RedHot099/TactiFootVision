"""01_data.ipynb: `tv.data`, datasets, format conversion, video."""

from common import REPO_URL, SYNTHETIC_PITCH_DATASET, Notebook

from tactifoot_vision.data.demo import ATTRIBUTION as DEMO_ATTRIBUTION


def notebook() -> Notebook:
    nb = Notebook("01_data")
    nb.md(f"""
# Data preparation with `tv.data`

{nb.badge}

Everything that comes before training: loading annotated datasets, deriving
new ones, converting between YOLO and COCO, and reading match video.

It runs on open data, the same from a checkout and on Google Colab, and
needs no GPU:

* the SoccerTrack v2 demo match from `tv.data.download_demo`: frames with
  player boxes and a 30-second clip (about 300 MB, CC BY 4.0);
* a small pitch keypoint dataset, rendered here from the package's synthetic
  pitch.

The demo data needs your own free Hugging Face token, as a Colab secret
named `HF_TOKEN` (or the environment variable of that name, or a saved
`hf auth login`). The [overview]({REPO_URL}/notebooks/00_overview.ipynb)
lists the three steps to get one.

A `Dataset` is a lightweight index. Images stay on disk, labels live in memory,
and every operation returns a new dataset, so nothing here changes the source
folders.

Part of the [TactiFoot Vision notebooks]({REPO_URL}/notebooks/README.md); the
[overview]({REPO_URL}/notebooks/00_overview.ipynb) shows how the stages connect.
""")
    nb.setup(
        imports="""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import tactifoot_vision as tv
"""
    )

    nb.md("""
## Loading a YOLO dataset

`tv.data.download_demo` fetches one match of
[SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2)
and lays it out for the package, among others as a YOLO detection dataset of
frames from its first minutes. A second call does nothing.

`tv.load_dataset` takes a YOLO `data.yaml` (or its folder) or a COCO folder and
picks the task from the files: detection here, pose further down. `summary()`
counts images and objects per split and class, which is the first thing to
check for class imbalance. The demo labels only the 22 players: the dataset
declares the package's four detection classes, and `goalkeeper`, `referee`
and `ball` stay empty. Every call below works the same on a dataset that has
all four.
""")
    nb.code("""
demo = tv.data.download_demo(DATA / "demo")
detection = tv.load_dataset(demo.detect_yaml)
print(detection)
detection.summary()
""")
    nb.md("""
`show_samples` draws random images of a split with their labels, a quick check
that the boxes line up with the pixels. The demo camera sees the whole pitch
through a fisheye lens, 4096x1080 pixels, so a player is about 25x60 pixels.
""")
    nb.code("""
tv.viz.show_samples(detection, "train", n=2, seed=1, cols=1)
""")
    nb.md("""
Each split is a list of `Sample`s: the image path, its size and the
`Annotations` in pixel coordinates. `read` loads the image (BGR, as in OpenCV)
together with a copy of its labels.
""")
    nb.code("""
sample = detection["train"][0]
image, annotations = detection.read("train", 0)
print("image:", image.shape, image.dtype, f"({sample.width}x{sample.height})")
print("objects:", len(annotations))
pd.DataFrame(annotations.boxes, columns=["x1", "y1", "x2", "y2"]).assign(
    class_name=[detection.class_names[i] for i in annotations.class_ids]
).head().round(1)
""")

    nb.md("""
## Deriving datasets

`subset`, `resplit`, `merge` and `with_split` each return a new dataset. They
cover the usual chores: a small subset for quick experiments, a fresh
train/valid/test split, combining two sources, or replacing one split.

`merge` and `with_split` take images from elsewhere and keep every image they
are given, so the sources should not overlap. Here the second source is a
`Dataset` built from detection images that `small` left out.
""")
    nb.code("""
small = detection.subset({"train": 200, "valid": 40}, seed=0)   # counts per split
tenth = detection.subset(0.1, seed=0)                           # a fraction of every split
resplit = small.resplit(train=0.7, valid=0.2, test=0.1, seed=0)  # pool and split again

in_small = {s.image_path for s in small}
unused = {split: [s for s in detection[split] if s.image_path not in in_small] for split in ("train", "valid")}
extra = tv.data.Dataset(detection.task, detection.class_names, {"train": unused["train"][:100]})
merged = small.merge(extra)                                     # same classes required
held_out = small.with_split("test", unused["valid"][:30])       # a test split small has not seen

print("distinct images in merged:", len({s.image_path for s in merged}), "of", len(merged))
pd.concat(
    {name: ds.summary()[["images", "objects"]] for name, ds in
     {"small": small, "tenth": tenth, "resplit": resplit, "merged": merged, "held_out": held_out}.items()},
    names=["dataset"],
)
""")

    nb.md("""
## Converting between YOLO and COCO

Ultralytics trains on YOLO folders and RF-DETR on COCO folders. `to_yolo` and
`to_coco` write either layout from any dataset. Images are symlinked by default
(`link="hardlink"` or `"copy"` also work) and labels are written fresh. The
trainers call these for you; here we call them directly and check that a
round trip YOLO → COCO → YOLO keeps every box and class.
""")
    nb.code("""
coco_root = small.to_coco(OUT / "demo_coco")
from_coco = tv.load_dataset(coco_root)
yolo_yaml = from_coco.to_yolo(OUT / "demo_yolo_again")
from_yolo = tv.load_dataset(yolo_yaml)

print("COCO:", from_coco)
print("YOLO:", from_yolo)
print("files:", sorted(p.name for p in coco_root.iterdir()), "+", yolo_yaml.name)
print("images are symlinks:", next((coco_root / "train").glob("*.jpg")).is_symlink())
""")
    nb.md("""
The check compares labels image by image (keyed by file name), since a format
may store images in a different order. Writers clip boxes to the image, because
Ultralytics rejects coordinates outside it, so the source boxes are clipped the
same way before comparing.
""")
    nb.code("""
def labels_by_image(dataset):
    # {(split, image stem): (class ids, boxes clipped to the image)}
    labels = {}
    for split in dataset.split_names:
        for sample in dataset[split]:
            a = sample.annotations
            size = [sample.width, sample.height] * 2
            labels[split, sample.image_path.stem] = (a.class_ids, a.boxes.clip(0, size))
    return labels


def same_labels(a, b, atol=1e-3):
    if a.keys() != b.keys():
        return False
    for key, (ids_a, boxes_a) in a.items():
        ids_b, boxes_b = b[key]
        # Same object order on both sides (YOLO stores 6 decimals, hence the rounding).
        order_a = np.lexsort((*np.round(boxes_a).T, ids_a))
        order_b = np.lexsort((*np.round(boxes_b).T, ids_b))
        if not (np.array_equal(ids_a[order_a], ids_b[order_b])
                and np.allclose(boxes_a[order_a], boxes_b[order_b], atol=atol)):
            return False
    return True


original = labels_by_image(small)
outside = sum(
    int((s.annotations.boxes != s.annotations.boxes.clip(0, [s.width, s.height] * 2)).any(axis=1).sum())
    for s in small
)
print("source boxes reaching outside their image:", outside)
print("YOLO -> COCO identical:        ", same_labels(original, labels_by_image(from_coco)))
print("YOLO -> COCO -> YOLO identical:", same_labels(original, labels_by_image(from_yolo)))
""")

    nb.md(f"""
## Pose datasets and `flip_idx`

A pitch keypoint dataset is a pose task: each image has one `pitch` object
with 32 landmark keypoints `(x, y, visibility)`, the vertices of
`tv.SoccerPitch`. `flip_idx` says which landmark takes the place of which when
the image is mirrored (left corner ↔ right corner). Augmentation needs it to
flip keypoints correctly, see
[02_augmentation]({REPO_URL}/notebooks/02_augmentation.ipynb).

This one is made here, from the package's synthetic pitch:
`broadcast_homographies` draws random views of a main broadcast camera,
`synthetic_pitch_clip` renders the pitch markings through them into an image
sequence, and `Dataset.from_clips(..., task="pose")` projects the pitch
landmarks into every frame. Any clip with calibration truth gives a dataset
the same way. The split counts clips, so all frames of a clip land in one
split. Four clips of eight small frames take seconds on a CPU.
""")
    nb.code(f"""
{SYNTHETIC_PITCH_DATASET.strip()}
print(pitch_data)
print("flip_idx:", pitch_data.flip_idx)
print("the pitch's:", list(pitch_data.flip_idx) == list(tv.SoccerPitch().flip_idx))
pitch_data.summary()
""")
    nb.code("""
tv.viz.show_samples(pitch_data, "train", n=3)
""")
    nb.md("""
Visibility follows YOLO and COCO: 0 not labelled, 1 occluded, 2 visible. A
broadcast frame shows only part of the pitch, so most landmarks of an image
are not labelled. A rendered frame knows every landmark exactly, so it uses
only 0 and 2; hand-labelled frames also mark landmarks hidden behind players
as occluded.
""")
    nb.code("""
keypoints = np.concatenate([s.annotations.keypoints for s in pitch_data["train"]])
visibility = pd.Series(keypoints[..., 2].ravel()).map(
    {tv.data.NOT_LABELLED: "not labelled", tv.data.OCCLUDED: "occluded", tv.data.VISIBLE: "visible"}
)
print("keypoints per object:", keypoints.shape[1])
visibility.value_counts(normalize=True).round(3).to_frame("share of keypoints")
""")
    nb.md("""
COCO has no field for `flip_idx`; the writer stores it on the category so the
round trip keeps it. Labelled keypoints come back unchanged; unlabelled ones
are written as `0 0 0`, so only their visibility is compared.
""")
    nb.code("""
pitch_coco = tv.load_dataset(pitch_data.to_coco(OUT / "pitch_coco"))
print(pitch_coco)
print("flip_idx kept:", pitch_coco.flip_idx == pitch_data.flip_idx)


def keypoints_by_image(dataset):
    return {(split, s.image_path.stem): s.annotations.keypoints for split in dataset.split_names for s in dataset[split]}


before, after = keypoints_by_image(pitch_data), keypoints_by_image(pitch_coco)
same = before.keys() == after.keys()
for key, kp in before.items():
    labelled = kp[..., 2] > tv.data.NOT_LABELLED
    same &= np.array_equal(kp[..., 2], after[key][..., 2])
    same &= np.allclose(kp[labelled], after[key][labelled], atol=1e-3)
print("keypoints kept:", bool(same))
""")

    nb.md("""
## Match video

`VideoReader` gives the video's properties and reads frames in order
(`frames(start, end, stride)` yields `(index, frame)` pairs) or one at a time
by index. Frames are BGR arrays, like dataset images. It reads a video file
or an image sequence, a pattern such as `img1/%06d.jpg` whose first image is
frame 0; the demo clip is such a sequence, laid out as SoccerNet clips are.
""")
    nb.code("""
clip = tv.data.load_labelled_clip(demo.clip_dir)
video = tv.VideoReader(clip.video_path)
print(video)
print(f"{video.width}x{video.height}, {video.fps:.0f} fps, {video.frame_count} frames, {video.duration:.0f} s")

every_ten_seconds = list(video.frames(start=0, end=video.frame_count, stride=250))
print("frames read:", [index for index, _ in every_ten_seconds])
tv.show([video.read(i) for i in (0, 375, 725)], titles=["0 s", "15 s", "29 s"], cols=1)
""")
    nb.md("""
`extract_frames` saves every `stride`-th frame of `start <= index < end` as an
image (the same frame range as `VideoReader.frames`), the usual first step when
annotating new footage. Files are named after the video, or after the folder
of an image sequence.
""")
    nb.code("""
paths = tv.data.extract_frames(clip.video_path, OUT / "frames", end=750, stride=250)
print([path.name for path in paths])
""")

    nb.md(f"""
## Sources

{DEMO_ATTRIBUTION}

The pitch keypoint frames are rendered by `tactifoot_vision`.
""")
    return nb
