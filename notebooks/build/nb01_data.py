"""01_data.ipynb: `tv.data`, datasets, format conversion, video and StatsBomb events."""

from common import Notebook


def notebook() -> Notebook:
    nb = Notebook("01_data")
    nb.md("""
# Data preparation with `tv.data`

Everything that comes before training: loading annotated datasets, deriving
new ones, converting between YOLO and COCO, reading match video, and loading
StatsBomb 360 event data.

A `Dataset` is a lightweight index. Images stay on disk, labels live in memory,
and every operation returns a new dataset, so nothing here changes the source
folders.

Part of the [TactiFoot Vision notebooks](README.md); the
[overview](00_overview.ipynb) shows how the stages connect.
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

`tv.load_dataset` takes a YOLO `data.yaml` (or its folder) or a COCO folder and
picks the task from the files: detection here, pose further down. `summary()`
counts images and objects per split and class, which is the first thing to
check for class imbalance.
""")
    nb.code("""
detection = tv.load_dataset(DATA / "datasets" / "football_yolo")
print(detection)
detection.summary()
""")
    nb.md("""
`show_samples` draws random images of a split with their labels, a quick check
that the boxes line up with the pixels.
""")
    nb.code("""
tv.viz.show_samples(detection, "train", n=6, seed=1)
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
""")
    nb.code("""
small = detection.subset({"train": 200, "valid": 40}, seed=0)   # counts per split
tenth = detection.subset(0.1, seed=0)                           # a fraction of every split
resplit = small.resplit(train=0.7, valid=0.2, test=0.1, seed=0)  # pool and split again
merged = small.merge(tenth)                                     # same classes required
held_out = small.with_split("test", detection.subset(30, seed=1)["valid"])

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
coco_root = small.to_coco(OUT / "football_coco")
from_coco = tv.load_dataset(coco_root)
yolo_yaml = from_coco.to_yolo(OUT / "football_yolo_again")
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

    nb.md("""
## Pose datasets and `flip_idx`

The pitch dataset is a pose task: each image has one `pitch` object with 32
landmark keypoints `(x, y, visibility)`. `flip_idx` says which landmark takes
the place of which when the image is mirrored (left corner ↔ right corner).
Augmentation needs it to flip keypoints correctly, see
[02_augmentation](02_augmentation.ipynb).
""")
    nb.code("""
pitch_data = tv.load_dataset(DATA / "keypoints")
print(pitch_data)
print("flip_idx:", pitch_data.flip_idx)
pitch_data.summary()
""")
    nb.code("""
tv.viz.show_samples(pitch_data, "train", n=3)
""")
    nb.md("""
Visibility follows YOLO and COCO: 0 not labelled, 1 occluded, 2 visible. A
broadcast frame shows only part of the pitch, so most landmarks of an image
are unlabelled.
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
pitch_subset = pitch_data.subset(20, seed=0)
pitch_coco = tv.load_dataset(pitch_subset.to_coco(OUT / "pitch_coco"))
print(pitch_coco)
print("flip_idx kept:", pitch_coco.flip_idx == pitch_data.flip_idx)


def keypoints_by_image(dataset):
    return {(split, s.image_path.stem): s.annotations.keypoints for split in dataset.split_names for s in dataset[split]}


before, after = keypoints_by_image(pitch_subset), keypoints_by_image(pitch_coco)
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
by index. Frames are BGR arrays, like dataset images.
""")
    nb.code("""
video = tv.VideoReader(VIDEO)
print(video)
print(f"{video.width}x{video.height}, {video.fps:.0f} fps, {video.frame_count} frames, {video.duration:.0f} s")

every_ten_seconds = list(video.frames(start=0, end=video.frame_count, stride=250))
print("frames read:", [index for index, _ in every_ten_seconds])
tv.show([video.read(i) for i in (0, 750, 1400)], titles=["0 s", "30 s", "56 s"])
""")
    nb.md("""
`extract_frames` saves every `stride`-th frame of `start <= index < end` as an
image (the same frame range as `VideoReader.frames`), the usual first step when
annotating new footage.
""")
    nb.code("""
paths = tv.data.extract_frames(VIDEO, OUT / "frames", end=1500, stride=375)
print([path.name for path in paths])
""")

    nb.md("""
## StatsBomb 360 events

`load_statsbomb` reads a match's `*_events.json` and `*_360.json` (or a CSV
saved from the same table) into one row per player visible in an event's
freeze frame. Locations are in StatsBomb's 120 × 80 pitch units. This is the
reference that pipeline positions can be compared against, see
[04_inference](04_inference.ipynb).
""")
    nb.code("""
statsbomb = tv.data.load_statsbomb(DATA / "statsbomb")
print(f"{len(statsbomb):,} objects in {statsbomb['event_uuid'].nunique():,} freeze frames")
statsbomb[["period", "minute", "second", "type_name", "team_name", "pitch_location", "teammate", "type"]].head()
""")
    nb.md("""
`tv.viz.draw_pitch` takes any pitch size, so a freeze frame can be checked at a
glance: the actor's team in blue, opponents in pink.
""")
    nb.code("""
event = statsbomb[statsbomb["event_uuid"] == statsbomb["event_uuid"].iloc[0]]
first = event.iloc[0]
xy = np.array(event["pitch_location"].tolist())

fig = tv.viz.draw_pitch(pitch=tv.SoccerPitch(120, 80), size=7)
ax = fig.axes[0]
ax.scatter(xy[:, 0], xy[:, 1], s=50, zorder=3, edgecolor="white",
           c=np.where(event["teammate"], "#00BFFF", "#FF1493"))
ax.set_title(f"{first.type_name} by {first.team_name}, {first.minute}:{first.second:02d}")
fig
""")
    return nb
