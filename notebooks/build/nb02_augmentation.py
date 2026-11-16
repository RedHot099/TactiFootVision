"""02_augmentation.ipynb: `tv.augment` transforms, containers and offline augmentation."""

from common import REPO_URL, SYNTHETIC_PITCH_DATASET, Notebook

from tactifoot_vision.data.demo import ATTRIBUTION as DEMO_ATTRIBUTION


def notebook() -> Notebook:
    nb = Notebook("02_augmentation")
    nb.md(f"""
# Data augmentation with `tv.augment`

{nb.badge}

Augmentation makes more training images from the ones you have: mirrored,
rotated, zoomed, recoloured, blurred. A transform takes and returns
`(image, annotations)`, so boxes and pitch keypoints move with the pixels and
the labels stay correct.

* Geometric transforms (`HorizontalFlip`, `RandomAffine`) move pixels, boxes
  and keypoints together.
* Photometric transforms (`ColorJitter`, `GaussianBlur`, `MotionBlur`,
  `GaussianNoise`, `JpegCompression`) change pixel values only.
* `Compose` and `OneOf` combine them; `augment_dataset` writes augmented
  copies of a training split to disk.

There is no vertical flip on purpose: broadcast footage is never upside down.

It runs on open data, the same from a checkout and on Google Colab, and
needs no GPU:

* frames with player boxes from the SoccerTrack v2 demo match of
  `tv.data.download_demo` (about 300 MB, CC BY 4.0);
* a small pitch keypoint dataset, rendered here from the package's synthetic
  pitch, as in [01_data]({REPO_URL}/notebooks/01_data.ipynb).

The demo data needs your own free Hugging Face token, as a Colab secret
named `HF_TOKEN` (or the environment variable of that name, or a saved
`hf auth login`). The [overview]({REPO_URL}/notebooks/00_overview.ipynb)
lists the three steps to get one.

Part of the [TactiFoot Vision notebooks]({REPO_URL}/notebooks/README.md);
datasets are introduced in [01_data]({REPO_URL}/notebooks/01_data.ipynb).
""")
    nb.setup(
        imports="""
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import tactifoot_vision as tv
from tactifoot_vision import augment as A
from tactifoot_vision.enums import Role, Split, Task
"""
    )

    nb.md("""
## Demo frames, cut to windows

`download_demo` fetches the demo match once; a second call does nothing. A
small subset of its detection frames is enough here.

The demo camera sees the whole pitch through a fisheye lens, 4096x1080
pixels, so a player is about 25x60 pixels and nearly vanishes when a frame
is shrunk to the width of a notebook. To keep the effect of each transform
visible, this notebook cuts a 1280x720 window around the players out of
every frame, with the boxes inside it, and works on those windows. The
transforms act the same on full frames, as the
[overview]({REPO_URL}/notebooks/00_overview.ipynb) shows.
""")
    nb.code("""
demo = tv.data.download_demo(DATA / "demo")
frames = tv.load_dataset(demo.detect_yaml).subset({Split.TRAIN: 60, Split.VALID: 20}, seed=0)


def window(sample, folder, size=(1280, 720)):
    \"\"\"The size window of a sample's frame around its players, as a new
    sample written to folder, and the window's top-left corner in the frame.\"\"\"
    width, height = size
    boxes = sample.annotations.boxes
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    cx, cy = np.median(centres, axis=0) if len(boxes) else (sample.width / 2, sample.height / 2)
    x0 = int(np.clip(cx - width / 2, 0, sample.width - width))
    y0 = int(np.clip(cy - height / 2, 0, sample.height - height))
    inside = ((centres >= [x0, y0]) & (centres < [x0 + width, y0 + height])).all(axis=1)
    labels = sample.annotations.select(inside)
    labels = replace(labels, boxes=(labels.boxes - [x0, y0, x0, y0]).clip(0, [width, height] * 2))
    path = folder / sample.image_path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sample.read_image()[y0 : y0 + height, x0 : x0 + width])
    return tv.data.Sample(path, width, height, labels), (x0, y0)


cuts = {split: [window(s, OUT / "windows" / split) for s in frames[split]] for split in frames.split_names}
windows = tv.data.Dataset(
    frames.task, frames.class_names, {split: [s for s, _ in cut] for split, cut in cuts.items()}
)
print(frames)
print(windows)

x0, y0 = cuts[Split.TRAIN][3][1]
frame = tv.viz.draw_annotations(*frames.read(Split.TRAIN, 3), frames.class_names)
cv2.rectangle(frame, (x0, y0), (x0 + 1280, y0 + 720), (255, 255, 255), 6)
tv.show(frame, "a demo frame and its window", width=14)
""")
    nb.md("""
Every transform has a probability `p` of being applied. Calls take a seeded
`np.random.Generator`, so a result can be reproduced. A small helper draws
the labels before and after with `tv.viz.draw_annotations`.
""")
    nb.code("""
image, labels = windows.read(Split.TRAIN, 3)


def before_after(transform, image, labels, class_names, seeds=(0,), width=14):
    images = [tv.viz.draw_annotations(image, labels, class_names)]
    titles = [f"original: {len(labels)} objects"]
    for seed in seeds:
        new_image, new_labels = transform(image, labels, np.random.default_rng(seed))
        images.append(tv.viz.draw_annotations(new_image, new_labels, class_names))
        titles.append(f"{type(transform).__name__} (seed {seed}): {len(new_labels)} objects")
    return tv.show(images, titles, cols=2, width=width)
""")

    nb.md("""
## Geometric transforms

`HorizontalFlip` mirrors the image and its boxes. `p=1.0` forces it here; in
training a flip on half the images (the default `p=0.5`) is typical.
""")
    nb.code("""
before_after(A.HorizontalFlip(p=1.0), image, labels, windows.class_names)
""")
    nb.md("""
`RandomAffine` rotates and scales about the image centre, then shifts. Each
box becomes the bounding box of its four moved corners, clipped to the image.
Objects pushed mostly out of frame (less than `min_area_ratio` of the box
left) are dropped, so the object count can shrink.
""")
    nb.code("""
affine = A.RandomAffine(degrees=8, translate=0.2, scale=(0.7, 1.3), p=1.0)
before_after(affine, image, labels, windows.class_names, seeds=(0, 1, 2))
""")

    nb.md("""
## Photometric transforms

These change only pixel values: lighting, focus and compression vary between
broadcasts. Blur, noise and JPEG artefacts are small, so the views below are
zoomed in on one player. The annotations come back as the very same object.
""")
    nb.code("""
photometric = {
    "ColorJitter": A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.05, p=1.0),
    "GaussianBlur": A.GaussianBlur(sigma=(2.0, 2.0), p=1.0),
    "MotionBlur": A.MotionBlur(kernel_size=(15, 15), p=1.0),
    "GaussianNoise": A.GaussianNoise(std=(12.0, 12.0), p=1.0),
    "JpegCompression": A.JpegCompression(quality=(8, 8), p=1.0),
}
player = labels.boxes[list(labels.class_ids).index(windows.class_names.index(Role.PLAYER))]
cx, cy = (player[:2] + player[2:]) / 2
x0, y0 = int(max(0, cx - 120)), int(max(0, cy - 90))
zoom = (slice(y0, y0 + 180), slice(x0, x0 + 240))

views, titles = [image[zoom]], ["original"]
for name, transform in photometric.items():
    new_image, new_labels = transform(image, labels, np.random.default_rng(0))
    assert new_labels is labels  # pixels only
    views.append(new_image[zoom])
    titles.append(name)
tv.show(views, titles, cols=3, width=12)
""")

    nb.md("""
## Combining transforms

`Compose` applies transforms in order, each with its own `p`. `OneOf` picks one
of its transforms at random. A typical training recipe flips half the time,
moves the camera a little, varies the colours, and sometimes adds one kind of
degradation. The repr shows the whole recipe.
""")
    nb.code("""
recipe = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomAffine(degrees=3, translate=0.05, scale=(0.8, 1.2), p=0.7),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.3, p=0.8),
    A.OneOf([A.MotionBlur(), A.GaussianBlur(), A.GaussianNoise()], p=0.4),
    A.JpegCompression(p=0.3),
])
recipe
""")
    nb.code("""
before_after(recipe, image, labels, windows.class_names, seeds=(0, 1, 2))
""")

    nb.md(f"""
## Keypoints under flips and affine transforms

Pitch landmarks have names: keypoint 0 is the corner at one end, keypoint 24
the matching corner at the other end. After a mirror flip, the landmark seen
where keypoint 0 was is really keypoint 24. `HorizontalFlip` renames keypoints
with the dataset's `flip_idx` (new keypoint `i` is the mirrored old keypoint
`flip_idx[i]`). A dataset loaded from YOLO files reads `flip_idx` from
`data.yaml`; this one comes from the pitch itself.

The keypoint frames are the small synthetic dataset of
[01_data]({REPO_URL}/notebooks/01_data.ipynb): random views of a broadcast
camera, rendered from the package's pitch model, with the pitch landmarks
projected into every frame. A view shows only part of the pitch, so the
examples below use the training frame with the most landmarks in view.
""")
    nb.code(f"""
{SYNTHETIC_PITCH_DATASET.strip()}
print(pitch_data)

in_view = [int((s.annotations.keypoints[..., 2] > 0).sum()) for s in pitch_data[Split.TRAIN]]
pitch_image, pitch_labels = pitch_data.read(Split.TRAIN, int(np.argmax(in_view)))
print("landmarks in view:", max(in_view), "of", pitch_labels.num_keypoints)
""")
    nb.code("""
flip = A.HorizontalFlip(p=1.0)
before_after(flip, pitch_image, pitch_labels, pitch_data.class_names, width=12)
""")
    nb.md("""
The numbers confirm it: each flipped keypoint sits at the mirrored position of
the old keypoint `flip_idx[i]`, and its visibility moves with it.
""")
    nb.code("""
width = pitch_image.shape[1]
_, flipped = flip(pitch_image, pitch_labels, np.random.default_rng(0))
old, new = pitch_labels.keypoints[0], flipped.keypoints[0]
expected = old[list(pitch_labels.flip_idx)]
print("x mirrored:  ", np.allclose(new[:, 0], width - expected[:, 0]))
print("y unchanged: ", np.allclose(new[:, 1], expected[:, 1]))
print("visibility:  ", np.array_equal(new[:, 2], expected[:, 2]))
""")
    nb.md("""
Without `flip_idx` a flip would keep the old names on mirrored positions. The
labels would then describe a mirror image of the pitch, which no camera can
film, and a keypoint model trained on them would learn contradictions.

A homography fitted to the labelled landmarks shows the difference: a real
view preserves the pitch's orientation (the sign of the mapping's Jacobian
determinant), a mirror image reverses it. A homography needs at least four
landmarks. Annotations without `flip_idx` raise an error instead of producing
such labels silently.
""")
    nb.code("""
def mirrored(annotations, image_size):
    \"\"\"True when the labelled landmarks only fit a mirror image of the pitch,
    None when fewer than four are labelled.\"\"\"
    h = tv.pitch.HomographyEstimator().update(annotations.to_keypoints())
    if h is None:
        return None
    width, height = image_size
    centre = h[2] @ [width / 2, height / 2, 1.0]
    return bool(np.linalg.det(h) * centre < 0)


image_size = pitch_image.shape[1::-1]
rng = np.random.default_rng(0)
_, renamed = flip(pitch_image, pitch_labels, rng)
_, not_renamed = flip(pitch_image, replace(pitch_labels, flip_idx=tuple(range(32))), rng)
_, moved = A.RandomAffine(degrees=10, translate=0.1, p=1.0)(pitch_image, pitch_labels, rng)
print("original mirrored:              ", mirrored(pitch_labels, image_size))
print("flip with flip_idx mirrored:    ", mirrored(renamed, image_size))
print("flip keeping old names mirrored:", mirrored(not_renamed, image_size))
print("affine mirrored:                ", mirrored(moved, image_size))

try:
    flip(pitch_image, replace(pitch_labels, flip_idx=None))
except ValueError as error:
    print("\\nwithout flip_idx:", error)
""")
    nb.md("""
The same check runs over a whole dataset, so a labelling error cannot slip into
training unnoticed. Every image of every split with at least four landmarks in
view is tested.
""")
    nb.code("""
checked = {
    (split, sample.image_path.name): mirrored(sample.annotations, (sample.width, sample.height))
    for split in pitch_data.split_names
    for sample in pitch_data[split]
}
fitted = {key: result for key, result in checked.items() if result is not None}
print(f"images with four or more landmarks: {len(fitted)} of {len(checked)}")
print(f"images mirrored: {sum(fitted.values())} of {len(fitted)}")
print([key for key, result in fitted.items() if result])
""")
    nb.md("""
None is flagged: the dataset's labels all describe a real view of the pitch. To
see what a flagged image looks like, the next cell builds a mirrored labelling
in memory from one sample. A mirrored labelling flips every labelled keypoint
from the top touchline to the bottom one, not just one pair: 13 and 16 swap,
and so do 14 and 15, 17 and 20, and so on. The points stay where they are in
the image and only their names change.
""")
    nb.code("""
pitch = tv.pitch.SoccerPitch()
touchline_mirror = [  # the vertex at the same place on the opposite touchline
    int(np.argmin(np.linalg.norm(pitch.vertices - [x, pitch.width - y], axis=1)))
    for x, y in pitch.vertices
]
print("swapped pairs:", [(i, j) for i, j in enumerate(touchline_mirror) if i < j])

wrong_keypoints = np.zeros_like(pitch_labels.keypoints)
wrong_keypoints[:, touchline_mirror] = pitch_labels.keypoints
wrong_labels = replace(pitch_labels, keypoints=wrong_keypoints)
print("sample mirrored:       ", mirrored(pitch_labels, image_size))
print("mirrored copy mirrored:", mirrored(wrong_labels, image_size))

tv.show(
    [
        tv.viz.draw_annotations(pitch_image, pitch_labels),
        tv.viz.draw_annotations(pitch_image, wrong_labels),
    ],
    ["sample labels", "same points, names flipped top to bottom"],
    cols=2,
    width=12,
)
""")
    nb.md("""
`RandomAffine` moves keypoints with the same matrix as the pixels. Keypoints
that leave the image become "not labelled" (visibility 0) instead of pointing
outside it.
""")
    nb.code("""
strong = A.RandomAffine(degrees=10, translate=0.25, scale=(1.2, 1.4), p=1.0)
_, zoomed = strong(pitch_image, pitch_labels, np.random.default_rng(1))
print("labelled keypoints before:", int((pitch_labels.keypoints[..., 2] > 0).sum()),
      " after:", int((zoomed.keypoints[..., 2] > 0).sum()))
before_after(strong, pitch_image, pitch_labels, pitch_data.class_names, seeds=(1,), width=12)
""")

    nb.md("""
## Augmenting a dataset

`show_augmentations` previews a recipe on random dataset images, original and
augmented side by side, before committing to it.
""")
    nb.code("""
tv.viz.show_augmentations(windows, recipe, n=2, seed=3)
""")
    nb.md("""
`augment_dataset` writes `copies` augmented variants of every training image
to `out_dir/<split>/images/<stem>_aug<k>` (the originals stay where they are)
and returns the bigger dataset: originals plus variants. Validation data is left
alone, so metrics keep measuring performance on real images. The same seed
gives the same images, and the result exports with `to_yolo` / `to_coco` like
any dataset.

It never overwrites: an earlier augmented dataset, and any export of it,
still reads its images, so an image the call would write must not exist yet
(the error names it). Use a new folder per call; this notebook removes its
previous run's folder first.
""")
    nb.code("""
shutil.rmtree(OUT / "augmented", ignore_errors=True)
augmented = A.augment_dataset(
    windows, recipe, OUT / "augmented", copies=2, seed=0, progress=False
)
print(augmented)
augmented.summary()
""")
    nb.code("""
originals = len(windows[Split.TRAIN])
tv.viz.show_samples(augmented.with_split(Split.TRAIN, augmented[Split.TRAIN][originals:]), Split.TRAIN, n=4, cols=2)
""")

    nb.md(f"""
## Sources

{DEMO_ATTRIBUTION} In this notebook the frames are also cut to windows and
augmented.

The pitch keypoint frames are rendered by `tactifoot_vision`.
""")
    return nb
