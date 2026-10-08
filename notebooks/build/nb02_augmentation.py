"""02_augmentation.ipynb: `tv.augment` transforms, containers and offline augmentation."""

from common import Notebook


def notebook() -> Notebook:
    nb = Notebook("02_augmentation")
    nb.md("""
# Data augmentation with `tv.augment`

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

Part of the [TactiFoot Vision notebooks](README.md); datasets are introduced
in [01_data](01_data.ipynb).
""")
    nb.setup(
        imports="""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import tactifoot_vision as tv
from tactifoot_vision import augment as A
""",
        body="""
detection = tv.load_dataset(DATA / "datasets" / "football_yolo").subset({"train": 60, "valid": 20}, seed=0)
pitch_data = tv.load_dataset(DATA / "keypoints")
image, labels = detection.read("train", 3)
pitch_image, pitch_labels = pitch_data.read("train", 4)
print(detection)
print(pitch_data)
""",
    )
    nb.md("""
Every transform has a probability `p` of being applied. Calls take a seeded
`np.random.Generator`, so a result can be reproduced. A small helper draws
the labels before and after with `tv.viz.draw_annotations`.
""")
    nb.code("""
def before_after(transform, image, labels, class_names, seeds=(0,), width=12):
    images = [tv.viz.draw_annotations(image, labels, class_names)]
    titles = [f"original: {len(labels)} objects"]
    for seed in seeds:
        new_image, new_labels = transform(image, labels, np.random.default_rng(seed))
        images.append(tv.viz.draw_annotations(new_image, new_labels, class_names))
        titles.append(f"{type(transform).__name__} (seed {seed}): {len(new_labels)} objects")
    return tv.show(images, titles, cols=min(len(images), 3), width=width)
""")

    nb.md("""
## Geometric transforms

`HorizontalFlip` mirrors the image and its boxes. `p=1.0` forces it here; in
training a flip on half the images (the default `p=0.5`) is typical.
""")
    nb.code("""
before_after(A.HorizontalFlip(p=1.0), image, labels, detection.class_names, width=10)
""")
    nb.md("""
`RandomAffine` rotates and scales about the image centre, then shifts. Each
box becomes the bounding box of its four moved corners, clipped to the image.
Objects pushed mostly out of frame (less than `min_area_ratio` of the box
left) are dropped, so the object count can shrink.
""")
    nb.code("""
affine = A.RandomAffine(degrees=8, translate=0.2, scale=(0.7, 1.3), p=1.0)
before_after(affine, image, labels, detection.class_names, seeds=(0, 1, 2, 3, 4), width=14)
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
player = labels.boxes[list(labels.class_ids).index(detection.class_names.index("player"))]
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
before_after(recipe, image, labels, detection.class_names, seeds=(0, 1, 2, 3, 4), width=14)
""")

    nb.md("""
## Keypoints under flips and affine transforms

Pitch landmarks have names: keypoint 0 is the corner at one end, keypoint 24
the matching corner at the other end. After a mirror flip, the landmark seen
where keypoint 0 was is really keypoint 24. `HorizontalFlip` renames keypoints
with the dataset's `flip_idx` (new keypoint `i` is the mirrored old keypoint
`flip_idx[i]`); the dataset loads `flip_idx` from `data.yaml`.
""")
    nb.code("""
flip = A.HorizontalFlip(p=1.0)
before_after(flip, pitch_image, pitch_labels, pitch_data.class_names, width=10)
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
determinant), a mirror image reverses it. Annotations without `flip_idx` raise
an error instead of producing such labels silently.
""")
    nb.code("""
def mirrored(annotations, image_size=640):
    \"\"\"True when the labelled landmarks only fit a mirror image of the pitch.\"\"\"
    h = tv.pitch.HomographyEstimator().update(annotations.to_keypoints())
    centre = h[2] @ [image_size / 2, image_size / 2, 1.0]
    return bool(np.linalg.det(h) * centre < 0)


rng = np.random.default_rng(0)
_, renamed = flip(pitch_image, pitch_labels, rng)
_, not_renamed = flip(pitch_image, replace(pitch_labels, flip_idx=tuple(range(32))), rng)
_, moved = A.RandomAffine(degrees=10, translate=0.1, p=1.0)(pitch_image, pitch_labels, rng)
print("original mirrored:              ", mirrored(pitch_labels))
print("flip with flip_idx mirrored:    ", mirrored(renamed))
print("flip keeping old names mirrored:", mirrored(not_renamed))
print("affine mirrored:                ", mirrored(moved))

try:
    flip(pitch_image, replace(pitch_labels, flip_idx=None))
except ValueError as error:
    print("\\nwithout flip_idx:", error)
""")
    nb.md("""
The same check runs over a whole dataset, so a labelling error cannot slip into
training unnoticed. Every image in every split of the pitch dataset is tested.
""")
    nb.code("""
images = [(split, sample) for split in pitch_data.split_names for sample in pitch_data[split]]
flagged = [(split, sample.image_path.name) for split, sample in images if mirrored(sample.annotations)]
print(f"images mirrored: {len(flagged)} of {len(images)}")
print(flagged)
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
print("sample mirrored:       ", mirrored(pitch_labels))
print("mirrored copy mirrored:", mirrored(wrong_labels))

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
before_after(strong, pitch_image, pitch_labels, pitch_data.class_names, seeds=(1,), width=10)
""")

    nb.md("""
## Augmenting a dataset

`show_augmentations` previews a recipe on random dataset images, original and
augmented side by side, before committing to it.
""")
    nb.code("""
tv.viz.show_augmentations(pitch_data, recipe, n=2, seed=3)
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
import shutil

shutil.rmtree(OUT / "augmented", ignore_errors=True)
augmented = A.augment_dataset(
    detection, recipe, OUT / "augmented", copies=2, seed=0, progress=False
)
print(augmented)
augmented.summary()
""")
    nb.code("""
tv.viz.show_samples(augmented.with_split("train", augmented["train"][60:]), "train", n=3)
""")
    return nb
