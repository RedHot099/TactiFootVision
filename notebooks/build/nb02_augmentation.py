"""02_augmentation.ipynb: `tv.augment` transforms, containers and offline augmentation."""

from common import Notebook


def notebook() -> Notebook:
    nb = Notebook("02_augmentation")
    nb.md("""
# Data augmentation with `tv.augment`

Augmentation makes more training images from the ones you have: mirrored,
rotated, zoomed, recoloured, blurred. A transform takes and returns
`(image, annotations)`, so boxes move with the pixels and the labels stay
correct. (The pitch line model augments its own training images: a zoom, a
mirror that renames the markings, colour and blur; see
[03_training_and_evaluation](03_training_and_evaluation.ipynb).)

* Geometric transforms (`HorizontalFlip`, `RandomAffine`) move pixels and
  boxes together.
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
from pathlib import Path

import numpy as np

import tactifoot_vision as tv
from tactifoot_vision import augment as A
""",
        body="""
detection = tv.load_dataset(DATA / "datasets" / "football_yolo").subset({"train": 60, "valid": 20}, seed=0)
image, labels = detection.read("train", 3)
print(detection)
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
## Augmenting a dataset

`show_augmentations` previews a recipe on random dataset images, original and
augmented side by side, before committing to it.
""")
    nb.code("""
tv.viz.show_augmentations(detection, recipe, n=2, seed=3)
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
