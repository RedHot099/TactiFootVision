import cv2
import numpy as np
import pytest

from tactifoot_vision.augment import (
    ColorJitter,
    Compose,
    GaussianBlur,
    GaussianNoise,
    HorizontalFlip,
    JpegCompression,
    MotionBlur,
    OneOf,
    RandomAffine,
    Transform,
    augment_dataset,
)
from tactifoot_vision.data import (
    NOT_LABELLED,
    OCCLUDED,
    VISIBLE,
    Annotations,
    Dataset,
    Sample,
)

W, H = 100, 80


def make_image(seed: int = 0, width: int = W, height: int = H) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )


def box_annotations() -> Annotations:
    return Annotations(boxes=[[10, 20, 30, 50], [60, 5, 90, 25]], class_ids=[0, 1])


def pose_annotations(flip_idx: tuple[int, ...] | None = (1, 0, 3, 2)) -> Annotations:
    keypoints = [
        [[10, 20, VISIBLE], [80, 30, OCCLUDED], [30, 60, VISIBLE], [0, 0, NOT_LABELLED]]
    ]
    return Annotations(
        boxes=[[5, 10, 90, 70]], class_ids=[0], keypoints=keypoints, flip_idx=flip_idx
    )


def assert_same_annotations(a: Annotations, b: Annotations, atol: float = 1e-4) -> None:
    np.testing.assert_allclose(a.boxes, b.boxes, atol=atol)
    np.testing.assert_array_equal(a.class_ids, b.class_ids)
    if a.keypoints is None or b.keypoints is None:
        assert a.keypoints is None and b.keypoints is None
    else:
        np.testing.assert_allclose(a.keypoints, b.keypoints, atol=atol)
    assert a.flip_idx == b.flip_idx


# ---------------------------------------------------------------- base


class Record(Transform):
    """Test helper: remembers how often it was applied."""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p)
        self.calls = 0

    def apply(self, image, annotations, rng):
        self.calls += 1
        return image, annotations


def test_probability_zero_and_one():
    image, annotations = make_image(), box_annotations()
    never, always = Record(p=0.0), Record(p=1.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out_image, out_annotations = never(image, annotations, rng)
        always(image, annotations, rng)
    assert out_image is image and out_annotations is annotations
    assert never.calls == 0 and always.calls == 20
    with pytest.raises(ValueError):
        Record(p=1.5)


def test_compose_runs_in_order_and_one_of_picks_exactly_one():
    first, second = Record(), Record()
    Compose([first, second])(make_image(), box_annotations())
    assert first.calls == second.calls == 1

    a, b = Record(), Record()
    one_of = OneOf([a, b])
    rng = np.random.default_rng(0)
    for _ in range(100):
        one_of(make_image(), box_annotations(), rng)
    assert a.calls + b.calls == 100 and a.calls > 20 and b.calls > 20
    with pytest.raises(ValueError):
        OneOf([])


def test_repr_shows_parameters():
    text = repr(Compose([HorizontalFlip(), OneOf([GaussianBlur(sigma=(1, 2))])]))
    assert "HorizontalFlip(p=0.5)" in text
    assert "GaussianBlur(sigma=(1.0, 2.0), p=0.5)" in text
    assert "RandomAffine(degrees=5.0, translate=0.1" in repr(RandomAffine())


# ------------------------------------------------------------ geometric


def test_flip_moves_boxes_and_permutes_keypoints():
    image, annotations = make_image(), pose_annotations()
    out_image, out = HorizontalFlip(p=1)(image, annotations)

    np.testing.assert_array_equal(out_image, image[:, ::-1])
    np.testing.assert_allclose(out.boxes, [[W - 90, 10, W - 5, 70]])
    # New keypoint i is the mirrored old keypoint flip_idx[i].
    np.testing.assert_allclose(
        out.keypoints[0],
        [[W - 80, 30, OCCLUDED], [W - 10, 20, VISIBLE], [W - 0, 0, 0], [W - 30, 60, 2]],
    )
    assert_same_annotations(annotations, pose_annotations())  # input untouched


def test_flip_twice_is_identity():
    flip = HorizontalFlip(p=1)
    image, annotations = flip(*flip(make_image(), pose_annotations()))
    np.testing.assert_array_equal(image, make_image())
    assert_same_annotations(annotations, pose_annotations())


def test_flip_without_flip_idx():
    with pytest.raises(ValueError, match="flip_idx"):
        HorizontalFlip(p=1)(make_image(), pose_annotations(flip_idx=None))
    with pytest.raises(ValueError, match="flip_idx"):
        HorizontalFlip(p=1)(make_image(), pose_annotations(flip_idx=(1, 0)))

    _, out = HorizontalFlip(p=1)(make_image(), box_annotations())
    np.testing.assert_allclose(
        out.boxes, [[W - 30, 20, W - 10, 50], [W - 90, 5, W - 60, 25]]
    )


@pytest.mark.parametrize("annotations", [box_annotations(), pose_annotations()])
def test_identity_affine_keeps_image_and_annotations(annotations):
    identity = RandomAffine(degrees=0, translate=0, scale=(1, 1), p=1)
    image = make_image()
    out_image, out = identity(image, annotations)
    np.testing.assert_array_equal(out_image, image)
    assert_same_annotations(out, annotations)


def test_rotation_and_scale_move_keypoints_like_cv2_transform():
    annotations = Annotations(
        boxes=[[0, 0, W, H]],
        class_ids=[0],
        keypoints=[[[50, 40, VISIBLE], [30, 30, VISIBLE], [70, 55, OCCLUDED]]],
    )
    affine = RandomAffine(degrees=(20, 20), translate=0, scale=(1.3, 1.3), p=1)
    _, out = affine(make_image(), annotations)

    matrix = cv2.getRotationMatrix2D((W / 2, H / 2), 20, 1.3)
    expected = cv2.transform(annotations.keypoints[:, :, :2], matrix)
    np.testing.assert_allclose(out.keypoints[..., :2], expected, atol=1e-4)
    np.testing.assert_array_equal(out.keypoints[..., 2], annotations.keypoints[..., 2])


@pytest.mark.parametrize(("degrees", "scale"), [(30, 1.0), (20, 1.3), (-10, 0.8)])
def test_pixels_move_with_keypoints(degrees, scale):
    # The centroid of a bright square must land exactly where its keypoint
    # lands (a half-pixel convention mismatch would be off by ~0.3 px).
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[18:23, 28:33] = 255  # centre of pixel (30, 20) = (30.5, 20.5)
    annotations = Annotations(
        boxes=[[0, 0, W, H]], class_ids=[0], keypoints=[[[30.5, 20.5, VISIBLE]]]
    )
    affine = RandomAffine(
        degrees=(degrees, degrees),
        translate=0,
        scale=(scale, scale),
        p=1,
        border_value=(0, 0, 0),
    )
    out_image, out = affine(image, annotations)
    weight = out_image[..., 0].astype(float)
    ys, xs = np.mgrid[0:H, 0:W]
    centroid = [(weight * xs).sum() / weight.sum(), (weight * ys).sum() / weight.sum()]
    np.testing.assert_allclose(
        out.keypoints[0, 0, :2], np.add(centroid, 0.5), atol=0.05
    )


def test_affine_clips_boxes_and_drops_mostly_outside_objects():
    # Zoom x2 about the centre (50, 40): x' = 2x - 50, y' = 2y - 40.
    annotations = Annotations(
        boxes=[
            [30, 25, 40, 35],  # -> [10, 10, 30, 30], inside
            [60, 30, 80, 50],  # -> [70, 20, 110, 60], clipped keeps 75 %
            [70, 30, 90, 50],  # -> [90, 20, 130, 60], clipped keeps 25 %
            [85, 70, 95, 78],  # -> fully outside
        ],
        class_ids=[0, 1, 2, 3],
    )
    zoom = RandomAffine(degrees=0, translate=0, scale=(2, 2), p=1, min_area_ratio=0.3)
    _, out = zoom(make_image(), annotations)
    np.testing.assert_array_equal(out.class_ids, [0, 1])
    np.testing.assert_allclose(
        out.boxes, [[10, 10, 30, 30], [70, 20, 100, 60]], atol=1e-4
    )


def test_keypoints_leaving_the_frame_become_not_labelled():
    annotations = Annotations(
        boxes=[[10, 10, 90, 70]],
        class_ids=[0],
        keypoints=[
            [[50, 40, VISIBLE], [20, 20, VISIBLE], [40, 50, OCCLUDED], [80, 40, 0]]
        ],
    )
    zoom = RandomAffine(degrees=0, translate=0, scale=(2, 2), p=1, min_area_ratio=0.1)
    _, out = zoom(make_image(), annotations)
    np.testing.assert_allclose(
        out.keypoints[0],
        [[50, 40, VISIBLE], [-10, 0, 0], [30, 60, OCCLUDED], [110, 40, 0]],
    )


# ----------------------------------------------------------- photometric

PHOTOMETRIC = [
    ColorJitter(p=1),
    GaussianBlur(p=1),
    MotionBlur(p=1),
    GaussianNoise(p=1),
    JpegCompression(p=1),
]


@pytest.mark.parametrize("transform", PHOTOMETRIC, ids=lambda t: type(t).__name__)
def test_photometric_changes_pixels_only(transform):
    image, annotations = make_image(), pose_annotations()
    out_image, out = transform(image, annotations, np.random.default_rng(1))
    assert out_image.shape == image.shape and out_image.dtype == np.uint8
    assert not np.array_equal(out_image, image)
    np.testing.assert_array_equal(image, make_image())  # input untouched
    assert_same_annotations(out, pose_annotations())


def test_motion_blur_keeps_a_uniform_image():
    image = np.full((H, W, 3), 77, dtype=np.uint8)
    for seed in range(5):
        out, _ = MotionBlur(p=1)(image, box_annotations(), np.random.default_rng(seed))
        np.testing.assert_array_equal(out, image)


def full_pipeline() -> Compose:
    return Compose(
        [
            HorizontalFlip(),
            RandomAffine(),
            OneOf([GaussianBlur(), MotionBlur()]),
            ColorJitter(),
            GaussianNoise(),
            JpegCompression(),
        ]
    )


def test_same_seed_same_output_and_inputs_untouched():
    image, annotations = make_image(), pose_annotations()
    runs = [
        full_pipeline()(image, annotations, np.random.default_rng(7)) for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert_same_annotations(runs[0][1], runs[1][1], atol=0)
    np.testing.assert_array_equal(image, make_image())
    assert_same_annotations(annotations, pose_annotations(), atol=0)

    outputs = {
        full_pipeline()(image, annotations, np.random.default_rng(s))[0].tobytes()
        for s in range(5)
    }
    assert len(outputs) > 1


# ------------------------------------------------------ augment_dataset


def make_dataset(tmp_path, pose: bool = False) -> Dataset:
    def sample(name: str, seed: int) -> Sample:
        path = tmp_path / "src" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), make_image(seed))
        return Sample(path, W, H, pose_annotations() if pose else box_annotations())

    return Dataset(
        task="pose" if pose else "detect",
        class_names=["a", "b"],
        splits={
            "train": [sample("img0", 0), sample("img1", 1)],
            "valid": [sample("val0", 2)],
        },
        num_keypoints=4 if pose else None,
        flip_idx=(1, 0, 3, 2) if pose else None,
        name="toy",
    )


def test_augment_dataset_writes_images_and_keeps_valid(tmp_path):
    dataset = make_dataset(tmp_path)
    out = augment_dataset(
        dataset, full_pipeline(), tmp_path / "aug", copies=2, image_format="png"
    )

    assert out.name == "toy-aug"
    assert len(out["train"]) == len(dataset["train"]) * (1 + 2)
    assert all(a is b for a, b in zip(out["train"][:2], dataset["train"], strict=True))
    assert len(out["valid"]) == 1 and out["valid"][0] is dataset["valid"][0]
    written = sorted(p.name for p in (tmp_path / "aug" / "train" / "images").iterdir())
    assert written == [
        "img0_aug0.png",
        "img0_aug1.png",
        "img1_aug0.png",
        "img1_aug1.png",
    ]
    assert not (tmp_path / "aug" / "valid").exists()
    for sample in out["train"][2:]:
        assert sample.image_path.is_file()
        assert sample.read_image().shape == (H, W, 3)
    assert len(dataset["train"]) == 2  # source dataset untouched


def test_augment_dataset_is_deterministic(tmp_path):
    dataset = make_dataset(tmp_path, pose=True)
    first = augment_dataset(dataset, full_pipeline(), tmp_path / "a", seed=3)
    second = augment_dataset(dataset, full_pipeline(), tmp_path / "b", seed=3)
    for x, y in zip(first["train"], second["train"], strict=True):
        np.testing.assert_array_equal(x.read_image(), y.read_image())
        assert_same_annotations(x.annotations, y.annotations, atol=0)


def test_augment_dataset_fills_flip_idx_and_rejects_bad_arguments(tmp_path):
    dataset = make_dataset(tmp_path, pose=True)
    bare = [
        Sample(s.image_path, s.width, s.height, pose_annotations(flip_idx=None))
        for s in dataset["train"]
    ]
    out = augment_dataset(
        dataset.with_split("train", bare), HorizontalFlip(p=1), tmp_path / "aug"
    )
    np.testing.assert_allclose(out["train"][2].annotations.keypoints[0, 0], [20, 30, 1])

    with pytest.raises(ValueError, match="test"):
        augment_dataset(dataset, HorizontalFlip(), tmp_path / "x", splits=("test",))
    with pytest.raises(ValueError, match="copies"):
        augment_dataset(dataset, HorizontalFlip(), tmp_path / "x", copies=0)


def test_augment_dataset_keeps_duplicate_stems_apart(tmp_path):
    dataset = make_dataset(tmp_path)
    other = tmp_path / "other" / "img0.png"
    other.parent.mkdir()
    cv2.imwrite(str(other), make_image(9))
    train = [*dataset["train"], Sample(other, W, H, box_annotations())]
    out = augment_dataset(
        dataset.with_split("train", train), HorizontalFlip(), tmp_path / "aug"
    )
    names = [s.image_path.name for s in out["train"][3:]]
    assert names == ["img0_aug0.jpg", "img1_aug0.jpg", "img0_1_aug0.jpg"]


def test_augment_dataset_refuses_to_write_into_the_source_folders(tmp_path):
    dataset = make_dataset(tmp_path)
    source_root = dataset["train"][0].image_path.parent
    with pytest.raises(ValueError, match="Refusing"):
        augment_dataset(dataset, HorizontalFlip(p=1.0), source_root)
    with pytest.raises(ValueError, match="Refusing"):
        augment_dataset(dataset, HorizontalFlip(p=1.0), source_root.parent)


def test_augment_dataset_refuses_an_export_folder(tmp_path):
    dataset = make_dataset(tmp_path)
    dataset.to_yolo(tmp_path / "export")
    with pytest.raises(ValueError, match="export"):
        augment_dataset(dataset, HorizontalFlip(), tmp_path / "export", progress=False)
    assert not list((tmp_path / "export" / "train" / "images").glob("*_aug*"))


def test_augment_dataset_progress_false_is_silent_for_every_split(tmp_path, capsys):
    augment_dataset(
        make_dataset(tmp_path), Compose([]), tmp_path / "aug",
        splits=("train", "valid"), progress=False,
    )  # fmt: skip
    assert capsys.readouterr().err == ""
