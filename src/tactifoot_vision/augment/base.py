"""Transform protocol and the containers that combine transforms."""

import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from tactifoot_vision.data.annotations import Annotations


class Transform(ABC):
    """An augmentation applied to an image and its annotations with probability ``p``.

    Call it as ``image, annotations = transform(image, annotations, rng)``.
    Images are ``uint8`` BGR ``H x W x 3`` arrays; annotations are in pixel
    coordinates of that image.

    Inputs are never modified. The returned objects may *be* the inputs when a
    transform leaves them unchanged (skipped by ``p``, or the annotations of a
    pixel-only transform), so copy them before editing in place.

    Pass a seeded ``np.random.Generator`` for reproducible results; without one
    a fresh, unseeded generator is used.

    Subclasses implement :meth:`apply`.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def __call__(
        self,
        image: np.ndarray,
        annotations: Annotations,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, Annotations]:
        rng = np.random.default_rng() if rng is None else rng
        if rng.random() >= self.p:
            return image, annotations
        return self.apply(image, annotations, rng)

    @abstractmethod
    def apply(
        self, image: np.ndarray, annotations: Annotations, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotations]:
        """Transform unconditionally (``p`` is handled by ``__call__``)."""

    def __repr__(self) -> str:
        params = {
            k: v for k, v in vars(self).items() if not k.startswith("_") and k != "p"
        }
        args = ", ".join(f"{k}={v!r}" for k, v in (params | {"p": self.p}).items())
        return f"{type(self).__name__}({args})"


class Compose(Transform):
    """Apply ``transforms`` in order (each with its own ``p``)."""

    def __init__(self, transforms: Sequence[Transform], p: float = 1.0) -> None:
        super().__init__(p)
        self.transforms = list(transforms)

    def apply(
        self, image: np.ndarray, annotations: Annotations, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotations]:
        for transform in self.transforms:
            image, annotations = transform(image, annotations, rng)
        return image, annotations

    def __repr__(self) -> str:
        return _container_repr(self, self.transforms)


class OneOf(Transform):
    """Pick one of ``transforms`` uniformly at random; the picked one still applies its own ``p``."""

    def __init__(self, transforms: Sequence[Transform], p: float = 1.0) -> None:
        super().__init__(p)
        self.transforms = list(transforms)
        if not self.transforms:
            raise ValueError("OneOf needs at least one transform")

    def apply(
        self, image: np.ndarray, annotations: Annotations, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotations]:
        transform = self.transforms[rng.integers(len(self.transforms))]
        return transform(image, annotations, rng)

    def __repr__(self) -> str:
        return _container_repr(self, self.transforms)


def _container_repr(container: Transform, transforms: list[Transform]) -> str:
    name = type(container).__name__
    if not transforms:
        return f"{name}([], p={container.p})"
    children = "".join(textwrap.indent(repr(t), "    ") + ",\n" for t in transforms)
    return f"{name}([\n{children}], p={container.p})"


def _as_range(value: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate a ``(min, max)`` parameter range."""
    low, high = (float(v) for v in value)
    if low > high:
        raise ValueError(f"{name} must be (min, max) with min <= max, got {value}")
    return low, high
