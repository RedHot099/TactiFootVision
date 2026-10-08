"""Photometric transforms: change pixel values only, never where objects are."""

from abc import abstractmethod

import cv2
import numpy as np

from tactifoot_vision.augment.base import Transform, _as_range
from tactifoot_vision.data.annotations import Annotations


class ImageOnlyTransform(Transform):
    """A transform that changes pixels but not geometry.

    The annotations are returned as the very same object (nothing about them
    changes). Subclasses implement :meth:`apply_image`.
    """

    def apply(
        self, image: np.ndarray, annotations: Annotations, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotations]:
        return self.apply_image(image, rng), annotations

    @abstractmethod
    def apply_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a new ``uint8`` image of the same shape."""


class ColorJitter(ImageOnlyTransform):
    """Random brightness, contrast, saturation and hue changes.

    Brightness, contrast and saturation factors are drawn from ``1 ± value``;
    the hue shift from ``± hue`` of the full colour circle.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.3,
        hue: float = 0.02,
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        for name, value in [
            ("brightness", brightness),
            ("contrast", contrast),
            ("saturation", saturation),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= hue <= 0.5:
            raise ValueError(f"hue must be in [0, 0.5], got {hue}")
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def apply_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        brightness = rng.uniform(1 - self.brightness, 1 + self.brightness)
        contrast = rng.uniform(1 - self.contrast, 1 + self.contrast)
        saturation = rng.uniform(1 - self.saturation, 1 + self.saturation)
        hue_shift = rng.uniform(-self.hue, self.hue) * 360.0

        pixels = np.clip(image.astype(np.float32) * (brightness / 255.0), 0.0, 1.0)
        mean = pixels.mean()
        pixels = np.clip((pixels - mean) * contrast + mean, 0.0, 1.0)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)  # float input: H in [0, 360)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
        return _to_uint8(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255.0)


class GaussianBlur(ImageOnlyTransform):
    """Gaussian blur with a random ``sigma`` (pixels) drawn from the ``(min, max)`` range."""

    def __init__(self, sigma: tuple[float, float] = (0.1, 1.5), p: float = 0.5) -> None:
        super().__init__(p)
        self.sigma = _as_range(sigma, "sigma")
        if self.sigma[0] <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

    def apply_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=rng.uniform(*self.sigma))


class MotionBlur(ImageOnlyTransform):
    """Blur along a line in a random direction, like a fast camera pan.

    The kernel length is a random odd number of pixels within ``kernel_size``
    (odd so the blur is centred and does not shift the image).
    """

    def __init__(self, kernel_size: tuple[int, int] = (3, 11), p: float = 0.5) -> None:
        super().__init__(p)
        low, high = (int(v) for v in _as_range(kernel_size, "kernel_size"))
        self.kernel_size = (low, high)
        self._sizes = [k for k in range(max(low, 3), high + 1) if k % 2 == 1]
        if not self._sizes:
            raise ValueError(
                f"kernel_size {kernel_size} contains no odd size >= 3 to blur with"
            )

    def apply_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = int(rng.choice(self._sizes))
        angle = rng.uniform(0.0, np.pi)
        centre = size // 2
        dx = round(np.cos(angle) * centre)
        dy = round(np.sin(angle) * centre)
        kernel = np.zeros((size, size), dtype=np.float32)
        cv2.line(kernel, (centre - dx, centre - dy), (centre + dx, centre + dy), 1.0)
        return cv2.filter2D(image, -1, kernel / kernel.sum())


class GaussianNoise(ImageOnlyTransform):
    """Additive per-pixel Gaussian noise; ``std`` range in 0-255 intensity units."""

    def __init__(self, std: tuple[float, float] = (2.0, 8.0), p: float = 0.5) -> None:
        super().__init__(p)
        self.std = _as_range(std, "std")
        if self.std[0] < 0:
            raise ValueError(f"std must be >= 0, got {std}")

    def apply_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = rng.uniform(*self.std)
        noise = rng.standard_normal(image.shape, dtype=np.float32) * std
        return _to_uint8(image + noise)


class JpegCompression(ImageOnlyTransform):
    """Re-encode as JPEG with a random quality in the ``(min, max)`` range (inclusive)."""

    def __init__(self, quality: tuple[int, int] = (40, 90), p: float = 0.5) -> None:
        super().__init__(p)
        low, high = (int(v) for v in _as_range(quality, "quality"))
        if low < 1 or high > 100:
            raise ValueError(f"quality must be within [1, 100], got {quality}")
        self.quality = (low, high)

    def apply_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        quality = int(rng.integers(self.quality[0], self.quality[1] + 1))
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(
                f"JPEG encoding failed for an image of shape {image.shape}"
            )
        return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)


def _to_uint8(pixels: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
