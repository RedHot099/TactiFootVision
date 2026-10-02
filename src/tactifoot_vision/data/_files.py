"""File helpers shared by the dataset writers."""

import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Written into every export folder, so a later export may safely clear it.
EXPORT_MARKER = ".tactifoot-export"


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` read from the image header only."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def link_file(source: Path, target: Path, mode: str) -> None:
    """Place ``source`` at ``target`` as a symlink, hard link or copy."""
    if target.is_symlink() or target.exists():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source.resolve(), target)
        except OSError:  # e.g. across file systems
            shutil.copy2(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"link must be 'symlink', 'hardlink' or 'copy', got {mode!r}")


def unique_stems(stems: Iterable[str]) -> list[str]:
    """Suffix repeated stems with ``_1``, ``_2``, ... (YOLO labels are keyed by stem)."""
    seen: set[str] = set()
    unique = []
    for stem in stems:
        candidate, counter = stem, 1
        while candidate in seen:
            candidate, counter = f"{stem}_{counter}", counter + 1
        seen.add(candidate)
        unique.append(candidate)
    return unique


def unique_names(paths: Sequence[Path]) -> list[str]:
    """File names for ``paths`` in one folder, with no two sharing a stem."""
    stems = unique_stems(p.stem for p in paths)
    return [stem + path.suffix for stem, path in zip(stems, paths, strict=True)]


def check_output_location(
    out_dir: Path, written_dirs: Iterable[Path], source_images: Iterable[Path]
) -> None:
    """Refuse outputs that could destroy or pollute the source dataset.

    ``out_dir`` must not contain a source image (clearing it would delete the
    data), and no folder we write to may sit in a source image folder (reloading
    the source would pick the new files up). Paths are compared both as given
    and with symlinks resolved, since exports link to their sources.
    """
    source_dirs = {p for image in source_images for p in _both(image.parent)}
    outs = _both(out_dir)
    targets = [p for folder in written_dirs for p in _both(folder)]
    for source in source_dirs:
        if any(out == source or out in source.parents for out in outs):
            raise ValueError(
                f"Refusing to write into {out_dir}: it contains source images ({source})"
            )
        if any(target == source or source in target.parents for target in targets):
            raise ValueError(
                f"Refusing to write into {out_dir}: it lies inside the source image folder {source}"
            )


def prepare_output(out_dir: Path, managed: Iterable[Path]) -> None:
    """Create ``out_dir`` and empty the sub-folders an export is about to rewrite.

    A non-empty ``out_dir`` must be an earlier export (it carries
    :data:`EXPORT_MARKER`); anything else is left alone with an error.
    """
    if (
        out_dir.is_dir()
        and any(out_dir.iterdir())
        and not (out_dir / EXPORT_MARKER).is_file()
    ):
        raise ValueError(
            f"{out_dir} is not empty and is not a tactifoot export; use a new folder"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EXPORT_MARKER).touch()
    for folder in managed:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)


def clipped_box(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """``xyxy`` box clipped to the image, or ``None`` if nothing of it is left."""
    clipped = np.clip(box, 0, [width, height, width, height])
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _both(path: Path) -> set[Path]:
    return {path.absolute(), path.resolve()}
