"""File helpers shared by the dataset writers."""

import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Written into every export folder, so a later export may safely clear it. It
# lists, relative to the export folder, the folders the export manages (lines
# ending in "/") and every file it writes; a later export deletes exactly these.
EXPORT_MARKER = ".tactifoot-export"
LINK_MODES = ("symlink", "hardlink", "copy")


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` read from the image header only."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def check_link_mode(mode: str) -> None:
    if mode not in LINK_MODES:
        raise ValueError(f"link must be 'symlink', 'hardlink' or 'copy', got {mode!r}")


def link_file(source: Path, target: Path, mode: str) -> None:
    """Place ``source`` at ``target`` as a symlink, hard link or copy."""
    check_link_mode(mode)
    if target.is_symlink() or target.exists():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source.resolve(), target)
        except OSError:  # e.g. across file systems
            shutil.copy2(source, target)
    else:
        shutil.copy2(source, target)


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

    Every source image must exist (``FileNotFoundError``), ``out_dir`` must not
    contain a source image (clearing it would delete the data), and no folder
    we write to may sit in a source image folder (reloading the source would
    pick the new files up). Paths are compared both as given and with symlinks
    resolved, since exports link to their sources.
    """
    source_images = list(source_images)
    missing = next((p for p in source_images if not p.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Source image not found: {missing}")
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


def prepare_output(
    out_dir: Path, managed: Sequence[Path], files: Iterable[Path]
) -> None:
    """Make ``out_dir`` ready for an export that writes ``files`` into ``managed`` folders.

    A non-empty ``out_dir`` must be an earlier export (it carries
    :data:`EXPORT_MARKER`); anything else is left alone with an error.

    An export only deletes or overwrites what an earlier export recorded in
    the marker: the folders it managed (a split the new export drops is
    removed with them) and the files it wrote. Before anything changes, a new
    file that already exists (a user's ``data.yaml`` next to a COCO export, or
    a symlink there) and a recorded folder holding any file the marker does
    not list (a user's notes, augmented images, ...) are left alone with an
    error naming the file. The marker is updated before the new files are
    written, so an interrupted export can be redone.
    """
    marker = out_dir / EXPORT_MARKER
    if out_dir.is_dir() and any(out_dir.iterdir()) and not marker.is_file():
        raise ValueError(
            f"{out_dir} is not empty and is not a tactifoot export; use a new folder"
        )
    old_folders, old_files = _read_marker(out_dir)
    files = list(files)
    for path in files:
        if (path.exists() or path.is_symlink()) and path not in old_files:
            raise ValueError(
                f"Refusing to overwrite {path}, which the export did not write; "
                "move it or use a new folder"
            )
    folders = list(dict.fromkeys([*managed, *old_folders]))
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_symlink() or not path.is_dir():
                if path not in old_files:
                    raise ValueError(
                        f"Refusing to clear {folder}: it holds {path}, which the "
                        "export did not write; move it or use a new folder"
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Until the old export is gone, the marker records both.
    _write_marker(out_dir, folders, [*old_files, *files])
    for path in old_files:
        if path.is_symlink() or path.is_file():
            path.unlink()
    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)
        parent = folder.parent
        while parent != out_dir and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    for folder in managed:
        folder.mkdir(parents=True)
    _write_marker(out_dir, managed, files)


def _read_marker(out_dir: Path) -> tuple[list[Path], set[Path]]:
    """The folders and files an earlier export in ``out_dir`` recorded."""
    marker = out_dir / EXPORT_MARKER
    lines = marker.read_text().splitlines() if marker.is_file() else []
    folders = [out_dir / line.rstrip("/") for line in lines if line.endswith("/")]
    files = {out_dir / line for line in lines if line and not line.endswith("/")}
    return folders, files


def _write_marker(
    out_dir: Path, folders: Iterable[Path], files: Iterable[Path]
) -> None:
    names = [f"{p.relative_to(out_dir).as_posix()}/" for p in folders]
    names += [p.relative_to(out_dir).as_posix() for p in files]
    (out_dir / EXPORT_MARKER).write_text("".join(f"{name}\n" for name in names))


def clipped_box(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """``xyxy`` box clipped to the image, or ``None`` if nothing of it is left."""
    clipped = np.clip(box, 0, [width, height, width, height])
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _both(path: Path) -> set[Path]:
    return {path.absolute(), path.resolve()}
