"""Small helpers shared across modules."""

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Send the package's log records to stderr at ``level``.

    The library itself only creates loggers; call this from scripts or notebooks
    when you want to see progress messages.
    """
    logger = logging.getLogger("tactifoot_vision")
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    if not any(getattr(h, "_tactifoot", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        handler._tactifoot = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def require_extra(
    module: str, extra: str | None = None, *, install: str | None = None, hint: str = ""
) -> ModuleType:
    """Import ``module``, or raise ``ImportError`` that says how to install it.

    ``extra`` names the package extra that provides it (``pip install
    "tactifoot-vision[extra]"``); ``install`` is the command when there is no
    extra. ``hint`` is appended to the message. A module that is installed
    but broken raises its own error.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module.partition(".")[0]:
            raise  # installed, but something inside it is missing: its own error
        command = install or f'pip install "tactifoot-vision[{extra}]"'
        raise ImportError(
            f"{module} is not installed: run `{command}`{hint}"
        ) from error


def resolve_device(device: str | None = None) -> str:
    """Return ``device`` or the best available one (``cuda`` if present, else ``cpu``)."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(name: str) -> Path:
    """``~/.cache/tactifoot_vision/<name>``, created; downloaded weights live here."""
    return ensure_dir(Path.home() / ".cache" / "tactifoot_vision" / name)


def next_run_dir(path: Path, exist_ok: bool = False) -> Path:
    """Create and return ``path``, or ``path2``, ``path3``, ... if it exists.

    With ``exist_ok`` an existing ``path`` is reused (Ultralytics' convention).
    """
    candidate, n = path, 2
    while candidate.exists() and not exist_ok:
        candidate, n = path.with_name(f"{path.name}{n}"), n + 1
    return ensure_dir(candidate)
