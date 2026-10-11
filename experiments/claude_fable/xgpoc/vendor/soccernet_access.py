"""Access to the SoccerNet server for the re-download (xg-vendor ticket 12): the password and the half downloader.

The password is the owner's NDA password. It is read from `SOCCERNET_PASSWORD` or from the private file
`~/.config/soccernet/password` (mode 600, outside the repo) and lives only inside a `Secret`, which prints as hidden.
It is never a command-line argument (shell history, process list) and never reaches a log or a record: every text the
downloader hands back is scrubbed, and the original exception is not chained to the one it raises.

`password_source` tells where a password would come from without reading the file (the dry run uses it, so it makes
no network call and opens no secret).

The real downloader wraps `SoccerNet.Downloader.SoccerNetDownloader` and is imported lazily, so merely planning or
testing never loads it. Two properties of that library shape the adapter:
 * `downloadFile` skips a path that exists and `downloadGame` ignores its return codes (a failed request leaves no
   file), so each half goes into its own fresh directory and a missing file is the error signal;
 * every successful download pings Google Analytics (`google_measurement_protocol.report`). That ping carries the file
   name and nothing the study needs, so it is switched off for the lifetime of this process.
"""

import importlib
import logging
import os
import socket
import stat
import tempfile
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

PASSWORD_ENV = "SOCCERNET_PASSWORD"
PASSWORD_FILE = Path.home() / ".config/soccernet/password"
HIDDEN = "***"
SOCKET_TIMEOUT_S = 120   # a stalled read raises instead of hanging forever (as in scripts/collect_soccernet.py)


class PasswordError(RuntimeError):
    """No usable password. The message never contains any part of it."""


class DownloadError(RuntimeError):
    """A half could not be fetched. The message is scrubbed of the password."""


class Secret:
    """A string that does not show itself: `repr`, `str` and pickling hide it, `reveal` is the one way out."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        if not value:
            raise PasswordError("The password is empty")
        object.__setattr__(self, "_value", value)

    def __setattr__(self, name, value):
        raise AttributeError("Secret is immutable")

    def __repr__(self) -> str:
        return "Secret(<hidden>)"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("Secret cannot be pickled")

    def reveal(self) -> str:
        return self._value

    def scrub(self, text: str) -> str:
        """`text` with the password, and its URL-quoted form, replaced by `***`."""
        value = self._value
        for form in sorted({value, quote(value, safe=""), quote(value)}, key=len, reverse=True):
            text = text.replace(form, HIDDEN)
        return text


class ScrubFilter(logging.Filter):
    """Handler filter that removes the password from every message and traceback before it is formatted."""

    def __init__(self, secret: Secret):
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = self.secret.scrub(record.getMessage()), None
        if record.exc_info:
            record.exc_text = self.secret.scrub("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        return True


def password_source(environ: Mapping[str, str] = os.environ, path: Path = PASSWORD_FILE) -> str | None:
    """`env`, `file` or None. Reads only whether the variable is set and whether the file is non-empty."""
    if environ.get(PASSWORD_ENV):
        return "env"
    try:
        return "file" if path.stat().st_size > 0 else None
    except OSError:
        return None


def load_password(environ: Mapping[str, str] = os.environ, path: Path = PASSWORD_FILE) -> Secret:
    """The password from the environment, else from the private file (refused unless only its owner can read it)."""
    if value := environ.get(PASSWORD_ENV, "").strip():
        return Secret(value)
    try:
        mode = path.stat().st_mode
    except OSError:
        raise PasswordError(f"No password: {PASSWORD_ENV} is not set and {path} does not exist") from None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PasswordError(f"{path} must be readable by its owner only (chmod 600)")
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise PasswordError(f"Cannot read {path}: {type(exc).__name__}") from None
    if not value:
        raise PasswordError(f"{path} is empty")
    return Secret(value)


class HalfDownloader(Protocol):
    def download(self, game: str, split: str, half: int, dest: Path) -> None:
        """Write the 720p half to `dest`, or raise `DownloadError`. A file at `dest` is complete."""


def half_file_name(half: int) -> str:
    return f"{half}_720p.mkv"


class SoccerNetHalfDownloader:
    """The SoccerNet library behind the `HalfDownloader` protocol."""

    def __init__(self, password: Secret, staging: Path, timeout_s: float = SOCKET_TIMEOUT_S):
        self.password = password
        self.staging = staging
        self.timeout_s = timeout_s

    def download(self, game: str, split: str, half: int, dest: Path) -> None:
        library = importlib.import_module("SoccerNet.Downloader")   # lazy: loads boto3 and huggingface_hub
        library.report = lambda *args, **kwargs: None   # no analytics ping per file
        socket.setdefaulttimeout(self.timeout_s)
        name = half_file_name(half)
        self.staging.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.staging, prefix="dl_") as fresh:
            failure = None
            try:
                downloader = library.SoccerNetDownloader(LocalDirectory=fresh)
                downloader.password = self.password.reveal()
                downloader.downloadGame(game, files=[name], spl=split, verbose=False)
            except Exception as exc:  # noqa: BLE001 - any failure of the library, with the password removed
                failure = self.password.scrub(f"{type(exc).__name__}: {exc}")[:300]
            if failure:   # raised outside the except block: nothing of the original exception stays attached
                raise DownloadError(failure)
            fetched = Path(fresh) / game / name
            if not fetched.is_file():
                raise DownloadError(f"No file after the download request for {game} half {half} "
                                    "(HTTP error, wrong password or unknown game)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(fetched, dest)
