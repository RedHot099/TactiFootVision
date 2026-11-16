"""One StatsBomb open-data match (events and 360 frames), fetched from StatsBomb's GitHub repository."""

import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

REPOSITORY_URL = "https://github.com/statsbomb/open-data"
_RAW_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
# Euro 2020 final, Italy v England: in the open data, with 360 frames.
DEFAULT_MATCH = 3795506
ATTEMPTS = 5
RETRY_WAIT_SECONDS = 15
_TIMEOUT_SECONDS = 60

# StatsBomb's terms: the repository README ("state the data source as StatsBomb
# and use our logo", from their media pack, https://statsbomb.com/media-pack/)
# and LICENSE.pdf, the Public Data User Agreement (1.4: accredit any publication
# of analysis formed from the data with the StatsBomb logo; 1.2.1: the data may
# not be redistributed to third parties, so it is fetched here, never re-hosted).
ATTRIBUTION = (
    f"Data: StatsBomb ({REPOSITORY_URL}), used under the StatsBomb Public Data "
    "User Agreement. Any publication of analysis based on it must credit "
    "StatsBomb and show the StatsBomb logo "
    "(https://statsbomb.com/media-pack/); the data may not be redistributed."
)

_FILES = {"events": "{m}_events.json", "three-sixty": "{m}_360.json"}


@dataclass(frozen=True)
class StatsBombMatch:
    """What :func:`download_statsbomb` leaves on disk.

    ``folder`` holds ``<match>_events.json`` and ``<match>_360.json``, the layout
    :func:`~tactifoot_vision.data.load_statsbomb` reads. ``attribution`` is the
    credit StatsBomb's terms ask for; show it, with the StatsBomb logo, wherever
    the data or an analysis of it is shown.
    """

    match: str
    folder: Path
    attribution: str = ATTRIBUTION


def download_statsbomb(
    match: int | str = DEFAULT_MATCH, root: str | Path = "data/statsbomb"
) -> StatsBombMatch:
    """Fetch one open-data match, its events and its 360 frames, into ``root/<match>``.

    The default is the Euro 2020 final. The source is StatsBomb's public
    repository (https://github.com/statsbomb/open-data); nothing is re-hosted,
    and its terms forbid passing the data on. The terms ask that any
    publication of analysis based on the data credits StatsBomb and shows
    their logo; the returned ``attribution`` is that credit and the call
    prints it. A file already in the folder is not fetched again, so a second
    call makes no request.

    A match that is missing from the open data, or has no 360 frames, is a
    ``ValueError`` (not every match has them); a failed download, after
    ``ATTEMPTS`` tries with a wait between them, a ``ConnectionError``.
    About 13 MB come down for the default match.
    """
    match = str(match)
    if not match.isdigit():
        raise ValueError(f"A StatsBomb match id is a number, got {match!r}")
    folder = Path(root) / match
    wanted = [
        (f"{_RAW_URL}/{source}/{match}.json", folder / name.format(m=match))
        for source, name in _FILES.items()
    ]
    missing = [(url, path) for url, path in wanted if not path.is_file()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        bar = tqdm(missing, desc=f"Downloading StatsBomb match {match}", unit="file")
        for url, path in bar:
            _fetch(url, path, match)
    else:
        logger.info("StatsBomb match %s is already in %s", match, folder)
    print(ATTRIBUTION)
    return StatsBombMatch(match=match, folder=folder)


def _fetch(url: str, path: Path, match: str) -> None:
    """Write ``url`` to ``path`` (through a part file, so a half download never counts)."""
    part = path.with_name(path.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
                part.write_bytes(response.read())
            part.replace(path)
            return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise ValueError(
                    f"StatsBomb match {match} has no {path.name} in the open data "
                    f"({url}); not every match is open or has 360 frames. See "
                    f"{REPOSITORY_URL}/tree/master/data/three-sixty for the ones that do."
                ) from error
            failure: Exception = error
        except OSError as error:  # network failures, timeouts
            failure = error
        if attempt == ATTEMPTS:
            raise ConnectionError(
                f"Could not download {url} after {ATTEMPTS} tries: {failure}"
            ) from failure
        logger.warning(
            "Download of %s failed (%s); try %d of %d in %d s",
            url,
            failure,
            attempt + 1,
            ATTEMPTS,
            RETRY_WAIT_SECONDS,
        )
        time.sleep(RETRY_WAIT_SECONDS)
