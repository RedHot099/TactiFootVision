"""Fetch StatsBomb open data at one pinned commit into a resumable raw store.

The open-data repository (https://github.com/statsbomb/open-data) is read at a single commit:
`resolve_sha` turns "master" into a commit sha once, and every file comes from
raw.githubusercontent.com/<repo>/<sha>/data/... A store records its sha in `pin.json` and
refuses another one, so a build never mixes commits.

Layout under the store root:

    pin.json                          repository and commit sha
    raw/competitions.json.gz          upstream bytes, gzip (mtime 0, so equal input gives equal file)
    raw/matches/<cid>_<sid>.json.gz   match list of one competition season
    raw/events/<match_id>.json.gz     full event JSON of one match
    raw/terms/README.md, LICENSE.pdf  the repository's terms at the pinned commit (stored as is)
    <file>.meta.json                  per file: commit, upstream path, sha256 of the upstream bytes and of the stored file

A file counts as downloaded only when its meta file is present and the stored file still matches
the recorded hash; the meta file is written last. A rerun therefore skips verified files, refetches
damaged ones and finishes interrupted ones. Network errors are retried with exponential backoff
(4xx answers other than 408/429 are permanent and not retried). After all downloads, any expected
match that is still missing raises `MissingMatchesError` with the list.

`fetch` is injected everywhere (`Fetch = Callable[[str], bytes]`), so tests never touch the network.
"""

import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

Fetch = Callable[[str], bytes]

REPO = "statsbomb/open-data"
COMMIT_URL = f"https://api.github.com/repos/{REPO}/commits/master"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{{sha}}/{{path}}"
ALLOWED_HOSTS = ("api.github.com", "raw.githubusercontent.com")
USER_AGENT = "tactifoot-vision-research (statsbomb open-data fetch)"
MAX_WORKERS = 4
ATTEMPTS = 5
BACKOFF_S = 2.0
TERMS_FILES = ("README.md", "LICENSE.pdf")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# (competition_id, season_id) of the 10 competition seasons the prototype's build covered: 2015/16 of the Premier League,
# Serie A, Bundesliga and Ligue 1, La Liga 2014/15 to 2016/17 and the Champions League 2014/15 to 2016/17 (1,626 matches).
COVERED = ((11, 26), (11, 27), (11, 2), (12, 27), (16, 26), (16, 27), (16, 2), (2, 27), (7, 27), (9, 27))


class PermanentFetchError(Exception):
    """The server answered with an error that a retry cannot fix (HTTP 404, 403, ...)."""


class FetchFailed(Exception):
    """All attempts to fetch one URL failed."""


class MissingMatchesError(RuntimeError):
    """Expected matches whose events could not be stored; `missing` maps match id to the last error."""

    def __init__(self, missing: Mapping[int, str]):
        self.missing = dict(missing)
        listing = "; ".join(f"{m}: {e}" for m, e in sorted(self.missing.items()))
        super().__init__(f"{len(self.missing)} expected matches are missing: {listing}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def http_fetch(url: str, timeout: float = 60) -> bytes:
    """GET `url` (GitHub hosts only). 4xx answers other than 408/429 raise PermanentFetchError."""
    if urllib.parse.urlsplit(url).hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Only {ALLOWED_HOSTS} may be fetched, not {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500 and exc.code not in (408, 429):
            raise PermanentFetchError(f"HTTP {exc.code} for {url}") from exc
        raise


def resolve_sha(fetch: Fetch = http_fetch) -> str:
    """The commit sha master points to now (one API call; the answer is then pinned)."""
    sha = json.loads(fetch(COMMIT_URL))["sha"]
    if not SHA_RE.match(sha):
        raise ValueError(f"Not a commit sha: {sha!r}")
    return sha


@dataclass(frozen=True)
class Source:
    """A fetch function bound to one commit and a retry policy."""

    fetch: Fetch
    sha: str
    attempts: int = ATTEMPTS
    backoff_s: float = BACKOFF_S
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self):
        if not SHA_RE.match(self.sha):
            raise ValueError(f"Not a commit sha: {self.sha!r}")

    def get(self, path: str, validate: Callable[[bytes], None] | None = None) -> bytes:
        """Upstream bytes of `path`. Network errors and a body that fails `validate` (a ValueError,
        e.g. truncated JSON) are retried with backoff 1x, 2x, 4x ... `backoff_s`."""
        url = RAW_URL.format(sha=self.sha, path=path)
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                body = self.fetch(url)
                if validate is not None:
                    validate(body)
                return body
            except PermanentFetchError:
                raise
            except Exception as exc:  # noqa: BLE001 - any transport or parse problem is retried
                last = exc
                if attempt + 1 < self.attempts:
                    self.sleep(self.backoff_s * 2**attempt)
        raise FetchFailed(f"{url}: {self.attempts} attempts failed, last error {last!r}")


def meta_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".meta.json")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def is_stored(dest: Path, sha: str, path: str) -> bool:
    """The file and its meta record exist, belong to this commit and path, and the file is unchanged."""
    try:
        meta = json.loads(meta_path(dest).read_text())
        return (meta["sha"], meta["path"]) == (sha, path) and file_sha256(dest) == meta["stored_sha256"]
    except (OSError, ValueError, KeyError):
        return False


def store(source: Source, path: str, dest: Path, compress: bool = True, validate=None) -> bool:
    """Download `path` to `dest` unless it is already stored and intact. True when it was downloaded."""
    if is_stored(dest, source.sha, path):
        return False
    body = source.get(path, validate)
    stored = gzip.compress(body, compresslevel=9, mtime=0) if compress else body
    _write_atomic(dest, stored)
    meta = {"sha": source.sha, "path": path, "upstream_bytes": len(body), "upstream_sha256": hashlib.sha256(body).hexdigest(),
            "stored_sha256": hashlib.sha256(stored).hexdigest()}
    _write_atomic(meta_path(dest), (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode())
    return True


def load_raw(dest: Path, sha: str, path: str, compress: bool = True) -> bytes:
    """Upstream bytes of a stored file, after checking its meta record, stored hash and upstream hash."""
    if not is_stored(dest, sha, path):
        raise ValueError(f"{dest} is missing, damaged or from another commit than {sha}")
    stored = Path(dest).read_bytes()
    body = gzip.decompress(stored) if compress else stored
    if hashlib.sha256(body).hexdigest() != json.loads(meta_path(dest).read_text())["upstream_sha256"]:
        raise ValueError(f"{dest}: upstream bytes differ from the recorded hash")
    return body


def _validate_json_list(body: bytes) -> None:
    if not isinstance(json.loads(body), list):
        raise ValueError("expected a JSON list")


def ensure_pin(root: Path, sha: str) -> None:
    """Record the commit in `pin.json`, or refuse when the store already holds another commit."""
    pin = root / "pin.json"
    if pin.exists():
        recorded = json.loads(pin.read_text())
        if recorded != {"repo": REPO, "sha": sha}:
            raise ValueError(f"{root} is pinned to {recorded}; refusing to mix commits (asked for {sha})")
        return
    _write_atomic(pin, (json.dumps({"repo": REPO, "sha": sha}, sort_keys=True, indent=2) + "\n").encode())


def read_pin(root: Path) -> str | None:
    pin = root / "pin.json"
    return json.loads(pin.read_text())["sha"] if pin.exists() else None


def competitions_path(root: Path) -> Path:
    return root / "raw/competitions.json.gz"


def match_list_path(root: Path, competition_id: int, season_id: int) -> Path:
    return root / f"raw/matches/{competition_id}_{season_id}.json.gz"


def events_path(root: Path, match_id: int) -> Path:
    return root / f"raw/events/{match_id}.json.gz"


def terms_path(root: Path, name: str) -> Path:
    return root / "raw/terms" / name


def load_match_lists(root: Path, sha: str, covered=COVERED) -> list[dict]:
    """All match records of the covered competition seasons, by match id; a match listed twice is an error."""
    matches: dict[int, dict] = {}
    for cid, sid in covered:
        for match in json.loads(load_raw(match_list_path(root, cid, sid), sha, f"data/matches/{cid}/{sid}.json")):
            if match["match_id"] in matches:
                raise ValueError(f"Match {match['match_id']} is listed twice")
            matches[match["match_id"]] = match
    return [matches[m] for m in sorted(matches)]


def fetch_all(fetch: Fetch, root: Path, sha: str, covered=COVERED, workers: int = MAX_WORKERS, attempts: int = ATTEMPTS,
              backoff_s: float = BACKOFF_S, sleep: Callable[[float], None] = time.sleep,
              progress: Callable[[str], None] | None = None) -> list[int]:
    """Store competitions, match lists, terms and the events of every match of `covered` under `root`.

    Returns the sorted expected match ids. Raises ValueError when a covered season is not in the
    competitions file or a match is listed twice, MissingMatchesError (after trying every match) when
    events could not be stored."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ensure_pin(root, sha)
    source = Source(fetch, sha, attempts, backoff_s, sleep)
    say = progress or (lambda _: None)

    store(source, "data/competitions.json", competitions_path(root), validate=_validate_json_list)
    listed = {(c["competition_id"], c["season_id"]) for c in json.loads(load_raw(competitions_path(root), sha, "data/competitions.json"))}
    if absent := [pair for pair in covered if pair not in listed]:
        raise ValueError(f"Covered competition seasons not in competitions.json at {sha}: {absent}")
    for cid, sid in covered:
        store(source, f"data/matches/{cid}/{sid}.json", match_list_path(root, cid, sid), validate=_validate_json_list)
    for name in TERMS_FILES:
        store(source, name, terms_path(root, name), compress=False)
    expected = [m["match_id"] for m in load_match_lists(root, sha, covered)]

    def one(match_id: int) -> tuple[int, str | None, bool]:
        try:
            downloaded = store(source, f"data/events/{match_id}.json", events_path(root, match_id), validate=_validate_json_list)
            return match_id, None, downloaded
        except (FetchFailed, PermanentFetchError) as exc:
            return match_id, str(exc), False

    missing, downloaded_count = {}, 0
    with ThreadPoolExecutor(workers) as pool:
        for i, (match_id, error, downloaded) in enumerate(pool.map(one, expected), start=1):
            downloaded_count += downloaded
            if error is not None:
                missing[match_id] = error
            if i % 100 == 0:
                say(f"{i}/{len(expected)} matches handled, {downloaded_count} downloaded, {len(missing)} missing")
    say(f"{len(expected)} matches expected, {downloaded_count} downloaded this run, {len(missing)} missing")
    still_missing = {m: missing.get(m, "stored file failed verification") for m in expected if not is_stored(events_path(root, m), sha, f"data/events/{m}.json")}
    if still_missing:
        raise MissingMatchesError(still_missing)
    return expected
