"""Read single members of a large remote zip through HTTP range requests (stdlib only).

    python -I scripts/detector_data/remote_zip.py list <url> [pattern]
    python -I scripts/detector_data/remote_zip.py get <url> <out_dir> <regex>

Used for Google Drive archives where only a few members are needed (TeamTrack: the soccer part of an all-sports zip).
Supports zip64 and stored/deflated members. Member names are checked before they become paths.
"""

import re
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ +-]*$")


def get_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``start..end`` (inclusive), retried."""
    for attempt in range(8):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"}
            )
            data = urllib.request.urlopen(request, timeout=180).read()
            if len(data) == end - start + 1:
                return data
            raise OSError(f"short read {len(data)}")
        except OSError as error:
            print(
                f"retry {attempt + 1} at {start}: {error}", file=sys.stderr, flush=True
            )
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"giving up on range {start}-{end}")


def total_size(url: str) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def central_directory(url: str) -> list[dict]:
    """Every member: ``name``, ``offset`` (local header), ``csize``, ``usize``, ``method``."""
    total = total_size(url)
    tail = get_range(url, max(0, total - (1 << 16)), total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    count, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10 : eocd + 20])
    locator = tail.rfind(b"PK\x06\x07")
    if locator >= 0:
        (zip64_eocd,) = struct.unpack("<Q", tail[locator + 8 : locator + 16])
        record = get_range(url, zip64_eocd, zip64_eocd + 55)
        count, cd_size, cd_offset = struct.unpack("<QQQ", record[32:56])
    cd = get_range(url, cd_offset, cd_offset + cd_size - 1)
    members, pos = [], 0
    for _ in range(count):
        (method,) = struct.unpack("<H", cd[pos + 10 : pos + 12])
        csize, usize = struct.unpack("<II", cd[pos + 20 : pos + 28])
        name_len, extra_len, comment_len = struct.unpack(
            "<HHH", cd[pos + 28 : pos + 34]
        )
        (offset,) = struct.unpack("<I", cd[pos + 42 : pos + 46])
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        values = [
            v
            for v, flag in ((usize, usize), (csize, csize), (offset, offset))
            if flag == 0xFFFFFFFF
        ]
        if values:  # zip64 extra field (id 1) holds the 64-bit values in order usize, csize, offset
            e = 0
            while e < len(extra):
                tag, size = struct.unpack("<HH", extra[e : e + 4])
                if tag == 1:
                    fields = list(
                        struct.unpack(f"<{size // 8}Q", extra[e + 4 : e + 4 + size])
                    )
                    if usize == 0xFFFFFFFF:
                        usize = fields.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = fields.pop(0)
                    if offset == 0xFFFFFFFF:
                        offset = fields.pop(0)
                e += 4 + size
        members.append(
            {
                "name": name,
                "offset": offset,
                "csize": csize,
                "usize": usize,
                "method": method,
            }
        )
        pos += 46 + name_len + extra_len + comment_len
    return members


def safe_path(out_dir: Path, name: str) -> Path:
    parts = [p for p in name.split("/") if p]
    if not parts or not all(SAFE_PART.match(p) for p in parts):
        raise ValueError(f"unsafe member name {name!r}")
    return out_dir.joinpath(*parts)


def extract(url: str, member: dict, target: Path) -> None:
    """Write one member to ``target`` (skipped when it already has the right size)."""
    if target.is_file() and target.stat().st_size == member["usize"]:
        return
    header = get_range(url, member["offset"], member["offset"] + 29)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = member["offset"] + 30 + name_len + extra_len
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    inflater = zlib.decompressobj(-15) if member["method"] == 8 else None
    if member["method"] not in (0, 8):
        raise ValueError(
            f"unsupported compression {member['method']} for {member['name']}"
        )
    with part.open("wb") as out:
        for chunk_start in range(start, start + member["csize"], 128 << 20):
            chunk_end = min(chunk_start + (128 << 20), start + member["csize"]) - 1
            data = get_range(url, chunk_start, chunk_end)
            out.write(inflater.decompress(data) if inflater else data)
        if inflater:
            out.write(inflater.flush())
    if part.stat().st_size != member["usize"]:
        raise RuntimeError(
            f"{member['name']}: wrote {part.stat().st_size}, expected {member['usize']}"
        )
    part.rename(target)


def main(argv: list[str]) -> None:
    command, url = argv[0], argv[1]
    members = central_directory(url)
    if command == "list":
        pattern = re.compile(argv[2]) if len(argv) > 2 else None
        for m in members:
            if pattern is None or pattern.search(m["name"]):
                print(m["usize"], m["name"])
        return
    out_dir, pattern = Path(argv[2]), re.compile(argv[3])
    for m in members:
        if pattern.search(m["name"]) and not m["name"].endswith("/"):
            extract(url, m, safe_path(out_dir, m["name"]))
            print(f"got {m['name']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
