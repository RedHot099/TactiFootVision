"""Checks before a large write: the directory sits on a mounted, writable filesystem and enough space stays free.

A directory under an unmounted mount point (`/mnt/storage` while the partition is away) is an ordinary folder of the
root filesystem; writing 200 GB there fills `/`. So the filesystem that holds the target must be mounted somewhere
other than `/`, mounted read-write, and accept a written and read-back probe file. The space rule has two parts: the
target needs room for the next write, and `/home` keeps `HOME_FLOOR_BYTES` free (when the target is on the same
filesystem as `/home`, the floor applies to the target's own free space).
"""

import os
import shlex
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HOME_FLOOR_BYTES = 5 * 2**30     # the owner's floor for /home
PROBE_BYTES = 2**20


class TargetRefused(RuntimeError):
    """The target is not a mounted, writable filesystem."""


class LowSpace(RuntimeError):
    """Not enough free space for the next write or below the /home floor."""


@dataclass(frozen=True)
class MountInfo:
    target: str          # mount point of the filesystem holding the path
    fstype: str
    options: tuple[str, ...]


def nearest_existing(path: Path) -> Path:
    path = path.absolute()
    while not path.exists():
        path = path.parent
    return path


def mount_info(path: Path) -> MountInfo:
    """Mount point, type and options of the filesystem that holds `path` (or its nearest existing parent), by findmnt."""
    out = subprocess.run(["findmnt", "-T", str(nearest_existing(path)), "-n", "-P", "-o", "TARGET,FSTYPE,OPTIONS"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise TargetRefused(f"findmnt cannot place {path} on a mounted filesystem")
    fields = dict(part.split("=", 1) for part in shlex.split(out.stdout.strip()))
    return MountInfo(fields["TARGET"], fields["FSTYPE"], tuple(fields["OPTIONS"].split(",")))


def write_probe(directory: Path) -> None:
    """Write PROBE_BYTES to a temporary file in `directory`, sync, read back, delete."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write_probe_{uuid.uuid4().hex}"
    payload = os.urandom(PROBE_BYTES)
    try:
        with probe.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_bytes() != payload:
            raise TargetRefused(f"{directory} returned different bytes than were written")
    except OSError as exc:
        raise TargetRefused(f"{directory} is not writable: {type(exc).__name__}: {exc.strerror}") from None
    finally:
        probe.unlink(missing_ok=True)


def check_target(path: Path, lookup: Callable[[Path], MountInfo] | None = None) -> MountInfo:
    """Refuse unless `path` is on a filesystem mounted away from `/`, read-write, and accepts a probe file."""
    info = (lookup or mount_info)(path)
    if info.target == "/":
        raise TargetRefused(f"{path} is on the root filesystem, not on a mounted target (mount point {info.target})")
    if "ro" in info.options:
        raise TargetRefused(f"{path} is on {info.target}, which is mounted read-only")
    write_probe(path)
    return info


def free_bytes(path: Path) -> int:
    stat = os.statvfs(nearest_existing(path))
    return stat.f_bavail * stat.f_frsize


def same_device(a: Path, b: Path) -> bool:
    return os.stat(nearest_existing(a)).st_dev == os.stat(nearest_existing(b)).st_dev


def check_space(path: Path, need_bytes: int, home: Path = Path("/home"), floor: int = HOME_FLOOR_BYTES,
                free: Callable[[Path], int] = free_bytes, shares_device: Callable[[Path, Path], bool] = same_device) -> None:
    """Raise `LowSpace` unless `path` has `need_bytes` free and `home` keeps `floor` free afterwards."""
    room = free(path)
    if shares_device(path, home):
        if room < need_bytes + floor:
            raise LowSpace(f"{path} shares a filesystem with {home}: {room / 2**30:.1f} GiB free, "
                           f"{need_bytes / 2**30:.1f} GiB needed plus the {floor / 2**30:.0f} GiB floor")
    else:
        if room < need_bytes:
            raise LowSpace(f"{path}: {room / 2**30:.1f} GiB free, {need_bytes / 2**30:.1f} GiB needed")
        if (home_room := free(home)) < floor:
            raise LowSpace(f"{home} has {home_room / 2**30:.1f} GiB free, below the {floor / 2**30:.0f} GiB floor")


def disk_summary(path: Path, home: Path = Path("/home")) -> dict:
    """Free space of `path` and `home` in GiB, for the dry run."""
    return {"target_free_gib": round(free_bytes(path) / 2**30, 1), "home_free_gib": round(shutil.disk_usage(home).free / 2**30, 1)}
