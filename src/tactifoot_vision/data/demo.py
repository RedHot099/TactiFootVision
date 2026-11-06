"""Demo data for the notebooks: one SoccerTrack v2 match, fetched from Hugging Face and laid out for the package readers."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tactifoot_vision.data.dataset import Dataset
from tactifoot_vision.data.labelled_clip import load_labelled_clip
from tactifoot_vision.data.video import VideoReader

logger = logging.getLogger(__name__)

DATASET = "atomscott/soccertrack-v2"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
REVISION = "v1.2"  # earlier tags stay available, the card says; a demo is reproducible
DEFAULT_MATCH = "118577"  # M5 of the train split: the smallest MOT clip (270 MB)
MARKER = ".tactifoot-demo.json"
LAYOUT = 1  # bump when what this module writes changes, so old downloads are redone

# Its "Changes" must stay true to what this module does to the data.
ATTRIBUTION = (
    "Video and annotations: SoccerTrack v2, A. Scott, I. Uchida, K. Kuroda, "
    f"Y. Kim, K. Fujii, {DATASET_URL}, licensed CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). Changes: cut to a short "
    "excerpt, frames extracted, annotations converted, and model outputs drawn "
    "by tactifoot_vision."
)

_SOURCE_FILES = (
    "mot/clips/{m}.mp4",
    "mot/{m}.txt",
    "raw/{m}/{m}_keypoints.json",
    "raw/{m}/{m}_homography.npy",
    "raw/{m}/{m}_camera_intrinsics.npz",
    "raw/{m}/{m}_mapx.npy",
    "raw/{m}/{m}_mapy.npy",
    "bas/{m}/{m}_12_class_events.json",
)
# The MOT boxes are the 22 players on the pitch, no role split (docs/format-mot.md
# of the dataset): every track is a package "player". The rows' frame numbers
# count from 0 (checked on match 118577: frames 0..5999 of a 6000-frame clip),
# unlike MOTChallenge's.
_TRACK_CLASS = "player"
_MOT_FIRST_FRAME = 0
# Of the frames before the demo window, this share trains and the rest validates.
_TRAIN_SHARE = 2 / 3
_JPEG_QUALITY = 92


@dataclass(frozen=True)
class DemoData:
    """What :func:`download_demo` leaves on disk, as the paths a notebook needs.

    ``clip_dir`` is a SoccerNet Tracking sequence folder (``img1/``,
    ``gt/gt.txt``, ``seqinfo.ini``, ``gameinfo.ini``) of the last
    ``clip_seconds`` of the match clip at full frame rate, for
    :func:`~tactifoot_vision.data.load_labelled_clip`, the pipeline and the
    tracking evaluators. ``detect_yaml`` is a YOLO ``data.yaml`` of frames
    taken every ``train_stride`` frames from the rest of the clip, split by
    time (the earlier part trains, the later validates), with the clip
    classes and only ``player`` populated. Frames and boxes stay in the
    clip's own pixels: the panorama is a fisheye image, whose undistorted
    version would keep only a third of the resolution. ``calibration`` is an
    ``.npz`` of the static camera: ``K``, ``D`` (4 fisheye coefficients) and
    ``Knew`` as OpenCV's ``cv2.fisheye`` takes them, and ``frame_to_pitch``,
    a 3x3 homography from undistorted frame pixels (``undistortPoints`` with
    ``P=Knew``) to :class:`~tactifoot_vision.pitch.SoccerPitch` metres; the
    pipeline's ``undistortion`` and ``fixed_homography`` arguments take it.
    The dataset's own fit is about 1.3 m off at the median over its 65 pitch
    correspondences, worse at the far corners. ``calibration_dir`` holds the
    match's ``raw/`` files as downloaded: the correspondences, the homography,
    camera intrinsics and undistortion maps. ``events_json`` is its
    ball-action events. ``attribution`` is the credit CC BY 4.0 requires; show
    it wherever the data is shown.
    """

    root: Path
    match: str
    clip_dir: Path
    detect_yaml: Path
    calibration: Path
    calibration_dir: Path
    events_json: Path
    attribution: str = ATTRIBUTION


def download_demo(
    root: str | Path = "data/demo",
    match: str = DEFAULT_MATCH,
    token: str | None = None,
    clip_seconds: float = 30.0,
    train_stride: int = 10,
) -> DemoData:
    """Fetch one SoccerTrack v2 match into ``root/<match>`` and lay it out for the package.

    The source is the dataset's Hugging Face repository at tag ``v1.2``;
    nothing is re-hosted. It is gated: accept the conditions once on its page,
    then pass a token or let ``huggingface_hub`` find one (the ``HF_TOKEN``
    environment variable, a Colab secret of that name, or a saved login). A
    refused download is a ``PermissionError`` that says so; a failed one a
    ``ConnectionError``. About 300 MB come down for the default match; frame
    extraction then writes about 1 GB of JPEGs.

    A second call with the same arguments does nothing and returns the same
    paths; the same ``root`` and ``match`` with other arguments is a
    ``ValueError`` (use another ``root``). The call prints the attribution
    and writes it to ``root/<match>/ATTRIBUTION.md``.
    """
    root = Path(root)
    home = root / match
    params = {
        "dataset": DATASET,
        "revision": REVISION,
        "layout": LAYOUT,
        "match": match,
        "clip_seconds": float(clip_seconds),
        "train_stride": int(train_stride),
    }
    marker = home / MARKER
    if marker.is_file():
        done = json.loads(marker.read_text())
        if done != params:
            raise ValueError(
                f"{home} already holds a demo made with {done}; asked for "
                f"{params}. Use another root, or delete it."
            )
        logger.info("Demo data for match %s is already in %s", match, home)
        print(ATTRIBUTION)
        return _paths(home, match)
    if train_stride < 1:
        raise ValueError(f"train_stride must be >= 1, got {train_stride}")
    if clip_seconds <= 0:
        raise ValueError(f"clip_seconds must be > 0, got {clip_seconds}")

    source = home / "source"
    _download(match, source, token)
    video = VideoReader(source / "mot" / "clips" / f"{match}.mp4")
    rows = _read_mot(source / "mot" / f"{match}.txt")
    window = int(round(clip_seconds * video.fps))
    if not 1 <= window < video.frame_count:
        raise ValueError(
            f"clip_seconds={clip_seconds} must be at least a frame and less than "
            f"the whole clip ({video.duration:.0f} s at {video.fps:g} fps)"
        )
    first_demo = video.frame_count - window
    train_end = int(first_demo * _TRAIN_SHARE)
    sequences = home / "sequences"
    for folder, frames in (
        (sequences / "train", range(0, train_end, train_stride)),
        (sequences / "valid", range(train_end, first_demo, train_stride)),
        (home / "clip", range(first_demo, video.frame_count)),
    ):
        _write_sequence(folder, video, rows, frames)
    _write_calibration(source / "raw" / match, match, home / "clip" / "calibration.npz")

    train = Dataset.from_clips(load_labelled_clip(sequences / "train"), 1.0, 0.0)
    valid = Dataset.from_clips(load_labelled_clip(sequences / "valid"), 0.0, 1.0)
    train.merge(valid).to_yolo(home / "detect")

    (home / "ATTRIBUTION.md").write_text(ATTRIBUTION + "\n")
    marker.write_text(json.dumps(params, indent=2) + "\n")
    print(ATTRIBUTION)
    return _paths(home, match)


def _paths(home: Path, match: str) -> DemoData:
    return DemoData(
        root=home.parent,
        match=match,
        clip_dir=home / "clip",
        detect_yaml=home / "detect" / "data.yaml",
        calibration=home / "clip" / "calibration.npz",
        calibration_dir=home / "source" / "raw" / match,
        events_json=home / "source" / "bas" / match / f"{match}_12_class_events.json",
    )


def _download(match: str, into: Path, token: str | None) -> None:
    """Fetch the match's files from the dataset repository into ``into`` (resumable)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    for name in (f.format(m=match) for f in _SOURCE_FILES):
        try:
            hf_hub_download(
                DATASET,
                name,
                repo_type="dataset",
                revision=REVISION,
                local_dir=into,
                token=token,
            )
        except GatedRepoError as error:
            raise PermissionError(
                f"{DATASET} is gated: open {DATASET_URL}, accept its conditions "
                "with your Hugging Face account, then pass a token "
                "(download_demo(token=...)), set HF_TOKEN, or add a Colab secret "
                f"named HF_TOKEN. ({error.__class__.__name__}: {error})"
            ) from error
        except HfHubHTTPError as error:
            if error.response is not None and error.response.status_code in (401, 403):
                raise PermissionError(
                    f"{DATASET} refused the download of {name} (HTTP "
                    f"{error.response.status_code}): accept its conditions at "
                    f"{DATASET_URL} and pass a valid token."
                ) from error
            raise ConnectionError(
                f"Could not download {name} from {DATASET}: {error}"
            ) from error
        except OSError as error:  # network and disk failures
            raise ConnectionError(
                f"Could not download {name} from {DATASET}: {error}"
            ) from error


def _read_mot(path: Path) -> np.ndarray:
    """``(N, 6)`` rows ``frame, track id, x, y, w, h`` of the match's MOT file."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    try:
        return np.loadtxt(lines, delimiter=",", usecols=range(6), ndmin=2)
    except ValueError as error:
        raise ValueError(f"{path} is not a MOT ground-truth file: {error}") from None


def _write_calibration(raw: Path, match: str, path: Path) -> None:
    """The match's static-camera calibration as the pipeline takes it.

    The dataset's ``mapx``/``mapy`` are exactly
    ``cv2.fisheye.initUndistortRectifyMap(K, D, I, Knew, frame size)`` of its
    intrinsics (checked on match 118577, to the pixel), and its homography
    maps pitch metres to that remapped frame; so the frame-to-pitch matrix
    of undistorted pixels is its inverse.
    """
    # The npz also holds object arrays (rvecs, tvecs); only these three are
    # read, so pickles stay off (allow_pickle is False by default).
    intrinsics = np.load(raw / f"{match}_camera_intrinsics.npz")
    pitch_to_frame = np.load(raw / f"{match}_homography.npy")
    np.savez(
        path,
        K=intrinsics["K"],
        D=intrinsics["D"],
        Knew=intrinsics["Knew"],
        frame_to_pitch=np.linalg.inv(pitch_to_frame),
    )


def _write_sequence(
    folder: Path, video: VideoReader, rows: np.ndarray, frames: range
) -> None:
    """Write the frames ``frames`` of ``video`` and their boxes as a SoccerNet Tracking sequence.

    Image ``n`` (from 1) is the ``n``-th frame of ``frames``, so a strided
    range becomes a gapless sequence the readers accept; the frame rate
    recorded is the video's divided by the stride.
    """
    images = folder / "img1"
    images.mkdir(parents=True, exist_ok=True)
    (folder / "gt").mkdir(exist_ok=True)
    total = len(frames)
    reader = video.frames(frames.start, frames.stop, frames.step)
    number = 0
    for number, (_, frame) in enumerate(
        tqdm(reader, total=total, desc=folder.name, unit="frame"), start=1
    ):
        path = images / f"{number:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]):
            raise RuntimeError(f"Could not write {path}")
    if number != total:
        raise RuntimeError(
            f"{video.path} gave {number} of the {total} frames {frames.start}.."
            f"{frames.stop - 1} (step {frames.step}); the clip is truncated"
        )
    mot_frames = rows[:, 0].astype(int) - _MOT_FIRST_FRAME
    kept = np.isin(mot_frames, list(frames))
    renumbered = rows[kept].copy()
    renumbered[:, 0] = (mot_frames[kept] - frames.start) // frames.step + 1
    with (folder / "gt" / "gt.txt").open("w") as out:
        for frame, track, x, y, w, h in renumbered:
            out.write(f"{int(frame)},{int(track)},{x:g},{y:g},{w:g},{h:g},1,-1,-1,-1\n")
    tracks = sorted(set(renumbered[:, 1].astype(int)))
    (folder / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={folder.name}\n"
        "imDir=img1\n"
        f"frameRate={video.fps / frames.step:g}\n"
        f"seqLength={total}\n"
        f"imWidth={video.width}\n"
        f"imHeight={video.height}\n"
        "imExt=.jpg\n"
    )
    (folder / "gameinfo.ini").write_text(
        "[Sequence]\n"
        f"name={folder.name}\n"
        f"num_tracklets={len(tracks)}\n"
        + "".join(f"trackletID_{t}= {_TRACK_CLASS};{t}\n" for t in tracks)
    )
