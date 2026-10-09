"""Frames at a fixed grid of offsets around the shot anchor, cut from the clip by the canonical decode path.

One extraction path: timing comes from `canonical_frames.probe_video` (ffprobe PTS relative to the clip start, checked
against OpenCV's decoder timestamps, anchor at 4000 ms) and the decode checks are those of
`canonical_frames.extract_inputs` (shape, timestamp agreement, decoded frame count equals the probed count). The loop
itself is not shared: `canonical_frames.py` is hashed into the completion record of `canonical_clip_v1`, so editing it
would break that version's verification. `tests/test_offset_frames.py` pins the two to the same frames.

Selection rule. Offset 0 is the anchor frame, the latest frame at or before the anchor (the canonical `t0`). Every other
offset takes the decoded frame within `TOLERANCE_MS` of the nominal time; none in range is `missing`, several is
`ambiguous`, and neither writes a frame (nothing is substituted). The tolerance is half the 25 fps frame step, so on a
25 fps clip at most one frame fits (two only at exactly 20 ms); clips whose frame step is not 40 ms are refused.
If the anchor frame itself lies more than the tolerance before the anchor, it is still the offset-0 frame but gets the
status `anchor_off_tolerance`, which the probe's complete-case rule excludes.

Cut flag. It reuses the frame-difference measurement of `scripts/report_data_quality.py` (64x36 grey thumbnails,
mean absolute change of consecutive frames) over the transitions strictly between the anchor frame and the offset
frame: the transition that produced the earlier of the two frames is not counted, so a cut that enters the -1600 ms
frame itself (the -1640 -> -1600 transition) does not flag it. This differs from the window flag of
`report_data_quality.window_changes`, which also counts the change at the window's first frame. For a negative offset
the span runs from the frame after the offset frame up to and including the anchor frame; for a positive offset from
the frame after the anchor frame up to and including the offset frame; offset 0 has no span.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from xgpoc.data.canonical_frames import VideoTimeline, probe_video

# The 30 offsets fixed by research/xgc_direction_probe_protocol.md ("Siatka offsetów").
GRID_MS = (-1600, -1400, -1200, -1000, -800, -600,
           -400, -360, -320, -280, -240, -200, -160, -120, -80, -40,
           0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400,
           600, 800, 1000)
ANCHOR_MS = 4000
FRAME_STEP_MS = 40.0         # 25 fps
STEP_TOLERANCE_MS = 0.5      # allowed deviation of the median frame step
TOLERANCE_MS = 20.0          # the probe keeps a frame only if its decoded time is within this of the nominal offset
CUT_THRESHOLD = 0.08         # default of `report_data_quality.py --cut-threshold`
TIMESTAMP_AGREEMENT_MS = .5  # same OpenCV/ffprobe agreement as canonical_frames.extract_inputs
EPS_MS = 1e-6


@dataclass(frozen=True)
class OffsetSelection:
    offset_ms: int
    status: str                       # ok | missing | ambiguous | anchor_off_tolerance
    frame_index: int | None           # set when a frame is chosen (ok, anchor_off_tolerance)
    decoded_ms: float | None          # decoded time relative to the anchor of the chosen frame
    candidates: tuple[int, ...]       # frame indices within tolerance of the nominal time
    nearest_ms: float | None          # decoded time of the nearest frame, kept for missing/ambiguous diagnosis


@dataclass(frozen=True)
class OffsetFrame:
    selection: OffsetSelection
    bgr: np.ndarray | None
    cut_max_change: float | None      # largest consecutive-frame change between anchor frame and this frame
    cut_between: bool | None


@dataclass(frozen=True)
class ShotOffsets:
    timeline: VideoTimeline
    anchor_index: int
    frames: dict[int, OffsetFrame]
    offset0_rule_differs: bool        # the plain nearest-within-tolerance rule would not have chosen the anchor frame


def select_offsets(relative_ms: np.ndarray, grid: tuple[int, ...] = GRID_MS,
                   tolerance_ms: float = TOLERANCE_MS) -> list[OffsetSelection]:
    """Nearest-within-tolerance rule on the decoded frame times (ms relative to the anchor), for every offset."""
    times = np.asarray(relative_ms, dtype=float)
    out = []
    for offset in grid:
        distance = np.abs(times - offset)
        within = tuple(int(i) for i in np.flatnonzero(distance <= tolerance_ms + EPS_MS))
        nearest = int(distance.argmin())
        if len(within) == 1:
            out.append(OffsetSelection(offset, "ok", within[0], float(times[within[0]]), within, float(times[nearest])))
        else:
            out.append(OffsetSelection(offset, "missing" if not within else "ambiguous", None, None, within,
                                       float(times[nearest])))
    return out


def anchor_frame_index(relative_ms: np.ndarray) -> int:
    """Canonical `t0`: the latest frame at or before the anchor."""
    before = np.flatnonzero(np.asarray(relative_ms) <= 0)
    if not len(before):
        raise ValueError("Clip has no frame at or before the annotation anchor")
    return int(before[-1])


def anchor_selection(relative_ms: np.ndarray, tolerance_ms: float = TOLERANCE_MS) -> OffsetSelection:
    index = anchor_frame_index(relative_ms)
    time = float(relative_ms[index])
    status = "ok" if abs(time) <= tolerance_ms + EPS_MS else "anchor_off_tolerance"
    return OffsetSelection(0, status, index, time, (index,), time)


def check_frame_rate(relative_ms: np.ndarray) -> None:
    """The ambiguity rule assumes the 25 fps clips of the index; refuse anything else instead of mislabelling it."""
    step = float(np.median(np.diff(relative_ms)))
    if abs(step - FRAME_STEP_MS) > STEP_TOLERANCE_MS:
        raise ValueError(f"Median frame step {step:.2f} ms is not the expected {FRAME_STEP_MS:.0f} ms (25 fps)")


def grey_thumbnail(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(float)


def max_change_between(changes: dict[int, float], anchor_index: int, frame_index: int) -> float:
    """Largest change over the transitions after the earlier frame up to and including the later one."""
    low, high = sorted((anchor_index, frame_index))
    return max((changes[k] for k in range(low + 1, high + 1)), default=0.0)


def decode_clip(path: Path, timeline: VideoTimeline, relative: np.ndarray, wanted: set[int],
                thumbnail_from: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Decode the whole clip with the checks of `canonical_frames.extract_inputs`; keep `wanted` frames."""
    kept, thumbnails = {}, {}
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    count, cv_first = 0, None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {path}")
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if count >= len(timeline.pts) or bgr.shape != (timeline.height, timeline.width, 3):
                raise ValueError("Decoder shape or frame count differs from ffprobe")
            cv_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            cv_first = cv_ms if cv_first is None else cv_first
            if not np.isfinite(cv_ms) or abs((cv_ms - cv_first - ANCHOR_MS) - relative[count]) > TIMESTAMP_AGREEMENT_MS:
                raise ValueError("OpenCV and ffprobe frame timestamps disagree")
            if count >= thumbnail_from:
                thumbnails[count] = grey_thumbnail(bgr)
            if count in wanted:
                kept[count] = bgr
            count += 1
    finally:
        capture.release()
    if count != len(timeline.pts):
        raise ValueError("Incomplete decode: frame count differs from ffprobe")
    return kept, thumbnails


def extract_offset_frames(path: Path) -> ShotOffsets:
    """One decode of the clip; frames are kept only where selected."""
    timeline = probe_video(path)
    relative = np.array([timeline.relative_ms(i, ANCHOR_MS) for i in range(len(timeline.pts))])
    if relative[-1] < 0:
        raise ValueError("Clip ends before the annotation anchor")
    check_frame_rate(relative)
    anchor = anchor_frame_index(relative)
    selections = select_offsets(relative)
    rule_differs = False
    for position, selection in enumerate(selections):
        if selection.offset_ms == 0:
            rule_differs = selection.status != "ok" or selection.frame_index != anchor
            selections[position] = anchor_selection(relative)
    wanted = {s.frame_index for s in selections if s.frame_index is not None} | {anchor}
    first, last = min(wanted), max(wanted)
    kept, thumbnails = decode_clip(path, timeline, relative, wanted, first)
    changes = {k: float(np.abs(thumbnails[k] - thumbnails[k - 1]).mean() / 255) for k in range(first + 1, last + 1)}

    frames = {}
    for selection in selections:
        if selection.frame_index is None:
            frames[selection.offset_ms] = OffsetFrame(selection, None, None, None)
            continue
        peak = max_change_between(changes, anchor, selection.frame_index)
        frames[selection.offset_ms] = OffsetFrame(selection, kept[selection.frame_index], peak, peak >= CUT_THRESHOLD)
    return ShotOffsets(timeline, anchor, frames, rule_differs)
