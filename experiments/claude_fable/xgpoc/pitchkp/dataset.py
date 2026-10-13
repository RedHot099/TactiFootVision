"""Per-image labels from a line fit, the rejection rule, and the YOLO-pose export (pitch-keypoints 01).

Rejection rule (`rejection`), applied to the line fit only, so the GSR position check stays independent of it:
1. no fit: fewer than 9 annotated ground-plane points, no straight line, or no camera view fits the lines;
2. `rms` > RMS_MAX_PX: the annotated points do not lie on the fitted pitch lines (720p px);
3. `max` > MAX_RESIDUAL_PX: some annotated point is far from its line (a misnamed or misplaced line);
4. `sd_max` > SD_MAX_PX: a visible vertex is poorly determined by the lines (largest 2-D standard deviation of the
   visible vertices' pixel positions, residual covariance with a 1 px floor); an underdetermined fit is infinite;
5. no pitch area in the frame;
6. not a main-side view: up the frame at its bottom centre must point to the far touchline within 66 degrees
   (`up_dy` < MAIN_SIDE_UP_DY on the unit pitch direction). A camera beyond the near touchline always looks that way
   (on GSR, main camera only, `up_dy` <= -0.50). SoccerNet names the lines in pitch coordinates, so a reverse-angle
   view (camera beyond the far touchline, `up_dy` > 0.4) or a view along the pitch (behind a goal or above it) gets
   labels rotated by 180 or 90 degrees against a main-camera view of a look-alike picture, which the image cannot tell
   apart. They are left out of the export (`fits.parquet` keeps them).
"""

import cv2
import numpy as np

from xgpoc.pitchkp import fit as F
from xgpoc.pitchkp.sources import Frame

RMS_MAX_PX = 4.0
MAX_RESIDUAL_PX = 16.0
SD_MAX_PX = 10.0
MAIN_SIDE_UP_DY = -0.4                                    # cos(66 deg)
FLIP_IDX = (24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 8, 6, 7,
            0, 1, 2, 3, 4, 5, 31, 30)                     # data/keypoints/data.yaml
IMAGE_SIZE = (960, 540)                                   # stored images (width, height)


def flipped_lines(lines: dict) -> dict:
    """SoccerNet annotations of the horizontally flipped image: x -> 1 - x and left/right names swapped."""
    return {F.FLIP_NAMES.get(name, name): [{"x": 1.0 - p["x"], "y": p["y"]} for p in points]
            for name, points in lines.items()}


def flip_labels(xy: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels of the flipped image as Ultralytics builds them: keypoint i <- keypoint flip_idx[i], x -> W - x."""
    idx = np.array(FLIP_IDX)
    out = xy[idx].copy()
    out[:, 0] = F.FRAME_W - out[:, 0]
    return np.where(visible[idx][:, None], out, 0.0), visible[idx]


def rejection(record: dict) -> str:
    """The first rule an image's record breaks, or "" when it is kept."""
    if not record["fitted"]:
        return "no fit"
    if record["rms"] > RMS_MAX_PX:
        return "rms"
    if record["max_residual"] > MAX_RESIDUAL_PX:
        return "max residual"
    if record["sd_max"] > SD_MAX_PX:
        return "vertex sd"
    if record["box"] is None:
        return "no pitch area"
    if record["view"] != "main side":
        return "view: " + record["view"]
    return ""


def view_class(up_dy: float) -> str:
    """Main side, opposite side or along the pitch, from the pitch direction of 'up the frame' (`fit.image_up_direction`)."""
    if not np.isfinite(up_dy):
        return "unresolved"
    if up_dy < MAIN_SIDE_UP_DY:
        return "main side"
    if up_dy > -MAIN_SIDE_UP_DY:
        return "opposite side"
    return "along the pitch"


def feet_errors(H: np.ndarray, feet_image: np.ndarray, feet_pitch: np.ndarray) -> np.ndarray:
    """Metres between persons' bottom-middle pixels mapped through `H` and their annotated pitch positions."""
    if len(feet_image) == 0:
        return np.zeros(0)
    q = np.column_stack([feet_image, np.ones(len(feet_image))]) @ H.T
    return np.linalg.norm(q[:, :2] / q[:, 2:] - feet_pitch, axis=1)


def process(frame: Frame, flip_check: bool = False) -> dict:
    """Fit one image's lines and build its record (labels, residuals, uncertainty, GSR position check)."""
    pts = F.annotation_points(frame.lines)
    fit = F.fit_lines(pts)
    record = {
        "key": frame.key, "source": frame.source, "split": frame.split, "group": frame.group, "frame": frame.frame,
        "game": frame.game, "image": str(frame.image), "n_points": fit.n_points, "n_lines": fit.n_lines,
        "lines": ",".join(sorted(pts)), "fitted": fit.ok, "fit_reason": fit.reason,
        "rms": fit.rms, "p50_residual": np.nan, "max_residual": fit.max, "sd_max": np.inf, "box": None,
        "H": None, "xy": None, "visible": None, "sd": None, "n_visible": 0, "up_dx": np.nan, "up_dy": np.nan,
        "view": "unresolved",
        "n_feet": len(frame.feet_image), "feet_median_m": np.nan, "feet_p90_m": np.nan, "feet_raw_median_m": np.nan,
        "flip_max_px": np.nan, "flip_visible_equal": None,
    }
    if fit.ok:
        lab = F.labels(fit.H, fit.front)
        sd = F.vertex_sd(fit)
        up = F.image_up_direction(fit.H)
        record.update(
            p50_residual=float(np.median(fit.residuals)), H=fit.H.ravel().tolist(), xy=lab.xy.ravel().tolist(),
            visible=lab.visible.tolist(), sd=np.where(lab.visible, sd, np.nan).tolist(),
            sd_max=float(sd[lab.visible].max()) if lab.visible.any() else 0.0, n_visible=int(lab.visible.sum()),
            box=list(lab.box) if lab.box is not None else None, up_dx=float(up[0]), up_dy=float(up[1]),
            view=view_class(float(up[1])),
        )
        if len(frame.feet_image):
            err = feet_errors(fit.H, frame.feet_image, frame.feet_pitch)
            raw = feet_errors(fit.H, frame.feet_image, frame.feet_pitch_raw)
            record.update(feet_median_m=float(np.median(err)), feet_p90_m=float(np.quantile(err, 0.9)),
                          feet_raw_median_m=float(np.nanmedian(raw)) if np.isfinite(raw).any() else np.nan)
        if flip_check:
            flipped = F.fit_lines(F.annotation_points(flipped_lines(frame.lines)))
            if flipped.ok:
                got = F.labels(flipped.H, flipped.front)
                want_xy, want_vis = flip_labels(lab.xy, lab.visible)
                both = got.visible & want_vis
                record["flip_visible_equal"] = bool((got.visible == want_vis).all())
                record["flip_max_px"] = float(np.abs(got.xy - want_xy)[both].max()) if both.any() else 0.0
    record["reject"] = rejection(record)
    return record


def yolo_line(record: dict) -> str:
    """One YOLO-pose label line: class 0, the pitch box and 32 (x, y, v) keypoints, all as fractions of the image."""
    x0, y0, x1, y1 = record["box"]
    xy = np.asarray(record["xy"], dtype=float).reshape(32, 2) / (F.FRAME_W, F.FRAME_H)
    visible = np.asarray(record["visible"], dtype=bool)
    values = [0, (x0 + x1) / 2 / F.FRAME_W, (y0 + y1) / 2 / F.FRAME_H, (x1 - x0) / F.FRAME_W, (y1 - y0) / F.FRAME_H]
    for (x, y), v in zip(xy, visible, strict=True):
        values += [x, y, 2] if v else [0, 0, 0]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)


def draw_overlay(image: np.ndarray, record: dict, lines: dict) -> np.ndarray:
    """The image (any 16:9 size) with the fitted pitch markings (cyan), annotated points (yellow), vertices (red)."""
    out = image.copy()
    sx, sy = out.shape[1] / F.FRAME_W, out.shape[0] / F.FRAME_H
    if record["H"] is not None:
        H = np.asarray(record["H"]).reshape(3, 3)
        from tactifoot_vision.pitch import pitch_markings

        markings, _ = pitch_markings(F.PITCH, samples=128)
        for poly in markings:
            dense = np.vstack([np.linspace(a, b, 40) for a, b in zip(poly[:-1], poly[1:], strict=True)])
            xy, ok = F.pitch_to_frame(H, dense)
            for a, b, oa, ob in zip(xy[:-1], xy[1:], ok[:-1], ok[1:], strict=True):
                if oa and ob and np.abs(np.concatenate([a, b])).max() < 1e5:
                    cv2.line(out, (int(a[0] * sx), int(a[1] * sy)), (int(b[0] * sx), int(b[1] * sy)), (255, 255, 0), 1)
    for points in F.annotation_points(lines).values():
        for x, y in points:
            cv2.circle(out, (int(x * sx), int(y * sy)), 2, (0, 255, 255), -1)
    if record["xy"] is not None:
        xy = np.asarray(record["xy"]).reshape(32, 2)
        for i, ((x, y), v) in enumerate(zip(xy, record["visible"], strict=True)):
            if v:
                c = (int(x * sx), int(y * sy))
                cv2.circle(out, c, 4, (0, 0, 255), -1)
                cv2.putText(out, str(i), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    text = f"{record['key']} rms {record['rms']:.2f} max {record['max_residual']:.1f} sd {record['sd_max']:.1f}"
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
