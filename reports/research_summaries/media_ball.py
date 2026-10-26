"""Stills and the overlay clip for the ball summary (CPU only, from the phase's saved outputs).

Run: uv run python reports/research_summaries/media_ball.py
Markers: truth (white ring), the detector's top-1 (orange square), the tracked ball (green dot with a
1 s trail; hollow when interpolated). A pitch inset shows the three positions in metres every 5th frame,
the frames the phase evaluates.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import ball_data
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sources import OUT

MEDIA = OUT / "ball/media"
FPS = 25
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
TRUTH = (255, 255, 255)
TOP1 = (235, 104, 52)
TRACK = (27, 175, 122)
SCALE = 2 / 3  # 1920x1080 frames are drawn at 1280x720

# (clip, first frame, frames) of the clip and the stills; picked where tracking and top-1 differ most
CLIP = ("SNGS-198", 26, 200)
# The frame where top-1 jumps to a referee while the track stays on the ball; the clip holds it.
FREEZE_FRAME, FREEZE_SECONDS = 56, 4.0


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _xy(row, name: str, scale: float) -> tuple[float, float] | None:
    u, v = row[f"{name}_u"], row[f"{name}_v"]
    return None if np.isnan(u) else (u * scale, v * scale)


def _pitch_inset(
    balls: ball_data.ClipBalls, frame: int, size=(330, 214)
) -> Image.Image:
    """Top-down pitch with truth, top-1 and track at the last evaluated frame <= frame."""
    w, h = size
    pad = 12
    img = Image.new("RGBA", size, (16, 18, 22, 215))
    d = ImageDraw.Draw(img)
    sx, sy = (w - 2 * pad) / 105, (h - 2 * pad - 26) / 68

    def p(x, y):
        return pad + x * sx, pad + 26 + y * sy

    line = (255, 255, 255, 150)
    d.rectangle([*p(0, 0), *p(105, 68)], outline=line, width=1)
    d.line([p(52.5, 0), p(52.5, 68)], fill=line, width=1)
    for x0, x1 in ((0, 16.5), (105 - 16.5, 105)):
        d.rectangle([*p(x0, 13.84), *p(x1, 54.16)], outline=line, width=1)
    cx, cy = p(52.5, 34)
    d.ellipse(
        [cx - 9.15 * sx, cy - 9.15 * sy, cx + 9.15 * sx, cy + 9.15 * sy],
        outline=line,
        width=1,
    )
    rows = balls.placement[balls.placement.frame <= frame]
    if len(rows):
        r = rows.iloc[-1]
        for name, color, radius in (("top", TOP1, 5), ("track", TRACK, 5)):
            x, y = r[f"x_{name}"], r[f"y_{name}"]
            if not np.isnan(x):
                px, py = p(np.clip(x, -3, 108), np.clip(y, -3, 71))
                d.line([p(r.truth_x, r.truth_y), (px, py)], fill=(*color, 160), width=1)
                d.ellipse(
                    [px - radius, py - radius, px + radius, py + radius], fill=color
                )
        tx, ty = p(r.truth_x, r.truth_y)
        d.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], outline=TRUTH, width=2)
        font = _font(13)
        parts = [("error  ", (200, 200, 200))]
        for name, color, label in (("track", TRACK, "tracked"), ("top", TOP1, "top-1")):
            e = r[f"error_m_{name}"]
            parts.append((f"{label} {'–' if np.isnan(e) else f'{e:.1f} m'}   ", color))
        x = pad
        for text, color in parts:
            d.text((x, 6), text, font=font, fill=color)
            x += d.textlength(text, font=font)
    return img


def draw(
    balls: ball_data.ClipBalls,
    frame: int,
    trail: int = 25,
    scale: float = SCALE,
    panels: bool = True,
) -> Image.Image:
    """The frame with the three markers; ``panels`` adds the legend and the pitch inset."""
    image = Image.open(balls.image(frame)).convert("RGB")
    image = image.resize(
        (round(1920 * scale), round(1080 * scale)), Image.LANCZOS
    ).convert("RGBA")
    k = scale / SCALE  # marker sizes are tuned for the 1280 px frame
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rows = balls.frames.set_index("frame")
    past = rows.loc[max(1, frame - trail) : frame]
    pts = [
        (u * scale, v * scale)
        for u, v in zip(past.track_u, past.track_v, strict=True)
        if not np.isnan(u)
    ]
    for i in range(1, len(pts)):
        if np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) > 40 * k:
            continue  # a new track segment, not motion
        a = int(220 * i / len(pts))
        d.line([pts[i - 1], pts[i]], fill=(*TRACK, a), width=round(3 * k))
    row = rows.loc[frame]
    truth = _xy(row, "truth", scale)
    if truth:
        r = max(9 * k, row.truth_w * scale * 0.9)
        d.ellipse(
            [truth[0] - r, truth[1] - r, truth[0] + r, truth[1] + r],
            outline=(*TRUTH, 255),
            width=round(2 * k),
        )
    top = _xy(row, "top", scale)
    if top:
        r = 11 * k
        d.rectangle(
            [top[0] - r, top[1] - r, top[0] + r, top[1] + r],
            outline=(*TOP1, 255),
            width=round(3 * k),
        )
    track = _xy(row, "track", scale)
    if track:
        r = 6 * k
        box = [track[0] - r, track[1] - r, track[0] + r, track[1] + r]
        if row.status == "interpolated":
            d.ellipse(box, outline=(*TRACK, 255), width=round(3 * k))
        else:
            d.ellipse(box, fill=(*TRACK, 255), outline=(255, 255, 255, 255), width=1)
    image.alpha_composite(layer)
    if not panels:
        return image
    _legend(image)
    inset = _pitch_inset(balls, frame)
    image.alpha_composite(
        inset, (image.width - inset.width - 14, image.height - inset.height - 14)
    )
    return image


def _legend(image: Image.Image) -> None:
    d = ImageDraw.Draw(image)
    font = _font(15, bold=True)
    x, y = 18, 16
    d.rounded_rectangle([x - 8, y - 6, x + 560, y + 26], 8, fill=(16, 18, 22, 210))
    d.ellipse([x, y + 3, x + 14, y + 17], outline=TRUTH, width=2)
    d.text((x + 20, y), "ground truth", font=font, fill=TRUTH)
    x += 135
    d.rectangle([x, y + 3, x + 14, y + 17], outline=TOP1, width=3)
    d.text((x + 20, y), "detector top-1", font=font, fill=TOP1)
    x += 150
    d.ellipse([x, y + 3, x + 14, y + 17], fill=TRACK)
    d.text((x + 20, y), "tracked (hollow = interpolated)", font=font, fill=TRACK)


def _callout(
    d: ImageDraw.ImageDraw, anchor, offset, text: str, color, alpha: float
) -> None:
    if alpha <= 0:
        return
    x, y = anchor
    tx, ty = x + offset[0], y + offset[1]
    d.line([(x, y), (tx, ty)], fill=(*color, int(255 * alpha)), width=2)
    font = _font(17, bold=True)
    box = d.multiline_textbbox((tx, ty), text, font=font, spacing=4)
    d.rounded_rectangle(
        [box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6],
        7,
        fill=(16, 18, 22, int(225 * alpha)),
    )
    d.multiline_text(
        (tx, ty), text, font=font, fill=(*color, int(255 * alpha)), spacing=4
    )


def freeze(balls: ball_data.ClipBalls, frame: int, t: float) -> Image.Image:
    """The held frame: dimmed, then the two picks explained one after the other."""
    image = draw(balls, frame)
    dim = Image.new("RGBA", image.size, (0, 0, 0, int(80 * min(1.0, t / 0.3))))
    image.alpha_composite(dim)
    image = Image.alpha_composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    row = balls.frames.set_index("frame").loc[frame]
    place = balls.placement.set_index("frame").loc[frame]
    top = _xy(row, "top", SCALE)
    track = _xy(row, "track", SCALE)
    a1, a2, a3 = (float(np.clip((t - s) / 0.35, 0, 1)) for s in (0.3, 1.3, 2.3))
    _callout(
        d, top, (40, -110),
        f"detector top-1: a referee ({row.top_w:.0f} px box)\n{place.error_m_top:.0f} m from the ball",
        TOP1, a1,
    )  # fmt: skip
    if a2 > 0:
        r = 20
        d.ellipse(
            [track[0] - r, track[1] - r, track[0] + r, track[1] + r],
            outline=(*TRACK, int(255 * a2)),
            width=3,
        )
    _callout(
        d, (track[0] + 14, track[1] - 14), (70, -70),
        f"tracked ball: {place.error_m_track:.1f} m from the truth",
        TRACK, a2,
    )  # fmt: skip
    font = _font(18, bold=True)
    text = "A 45 px size filter drops the referee; the tracker only accepts candidates\nconsistent with the ball's motion on the pitch (gate in metres)."
    if a3 > 0:
        box = d.multiline_textbbox((24, 64), text, font=font, spacing=4)
        d.rounded_rectangle(
            [box[0] - 10, box[1] - 8, box[2] + 10, box[3] + 8],
            8,
            fill=(16, 18, 22, int(225 * a3)),
        )
        d.multiline_text(
            (24, 64), text, font=font, fill=(255, 255, 255, int(255 * a3)), spacing=4
        )
    image.alpha_composite(layer)
    return image


def clip(balls: ball_data.ClipBalls, start: int, length: int) -> Path:
    path = MEDIA / "ball_clip.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        i = 0
        for frame in range(start, start + length):
            draw(balls, frame).convert("RGB").save(Path(tmp) / f"{i:04d}.png")
            i += 1
            if frame == FREEZE_FRAME:
                for k in range(int(FREEZE_SECONDS * FPS)):
                    freeze(balls, frame, k / FPS).convert("RGB").save(
                        Path(tmp) / f"{i:04d}.png"
                    )
                    i += 1
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
             "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            check=True,
        )  # fmt: skip
    freeze(balls, FREEZE_FRAME, 10.0).convert("RGB").save(
        MEDIA / "freeze.jpg", quality=88
    )
    return path


def still(balls: ball_data.ClipBalls, frame: int, name: str) -> Path:
    path = MEDIA / f"{name}.jpg"
    draw(balls, frame).convert("RGB").save(path, quality=88)
    return path


def zoom(balls: ball_data.ClipBalls, frame: int, name: str, size=(900, 506)) -> Path:
    """A full-resolution crop around the truth (or the track), for the gallery."""
    image = draw(balls, frame, scale=1.0, panels=False)
    row = balls.frames.set_index("frame").loc[frame]
    u, v = (
        (row.truth_u, row.truth_v)
        if not np.isnan(row.truth_u)
        else (row.track_u, row.track_v)
    )
    w, h = size
    if (
        not np.isnan(row.track_u)
        and np.hypot(row.track_u - u, row.track_v - v) < 0.8 * w
    ):
        u, v = (
            (u + row.track_u) / 2,
            (v + row.track_v) / 2,
        )  # keep a wrong track in the crop too
    x0 = int(np.clip(u - w / 2, 0, 1920 - w))
    y0 = int(np.clip(v - h / 2, 0, 1080 - h))
    path = MEDIA / f"{name}.jpg"
    image.crop((x0, y0, x0 + w, y0 + h)).convert("RGB").save(path, quality=88)
    return path


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    name, start, length = CLIP
    balls = ball_data.load(name)
    clip(balls, start, length)
    gallery = {
        "g_fog": ("SNGS-132", 591),
        "g_interp": ("SNGS-196", 66),
        "g_small": ("SNGS-194", 146),
        "g_fail": ("SNGS-144", 116),
    }
    meta = {
        "clip": {"name": name, "start": start, "frames": length, "freeze": FREEZE_FRAME}
    }
    for key, (clip_name, frame) in gallery.items():
        clip_balls = ball_data.load(clip_name)
        zoom(clip_balls, frame, key)
        place = clip_balls.placement.set_index("frame").loc[frame]
        status = clip_balls.frames.set_index("frame").loc[frame, "status"]
        meta[key] = {
            "clip": clip_name,
            "frame": frame,
            "status": status,
            "error_track": None
            if np.isnan(place.error_m_track)
            else float(place.error_m_track),
            "error_top": None
            if np.isnan(place.error_m_top)
            else float(place.error_m_top),
        }
    (MEDIA / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
