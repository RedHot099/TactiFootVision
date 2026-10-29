"""Shot stills and the short overlay clip for the xG summary (CPU only).

Run: CUDA_VISIBLE_DEVICES= uv run python reports/research_summaries/media_xg.py
Each shot's 8 s clip goes through tv.Pipeline once (cached as a PipelineResult);
stills are the frame at the shot anchor, the clip is 6 s around it with an xG card.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import geometry_xg
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from sources import CLIP_ANCHOR_MS, DETECTOR, KEYPOINTS, OUT, SHOTS, paired_shots

import tactifoot_vision as tv
from tactifoot_vision.models.ultralytics import YOLOPoseModel

MEDIA = OUT / "xg/media"
CACHE = OUT / "xg/cache"
FPS = 25
ANCHOR = CLIP_ANCHOR_MS * FPS // 1000

# name -> (SoccerNet shot id, caption)
GALLERY = {
    "agree": (
        "soccernet__italy_serie-a_2015-2016_2015-09-27---21-45-Inter-1---4-Fiorentina__1__1057417",
        "Close range in open play: the easy case, where even the frame alone rates the chance highly.",
    ),
    "angle": (
        "soccernet__spain_laliga_2015-2016_2016-03-02---23-00-Levante-1---3-Real-Madrid__1__1148208",
        "Sharp angle (< 15°): the over-prediction seen in 2.4, corrected by the measured goal-mouth angle.",
    ),
    "close": (
        "soccernet__germany_bundesliga_2015-2016_2015-08-29---19-30-Bayern-Munich-3---0-Bayer-Leverkusen__2__1999537",
        "Crowded six-yard box: the measured positions temper a high estimate from the frame.",
    ),
    "pressure": (
        "soccernet__england_epl_2015-2016_2015-10-17---17-00-Chelsea-2---0-Aston-Villa__2__1536623",
        "Pressured header: both video models rate it above StatsBomb, which annotates the pressure. Across all pressured shots the gap is within chance (2.4).",
    ),
}
FEATURE = "soccernet__england_epl_2015-2016_2015-08-16---18-00-Manchester-City-3---0-Chelsea__1__1859901"
# Text a pirate stream burned into the SoccerNet video (x0, y0, x1, y1), blurred out for presentation.
WATERMARKS = {FEATURE: [(960, 0, 1280, 30), (1030, 694, 1280, 720)]}
FREEZE_SECONDS = 4.4
# Where the radar goes when the default top-right corner would hide the goal.
RADAR_CORNER = {FEATURE: "bottom-right"}
TEAM_COLORS = ["#3987e5", "#3987e5"]
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"


class SquarePoseModel(YOLOPoseModel):
    """The sn960 pitch model was trained on frames padded to a centred grey square;
    on a letterboxed 16:9 frame it finds nothing, so pad here and shift back."""

    def predict(self, image: np.ndarray) -> sv.KeyPoints:
        h, w = image.shape[:2]
        top = (w - h) // 2
        square = cv2.copyMakeBorder(
            image, top, w - h - top, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        keypoints = super().predict(square)
        keypoints.xy[..., 1] -= top
        if "xyxy" in keypoints.data:
            keypoints.data["xyxy"][:, [1, 3]] -= top
        return keypoints


def analyse(shot_id: str) -> tv.PipelineResult:
    path = CACHE / f"{shot_id}.pkl"
    if path.exists():
        return tv.PipelineResult.load(path)
    pipeline = tv.Pipeline(
        detector=tv.load_model(
            "yolo", DETECTOR, conf=0.3, class_conf={"ball": 0.15}, iou=0.5, device="cpu"
        ),
        keypoint_model=SquarePoseModel(KEYPOINTS, conf=0.5, imgsz=960, device="cpu"),
        homography=tv.pitch.HomographyEstimator(min_confidence=0.3),
        team_classifier=tv.teams.TeamClassifier(embedder="siglip", device="cpu"),
    )
    result = pipeline.run(SHOTS / "clips" / f"{shot_id}.mp4", progress=False)
    result = tv.pipeline.smooth_homographies(result)
    result = tv.pipeline.join_tracks(result)
    result = tv.pipeline.assign_roles(result)
    result = tv.pipeline.track_ball(result)
    result.save(path)
    return result


def annotators() -> tuple[tv.viz.FrameAnnotator, tv.viz.PitchRadar]:
    annotator = tv.viz.FrameAnnotator(
        team_colors=TEAM_COLORS,
        draw_labels=False,
        draw_keypoints=False,
        trail_seconds=1.0,
    )
    radar = tv.viz.PitchRadar(
        width_px=900, team_colors=TEAM_COLORS, draw_visible_area=True, trail_seconds=1.5
    )
    return annotator, radar


def draw(
    result: tv.PipelineResult, frame: np.ndarray, index: int, corner: str = "top-right"
) -> np.ndarray:
    annotator, radar = annotators()
    frame_result = result.frame(index)
    image = annotator.annotate(frame, frame_result, history=result.window(index, 1.0))
    pitch = radar.draw(
        frame_result, history=result.window(index, 1.5), frame_size=result.frame_size
    )
    pitch = attacking_half(pitch, result)
    return tv.viz.overlay(
        image, pitch, position=corner, width_fraction=0.24, alpha=0.9, padding=14
    )


def attacking_half(radar: np.ndarray, result: tv.PipelineResult) -> np.ndarray:
    """The 56% of the radar on the side the shot is taken, so the dots stay readable."""
    xs = [f.pitch_xy[:, 0] for f in result.frames if len(f.detections)]
    side_right = (
        np.nanmean(np.concatenate(xs)) > result.pitch.length / 2 if xs else True
    )
    width = radar.shape[1]
    keep = int(width * 0.56)
    return radar[:, width - keep :] if side_right else radar[:, :keep]


def still(shot_id: str, name: str) -> Path:
    result = analyse(shot_id)
    frame = clean(
        tv.VideoReader(SHOTS / "clips" / f"{shot_id}.mp4").read(ANCHOR), shot_id
    )
    path = MEDIA / f"{name}.jpg"
    cv2.imwrite(
        str(path),
        draw(result, frame, ANCHOR, RADAR_CORNER.get(shot_id, "top-right")),
        [cv2.IMWRITE_JPEG_QUALITY, 86],
    )
    return path


def card(
    image: Image.Image, values: dict, t: float, goal: bool, outcome: bool = True
) -> None:
    """xG card, bottom left: hidden for t < 0, fades in, the outcome from t = 1 when ``outcome``."""
    if t < 0:
        return
    alpha = min(1.0, t / 0.3)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, w, h = 24, image.height - 196, 400, 172
    d.rounded_rectangle(
        [x0, y0, x0 + w, y0 + h], 14, fill=(16, 18, 22, int(215 * alpha))
    )
    title = ImageFont.truetype(FONT_BOLD, 19)
    body = ImageFont.truetype(FONT, 16)
    d.text(
        (x0 + 18, y0 + 14),
        "Shot quality at contact (xG)",
        font=title,
        fill=(255, 255, 255, int(255 * alpha)),
    )
    for i, (label, value, color) in enumerate(values["rows"]):
        y = y0 + 52 + i * 34
        d.text((x0 + 18, y), label, font=body, fill=(220, 220, 220, int(255 * alpha)))
        bar = int(150 * value)
        rgb = tuple(int(color[k : k + 2], 16) for k in (1, 3, 5))
        d.rounded_rectangle(
            [x0 + 196, y + 2, x0 + 196 + max(bar, 4), y + 18],
            4,
            fill=(*rgb, int(255 * alpha)),
        )
        d.text(
            (x0 + 204 + bar, y),
            f"{value:.2f}",
            font=body,
            fill=(255, 255, 255, int(255 * alpha)),
        )
    if outcome and t >= 1.0:
        result = "GOAL" if goal else "NO GOAL"
        d.rounded_rectangle(
            [x0 + w - 104, y0 + 12, x0 + w - 14, y0 + 40],
            8,
            fill=(27, 175, 122, 235) if goal else (82, 81, 78, 235),
        )
        d.text((x0 + w - 95, y0 + 15), result, font=title, fill=(255, 255, 255, 255))
    image.alpha_composite(layer)


def clean(frame: np.ndarray, shot_id: str) -> np.ndarray:
    out = frame.copy()
    for x0, y0, x1, y1 in WATERMARKS.get(shot_id, []):
        out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), 9)
    return out


def freeze_frame(
    shot_id: str, geo: geometry_xg.ShotGeometry, t: float, inset: Image.Image | None
) -> Image.Image:
    """The contact frame with the geometry building up, and the top-down view in the radar's place."""
    frame = clean(
        tv.VideoReader(SHOTS / "clips" / f"{shot_id}.mp4").read(ANCHOR), shot_id
    )
    image = geometry_xg.freeze_overlay(frame, geo, t)
    alpha = float(np.clip((t - 2.2) / 0.35, 0, 1))
    if inset is not None and alpha > 0:
        panel = inset.copy()
        panel.putalpha(panel.getchannel("A").point(lambda v: int(v * alpha)))
        image.alpha_composite(
            panel, (image.width - panel.width - 14, image.height - panel.height - 14)
        )
    return image


def save_png(image: Image.Image, folder: Path, i: int) -> None:
    image.convert("RGB").save(folder / f"{i:04d}.png")


def clip(
    shot_id: str, values: dict, goal: bool, seconds: tuple[float, float] = (-3.4, 2.4)
) -> Path:
    """Play up to contact, freeze while the measured geometry and the xG card appear, then play on."""
    result = analyse(shot_id)
    geo = geometry_xg.load(shot_id)
    geometry_xg.check(geo)
    inset = geometry_inset(geo)
    reader = tv.VideoReader(SHOTS / "clips" / f"{shot_id}.mp4")
    start, end = ANCHOR + int(seconds[0] * FPS), ANCHOR + int(seconds[1] * FPS)
    path = MEDIA / "feature_clip.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        folder, i = Path(tmp), 0

        def played(index: int, card_t: float | None) -> Image.Image:
            frame = draw(
                result,
                clean(reader.read(index), shot_id),
                index,
                RADAR_CORNER.get(shot_id, "top-right"),
            )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert(
                "RGBA"
            )
            if card_t is not None:
                card(image, values, card_t, goal)
            return image

        for index in range(start, ANCHOR):
            save_png(played(index, None), folder, i)
            i += 1
        for k in range(int(FREEZE_SECONDS * FPS)):
            t = k / FPS
            image = freeze_frame(shot_id, geo, t, inset)
            card(image, values, t - 2.9, goal, outcome=False)
            save_png(image, folder, i)
            i += 1
        for index in range(ANCHOR, end):
            save_png(played(index, 0.4 + (index - ANCHOR) / FPS), folder, i)
            i += 1
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                f"{tmp}/%04d.png",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
        )
    still = freeze_frame(shot_id, geo, 10.0, None)
    still.convert("RGB").save(MEDIA / "freeze.jpg", quality=88)
    geometry_xg.diagram(geo, OUT / "xg/figures/geometry.svg")
    return path


def geometry_inset(geo: geometry_xg.ShotGeometry) -> Image.Image:
    path = CACHE / "geometry_inset.png"
    geometry_xg.diagram(geo, path, inset=True)
    panel = Image.open(path).convert("RGBA")
    width = 300
    panel = panel.resize(
        (width, round(panel.height * width / panel.width)), Image.LANCZOS
    )
    backdrop = Image.new("RGBA", panel.size, (16, 18, 22, 215))
    backdrop.alpha_composite(panel)
    return backdrop


def xg_rows(row) -> list:
    return [
        ("Video only", float(row.p_video), "#2a78d6"),
        ("Video + geometry", float(row.p_geometry), "#1baf7a"),
        ("StatsBomb", float(row.p_vendor), "#9a9994"),
    ]


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    shots = paired_shots().set_index("shot_id")
    meta = {}
    for name, (shot_id, caption) in [*GALLERY.items(), ("feature", (FEATURE, ""))]:
        row = shots.loc[shot_id]
        still(shot_id, name)
        meta[name] = {
            "shot_id": shot_id,
            "caption": caption,
            "match": row.match,
            "goal": bool(row.goal),
            "xg": {
                "video": row.p_video,
                "geometry": row.p_geometry,
                "vendor": row.p_vendor,
            },
        }
        print(name, "done", flush=True)
    row = shots.loc[FEATURE]
    clip(FEATURE, {"rows": xg_rows(row)}, bool(row.goal))
    meta["feature"]["geometry"] = geometry_xg.load(FEATURE).features
    (MEDIA / "meta.json").write_text(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
