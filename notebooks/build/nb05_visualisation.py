"""05_visualisation.ipynb: `tv.viz` frame drawing, radar, video and analysis plots."""

from common import (
    DEMO_DETECTOR,
    REPO_URL,
    SAM2_GUARD,
    SAM2_INSTALL,
    Notebook,
)

from tactifoot_vision.data.demo import ATTRIBUTION as DEMO_ATTRIBUTION


def notebook() -> Notebook:
    nb = Notebook("05_visualisation")
    nb.md(f"""
# Results visualisation with `tv.viz`

{nb.badge}

Two kinds of output:

* **Images and video** (OpenCV, BGR arrays): `FrameAnnotator` draws a
  `FrameResult` onto its frame, `PitchRadar` draws a top-down pitch,
  `overlay` combines them and `render_video` writes the whole clip.
* **Analysis figures** (matplotlib): `plot_heatmap`, `plot_tracks`,
  `plot_distance_histogram`, `draw_pitch` and `show`.

Colours follow the team ids from the pipeline everywhere, so figures match the
video.

It runs on open data, the same from a checkout and on Google Colab: the
SoccerTrack v2 demo match from `tv.data.download_demo` (about 300 MB,
CC BY 4.0), on which a player detector trains for a few minutes as in the
[overview]({REPO_URL}/notebooks/00_overview.ipynb). **On Colab**, pick a GPU first (Runtime,
Change runtime type, T4 GPU); the notebook then takes about ten minutes. The
SAM2 masks need SAM2, which has an install cell that is off by default (see
Annotated frames).

The demo data needs your own free Hugging Face token, as a Colab secret
named `HF_TOKEN` (or the environment variable of that name, or a saved
`hf auth login`). The [overview]({REPO_URL}/notebooks/00_overview.ipynb)
lists the three steps to get one.

Part of the [TactiFoot Vision notebooks]({REPO_URL}/notebooks/README.md). The
pipeline itself is explained in
[04_inference]({REPO_URL}/notebooks/04_inference.ipynb); here it only
produces something to draw.
""")
    nb.setup(
        imports="""
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

import tactifoot_vision as tv
"""
    )

    nb.md(f"""
## Something to draw

The demo match, the overview's detector, and the pipeline of
[04_inference]({REPO_URL}/notebooks/04_inference.ipynb) on the first ten
seconds of the demo clip: the camera does not move, so `fixed_homography`
and `undistortion` take its calibration. The clip is one static fisheye
camera that sees the whole pitch, 4096x1080 pixels, so the comparisons below
show a window around the players.
""")
    nb.code("""
demo = tv.data.download_demo(DATA / "demo")
dataset = tv.load_dataset(demo.detect_yaml)
clip = tv.data.load_labelled_clip(demo.clip_dir)
video = tv.VideoReader(clip.video_path)
print(dataset)
""")
    nb.code(DEMO_DETECTOR, live_log=True)
    nb.code("""
detector = training.load(conf=0.3)
pipeline = tv.Pipeline(
    detector=detector,
    fixed_homography=demo.calibration,
    undistortion=demo.calibration,
    tracker="bytetrack",
    team_classifier=tv.teams.TeamClassifier(embedder="siglip"),
)
result = pipeline.run(clip.video_path, end=250, progress=False)  # 10 seconds
FRAME = 200  # the frame most cells draw
frame_result = result[FRAME]
frame = video.read(frame_result.index)
print(len(result), "frames,", len(result.track_ids), "tracks")

# 1280x540 pixels around the middle of the players
centre_x, centre_y = np.median((frame_result.detections.xyxy[:, :2] + frame_result.detections.xyxy[:, 2:]) / 2, axis=0)
left = int(np.clip(centre_x - 640, 0, video.width - 1280))
top = int(np.clip(centre_y - 270, 0, video.height - 540))
window = np.s_[top : top + 540, left : left + 1280]
""")

    nb.md("""
## Annotated frames

`FrameAnnotator` has two styles. `standard` draws boxes; `video_game` draws
ellipses at the players' feet and a triangle over the ball. Both label people
with their track id, colour them by team (referees in the default colour),
draw the confident pitch keypoints and project the pitch lines with the
homography.

Two things the demo cannot show. It labels the 22 players only, so there
is no ball, goalkeeper or referee: the ball's triangle, its trail and the
ball-carrier ring below draw nothing here, and on footage with a ball
detector the same calls draw them. And the pitch lines are off on the frame
as recorded (`draw_pitch_lines=False`, as in the overview), because the
demo's homography holds in the undistorted image, where the lines are
straight, while this fisheye frame bends them; they are drawn on the
undistorted frame further down.
""")
    nb.code("""
styles = {
    style: tv.viz.FrameAnnotator(style=style, draw_pitch_lines=False).annotate(frame, frame_result)
    for style in ("standard", "video_game")
}
tv.show(list(styles.values()), titles=list(styles), cols=1, width=16)
""")
    nb.md("""
Every layer has a flag: `draw_boxes`, `draw_labels`, `draw_keypoints`,
`draw_pitch_lines` and `draw_masks`. Turning one off at a time shows what each
adds. `draw_keypoints` draws the pitch landmarks a keypoint model found,
orange where the homography agrees with them and red where it does not, a
quick check of the pitch fit; with a fixed homography no keypoint model runs,
so the demo has none (04_inference trains one on rendered pitch views).
""")
    nb.code("""
views = {
    "all layers": tv.viz.FrameAnnotator(draw_pitch_lines=False).annotate(frame, frame_result),
    "draw_boxes=False": tv.viz.FrameAnnotator(draw_boxes=False, draw_pitch_lines=False).annotate(frame, frame_result),
    "draw_labels=False": tv.viz.FrameAnnotator(draw_labels=False, draw_pitch_lines=False).annotate(frame, frame_result),
}
tv.show([view[window] for view in views.values()], titles=list(views), cols=1, width=12)
""")
    nb.md("""
`draw_pitch_lines` projects the pitch markings into the frame through the
inverse homography. `cv2.fisheye.undistortImage`, with the camera of the
demo's `FisheyeUndistortion`, gives the undistorted frame the homography is
of, and there the projected lines lie on the painted ones. The people's
boxes are in the frame as recorded, so this view draws the pitch layers
only.
""")
    nb.code("""
undistortion = tv.pitch.FisheyeUndistortion.load(demo.calibration)
undistorted = cv2.fisheye.undistortImage(
    frame, undistortion.camera, undistortion.distortion, Knew=undistortion.new_camera)
pitch_layers = tv.viz.FrameAnnotator(draw_boxes=False, draw_labels=False)
tv.show(pitch_layers.annotate(undistorted, frame_result), titles="pitch layers only, on the undistorted frame",
        width=16)
""")
    nb.md("""
Colours and thresholds are arguments too: `team_colors`, `default_color`,
`ball_color`, `pitch_line_color` and `keypoint_threshold`.
""")
    nb.code("""
custom = tv.viz.FrameAnnotator(
    style="video_game", team_colors=["#E63946", "#F1FAEE"], default_color="#222222",
    ball_color="#FFD400", pitch_line_color="#A8DADC", keypoint_threshold=0.8, draw_pitch_lines=False,
)
tv.show(custom.annotate(frame, frame_result)[window], width=12)
""")
    nb.md("""
`draw_points` puts arbitrary pitch positions on the radar, for example
positions from another source. Here: the pitch positions of everyone in the
frame, all in one colour.
""")
    nb.code("""
tracks = result.to_dataframe()
in_frame = tracks[(tracks["frame"] == frame_result.index) & tracks["pitch_x"].notna()]
tv.show(radar.draw_points(in_frame[["pitch_x", "pitch_y"]].to_numpy(), "#FFD400"),
        titles="draw_points: positions of everyone in the frame", width=7)
""")
    nb.md("""
`trail_seconds` draws the ball's recent path. Drawing keeps no memory between
frames, so `annotate` takes the earlier frames as `history`, oldest first, and
`result.window(index, seconds)` gives them (ADR 0003). Each earlier ball
position is projected through the current frame's homography, so the trail
stays on the grass when the camera pans. It breaks where the ball was not
seen, fades to nothing at `trail_seconds`, and is not drawn on a frame
without a homography. Positions that `track_ball` filled in draw like any
other.
""")
    nb.code("""
TRAIL = 1.0
trail = tv.viz.FrameAnnotator(style="video_game", trail_seconds=TRAIL, draw_pitch_lines=False).annotate(
    frame, frame_result, history=result.window(frame_result.index, TRAIL))
tv.show(trail[window], titles=f"trail_seconds={TRAIL}", width=12)
""")
    nb.md("""
`annotate(..., ball_carrier=tracker_id)` rings the ball carrier at their
feet in `ball_color`. `tv.analysis.ball_carriers(result)` gives one
`tracker_id`, or `None` for a loose ball, per frame of `result.frames`; `None`
or an id nobody in the frame has draws no ring. Here: the first frame with a
carrier, if there is one.
""")
    nb.code("""
carriers = tv.analysis.ball_carriers(result)
carried = next(((f, c) for f, c in zip(result, carriers) if c is not None), None)
if carried is None:
    print("no frame of the demo clip has a ball carrier")
else:
    carrier_result, carrier = carried
    ring = tv.viz.FrameAnnotator(style="video_game", draw_pitch_lines=False).annotate(
        video.read(carrier_result.index), carrier_result, ball_carrier=carrier)
    display(tv.show(ring, titles=f"ball_carrier={carrier} at frame {carrier_result.index}", width=16))
""")
    nb.md(f"""
`draw_masks=True` draws segmentation masks, which only a mask tracker (SAM2)
produces and only `Pipeline(keep_masks=True)` keeps. With a box tracker there
are no masks; the annotator then warns once and draws the rest. SAM2 is not
on PyPI and needs a GPU: as in
[04_inference]({REPO_URL}/notebooks/04_inference.ipynb), set
`INSTALL_SAM2 = True` in the next cell to install it here, in about a
minute on Colab. Without it the SAM2 cell prints one line and shows the
warning instead. The saved outputs of this notebook come from a run with
SAM2.
""")
    nb.code(SAM2_INSTALL)
    nb.code(SAM2_GUARD)
    nb.code("""
if has_sam2:
    masked = tv.Pipeline(
        detector=detector, fixed_homography=demo.calibration, undistortion=demo.calibration,
        tracker=sam2_tracker, keep_masks=True,
    ).run(clip.video_path, start=frame_result.index - 30, end=frame_result.index + 1, progress=False)
    masks_view = tv.viz.FrameAnnotator(draw_masks=True, draw_keypoints=False, draw_pitch_lines=False).annotate(
        frame, masked[-1])
    display(tv.show(masks_view[window], titles="SAM2 masks", width=12))
else:
    no_masks = tv.viz.FrameAnnotator(draw_masks=True, draw_pitch_lines=False).annotate(frame, frame_result)
    display(tv.show(no_masks[window], titles="draw_masks=True without masks", width=12))
""")

    nb.md("""
## Pitch radar and overlay

`PitchRadar` draws a frame's pitch positions top-down, coloured like the video.
`draw_ids` writes track ids into the dots; size, colours and line widths are
arguments.
""")
    nb.code("""
radar = tv.viz.PitchRadar()
radar_ids = tv.viz.PitchRadar(draw_ids=True, width_px=900, pitch_color="#1D3557", line_color="#A8DADC")
tv.show([radar.draw(frame_result), radar_ids.draw(frame_result)], titles=["PitchRadar()", "draw_ids, custom colours"],
        cols=2, width=13)
""")
    nb.md("""
`trail_seconds` draws a tail behind each person, so the teams' movement shows.
Like the ball trail, `draw` takes the earlier frames as `history`. Each tail
joins the earlier pitch positions of the same track id, in the person's
colour (the default colour without a team), breaks where the track was not
seen, fades to nothing at `trail_seconds` and lies under the dots.
""")
    nb.code("""
tails = tv.viz.PitchRadar(trail_seconds=2.0).draw(frame_result, history=result.window(frame_result.index, 2.0))
tv.show(tails, titles="trail_seconds=2.0", width=7)
""")
    nb.md("""
`draw_visible_area` outlines the frame's visible area, the part of the pitch
the camera shows, under the players, in `visible_area_color`. It is
`tv.pitch.visible_area`, the polygon freeze frames export, and needs the
frame size: pass `frame_size=result.frame_size`, or the `frame`. There is no
outline without a homography or when part of the frame lies beyond the
horizon. On the demo the homography is of the undistorted image, so the
outline of the fisheye frame is approximate.
""")
    nb.code("""
area = tv.pitch.visible_area(frame_result.homography, result.frame_size)
print("visible area:", "none (part of the frame lies beyond the horizon)" if area is None else area.round(1).tolist())
visible = tv.viz.PitchRadar(draw_visible_area=True).draw(frame_result, frame_size=result.frame_size)
tv.show(visible, titles="draw_visible_area=True", width=7)
""")
    nb.md("""
`velocity_seconds` draws each person's velocity as an arrow in their colour,
from the dot to where the velocity takes them in that many seconds.
`draw` takes the frame's velocities, as `tv.analysis.velocities(result)`
gives them: one m/s array per frame of `result.frames`, row for row with
the detections. A person without a velocity (no track, too few positions in
the window, or too fast) gets no arrow.
""")
    nb.code("""
velocities = tv.analysis.velocities(result)
arrows = tv.viz.PitchRadar(velocity_seconds=1.0).draw(frame_result, velocities=velocities[FRAME])
tv.show(arrows, titles="velocity_seconds=1.0", width=7)
""")
    nb.md("""
`overlay` pastes the radar onto the frame: `position` picks the corner,
`width_fraction` the size and `alpha` the opacity.
""")
    nb.code("""
annotated = styles["video_game"]
image = radar.draw(frame_result)
tv.show([
    tv.viz.overlay(annotated, image),
    tv.viz.overlay(annotated, image, position="top-right", width_fraction=0.35, alpha=1.0),
], titles=["default: bottom-center, 25%, alpha 0.8", "top-right, 35%, opaque"], cols=1, width=16)
""")

    nb.md("""
## Rendering a video

`render_video` re-reads the source video and writes every processed frame,
annotated and with the radar overlay, to an mp4: here the ten seconds the
pipeline ran on. It refuses a video whose frame size differs from the
result's and a radar or annotator drawing another pitch, and warns when the
frame rates differ. Pass your own annotator and radar, or `radar=False` for
none. With a `trail_seconds` annotator or radar, `render_video` passes each
frame its `history` itself, for the longer of the two, and the radar the
frame size its visible area needs. A `velocity_seconds` radar gets each
frame's `tv.analysis.velocities` too. `ball_carriers`, one per frame as
`tv.analysis.ball_carriers` returns them, rings each frame's carrier. The
video stays in `OUT`, outside the notebook: three of its frames show here.
""")
    nb.code("""
video_path = tv.viz.render_video(
    result, clip.video_path, OUT / "annotated.mp4",
    annotator=tv.viz.FrameAnnotator(style="video_game", trail_seconds=TRAIL, draw_pitch_lines=False),
    radar=tv.viz.PitchRadar(draw_ids=True, trail_seconds=2.0, draw_visible_area=True, velocity_seconds=1.0),
    overlay_position="bottom-right", overlay_width_fraction=0.3,
    ball_carriers=carriers, progress=False,
)
rendered = tv.VideoReader(video_path)
print(rendered)
shown = (0, len(rendered) // 2, len(rendered) - 1)
tv.show([rendered.read(i) for i in shown], titles=[f"{i / rendered.fps:.1f} s" for i in shown], cols=1, width=16)
""")
    nb.md("""
OpenCV writes MPEG-4 Part 2, which browsers do not play. With `ffmpeg`
installed (Colab has it) the next cell re-encodes a small H.264 copy next to
it, to download and play anywhere; on Colab it is in the Files panel on the
left.
""")
    nb.code("""
if shutil.which("ffmpeg"):
    web_video = video_path.with_name("annotated_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vf", "scale=960:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(web_video)], check=True)
    print(f"{web_video} ({web_video.stat().st_size / 1e6:.1f} MB)")
else:
    print("ffmpeg not found; open", video_path, "in a video player")
""")

    nb.md("""
## Analysis plots

`plot_heatmap` shows where players spent time, for everybody or one team in
its team colour. Like the other plotting helpers it draws into a given `ax`,
so several fit in one figure.
""")
    nb.code("""
fig = Figure(figsize=(16, 4), layout="constrained")
for ax, team in zip(fig.subplots(1, 3), (None, 0, 1), strict=True):
    tv.viz.plot_heatmap(result, team=team, ax=ax)
fig
""")
    nb.md("""
`plot_tracks` draws trajectories on the pitch, by default of the tracks with
the most positions; `track_ids` picks specific ones.
""")
    nb.code("""
tv.viz.plot_tracks(result, max_tracks=8)
""")
    nb.code("""
tv.viz.plot_tracks(result, track_ids=result.track_ids[:3])
""")
    nb.md("""
`plot_distance_histogram` plots any set of distances with the median marked.
Here it shows how
far each track moved during the ten seconds. Detection jitter adds to these
distances, so they overestimate what the players really ran.
""")
    nb.code("""
tracks = result.to_dataframe()
people = tracks[(tracks["object"] == "person") & tracks["pitch_x"].notna()].sort_values("frame")
steps = people.groupby("track_id")[["pitch_x", "pitch_y"]].diff()
covered = np.hypot(steps["pitch_x"], steps["pitch_y"]).groupby(people["track_id"]).sum()
tv.viz.plot_distance_histogram(covered[covered > 0], bins=15, title="Distance covered per track in 10 s",
                               xlabel="metres")
""")
    nb.md("""
`draw_pitch` gives an empty pitch to plot anything on, in pitch units with `y`
pointing down like the radar. Here: every player position of the ten
seconds.
""")
    nb.code("""
fig = tv.viz.draw_pitch(pitch=result.pitch, size=8)
fig.axes[0].scatter(people["pitch_x"], people["pitch_y"], s=2, alpha=0.3, color="#FFD400")
fig
""")
    nb.md("""
`show` is the image grid used throughout these notebooks: one image or a list,
optional titles, `cols` per row and `width` in inches. Images are BGR by
default (pass `bgr=False` for RGB arrays); 2-D arrays show in grayscale.
""")
    nb.code("""
small = [video.read(round(seconds * video.fps))[::4, ::4] for seconds in (0, 10, 20)]
gray = [img.mean(axis=2).astype(np.uint8) for img in small]
tv.show(small + gray, titles=["0 s", "10 s", "20 s"] * 2, cols=3, width=14)
""")

    nb.md(f"""
## Sources

{DEMO_ATTRIBUTION}
""")
    return nb
