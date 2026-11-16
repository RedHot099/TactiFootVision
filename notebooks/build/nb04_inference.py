"""04_inference.ipynb: predictions, pitch geometry, trackers, teams and the full pipeline."""

from common import (
    DEMO_DETECTOR,
    DETECTOR_EPOCHS,
    DETECTOR_IMGSZ,
    DETECTOR_WEIGHTS,
    PITCH_TRAINING_DATASET,
    REPO_URL,
    RFDETR_GUARD,
    SAM2_GUARD,
    SAM2_INSTALL,
    Notebook,
)

from tactifoot_vision.data.demo import ATTRIBUTION as DEMO_ATTRIBUTION

# Epochs of the pitch keypoint model: fewer than 03's 30, to keep the
# notebook near ten minutes on a T4.
PITCH_EPOCHS = 40


def notebook() -> Notebook:
    nb = Notebook("04_inference")
    nb.md(f"""
# Inference: from frames to tracked players on the pitch

{nb.badge}

The inference modules turn match video into data:

1. **Models** (`tv.models`) detect people and the ball, and find pitch landmarks.
2. **Pitch geometry** (`tv.pitch`) maps frame pixels to pitch metres.
3. **Trackers** (`tv.tracking`) give each person an id that lasts across frames.
4. **Teams** (`tv.teams`) split players into teams by kit, without labels.
5. **The pipeline** (`tv.pipeline`) chains all of it over a video and returns a
   `PipelineResult` with tables, StatsBomb-style freeze frames and files.

Each part works on its own; the pipeline only connects them.

It runs on open data, the same from a checkout and on Google Colab: the
SoccerTrack v2 demo match from `tv.data.download_demo` (about 300 MB,
CC BY 4.0), on which a player detector trains for a few minutes as in the
[overview]({REPO_URL}/notebooks/00_overview.ipynb), and pitch views rendered
by the package, on which a short pitch keypoint model trains. Both are
demonstrations of the interface, not models for real matches. **On Colab**,
pick a GPU first (Runtime, Change runtime type, T4 GPU); the notebook then
takes about ten minutes. The RF-DETR and SAM2 cells run only where those
optional parts are installed: Colab skips RF-DETR, and SAM2 has an install
cell that is off by default (see Trackers).

The demo data needs your own free Hugging Face token, as a Colab secret
named `HF_TOKEN` (or the environment variable of that name, or a saved
`hf auth login`). The [overview]({REPO_URL}/notebooks/00_overview.ipynb)
lists the three steps to get one.

Part of the [TactiFoot Vision notebooks]({REPO_URL}/notebooks/README.md);
training is covered in
[03_training_and_evaluation]({REPO_URL}/notebooks/03_training_and_evaluation.ipynb).
""")
    nb.setup(
        imports="""
import logging
import os
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

import tactifoot_vision as tv
"""
    )

    nb.md(f"""
## The demo match and a detector

`download_demo` fetches the demo match once; a second call does nothing. It
holds a detection dataset and a 30-second clip with tracking truth, both from
one static camera that sees the whole pitch through a fisheye lens, 4096x1080
pixels, so a player is about 25x60 pixels. The demo labels the 22 players
only: no ball, referee or goalkeeper boxes.

The detector is the overview's: the COCO weights `{DETECTOR_WEIGHTS}` that
Ultralytics publishes, trained for {DETECTOR_EPOCHS} epochs on the demo frames
at {DETECTOR_IMGSZ} pixels on the long side, about five minutes on a Colab T4.
""")
    nb.code("""
demo = tv.data.download_demo(DATA / "demo")
dataset = tv.load_dataset(demo.detect_yaml)
clip = tv.data.load_labelled_clip(demo.clip_dir)
video = tv.VideoReader(clip.video_path)
print(dataset)
print(f"demo clip: {len(clip)} frames at {clip.fps:g} fps, {video.width}x{video.height} pixels")
""")
    nb.code(DEMO_DETECTOR, live_log=True)

    nb.md("""
## Single-image predictions

A model is a callable: give it a BGR frame. Detectors return `sv.Detections`
with the class names in `data["class_name"]`, whatever the backend, so YOLO
and RF-DETR are interchangeable. Supervision's annotators draw them directly.
`TrainResult.load` rebuilds the trained detector, with its options.
""")
    nb.code("""
detector = training.load(conf=0.3)
frame = video.read(100)
detections = detector(frame)
print(f"{len(detections)} objects", dict(Counter(detections.data["class_name"].tolist())))
annotated = sv.BoxAnnotator(thickness=2).annotate(frame.copy(), detections)
tv.show(annotated, width=16)
""")
    nb.md("""
A closer look at the middle of the players, and the same window through
RF-DETR with the COCO weights its library publishes. Those know people as
`person`, COCO's class, where the detector trained on the demo says `player`:
the class names are the model's own. The RF-DETR cells run only where the
`rfdetr` extra is installed: the extra pins numpy and transformers below the
versions Colab ships, so on Colab they print one line and skip, and a local
environment with the extra (`pip install "tactifoot-vision[rfdetr]"`, or
`uv sync --extra rfdetr` in a checkout) runs them. The saved outputs of this
notebook come from such a run.
""")
    nb.code(RFDETR_GUARD)
    nb.code("""
# 640x320 pixels around the middle of the players
centre_x, centre_y = np.median(detections.get_anchors_coordinates(sv.Position.CENTER), axis=0)
left = int(np.clip(centre_x - 320, 0, video.width - 640))
top = int(np.clip(centre_y - 160, 0, video.height - 320))
window = np.s_[top : top + 320, left : left + 640]
views, titles = [annotated[window]], ["trained YOLO11s, run on the whole frame"]
if has_rfdetr:
    # RF-DETR suggests optimize_for_inference() on its first prediction and torch warns
    # about a meshgrid argument inside RF-DETR; both are benign here.
    logging.getLogger("rfdetr.detr").addFilter(lambda record: "not optimized for inference" not in record.getMessage())
    warnings.filterwarnings("ignore", message="torch.meshgrid")
    rfdetr = tv.load_model("rfdetr", size="base", conf=0.5)  # weights=None: the COCO weights
    found = rfdetr(frame[window])
    print("RF-DETR (COCO weights):", dict(Counter(found.data["class_name"].tolist())))
    view = sv.BoxAnnotator(thickness=1).annotate(frame[window].copy(), found)
    views.append(sv.LabelAnnotator(text_scale=0.4).annotate(view, found, labels=list(found.data["class_name"])))
    titles.append("RF-DETR with COCO weights, run on this window")
tv.show(views, titles=titles, cols=1, width=10)
""")

    nb.md("""
## Pitch geometry

`SoccerPitch` describes the pitch: 32 landmarks in pitch units, with the origin
in a corner, `x` along the length and `y` across. The default is 105 × 68 m;
`SoccerPitch(120, 80)` gives StatsBomb units with the same proportions.
""")
    nb.code("""
pitch = tv.SoccerPitch()
print(pitch, "|", tv.SoccerPitch(120, 80))
fig = tv.viz.draw_pitch(pitch=pitch, size=8)
ax = fig.axes[0]
ax.scatter(*pitch.vertices.T, s=25, color="#FFA500", zorder=3)
for label, (x, y) in zip(pitch.labels, pitch.vertices, strict=True):
    ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="white")
fig
""")
    nb.md("""
### A calibrated static camera

A homography maps frame pixels to the pitch, but only in an image without
lens distortion. The demo camera does not move and comes calibrated:
`demo.calibration` holds its fisheye lens (`K`, `D` and `Knew` as OpenCV's
`cv2.fisheye` takes them) and `frame_to_pitch`, the homography of the
undistorted image. `FisheyeUndistortion` moves frame points to where the
undistorted image shows them, and `frame_to_pitch` then maps those pixels to
metres; here the players' feet, the bottom centre of their boxes. The
dataset's own fit is about 1.3 m off at the median, worse at the far corners.
""")
    nb.code("""
undistortion = tv.pitch.FisheyeUndistortion.load(demo.calibration)
with np.load(demo.calibration) as calibration:
    homography = calibration["frame_to_pitch"]
feet = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
on_pitch = tv.pitch.frame_to_pitch(undistortion.points(feet), homography)
pd.DataFrame(on_pitch, columns=["x (m)", "y (m)"]).assign(
    class_name=detections.data["class_name"]).head().round(1)
""")
    nb.md("""
`pitch_to_frame` goes the other way. In the undistorted frame the pitch's
lines between landmarks (white), projected through the homography, should
run along the painted ones, with the players' feet (orange) on the grass.
""")
    nb.code("""
undistorted = cv2.fisheye.undistortImage(
    frame, undistortion.camera, undistortion.distortion, Knew=undistortion.new_camera)
landmarks = tv.pitch.pitch_to_frame(pitch.vertices, homography).round().astype(int)
for i, j in tv.pitch.EDGES:
    cv2.line(undistorted, tuple(landmarks[i].tolist()), tuple(landmarks[j].tolist()), (255, 255, 255), 2)
for x, y in undistortion.points(feet):
    cv2.circle(undistorted, (int(x), int(y)), 8, (0, 165, 255), -1)
tv.show(undistorted, width=16)
""")

    nb.md(f"""
### Moving cameras: a pitch keypoint model

Broadcast footage pans and zooms, so no single matrix serves a whole match
and there is no calibration file. There a pose model finds the 32 landmarks
in every frame, and `HomographyEstimator` fits that frame's homography to
them. The model here learns from pitch views the package renders, as in
[03_training_and_evaluation]({REPO_URL}/notebooks/03_training_and_evaluation.ipynb):
`broadcast_homographies` draws random views of a main broadcast camera,
`synthetic_pitch_clip` renders the markings through them with noise and
player-sized boxes in front of the lines, and `Dataset.from_clips(...,
task="pose")` projects the landmarks into every frame. Rendering takes
seconds on a CPU, and {PITCH_EPOCHS} epochs of training a couple of minutes on a
T4. This model only knows rendered pitches: one for real footage needs real
labelled frames, which the synthetic pitch does not replace.
""")
    nb.code(f"""
{PITCH_TRAINING_DATASET.strip()}
print(pitch_data)
""")
    nb.code(
        f"""
pose_run = tv.train("yolo_pose", pitch_data, weights="yolov8n-pose.pt", epochs={PITCH_EPOCHS}, batch_size=16,
                    mosaic=0.0, output_dir=OUT / "runs", name="pitch_pose", exist_ok=True)
""",
        live_log=True,
    )
    nb.md("""
How a frame becomes the model's input matters to a pose model: `resize`
`"letterbox"` (the default) pads it only to a multiple of the stride, while
`"square"` pads it to a grey square, as Ultralytics' training does with every
image. A model that learnt from those bands can miss the pitch without them.
Scoring both ways on the test clip, which took no part in training, tells
which one this model needs; the better one is kept. A low confidence
threshold keeps the uncertain landmarks of a short run.
""")
    nb.code("""
pitch_scores = {
    resize: tv.evaluate(pose_run.load(conf=0.05, resize=resize), pitch_data, "test")
    for resize in ("letterbox", "square")
}
resize = max(pitch_scores, key=lambda name: pitch_scores[name].pck)
pitch_model = pose_run.load(conf=0.05, resize=resize)
print("kept:", resize)
pd.DataFrame({name: m.as_dict() for name, m in pitch_scores.items()}).T.round(3)
""")
    nb.md("""
A view the model has never seen: one frame of a new set of rendered views.
The model returns `sv.KeyPoints`, for each pitch it finds 32 landmark
positions with a confidence each, in the order of `SoccerPitch`'s vertices.
""")
    nb.code("""
held_out = tv.data.synthetic_pitch_clip(
    OUT / "pitch_views" / "held_out" / "held_out_%06d.jpg",
    tv.data.broadcast_homographies(25, size, seed=100),
    size,
    line_width=0.25,
    noise=4,
    occluders=3,
    seed=100,
)
view = tv.VideoReader(held_out.video_path).read(0)
keypoints = pitch_model(view)
print("xy:", keypoints.xy.shape, " confidence:", keypoints.confidence.shape)
""")
    nb.md("""
`HomographyEstimator.update(keypoints)` fits the homography to the landmarks
with a confidence of at least `min_confidence` (RANSAC; `used_indices` lists
them) and averages the last `smoothing_window` fits to steady it over time.
These views are independent, like camera cuts, so nothing is averaged. The
clip knows each view's true homography, so the distance of every landmark
the model found from where it really is can be measured in metres. In the
frame, the landmarks projected through the fitted homography (white) should
sit on the detected ones (orange).
""")
    nb.code("""
estimator = tv.pitch.HomographyEstimator(pitch, min_confidence=0.3, smoothing_window=1)
fitted = estimator.update(keypoints)
check = view.copy()
if fitted is None:
    print("No homography: fewer than four landmarks reach min_confidence in this view")
else:
    used = estimator.used_indices
    found_at = tv.pitch.frame_to_pitch(keypoints.xy[0][used], held_out.frames[0].homography)
    error = np.linalg.norm(found_at - pitch.vertices[used], axis=1)
    print(f"{len(used)} landmarks given to RANSAC, a median {np.median(error):.2f} m from where they are")
    for x, y in keypoints.xy[0][used]:
        cv2.circle(check, (int(x), int(y)), 6, (0, 165, 255), -1)
    for x, y in tv.pitch.pitch_to_frame(pitch.vertices, fitted):
        if 0 <= x < size[0] and 0 <= y < size[1]:
            cv2.circle(check, (int(x), int(y)), 3, (255, 255, 255), -1)
tv.show(check, width=8)
""")

    nb.md("""
## Trackers

A tracker gives each person a `tracker_id` that stays the same from frame to
frame. Trackers are created by name with `create_tracker`; all of them have
`reset(fps)` and `update(detections, frame)`.
""")
    nb.code("""
print("available:", tv.tracking.available_trackers())
tracker = tv.tracking.create_tracker("bytetrack", lost_track_buffer=30)
tracker.reset(fps=video.fps)
ids_per_frame = []
for _, image in video.frames(start=0, end=50):
    tracked = tracker.update(detector(image), image)
    ids_per_frame.append(set(tracked.tracker_id.tolist()))
stable = set.intersection(*ids_per_frame)
print(f"{len(set.union(*ids_per_frame))} ids over 50 frames; {len(stable)} present in every frame")
""")
    nb.md("""
The clip comes with the true boxes and identities, so a tracker can be
scored. `Pipeline` runs the detector and the tracker over the frames (the
other parts come later), and `evaluate_tracking` compares the tracks with the
truth: MOTA and IDF1 of 1 would be a perfect run, and `id_switches` counts
the times a true player changed id. HOTA needs TrackEval, installed by hand,
so it is off here.
""")
    nb.code("""
TRACKED = 250  # frames scored: the first ten seconds of the clip
bytetrack_run = tv.Pipeline(detector=detector, tracker="bytetrack").run(clip.video_path, end=TRACKED, progress=False)
tracking = {"ByteTrack": tv.evaluation.evaluate_tracking(bytetrack_run, clip, hota=False)}
pd.Series(tracking["ByteTrack"].as_dict()).dropna().round(3).to_frame("ByteTrack")
""")
    nb.md("""
### SAM2

ByteTrack matches boxes only and is fast. SAM2 tracks segmentation masks,
which holds identities better through overlaps, and is re-seeded from the
detector when new people enter. It is not on PyPI: the tracker imports it
from a clone of the
[real-time SAM2 repository](https://github.com/Gy920/segment-anything-2-real-time)
and loads a checkpoint (`sam2.1_hiera_tiny.pt`, from Meta) from the clone,
and it needs the `sam2` extra and a GPU. Without them the SAM2 cells print
one line and skip. To try it, set `INSTALL_SAM2 = True` in the next cell and
run it: it installs the extra, clones the repository into `DATA` and
downloads the checkpoint, in about a minute on Colab, and prints how long it
took. A checkout can instead run `uv sync --extra sam2` and clone the
repository there. The saved outputs of this notebook come from a run with
SAM2.
""")
    nb.code(SAM2_INSTALL)
    nb.code(SAM2_GUARD)
    nb.md("""
SAM2 on the same frames, scored the same way. With `keep_masks=True` the
pipeline keeps each person's mask in `FrameResult.masks`, cropped to the
pixels it covers, so memory grows with the people's area, not the frame's;
`FrameAnnotator(draw_masks=True)` draws them.
""")
    nb.code("""
if has_sam2:
    sam2_run = tv.Pipeline(detector=detector, tracker=sam2_tracker, keep_masks=True).run(
        clip.video_path, end=TRACKED, progress=False)
    tracking["SAM2"] = tv.evaluation.evaluate_tracking(sam2_run, clip, hota=False)
pd.DataFrame({name: m.as_dict() for name, m in tracking.items()}).dropna().round(3)
""")
    nb.code("""
if has_sam2:
    last = sam2_run[-1]
    print(f"{len(last.masks)} masks in the last frame, {sum(c.size for c in last.masks.crops) / 1e3:.0f} kB")
    masks_view = tv.viz.FrameAnnotator(draw_masks=True, draw_pitch_lines=False, draw_keypoints=False).annotate(
        video.read(last.index), last)
    display(tv.show(masks_view[window], width=10))
""")

    nb.md("""
## Team classification

Teams are found without labels. `extract_crops` cuts the central part of each
player box (shirt and shorts, little grass); an embedder turns each crop into a
feature vector; `TeamClassifier` reduces them with UMAP and clusters them into
two teams. Referees and goalkeepers are handled by the pipeline later.
""")
    nb.code("""
crops = []
for _, image in video.frames(start=0, end=len(clip), stride=75):
    people = detector(image)
    players = people[people.data["class_name"] == "player"]
    crops += [crop for crop in tv.teams.extract_crops(image, players.xyxy, scale=0.8) if crop is not None]
print(len(crops), "player crops; embedders:", tv.teams.available_embedders())
""")
    nb.code("""
classifier = tv.teams.TeamClassifier(embedder="siglip")
teams = classifier.fit_predict(crops)
print(classifier, "| team sizes:", np.bincount(teams).tolist())

examples = [crops[i] for team in (0, 1) for i in np.flatnonzero(teams == team)[:6]]
tv.show([cv2.resize(c, (64, 128)) for c in examples], titles=[f"team {t}" for t in (0, 1) for _ in range(6)],
        cols=6, width=9)
""")
    nb.md("""
The embedder is swappable. ResNet-18 features are cheaper than SigLIP; team
numbers are arbitrary, so agreement is measured up to swapping 0 and 1.
""")
    nb.code("""
resnet_teams = tv.teams.TeamClassifier(embedder="resnet").fit_predict(crops)
agreement = max(np.mean(resnet_teams == teams), np.mean(resnet_teams != teams))
print(f"ResNet and SigLIP agree on {agreement:.0%} of the crops")
""")

    nb.md("""
## The full pipeline

`tv.Pipeline` chains everything per frame: detect, split off the ball, track
the people, find the homography and project everyone onto the pitch. At the
end of the pass each track gets one team by majority vote over the crops
sampled during the pass, goalkeepers join the team whose players are
nearest, and ball positions that jump implausibly far are dropped. Every run
fits the team classifier on its own crops, so one pipeline can process
several matches; `refit_teams=False` keeps a classifier that is already
fitted. Every part is an object you pass in; a tracker can also be given by
name. The demo camera does not move, so `fixed_homography` and
`undistortion` take its calibration, as above, and no keypoint model runs.
""")
    nb.code("""
pipeline = tv.Pipeline(
    detector=detector,
    fixed_homography=demo.calibration,
    undistortion=demo.calibration,
    tracker="bytetrack",
    team_classifier=tv.teams.TeamClassifier(embedder="siglip"),
)
result = pipeline.run(clip.video_path, end=TRACKED, progress=False)
print(len(result), "frames,", len(result.track_ids), "tracks,", f"{result.fps:.0f} fps")
""")
    nb.md("""
A `PipelineResult` is a list of `FrameResult`s. Each holds the tracked people
as `sv.Detections` with `pitch_xy` and `team_id` attached, the ball as
`sv.Detections` with `pitch_xy` (the ball has no team), the pitch keypoints
(none here, with a fixed homography) and the homography.
""")
    nb.code("""
frame_result = result[100]
print("frame", frame_result.index, f"at {frame_result.timestamp:.2f} s")
print("people:", len(frame_result.detections), "ball:", len(frame_result.ball))
print("teams:", Counter(frame_result.team_ids.tolist()), "(-1 = no team: referees)")
print("homography:", None if frame_result.homography is None else frame_result.homography.shape)
""")
    nb.md("""
Because teams are voted per track, every frame of a track carries the same
team. The table counts tracks per class and team. The demo labels players
only, so its detector finds no goalkeeper, referee or ball, and those rules
are described here rather than shown.
""")
    nb.code("""
tracks = result.to_dataframe()
people = tracks[tracks["object"] == "person"]
print("tracks with more than one team:", int((people.groupby("track_id")["team_id"].nunique() > 1).sum()))
people.groupby(["class_name", "team_id"])["track_id"].nunique().unstack(fill_value=0)
""")

    nb.md("""
### Tables, freeze frames and the run folder

`to_dataframe` gives one row per object per frame. `to_freeze_frames` gives the
StatsBomb 360 shape: one row per object with match-clock time, JSON-encoded
pitch location and the visible area of the pitch (StatsBomb's flat
`[x1, y1, x2, y2, ...]` polygon). `period_start` is the match clock at the
first frame, in seconds; left out, it is the period's kick-off (0', 45', 90'
or 105'). `timestamp` is that match clock too, while StatsBomb's own event
timestamps restart every period. The demo does not say when in the match its
clip starts, so the times here count from the clip's first frame as if it
were the kick-off.
""")
    nb.code("""
tracks.head()
""")
    nb.code("""
freeze_frames = result.to_freeze_frames(period=1)
freeze_frames[["frame_id", "period", "timestamp", "minute", "second", "player_id", "type", "team_id", "location"]].head()
""")
    nb.md(f"""
`export` writes the run folder's data files: `result.pkl`, `tracks.csv`,
`freeze_frames.csv`, and the `tv.analysis` tables `carriers.csv` and
`passes.csv`. `tactifoot run` writes them through
`tv.run_file.load(run_file).run(video, output_dir, ...)`, which also renders
`annotated.mp4`. `save` and `PipelineResult.load` keep the whole result for
later, for example to render it again with other settings (see
[05_visualisation]({REPO_URL}/notebooks/05_visualisation.ipynb)).
""")
    nb.code("""
run_folder = result.export(OUT / "run", period=1)
print(sorted(p.name for p in run_folder.iterdir()))

saved = result.save(OUT / "result.pkl")
loaded = tv.PipelineResult.load(saved)
print("reloaded:", len(loaded), "frames; same tables:", loaded.to_dataframe().equals(tracks))
""")
    nb.md("""
What follows the ball is empty on the demo, which has no ball: the
`track_ball` post-process, which picks the live ball among the ball
candidates of every frame, finds none, so the ball carrier table has no
carrier and the pass table no pass. On footage with a ball detector they
fill in.
""")
    nb.code("""
ball_run = tv.pipeline.track_ball(result)
carriers = pd.read_csv(run_folder / "carriers.csv")
passes = pd.read_csv(run_folder / "passes.csv")
print("frames with a ball:", sum(len(f.ball) > 0 for f in ball_run))
print("frames with a ball carrier:", int(carriers["track_id"].notna().sum()), "| passes:", len(passes))
""")

    nb.md("""
### Swapping components

Components are plain objects, so changing one is a change of argument:
`tracker=sam2_tracker` above swapped the tracker, and leaving out
`team_classifier` skips the teams. For a moving camera, `keypoint_model` and
`homography` take the place of `fixed_homography` and `undistortion`: the
pitch model finds each frame's landmarks and the estimator fits that frame's
homography. Run on the held-out rendered views, whose true homographies are
known, `evaluate_homography` measures how far from the truth the pipeline
puts pitch points, in metres. On real broadcast footage the same pipeline
takes a pitch model trained on real labelled frames.
""")
    nb.code("""
moving = tv.Pipeline(detector=detector, keypoint_model=pitch_model, homography=estimator)
moving_run = moving.run(held_out.video_path, progress=False)
homography_metrics = tv.evaluation.evaluate_homography(moving_run, held_out)
pd.Series(homography_metrics.as_dict()).drop("truth").astype(float).round(3).to_frame("held-out views")
""")

    nb.md("""
## Comparing with StatsBomb 360

`compare_with_statsbomb(freeze_frames, statsbomb, period)` matches every
StatsBomb object to the nearest detected player or goalkeeper (the ball and
referees never match) and reports the distance. With sub-second times on
both sides (`timestamp_seconds`) the candidates come from the processed frame
closest to the event; otherwise from every frame of that second. Both sides
must use the same pitch units, so the pipeline is built with
`pitch=tv.SoccerPitch(120, 80)`, and the video must show the match the
events describe, with `period_start` set to the match clock at its first
frame.

No open source pairs match footage with the 360 data of the same match, so
this notebook shows the call and does not run it. StatsBomb's open data has
the 360 side: `tv.data.download_statsbomb` fetches one match into the layout
`load_statsbomb` reads (see [01_data]({REPO_URL}/notebooks/01_data.ipynb));
the footage of that match is yours to bring:

```python
match = tv.data.download_statsbomb(match_id, DATA / "statsbomb")
statsbomb = tv.data.load_statsbomb(match.folder)
pipeline = tv.Pipeline(..., pitch=tv.SoccerPitch(120, 80))
result = pipeline.run(match_video)
comparison = tv.evaluation.compare_with_statsbomb(
    result.to_freeze_frames(period=1, period_start=kickoff_offset_seconds), statsbomb, period=1)
tv.viz.plot_distance_histogram(comparison["euclidean_distance"].dropna(), bins=30)
```
""")

    nb.md(f"""
## Sources

{DEMO_ATTRIBUTION}

The pitch keypoint frames are rendered by `tactifoot_vision`.
""")
    return nb
