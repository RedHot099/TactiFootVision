"""00_overview.ipynb: the whole workflow in a few cells, with links to the module notebooks."""

from common import Notebook

from tactifoot_vision.data.demo import ATTRIBUTION

# The demo training run. Of three tried on the demo set for 10 epochs (YOLO11n
# at 640 and at 1280, YOLO11s at 1920), only this one finds all 22 players of
# the 4096-pixel panorama: MOTA 0.63 on the demo clip against 0.49 and 0.28,
# for under a minute of training on an RTX 4070 Ti SUPER.
WEIGHTS, IMGSZ, EPOCHS = "yolo11s.pt", 1920, 10


REPO_URL = "https://github.com/RedHot099/TactiFootVision/blob/master"
# The badge opens the notebook of the latest release, the package its first
# cell installs; `master` may be ahead of it. docs/RELEASING.md moves the tag.
COLAB_URL = "https://colab.research.google.com/github/RedHot099/TactiFootVision/blob/v0.3.0/notebooks/00_overview.ipynb"


def notebook() -> Notebook:
    nb = Notebook("00_overview")
    nb.md(f"""
# TactiFoot Vision: from dataset to tactical map

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

`tactifoot_vision` turns football video into tactical data: who is where on
the pitch, in which team, frame by frame. This notebook runs the whole
workflow once on open data, one short step per stage: it downloads a demo
match, trains a player detector on it for a few minutes, tracks the players
through a clip and draws them on the pitch. It needs no local data or
weights, so it runs the same from a checkout and on Google Colab.

**On Colab**, pick a GPU first (Runtime, Change runtime type, T4 GPU) and add
your Hugging Face token as a secret named `HF_TOKEN` (the key icon on the
left) with notebook access switched on; step 1 says why.

Each stage has its own notebook with the details. Those still read local data
and run from a checkout only:

| stage | module | notebook |
|---|---|---|
| Data preparation | `tv.data` | [01_data]({REPO_URL}/notebooks/01_data.ipynb) |
| Data augmentation | `tv.augment` | [02_augmentation]({REPO_URL}/notebooks/02_augmentation.ipynb) |
| Training and evaluation | `tv.models`, `tv.evaluation` | [03_training_and_evaluation]({REPO_URL}/notebooks/03_training_and_evaluation.ipynb) |
| Inference | `tv.tracking`, `tv.pitch`, `tv.teams`, `tv.pipeline` | [04_inference]({REPO_URL}/notebooks/04_inference.ipynb) |
| Visualisation | `tv.viz` | [05_visualisation]({REPO_URL}/notebooks/05_visualisation.ipynb) |

Everything is plain Python: build the objects you want and call them. The
same steps exist as a command line (`tactifoot run`, `tactifoot train`,
`tactifoot evaluate`), described under "Run files and the CLI" in
[docs/ARCHITECTURE.md]({REPO_URL}/docs/ARCHITECTURE.md).
""")
    nb.setup(standalone=True)

    nb.md("""
## 1. Data

`tv.data.download_demo` fetches one match of
[SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2)
(about 300 MB, CC BY 4.0) and lays it out for the package: a detection
dataset of frames from the first minutes and a 30-second clip with tracking
truth from the end. The dataset is gated: accept its conditions once on its
page, then the call finds your token in the `HF_TOKEN` Colab secret, the
environment variable of that name or a saved `hf auth login`. A second call
does nothing.

The footage is one static camera that sees the whole pitch through a fisheye
lens, 4096x1080 pixels, so a player is about 25x60 pixels. The labels are
the 22 players; this data has no ball, referee or goalkeeper boxes.
""")
    nb.code("""
demo = tv.data.download_demo(DATA / "demo")
dataset = tv.load_dataset(demo.detect_yaml)
print(dataset)
tv.viz.show_samples(dataset, "train", n=2, cols=1)
""")

    nb.md("""
## 2. Augmentation

Transforms move boxes and keypoints with the pixels. `augment_dataset` adds
augmented copies of the training images; validation images stay real. It
never overwrites earlier output, so a re-run starts from an empty folder.
""")
    nb.code("""
import shutil

recipe = tv.augment.Compose([
    tv.augment.HorizontalFlip(),
    tv.augment.RandomAffine(degrees=3, translate=0.05),
    tv.augment.ColorJitter(),
])
small = dataset.subset({"train": 40, "valid": 10}, seed=0)
shutil.rmtree(OUT / "augmented", ignore_errors=True)
augmented = tv.augment.augment_dataset(small, recipe, OUT / "augmented", copies=1, progress=False)
augmented.summary()
""")

    nb.md(f"""
## 3. Training and evaluation

`tv.train` works the same for every backend (`yolo`, `rfdetr`, `yolo_pose`).
This run starts from the COCO weights Ultralytics publishes and trains for
{EPOCHS} epochs on the demo frames, at {IMGSZ} pixels on the long side
because the players are small: about a minute on a desktop GPU, several on
a Colab T4. The score comes from `evaluate`, the same code for every backend.
""")
    nb.code(f"""
%%capture train_log
training = tv.train("yolo", dataset, weights="{WEIGHTS}", epochs={EPOCHS}, batch_size=8,
                    imgsz={IMGSZ}, rect=True, output_dir=OUT / "runs", name="detector", exist_ok=True)
""")
    nb.code("""
import pandas as pd

detector = training.load(conf=0.3)
pd.Series(detector.evaluate(dataset).as_dict()).round(3).to_frame("validation")
""")

    nb.md(f"""
## 4. Inference

The pipeline detects the players, tracks them, maps them onto the pitch and
splits them into teams by kit. The camera does not move, so one homography
serves the whole clip: `fixed_homography` and `undistortion` take the
calibration that came with the data. (Broadcast footage has no such file;
there a pitch-landmark model passed as `keypoint_model` finds the homography
frame by frame, as in [04_inference]({REPO_URL}/notebooks/04_inference.ipynb).)
""")
    nb.code("""
clip = tv.data.load_labelled_clip(demo.clip_dir)
pipeline = tv.Pipeline(
    detector=detector,
    fixed_homography=demo.calibration,
    undistortion=demo.calibration,
    tracker="bytetrack",
    team_classifier=tv.teams.TeamClassifier(embedder="siglip"),
)
result = pipeline.run(clip.video_path, end=250, progress=False)  # 10 seconds
print(len(result), "frames,", len(result.track_ids), "tracks")
result.to_dataframe().query("object == 'person'").head()
""")
    nb.md("""
The clip comes with the true boxes and identities, so the tracks can be
scored (MOTA and IDF1 of 1 would be a perfect run):
""")
    nb.code("""
tracking = tv.evaluation.evaluate_tracking(result, clip, hota=False)
pd.Series(tracking.as_dict()).dropna().round(3).to_frame("demo clip")
""")
    nb.md("""
`export` writes the run folder (`result.pkl`, `tracks.csv`, StatsBomb-style
`freeze_frames.csv`, and the ball carrier and pass tables `carriers.csv` and
`passes.csv`, empty here because the demo has no ball), the same files
`tactifoot run` produces.
""")
    nb.code("""
run_folder = result.export(OUT / "run")
sorted(p.name for p in run_folder.iterdir())
""")

    nb.md("""
## 5. Tactical map

One frame with tracks and teams, and the same moment on the top-down radar;
then where each team spent the clip.
""")
    nb.code("""
frame_result = result[200]
frame = tv.VideoReader(clip.video_path).read(frame_result.index)
# The pitch lines are left out: they are straight in the model and curved in a fisheye frame.
annotator = tv.viz.FrameAnnotator(style="video_game", draw_pitch_lines=False)
tv.show(annotator.annotate(frame, frame_result), width=16)
""")
    nb.code("""
tv.show(tv.viz.PitchRadar().draw(frame_result), width=8)
""")
    nb.code("""
from matplotlib.figure import Figure

fig = Figure(figsize=(12, 4), layout="constrained")
for ax, team in zip(fig.subplots(1, 2), (0, 1), strict=True):
    tv.viz.plot_heatmap(result, team=team, ax=ax)
fig
""")
    nb.md(f"""
## Next

* The module notebooks above go through every option.
* `tv.viz.render_video(result, clip.video_path, "annotated.mp4")` writes the
  annotated clip ([05_visualisation]({REPO_URL}/notebooks/05_visualisation.ipynb)).
* The command line runs the same steps on whole matches:
  `tactifoot init pipeline.yaml` writes a commented run file, then
  `tactifoot run pipeline.yaml --video match.mp4 --output-dir outputs/match`,
  which is `tv.run_file.load("pipeline.yaml").run("match.mp4", "outputs/match")`.

{ATTRIBUTION}
""")
    return nb
