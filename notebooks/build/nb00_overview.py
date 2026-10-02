"""00_overview.ipynb: the whole workflow in a few cells, with links to the module notebooks."""

from common import Notebook


def notebook() -> Notebook:
    nb = Notebook("00_overview")
    nb.md("""
# TactiFoot Vision: from dataset to tactical map

`tactifoot_vision` turns football broadcast video into tactical data: who is
where on the pitch, in which team, frame by frame. This notebook runs the
whole workflow once, one short step per stage. Each stage has its own
notebook with the details:

| stage | module | notebook |
|---|---|---|
| Data preparation | `tv.data` | [01_data](01_data.ipynb) |
| Data augmentation | `tv.augment` | [02_augmentation](02_augmentation.ipynb) |
| Training and evaluation | `tv.models`, `tv.evaluation` | [03_training_and_evaluation](03_training_and_evaluation.ipynb) |
| Inference | `tv.tracking`, `tv.pitch`, `tv.teams`, `tv.pipeline` | [04_inference](04_inference.ipynb) |
| Visualisation | `tv.viz` | [05_visualisation](05_visualisation.ipynb) |

Everything is plain Python: build the objects you want and call them. The
same steps exist as a command line (`tactifoot run`, `tactifoot train`), see
the README. Requirements and local data are listed in [README.md](README.md).
""")
    nb.setup()

    nb.md("""
## 1. Data

`tv.load_dataset` reads a YOLO or COCO dataset into a `Dataset`: images stay
on disk, labels in memory.
""")
    nb.code("""
dataset = tv.load_dataset(DATA / "datasets" / "football_yolo")
print(dataset)
tv.viz.show_samples(dataset, "train", n=3)
""")

    nb.md("""
## 2. Augmentation

Transforms move boxes and keypoints with the pixels. `augment_dataset` adds
augmented copies of the training images; validation images stay real.
""")
    nb.code("""
recipe = tv.augment.Compose([
    tv.augment.HorizontalFlip(),
    tv.augment.RandomAffine(degrees=3, translate=0.05),
    tv.augment.ColorJitter(),
])
small = dataset.subset({"train": 100, "valid": 40}, seed=0)
augmented = tv.augment.augment_dataset(small, recipe, OUT / "augmented", copies=1)
augmented.summary()
""")

    nb.md("""
## 3. Training and evaluation

`tv.train` works the same for every backend (`yolo`, `rfdetr`, `yolo_pose`).
This is a 15-epoch demo on 200 images; the checkpoints in `models/` were
trained the same way on the full datasets for 200 epochs and are used from
here on. Both are scored
by the same evaluation code.
""")
    nb.code("""
%%capture train_log
demo_run = tv.train("yolo", augmented, weights="yolo11n.pt", epochs=15, batch_size=16,
                    output_dir=OUT / "runs", name="yolo11n", exist_ok=True)
""")
    nb.code("""
import pandas as pd

detector = tv.load_model("yolo", MODELS / "football_yolo11m.pt", conf=0.3)
scores = {"YOLO11n, 15-epoch demo": demo_run.load().evaluate(dataset),
          "YOLO11m, models/": detector.evaluate(dataset)}
pd.DataFrame({name: m.as_dict() for name, m in scores.items()}).T.round(3)
""")

    nb.md("""
## 4. Inference

The pipeline detects people and the ball, tracks them, maps them onto the
pitch through the pitch-landmark model, and splits players into teams by kit.
""")
    nb.code("""
pipeline = tv.Pipeline(
    detector=detector,
    keypoint_model=tv.load_model("yolo_pose", MODELS / "pitch_yolov8n_pose.pt"),
    tracker="bytetrack",
    team_classifier=tv.teams.TeamClassifier(embedder="siglip"),
)
result = pipeline.run(VIDEO, max_frames=250, progress=False)  # 10 seconds
print(len(result), "frames,", len(result.track_ids), "tracks")
result.to_dataframe().query("object == 'person'").head()
""")
    nb.md("""
`export` writes the run folder (`result.pkl`, `tracks.csv`, StatsBomb-style
`freeze_frames.csv`), the same files `tactifoot run` produces.
""")
    nb.code("""
run_folder = result.export(OUT / "run")
sorted(p.name for p in run_folder.iterdir())
""")

    nb.md("""
## 5. Tactical map

One frame with tracks, teams and the projected pitch lines, plus the top-down
radar; then where each team spent the clip.
""")
    nb.code("""
frame_result = result[200]
frame = tv.VideoReader(VIDEO).read(frame_result.index)
annotated = tv.viz.FrameAnnotator(style="video_game").annotate(frame, frame_result)
radar = tv.viz.PitchRadar().draw(frame_result)
tv.show(tv.viz.overlay(annotated, radar, position="bottom-right", width_fraction=0.3), width=12)
""")
    nb.code("""
from matplotlib.figure import Figure

fig = Figure(figsize=(12, 4), layout="constrained")
for ax, team in zip(fig.subplots(1, 2), (0, 1), strict=True):
    tv.viz.plot_heatmap(result, team=team, ax=ax)
fig
""")
    nb.md("""
## Next

* The module notebooks above go through every option.
* `tv.viz.render_video(result, VIDEO, "annotated.mp4")` writes the annotated
  clip ([05_visualisation](05_visualisation.ipynb)).
* The command line runs the same steps on whole matches:
  `tactifoot run configs/pipeline.yaml --video match.mp4 --output-dir outputs/match`.
""")
    return nb
