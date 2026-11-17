# TactiFoot Vision

![TactiFoot Vision on one attacking move](https://raw.githubusercontent.com/RedHot099/TactiFootVision/master/docs/media/tactifoot_promo.avif)

TactiFoot Vision turns football match video into tactical data. It finds and follows every player and the ball, works out who plays for which team, and maps the action onto the pitch in metres.

- **Data:** load, convert, split and augment annotated football datasets.
- **Models:** train and evaluate YOLO and RF-DETR detectors and the pitch line model through one interface.
- **Pipeline:** detection, tracking, team classification, and a homography fitted to the painted pitch markings.
- **Output:** tracks, StatsBomb-style freeze frames, annotated video and a tactical radar.

## Quick start

To see the whole workflow without installing anything, open the overview
notebook in Google Colab. It downloads an open demo match, trains a player
detector for a few minutes, tracks the players and draws the tactical map:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RedHot099/TactiFootVision/blob/v0.3.0/notebooks/00_overview.ipynb)

To use the package on your own footage, install it from PyPI (Python 3.13):

```bash
pip install tactifoot-vision
```

```python
import tactifoot_vision as tv

pipeline = tv.Pipeline(
    detector=tv.load_model("yolo", "models/detector_yolo11l.pt"),
    line_model=tv.load_model("lines", "models/pitch_lines.pt"),
    team_classifier=tv.teams.TeamClassifier("siglip"),
)
result = pipeline.run("match.mp4")
result = tv.pipeline.smooth_homographies(result)  # over the whole video
tv.viz.render_video(result, "match.mp4", "annotated.mp4")
```

`pipeline.run` returns each frame's homography as the frame's own fit, steadied
by a causal Kalman filter. A fit the filter rejects is returned as it is
until three rejects in a row confirm a cut, so the new view after a cut is
returned on its first frame. `smooth_homographies` is the whole-video pass, and it
gives the steadier `final` homographies. From the command line, add it to the
run file as `postprocess: [{type: smooth_homographies}]`, or pass
`--set 'postprocess=[{type: smooth_homographies}]'`.

On the 49 clips of SoccerNet's held-out GSR test set, the `final`
homography's pitch error is under 0.5 m on 70 % and under 1 m on 93 % of a
clip's frames, on average over the clips (`A1_success@0.5` 0.696 and
`A1_success@1` 0.934 in
`docs/research/homography-hill-climb.md`, section 5). The 90th percentile of a clip's
error is 1.1 m on average, against 2.9 m for the keypoint estimator it
replaced. It takes 6 ms per 1080p frame on a GPU (docs/adr/0005). On
TS-WorldCup (43 broadcast sequences of the 2014 and 2018 World Cups, not
SoccerNet footage), the mean error of the visible pitch-grid points has a
90th percentile of 0.34 m on average over the sequences, against 0.68 m for
the keypoint estimator.

**Known limits of the homography:**

- **Views from the opposite side.** A view from the camera side opposite the
  main camera looks like a main view of the other half with every line name
  turned. The estimator then returns a wrong matrix and cannot flag it.
- **Single images along the pitch.** Behind or above a goal, the visible
  markings are often symmetric. Without a previous frame the fit is refused,
  so a single image gets no matrix. In video, the previous frame decides.
- **The causal track depends on the training run.** A network may give a
  wrong fit on single frames of a few clips; it is returned as it is, so the
  causal track jitters there, by how much depends on the training run: over
  five training seeds of the declared recipe, the worst SoccerNet GSR valid
  clip's causal jitter ranges from 0.26 m to 27.6 m. The `final` track
  repairs it (0.032 to 0.044 m); use it where jitter matters.
- **Fewer single images get a matrix.** On SoccerNet's Camera Calibration
  test images, 77.1 % get one, against 83.9 % for the keypoint estimator. A
  fit that the visible lines leave undetermined is refused rather than
  guessed.
- **The filter's time base, and strides.** Its noise settings are per 1/25 s.
  The pipeline passes the processed frame rate (fps / stride), and
  `smooth_homographies` takes it from the frame indices, so strided or 50 fps
  runs are filtered in the same physical time. A strided run still loses
  accuracy, since a frame between two processed frames holds the last matrix:
  on SoccerNet GSR valid, causal AUC5 is 0.879 at stride 2 and 0.850 at
  stride 5, against 0.885 on every frame.

No weights ship with the package. The pitch line model trains on SoccerNet's
line labels (the Camera Calibration and Game State Reconstruction downloads)
with the recipe the package declares:

```python
lines = tv.load_pitch_lines("data/SoccerNetCalibration", "data/SoccerNetGS")
run = tv.train("lines", lines, epochs=20, batch_size=12, seed=0)
print(run.weights)  # runs/lines/weights/final.pt, with runs/lines/manifest.json
```

The same pipeline runs from the command line:

```bash
tactifoot init pipeline.yaml   # a commented run file to edit
tactifoot run pipeline.yaml --video match.mp4 --output-dir outputs/match
tactifoot train lines --calibration data/SoccerNetCalibration --gamestate data/SoccerNetGS \
    --epochs 20 --batch-size 12
```

## Contributing

```bash
git clone https://github.com/RedHot099/TactiFootVision.git && cd TactiFootVision
uv sync                 # the package, its dev tools and the notebook kernel
uv run pytest -q        # the fast suite
```

The notebooks are generated from `notebooks/build/`. All but the overview
notebook, and the examples, expect local data and weights; see the
[notebooks README](https://github.com/RedHot099/TactiFootVision/blob/master/notebooks/README.md).

## Learn more

- **[Notebooks](https://github.com/RedHot099/TactiFootVision/blob/master/notebooks/README.md):** executed walkthroughs of every module, from data to visualisation. Start with `00_overview`, which also runs in Colab.
- **[Architecture](https://github.com/RedHot099/TactiFootVision/blob/master/docs/ARCHITECTURE.md):** modules, interfaces, run files and the CLI.
- **[Examples](https://github.com/RedHot099/TactiFootVision/tree/master/examples):** runnable scripts for each area and the CLI.
