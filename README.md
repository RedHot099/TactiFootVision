# TactiFoot Vision

![TactiFoot Vision on one attacking move](https://raw.githubusercontent.com/RedHot099/TactiFootVision/master/docs/media/tactifoot_promo.avif)

TactiFoot Vision turns football match video into tactical data. It finds and follows every player and the ball, works out who plays for which team, and maps the action onto the pitch in metres.

- **Data:** load, convert, split and augment annotated football datasets.
- **Models:** train and evaluate YOLO, RF-DETR, pitch-keypoint and pitch-line models through one interface.
- **Pipeline:** detection, tracking, team classification and homography to pitch coordinates.
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
    keypoint_model=tv.load_model("yolo_pose", "models/pitch_yolo11m_pose.pt"),
    team_classifier=tv.teams.TeamClassifier("siglip"),
)
result = pipeline.run("match.mp4")
tv.viz.render_video(result, "match.mp4", "annotated.mp4")
```

The same pipeline runs from the command line:

```bash
tactifoot init pipeline.yaml   # a commented run file to edit
tactifoot run pipeline.yaml --video match.mp4 --output-dir outputs/match
```

### Opt-in: the homography from the painted pitch lines

A second way to get the homography is a line network and one joint fit
([ADR 0005](https://github.com/RedHot099/TactiFootVision/blob/master/docs/adr/0005-the-homography-is-one-joint-fit-to-a-line-network.md)).
A `lines` model finds points along the 20 painted markings, and
`tv.pitch.LineHomographyEstimator` fits one homography to all of them,
steadied by a Kalman filter. It is opt-in: the keypoint model above stays the
default. The names `LineHomographyEstimator` and `smooth_line_homographies` are
temporary: when the line method becomes the default, they become
`HomographyEstimator` and `smooth_homographies`. No line weights ship with the package; they train on SoccerNet's
line labels (the Camera Calibration and Game State Reconstruction downloads):

```python
lines = tv.load_pitch_lines("data/SoccerNetCalibration", "data/SoccerNetGS")
run = tv.train("lines", lines, epochs=20, batch_size=12, seed=0)

pipeline = tv.Pipeline(
    detector=tv.load_model("yolo", "models/detector_yolo11l.pt"),
    line_model=tv.load_model("lines", run.weights),  # instead of keypoint_model
)
result = pipeline.run("match.mp4")  # the causal track
result = tv.pipeline.smooth_line_homographies(result)  # the final track, over the whole video
```

In a run file, use a `lines:` section instead of `keypoints:` and add
`postprocess: [{type: smooth_line_homographies}]`. The command
`tactifoot train lines --calibration ... --gamestate ...` trains the weights.

On the 49 clips of SoccerNet's held-out GSR test set, the `final` track's
pitch error is under 1 m on 93 % of a clip's frames, on average over the
clips. The 90th percentile of a clip's error is 1.1 m on average, against
2.9 m for the keypoint estimator, and a 1080p frame takes 6 ms on a GPU
against 85 ms. The evidence is in
[`docs/research/homography-hill-climb.md`](https://github.com/RedHot099/TactiFootVision/blob/master/docs/research/homography-hill-climb.md).

Its known limits:

- **Views from the opposite side** look like a main view of the other half,
  with every line name turned. The estimator returns a wrong matrix there and
  does not flag it.
- **Single images along the pitch** often show symmetric markings, so the fit
  is refused and the image gets no matrix. In video, the previous frame
  decides.
- **The causal track** can jitter on a few clips, by an amount that depends
  on the training run. Use the `final` track where jitter matters.
- **Fewer single images get a matrix:** 77.1 % of SoccerNet's Camera
  Calibration test images, against 83.9 % for the keypoint estimator.
  A fit the visible lines leave undetermined is refused rather than guessed.

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
