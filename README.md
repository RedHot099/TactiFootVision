# TactiFoot Vision

![TactiFoot Vision on one attacking move](https://raw.githubusercontent.com/RedHot099/TactiFootVision/master/docs/media/tactifoot_promo.avif)

TactiFoot Vision turns football match video into tactical data. It finds and follows every player and the ball, works out who plays for which team, and maps the action onto the pitch in metres.

- **Data:** load, convert, split and augment annotated football datasets.
- **Models:** train and evaluate YOLO, RF-DETR and pitch-keypoint models through one interface.
- **Pipeline:** detection, tracking, team classification and homography to pitch coordinates.
- **Output:** tracks, StatsBomb-style freeze frames, annotated video and a tactical radar.

## Quick start

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

## Contributing

```bash
git clone https://github.com/RedHot099/TactiFootVision.git && cd TactiFootVision
uv sync                 # the package, its dev tools and the notebook kernel
uv run pytest -q        # the fast suite
```

The notebooks and examples expect local data and weights; see the
[notebooks README](https://github.com/RedHot099/TactiFootVision/blob/master/notebooks/README.md).

## Learn more

- **[Notebooks](https://github.com/RedHot099/TactiFootVision/blob/master/notebooks/README.md):** executed walkthroughs of every module, from data to visualisation. Start with `00_overview`.
- **[Architecture](https://github.com/RedHot099/TactiFootVision/blob/master/docs/ARCHITECTURE.md):** modules, interfaces, run files and the CLI.
- **[Examples](https://github.com/RedHot099/TactiFootVision/tree/master/examples):** runnable scripts for each area and the CLI.
