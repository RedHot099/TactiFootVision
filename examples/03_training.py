"""Model training and evaluation: three model types, one interface.

A short demo schedule (a few epochs on a subset); raise EPOCHS and drop the
subsets for real training. Run from the repository root:
``uv run python examples/03_training.py``
"""

import shutil
from pathlib import Path

import tactifoot_vision as tv
from tactifoot_vision import augment as A

OUT = Path("outputs/examples/03_training")
EPOCHS = 3
tv.setup_logging("INFO")

detection = tv.load_dataset("data/datasets/football_yolo")
# The pitch line model learns SoccerNet's line labels (Calibration and/or GSR folders);
# the shipped rule keeps the 8,316 Calibration train images the network trained on.
lines = tv.load_pitch_lines("data/SoccerNetCalibration").subset(
    {"train": 200, "valid": 50}, seed=0
)
train_subset = detection.subset({"train": 150, "valid": 40}, seed=0)

transform = A.Compose(
    [A.HorizontalFlip(), A.RandomAffine(degrees=3, translate=0.05), A.ColorJitter()]
)
shutil.rmtree(OUT / "augmented", ignore_errors=True)  # augment_dataset never overwrites
augmented = A.augment_dataset(train_subset, transform, OUT / "augmented", copies=1)

# Same call for every backend; backend-specific options pass straight through.
yolo_run = tv.train("yolo", augmented, weights="yolo11n.pt", epochs=EPOCHS, batch_size=16,
                    output_dir=OUT / "runs", name="yolo11n")  # fmt: skip
# The declared recipe is epochs=20, batch_size=12 on the full set (docs/adr/0005).
lines_run = tv.train("lines", lines, epochs=1, batch_size=12, output_dir=OUT / "runs",
                     name="pitch_lines")  # fmt: skip
rfdetr_run = tv.train("rfdetr", train_subset, weights=None, epochs=1,
                      batch_size=4, output_dir=OUT / "runs", name="rfdetr",
                      grad_accum_steps=2)  # fmt: skip

for run in (yolo_run, lines_run, rfdetr_run):
    print(run.model_type, run.weights, run.metrics)
    tv.viz.plot_training(run).savefig(OUT / f"training_{run.model_type}.png")

# Backend-independent metrics on the same validation split.
valid = detection.subset({"valid": 100}, seed=1)
models = {
    "YOLO11n (demo run)": yolo_run.load(),
    "RF-DETR (demo run)": rfdetr_run.load(),
    "YOLO11l (football)": tv.load_model("yolo", "models/detector_yolo11l.pt"),
    "RF-DETR base (football)": tv.load_model(
        "rfdetr", "models/rfdetr_smoketest_sample.pth"
    ),
}
metrics = {name: model.evaluate(valid) for name, model in models.items()}
for name, m in metrics.items():
    print(f"{name:26s} mAP50-95={m.map50_95:.3f} mAP50={m.map50:.3f}")
tv.viz.plot_metrics(metrics).savefig(OUT / "detector_comparison.png")

print("figures in", OUT)
