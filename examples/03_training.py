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
pitch = tv.load_dataset("data/keypoints")
train_subset = detection.subset({"train": 150, "valid": 40}, seed=0)

transform = A.Compose(
    [A.HorizontalFlip(), A.RandomAffine(degrees=3, translate=0.05), A.ColorJitter()]
)
shutil.rmtree(OUT / "augmented", ignore_errors=True)  # augment_dataset never overwrites
augmented = A.augment_dataset(train_subset, transform, OUT / "augmented", copies=1)

# Same call for every backend; backend-specific options pass straight through.
yolo_run = tv.train("yolo", augmented, weights="yolo11n.pt", epochs=EPOCHS, batch_size=16,
                    output_dir=OUT / "runs", name="yolo11n")  # fmt: skip
pose_run = tv.train("yolo_pose", pitch, weights="yolov8n-pose.pt", epochs=EPOCHS,
                    batch_size=16, output_dir=OUT / "runs", name="pitch_pose")  # fmt: skip
rfdetr_run = tv.train("rfdetr", train_subset, weights=None, epochs=1,
                      batch_size=4, output_dir=OUT / "runs", name="rfdetr",
                      grad_accum_steps=2)  # fmt: skip

for run in (yolo_run, pose_run, rfdetr_run):
    print(run.model_type, run.weights, run.metrics)
    tv.viz.plot_training(run).savefig(OUT / f"training_{run.model_type}.png")

# Backend-independent metrics on the same validation split.
valid = detection.subset({"valid": 100}, seed=1)
models = {
    "YOLO11n (demo run)": yolo_run.load(),
    "RF-DETR (demo run)": rfdetr_run.load(),
    "YOLO11m (football)": tv.load_model("yolo", "models/football_yolo11m.pt"),
    "RF-DETR base (football)": tv.load_model(
        "rfdetr", "models/football_rfdetr_base.pth"
    ),
}
metrics = {name: model.evaluate(valid) for name, model in models.items()}
for name, m in metrics.items():
    print(f"{name:26s} mAP50-95={m.map50_95:.3f} mAP50={m.map50:.3f}")
tv.viz.plot_metrics(metrics).savefig(OUT / "detector_comparison.png")

pitch_metrics = {
    "YOLOv8n-pose (demo run)": pose_run.load().evaluate(pitch),
    "YOLOv8n-pose (football)": tv.load_model("yolo_pose", "models/pitch_yolov8n_pose.pt").evaluate(pitch),
}  # fmt: skip
for name, m in pitch_metrics.items():
    print(
        f"{name:26s} error={m.mean_error_px:.1f}px PCK={m.pck:.3f} found={m.detection_rate:.2f}"
    )
tv.viz.plot_metrics(pitch_metrics).savefig(OUT / "keypoint_comparison.png")
print("figures in", OUT)
