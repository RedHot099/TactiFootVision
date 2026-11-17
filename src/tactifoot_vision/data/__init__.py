"""Data preparation: annotated datasets, pitch line labels, labelled clips (read or synthetic), format conversion, video frames, StatsBomb events."""

from tactifoot_vision.data.annotations import Annotations, Task
from tactifoot_vision.data.dataset import Dataset, Sample, load_dataset
from tactifoot_vision.data.demo import DemoData, download_demo
from tactifoot_vision.data.labelled_clip import (
    UNKNOWN_TEAM,
    LabelledClip,
    LabelledFrame,
    Truth,
    load_labelled_clip,
)
from tactifoot_vision.data.pitch_lines import load_pitch_lines, render_line_targets
from tactifoot_vision.data.statsbomb import load_statsbomb
from tactifoot_vision.data.synthetic import synthetic_pitch_clip
from tactifoot_vision.data.video import VideoReader, extract_frames

__all__ = [
    "UNKNOWN_TEAM",
    "Annotations",
    "Dataset",
    "DemoData",
    "LabelledClip",
    "LabelledFrame",
    "Sample",
    "Task",
    "Truth",
    "VideoReader",
    "download_demo",
    "extract_frames",
    "load_dataset",
    "load_labelled_clip",
    "load_pitch_lines",
    "load_statsbomb",
    "render_line_targets",
    "synthetic_pitch_clip",
]
