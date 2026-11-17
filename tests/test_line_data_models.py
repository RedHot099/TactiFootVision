"""What the line method adds to data, models, evaluation and the vocabularies, beside pose and keypoints."""

import numpy as np
import pytest

import tactifoot_vision as tv
from tactifoot_vision.data import Annotations, Dataset
from tactifoot_vision.enums import (
    ModelType,
    PitchView,
    PostProcessName,
    RunFileSection,
    Task,
)
from tactifoot_vision.evaluation import evaluate
from tactifoot_vision.models import MODELS, PitchLineModel, YOLOPoseModel
from tactifoot_vision.pipeline import POSTPROCESSES
from test_evaluation import FakeModel


def test_the_lines_backend_is_registered_beside_the_pose_backend():
    assert {"yolo", "yolo_pose", "rfdetr", "lines"} <= set(tv.models.available_models())
    assert MODELS.get(ModelType.LINES) is PitchLineModel
    assert PitchLineModel.task is Task.LINES
    assert MODELS.get(ModelType.YOLO_POSE) is YOLOPoseModel


def test_both_whole_video_steps_are_registered_post_processes():
    names = set(POSTPROCESSES.names())
    assert {
        PostProcessName.SMOOTH_HOMOGRAPHIES,
        PostProcessName.SMOOTH_LINE_HOMOGRAPHIES,
    } <= names
    assert {RunFileSection.KEYPOINTS, RunFileSection.LINES} <= set(RunFileSection)


def test_annotations_carry_line_labels_beside_keypoints():
    lines = Annotations(
        np.zeros((0, 4)),
        [],
        lines={"Middle line": [[1, 2], [3, 4]]},
        weight=0.5,
        view="along",
    )
    assert lines.view is PitchView.ALONG and lines.weight == 0.5
    assert lines.lines["Middle line"].dtype == np.float32
    copy = lines.copy()
    copy.lines["Middle line"][0, 0] = 99
    assert lines.lines["Middle line"][0, 0] == 1  # a deep copy
    with pytest.raises(ValueError, match="view"):
        Annotations(np.zeros((0, 4)), [], view="sideways")
    # the keypoint fields are as before
    pose = Annotations([[0, 0, 1, 1]], [0], keypoints=np.zeros((1, 3, 3)))
    assert pose.num_keypoints == 3 and pose.lines is None
    assert pose.view is PitchView.MAIN


def test_a_lines_dataset_has_no_export_layout():
    dataset = Dataset(Task.LINES, ["Middle line"], {"train": []})
    for layout in ("to_yolo", "to_coco"):
        with pytest.raises(ValueError, match="no (YOLO|COCO) layout"):
            getattr(dataset, layout)("unused")


def test_evaluate_points_a_lines_model_to_evaluate_lines():
    dataset = Dataset(Task.DETECT, ["ball"], {"valid": []})
    with pytest.raises(
        ValueError, match="No evaluation for lines models.*evaluate_lines"
    ):
        evaluate(FakeModel(Task.LINES, {}, ["Middle line"]), dataset)
