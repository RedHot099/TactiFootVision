import json
from pathlib import Path

import pytest

from tactifoot_vision.datasets import iter_gsr_sequence_dirs, read_gsr_labels


def make_gsr_sequence(root: Path, name: str = "SNGS-001") -> Path:
    sequence = root / name
    sequence.mkdir(parents=True)
    payload = {
        "info": {"version": "1.3"},
        "images": [
            {
                "image_id": "1001000001",
                "file_name": "000001.jpg",
                "width": 1920,
                "height": 1080,
                "has_labeled_pitch": True,
                "has_labeled_camera": True,
                "has_labeled_person": True,
            }
        ],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "object"},
            {"id": 20, "name": "Side line top", "supercategory": "pitch"},
        ],
        "annotations": [
            {
                "id": "person-1",
                "image_id": "1001000001",
                "track_id": 7,
                "supercategory": "object",
                "bbox_image": {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
                "bbox_pitch": {
                    "x_bottom_left": -11.0,
                    "y_bottom_left": 5.5,
                    "x_bottom_middle": -10.5,
                    "y_bottom_middle": 5.0,
                    "x_bottom_right": -10.0,
                    "y_bottom_right": 4.5,
                },
                "attributes": {
                    "role": "player",
                    "jersey": "9",
                    "team": "left",
                },
            },
            {
                "id": "line-1",
                "image_id": "1001000001",
                "category_id": 20,
                "supercategory": "pitch",
                "points": [{"x": 0.0, "y": 1.0}, {"x": 2.0, "y": 3.0}],
            },
        ],
    }
    (sequence / "Labels-GameState.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return sequence


def test_read_gsr_labels_parses_players_pitch_bbox_and_lines(tmp_path: Path) -> None:
    sequence = make_gsr_sequence(tmp_path)

    labels = read_gsr_labels(sequence)

    assert labels.sequence == "SNGS-001"
    assert labels.version == "1.3"
    assert labels.frames[0].frame == 1
    assert labels.frames[0].has_labeled_pitch
    assert len(labels.athletes) == 1
    athlete = labels.athletes[0]
    assert athlete.track_id == 7
    assert athlete.role == "player"
    assert athlete.jersey == "9"
    assert athlete.team == "left"
    assert athlete.image_bottom_middle == (25.0, 60.0)
    assert athlete.pitch_bottom_middle == (-10.5, 5.0)
    assert labels.athletes_for_frame(1) == (athlete,)
    assert len(labels.lines) == 1
    assert labels.lines[0].line_name == "Side line top"
    assert labels.lines[0].points[1].x == 2.0


def test_read_gsr_labels_rejects_old_version(tmp_path: Path) -> None:
    sequence = make_gsr_sequence(tmp_path)
    labels_path = sequence / "Labels-GameState.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    payload["info"]["version"] = "1.2"
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=">= 1.3"):
        read_gsr_labels(sequence)


def test_read_gsr_labels_accepts_center_based_gt_image_bbox(tmp_path: Path) -> None:
    sequence = make_gsr_sequence(tmp_path)
    labels_path = sequence / "Labels-GameState.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["bbox_image"] = {
        "x_center": 25.0,
        "y_center": 40.0,
        "w": 30.0,
        "h": 40.0,
    }
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    labels = read_gsr_labels(sequence)

    assert labels.athletes[0].image_bottom_middle == (25.0, 60.0)


def test_read_gsr_labels_parses_pitch_row_lines_mapping(tmp_path: Path) -> None:
    sequence = make_gsr_sequence(tmp_path)
    labels_path = sequence / "Labels-GameState.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    payload["annotations"][1] = {
        "id": "pitch-1",
        "image_id": "1001000001",
        "video_id": "001",
        "category_id": 20,
        "supercategory": "pitch",
        "lines": {
            "Middle line": [[12.0, 0.0], [12.0, 68.0]],
            "Side line top": [{"x": 0.0, "y": 0.0}, {"x": 105.0, "y": 0.0}],
        },
    }
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    labels = read_gsr_labels(sequence)

    assert [line.line_name for line in labels.lines] == [
        "Middle line",
        "Side line top",
    ]
    assert labels.lines[0].points[1].y == 68.0


def test_read_gsr_labels_ignores_line_with_unknown_nonnumeric_category(
    tmp_path: Path,
) -> None:
    sequence = make_gsr_sequence(tmp_path)
    labels_path = sequence / "Labels-GameState.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    payload["annotations"][1].pop("category_id")
    payload["annotations"][1]["category_id"] = "pitch-line"
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    labels = read_gsr_labels(sequence)

    assert labels.lines == ()


def test_iter_gsr_sequence_dirs_supports_dataset_root_and_split(tmp_path: Path) -> None:
    root = tmp_path / "SoccerNetGS"
    make_gsr_sequence(root / "valid", "SNGS-001")
    make_gsr_sequence(root / "train", "SNGS-002")

    assert [path.name for path in iter_gsr_sequence_dirs(root, split="valid")] == [
        "SNGS-001"
    ]
    assert [path.name for path in iter_gsr_sequence_dirs(root)] == [
        "SNGS-002",
        "SNGS-001",
    ]
