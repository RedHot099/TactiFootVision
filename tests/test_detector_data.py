"""detector-data 03: source converters, manifests and leak checks, on tiny synthetic fixtures."""

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.detector_data import common, roboflow  # noqa: E402
from scripts.detector_data import soccernet_v3 as v3_script  # noqa: E402

from tactifoot_vision.data import Annotations, Dataset, Sample, Task  # noqa: E402
from tactifoot_vision.data.soccernet_v3 import V3Image, read_labels  # noqa: E402

GAME = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
HELD = "spain_laliga/2016-2017/2017-04-08 - 21-45 Malaga 2 - 0 Barcelona"


def picture(seed: int, size=(360, 640)) -> np.ndarray:
    """A smooth random picture (blurred noise), so resized copies keep their structure."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


def write(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def sample(path: Path, boxes=((10, 10, 50, 80),), classes=(2,)) -> Sample:
    h, w = cv2.imread(str(path)).shape[:2]
    return Sample(
        path, w, h, Annotations(np.asarray(boxes, np.float32), np.asarray(classes))
    )


def v3_labels(images: dict) -> dict:
    """A Labels-v3.json with the given ``{name: [(class, x1, y1, x2, y2), ...]}`` (action or replay by name)."""
    labels: dict = {"GameMetadata": {"UrlLocal": GAME}, "actions": {}, "replays": {}}
    for name, boxes in images.items():
        part = "replays" if "_" in name else "actions"
        labels[part][name] = {
            "imageMetadata": {
                "localpath": GAME,
                "imageType": part[:-1],
                "width": 640,
                "height": 360,
                "set": "train",
                "half": 1,
                "position": 1000,
                "label": "Goal",
            },
            "bboxes": [
                {
                    "class": c,
                    "points": {"x1": x1, "x2": x2, "y1": y1, "y2": y2},
                    "ID": None,
                    "UAI": "",
                }
                for c, x1, y1, x2, y2 in boxes
            ],
        }
    return labels


# ------------------------------------------------------------- v3 reader
def test_read_labels_maps_classes_ignores_staff_and_drops_cards(tmp_path):
    path = tmp_path / "Labels-v3.json"
    path.write_text(
        json.dumps(
            v3_labels(
                {
                    "0.png": [
                        ("Player team left", 10, 10, 30, 60),
                        ("Goalkeeper team right", 40, 10, 60, 60),
                        ("Side referee", 70, 10, 90, 60),
                        ("Ball", 102, 52, 100, 50),  # x1 > x2: normalised
                        ("Staff members", 200, 10, 230, 60),
                        ("Yellow card", 5, 5, 8, 8),
                    ],
                    "0_1.png": [("Main referee", 1, 1, 9, 9)],
                }
            )
        )
    )
    images = read_labels(path)
    assert [i.name for i in images] == ["0.png", "0_1.png"]
    action, replay = images
    assert action.classes == ["player", "goalkeeper", "referee", "ball"]
    np.testing.assert_allclose(action.boxes[3], [100, 50, 102, 52])
    np.testing.assert_allclose(action.ignore_boxes, [[200, 10, 230, 60]])
    assert action.game == GAME and action.image_type == "action"
    assert replay.image_type == "replay" and replay.classes == ["referee"]


def test_read_labels_refuses_unknown_class(tmp_path):
    path = tmp_path / "Labels-v3.json"
    path.write_text(json.dumps(v3_labels({"0.png": [("Linesman drone", 1, 1, 5, 5)]})))
    with pytest.raises(ValueError, match="unknown class"):
        read_labels(path)


# ------------------------------------------------------------- game rules
def test_held_out_games_reads_evaluation_and_gate_roles(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "game,league,season,home,official_split,role,fold\n"
        f"{GAME},england_epl,2014-2015,Chelsea,train,training,2\n"
        f"{HELD},spain_laliga,2016-2017,Malaga,test,evaluation,\n"
        "a/b/c,x,y,z,valid,gate,\n"
    )
    assert common.held_out_games(plan) == {HELD, "a/b/c"}


def test_check_games_rejects_held_out_and_gsr_games():
    assert common.check_games([GAME], {HELD})["games"] == 1
    with pytest.raises(AssertionError, match="held-out"):
        common.check_games([GAME, HELD], {HELD})
    with pytest.raises(AssertionError, match="GSR"):
        common.check_games(["switzerland_super-league/2019-2020/x"], set())
    assert common.is_gsr_game("anything/2019-2020/match")
    assert not common.is_gsr_game(GAME)


# ------------------------------------------------------- near duplicates
def test_same_picture_resized_and_recompressed_is_a_duplicate_other_is_not(tmp_path):
    original = write(tmp_path / "ref" / "a.jpg", picture(1, (1080, 1920)))
    other = write(tmp_path / "ref" / "b.jpg", picture(2, (1080, 1920)))
    copy = tmp_path / "query" / "a_640.jpg"
    copy.parent.mkdir()
    cv2.imwrite(
        str(copy),
        cv2.resize(cv2.imread(str(original)), (640, 640)),
        [cv2.IMWRITE_JPEG_QUALITY, 70],
    )
    fresh = write(tmp_path / "query" / "c.png", picture(3))
    hits = common.near_duplicates(
        common.Fingerprints.of([copy, fresh]), common.Fingerprints.of([original, other])
    )
    assert [(Path(h["image"]).name, Path(h["reference"]).name) for h in hits] == [
        ("a_640.jpg", "a.jpg")
    ]


def test_hash_candidate_with_different_picture_is_not_a_duplicate():
    a = common.Fingerprints(
        ["a"], np.array([0], np.uint64), picture(1, (36, 64))[..., 0][None]
    )
    b = common.Fingerprints(
        ["b"], np.array([1], np.uint64), picture(2, (36, 64))[..., 0][None]
    )
    assert len(common.hamming_pairs(a.hashes, b.hashes)) == 1
    assert common.near_duplicates(a, b) == []


def test_dedup_within_keeps_first_of_each_duplicate_group(tmp_path):
    paths = [
        write(tmp_path / "f0.png", picture(1)),
        write(tmp_path / "f1.png", picture(1) // 2 * 2),  # same picture
        write(tmp_path / "f2.png", picture(2)),
        write(tmp_path / "f3.png", picture(1)),
    ]
    assert common.dedup_within(common.Fingerprints.of(paths)) == [1, 3]


def test_leak_check_drops_benchmark_duplicates_and_reports_them(tmp_path):
    bench = write(tmp_path / "bench" / "000001.jpg", picture(7))
    leaked = write(tmp_path / "src" / "leak.jpg", picture(7))
    clean = write(tmp_path / "src" / "clean.jpg", picture(8))
    dataset = Dataset(
        Task.DETECT, common.CLASS_NAMES, {"train": [sample(leaked), sample(clean)]}
    )
    kept, report = common.leak_check(
        dataset, {"gsr_valid_test": common.Fingerprints.of([bench])}
    )
    assert [s.image_path for s in kept["train"]] == [clean]
    assert report["dropped"] == [str(leaked)]
    assert report["hits"]["gsr_valid_test"][0]["reference"] == str(bench)


# ---------------------------------------------------------------- export
def test_grey_out_blanks_only_the_region():
    image = np.zeros((20, 20, 3), np.uint8)
    out = common.grey_out(image, np.array([[2.4, 3, 6, 25]]))
    assert (out[3:20, 2:6] == common.GREY).all()
    assert out[:3].sum() == 0 and out[:, 6:].sum() == 0 and image.sum() == 0


def test_export_set_writes_yolo_manifest_and_label_hashes(tmp_path):
    image = write(tmp_path / "src" / "img.jpg", picture(1))
    dataset = Dataset(
        Task.DETECT,
        common.CLASS_NAMES,
        {"train": [sample(image, ((10, 10, 20, 20), (30, 30, 60, 90)), (0, 2))]},
    )
    out = tmp_path / "sets" / "toy"
    manifest = common.export_set(
        dataset,
        out,
        {"source": "toy", "licence": "x"},
        ["ball", "player"],
        {str(image): {"game": "g1", "image": "1.png", "split": "x"}},
        {"dropped": []},
    )
    assert manifest["not_annotated_classes"] == ["goalkeeper", "referee"]
    assert manifest["boxes_total"] == {
        "ball": 1,
        "goalkeeper": 0,
        "player": 1,
        "referee": 0,
    }
    assert manifest["per_image"] == [
        {"game": "g1", "split": "train", "image": str(image)}
    ]
    written = json.loads((out / "manifest.json").read_text())
    assert written["label_sha256_of_list"] == common.sha256_file(
        out / "label_sha256.txt"
    )
    sha, rel = (out / "label_sha256.txt").read_text().split()
    assert rel == "train/labels/img.txt" and sha == common.sha256_file(out / rel)
    assert (out / "leak_report.json").is_file()
    reloaded = Dataset.load(out)
    assert reloaded.summary().loc["train", "player"] == 1
    # re-running into the same folder works (manifest files beside the export are kept apart)
    common.export_set(dataset, out, {"source": "toy"}, ["ball"])


# ------------------------------------------------------------- v3 converter
def v3_image(boxes, classes) -> V3Image:
    return V3Image(
        GAME, "3.png", "action", 640, 360, np.asarray(boxes, np.float32), list(classes)
    )


def test_refine_ball_uses_v3d_diameter_on_original_centre():
    image = v3_image([[100, 100, 120, 110], [0, 0, 10, 30]], ["ball", "player"])
    row = {"bbox": [100, 100, 20, 10], "d": 12.0, "rep_error": 2.0}
    assert v3_script.refine_ball(image, row) == "refined"
    np.testing.assert_allclose(image.boxes[0], [104, 99, 116, 111])
    np.testing.assert_allclose(image.boxes[1], [0, 0, 10, 30])


@pytest.mark.parametrize(
    ("row", "status"),
    [
        (None, "no_v3d"),
        ({"bbox": [50, 50, 20, 10], "d": 12.0, "rep_error": 2.0}, "unmatched"),
        ({"bbox": [100, 100, 20, 10], "d": 12.0, "rep_error": 30.0}, "rep_error"),
        ({"bbox": [100, 100, 20, 10], "d": 60.0, "rep_error": 2.0}, "size_ratio"),
    ],
)
def test_refine_ball_keeps_v3_box_when_v3d_is_unusable(row, status):
    image = v3_image([[100, 100, 120, 110]], ["ball"])
    assert v3_script.refine_ball(image, row) == status
    np.testing.assert_allclose(image.boxes[0], [100, 100, 120, 110])


def test_refine_index_keys_actions_and_replays(tmp_path):
    path = tmp_path / "SNv3D.csv"
    path.write_text(
        ",league,season,match,main_action,action,replay,img_w,img_h,is_ball,ball_bbox,optimized_d,rep_error,set\n"
        '0,england_epl,2014-2015,m,11,1.0,,1280,720,True,"[1.0, 2.0, 3.0, 4.0]",3.5,1.2,train\n'
        '1,england_epl,2014-2015,m,11,,0.0,1280,720,True,"[5.0, 6.0, 7.0, 8.0]",7.5,0.5,test\n'
    )
    index = v3_script.refine_index(path)
    assert index[("england_epl/2014-2015/m", "11.png")]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert index[("england_epl/2014-2015/m", "11_0.png")]["d"] == 7.5


def test_convert_game_greys_staff_and_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(v3_script, "WORK", tmp_path / "work")
    game_dir = tmp_path / "raw" / GAME
    game_dir.mkdir(parents=True)
    labels = game_dir / "Labels-v3.json"
    labels.write_text(
        json.dumps(
            v3_labels(
                {
                    "0.png": [
                        ("Player team left", 10, 10, 30, 60),
                        ("Staff members", 200, 100, 240, 200),
                    ],
                    "0_0.png": [("Ball", 5, 5, 9, 9)],
                }
            )
        )
    )
    frame = np.full((360, 640, 3), 30, np.uint8)
    with zipfile.ZipFile(game_dir / "Frames-v3.zip", "w") as archive:
        for name in ("0.png", "0_0.png"):
            archive.writestr(
                f"Frames-v3/{name}", cv2.imencode(".png", frame)[1].tobytes()
            )
    records = v3_script.convert_game(str(labels), {})
    assert [r["v3_name"] for r in records] == ["0.png", "0_0.png"]
    action = cv2.imread(records[0]["path"])
    assert (
        abs(int(action[150, 220, 0]) - common.GREY) <= 2
        and abs(int(action[5, 5, 0]) - 30) <= 2
    )
    assert records[0]["game"] == GAME and records[1]["image_type"] == "replay"
    dataset = v3_script.to_dataset(records)
    assert (
        dataset.summary().loc["train", "player"] == 1
        and dataset.summary().loc["train", "ball"] == 1
    )


# --------------------------------------------------------------- roboflow
def test_roboflow_rf_copies_collapse_to_one_image_per_source_stem(tmp_path):
    a1 = write(tmp_path / "a_jpg.rf.0123abcd.jpg", picture(1))
    a2 = write(tmp_path / "a_jpg.rf.89ef0011.jpg", picture(2))
    b = write(tmp_path / "b_jpg.rf.4567.jpg", picture(3))
    assert roboflow.source_stem(a1) == "a_jpg"
    dataset = Dataset(
        Task.DETECT,
        common.CLASS_NAMES,
        {"train": [sample(a2), sample(b)], "valid": [sample(a1)]},
    )
    kept, dropped = roboflow.dedup_rf_copies(dataset)
    assert dropped == 1
    assert [s.image_path.name for s in kept["train"]] == [a2.name, b.name] and kept[
        "valid"
    ] == []


def test_roboflow_remap_reindexes_classes_by_name(tmp_path):
    image = write(tmp_path / "x.jpg", picture(1))
    ball_only = Dataset(Task.DETECT, ["ball"], {"train": [sample(image, classes=(0,))]})
    players = Dataset(
        Task.DETECT,
        ["player", "ball"],
        {"train": [sample(image, ((1, 1, 5, 5), (2, 2, 4, 4)), (0, 1))]},
    )
    assert roboflow.remap(ball_only)["train"][0].annotations.class_ids.tolist() == [0]
    assert roboflow.remap(players)["train"][0].annotations.class_ids.tolist() == [2, 0]


# ------------------------------------------------------------- soccersynth
def test_soccersynth_parse_yolo_maps_person_to_player_and_ball():
    from scripts.detector_data import soccersynth

    boxes, classes = soccersynth.parse_yolo(
        "1 0.5 0.5 0.1 0.2\r\n0 0.25 0.5 0.5 1.0\r\n\n", 200, 100
    )
    assert classes == ["ball", "player"]
    np.testing.assert_allclose(boxes, [[90, 40, 110, 60], [0, 0, 100, 100]])
    with pytest.raises(ValueError, match="unknown"):
        soccersynth.parse_yolo("2 0.5 0.5 0.1 0.1", 10, 10)


def test_soccersynth_convert_members_reads_from_the_archive(tmp_path):
    from scripts.detector_data import soccersynth

    archive_path = tmp_path / "synth.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "train/0.png", cv2.imencode(".png", picture(1, (600, 800)))[1].tobytes()
        )
        archive.writestr("train/0.txt", "1 0.5 0.5 0.01 0.01\r\n")
        archive.writestr(
            "val/7.png", cv2.imencode(".png", picture(2, (1080, 1920)))[1].tobytes()
        )
        archive.writestr("val/7.txt", "0 0.5 0.5 0.1 0.3\r\n")
    records = soccersynth.convert_members(
        str(archive_path), ["train/0", "val/7"], str(tmp_path / "work")
    )
    assert [(r["split"], r["variant"], r["classes"]) for r in records] == [
        ("train", "motion_blur", ["ball"]),
        ("valid", "clear", ["player"]),
    ]
    assert Path(records[1]["path"]).name == "7.jpg" and cv2.imread(
        records[1]["path"]
    ).shape == (1080, 1920, 3)
    dataset = soccersynth.to_dataset(records)
    assert dataset.split_names == ["train", "valid"]


# --------------------------------------------------------------- sportsmot
def test_sportsmot_download_keeps_only_football_gt_info_and_strided_frames():
    from scripts.detector_data import download_sportsmot as dl

    clips = {"v_abc_c001"}
    assert dl.keep("train/v_abc_c001/gt/gt.txt", clips, 10) == Path(
        "v_abc_c001/gt/gt.txt"
    )
    assert dl.keep("train/v_abc_c001/seqinfo.ini", clips, 10) == Path(
        "v_abc_c001/seqinfo.ini"
    )
    assert dl.keep("train/v_abc_c001/img1/000011.jpg", clips, 10) == Path(
        "v_abc_c001/img1/000011.jpg"
    )
    assert dl.keep("train/v_abc_c001/img1/000012.jpg", clips, 10) is None
    assert dl.keep("train/v_basket_c001/img1/000001.jpg", clips, 10) is None
    assert dl.keep("train/v_abc_c001/img1/../../evil.jpg", clips, 1) is None


def test_sportsmot_clip_samples_reads_gt_at_stride(tmp_path):
    from scripts.detector_data import sportsmot

    clip = tmp_path / "v_abc_c001"
    (clip / "gt").mkdir(parents=True)
    (clip / "seqinfo.ini").write_text("[Sequence]\nimWidth=640\nimHeight=360\n")
    (clip / "gt" / "gt.txt").write_text(
        "1,1,10,20,30,40,1,1,1\n1,2,50,60,10,10,1,1,1\n11,1,0,0,5,5,1,1,1\n2,1,0,0,5,5,1,1,1\n"
    )
    for frame in (1, 2, 11):
        write(clip / "img1" / f"{frame:06d}.jpg", picture(frame, (360, 640)))
    samples, provenance = sportsmot.clip_samples(clip)
    assert [s.image_path.name for s in samples] == ["000001.jpg", "000011.jpg"]
    np.testing.assert_allclose(
        samples[0].annotations.boxes, [[10, 20, 40, 60], [50, 60, 60, 70]]
    )
    assert set(samples[0].annotations.class_ids.tolist()) == {
        common.CLASS_NAMES.index("player")
    }
    assert provenance[str(samples[1].image_path)] == {
        "video": "v_abc",
        "clip": "v_abc_c001",
        "frame": 11,
    }
