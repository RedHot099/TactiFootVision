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
def test_xg_roles_reads_every_plan_role(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "game,league,season,home,official_split,role,fold\n"
        f"{GAME},england_epl,2014-2015,Chelsea,train,training,2\n"
        f"{HELD},spain_laliga,2016-2017,Malaga,test,evaluation,\n"
        "a/b/c,x,y,z,valid,gate,\n"
    )
    assert common.xg_roles(plan) == {
        GAME: "training",
        HELD: "evaluation",
        "a/b/c": "gate",
    }


def test_game_tags_count_roles_and_reject_gsr_games():
    report = common.game_tags(
        [GAME, HELD, HELD, "x/2015-2016/y"], {GAME: "training", HELD: "evaluation"}
    )
    assert report["games_by_xg_role"] == {"training": 1, "evaluation": 1, "none": 1}
    with pytest.raises(AssertionError, match="GSR"):
        common.game_tags(["switzerland_super-league/2019-2020/x"], {})
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


def test_near_duplicates_are_tagged_with_the_gsr_frame_and_split_never_dropped(
    tmp_path,
):
    gsr_frame = write(
        tmp_path / "SoccerNetGS" / "valid" / "SNGS-021" / "img1" / "000007.jpg",
        picture(7),
    )
    yolo_valid = write(
        tmp_path / "football_yolo" / "valid" / "images" / "x.jpg", picture(9)
    )
    leaked = write(tmp_path / "src" / "leak.jpg", picture(7))
    other = write(tmp_path / "src" / "other.jpg", picture(9))
    clean = write(tmp_path / "src" / "clean.jpg", picture(8))
    tags, report = common.tag_near_duplicates(
        [leaked, other, clean],
        {
            "gsr_valid_test": common.Fingerprints.of([gsr_frame]),
            "football_yolo_valid": common.Fingerprints.of([yolo_valid]),
        },
    )
    assert set(tags) == {str(leaked), str(other), str(clean)}
    assert tags[str(leaked)]["near_dup_of"] == "gsr:valid/SNGS-021/000007"
    assert tags[str(leaked)]["near_dup_set"] == "gsr_valid"
    assert tags[str(leaked)]["gsr_split"] == "valid"
    assert tags[str(other)]["near_dup_of"] == "football_yolo_valid:x.jpg"
    assert tags[str(other)]["gsr_split"] is None
    assert (
        tags[str(clean)]["near_dup_of"] is None
        and tags[str(clean)]["near_dup_hits"] == 0
    )
    assert report["images_tagged"] == 2 and "dropped" not in report


def test_thin_video_frames_counts_what_it_thins_and_no_thin_keeps_all(tmp_path):
    frames = [
        write(tmp_path / f"f{i}.png", picture(1 if i < 2 else i)) for i in range(4)
    ]
    dataset = Dataset(
        Task.DETECT, common.CLASS_NAMES, {"train": [sample(p) for p in frames]}
    )
    thinned, removed = common.thin_video_frames(dataset)
    assert removed == 1 and [s.image_path.name for s in thinned["train"]] == [
        "f0.png",
        "f2.png",
        "f3.png",
    ]
    kept, removed = common.thin_video_frames(dataset, thin=False)
    assert removed == 0 and len(kept) == 4


# ---------------------------------------------------------------- export
def test_grey_out_blanks_only_the_region():
    image = np.zeros((20, 20, 3), np.uint8)
    out = common.grey_out(image, np.array([[2.4, 3, 6, 25]]))
    assert (out[3:20, 2:6] == common.GREY).all()
    assert out[:3].sum() == 0 and out[:, 6:].sum() == 0 and image.sum() == 0


MANIFEST = {
    "source": "toy",
    "licence_tag": "CC0",
    "licence": "x",
    "access_route": "local",
}


def test_export_set_writes_yolo_manifest_schema_rows_and_label_hashes(tmp_path):
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
        MANIFEST,
        ["ball", "player"],
        {str(image): {"game": "g1", "xg_role": "gate"}},
        {"hits": {}},
    )
    assert manifest["not_annotated_classes"] == ["goalkeeper", "referee"]
    assert manifest["boxes_total"] == {
        "ball": 1,
        "goalkeeper": 0,
        "player": 1,
        "referee": 0,
    }
    (row,) = manifest["per_image"]
    assert set(row) == set(common.SCHEMA)
    assert (
        row["image"] == str(out.resolve() / "train" / "images" / "img.jpg")
        and Path(row["image"]).is_file()
    )
    assert (
        row["source_image"] == str(image)
        and row["set"] == "toy"
        and row["split"] == "train"
    )
    assert (row["game"], row["xg_role"], row["near_dup_of"]) == ("g1", "gate", None)
    assert (row["n_ball"], row["n_player"], row["n_referee"]) == (1, 1, 0)
    assert row["annotated_classes"] == "ball,player" and row["licence"] == "CC0"
    assert manifest["images_by_xg_role"] == {"gate": 1}
    written = json.loads((out / "manifest.json").read_text())
    assert written["label_sha256_of_list"] == common.sha256_file(
        out / "label_sha256.txt"
    )
    sha, rel = (out / "label_sha256.txt").read_text().split()
    assert rel == "train/labels/img.txt" and sha == common.sha256_file(out / rel)
    assert (out / "leak_report.json").is_file()
    assert Dataset.load(out).summary().loc["train", "player"] == 1
    # re-running into the same folder works (manifest files beside the export are kept apart)
    common.export_set(dataset, out, MANIFEST, ["ball"])


def test_export_set_refuses_tags_outside_the_schema_and_incomplete_manifests(tmp_path):
    image = write(tmp_path / "src" / "img.jpg", picture(1))
    dataset = Dataset(Task.DETECT, common.CLASS_NAMES, {"train": [sample(image)]})
    with pytest.raises(ValueError, match="outside the schema"):
        common.export_set(
            dataset,
            tmp_path / "a",
            MANIFEST,
            ["player"],
            {str(image): {"image": "1.png"}},
        )
    with pytest.raises(ValueError, match="licence_tag"):
        common.export_set(dataset, tmp_path / "b", {"source": "toy"}, ["player"])


def test_pool_index_concatenates_sets_and_rejects_off_schema_rows(tmp_path):
    from scripts.detector_data import pool_index

    images = [write(tmp_path / "src" / f"{i}.jpg", picture(i)) for i in range(3)]
    for name, paths in (("one", images[:2]), ("two", images[2:])):
        dataset = Dataset(
            Task.DETECT, common.CLASS_NAMES, {"train": [sample(p) for p in paths]}
        )
        common.export_set(
            dataset,
            tmp_path / "sets" / name,
            MANIFEST,
            ["player"],
            {str(paths[0]): {"xg_role": "training"}},
        )
    (tmp_path / "sets" / "foreign").mkdir()
    (tmp_path / "sets" / "foreign" / "manifest.json").write_text(
        "{}"
    )  # no rows yet: skipped
    pool = pool_index.build(tmp_path / "sets")
    assert list(pool.columns) == list(common.SCHEMA)
    assert pool.groupby("set").size().to_dict() == {"one": 2, "two": 1}
    assert pool.xg_role.isna().sum() == 1 and pool.n_player.sum() == 3
    assert not pool_index.summary(pool).empty
    manifest = tmp_path / "sets" / "two" / "manifest.json"
    data = json.loads(manifest.read_text())
    del data["per_image"][0]["xg_role"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="off schema"):
        pool_index.build(tmp_path / "sets")


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
        "game": "youtube:abc",
        "clip": "v_abc_c001",
        "frame": 11,
    }


# ------------------------------------------------------------------- issia
XGTF = """<?xml version="1.0" encoding="UTF-8"?>
<viper xmlns="http://lamp.cfar.umd.edu/viper#" xmlns:data="http://lamp.cfar.umd.edu/viperdata#">
 <data><sourcefile filename="x">
  <object framespan="0:200" id="0" name="BALL">
   <attribute name="BallPos"><data:point framespan="100:101" x="30" y="40"/></attribute>
   <attribute name="BallShot"><data:bvalue framespan="100:100" value="true"/></attribute>
  </object>
  <object framespan="0:200" id="1" name="Person">
   <attribute name="LOCATION"><data:bbox framespan="40:110" height="100" width="20" x="10" y="20"/></attribute>
  </object>
 </sourcefile></data>
</viper>"""


def test_issia_xgtf_parse_frame_shift_mirroring_and_ball_box(tmp_path, monkeypatch):
    from scripts.detector_data import issia

    path = tmp_path / "cam.xgtf"
    path.write_text(XGTF)
    persons, balls = issia.parse_xgtf(path)
    assert persons[40] == [[10, 20, 30, 120]] and len(persons) == 71
    assert balls == {100: [[30, 40]], 101: [[30, 40]]}
    monkeypatch.setattr(issia, "STRIDE", 5)
    labels = issia.frame_labels(persons, balls, 1920, camera=1)
    # first annotated frame 40 + 50 skipped -> 90, 95, ..., 110; video frame = annotation frame - 8
    assert sorted(labels) == [82, 87, 92, 97, 102]
    boxes, ids = labels[92]
    assert ids == [2, 0]
    np.testing.assert_allclose(
        boxes[1], [24, 34, 36, 46]
    )  # 0.12 x 100 px = 12 px square around (30, 40)
    mirrored = issia.ball_box([30, 40], [[10, 20, 30, 120]], 1920, camera=2)
    np.testing.assert_allclose(mirrored, [1884, 34, 1896, 46])


# ------------------------------------------------ soccer player detection
def test_soccer_player_detection_reads_mat_annotations(tmp_path):
    from scipy.io import savemat
    from scripts.detector_data import soccer_player_detection as spd

    annot = np.zeros((1, 2), dtype=[("BBox", "O"), ("ImgName", "O")])
    annot[0, 0] = (np.array([[1.0, 2.0, 11.0, 22.0]]), np.array(["0001.jpg"]))
    annot[0, 1] = (np.zeros((0, 4)), np.array(["0002.jpg"]))
    for game in (1, 2):
        savemat(tmp_path / f"annotation_{game}.mat", {"annot": annot})
        write(tmp_path / f"DataSet_{game:03d}" / "0001.jpg", picture(game))
    read = spd.read_annotations(tmp_path / "annotation_1.mat")
    np.testing.assert_allclose(read["0001.jpg"], [[1, 2, 11, 22]])
    assert read["0002.jpg"].shape == (0, 4)
    dataset, tags = spd.build(tmp_path)  # 0002.jpg has no image file: skipped
    assert len(dataset) == 2
    first = dataset["train"][0]
    assert tags[str(first.image_path)] == {
        "game": "youtube:XRfXrj9Bkiw",
        "clip": "DataSet_001",
        "frame": 1,
    }
    assert first.annotations.class_ids.tolist() == [2]


# -------------------------------------------------------------- remote zip
def test_remote_zip_lists_and_extracts_members_through_byte_ranges(
    tmp_path, monkeypatch
):
    from scripts.detector_data import remote_zip

    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "teamtrack/soccer_side/train/a.csv",
            "x" * 5000,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        z.writestr(
            "teamtrack/basketball/b.csv", "stored", compress_type=zipfile.ZIP_STORED
        )
        z.writestr("../evil.txt", "no")
    blob = archive.read_bytes()
    monkeypatch.setattr(remote_zip, "total_size", lambda url: len(blob))
    monkeypatch.setattr(
        remote_zip, "get_range", lambda url, start, end: blob[start : end + 1]
    )
    members = {m["name"]: m for m in remote_zip.central_directory("u")}
    assert set(members) == {
        "teamtrack/soccer_side/train/a.csv",
        "teamtrack/basketball/b.csv",
        "../evil.txt",
    }
    out = tmp_path / "out"
    for name in ("teamtrack/soccer_side/train/a.csv", "teamtrack/basketball/b.csv"):
        remote_zip.extract("u", members[name], remote_zip.safe_path(out, name))
    assert (out / "teamtrack/soccer_side/train/a.csv").read_text() == "x" * 5000
    assert (out / "teamtrack/basketball/b.csv").read_text() == "stored"
    with pytest.raises(ValueError, match="unsafe"):
        remote_zip.safe_path(out, "../evil.txt")


# --------------------------------------------------------------- teamtrack
def test_teamtrack_csv_reads_players_and_ball_and_skips_empty_cells(tmp_path):
    from scripts.detector_data import teamtrack

    path = tmp_path / "F_20220220_1_0000_0030.csv"
    path.write_text(
        "TeamID,0,0,0,0,BALL,BALL,BALL,BALL\n"
        "PlayerID,7,7,7,7,BALL,BALL,BALL,BALL\n"
        "Attributes,bb_height,bb_left,bb_top,bb_width,bb_height,bb_left,bb_top,bb_width\n"
        "frame,,,,,,,,\n"
        "1,40.0,100.0,50.0,20.0,6.0,300.0,80.0,6.0\n"
        "2,41.0,101.0,51.0,21.0,,,,\n"
    )
    labels = teamtrack.read_annotations(path)
    boxes, ids = labels[1]
    np.testing.assert_allclose(boxes, [[100, 50, 120, 90], [300, 80, 306, 86]])
    assert ids == [2, 0]
    assert labels[2][1] == [2]
