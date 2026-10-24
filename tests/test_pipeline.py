import dataclasses
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.data import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.data.annotations import Task
from tactifoot_vision.enums import TrackerType
from tactifoot_vision.evaluation import evaluate_tracking
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline import (
    NO_TEAM,
    FrameAnalyzer,
    Pipeline,
    PipelineResult,
    fill_ball,
    track_ball,
)
from tactifoot_vision.pitch import SoccerPitch, frame_to_pitch, pitch_to_frame
from tactifoot_vision.teams import Embedder, TeamClassifier
from tactifoot_vision.tracking import ByteTrackTracker, Tracker

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, FRAMES = 320, 240, 25.0, 12
# frame <- pitch: ~2.9 px per metre plus a little perspective
PITCH_TO_FRAME = np.array([[2.9, 0.1, 5.0], [0.0, 3.1, 12.0], [0.0, 0.0005, 1.0]])
FRAME_TO_PITCH = np.linalg.inv(PITCH_TO_FRAME)
COLORS = {  # BGR kit colour -> detector class
    (0, 0, 230): "player",  # red team
    (230, 0, 0): "player",  # blue team
    (0, 230, 230): "goalkeeper",  # yellow
    (0, 0, 0): "referee",
    (255, 255, 255): "ball",
}
RED_PLAYERS = [(40, 60), (60, 150), (90, 100)]  # top-left corners at frame 0 (pixels)
BLUE_PLAYERS = [(180, 50), (210, 140), (240, 90)]
KITS = ((0, 0, 230), (230, 0, 0))  # BGR kits of the RED_PLAYERS and BLUE_PLAYERS
GOALKEEPER = (12, 110)
REFEREE = (150, 190)
TELEPORT_FRAME = 6  # the ball jumps across the pitch for one frame


def _scene(
    t: int, kits=KITS, goalkeeper=GOALKEEPER
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """(colour, xyxy) of every object in frame ``t``; people move 1 px per frame."""
    objects = []
    for color, corners in zip(kits, (RED_PLAYERS, BLUE_PLAYERS), strict=True):
        objects += [(color, (x + t, y, x + t + 8, y + 18)) for x, y in corners]
    gx, gy = goalkeeper
    objects.append(((0, 230, 230), (gx, gy + t, gx + 8, gy + t + 18)))
    rx, ry = REFEREE
    objects.append(((0, 0, 0), (rx - t, ry, rx - t + 8, ry + 18)))
    bx = 150 + 2 * t if t != TELEPORT_FRAME else 290
    objects.append(((255, 255, 255), (bx, 120, bx + 6, 126)))
    return objects


def _write_video(path: Path, **scene) -> Path:
    """The synthetic clip; ``scene`` changes the kits or the goalkeeper's place."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    for t in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (40, 140, 40), dtype=np.uint8)
        for color, (x1, y1, x2, y2) in _scene(t, **scene):
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, thickness=-1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    return _write_video(tmp_path_factory.mktemp("video") / "synthetic.avi")


class ColorDetector(Model):
    """Finds the solid rectangles of the synthetic video by colour."""

    name = "color_detector"
    task = Task.DETECT

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        return ["ball", "goalkeeper", "player", "referee"]

    def predict(self, image: np.ndarray) -> sv.Detections:
        boxes, names = [], []
        for color, name in COLORS.items():
            distance = np.abs(image.astype(int) - np.array(color)).sum(axis=2)
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                (distance < 120).astype(np.uint8)
            )
            for x, y, w, h, area in stats[1:]:
                if area >= 12:
                    boxes.append([x, y, x + w, y + h])
                    names.append(name)
        names = np.array(names, dtype=str)
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.full(len(names), 0.9, dtype=np.float32),
            class_id=np.array([self.class_names.index(n) for n in names], dtype=int),
            data={"class_name": names},
        )


class KnownHomographyKeypoints(Model):
    """Returns the 32 pitch landmarks projected with a fixed homography."""

    name = "known_keypoints"
    task = Task.POSE

    def _load(self, weights):
        pass

    def _train(self, dataset, config, run_dir):
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        return ["pitch"]

    def predict(self, image: np.ndarray) -> sv.KeyPoints:
        xy = pitch_to_frame(SoccerPitch().vertices, FRAME_TO_PITCH)
        return sv.KeyPoints(
            xy=xy[None].astype(np.float32), confidence=np.ones((1, len(xy)), np.float32)
        )


class MeanColorEmbedder(Embedder):
    name = "mean_color"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        self.calls += 1
        return np.stack([c.reshape(-1, 3).mean(axis=0) / 255 for c in crops]).astype(
            np.float32
        )


def _pipeline(**kwargs) -> Pipeline:
    options = {
        "keypoint_model": KnownHomographyKeypoints(device="cpu"),
        "team_classifier": TeamClassifier(MeanColorEmbedder(), reducer=None),
    } | kwargs
    return Pipeline(ColorDetector(device="cpu"), **options)


def _track_of(frame, corner_x: float, corner_y: float) -> int:
    """Index of the detection whose box starts at the given corner."""
    distance = np.abs(frame.detections.xyxy[:, :2] - [corner_x, corner_y]).sum(axis=1)
    assert distance.min() < 3
    return int(distance.argmin())


@pytest.fixture(scope="module")
def result(video) -> PipelineResult:
    return _pipeline().run(video, progress=False)


def test_result_metadata(result, video):
    assert len(result) == FRAMES
    assert [f.index for f in result] == list(range(FRAMES))
    assert result[3].timestamp == pytest.approx(3 / FPS)
    assert result.fps == FPS and result.frame_size == (WIDTH, HEIGHT)
    assert result.video_path == video
    assert result.class_names == ["ball", "goalkeeper", "player", "referee"]


def test_people_are_tracked_with_class_names(result):
    for f in result:
        assert len(f.detections) == 8
        assert f.detections.tracker_id is not None
        assert f.detections.mask is None
        assert sorted(f.detections.data["class_name"]) == sorted(
            ["player"] * 6 + ["goalkeeper", "referee"]
        )
    # every object keeps its id through the clip
    ids = [
        [
            int(f.detections.tracker_id[_track_of(f, x + t, y)])
            for x, y in RED_PLAYERS + BLUE_PLAYERS
        ]
        for t, f in enumerate(result)
    ]
    assert all(row == ids[0] for row in ids)
    assert len(result.track_ids) == 8


def test_pitch_positions_follow_the_homography(result):
    frame = result[2]
    assert frame.homography is not None
    np.testing.assert_allclose(
        frame.homography, FRAME_TO_PITCH / FRAME_TO_PITCH[2, 2], atol=1e-5
    )
    anchors = frame.detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    np.testing.assert_allclose(
        frame.pitch_xy, frame_to_pitch(anchors, FRAME_TO_PITCH), atol=1e-3
    )
    pitch = SoccerPitch()
    assert (frame.pitch_xy >= 0).all() and (frame.pitch_xy[:, 0] <= pitch.length).all()
    ball_centre = frame.ball.get_anchors_coordinates(sv.Position.CENTER)
    np.testing.assert_allclose(
        frame.ball_xy, frame_to_pitch(ball_centre, FRAME_TO_PITCH), atol=1e-3
    )
    assert frame.keypoints is not None and frame.keypoints.xy.shape == (1, 32, 2)


def test_teams_by_vote_goalkeeper_by_position_referee_none(result):
    for t, f in enumerate(result):
        teams = f.team_ids
        red = {teams[_track_of(f, x + t, y)] for x, y in RED_PLAYERS}
        blue = {teams[_track_of(f, x + t, y)] for x, y in BLUE_PLAYERS}
        assert red == {0} and blue == {1}  # deterministic numbering
        names = f.detections.data["class_name"]
        assert teams[names == "referee"].tolist() == [NO_TEAM]
        # the yellow goalkeeper stands next to the red team
        assert teams[names == "goalkeeper"].tolist() == [0]


def test_the_pass_keeps_each_tracks_mean_team_embedding(result):
    # the team-class people's crops were embedded (mean colour); the referee's were not
    frame = result[0]
    names = frame.detections.data["class_name"]
    ids = frame.detections.tracker_id
    assert set(result.track_embeddings) == {int(i) for i in ids[names != "referee"]}
    keeper = int(ids[names == "goalkeeper"][0])
    red = int(ids[_track_of(frame, *RED_PLAYERS[0])])
    yellow, red_kit = np.array([0, 230, 230]) / 255, np.array(KITS[0]) / 255
    assert result.track_embeddings[keeper].dtype == np.float16
    # the vote samples the first frame of a track and every 10th: frames 0 and 10
    assert result.track_embedding_samples == dict.fromkeys(result.track_embeddings, 2)
    assert np.linalg.norm(result.track_embeddings[keeper] - yellow) < 0.15
    assert np.linalg.norm(result.track_embeddings[red] - red_kit) < 0.15


def test_ball_is_single_and_the_pass_keeps_its_jumps(result):
    # choosing the ball is track_ball's job: the pass keeps the teleport
    for t, f in enumerate(result):
        assert len(f.ball) == 1
        assert f.ball.data["class_name"].tolist() == ["ball"]
        assert np.isfinite(f.ball_xy).all()
        if t == TELEPORT_FRAME:
            assert f.ball.xyxy[0, 0] == pytest.approx(290, abs=2)


def test_table_export(result):
    table = result.to_dataframe()
    assert len(table) == FRAMES * 9
    people = table[table.object == "person"]
    assert set(people.team_id) == {NO_TEAM, 0, 1}


def test_frame_range_start_end_stride(video):
    embedder = MeanColorEmbedder()
    pipeline = _pipeline(team_classifier=TeamClassifier(embedder, reducer=None))
    result = pipeline.run(video, start=1, end=9, stride=2, progress=False)
    assert [f.index for f in result] == [1, 3, 5, 7]
    assert result[1].timestamp == pytest.approx(3 / FPS)
    assert {tuple(f.team_ids) for f in result} != {(NO_TEAM,) * 8}
    assert [f.index for f in pipeline.run(video, start=8, progress=False)] == [
        8, 9, 10, 11,
    ]  # fmt: skip


def test_include_classes_and_no_keypoints(video):
    pipeline = _pipeline(keypoint_model=None, include_classes=["player"])
    result = pipeline.run(video, end=3, progress=False)
    for f in result:
        assert set(f.detections.data["class_name"]) == {"player"}
        assert len(f.ball) == 1  # the ball is always kept
        assert f.homography is None and np.isnan(f.pitch_xy).all()
        assert set(f.team_ids) == {0, 1}


def test_without_tracker_each_detection_gets_a_team(video):
    result = _pipeline(tracker=None).run(video, end=3, progress=False)
    for t, f in enumerate(result):
        assert f.detections.tracker_id is None
        assert {f.team_ids[_track_of(f, x + t, y)] for x, y in RED_PLAYERS} == {0}
        assert {f.team_ids[_track_of(f, x + t, y)] for x, y in BLUE_PLAYERS} == {1}
    assert result.track_embeddings == {}  # no tracks to keep an embedding for


def test_prefitted_classifier_is_reused_when_asked(video):
    classifier = TeamClassifier(MeanColorEmbedder(), reducer=None)
    blue_first = [
        np.full((10, 6, 3), c, np.uint8) for c in [(230, 0, 0)] * 3 + [(0, 0, 230)] * 3
    ]
    classifier.fit(blue_first)
    before = classifier._kmeans
    result = _pipeline(team_classifier=classifier, refit_teams=False).run(
        video, end=2, progress=False
    )
    assert classifier._kmeans is before
    f = result[0]
    assert (
        f.team_ids[_track_of(f, *RED_PLAYERS[0])]
        == classifier.predict(blue_first[3:4])[0]
    )


def test_sampling_stride_limits_embedding_calls(video):
    embedder = MeanColorEmbedder()
    pipeline = _pipeline(
        team_classifier=TeamClassifier(embedder, reducer=None), team_sample_stride=4
    )
    pipeline.run(video, progress=False)
    assert embedder.calls == len(
        range(0, FRAMES, 4)
    )  # new tracks only appear in frame 0


def test_no_team_classifier_leaves_no_team(video):
    result = _pipeline(team_classifier=None, tracker="bytetrack").run(
        video, end=2, progress=False
    )
    assert all((f.team_ids == NO_TEAM).all() for f in result)
    assert result.track_embeddings == {}


def test_validation():
    detector, keypoints = (
        ColorDetector(device="cpu"),
        KnownHomographyKeypoints(device="cpu"),
    )
    with pytest.raises(ValueError, match="detection model"):
        Pipeline(keypoints)
    with pytest.raises(ValueError, match="pose model"):
        Pipeline(detector, keypoint_model=detector)
    with pytest.raises(ValueError, match="Unknown tracker"):
        Pipeline(detector, tracker="nope")
    with pytest.raises(ValueError, match="differs"):
        Pipeline(
            detector,
            pitch=SoccerPitch(120, 80),
            homography=tv.pitch.HomographyEstimator(SoccerPitch()),
        )
    pipeline = Pipeline(
        detector, homography=tv.pitch.HomographyEstimator(SoccerPitch(120, 80))
    )
    assert pipeline.pitch == SoccerPitch(120, 80)


class BoxMaskTracker(Tracker):
    """ByteTrack plus a full-frame mask over the top half of every tracked box (like SAM2)."""

    name = "box_mask"

    def __init__(self) -> None:
        self._inner = ByteTrackTracker()

    def reset(self, fps: float | None = None) -> None:
        self._inner.reset(fps)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        tracked = self._inner.update(detections, frame)
        tracked.mask = np.zeros((len(tracked), *frame.shape[:2]), dtype=bool)
        for mask, (x1, y1, x2, y2) in zip(
            tracked.mask, tracked.xyxy.astype(int), strict=True
        ):
            mask[y1 : (y1 + y2) // 2, x1:x2] = True
        return tracked


def test_keep_masks_stores_crops_the_size_of_the_mask(video, tmp_path):
    kept = _pipeline(tracker=BoxMaskTracker(), keep_masks=True).run(
        video, end=3, progress=False
    )
    dropped = _pipeline(tracker=BoxMaskTracker()).run(video, end=3, progress=False)
    assert all(f.masks is None and f.detections.mask is None for f in dropped)
    for frame in kept:
        assert frame.detections.mask is None
        assert len(frame.masks) == len(frame.detections) > 0
        dense = frame.masks.to_dense(WIDTH, HEIGHT)
        boxes = frame.detections.xyxy.astype(int)
        for mask, crop, (x1, y1, x2, y2) in zip(
            dense, frame.masks.crops, boxes, strict=True
        ):
            assert crop.shape == ((y1 + y2) // 2 - y1, x2 - x1)
            assert mask[y1, x1] and not mask[y2 - 1, x1] and mask.sum() == crop.size

    loaded = PipelineResult.load(kept.save(tmp_path / "result.pkl"))
    for before, after in zip(kept, loaded, strict=True):
        np.testing.assert_array_equal(
            after.masks.to_dense(WIDTH, HEIGHT), before.masks.to_dense(WIDTH, HEIGHT)
        )
    pd.testing.assert_frame_equal(kept.to_dataframe(), dropped.to_dataframe())
    pd.testing.assert_frame_equal(kept.to_freeze_frames(), dropped.to_freeze_frames())


# ----------------------------------------------------------------- real models
@pytest.mark.model
def test_real_video_pipeline(tmp_path, detector_weights, pitch_weights):
    video = ROOT / "data" / "videos" / "broadcast_60s.mp4"
    if not video.is_file():
        pytest.skip("real video not available")
    if "yolo" not in tv.models.available_models():
        pytest.skip("yolo backend not available")
    pipeline = Pipeline(
        tv.load_model("yolo", detector_weights),
        keypoint_model=tv.load_model("yolo_pose", pitch_weights, resize="square"),
        team_classifier=TeamClassifier("siglip"),
    )
    result = pipeline.run(video, end=50, progress=False)
    assert len(result) == 50
    table = result.to_dataframe()
    players = table[(table.object == "person") & (table.class_name == "player")]
    assert players.track_id.nunique() >= 15
    # both teams are present and most players have one
    assert set(players.team_id) >= {0, 1}
    assert (players.team_id != NO_TEAM).mean() > 0.9
    # the homography puts players on the pitch
    pitch = result.pitch
    on_pitch = players.pitch_x.between(-5, pitch.length + 5) & players.pitch_y.between(
        -5, pitch.width + 5
    )
    assert on_pitch.mean() > 0.9
    # each track keeps one team
    assert (players.groupby("track_id").team_id.nunique() == 1).all()
    assert result.save(tmp_path / "result.pkl").is_file()


def test_team_samples_keep_a_bounded_uniform_sample_per_track():
    from tactifoot_vision.pipeline.pipeline import _TeamSamples

    samples = _TeamSamples(per_key=3)
    accepted = 0
    for i in range(200):
        if samples.accepts(7):
            samples.add(7, np.full(4, i, dtype=np.float32))
            accepted += 1
    assert samples.keys == [7, 7, 7]
    assert samples.embeddings().shape == (3, 4)
    assert 3 < accepted < 200  # later crops still get a chance to enter
    assert samples.embeddings().max() > 2  # not just the first three crops


def test_a_frame_without_a_valid_crop_uses_up_no_sample(monkeypatch):
    from tactifoot_vision.pipeline import pipeline as module

    pipeline = _pipeline()
    samples = module._TeamSamples(per_key=2)
    people = sv.Detections(
        xyxy=np.float32([[0, 0, 10, 20]]),
        tracker_id=np.array([5]),
        data={"class_name": np.array(["player"])},
    )
    monkeypatch.setattr(module, "extract_crops", lambda frame, boxes, **_: [None])
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    pipeline._sample_crops(frame, people, np.array([5]), 1, samples)
    # The track still counts as new, so its next frame is sampled.
    assert 5 not in samples.seen and samples.offered[5] == 0
    assert not samples._pending


def _frame_with(keys, names, xy) -> "tv.pipeline.FrameResult":
    detections = sv.Detections(
        xyxy=np.zeros((len(keys), 4), np.float32),
        tracker_id=np.array(keys),
        data={"class_name": np.array(names), "pitch_xy": np.array(xy, float)},
    )
    return tv.pipeline.FrameResult(
        0, 0.0, detections, sv.Detections.empty(), ball_candidates=sv.Detections.empty()
    )


def test_goalkeeper_without_pitch_positions_keeps_its_crop_vote():
    from collections import Counter

    from tactifoot_vision.pipeline.pipeline import _goalkeeper_teams

    keys = [1, 2, 9, 10]
    frame = _frame_with(
        keys, ["player", "player", "goalkeeper", "goalkeeper"], [[np.nan, np.nan]] * 4
    )
    votes = {9: Counter({1: 3, 0: 1})}  # goalkeeper 10 has no crops
    teams = _goalkeeper_teams([9, 10], [frame], [np.array(keys)], {1: 0, 2: 1}, votes)
    assert teams == {9: 1}


def test_prefitted_classifier_with_no_team_crops(video):
    classifier = TeamClassifier(MeanColorEmbedder(), reducer=None)
    classifier.fit(
        [np.full((10, 6, 3), c, np.uint8) for c in [(230, 0, 0), (0, 0, 230)] * 3]
    )
    result = _pipeline(
        team_classifier=classifier, refit_teams=False, include_classes=["referee"]
    ).run(video, end=3, progress=False)
    assert all(set(f.detections.data["class_name"]) == {"referee"} for f in result)
    assert all((f.team_ids == NO_TEAM).all() for f in result)


# ------------------------------------------------------- review round 2
class _ResetSpy(ByteTrackTracker):
    resets = 0

    def reset(self, fps=None):
        type(self).resets += 1
        super().reset(fps=fps)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start": -2}, "start must be >= 0"),
        ({"start": 4, "end": 2}, "must be >= start"),
        ({"start": 3, "end": 3}, "empty"),
        ({"end": 0}, "empty"),
        ({"stride": 0}, "stride must be >= 1"),
        ({"start": FRAMES}, "past the end"),
    ],
)
def test_run_checks_the_frame_range_before_any_work(video, kwargs, match):
    detector = ColorDetector(device="cpu")
    detector.predict = lambda image: pytest.fail("the detector ran")
    pipeline = Pipeline(detector, tracker=_ResetSpy())
    _ResetSpy.resets = 0
    with pytest.raises(ValueError, match=match):
        pipeline.run(video, progress=False, **kwargs)
    assert _ResetSpy.resets == 0


def test_run_carries_the_ball_class_into_the_freeze_frames(video):
    class SportsBall(ColorDetector):
        def predict(self, image):
            detections = super().predict(image)
            names = detections.data["class_name"]
            detections.data["class_name"] = np.where(
                names == "ball", "sports ball", names
            )
            return detections

    result = Pipeline(SportsBall(device="cpu"), ball_class="sports ball").run(
        video, end=1, progress=False
    )
    frames = result.to_freeze_frames()
    assert frames.loc[frames["type"] == "ball", "class_name"].tolist() == [
        "sports ball"
    ]


# ------------------------------------------------------- review round 3 (Fable)
def _player_teams(frame, t: int = 0) -> tuple[set[int], set[int]]:
    """Team ids of the players at RED_PLAYERS' and at BLUE_PLAYERS' places."""
    teams = frame.team_ids
    return (
        {int(teams[_track_of(frame, x + t, y)]) for x, y in RED_PLAYERS},
        {int(teams[_track_of(frame, x + t, y)]) for x, y in BLUE_PLAYERS},
    )


def test_each_run_fits_the_teams_again_unless_asked_to_keep_the_fit(video, tmp_path):
    # Another match: both teams wear red, one dark and one bright. A fit on the
    # first clip (red against blue) puts every one of them in the red team.
    reds = _write_video(tmp_path / "reds.avi", kits=((0, 0, 140), (0, 0, 230)))

    pipeline = _pipeline()
    pipeline.run(video, progress=False)
    first_fit = pipeline.team_classifier._kmeans
    dark, bright = _player_teams(pipeline.run(reds, progress=False)[0])
    assert pipeline.team_classifier._kmeans is not first_fit
    assert len(dark) == len(bright) == 1 and dark != bright

    keep = _pipeline(refit_teams=False)
    keep.run(video, progress=False)
    first_fit = keep.team_classifier._kmeans
    dark, bright = _player_teams(keep.run(reds, progress=False)[0])
    assert keep.team_classifier._kmeans is first_fit
    assert dark == bright == {0}  # the red team of the first clip


def test_goalkeeper_team_follows_position_when_the_kit_says_otherwise(tmp_path):
    # The yellow kit is nearer the red kit (team 0), but this goalkeeper stands
    # beside the blue team (team 1).
    clip = _write_video(tmp_path / "keeper.avi", goalkeeper=(270, 110))
    result = _pipeline().run(clip, progress=False)
    for t, frame in enumerate(result):
        assert _player_teams(frame, t) == ({0}, {1})
        names = frame.detections.data["class_name"]
        assert frame.team_ids[names == "goalkeeper"].tolist() == [1]


@pytest.mark.parametrize("stride", [1, 3])
def test_track_ball_drops_the_teleport_and_fills_it_at_any_stride(video, stride):
    # The ball moves ~17 m/s and the teleport jumps ~44 m. The gate widens
    # with the time since the last ball, not per processed frame, so at
    # stride 3 the ~2 m between processed frames still pass it.
    result = track_ball(_pipeline().run(video, stride=stride, progress=False))
    assert [f.index for f in result] == list(range(0, FRAMES, stride))
    for frame in result:
        assert len(frame.ball) == 1, frame.index
        assert frame.ball.data["interpolated"].tolist() == [
            frame.index == TELEPORT_FRAME
        ]
        bx = 150 + 2 * frame.index + 3  # the box centre without the teleport
        assert frame.ball.get_anchors_coordinates(sv.Position.CENTER)[0, 0] == (
            pytest.approx(bx, abs=2)
        )


def _labelled_clip(video: Path) -> LabelledClip:
    """The synthetic clip's truth: every object of every frame, one id each."""
    frames = []
    for t in range(FRAMES):
        objects = [(i + 1, COLORS[c], box) for i, (c, box) in enumerate(_scene(t))]
        people = [o for o in objects if o[1] != "ball"]
        ball = [o for o in objects if o[1] == "ball"]
        frames.append(
            LabelledFrame(
                t,
                *(
                    sv.Detections(
                        xyxy=np.array([b for _, _, b in rows], float).reshape(-1, 4),
                        tracker_id=np.array([i for i, _, _ in rows], int),
                        data={"class_name": np.array([n for _, n, _ in rows])},
                    )
                    for rows in (people, ball)
                ),
            )
        )
    return LabelledClip(
        "synthetic", video, FPS, WIDTH, HEIGHT, tuple(frames),
        frozenset({Truth.IDENTITIES, Truth.BOXES, Truth.BALL}),
    )  # fmt: skip


def test_a_strided_run_is_scored_on_its_processed_frames_only(video):
    result = _pipeline().run(video, stride=2, progress=False)
    metrics = evaluate_tracking(result, _labelled_clip(video), hota=False)
    # The odd frames hold truth but were skipped: no misses there.
    assert metrics.num_frames == FRAMES // 2
    assert metrics.per_class.loc["player", "instances"] == 6 * FRAMES // 2
    assert (metrics.false_negatives, metrics.false_positives) == (0, 0)
    assert (metrics.mota, metrics.idf1, metrics.id_switches) == (1.0, 1.0, 0)


def test_too_few_player_crops_leave_an_unfitted_classifier_alone(video, caplog):
    classifier = TeamClassifier(MeanColorEmbedder(), n_teams=7, reducer=None)
    with caplog.at_level("WARNING", logger="tactifoot_vision"):
        result = _pipeline(team_classifier=classifier).run(video, end=1, progress=False)
    assert "Only 6 player crops" in caplog.text
    assert not classifier.is_fitted
    assert (result[0].team_ids == NO_TEAM).all()


class _ThreeBalls(ColorDetector):
    """Adds two decoy balls around the real one: confidences 0.5, 0.95 (real), 0.7."""

    def predict(self, image):
        detections = super().predict(image)
        is_ball = detections.data["class_name"] == "ball"
        ball = detections[is_ball]
        decoys = sv.Detections(
            xyxy=np.float32([[10, 10, 16, 16], [300, 10, 306, 16]]),
            confidence=np.float32([0.5, 0.7]),
            class_id=np.array([0, 0]),
            data={"class_name": np.array(["ball", "ball"])},
        )
        ball.confidence = np.full(len(ball), 0.95, np.float32)
        return sv.Detections.merge([detections[~is_ball], decoys[:1], ball, decoys[1:]])


def test_the_most_confident_ball_is_kept(video):
    result = Pipeline(_ThreeBalls(device="cpu"), tracker=None).run(
        video, end=3, progress=False
    )
    for t, frame in enumerate(result):
        assert len(frame.ball) == 1
        assert frame.ball.confidence.tolist() == [pytest.approx(0.95)]
        assert frame.ball.xyxy[0, 0] == pytest.approx(150 + 2 * t, abs=2)


def test_every_ball_candidate_is_kept_best_first(video):
    result = Pipeline(
        _ThreeBalls(device="cpu"),
        keypoint_model=KnownHomographyKeypoints(device="cpu"),
        tracker=None,
    ).run(video, end=3, progress=False)
    for t, frame in enumerate(result):
        candidates = frame.ball_candidates
        assert candidates.confidence.tolist() == pytest.approx([0.95, 0.7, 0.5])
        assert candidates.xyxy[1:, 0].tolist() == [300, 10]
        assert candidates.xyxy[0, 0] == pytest.approx(150 + 2 * t, abs=2)
        np.testing.assert_array_equal(frame.ball.xyxy, candidates.xyxy[:1])
        centres = candidates.get_anchors_coordinates(sv.Position.CENTER)
        np.testing.assert_allclose(
            candidates.data["pitch_xy"],
            frame_to_pitch(centres, FRAME_TO_PITCH),
            atol=1e-3,
        )


def test_track_ball_leaves_the_candidates_alone(result):
    frame = track_ball(result).frame(TELEPORT_FRAME)
    assert frame.ball.data["interpolated"].tolist() == [True]
    assert frame.ball_candidates.xyxy[0, 0] == pytest.approx(290, abs=2)
    assert np.isfinite(frame.ball_candidates.data["pitch_xy"]).all()


def test_the_ball_shares_no_memory_with_its_candidates(video):
    result = Pipeline(_ThreeBalls(device="cpu"), tracker=None).run(
        video, end=1, progress=False
    )
    frame = result[0]
    frame.ball.xyxy[:] = 0
    frame.ball.data["pitch_xy"][:] = 0
    assert frame.ball_candidates.xyxy[0, 0] == pytest.approx(150, abs=2)
    assert np.isnan(frame.ball_candidates.data["pitch_xy"][0]).all()


# ------------------------------------------------------------- enums (ADR 0004)
def test_saved_files_hold_plain_strings_whatever_the_enum_members(video, tmp_path):
    detector = ColorDetector(device="cpu")
    result = Pipeline(detector, tracker=TrackerType.BYTETRACK).run(
        video, end=3, progress=False
    )
    by_name = Pipeline(detector, tracker="bytetrack").run(video, end=3, progress=False)
    pd.testing.assert_frame_equal(result.to_dataframe(), by_name.to_dataframe())

    path = result.save(tmp_path / "result.pkl")
    assert b"tactifoot_vision.enums" not in path.read_bytes()
    for frame in PipelineResult.load(path):
        assert all(type(key) is str for key in frame.detections.data)
        assert frame.detections.data["class_name"].dtype.kind == "U"
    tracks, freeze_frames = result.to_dataframe(), result.to_freeze_frames()
    assert {type(v) for v in tracks["object"]} == {str}
    assert {type(v) for v in freeze_frames["type"]} == {str}
    assert set(freeze_frames["type"]) == {"player", "goalkeeper", "referee", "ball"}
    parquet = result.to_parquet(tmp_path / "tracks.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), tracks)


# ------------------------------------------------------------ single frames
def _frame(t: int = 0, **scene) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (40, 140, 40), dtype=np.uint8)
    for color, (x1, y1, x2, y2) in _scene(t, **scene):
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, thickness=-1)
    return frame


def _analyzer(detector: Model | None = None, **kwargs) -> FrameAnalyzer:
    options = {
        "keypoint_model": KnownHomographyKeypoints(device="cpu"),
        "team_classifier": TeamClassifier(MeanColorEmbedder(), reducer=None),
    } | kwargs
    return FrameAnalyzer(detector or ColorDetector(device="cpu"), **options)


def test_frame_analyzer_places_and_splits_one_frame_on_its_own():
    analyzer = _analyzer()
    frame = analyzer(_frame(3), index=7, timestamp=1.5)
    assert (frame.index, frame.timestamp) == (7, 1.5)
    assert frame.detections.tracker_id is None and len(frame.detections) == 8
    np.testing.assert_allclose(
        frame.homography, FRAME_TO_PITCH / FRAME_TO_PITCH[2, 2], atol=1e-5
    )
    anchors = frame.detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    np.testing.assert_allclose(
        frame.pitch_xy, frame_to_pitch(anchors, FRAME_TO_PITCH), atol=1e-3
    )
    assert len(frame.ball) == 1 and np.isfinite(frame.ball_xy).all()
    teams = frame.team_ids
    red = {teams[_track_of(frame, x + 3, y)] for x, y in RED_PLAYERS}
    blue = {teams[_track_of(frame, x + 3, y)] for x, y in BLUE_PLAYERS}
    assert len(red) == len(blue) == 1 and red != blue and red | blue == {0, 1}
    names = frame.detections.data["class_name"]
    assert teams[names == "referee"].tolist() == [NO_TEAM]
    assert teams[names == "goalkeeper"].tolist() == list(red)  # beside the red team
    assert analyzer.homography.matrix is None  # no state kept between frames


def test_frame_analyzer_keeps_every_ball_candidate_best_first():
    frame = _analyzer(detector=_ThreeBalls(device="cpu"))(_frame())
    candidates = frame.ball_candidates
    assert candidates.confidence.tolist() == pytest.approx([0.95, 0.7, 0.5])
    np.testing.assert_array_equal(frame.ball.xyxy, candidates.xyxy[:1])
    assert frame.ball.xyxy is not candidates.xyxy  # a copy, as in the pipeline
    centres = candidates.get_anchors_coordinates(sv.Position.CENTER)
    np.testing.assert_allclose(
        candidates.data["pitch_xy"], frame_to_pitch(centres, FRAME_TO_PITCH), atol=1e-3
    )


def test_frame_analyzer_without_keypoints_or_teams():
    frame = _analyzer(keypoint_model=None, team_classifier=None)(_frame())
    assert frame.homography is None and frame.keypoints is None
    assert np.isnan(frame.pitch_xy).all() and np.isnan(frame.ball_xy).all()
    assert (frame.team_ids == NO_TEAM).all()
    # Teams without a homography: the goalkeeper is placed in frame pixels.
    frame = _analyzer(keypoint_model=None)(_frame())
    names = frame.detections.data["class_name"]
    red = frame.team_ids[_track_of(frame, *RED_PLAYERS[0])]
    assert frame.team_ids[names == "goalkeeper"].tolist() == [red]


def test_frame_analyzer_validation():
    with pytest.raises(ValueError, match="reducer=None"):
        _analyzer(team_classifier=TeamClassifier(MeanColorEmbedder()))
    with pytest.raises(ValueError, match="detector must be a detection model"):
        FrameAnalyzer(KnownHomographyKeypoints(device="cpu"))
    with pytest.raises(ValueError, match="keypoint_model must be a pose model"):
        _analyzer(keypoint_model=ColorDetector(device="cpu"))


def _with_ball(frame, xy):
    ball = sv.Detections(
        xyxy=np.float32([[1, 1, 3, 3]]),
        confidence=np.float32([0.5]),
        data={"class_name": np.array(["ball"]), "pitch_xy": np.float32([xy])},
    )
    return dataclasses.replace(frame, ball=ball)


def test_fill_ball_takes_the_first_earlier_ball_on_the_pitch():
    t0 = _analyzer()(_frame())
    without = dataclasses.replace(t0, ball=sv.Detections.empty())
    off_pitch = _with_ball(t0, [np.nan, np.nan])
    earlier = [off_pitch, _with_ball(t0, [30.0, 20.0]), _with_ball(t0, [1.0, 1.0])]
    consumed = []

    def lazily():
        for frame in earlier:
            consumed.append(frame)
            yield frame

    filled = fill_ball(without, lazily())
    assert filled.ball_xy.tolist() == [[30.0, 20.0]]
    assert filled.ball.data["fallback"].tolist() == [2]
    assert len(consumed) == 2  # the third frame was never needed
    assert len(without.ball) == 0 and "fallback" not in earlier[1].ball.data
    assert filled.detections is without.detections  # people stay those of the frame

    own = fill_ball(t0, earlier)
    np.testing.assert_array_equal(own.ball_xy, t0.ball_xy)
    assert own.ball.data["fallback"].tolist() == [0] and "fallback" not in t0.ball.data
    # An off-pitch ball of the frame itself counts as missing; no ball anywhere keeps it.
    assert fill_ball(off_pitch, earlier[1:]).ball.data["fallback"].tolist() == [1]
    kept = fill_ball(off_pitch, [without])
    assert np.isnan(kept.ball_xy).all() and kept.ball.data["fallback"].tolist() == [0]
    assert len(fill_ball(without, []).ball) == 0


class _SeeingEstimator(tv.pitch.HomographyEstimator):
    """Records the frame and the boxes the pipeline hands to the homography."""

    def __init__(self) -> None:
        super().__init__(SoccerPitch())
        self.seen: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def fit(self, keypoints, frame=None, boxes=None):
        self.seen.append((frame, boxes))
        return super().fit(keypoints, frame, boxes)

    def update(self, keypoints, frame=None, boxes=None):
        self.seen.append((frame, boxes))
        return super().update(keypoints, frame, boxes)


def test_the_homography_gets_the_frame_and_the_people_boxes(video):
    # Line refinement (refine_with_lines) needs the frame, and the boxes keep
    # kits out of its line mask.
    estimator = _SeeingEstimator()
    image = _frame(3)
    frame = _analyzer(homography=estimator)(image)
    seen_frame, seen_boxes = estimator.seen[-1]
    assert seen_frame is image
    np.testing.assert_array_equal(seen_boxes, frame.detections.xyxy)

    estimator = _SeeingEstimator()
    result = Pipeline(
        ColorDetector(device="cpu"),
        keypoint_model=KnownHomographyKeypoints(device="cpu"),
        tracker=None,
        homography=estimator,
    ).run(video, end=2, progress=False)
    assert len(estimator.seen) == 2
    for (seen_frame, seen_boxes), frame in zip(estimator.seen, result, strict=True):
        assert seen_frame.shape == (HEIGHT, WIDTH, 3)
        np.testing.assert_array_equal(seen_boxes, frame.detections.xyxy)
