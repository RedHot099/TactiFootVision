"""ball-detection 05: the views of the ball fine-tune (scripts/ball_view.py): bins, oversampling, neighbour
thinning, constant epoch length and the exclusions, on synthetic pools."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_view as bv  # noqa: E402
import detector_train as dt  # noqa: E402

ALL = "ball,goalkeeper,player,referee"


def row(
    set_name,
    image,
    *,
    source=None,
    split="train",
    game=None,
    clip=None,
    frame=None,
    gsr_split=None,
    annotated=ALL,
):
    return {
        "set": set_name,
        "source": source or set_name,
        "split": split,
        "image": image,
        "width": 1920,
        "height": 1080,
        "game": game,
        "clip": clip,
        "frame": frame,
        "gsr_split": gsr_split,
        "near_dup_set": None,
        "annotated_classes": annotated,
        **{f"n_{c}": 1 for c in dt.CLASSES},
    }


def pool(frames: int = 20) -> pd.DataFrame:
    """Run 2c's sets, two GSR train clips of ``frames`` frames, the benchmark games and the Ekstraklasa dev clips."""
    rows = [
        row(s, f"/{s}/a.jpg")
        for s in dt.RUN1_SETS
        if s not in ("soccernet_gsr", "sportsmot_football")
    ]
    for source in ("issia", "teamtrack", "soccer_player_detection"):
        rows = [r for r in rows if r["set"] != source]
        rows.append(row(source, f"/{source}/a.jpg", annotated="ball,player"))
    rows.append(
        row("sportsmot_football", "/sm/a.jpg", source="sportsmot", annotated="player")
    )
    for clip in ("SNGS-060", "SNGS-061"):
        for f in range(1, frames + 1):
            rows.append(
                row(
                    "soccernet_gsr",
                    f"/gsr/{clip}_{f:06d}.jpg",
                    game="gsr:4",
                    clip=clip,
                    frame=f,
                    gsr_split="train",
                )
            )
    rows += [
        row(
            "soccernet_gsr",
            "/gsr/valid.jpg",
            game="gsr:2",
            clip="SNGS-021",
            frame=1,
            gsr_split="valid",
            split="valid",
        ),
        row(
            "soccernet_gsr",
            "/gsr/test.jpg",
            game="gsr:7",
            clip="SNGS-116",
            frame=1,
            gsr_split="test",
            split="test",
        ),
        row("soccernet_v3", "/v3/dup.jpg"),
        row("football_yolo", "/fy/train.jpg"),
        row("football_yolo", "/fy/dev.jpg", split="test"),
        row("soccersynth", "/synth/a.jpg", annotated="ball,player"),
    ]
    frame = pd.DataFrame(rows)
    frame.loc[frame.image == "/v3/dup.jpg", "near_dup_set"] = "gsr_valid"
    for column in ("game", "clip", "gsr_split", "near_dup_set"):
        frame[column] = frame[column].astype("string")
    frame["frame"] = frame["frame"].astype("Int64")
    return frame


def view(frames: int = 20) -> pd.DataFrame:
    return (
        dt.build_view(pool(frames), "bd05_B")
        .sort_values("image")
        .reset_index(drop=True)
    )


def bins_for(
    v: pd.DataFrame, small: set[str] = (), blurred: set[tuple[str, int]] = ()
) -> pd.DataFrame:
    diameter = pd.Series(np.where(v.image.isin(small), 8.0, 20.0), index=v.index)
    gsr = v[bv.is_gsr_train(v)]
    blur = pd.DataFrame(
        {
            "clip": gsr["clip"].astype(str),
            "frame": gsr["frame"].astype(int),
            "blur": [
                (c, int(f)) in blurred
                for c, f in zip(gsr["clip"], gsr["frame"], strict=True)
            ],
            "blur_known": True,
        }
    )
    return bv.ball_bins(v, diameter, blur)


def test_both_arms_drop_the_benchmark_and_the_ekstraklasa_dev_clips():
    for run in bv.ARMS:
        images = set(dt.build_view(pool(), run).image)
        assert not images & {
            "/gsr/valid.jpg",
            "/gsr/test.jpg",
            "/v3/dup.jpg",
            "/fy/dev.jpg",
        }
        assert "/fy/train.jpg" in images and "/gsr/SNGS-060_000001.jpg" in images
    assert dt.RUNS["bd05_A"]["oversample"] == dt.RUNS["run2c"]["oversample"]
    for run in bv.ARMS:
        assert dt.RUNS[run]["settings"] == dt.RUNS["run2c"]["settings"] | {"epochs": 2}
        assert dt.RUNS[run]["init"].name == "CANDIDATE_epoch1.pt"


def test_detector_train_refuses_to_build_the_bd05_views():
    from types import SimpleNamespace

    with pytest.raises(SystemExit, match="ball_view"):
        dt.view(SimpleNamespace(run="bd05_B"))


def test_bins_are_small_or_blurred_and_blur_comes_only_from_gsr_train():
    v = view()
    b = bins_for(v, small={"/issia/a.jpg"}, blurred={("SNGS-060", 3)})
    flagged = set(b.image[b.in_bin])
    assert flagged == {"/issia/a.jpg", "/gsr/SNGS-060_000003.jpg"}
    assert not b.blur[~bv.is_gsr_train(v)].any()
    assert b.small.sum() == 1 and b.blur.sum() == 1


def test_diameters_are_scaled_to_the_1080p_equivalent_of_the_network_input(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.01 0.02\n2 0.5 0.5 0.1 0.2\n")
    # 3840 x 2160: box 38.4 x 43.2 px, mean 40.8, halved to the 1920-wide scale
    assert bv.ball_diameters(label, 3840, 2160) == pytest.approx([20.4])
    # 6500 x 1000 panorama: the long side sets the scale
    assert bv.ball_diameters(label, 6500, 1000) == pytest.approx(
        [(65 + 20) / 2 * 1920 / 6500]
    )
    (tmp_path / "empty.txt").write_text("")
    assert len(bv.ball_diameters(tmp_path / "empty.txt", 1920, 1080)) == 0
    assert bv.label_path("/x/images/a.jpg") == Path("/x/labels/a.txt")


def test_frame_blur_uses_the_image_speed_in_ball_diameters_per_frame():
    balls = pd.DataFrame(
        {
            "clip": "c",
            "track_id": 1,
            "frame": [1, 2, 3, 4],
            "cx": [0.0, 5.0, 10.0, 30.0],
            "cy": 0.0,
            "box": [
                (x - 5, -5, x + 5, 5) for x in (0.0, 5.0, 10.0, 30.0)
            ],  # diameter 10
            "pitch_x": 0.0,
            "pitch_y": 0.0,
        }
    )
    frames = bv.frame_blur(balls).set_index("frame")
    # speeds 5, 5, 12.5, 20 px/frame -> 0.5, 0.5, 1.25, 2 diameters per frame
    assert frames.blur.tolist() == [False, False, True, True]
    assert frames.blur_known.all()


def test_arm_b_oversamples_the_bins_and_keeps_a_s_epoch_length():
    v = view(frames=40)
    small = {"/issia/a.jpg", "/fy/train.jpg", "/gsr/SNGS-060_000004.jpg"}
    b = bins_for(v, small=small, blurred={("SNGS-061", 7)})
    oversample = dt.RUNS["bd05_B"]["oversample"]
    a_times = bv.arm_times(v, b.in_bin.to_numpy(), oversample, 1)
    b_times = bv.arm_times(v, b.in_bin.to_numpy(), oversample, 2)
    assert a_times.sum() == b_times.sum()
    assert len(bv.listing(v, a_times)) == len(bv.listing(v, b_times))
    by_image = dict(zip(v.image, b_times, strict=True))
    assert by_image["/fy/train.jpg"] == 40  # football_yolo's x20, doubled
    assert by_image["/issia/a.jpg"] == 2
    # 21 extra non-GSR lines (fy 20, issia 1) + 2 GSR in-bin frames: thinning removes exactly 23 GSR lines
    gsr = bv.is_gsr_train(v).to_numpy()
    assert a_times[gsr].sum() + 2 - b_times[gsr].sum() == 23
    assert not np.isin(v.image[b_times == 0], ["/synth/a.jpg"]).any()
    # every non-GSR image keeps its listing times the factor of its bin
    others = ~gsr
    expected = np.where(b.in_bin.to_numpy(), a_times * 2, a_times)
    assert np.array_equal(b_times[others], expected[others])
    assert dict(zip(v.image, a_times, strict=True))["/fy/train.jpg"] == 20


def test_thinning_drops_only_odd_gsr_train_frames_and_keeps_their_neighbours():
    v = view(frames=40)
    times = bv.base_times(v, {})
    thinned = bv.thin(v, times, 15)
    dropped = v[thinned == 0]
    assert len(dropped) == 15 and times.sum() - thinned.sum() == 15
    assert bv.is_gsr_train(dropped).all() and (dropped.frame % 2 == 1).all()
    listed = v[(thinned > 0) & bv.is_gsr_train(v).to_numpy()]
    kept = {(c, int(f)) for c, f in zip(listed["clip"], listed.frame, strict=True)}
    for clip, f in zip(dropped["clip"], dropped.frame, strict=True):
        assert (clip, int(f) - 1) in kept or (clip, int(f) + 1) in kept
    assert np.array_equal(bv.thin(v, times, 15), thinned)  # seeded
    assert not np.array_equal(bv.thin(v, times, 15, seed=1), thinned)


def test_thinning_hits_the_excess_exactly_with_doubled_frames_and_refuses_too_much():
    v = view(frames=10)
    times = bv.base_times(v, {})
    doubled = np.where(bv.is_gsr_train(v), 2, times)
    # an odd excess cannot be met by doubled frames alone: every odd frame is listed twice
    with pytest.raises(ValueError, match="removes at most"):
        bv.thin(v, doubled, 3)
    thinned = bv.thin(v, doubled, 4)
    assert doubled.sum() - thinned.sum() == 4
    # 2 clips x 5 odd frames x 2 lines = 20 removable lines at most
    with pytest.raises(ValueError, match="at most 20"):
        bv.thin(v, doubled, 22)


def test_bin_counts_report_images_lines_and_epoch_shares_per_source():
    v = view()
    b = bins_for(v, small={"/issia/a.jpg"}, blurred={("SNGS-060", 3)})
    times = bv.arm_times(v, b.in_bin.to_numpy(), dt.RUNS["bd05_B"]["oversample"], 2)
    counts = bv.bin_counts(b, times)
    assert counts["epoch_lines"] == times.sum()
    assert counts["per_source"]["issia"]["small"] == {
        "images": 1,
        "lines": 2,
        "epoch_share": round(2 / times.sum(), 4),
    }
    total = counts["all"]
    assert total["small_or_blur"]["lines"] + total["none"]["lines"] == times.sum()
    assert counts["gsr_train"]["frames_dropped"] == int(
        (times[bv.is_gsr_train(v)] == 0).sum()
    )
