"""Attributes Xa, roles and mask regions of xg-dino-info (ticket 09): planted frames replayed end to end, the own-frame
ball kept apart from the fallback ball, the motion validity rules, the SoccerNet flags' invariance to Goal records and
to the shot's label type, the label-read guard, and the patch-mask regions and placebos."""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgd_fixtures import frame_with
from xgpoc.detector import extract as ex
from xgpoc.detector import location as loc
from xgpoc.detector.guard import StatsBombRead
from xgpoc.dino import attributes as at
from xgpoc.dino import regions as rg
from xgpoc.dino.records import import_closure

ROOT = Path(__file__).resolve().parents[1]
SCALE = 10.0                                    # planted frames: image px = 10 x pitch metres


def scaled(frame, scale: float = SCALE):
    """A fixture frame with image boxes at `scale` px per metre and the matching image -> pitch homography."""
    people = frame.detections
    people.xyxy = (people.xyxy * scale).astype(np.float32)
    ball = frame.ball
    ball.xyxy = np.float32([[x * scale - 3, y * scale - 3, x * scale + 3, y * scale + 3] for x, y in ball.data["pitch_xy"]]).reshape(-1, 4)
    return dataclasses.replace(frame, detections=people, ball=ball, ball_candidates=ball,
                               homography=np.diag([1 / scale, 1 / scale, 1.0]))


SHOOTER, MATE, DEFENDER, KEEPER = (95.0, 40.0, 0, "player"), (90.0, 30.0, 0, "player"), (97.0, 38.0, 1, "player"), (104.0, 34.0, 1, "goalkeeper")


def cache(frames_by_shot: dict[str, dict[int, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(frames.parquet rows, features.parquet rows under feet_t0) of planted frames {shot: {offset: FrameResult}}."""
    rows = [ex.frame_row(shot, offset, f"{shot}/{offset}.jpg", frame)
            for shot, by_offset in frames_by_shot.items() for offset, frame in by_offset.items()]
    frames = pd.DataFrame(rows)
    return frames, loc.feature_rows(frames, loc.FEET_T0)


def build(frames_by_shot, cuts=None):
    frames, features = cache(frames_by_shot)
    shots = pd.DataFrame({"game": "g", "half": 1, "position_ms": 1000}, index=pd.Index(sorted(frames_by_shot), name="shot_id"))
    cuts = pd.Series(cuts if cuts is not None else dict.fromkeys(frames_by_shot, False))
    return at.build(frames, features, shots, {"g": []}, cuts)


def planted(ball0=(95.2, 40.0), ball4=(94.0, 40.0), shooter4=(94.5, 40.0), fallback0=None):
    f0 = scaled(frame_with([SHOOTER, MATE, DEFENDER, KEEPER], ball0, fallback=fallback0))
    f4 = scaled(frame_with([(*shooter4, 0, "player"), MATE, DEFENDER, KEEPER], ball4))
    return {0: f0, -400: f4}


# ------------------------------------------------------------------ attributes
def test_planted_shot_end_to_end():
    table, roles, t0 = build({"s": planted()})
    r = table.loc["s"]
    assert at.replay_match(roles) == {"-400": {"rows": 1, "readable": 1, "reproduced": 1, "share": 1.0},
                                      "0": {"rows": 1, "readable": 1, "reproduced": 1, "share": 1.0}}
    assert r["roles_ok@0"] == 1 and t0["s"].shooter == 0 and t0["s"].gk_detected
    assert r["box_players@0"] == 4                                  # every person is inside the box
    assert r["near5_players@0"] == 1                                # defender 2.8 m, mate 11.2 m, keeper 10.8 m
    assert r["box_attackers@0"] == 1                                # the mate, shooter excluded
    assert r["px_per_m@0"] == pytest.approx(SCALE)
    assert r["ball_own@0"] == 1 and r["ball_above@0"] == 0          # the ball sits at the shooter's feet
    assert r["ball_height_rel@0"] == pytest.approx(0.0)
    assert r["shooter_h_px@0"] == pytest.approx(20) and r["shooter_aspect@0"] == pytest.approx(2)
    assert r["motion_status"] == "ok" and r["motion_shot_disp_m"] == pytest.approx(0.5)
    assert r["motion_gk_disp_m"] == 0 and r["motion_box_change"] == 0
    assert r["ball_height_rel@-400"] == pytest.approx(0.0)          # own ball at both offsets


def test_a_fallback_ball_is_not_an_own_ball():
    shot = planted()
    shot[0] = scaled(frame_with([SHOOTER, MATE, DEFENDER, KEEPER], None))     # t0 takes the -400 ms ball
    table, roles, _ = build({"s": shot})
    r = table.loc["s"]
    assert r["ball_ok@0"] == 1 and r["ball_fallback@0"] == 1        # the fallback-filled cache flags
    assert r["ball_own@0"] == 0
    assert np.isnan(r["ball_above@0"]) and np.isnan(r["ball_dist_rel@0"])
    assert r["ball_own@-400"] == 1 and np.isnan(r["ball_above@-400"])   # -400 ms ball rows need an own ball at both
    assert not roles.set_index("offset_ms").loc[0, "ball_own"]


def test_a_row_the_replay_does_not_reproduce_keeps_its_roles_missing():
    frames, features = cache({"s": planted()})
    features.loc[features.offset_ms == 0, "shot_x"] += 1.0
    shots = pd.DataFrame({"game": "g", "half": 1, "position_ms": 1000}, index=pd.Index(["s"], name="shot_id"))
    table, roles, t0 = at.build(frames, features, shots, {"g": []}, pd.Series({"s": False}))
    r = table.loc["s"]
    assert r["roles_ok@0"] == 0 and t0["s"].shooter is None
    assert all(np.isnan(r[at.col(c, 0)]) for c in [*at.CROWDING, *at.SHOOTER, "ball_above", "px_per_m"])
    assert r["n_players@0"] == 3 and r["motion_status"] == "roles_missing"
    assert at.replay_match(roles)["0"]["reproduced"] == 0


@pytest.mark.parametrize(("cut", "status"), [(True, "camera_cut"), (None, "cut_unknown")])
def test_motion_needs_no_camera_cut(cut, status):
    table, _, _ = build({"s": planted()}, cuts={"s": cut})
    assert table.loc["s", "motion_status"] == status
    assert table.loc["s", at.MOTION].isna().all()


def test_motion_needs_the_same_shooter():
    # at -400 ms the ball is with the mate, so the rule's shooter there is another player
    table, _, _ = build({"s": planted(ball4=(90.0, 30.0))})
    assert table.loc["s", "motion_status"] == "shooter_differs"
    assert table.loc["s", at.MOTION].isna().all()


def test_motion_needs_the_same_attacked_goal_and_homographies():
    shot = planted()
    shot[-400] = dataclasses.replace(shot[-400], homography=None)
    assert build({"s": shot})[0].loc["s", "motion_status"] in ("no_homography", "roles_missing")
    far = {0: shot[0], -400: scaled(frame_with([(10.0, 40.0, 0, "player"), (12.0, 30.0, 1, "player")], (10.0, 40.0)))}
    assert build({"s": far})[0].loc["s", "motion_status"] == "direction_differs"


def test_mutual_nearest_neighbour_identity():
    def shot(shooter4):
        f0 = scaled(frame_with([SHOOTER, MATE, KEEPER], (95.2, 40.0)))
        f4 = scaled(frame_with([(*shooter4, 0, "player"), MATE, KEEPER], shooter4))
        return {0: f0, -400: f4}
    assert build({"s": shot((95.0, 44.0))})[0].loc["s", "motion_status"] == "ok"                 # 4.0 m: at the bound
    assert build({"s": shot((95.0, 44.5))})[0].loc["s", "motion_status"] == "shooter_differs"    # 4.5 m in 400 ms


# ------------------------------------------------------------------ flags
def annotation(label, position, team="home", half=1):
    return {"gameTime": f"{half} - 00:00", "position": str(position), "label": label, "team": team, "visibility": "visible"}


def test_flags_windows_and_the_le0_variant():
    raw = [annotation("Shots on target", 10000), annotation("Direct free-kick", 10200)]
    flags = at.shot_flags(at.si.parse_annotations(raw), 1, 10000)
    assert flags["sn_direct_fk"] == 1 and flags["sn_direct_fk_le0"] == 0
    raw = [annotation("Shots on target", 10000), annotation("Direct free-kick", 9000)]
    flags = at.shot_flags(at.si.parse_annotations(raw), 1, 10000)
    assert flags["sn_direct_fk"] == 1 and flags["sn_direct_fk_le0"] == 1
    assert "sn_prev_shot_5s" not in at.FLAGS


@pytest.mark.parametrize("label", ["Shots on target", "Shots off target"])
def test_flags_are_invariant_to_goal_records_and_the_shot_label_type(label):
    base = [annotation("Direct free-kick", 9800), annotation("Corner", 0), annotation("Indirect free-kick", 8000),
            annotation("Direct free-kick", 10250, team="away")]
    reference = at.shot_flags(at.si.parse_annotations([annotation("Shots on target", 10000), *base]), 1, 10000)
    for extra in ([], [annotation("Goal", 10000)], [annotation("Goal", 10100), annotation("Goal", 9900, team="away")]):
        raw = [annotation(label, 10000), *base, *extra]
        assert at.shot_flags(at.si.parse_annotations(raw), 1, 10000) == reference
    assert reference["sn_direct_fk"] == 1 and reference["sn_corner_20s"] == 1 and reference["sn_indirect_fk_3s"] == 1


def test_an_unknown_shooter_team_sets_every_flag_to_zero():
    raw = [annotation("Shots on target", 10000, team="not applicable"), annotation("Direct free-kick", 9800)]
    flags = at.shot_flags(at.si.parse_annotations(raw), 1, 10000)
    assert flags["sn_team_unknown"] == 1 and sum(flags[f] for f in at.FLAG_WINDOWS) == 0


# ------------------------------------------------------------------ guard
@pytest.mark.parametrize("name", ["captions/labels.parquet", "gpt_labels/x.parquet", "statsbomb/shots.parquet",
                                  "xgdino_info_Caption.parquet"])
def test_label_reads_are_refused(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [1]}).to_parquet(path)
    with at.no_label_reads(), pytest.raises(StatsBombRead):
        pd.read_parquet(path)
    with at.no_label_reads(), pytest.raises(StatsBombRead), open(path, "rb"):
        pass
    assert len(pd.read_parquet(path)) == 1                       # outside the block


def test_other_reads_pass_and_the_guard_is_restored(tmp_path):
    path = tmp_path / "attributes.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path)
    with at.no_label_reads():
        assert len(pd.read_parquet(path)) == 1
    assert "caption" not in at.guard.FORBIDDEN.pattern


def test_no_label_reader_is_imported():
    files = import_closure([ROOT / "xgpoc/dino/attributes.py", ROOT / "xgpoc/dino/regions.py",
                            ROOT / "scripts/xgdino_info_attributes.py"])
    names = {str(p.relative_to(ROOT)) for p in files}
    assert not names & {"xgpoc/detector/audit.py", "xgpoc/dino/targets.py", "xgpoc/dino/agreement.py"}
    own = [ROOT / "xgpoc/dino/attributes.py", ROOT / "xgpoc/dino/regions.py", ROOT / "scripts/xgdino_info_attributes.py"]
    for p in own:                    # the runtime guard (above) covers every read; this names no label path at all
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != at.LABEL_PATHS:
                paths = [t.lower() for t in node.value.split() if "/" in t]      # path-like tokens
                assert not [t for t in paths if any(k in t for k in ("caption", "gpt", "statsbomb"))], (p, node.value)


# ------------------------------------------------------------------ regions
def test_patch_masks():
    m = rg.box_patches(np.array([0, 0, 1, 1]) / rg.SCALE * 14)  # one input patch
    assert m.sum() == 1 and m[0, 0]
    assert rg.dilate(m).sum() == 4                               # corner: 2 x 2
    inner = rg.empty()
    inner[5, 5] = True
    assert rg.dilate(inner).sum() == 9
    assert rg.shift(inner, 3, -1)[4, 8] and rg.shift(inner, 3, -1).sum() == 1
    assert rg.shift(inner, 40, 0).sum() == 0
    assert np.array_equal(rg.unpack(rg.pack(rg.dilate(inner))), rg.dilate(inner))


def test_placebos_keep_shape_and_band_and_avoid_the_exclusion():
    mask = rg.empty()
    mask[8:10, 3:5] = True
    allowed = np.ones(rg.GRID_HW, bool)
    allowed[:, :20] = False
    shifts, n = rg.placebos(mask, allowed, np.random.default_rng(0))
    assert len(shifts) == rg.N_PLACEBO and n > rg.N_PLACEBO
    for dx, dy in shifts:
        moved = rg.shift(mask, dx, dy)
        assert abs(dy) <= rg.Y_BAND and moved.sum() == mask.sum() and not (moved & ~allowed).any()
    assert rg.placebos(mask, rg.empty(), np.random.default_rng(0)) == ([], 0)


def test_placebo_draws_are_seeded_by_shot_and_region():
    mask = rg.empty()
    mask[8, 3] = True
    allowed = np.ones(rg.GRID_HW, bool)
    first = rg.placebos(mask, allowed, rg.rng_for("s", "ball"))
    assert first == rg.placebos(mask, allowed, rg.rng_for("s", "ball"))
    assert first != rg.placebos(mask, allowed, rg.rng_for("s", "goal"))


def test_goal_polygon_is_projected_from_the_posts():
    H = np.diag([1 / SCALE, 1 / SCALE, 1.0])
    polygon, status = rg.goal_polygon(H, attack_right=True)
    assert status == "ok"
    np.testing.assert_allclose(polygon[:2], [[1050, 303.4], [1050, 376.6]])
    width = 73.2
    np.testing.assert_allclose(polygon[2:], [[1050, 376.6 - width * 2.44 / 7.32], [1050, 303.4 - width * 2.44 / 7.32]])
    assert rg.goal_polygon(np.diag([1 / 0.1, 1 / 0.1, 1.0]), True) == (None, "posts_too_close")


def test_shot_regions_of_a_planted_shot():
    _, _, t0 = build({"s": planted()})
    summary, masks, goal = rg.shot_regions("s", t0["s"], rg.empty())
    summary = pd.DataFrame(summary).set_index("region")
    assert summary.present.all()
    assert summary.loc["background", "placebo_status"] == "not_applicable" and goal["goal_status"] == "ok"
    masks = pd.DataFrame(masks)
    region = {r: rg.unpack(m) for r, m in masks[masks.kind == "region"].set_index("region")["mask"].items()}
    assert not (region["background"] & (region["players"] | region["ball"] | region["goal"])).any()
    assert (region["shooter"] <= region["players"]).all()
    persons = np.zeros(rg.GRID_HW, bool)
    for box in t0["s"].boxes():
        persons |= rg.box_patches(box)
    for row in masks[masks.kind == "placebo"].itertuples():
        m = rg.unpack(row.mask)
        assert m.sum() == region[row.region].sum() and not (m & persons).any()


def test_a_proxy_goalkeeper_has_no_region():
    shot = planted()
    shot[0] = scaled(frame_with([SHOOTER, MATE, DEFENDER, (104.0, 34.0, 1, "player")], (95.2, 40.0)))
    _, _, t0 = build({"s": shot})
    summary, _, _ = rg.shot_regions("s", t0["s"], rg.empty())
    gk = next(r for r in summary if r["region"] == "goalkeeper")
    assert not gk["present"] and gk["reason"] == "proxy_only" and gk["placebo_status"] == "region_absent"


def test_input_space_is_the_detector_frame_scaled_onto_whole_patches():
    assert rg.GRID_HW == (18, 32) and rg.SCALE == pytest.approx(0.35)
