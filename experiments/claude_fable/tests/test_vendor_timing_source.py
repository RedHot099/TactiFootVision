"""Timing-source test (xg-vendor ticket 04) on fixtures with planted clocks and known answers.

The fixture world has one match, Alpha 2 - 1 Beta (StatsBomb match 11), with a half offset per half:
SoccerNet position = StatsBomb time + OFFSET[half] + planted residual. Half 2's video starts 52 s before kick-off.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_timing_source as script  # noqa: E402
from xgpoc.paths import LABELS_ROOT, STATSBOMB  # noqa: E402
from xgpoc.vendor import timing_source as ts  # noqa: E402
from xgpoc.vendor import timing_verdict  # noqa: E402
from xgpoc.vendor import unpaired_reasons as ur  # noqa: E402
from xgpoc.vendor.statsbomb_fetch import file_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAME = "england_epl/2015-2016/2015-08-16 - 18-00 Alpha 2 - 1 Beta"
MATCH_ID = 11
OFFSET = {1: 7300, 2: 52_100}
KICK_OFF_MS = {1: 500, 2: 600}
PAIR_COLUMNS = ["shot_id", "game", "half", "split", "position_ms", "sb_event_id", "sb_timestamp_ms", "offset_ms", "residual_ms", "paired"]
SHOT_COLUMNS = ["match_id", "event_id", "period", "timestamp_ms", "team", "outcome", "event_json"]


class World:
    """Builds the inputs of `timing_source_test`. `plant` adds a StatsBomb event and its SoccerNet label."""

    def __init__(self):
        self.annotations, self.timing, self.shots, self.pairs = [], [], [], []
        for period, ms in KICK_OFF_MS.items():
            self.sb_event(period, ms, "Pass", "Kick Off", label="Kick-off")

    def sb_event(self, period, ms, kind, detail="", team="Alpha", ball_out=False, label=None):
        event_id = f"t{len(self.timing)}"
        self.timing.append({"match_id": MATCH_ID, "event_id": event_id, "index": len(self.timing), "period": period, "timestamp_ms": ms,
                            "type": kind, "detail": detail, "team": team, "ball_out": ball_out, "soccernet_label": label})
        return event_id

    def sb_shot(self, period, ms, outcome="Saved", out=False):
        event = {"id": f"s{len(self.shots)}"} | ({"out": True} if out else {})
        self.shots.append({"match_id": MATCH_ID, "event_id": event["id"], "period": period, "timestamp_ms": ms, "team": "Alpha",
                           "outcome": outcome, "event_json": json.dumps(event)})
        return event["id"]

    def sn_label(self, half, position_ms, label, team="home", visibility="visible"):
        self.annotations.append({"gameTime": f"{half} - 00:00", "label": label, "position": str(position_ms), "team": team,
                                 "visibility": visibility})

    def plant(self, label, half, sb_ms, residual, team="home", visibility="visible"):
        kind, detail = {"Substitution": ("Substitution", "Tactical"), "Corner": ("Pass", "Corner"), "Throw-in": ("Pass", "Throw-in"),
                        "Foul": ("Foul Committed", "")}[label]
        self.sb_event(half, sb_ms, kind, detail, "Alpha" if team == "home" else "Beta", label=label)
        self.sn_label(half, sb_ms + OFFSET[half] + residual, label, team, visibility)

    def paired_shot(self, half, sb_ms, residual):
        event_id = self.sb_shot(half, sb_ms)
        position = sb_ms + OFFSET[half] + residual
        self.pairs.append({"shot_id": f"sn{len(self.pairs)}", "game": GAME, "half": half, "split": "train", "position_ms": position,
                           "sb_event_id": event_id, "sb_timestamp_ms": sb_ms, "offset_ms": float(OFFSET[half]),
                           "residual_ms": float(residual), "paired": True})

    def inputs(self):
        match_map = pd.DataFrame([{"game": GAME, "sb_match_id": MATCH_ID, "sb_home": "Alpha", "sb_away": "Beta", "reason": "paired"}])
        half_offsets = pd.DataFrame([{"game": GAME, "half": h, "sb_match_id": MATCH_ID, "offset_ms": float(o), "offset_source": "refit"}
                                     for h, o in OFFSET.items()])
        labels = [{"UrlLocal": GAME + "/", "annotations": self.annotations}]
        return dict(labels=labels, half_offsets=half_offsets, match_map=match_map, pairs=pd.DataFrame(self.pairs, columns=PAIR_COLUMNS),
                    timing=pd.DataFrame(self.timing), shots=pd.DataFrame(self.shots, columns=SHOT_COLUMNS))

    def run(self):
        return ts.timing_source_test(**self.inputs(), rule_sha256=ts.RULE_SHA256)


def planted_world(sub_residuals, corner_residuals, shot_residual=(-20, 5, 15, -8, 30, 0, 11, -14)):
    """Substitutions every 80 s from 200 s in both halves, corners 40 s after each, 8 paired shots per half."""
    world = World()
    for k, residual in enumerate(sub_residuals):
        world.plant("Substitution", 1 + k % 2, 200_000 + 80_000 * (k // 2), residual)
    for k, residual in enumerate(corner_residuals):
        world.plant("Corner", 1 + k % 2, 240_000 + 80_000 * (k // 2), residual)
    for half in (1, 2):
        for k, residual in enumerate(shot_residual):
            world.paired_shot(half, 100_000 + 300_000 * k, residual)
    return world


def uniform(low, high, n, seed):
    return np.random.default_rng(seed).uniform(low, high, n).round().astype(int)


# ------------------------------------------------------------------------------------------------ verdicts
def test_planted_shared_clock_gives_shared():
    result = planted_world(uniform(-150, 150, 60, 0), uniform(-100, 100, 60, 1)).run()
    record = result.verdict
    assert (record["verdict"], record["step"]) == ("shared", 2)
    assert record["statistics"]["n_substitutions"] == 60 and record["statistics"]["substitution_share_within_200ms"] == 1.0
    parsed = timing_verdict.parse_timing_verdict(json.loads(json.dumps(record)), ts.RULE_SHA256)
    with pytest.raises(timing_verdict.TimingVerdictError, match="needs the verdict 'independent'"):
        timing_verdict.residual_arm_permission(parsed, "confirmatory")


def test_planted_substitution_lags_with_agreeing_kicks_give_independent():
    result = planted_world(uniform(2000, 8000, 60, 0), uniform(-100, 100, 60, 1)).run()
    record = result.verdict
    assert (record["verdict"], record["step"]) == ("independent", 3)
    assert record["statistics"]["substitution_median_abs_ms"] >= 2000
    assert record["statistics"]["shot_loo_share_within_200ms"] == 1.0 and record["statistics"]["corner_share_within_200ms"] == 1.0
    assert timing_verdict.residual_arm_permission(timing_verdict.parse_timing_verdict(record), "confirmatory").startswith("timing verdict 'independent'")


def test_lagging_substitutions_with_lagging_corners_stay_unresolved():
    record = planted_world(uniform(2000, 8000, 60, 0), uniform(800, 1500, 60, 1)).run().verdict
    assert (record["verdict"], record["step"]) == ("unresolved", 4)


def test_substitutions_between_the_thresholds_stay_unresolved():
    record = planted_world(uniform(250, 900, 60, 0), uniform(-100, 100, 60, 1)).run().verdict
    assert (record["verdict"], record["step"]) == ("unresolved", 4)


def test_too_few_substitutions_or_too_few_matched_are_unresolved():
    few = planted_world(uniform(-50, 50, 49, 0), uniform(-100, 100, 60, 1)).run().verdict
    assert (few["verdict"], few["step"]) == ("unresolved", 1)
    world = planted_world(uniform(-50, 50, 60, 0), uniform(-100, 100, 60, 1))
    for k in range(61):   # SoccerNet substitutions StatsBomb never logged, far from every StatsBomb substitution
        world.sn_label(1, 3_000_000 + 1000 * k + OFFSET[1], "Substitution")
    unmatched = world.run()
    assert (unmatched.verdict["verdict"], unmatched.verdict["step"]) == ("unresolved", 1)
    assert unmatched.verdict["statistics"]["substitution_match_rate"] == pytest.approx(60 / 121)


def test_decide_boundaries_are_inclusive():
    shots, corners = np.zeros(10), np.zeros(50)
    exactly_90 = np.r_[np.full(45, 200.0), np.full(5, 201.0)]
    assert ts.decide(exactly_90, 1.0, shots, corners)["verdict"] == "shared"
    lag = np.full(50, -1000.0)
    assert ts.decide(lag, 0.5, shots, corners)["verdict"] == "independent"
    assert ts.decide(lag, 0.5, shots, corners[:49])["verdict"] == "unresolved"
    corner_80 = np.r_[np.zeros(40), np.full(10, 500.0)]
    assert ts.decide(lag, 0.5, shots, corner_80)["verdict"] == "independent"
    assert ts.decide(lag, 0.5, np.r_[np.zeros(7), np.full(3, 300.0)], corners)["verdict"] == "unresolved"


def test_half_time_substitutions_leave_the_rule_set():
    world = planted_world(uniform(-100, 100, 60, 0), uniform(-100, 100, 60, 1))
    for k in range(10):   # StatsBomb pins half-time substitutions to 0; SoccerNet shows them 30-50 s before kick-off
        world.sb_event(2, 0, "Substitution", "Tactical", label="Substitution")
        world.sn_label(2, OFFSET[2] - 30_000 - 2000 * k, "Substitution")
    result = world.run()
    excluded = result.label_matches[result.label_matches.excluded.ne("")]
    assert len(excluded) == 10 and set(excluded.excluded) == {ts.HALF_TIME}
    assert (excluded.residual_ms < -29_000).all()
    assert result.verdict["verdict"] == "shared"   # 60 of 70 (0.857) would not be
    assert result.verdict["statistics"]["n_substitutions"] == 60
    assert result.verdict["statistics"]["substitution_match_rate"] == 1.0


def test_half_time_flag_uses_the_second_half_kick_off():
    world = World()
    early = world.sb_event(2, 400, "Substitution", "Tactical", label="Substitution")
    late = world.sb_event(2, 700, "Substitution", "Tactical", label="Substitution")
    first = world.sb_event(1, 0, "Substitution", "Injury", label="Substitution")
    inputs = world.inputs()
    events = ts.sb_label_events(inputs["timing"], inputs["shots"], inputs["match_map"]).set_index("event_id")
    assert events.half_time[early] and not events.half_time[late] and not events.half_time[first]


# ------------------------------------------------------------------------------------------------ matching
def test_residual_sign_one_to_one_and_window():
    world = World()
    world.plant("Throw-in", 1, 600_000, 250)
    world.sn_label(1, 600_000 + OFFSET[1] + 900, "Throw-in")          # second label near the same event: no second partner
    world.plant("Throw-in", 2, 900_000, 9_000)                        # inside the 10 s window
    world.plant("Throw-in", 2, 1_500_000, -11_000)                    # outside
    world.plant("Substitution", 1, 1_800_000, 100_000, team="away")   # inside the 120 s window; team agrees
    world.sn_label(2, 300_000 + OFFSET[2], "Substitution", team="away")
    world.sb_event(2, 300_000, "Substitution", "Tactical", team="Alpha", label="Substitution")   # team disagrees
    matches = world.run().label_matches.set_index(["half", "position_ms"])
    assert matches.residual_ms[(1, 600_000 + OFFSET[1] + 250)] == 250
    assert not matches.matched[(1, 600_000 + OFFSET[1] + 900)]
    assert matches.residual_ms[(2, 900_000 + OFFSET[2] + 9_000)] == 9_000
    assert not matches.matched[(2, 1_500_000 + OFFSET[2] - 11_000)]
    assert matches.residual_ms[(1, 1_800_000 + OFFSET[1] + 100_000)] == 100_000
    assert matches.team_agree[(1, 1_800_000 + OFFSET[1] + 100_000)]
    assert not matches.team_agree[(2, 300_000 + OFFSET[2])]


def test_labels_match_only_their_mapped_events():
    world = World()
    world.sb_event(1, 600_000, "Pass", "Corner", label="Corner")
    world.sn_label(1, 600_000 + OFFSET[1], "Throw-in")
    matches = world.run().label_matches
    assert not matches.matched.any()


def test_ball_out_events_include_out_shots_and_passes_that_went_out():
    world = World()
    out_pass = world.sb_event(1, 100_000, "Clearance", ball_out=True, label="Ball out of play")
    throw = world.sb_event(1, 200_000, "Pass", "Throw-in", ball_out=True, label="Throw-in")
    wide = world.sb_shot(1, 300_000, "Off T")
    post_out = world.sb_shot(1, 400_000, "Post", out=True)
    world.sb_shot(1, 500_000, "Saved")
    world.sb_shot(1, 600_000, "Post")
    inputs = world.inputs()
    events = ts.sb_label_events(inputs["timing"], inputs["shots"], inputs["match_map"])
    assert set(events.event_id[events.label.eq("Ball out of play")]) == {out_pass, throw, wide, post_out}
    assert set(events.label[events.event_id.eq(throw)]) == {"Throw-in", "Ball out of play"}


def test_leave_one_out_shot_residuals():
    pairs = pd.DataFrame({"shot_id": ["a", "b", "c", "d"], "game": GAME, "half": [1, 1, 1, 2], "split": "train",
                          "position_ms": [1100, 2110, 3130, 5000], "sb_event_id": list("wxyz"), "sb_timestamp_ms": [1000, 2000, 3000, 4000],
                          "offset_ms": 110.0, "residual_ms": [-10.0, 0.0, 20.0, 0.0], "paired": True})
    shots = ts.shot_residuals(pairs).set_index("shot_id")
    assert shots.loo_offset_ms["a"] == 120 and shots.loo_residual_ms["a"] == -20
    assert shots.loo_offset_ms["b"] == 115 and shots.loo_residual_ms["b"] == -5
    assert shots.loo_offset_ms["c"] == 105 and shots.loo_residual_ms["c"] == 25
    assert np.isnan(shots.loo_residual_ms["d"])


def test_agreement_table_and_sensitivity():
    world = planted_world(uniform(-100, 100, 60, 0), uniform(-100, 100, 60, 1))
    world.plant("Substitution", 1, 2_700_000, 400, visibility="not shown")
    result = world.run()
    table = result.agreement.set_index(["event", "subset"])
    assert table.n_labels[("Substitution", "all")] == 61 and table.n_labels[("Substitution", "not shown")] == 1
    assert table.median_ms[("Substitution", "not shown")] == 400
    assert table.n_candidates[("Substitution", "all")] == 61 and table.n_candidates[("Corner", "all")] == 60
    assert table.share_within_200ms[("Shot (paired)", "leave-one-out offset")] == 1.0
    sensitivity = result.substitution_sensitivity.set_index(["subset", "window_ms"])
    assert sensitivity.n_labels[("visible", 120_000)] == 60 and sensitivity.n_labels[("all", 30_000)] == 61
    assert sensitivity.frozen_window[("all", 120_000)] and not sensitivity.frozen_window[("all", 30_000)]


def test_refuses_another_rule():
    with pytest.raises(ValueError, match="not the pinned rule"):
        ts.timing_source_test(**World().inputs(), rule_sha256="0" * 64)


def test_rule_note_is_the_pinned_one():
    assert ts.rule_note_sha256(ROOT) == ts.RULE_SHA256


# ------------------------------------------------------------------------------------------------ unpaired shots
def raw(index, period, timestamp, kind, **extra):
    return {"id": f"r{index}", "index": index, "period": period, "timestamp": timestamp, "type": {"name": kind},
            "team": {"name": "Alpha"}} | extra


def test_event_frame_flags():
    frame = ur.event_frame(MATCH_ID, [
        raw(1, 1, "00:00:00.000", "Half Start"),
        raw(2, 1, "00:01:00.500", "Pass", **{"pass": {"type": {"name": "Throw-in"}}}),
        raw(3, 1, "00:02:00.000", "Pass", **{"pass": {"outcome": {"name": "Pass Offside"}}}),
        raw(4, 1, "00:03:00.000", "Pass", **{"pass": {"outcome": {"name": "Out"}}}),
        raw(5, 1, "00:04:00.000", "Shot", **{"shot": {"outcome": {"name": "Off T"}}}),
        raw(6, 1, "00:05:00.000", "Clearance", out=True),
        raw(7, 1, "00:06:00.000", "Carry"),
    ]).set_index("index")
    assert frame.timestamp_ms[2] == 60_500 and frame.detail[2] == "Throw-in" and frame.detail[3] == "Pass Offside"
    assert frame.restarts_play.tolist() == [True, True, False, False, False, False, False]
    assert frame.stops_play.tolist() == [False, False, True, True, True, True, False]
    assert frame.ball_out.tolist() == [False, False, False, True, True, True, False]


def test_unpaired_shot_reasons():
    events = ur.event_frame(MATCH_ID, [
        raw(1, 1, "00:00:00.000", "Half Start"),
        raw(2, 1, "00:05:00.000", "Offside"),                                                     # B: 3 s before
        raw(3, 1, "00:10:00.000", "Foul Committed"),                                              # C: 6 s before, then restarted
        raw(4, 1, "00:10:04.000", "Pass", **{"pass": {"type": {"name": "Free Kick"}}}),
        raw(5, 1, "00:10:05.000", "Pass"),
        raw(6, 1, "00:20:00.000", "Pass", **{"pass": {"outcome": {"name": "Pass Offside"}}}),     # G: 2 s before
        raw(7, 1, "00:30:00.000", "Pass"),                                                        # E: 20 s after
        raw(8, 1, "00:45:00.000", "Half End"),                                                    # D: 30 s before
    ])
    shot_ms = {"A": 300_500, "B": 303_000, "C": 606_000, "D": 2_730_000, "E": 1_780_000, "F": np.nan, "G": 1_202_000}
    unpaired = pd.DataFrame([{"shot_id": name, "game": GAME, "split": "test", "half": 1, "position_ms": 0, "label": "Shots on target",
                              "team": "home", "goal": False, "sb_match_id": MATCH_ID, "sb_clock_ms": clock,
                              "nearest_sb_ms": 5_000.0 if name == "A" else 60_000.0} for name, clock in shot_ms.items()])
    reasons = ur.unpaired_reasons(unpaired, events).set_index("shot_id")
    assert reasons.reason.to_dict() == {"A": "sb_shot_within_10s", "B": "dead_ball", "C": "other_event_within_2_5s", "D": "outside_half",
                                        "E": "no_event_within_2_5s", "F": "no_half_offset", "G": "dead_ball"}
    assert reasons.stoppage_type["B"] == "Offside" and reasons.stoppage_dt_ms["B"] == -3000
    assert reasons.stoppage_type["G"] == "Pass (Pass Offside)"
    assert reasons.nearest_event_type["C"] == "Pass" and reasons.nearest_event_dt_ms["C"] == -1000
    assert reasons.nearest_event_dt_ms["E"] == 20_000
    assert reasons.no_sb_shot_within_10s.to_dict() == {k: k != "A" for k in shot_ms}
    summary = ur.reason_summary(reasons)
    assert summary["unpaired_shots"] == 7 and summary["unpaired_shots_without_sb_shot_within_10s"] == 6
    assert summary["reasons_without_sb_shot_within_10s"]["dead_ball"] == 2


# ------------------------------------------------------------------------------------------------ script inputs
def write_record(folder: Path, files: dict[str, bytes], extra: dict) -> None:
    for name, body in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(body)
    record = extra | {"files": {name: file_sha256(folder / name) for name in files}}
    (folder / "complete.json").write_text(json.dumps(record))


def test_pairing_folder_is_verified(tmp_path):
    write_record(tmp_path, {"half_offsets.csv": b"a\n1\n", "match_map.csv": b"b\n2\n"}, {"identity": {}})
    assert set(script.verify_record(tmp_path)["files"]) == {"half_offsets.csv", "match_map.csv"}
    (tmp_path / "match_map.csv").write_bytes(b"b\n3\n")
    with pytest.raises(ValueError, match="differs from the hash"):
        script.verify_record(tmp_path)
    with pytest.raises(ValueError, match="no completion record"):
        script.verify_record(tmp_path / "missing")


def test_statsbomb_table_outside_the_record_is_refused(tmp_path):
    write_record(tmp_path, {"tables/timing.parquet": b"x"}, {"identity": {}})
    assert set(script.verify_record(tmp_path, ["tables/timing.parquet"])["files"]) == {"tables/timing.parquet"}
    with pytest.raises(ValueError, match="does not list tables/shots.parquet"):
        script.verify_record(tmp_path, ["tables/timing.parquet", "tables/shots.parquet"])


def test_finished_folder_is_verified_not_rebuilt(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.csv").write_text("x\n")
    identity = {"version": script.VERSION}
    (out / "complete.json").write_text(json.dumps({"identity": identity, "files": {"a.csv": file_sha256(out / "a.csv")}}))
    assert script.verify_complete(out, identity)["files"] == {"a.csv": file_sha256(out / "a.csv")}
    with pytest.raises(ValueError, match="changed"):
        script.verify_complete(out, {"version": "other"})


def fixture_sources(tmp_path: Path, world: World) -> dict:
    """A StatsBomb data version, a frozen pairing folder and a Labels-v2 root holding `world`, with completion records."""
    sha = "a" * 40
    inputs = world.inputs()
    statsbomb, pairing_root, labels_root = tmp_path / "statsbomb", tmp_path / "pairing", tmp_path / "labels"
    (statsbomb / "tables").mkdir(parents=True)
    inputs["timing"].to_parquet(statsbomb / "tables/timing.parquet")
    inputs["shots"].to_parquet(statsbomb / "tables/shots.parquet")
    body = json.dumps([raw(1, 1, "00:00:00.000", "Half Start"), raw(2, 1, "00:05:00.000", "Offside")]).encode()
    stored = statsbomb / f"raw/events/{MATCH_ID}.json.gz"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(gzip.compress(body, mtime=0))
    meta = {"sha": sha, "path": f"data/events/{MATCH_ID}.json", "stored_sha256": file_sha256(stored),
            "upstream_sha256": hashlib.sha256(body).hexdigest()}
    stored.with_name(stored.name + ".meta.json").write_text(json.dumps(meta))
    names = ["tables/timing.parquet", "tables/shots.parquet", f"raw/events/{MATCH_ID}.json.gz"]
    (statsbomb / "complete.json").write_text(json.dumps({"identity": {"open_data": {"repo": "x", "sha": sha}},
                                                          "files": {n: file_sha256(statsbomb / n) for n in names}}))
    (labels_root / GAME).mkdir(parents=True)
    (labels_root / GAME / "Labels-v2.json").write_text(json.dumps(inputs["labels"][0]))
    labels_digest = script.pairing_inputs.read_labels(labels_root)[1]
    pairing_root.mkdir()
    inputs["half_offsets"].to_csv(pairing_root / "half_offsets.csv", index=False)
    inputs["match_map"].to_csv(pairing_root / "match_map.csv", index=False)
    inputs["pairs"].to_parquet(pairing_root / "pairs.parquet")
    pd.DataFrame([{"shot_id": "u1", "game": GAME, "split": "test", "half": 1, "position_ms": 303_000 + OFFSET[1], "label": "Shots on target",
                   "team": "home", "goal": False, "offset_ms": float(OFFSET[1]), "sb_clock_ms": 303_000.0, "sb_match_id": MATCH_ID,
                   "nearest_sb_ms": 60_000.0}]).astype({"sb_match_id": "Int64"}).to_parquet(pairing_root / "unpaired_sn.parquet")
    identity = {"inputs": {"statsbomb": {"complete_json_sha256": file_sha256(statsbomb / "complete.json")}, "labels_v2": {"sha256": labels_digest}}}
    (pairing_root / "complete.json").write_text(json.dumps({"identity": identity, "files": {
        n: file_sha256(pairing_root / n) for n in script.PAIRING_FILES}}))
    return {"STATSBOMB": statsbomb, "PAIRING": pairing_root, "LABELS_ROOT": labels_root, "OPEN_DATA_SHA": sha,
            "PAIRING_COMPLETE_SHA256": file_sha256(pairing_root / "complete.json")}


def test_script_end_to_end_publishes_verifies_and_refuses(tmp_path, monkeypatch, capsys):
    sources = fixture_sources(tmp_path, planted_world(uniform(2000, 8000, 60, 0), uniform(-100, 100, 60, 1)))
    for name, value in sources.items():
        monkeypatch.setattr(script, name, value)
    out = tmp_path / "out"
    script.main(["--out", str(out)])
    record = timing_verdict.load_timing_verdict(out / "timing_verdict.json", ts.RULE_SHA256)
    assert record.verdict == "independent"
    complete = json.loads((out / "complete.json").read_text())
    assert set(complete["files"]) == {"label_matches.parquet", "shot_residuals.parquet", "agreement.csv", "substitution_sensitivity.csv",
                                      "unpaired_reasons.csv", "timing_verdict.json"}
    assert complete["identity"]["rule"]["sha256"] == ts.RULE_SHA256 and complete["summary"]["reasons"] == {"dead_ball": 1}
    assert pd.read_csv(out / "unpaired_reasons.csv").stoppage_type.tolist() == ["Offside"]
    script.main(["--out", str(out)])
    assert "nothing rebuilt" in capsys.readouterr().out
    (out / "agreement.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="differs from the hash"):
        script.main(["--out", str(out)])
    (sources["PAIRING"] / "match_map.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="differs from the hash"):
        script.main(["--out", str(tmp_path / "other")])
    monkeypatch.setattr(script, "PAIRING_COMPLETE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not the frozen pairing"):
        script.main(["--out", str(tmp_path / "other")])
    monkeypatch.setattr(script, "OPEN_DATA_SHA", "b" * 40)
    with pytest.raises(ValueError, match="open-data commit"):
        script.main(["--out", str(tmp_path / "other")])


@pytest.mark.skipif(not (LABELS_ROOT.exists() and (STATSBOMB / "complete.json").exists() and (script.PAIRING / "complete.json").exists()),
                    reason="needs the real data and the frozen pairing")
def test_real_inputs_are_the_frozen_ones():
    inputs = script.load_inputs(STATSBOMB, script.PAIRING, LABELS_ROOT, script.OPEN_DATA_SHA, script.PAIRING_COMPLETE_SHA256)
    games = inputs.match_map.game[inputs.match_map.reason.eq("paired")]
    labels = ts.sn_labels(inputs.labels, games)
    assert len(games) == 135 and (labels.label == "Substitution").sum() == 749
    assert len(inputs.unpaired_sn) == 285 and (inputs.unpaired_sn.nearest_sb_ms > ur.SHOT_NEAR_MS).sum() == 249
