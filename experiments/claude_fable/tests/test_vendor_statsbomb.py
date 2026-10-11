"""StatsBomb data version on a two-match fixture: no network, the fetch function is injected."""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_fetch_statsbomb as builder  # noqa: E402
from xgpoc.vendor import statsbomb_fetch as fetching  # noqa: E402
from xgpoc.vendor.statsbomb_tables import (  # noqa: E402
    build_tables,
    event_rows,
    timestamp_ms,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40
COVERED = ((1, 1),)
A, B = 101, 102


def ev(index, kind, period=1, timestamp="00:00:01.000", team="Home", **extra):
    base = {"id": f"e{index}", "index": index, "period": period, "timestamp": timestamp, "minute": 0, "second": 1,
            "type": {"id": 0, "name": kind}, "possession": 1, "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 7 if team == "Home" else 8, "name": team}, "player": {"id": 70, "name": "P"}, "location": [100.0, 40.0]}
    return base | extra


def shot(index, key_pass_id=None, freeze=True, outcome="Goal", timestamp="00:00:10.250"):
    body = {"statsbomb_xg": 0.25, "outcome": {"name": outcome}, "body_part": {"name": "Right Foot"},
            "type": {"name": "Open Play"}, "technique": {"name": "Normal"}}
    if key_pass_id:
        body["key_pass_id"] = key_pass_id
    if freeze:
        body["freeze_frame"] = [{"location": [110, 40], "player": {"id": 1}, "teammate": False},
                                {"location": [90, 35], "player": {"id": 2}, "teammate": True}]
    return ev(index, "Shot", timestamp=timestamp, shot=body)


EVENTS_A = [
    ev(1, "Half Start", timestamp="00:00:00.000"),
    ev(2, "Pass", pass_={"type": {"name": "Kick Off"}}),
    ev(3, "Pass", pass_={}),
    shot(4, key_pass_id="e3"),
    ev(5, "Substitution", period=2, timestamp="00:00:00.000", substitution={"outcome": {"name": "Tactical"}}),
    ev(6, "Pass", timestamp="00:01:02.500", pass_={"type": {"name": "Throw-in"}}),
    ev(7, "Pass", timestamp="00:02:00.000", pass_={"outcome": {"name": "Out"}}),
    ev(8, "Clearance", timestamp="00:03:00.000", out=True),
    ev(9, "Foul Committed", timestamp="00:04:00.000"),
    ev(10, "Offside", timestamp="00:05:00.000"),
    ev(11, "Own Goal Against", timestamp="00:06:00.000", team="Away"),
    ev(12, "Pass", timestamp="00:07:00.000", pass_={"type": {"name": "Corner"}, "outcome": {"name": "Out"}}),
    ev(13, "Ball Receipt*", timestamp="00:08:00.000"),
    ev(14, "Pass", timestamp="00:09:00.000", pass_={"outcome": {"name": "Incomplete"}}),
    ev(15, "Pass", timestamp="00:10:00.000", pass_={"outcome": {"name": "Pass Offside"}}),
]
EVENTS_B = [ev(1, "Pass", pass_={}), shot(2, freeze=False, outcome="Saved", timestamp="01:02:03.004")]


def prepare(events):
    """The JSON key `pass` is a keyword in Python; the fixture writes `pass_`."""
    return [{("pass" if k == "pass_" else k): v for k, v in e.items()} for e in events]


def match_record(match_id, home_score):
    return {"match_id": match_id, "match_date": "2015-08-23", "kick_off": "17:00:00.000", "match_status": "available",
            "competition": {"competition_id": 1, "competition_name": "League"}, "season": {"season_id": 1, "season_name": "2015/2016"},
            "home_team": {"home_team_id": 7, "home_team_name": "Home"}, "away_team": {"away_team_id": 8, "away_team_name": "Away"},
            "home_score": home_score, "away_score": 0}


class FakeRepo:
    """Serves a two-match repository; records calls; failures are scripted per upstream path."""

    def __init__(self, sha=SHA):
        self.sha = sha
        prefix = f"https://raw.githubusercontent.com/statsbomb/open-data/{sha}/"
        self.files = {
            prefix + "data/competitions.json": json.dumps([{"competition_id": 1, "season_id": 1}, {"competition_id": 9, "season_id": 9}]).encode(),
            prefix + "data/matches/1/1.json": json.dumps([match_record(A, 1), match_record(B, 0)]).encode(),
            prefix + f"data/events/{A}.json": json.dumps(prepare(EVENTS_A)).encode(),
            prefix + f"data/events/{B}.json": json.dumps(prepare(EVENTS_B)).encode(),
            prefix + "README.md": b"terms", prefix + "LICENSE.pdf": b"%PDF terms",
        }
        self.prefix = prefix
        self.calls: list[str] = []
        self.transient: dict[str, int] = {}   # path -> failures left
        self.permanent: set[str] = set()
        self.truncated: dict[str, int] = {}   # path -> truncated bodies left

    def __call__(self, url):
        self.calls.append(url)
        if url == fetching.COMMIT_URL:
            return json.dumps({"sha": self.sha}).encode()
        path = url.removeprefix(self.prefix)
        if path in self.permanent:
            raise fetching.PermanentFetchError(f"HTTP 404 for {url}")
        if self.transient.get(path, 0) > 0:
            self.transient[path] -= 1
            raise OSError("connection reset")
        body = self.files[url]
        if self.truncated.get(path, 0) > 0:
            self.truncated[path] -= 1
            return body[: len(body) // 2]
        return body

    def event_calls(self, match_id):
        return [c for c in self.calls if c.endswith(f"data/events/{match_id}.json")]


def run_build(out, repo, **kwargs):
    kwargs.setdefault("sleep", lambda _: None)
    return builder.build(out, repo, covered=COVERED, progress=None, **kwargs)


def test_timestamp_ms():
    assert timestamp_ms("00:09:59.015") == 599015
    assert timestamp_ms("01:02:03.004") == 3723004


def test_event_tables_known_answers():
    tables = build_tables([match_record(B, 0), match_record(A, 1)], [(A, prepare(EVENTS_A)), (B, prepare(EVENTS_B))])
    assert list(tables["matches"].match_id) == [A, B]
    shots = tables["shots"]
    assert list(shots.event_id) == ["e4", "e2"] and list(shots.match_id) == [A, B]
    first, second = shots.iloc[0], shots.iloc[1]
    assert (first.outcome, first.statsbomb_xg, first.timestamp_ms, first.key_pass_id) == ("Goal", 0.25, 10250, "e3")
    assert first.n_freeze_frame == 2 and len(json.loads(first.freeze_frame_json)) == 2
    assert json.loads(first.event_json)["shot"]["freeze_frame"] == json.loads(first.freeze_frame_json)
    assert second.n_freeze_frame == 0 and second.freeze_frame_json is None and second.timestamp_ms == 3723004
    # context: the named key pass and the own goal, nothing else
    assert list(tables["context"].event_id) == ["e3", "e11"]
    assert list(tables["context"].type) == ["Pass", "Own Goal Against"]
    a = tables["timing"][tables["timing"].match_id == A].set_index("event_id")
    assert list(a.index) == ["e1", "e2", "e5", "e6", "e7", "e8", "e9", "e10", "e12", "e15"]   # receipt, plain passes and the shot are out
    assert a.soccernet_label.to_dict() == {"e1": None, "e2": "Kick-off", "e5": "Substitution", "e6": "Throw-in", "e7": "Ball out of play",
                                           "e8": "Ball out of play", "e9": "Foul", "e10": "Offside", "e12": "Corner", "e15": "Offside"}
    assert a.ball_out.to_dict() == {"e1": False, "e2": False, "e5": False, "e6": False, "e7": True, "e8": True, "e9": False,
                                    "e10": False, "e12": True, "e15": False}
    assert (a.loc["e5", "period"], a.loc["e5", "timestamp_ms"], a.loc["e5", "detail"], a.loc["e6", "timestamp_ms"]) == (2, 0, "Tactical", 62500)
    assert tables["timing"].match_id.eq(A).all()   # match B has no timing event


def test_event_rows_follow_event_index_not_file_order():
    shots, _, _ = event_rows(A, list(reversed(prepare(EVENTS_A))))
    assert [s["event_id"] for s in shots] == ["e4"]


def test_retry_with_backoff_then_success(tmp_path):
    repo, slept = FakeRepo(), []
    repo.transient[f"data/events/{A}.json"] = 2
    run_build(tmp_path / "v", repo, sleep=slept.append, backoff_s=1.5, sha=SHA)
    assert len(repo.event_calls(A)) == 3 and slept == [1.5, 3.0]


def test_truncated_body_is_retried(tmp_path):
    repo = FakeRepo()
    repo.truncated[f"data/events/{B}.json"] = 1
    run_build(tmp_path / "v", repo, sha=SHA)
    assert len(repo.event_calls(B)) == 2


def test_missing_match_fails_listing_it_and_publishes_nothing(tmp_path):
    repo, out = FakeRepo(), tmp_path / "v"
    repo.permanent.add(f"data/events/{B}.json")
    with pytest.raises(fetching.MissingMatchesError) as error:
        run_build(out, repo, sha=SHA)
    assert list(error.value.missing) == [B] and str(B) in str(error.value)
    assert len(repo.event_calls(B)) == 1   # a permanent error is not retried
    assert not (out / "complete.json").exists() and not (out / "tables").exists()
    assert fetching.is_stored(fetching.events_path(out, A), SHA, f"data/events/{A}.json")   # the other match is kept


def test_exhausted_retries_count_as_missing(tmp_path):
    repo = FakeRepo()
    repo.transient[f"data/events/{A}.json"] = 99
    with pytest.raises(fetching.MissingMatchesError) as error:
        run_build(tmp_path / "v", repo, sha=SHA, attempts=3)
    assert list(error.value.missing) == [A] and len(repo.event_calls(A)) == 3


def test_resume_skips_verified_files_and_refetches_damaged(tmp_path):
    out, repo = tmp_path / "v", FakeRepo()
    repo.permanent.add(f"data/events/{B}.json")
    with pytest.raises(fetching.MissingMatchesError):
        run_build(out, repo, sha=SHA)
    repo.permanent.clear()
    repo.calls.clear()
    # damage match A's stored file: its hash no longer verifies
    path = fetching.events_path(out, A)
    path.write_bytes(path.read_bytes()[:-3])
    run_build(out, repo)   # the sha comes from pin.json, the API is not asked
    assert fetching.COMMIT_URL not in repo.calls
    assert len(repo.event_calls(A)) == 1 and len(repo.event_calls(B)) == 1
    assert not [c for c in repo.calls if "matches/" in c or "competitions" in c]   # verified files are skipped
    assert (out / "complete.json").exists()


def test_store_never_mixes_commits(tmp_path):
    out = tmp_path / "v"
    run_build(out, FakeRepo(), sha=SHA)
    with pytest.raises(ValueError, match="refusing to mix"):
        run_build(out, FakeRepo(OTHER_SHA), sha=OTHER_SHA)
    partial = tmp_path / "w"
    fetching.ensure_pin(partial, SHA)
    with pytest.raises(ValueError, match="refusing to mix"):
        fetching.fetch_all(FakeRepo(OTHER_SHA), partial, OTHER_SHA, COVERED)


def test_resolves_master_once_and_pins_it(tmp_path):
    repo = FakeRepo()
    record = run_build(tmp_path / "v", repo)
    assert repo.calls.count(fetching.COMMIT_URL) == 1
    assert record["identity"]["open_data"]["sha"] == SHA and fetching.read_pin(tmp_path / "v") == SHA
    with pytest.raises(ValueError, match="Not a commit sha"):
        fetching.resolve_sha(lambda _: b'{"sha": "master"}')


def test_workers_are_capped_at_four(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        fetching.fetch_all(FakeRepo(), tmp_path / "v", SHA, COVERED, workers=5)


def test_uncovered_competition_season_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not in competitions.json"):
        fetching.fetch_all(FakeRepo(), tmp_path / "v", SHA, ((1, 1), (2, 2)))


def test_only_github_hosts_are_fetched():
    with pytest.raises(ValueError, match="may be fetched"):
        fetching.http_fetch("https://example.com/x")


def test_completion_record_and_verification(tmp_path):
    out, repo = tmp_path / "v", FakeRepo()
    record = run_build(out, repo, sha=SHA)
    assert record["summary"]["matches"] == 2 and record["summary"]["shots"] == 2 and record["summary"]["goals"] == 1
    assert record["summary"]["shots_with_freeze_frame"] == 1
    assert record["summary"]["timing_events"]["Throw-in"] == 1
    files = record["files"]
    assert {"pin.json", "tables/shots.parquet", "tables/timing.parquet", f"raw/events/{A}.json.gz", f"raw/events/{A}.json.gz.meta.json",
            "raw/terms/README.md", "raw/competitions.json.gz"} <= set(files)
    assert files[f"raw/events/{A}.json.gz"] == fetching.file_sha256(fetching.events_path(out, A))
    assert set(record["identity"]["sources"]) == {str(p) for p in builder.SOURCE_FILES}
    assert gzip.decompress(fetching.events_path(out, A).read_bytes()) == repo.files[repo.prefix + f"data/events/{A}.json"]
    # a second run verifies and fetches nothing
    repo.calls.clear()
    assert run_build(out, repo) == record and repo.calls == []
    # any change is caught: a table, an extra file
    (out / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="not in the completion record"):
        run_build(out, repo)
    (out / "stray.txt").unlink()
    (out / "tables/matches.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Output changed"):
        run_build(out, repo)


def test_rebuild_from_the_same_raw_files_gives_identical_tables(tmp_path):
    first = run_build(tmp_path / "one", FakeRepo(), sha=SHA)
    second = run_build(tmp_path / "two", FakeRepo(), sha=SHA)
    assert first["files"] == second["files"]


def test_tables_are_not_published_when_the_derivation_fails(tmp_path, monkeypatch):
    out = tmp_path / "v"

    def boom(*args, **kwargs):
        raise RuntimeError("derivation failed")

    monkeypatch.setattr(builder, "build_tables", boom)
    with pytest.raises(RuntimeError):
        run_build(out, FakeRepo(), sha=SHA)
    assert not (out / "tables").exists() and not (out / "complete.json").exists()


def test_prototype_crosscheck_reports_equal_counts_and_a_planted_difference(tmp_path):
    out = tmp_path / "v"
    run_build(out, FakeRepo(), sha=SHA)
    proto = tmp_path / "proto"
    proto.mkdir()
    for name in ("matches", "shots", "context"):
        frame = pd.read_parquet(out / "tables" / f"{name}.parquet")
        frame[[c for c in frame.columns if c not in ("timestamp_ms", "team_id", "player_id", "possession", "freeze_frame_json")]].to_parquet(proto / f"{name}.parquet")
    clean = builder.prototype_crosscheck(out, proto)
    assert clean["shots"]["prototype_rows"] == clean["shots"]["new_rows"] == 2 and clean["shots"]["same_keys"]
    assert all(not part["columns_with_differences"] for part in clean.values())
    shots = pd.read_parquet(proto / "shots.parquet")
    shots.loc[0, "x"] += 1
    shots.iloc[:1].to_parquet(proto / "shots.parquet")
    planted = builder.prototype_crosscheck(out, proto)
    assert planted["shots"]["only_in_new"] == 1 and not planted["shots"]["same_keys"]
