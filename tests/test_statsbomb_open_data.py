"""download_statsbomb lays one open-data match out for load_statsbomb, from a stubbed HTTP layer."""

import io
import json
import urllib.error

import pytest

import tactifoot_vision as tv
from tactifoot_vision.data import statsbomb_open_data as sb

EVENT = {
    "id": "e1",
    "index": 1,
    "period": 1,
    "timestamp": "00:00:01.000",
    "minute": 0,
    "second": 1,
    "type": {"name": "Pass"},
    "possession": 1,
    "possession_team": {"name": "A"},
    "play_pattern": {"name": "Regular Play"},
    "team": {"name": "A"},
    "player": {"name": "P"},
}
FRAME = {
    "event_uuid": "e1",
    "visible_area": [0, 0, 10, 0, 10, 10],
    "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [5.0, 5.0]},
        {"teammate": False, "actor": False, "keeper": True, "location": [9.0, 1.0]},
    ],
}


@pytest.fixture
def server(monkeypatch):
    """A stub for urlopen: the urls asked, the failures to raise first per url suffix, the sleeps."""
    asked: list[str] = []
    failures: dict[str, list[Exception]] = {}
    payloads = {"/events/7.json": [EVENT], "/three-sixty/7.json": [FRAME]}

    def urlopen(url, timeout=None):
        asked.append(url)
        suffix = url.split("/data", 1)[1]
        if failures.get(suffix):
            raise failures[suffix].pop(0)
        return io.BytesIO(json.dumps(payloads[suffix]).encode())

    waits: list[float] = []
    monkeypatch.setattr(sb.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sb.time, "sleep", waits.append)
    return asked, failures, waits


def _http(code):
    return urllib.error.HTTPError("u", code, "x", {}, None)


def test_a_match_lands_in_the_layout_load_statsbomb_reads(tmp_path, server, capsys):
    asked, _, _ = server

    data = tv.data.download_statsbomb(7, tmp_path)

    assert data.folder == tmp_path / "7" and data.match == "7"
    assert sorted(p.name for p in data.folder.iterdir()) == [
        "7_360.json",
        "7_events.json",
    ]
    assert [u.rsplit("/data/", 1)[1] for u in asked] == [
        "events/7.json",
        "three-sixty/7.json",
    ]
    table = tv.data.load_statsbomb(data.folder)
    assert len(table) == 2 and set(table["type"]) == {"player", "goalkeeper"}
    assert data.attribution == sb.ATTRIBUTION
    assert "StatsBomb" in data.attribution and "logo" in data.attribution
    assert sb.ATTRIBUTION in capsys.readouterr().out


def test_a_second_call_makes_no_request(tmp_path, server):
    asked, _, _ = server
    first = tv.data.download_statsbomb("7", tmp_path)
    asked.clear()

    assert tv.data.download_statsbomb("7", tmp_path) == first
    assert asked == []


def test_only_the_missing_file_is_fetched(tmp_path, server):
    asked, _, _ = server
    data = tv.data.download_statsbomb(7, tmp_path)
    (data.folder / "7_360.json").unlink()
    asked.clear()

    tv.data.download_statsbomb(7, tmp_path)

    assert [u.rsplit("/data/", 1)[1] for u in asked] == ["three-sixty/7.json"]


def test_an_unknown_match_is_a_clear_error_and_leaves_nothing(tmp_path, server):
    _, failures, waits = server
    failures["/three-sixty/7.json"] = [_http(404)]

    with pytest.raises(ValueError, match="7_360.json.*not every match"):
        tv.data.download_statsbomb(7, tmp_path)
    assert waits == []  # a 404 is not retried
    assert not list((tmp_path / "7").glob("*.part"))


def test_a_match_id_must_be_a_number(tmp_path, server):
    with pytest.raises(ValueError, match="number"):
        tv.data.download_statsbomb("../x", tmp_path)


def test_network_errors_are_retried_then_succeed(tmp_path, server):
    _, failures, waits = server
    failures["/events/7.json"] = [urllib.error.URLError("dns"), _http(503)]

    data = tv.data.download_statsbomb(7, tmp_path)

    assert (data.folder / "7_events.json").is_file()
    assert waits == [sb.RETRY_WAIT_SECONDS] * 2
    assert 10 <= sb.RETRY_WAIT_SECONDS <= 30


def test_a_download_that_keeps_failing_is_a_connection_error(tmp_path, server):
    _, failures, waits = server
    failures["/events/7.json"] = [urllib.error.URLError("no route")] * sb.ATTEMPTS

    with pytest.raises(ConnectionError, match="after 5 tries"):
        tv.data.download_statsbomb(7, tmp_path)
    assert len(waits) == sb.ATTEMPTS - 1
    assert not (tmp_path / "7" / "7_events.json").exists()
