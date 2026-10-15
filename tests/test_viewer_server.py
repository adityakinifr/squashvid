import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from squashvid.viewer.server import create_app, parse_range  # noqa: E402

DATA = bytes(range(256)) * 40  # 10240 bytes


@pytest.fixture()
def client(tmp_path):
    (tmp_path / "clip" / "viewer").mkdir(parents=True)
    (tmp_path / "clip" / "proxy.mp4").write_bytes(DATA)
    (tmp_path / "clip" / "viewer" / "meta.json").write_text(json.dumps({"n_frames": 1}))
    return TestClient(create_app(tmp_path)), tmp_path


def test_parse_range_forms():
    assert parse_range(None, 100) is None
    assert parse_range("bytes=0-", 100) == (0, 99)
    assert parse_range("bytes=10-19", 100) == (10, 19)
    assert parse_range("bytes=90-500", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    for bad in ("bytes=100-", "bytes=5-2", "bytes=-", "items=0-1"):
        with pytest.raises(ValueError):
            parse_range(bad, 100)


def test_range_requests_return_the_requested_bytes(client):
    c, _ = client
    r = c.get("/api/clip/proxy.mp4", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and r.content == DATA[100:200]
    assert r.headers["content-range"] == f"bytes 100-199/{len(DATA)}" and r.headers["accept-ranges"] == "bytes"
    r = c.get("/api/clip/proxy.mp4", headers={"Range": "bytes=-16"})
    assert r.status_code == 206 and r.content == DATA[-16:]
    r = c.get("/api/clip/proxy.mp4", headers={"Range": f"bytes={len(DATA)}-"})
    assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{len(DATA)}"


def test_full_and_head_requests(client):
    c, _ = client
    r = c.get("/api/clip/proxy.mp4")
    assert r.status_code == 200 and r.content == DATA
    h = c.head("/api/clip/proxy.mp4", headers={"Range": "bytes=0-9"})
    assert h.status_code == 206 and h.headers["content-length"] == "10"


def test_unknown_clip_and_path_traversal(client):
    c, _ = client
    assert c.get("/api/nope/meta").status_code == 404
    assert c.get("/api/clip/array/..%2f..%2fsecret").status_code == 404
    assert c.get("/api/clip/proxy.mp4?x=1").status_code == 200


def test_settings_persist_in_work_dir(client):
    c, work = client
    assert c.get("/api/settings").json()["names"] == {"A": "Player A", "B": "Player B"}
    r = c.post("/api/settings", json={"names": {"A": " Deepti ", "B": ""}})
    assert r.json()["names"] == {"A": "Deepti", "B": "Player B"}
    assert json.loads((work / "viewer_settings.json").read_text())["names"]["A"] == "Deepti"
    assert c.get("/api/settings").json()["names"]["A"] == "Deepti"
    assert [x["stem"] for x in c.get("/api/clips").json()["clips"]] == ["clip"]


def _analysis(work):
    """A minimal analysed clip: one improvement for A that names both players."""
    from tests.test_analysis_report import _stats

    stats = {**_stats(), "match": {"rallies": 6, "shots_total": 80, "clip_seconds": 60.0, "rally_shots": {"median": 7}, "rally_seconds": {"median": 9.0}}}
    item = {
        "theme": "position", "title": "Getting back to the T", "kind": "try", "play": "t_overall",
        "headline": "Stay closer to the T between shots.", "why": ["{A} was 2.0 m from the T, against 1.7 m for {B}."],
        "look_for": "how far {A} drifts", "drill": "Ghosting.", "detail": [], "evidence": [], "cites": ["c1"], "confidence": "likely",
    }
    ins = {"n_rallies": 6, "players": {"A": {"try": [item], "keep": []}, "B": {"try": [], "keep": []}}}
    a = work / "clip" / "analysis"
    a.mkdir(parents=True)
    (a / "stats.json").write_text(json.dumps(stats))
    (a / "insights.json").write_text(json.dumps(ins))
    return a


def test_report_carries_names_saved_after_the_analysis(client):
    c, work = client
    a = _analysis(work)
    assert "Player A was 2.0 m" in c.get("/api/clip/report").json()["markdown"]
    c.post("/api/settings", json={"names": {"A": "Deepti", "B": "Maya"}})
    md = c.get("/api/clip/report").json()["markdown"]
    assert "Deepti was 2.0 m from the T, against 1.7 m for Maya." in md and "## Deepti" in md
    assert "Deepti was 2.0 m" in (a / "report.md").read_text()  # the file on disk follows the names too


def test_settings_say_how_to_tell_the_players_apart(client):
    c, _ = client
    looks = c.get("/api/settings").json()["looks"]
    assert set(looks) == {"A", "B"} and all(looks.values())


def test_pooled_coaching_is_served_with_the_saved_names(client):
    c, work = client
    assert c.get("/api/pooled/insights").status_code == 404 and c.get("/api/pooled/report").status_code == 404
    item = {
        "theme": "position", "title": "Getting back to the T", "kind": "try", "play": "t_share",
        "headline": "After every shot, move back to the T before {B} hits.", "why": ["{A} spent 16% of rally time near the T."],
        "look_for": "where {A} stands", "drill": "Ghosting.", "detail": [], "evidence": [], "cites": ["c1"], "confidence": "likely",
    }
    (work / "analysis").mkdir()
    (work / "analysis" / "insights.json").write_text(json.dumps({"clip": "clip+other", "n_rallies": 40, "players": {"A": {"try": [item], "keep": []}, "B": {"try": [], "keep": []}}}))
    assert c.get("/api/pooled/insights").json()["n_rallies"] == 40
    c.post("/api/settings", json={"names": {"A": "Deepti"}})
    md = c.get("/api/pooled/report").json()["markdown"]
    assert "Deepti spent 16% of rally time near the T." in md and "before Player B hits" in md
