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
