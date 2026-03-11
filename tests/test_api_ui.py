from __future__ import annotations

from fastapi.testclient import TestClient

from squashvid.api import app

client = TestClient(app)


def test_ui_home_and_assets() -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert "text/html" in home.headers.get("content-type", "")
    assert "SquashVid" in home.text

    css = client.get("/static/css/app.css")
    js = client.get("/static/js/app.js")

    assert css.status_code == 200
    assert "text/css" in css.headers.get("content-type", "")
    assert js.status_code == 200
    assert "javascript" in js.headers.get("content-type", "")
