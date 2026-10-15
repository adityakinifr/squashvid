"""Labelling app and window planning: labellers stay independent, labels round-trip, rally windows are chosen sanely."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from squashvid.ball import label_app, predict, windows
from squashvid.ball import labels as L

TS = np.arange(0, 400) / 30.0  # a fake 30 fps clip


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    (tmp_path / "windows.json").write_text(json.dumps([{"id": "w1", "stem": "clip", "start": 0.0, "end": 3.0, "i0": 0, "i1": 90, "note": ""}]))
    balls = tmp_path / "ball"
    balls.mkdir()
    pd.DataFrame({"frame": [5, 6], "x": [10.0, np.nan], "y": [20.0, np.nan], "score": [0.9, 0.1], "visible": [True, False]}).to_parquet(balls / "m@default.parquet")
    pd.DataFrame({"frame": [5], "x": [9.0], "y": [19.0], "score": [0.8], "visible": [True]}).to_parquet(balls / f"{label_app.DEFAULT_LAYERS[0]}.parquet")
    pd.DataFrame({"frame": [5], "x": [8.0], "y": [18.0], "score": [0.7], "visible": [True]}).to_parquet(balls / "a@default-tone.parquet")
    L.append("clip", L.Label(5, TS[5], 11.0, 21.0, "visible", source="self", labeller="other"), tmp_path)
    L.append("clip", L.Label(7, TS[7], None, None, "out", labeller="other"), tmp_path)
    monkeypatch.setattr(label_app, "video", lambda stem: SimpleNamespace(timestamps=TS))
    monkeypatch.setattr(label_app, "ball_dir", lambda stem: balls)
    monkeypatch.setattr(label_app.FrameStore, "ensure", lambda self, *a: None)
    return TestClient(label_app.create_app("me", tmp_path))


def test_other_labellers_are_a_layer_not_my_labels(client):
    w = client.get("/api/window/w1").json()
    assert w["labels"] == []
    assert w["prelabels"]["labels:other"] == [[5, 11.0, 21.0, 1.0, True]]  # 'out' has no position to show
    assert w["prelabels"]["m@default"][0] == [5, 10.0, 20.0, 0.9, True]
    assert client.get("/api/windows").json()["labelled"] == 0
    models = [k for k in w["prelabels"] if not k.startswith("labels:")]
    # the preferred model is listed (and selected) first; outputs from a superseded decode come after current ones
    assert models == [label_app.DEFAULT_LAYERS[0], "m@default", "a@default-tone"]
    assert w["default_layers"][0] == label_app.DEFAULT_LAYERS[0]


def test_posted_labels_round_trip_and_count(client, tmp_path):
    r = client.post("/api/label", json={"stem": "clip", "frame": 6, "t": float(TS[6]), "x": 12.0, "y": 22.0, "visibility": "visible"})
    assert r.status_code == 200
    assert client.post("/api/label", json={"stem": "clip", "frame": 8, "t": 0.27, "visibility": "visible"}).status_code == 422
    w = client.get("/api/window/w1").json()
    assert [(lab["frame"], lab["labeller"]) for lab in w["labels"]] == [(6, "me")]
    assert client.get("/api/windows").json()["labelled"] == 1
    assert len(L.read_log("clip", tmp_path)) == 3  # appended, nothing rewritten


def test_labelled_span_pads_the_labelled_stretch(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "video", lambda stem: SimpleNamespace(timestamps=TS))
    w = {"stem": "clip", "i0": 100, "i1": 300}
    assert predict.labelled_span(w, "me", tmp_path) is None
    for f in (150, 160, 400):  # 400 lies outside the window and is ignored
        L.append("clip", L.Label(f, 0.0, 1.0, 1.0, "visible", labeller="me"), tmp_path)
    s, e = predict.labelled_span(w, "me", tmp_path)
    assert (s, e) == (TS[150 - predict.LABEL_PAD], TS[160 + predict.LABEL_PAD])


def test_windows_skip_knockups_and_stay_apart():
    r = pd.DataFrame({
        "rally": [1, 2, 3, 4],
        "first_hit": [10.0, 30.0, 100.0, 50.0],
        "last_hit": [25.0, 50.0, 120.0, 60.0],
        "n_hits": [30, 40, 20, 12],
        "median_interval": [0.6, 0.30, 1.0, 0.7],  # rally 2 is a fast regular knock-up chain
    })
    picked = windows.choose(r)
    assert [p["rally"] for p in picked] == [1, 3]  # busiest real rally, then a far one with a different tempo
    assert all(p["end"] - p["start"] <= windows.WINDOW_S for p in picked)
    assert [p["rally"] for p in windows.choose(r, exclude=[(0.0, 40.0)])][0] == 3


def test_full_clip_chunks_tile_the_clip_with_context_on_both_sides():
    duration = 150.0
    spans = predict.chunks(duration, size=60.0, lead=1.0)
    kept = [(k0, k1) for _, k0, k1, _ in spans]
    assert kept[0][0] == 0.0 and kept[-1][1] == duration
    assert all(a[1] == b[0] for a, b in zip(kept, kept[1:]))  # no gap, no overlap
    for run_from, k0, k1, run_to in spans:
        assert run_from == max(0.0, k0 - 1.0) and run_to == min(duration, k1 + 1.0)
