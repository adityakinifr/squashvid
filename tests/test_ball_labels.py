"""Ball label log: validation, append-only replay, clearing and crash tolerance."""

import json

import pytest

from squashvid.ball import labels as L


def test_out_drops_position_and_visible_needs_one():
    lab = L.Label(5, 0.17, 10.0, 20.0, "out")
    assert lab.x is None and lab.y is None
    with pytest.raises(ValueError):
        L.Label(5, 0.17, None, None, "visible")
    with pytest.raises(ValueError):
        L.Label(5, 0.17, 1.0, 2.0, "gone")
    with pytest.raises(ValueError):
        L.Label(5, 0.17, 1.0, 2.0, "visible", source="robot")


def test_latest_record_per_frame_wins_and_cleared_deletes(tmp_path):
    L.append("clip", L.Label(1, 0.03, 10.0, 10.0, "visible"), tmp_path)
    L.append("clip", L.Label(2, 0.07, 20.0, 20.0, "occluded"), tmp_path)
    L.append("clip", L.Label(1, 0.03, 11.0, 12.0, "visible", event="racket"), tmp_path)
    L.append("clip", L.Label(2, 0.07, None, None, "out", source="cleared"), tmp_path)
    L.append("clip", L.Label(3, 0.10, None, None, "out"), tmp_path)
    df = L.load("clip", tmp_path)
    assert df.frame.tolist() == [1, 3]
    assert (df.x[0], df.y[0], df.event[0]) == (11.0, 12.0, "racket")
    assert len(L.read_log("clip", tmp_path)) == 5  # history kept


def test_torn_final_line_is_skipped(tmp_path):
    L.append("clip", L.Label(1, 0.03, 10.0, 10.0, "visible"), tmp_path)
    with L.label_path("clip", tmp_path).open("a") as fh:
        fh.write('{"frame": 2, "t": 0.0')
    assert L.load("clip", tmp_path).frame.tolist() == [1]


def test_filter_by_labeller_and_load_all(tmp_path):
    L.append("a", L.Label(1, 0.0, 1.0, 1.0, "visible", labeller="me"), tmp_path)
    L.append("a", L.Label(2, 0.0, 1.0, 1.0, "visible", labeller="you"), tmp_path)
    L.append("b", L.Label(7, 0.0, None, None, "out", labeller="me"), tmp_path)
    assert L.load("a", tmp_path, labeller="me").frame.tolist() == [1]
    both = L.load_all(tmp_path, labeller="me")
    assert sorted(zip(both.stem, both.frame)) == [("a", 1), ("b", 7)]


def test_records_are_plain_json(tmp_path):
    L.append("clip", L.Label(4, 0.13, 3.5, 4.5, "visible", source="accepted-prelabel", prelabel="blurball@default"), tmp_path)
    rec = json.loads(L.label_path("clip", tmp_path).read_text())
    assert rec["source"] == "accepted-prelabel" and rec["prelabel"] == "blurball@default" and rec["frame"] == 4
