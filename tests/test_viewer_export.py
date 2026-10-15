import json

import numpy as np
import pandas as pd

from squashvid.geom import court
from squashvid.viewer import export as X
from squashvid.viewer.proxy import compare_pts

VFR = np.array([0.0, 0.0333, 0.0667, 0.1333, 0.1667, 0.2, 0.5, 0.5333])  # a dropped-frame gap and a long gap


def read_array(folder, meta, name):
    info = meta["arrays"][name]
    a = np.fromfile(folder / info["file"], dtype=np.dtype(info["dtype"]).newbyteorder("<")).reshape(info["shape"])
    return X.dequantise(a, info["scale"]) if info["missing"] is not None and info["dtype"] == "int16" else a


def test_frame_of_time_on_vfr_timestamps():
    f = X.frame_of_time(VFR, [0.0, 0.05, 0.0667, 0.1, 0.1333, 0.4, 0.5, 9.0, -1.0])
    assert f.tolist() == [0, 1, 2, 2, 3, 5, 6, 7, 0]
    # every frame's own PTS maps back to itself
    assert X.frame_of_time(VFR, VFR).tolist() == list(range(len(VFR)))


def test_quantise_roundtrip_and_sentinel():
    a = np.array([0.0, 1.2345, np.nan, np.inf, 1e9])
    q = X.quantise(a, 1000)
    back = X.dequantise(q, 1000)
    assert abs(back[1] - 1.2345) < 1e-3 and back[0] == 0
    assert np.isnan(back[2:]).all()


def test_export_with_only_timestamps(tmp_path):
    (tmp_path / "clip").mkdir()
    out = X.export("clip", root=tmp_path, timestamps=VFR)
    meta = json.loads((tmp_path / "clip" / "viewer" / "meta.json").read_text())
    assert meta["schema_version"] == X.SCHEMA_VERSION and meta["n_frames"] == len(VFR)
    assert meta["rallies"] == [] and meta["events"] == [] and meta["stats"] is None and meta["insights"] is None
    assert not any(v for k, v in out["sources"].items() if k != "venue" and v)
    assert "cam" not in meta["arrays"] and "ball_xyz" not in meta["arrays"]
    folder = tmp_path / "clip" / "viewer"
    assert np.allclose(np.fromfile(folder / "t.bin", "<f8"), VFR)


def test_export_reads_optional_inputs(tmp_path):
    w = tmp_path / "clip"
    (w / "analysis").mkdir(parents=True)
    n = len(VFR)
    rows = []
    for f in range(n):
        for p in "AB":
            rows.append(dict(frame=f, player=p, detected=p == "A", sx=1.0 + f * 0.1, sy=2.0, x1=10.0, y1=20.0, x2=30.0, y2=90.0,
                             confidence=0.9, kpts=np.arange(34, dtype=float), kscore=np.full(17, 0.5), interpolated=False))
    pd.DataFrame(rows).to_parquet(w / "players.parquet")
    (w / "ball").mkdir()
    pd.DataFrame({"frame": [1, 2], "t": VFR[1:3], "x": [100.0, np.nan], "y": [50.0, np.nan], "score": [0.9, 0.0], "visible": [True, False]}).to_parquet(
        w / "ball" / "blurball@576x1024-coreml.parquet")
    pd.DataFrame({"t": [0.1, 0.2], "kind": ["front_wall", "floor"], "x": [1.0, 2.0], "y": [9.75, 3.0], "z": [1.5, 0.0], "player": ["A", "B"],
                  "sigma_a": [0.1, 0.2], "sigma_b": [0.1, 0.2]}).to_parquet(w / "ball_events.parquet")
    pd.DataFrame({"rally": [1], "start_t": [0.0], "end_t": [0.3]}).to_parquet(w / "analysis" / "rallies.parquet")
    (w / "analysis" / "stats.json").write_text(json.dumps({"A": {"shots": 3}}))
    X.export("clip", root=tmp_path, timestamps=VFR)
    folder = w / "viewer"
    meta = json.loads((folder / "meta.json").read_text())
    pos = read_array(folder, meta, "A_pos")
    assert pos.shape == (n, 2) and np.allclose(pos[:, 0], 1.0 + np.arange(n) * 0.1, atol=1e-3)
    assert read_array(folder, meta, "A_status").tolist() == [2] * n and read_array(folder, meta, "B_status").tolist() == [1] * n
    assert read_array(folder, meta, "ball_state").tolist()[:3] == [0, 1, 0]
    ev = meta["events"]
    assert [e["kind"] for e in ev] == ["front_wall", "floor"] and ev[0]["frame"] == 2  # t=0.1 shows frame 2 (PTS 0.0667)
    assert meta["rallies"][0]["start_t"] == 0.0 and meta["stats"] == {"A": {"shots": 3}}


def test_court_geometry_matches_court_module():
    spec = court.WSF.with_(width=6.15, box_width=1.61)
    payload = X.court_payload(spec)
    lines = {ln["name"]: ln for ln in payload["lines"]}
    expect = {ln.name: ln for ln in court.court_lines(spec)}
    assert set(lines) == set(expect)
    for name, ln in expect.items():
        pts = np.asarray(lines[name]["pts"])
        assert np.allclose(pts[0], ln.a, atol=1e-3) and np.allclose(pts[-1], ln.b, atol=1e-3)
        assert lines[name]["surface"] == ln.surface
    assert payload["spec"]["width"] == 6.15
    assert abs(payload["derived"]["short_y"] - spec.short_y) < 1e-12


def test_compare_pts_flags_missing_and_shifted_frames():
    orig = np.cumsum(np.full(100, 1 / 30))
    assert compare_pts(orig, orig)["ok"]
    # undecodable tail packets after the last proxy frame are allowed
    assert compare_pts(orig[:90], orig)["ok"]
    assert compare_pts(orig[:90], orig)["n_original_tail_not_in_proxy"] == 10
    # a dropped frame in the middle shifts everything after it
    assert not compare_pts(np.delete(orig, 40), orig)["ok"]
    # a constant offset (audio would drift against it) is caught
    assert not compare_pts(orig + 0.05, orig)["ok"]
    assert not compare_pts(np.concatenate([orig, [9.0]]), orig)["ok"]
