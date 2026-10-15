"""The ball path the viewer draws: curves through tracking gaps, corners at impacts, a 3D point on every frame."""

import numpy as np
import pandas as pd

from squashvid.viewer import ballpath as B

FPS = 30.0


def ts(seconds=4.0):
    return np.arange(0, seconds, 1 / FPS)


def test_a_gap_on_a_straight_flight_is_filled_on_the_line():
    t = ts()
    uv = np.c_[100 + 300 * t, 200 + 60 * t]
    uv[40:50] = np.nan
    path, kind = B.image_path(uv, np.zeros(len(t), int), t, [(0, len(t) - 1)])
    assert (kind[40:50] == B.FILLED).all() and (kind[:40] == B.MEASURED).all()
    assert np.allclose(path[40:50], np.c_[100 + 300 * t[40:50], 200 + 60 * t[40:50]], atol=1e-6)


def test_a_gap_on_a_curve_follows_the_arc_not_the_chord():
    t = ts()
    uv = np.c_[100 + 300 * t, 600 - 400 * t + 200 * t**2]  # a dropping arc
    true = uv.copy()
    uv[45:57] = np.nan
    path, _ = B.image_path(uv, np.zeros(len(t), int), t, [(0, len(t) - 1)])
    chord = np.linspace(true[44], true[57], 14)[1:-1]
    assert np.abs(path[45:57] - true[45:57]).max() < 0.3 * np.abs(chord - true[45:57]).max()


def test_an_impact_keeps_its_corner():
    t = ts(2.0)
    hit = 30
    u = np.where(np.arange(len(t)) <= hit, 100 + 400 * t, 100 + 400 * t[hit] - 400 * (t - t[hit]))  # bounces back
    uv = np.c_[u, np.full(len(t), 300.0)]
    uv[hit - 3 : hit] = np.nan  # unseen just before the wall
    events = pd.DataFrame({"t": [t[hit]], "u": [u[hit]], "v": [300.0]})
    path, kind = B.image_path(uv, np.zeros(len(t), int), t, [(0, len(t) - 1)], events)
    assert kind[hit] == B.MEASURED
    assert np.allclose(path[hit - 3 : hit, 0], u[hit - 3 : hit], atol=1e-6)  # straight into the corner, no rounding off


def test_long_gaps_and_frames_outside_rallies_stay_empty():
    t = ts()
    uv = np.c_[100 + 300 * t, 200 + 0 * t]
    uv[20:20 + int(B.MAX_GAP_S * FPS) + 5] = np.nan
    path, kind = B.image_path(uv, np.zeros(len(t), int), t, [(0, 80)])
    assert (kind[20:20 + int(B.MAX_GAP_S * FPS) + 5] == 0).all()
    assert (kind[81:] == 0).all() and np.isnan(path[81:]).all()


def test_rally_spans_are_padded_and_clipped():
    t = ts(10.0)
    spans = B.rally_spans([{"start_t": 0.1, "end_t": 2.0}, {"start_t": 5.0, "end_t": 9.9}], t)
    assert spans[0][0] == 0 and abs(t[spans[0][1]] - (2.0 + B.PAD_S)) < 1 / FPS
    assert spans[1][1] == len(t) - 1
    assert B.rally_spans([], t) == [(0, len(t) - 1)]


def test_3d_fill_never_overshoots_its_knots_and_lands_on_the_camera_ray():
    t = ts(2.0)
    xyz = np.full((len(t), 3), np.nan)
    xyz[:20] = np.c_[np.full(20, 3.0), 2 + 5 * t[:20], 1.0 + 0 * t[:20]]
    xyz[40:] = np.c_[np.full(len(t) - 40, 3.0), 2 + 5 * t[40:], 1.0 + 0 * t[40:]]
    path, kind = B.path_3d(xyz, t, [(0, len(t) - 1)])
    assert (kind[20:40] == B.FILLED).all() and np.allclose(path[20:40, 2], 1.0) and np.allclose(path[20:40, 1], 2 + 5 * t[20:40])
    centre = np.array([3.2, -2.0, 2.0])
    uv = np.zeros((len(t), 2))

    def camera_of(fr):  # a ray through a point 0.4 m above the interpolated one
        target = np.array([3.0, 2 + 5 * t[fr], 1.4])
        d = target - centre
        return centre, d / np.linalg.norm(d)

    snapped, _ = B.path_3d(xyz, t, [(0, len(t) - 1)], uv=uv, camera_of=camera_of)
    for fr in range(20, 40):
        c, d = camera_of(fr)
        off = snapped[fr] - c
        assert np.linalg.norm(off - np.dot(off, d) * d) < 1e-9  # on the ray
    assert (snapped[20:40, 2] > 1.0).all()  # moved up toward what the camera saw
