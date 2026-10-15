"""Pure parts of the ball-track stage: kink detection, linking, gap filling, ray-plane mapping, bounce physics."""

import cv2
import numpy as np
import pandas as pd

from squashvid.balltrack import kinks, link, world
from squashvid.geom import court
from squashvid.geom.camera import Camera


def _camera() -> Camera:
    """Behind the back wall, 2 m up, looking down the court (+y)."""
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    C = np.array([3.2, -2.0, 2.0])
    return Camera(f=1500.0, cx=960.0, cy=540.0, rvec=cv2.Rodrigues(R)[0].reshape(3), tvec=-R @ C)


def _flight(frames, p0, v, a=(0.0, 0.3)):
    f = np.asarray(frames, float) - frames[0]
    return p0[0] + v[0] * f + 0.5 * a[0] * f**2, p0[1] + v[1] * f + 0.5 * a[1] * f**2


def _cands(frames, u, v, peak=0.9):
    return pd.DataFrame({"frame": frames, "x": u, "y": v, "peak": peak, "mass": 20.0})


def test_ray_plane_recovers_projected_points():
    cam = _camera()
    spec = court.WSF
    for kind, point in (("floor", (2.0, 5.0, 0.02)), ("front_wall", (3.0, spec.length, 2.0)), ("side_wall_left", (0.0, 4.0, 1.5))):
        uv = cam.project(np.array([point]))
        got = world.locate(cam, uv, kind, spec)
        assert np.allclose(got.reshape(-1)[:3], point, atol=1e-3), kind


def test_ray_plane_behind_camera_is_nan():
    cam = _camera()
    pts = world.ray_plane(cam.centre, cam.rays(np.array([[960.0, 540.0]])), 1, -10.0)
    assert np.isnan(pts).all()


def test_smooth_flight_has_no_kink():
    f = np.arange(0, 40)
    u, v = _flight(f, (300.0, 200.0), (9.0, -4.0))
    assert kinks.detect(f, u, v) == []


def test_velocity_reversal_is_found_at_the_right_frame():
    f = np.arange(0, 40)
    u1, v1 = _flight(f[:20], (300.0, 200.0), (9.0, 6.0))
    u2, v2 = _flight(f[20:], (u1[-1] + 9.0, v1[-1]), (9.0, -7.0))
    found = kinks.detect(f, np.r_[u1, u2], np.r_[v1, v2])
    assert found, "an obvious bounce must be detected"
    best = max(found, key=lambda k: k.gain)
    assert abs(best.frame - 19.5) <= 1.5


def test_linking_follows_the_ball_through_clutter_and_gaps():
    f = np.arange(0, 60)
    u, v = _flight(f, (200.0, 300.0), (10.0, -3.0))
    keep = ~np.isin(f, [20, 21, 22, 40])  # missed frames
    ball = _cands(f[keep], u[keep], v[keep])
    rng = np.random.default_rng(0)
    clutter = _cands(rng.integers(0, 60, 40), rng.uniform(100, 900, 40), rng.uniform(100, 600, 40), peak=0.6)
    static = _cands(np.arange(0, 60), np.full(60, 700.0), np.full(60, 100.0), peak=0.8)
    cands = pd.concat([ball, clutter, static], ignore_index=True).sort_values(["frame", "peak"], ascending=[True, False])
    out = link.link(cands, np.arange(60) / 30.0)
    ids = out[np.hypot(out.x - np.interp(out.frame, f, u), out.y - np.interp(out.frame, f, v)) < 3]
    assert len(ids) >= 0.9 * keep.sum()
    assert ids.track_id.nunique() == 1
    wrong = out[np.hypot(out.x - np.interp(out.frame, f, u), out.y - np.interp(out.frame, f, v)) >= 3]
    assert len(wrong) <= 0.1 * len(out)
    assert not ((out.x == 700.0) & (out.y == 100.0)).any(), "a blob that never moves is not the ball"


def test_gap_fill_never_smooths_across_an_impact():
    f = np.arange(0, 50)
    u1, v1 = _flight(f[:25], (300.0, 200.0), (9.0, 6.0))
    u2, v2 = _flight(f[25:], (u1[-1] + 9.0, v1[-1]), (9.0, -8.0))
    u, v = np.r_[u1, u2], np.r_[v1, v2]
    gap = (f >= 23) & (f <= 27)  # the impact sits in the missing frames
    kept = ~gap
    track = pd.DataFrame({"frame": f[kept], "x": u[kept], "y": v[kept], "peak": 0.9, "state": "detected", "track_id": 0})
    res = link.fill_gaps(track, 50)
    fr = res.frames.set_index("frame")
    filled = fr[fr.state == "interpolated"]
    assert len(filled) > 0
    # every filled point lies on one of the two true flights, not on a smooth curve between them
    d1 = np.hypot(filled.u - np.interp(filled.index, f[:25], u1, right=np.nan), filled.v - np.interp(filled.index, f[:25], v1, right=np.nan))
    d2 = np.hypot(filled.u - np.interp(filled.index, f[25:], u2, left=np.nan), filled.v - np.interp(filled.index, f[25:], v2, left=np.nan))
    assert (np.fmin(d1, d2) < 12).all()
    assert fr.segment_id.loc[20] != fr.segment_id.loc[30], "the impact must start a new segment"
    assert len(res.kinks) >= 1


def test_bounce_residual_prefers_the_right_plane():
    cam = _camera()
    point = np.array([3.0, 5.0, 0.02])
    p0, p1 = cam.project(np.array([point + [0.0, -0.3, 0.25]]))[0], cam.project(np.array([point]))[0]
    p2 = cam.project(np.array([point + [0.0, 0.2, 0.15]]))[0]
    vb, va = p1 - p0, p2 - p1
    on_floor, _ = world.bounce_residual(cam, point, "floor", vb, va)
    # the same kink explained as a front-wall bounce would need the ball far out of the right place
    on_wall, _ = world.bounce_residual(cam, np.array([3.0, court.WSF.length, 0.5]), "front_wall", vb, va)
    assert np.isfinite(on_floor)
    assert on_floor <= on_wall + 1e-9 or on_floor < 1.0
