"""3D ballistic model: gravity, drag, and recovery of a synthetic flight from its reprojection."""

import numpy as np

from squashvid.balltrack import traj3d
from tests.test_balltrack_core import _camera


def test_free_fall_matches_closed_form_without_drag():
    t = np.linspace(0.0, 0.5, 6)
    pos = traj3d.simulate(np.array([1.0, 2.0, 3.0]), np.array([4.0, 0.0, 0.0]), t, drag=0.0, substeps=8)
    assert np.allclose(pos[:, 2], 3.0 - 0.5 * traj3d.G * t**2, atol=0.02)
    assert np.allclose(pos[:, 0], 1.0 + 4.0 * t, atol=1e-6)


def test_drag_slows_the_ball():
    t = np.array([0.0, 0.5])
    p0, v0 = np.array([0.0, 0.0, 10.0]), np.array([0.0, 30.0, 0.0])
    a = traj3d.simulate(p0, v0, t, drag=0.0)
    b = traj3d.simulate(p0, v0, t, drag=0.016)
    assert b[1, 1] < a[1, 1]


def test_anchored_fit_recovers_the_flight_from_pixels():
    cam = _camera()
    fp = traj3d.FitParams()
    t = np.arange(12) / 30.0
    truth = traj3d.simulate(np.array([2.5, 3.0, 1.2]), np.array([2.0, 12.0, 3.0]), t, fp.drag)
    uv = cam.project(truth)
    frames = np.arange(12)
    T = t

    def cam_of(_):
        return cam

    res = traj3d._fit_segment(cam_of, frames, T, uv, [(0.0, truth[0])], fp)
    assert res is not None
    theta, _ = res
    pos = traj3d.simulate(theta[:3], theta[3:], t, fp.drag)
    assert np.abs(pos - truth).max() < 0.15
