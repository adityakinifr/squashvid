from __future__ import annotations

import cv2
import numpy as np
import pytest

from squashvid.geom import court
from squashvid.geom.camera import Camera, plane, solve_pnp, solve_pnp_focal

SPEC = court.WSF.with_(width=6.15, length=9.9)


def behind_glass_camera(f: float = 478.0, dist=(0.06, -0.04, 0.0, 0.0, 0.0)) -> Camera:
    """A camera like the pilot footage: ~1.8 m behind the back wall, chest height, looking down the court."""
    centre = np.array([2.9, -1.8, 1.4])
    target = np.array([3.0, 6.0, 0.4])
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    rvec = cv2.Rodrigues(R)[0].reshape(3)
    return Camera(f, 512.0, 288.0, rvec, -R @ centre, np.asarray(dist, np.float64))


def test_spec_lines_are_symmetric_and_sit_on_their_surfaces():
    s = SPEC
    assert s.box_inner_left_x + s.box_inner_right_x == pytest.approx(s.width)
    assert s.t_point[0] == pytest.approx(s.width / 2)
    assert s.side_out_z(0.0) == pytest.approx(s.back_out_z)
    assert s.side_out_z(s.length) == pytest.approx(s.front_out_z)
    for line in court.court_lines(s):
        n, d = plane(line.surface, s)
        assert np.dot(n, line.a) == pytest.approx(d)
        assert np.dot(n, line.b) == pytest.approx(d)


def test_landmarks_follow_the_spec():
    narrow = court.landmarks(SPEC)
    wsf = court.landmarks(court.WSF)
    assert narrow["short_x_right_wall"][0] == pytest.approx(SPEC.width)
    assert narrow["front_floor_left"][1] == pytest.approx(SPEC.length)
    assert wsf["t_point"][0] - narrow["t_point"][0] == pytest.approx((court.WSF.width - SPEC.width) / 2)


@pytest.mark.parametrize("surface", ["floor", "front", "left", "right"])
def test_pixel_to_plane_round_trip(surface):
    cam = behind_glass_camera()
    rng = np.random.default_rng(0)
    if surface == "floor":
        pts = np.column_stack([rng.uniform(0.3, SPEC.width - 0.3, 50), rng.uniform(0.5, SPEC.length - 0.5, 50), np.zeros(50)])
    elif surface == "front":
        pts = np.column_stack([rng.uniform(0.3, SPEC.width - 0.3, 50), np.full(50, SPEC.length), rng.uniform(0.2, 4.0, 50)])
    else:
        x = 0.0 if surface == "left" else SPEC.width
        pts = np.column_stack([np.full(50, x), rng.uniform(2.0, SPEC.length - 0.2, 50), rng.uniform(0.2, 3.0, 50)])
    uv = cam.project(pts)
    back = cam.to_plane(uv, surface, spec=SPEC)
    assert np.max(np.linalg.norm(back - pts, axis=1)) < 1e-6


def test_plane_offset_intersects_a_raised_plane():
    cam = behind_glass_camera()
    p = np.array([[3.0, 4.0, 0.53]])
    back = cam.to_plane(cam.project(p), "floor", offset=0.53, spec=SPEC)
    assert np.allclose(back, p, atol=1e-6)


def test_rays_behind_the_camera_are_nan():
    cam = behind_glass_camera()
    # A pixel near the top of the frame looks above the horizon, so it never meets the floor ahead.
    out = cam.to_plane(np.array([[512.0, 0.0]]), "floor", spec=SPEC)
    assert np.isnan(out).all()


@pytest.mark.parametrize("surface", ["floor", "front", "left", "right"])
def test_plane_homography_matches_projection_without_distortion(surface):
    cam = behind_glass_camera(dist=(0, 0, 0, 0, 0))
    H = cam.plane_homography(surface, SPEC)
    if surface == "floor":
        pts2, pts3 = np.array([[1.0, 2.0], [5.0, 7.0]]), np.array([[1.0, 2.0, 0.0], [5.0, 7.0, 0.0]])
    elif surface == "front":
        pts2, pts3 = np.array([[1.0, 0.5], [5.0, 3.0]]), np.array([[1.0, SPEC.length, 0.5], [5.0, SPEC.length, 3.0]])
    elif surface == "left":
        pts2, pts3 = np.array([[3.0, 0.5], [8.0, 2.0]]), np.array([[0.0, 3.0, 0.5], [0.0, 8.0, 2.0]])
    else:
        pts2, pts3 = np.array([[3.0, 0.5], [8.0, 2.0]]), np.array([[SPEC.width, 3.0, 0.5], [SPEC.width, 8.0, 2.0]])
    h = np.column_stack([pts2, np.ones(len(pts2))]) @ H.T
    assert np.allclose(h[:, :2] / h[:, 2:], cam.project(pts3), atol=1e-6)


def test_camera_dict_round_trip():
    cam = behind_glass_camera()
    back = Camera.from_dict(cam.to_dict())
    pts = np.array([[1.0, 2.0, 0.0], [3.0, 9.0, 1.0]])
    assert np.allclose(back.project(pts), cam.project(pts))


def test_pnp_recovers_focal_and_pose_from_landmarks():
    cam = behind_glass_camera(dist=(0, 0, 0, 0, 0))
    lm = court.landmarks(SPEC)
    world = np.array(list(lm.values()))
    image = cam.project(world)
    est, rms = solve_pnp_focal(world, image, cam.cx, cam.cy, f_range=(400.0, 560.0), step=1.0)
    assert rms < 0.5
    assert est.f == pytest.approx(cam.f, abs=1.5)
    assert np.allclose(est.centre, cam.centre, atol=0.02)
    exact, rms_exact = solve_pnp(world, image, cam.f, cam.cx, cam.cy)
    assert rms_exact < 1e-3
