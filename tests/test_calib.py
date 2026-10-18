from __future__ import annotations

import cv2
import numpy as np
import pytest

from squashvid.calib.coarse import coarse_align
from squashvid.calib.lines import match_lines, model_samples, refine_joint, response_maps
from squashvid.calib.register import motion_px, segment_and_smooth, to_frame, FramePose, _probe_points
from squashvid.calib.venue import Venue, VenueFrame, scan_width
from squashvid.geom import court
from squashvid.geom.camera import Camera

from tests.test_geometry import behind_glass_camera

TRUE = court.WSF.with_(width=6.15, length=9.85)
SHAPE = (576, 1024)


def render(cam: Camera, spec: court.CourtSpec, ss: int = 2, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """A synthetic frame ray-cast through the camera model, so lines and junctions sit exactly where it projects them.

    Surfaces get distinct greys (junctions are brightness steps), painted lines
    are 50 mm dark stripes, and each pixel averages ss x ss sub-rays.
    """
    h, w = shape
    ys, xs = np.mgrid[0 : h * ss, 0 : w * ss]
    uv = np.column_stack([(xs.ravel() + 0.5) / ss - 0.5, (ys.ravel() + 0.5) / ss - 0.5])
    rays, c = cam.rays(uv), cam.centre
    grey = np.full(len(uv), 200.0)
    best = np.full(len(uv), np.inf)
    hit = np.zeros((len(uv), 3))
    surf = np.full(len(uv), "", dtype=object)
    W, L = spec.width, spec.length
    for name, n, d, shade in (("floor", 2, 0.0, 150.0), ("front", 1, L, 205.0), ("left", 0, 0.0, 185.0), ("right", 0, W, 192.0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - c[n]) / rays[:, n]
        p = c + rays * t[:, None]
        ok = (t > 0) & (t < best) & (p[:, 0] >= -1e-9) & (p[:, 0] <= W + 1e-9) & (p[:, 1] <= L + 1e-9) & (p[:, 2] >= -1e-9) & (p[:, 2] <= 5.5)
        best[ok], hit[ok], surf[ok], grey[ok] = t[ok], p[ok], name, shade
    for line in court.court_lines(spec):
        if line.kind != "paint":
            continue
        on = surf == line.surface
        a, b = np.asarray(line.a, float), np.asarray(line.b, float)
        ab = b - a
        q = hit[on] - a
        s = np.clip(q @ ab / (ab @ ab), 0.0, 1.0)
        dist = np.linalg.norm(q - s[:, None] * ab, axis=1)
        idx = np.flatnonzero(on)[dist <= court.HALF_LINE]
        grey[idx] = 60.0
    img = grey.reshape(h, ss, w, ss).mean(axis=(1, 3))
    rng = np.random.default_rng(1)
    noisy = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)


def nudge(cam: Camera, yaw: float = 0.0, shift=(0.0, 0.0, 0.0)) -> Camera:
    R = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0] @ cam.R
    centre = cam.centre + np.asarray(shift)
    return cam.copy(rvec=cv2.Rodrigues(R)[0].reshape(3), tvec=-R @ centre)


@pytest.fixture(scope="module")
def scene():
    cam = behind_glass_camera()
    return cam, response_maps(render(cam, TRUE))


def test_refine_recovers_a_perturbed_pose(scene):
    cam, maps = scene
    samples = model_samples(spec=TRUE)
    start = nudge(cam, yaw=0.01, shift=(0.05, -0.05, 0.03))
    fit = refine_joint([start], [maps], samples, radii=(10, 6, 4, 3, 2), spec=TRUE)
    probe = _probe_points(Venue("t", TRUE, cam.f, cam.cx, cam.cy, cam.dist))
    assert motion_px(fit.cameras[0], cam, probe) < 0.3
    assert fit.frames[0].rms < 0.5


def test_coarse_align_recovers_a_pan_far_outside_refinement_range(scene):
    cam, maps = scene
    samples = model_samples(spec=TRUE)
    start = nudge(cam, yaw=0.12)  # ~55 px pan, like the phone being re-aimed
    probe = _probe_points(Venue("t", TRUE, cam.f, cam.cx, cam.cy, cam.dist))
    assert motion_px(start, cam, probe) > 40
    coarse = coarse_align(start, maps, samples, TRUE).camera
    fit = refine_joint([coarse], [maps], samples, radii=(8, 5, 3, 2), spec=TRUE)
    assert motion_px(fit.cameras[0], cam, probe) < 0.5


def test_coarse_align_recovers_a_hand_held_tilt(scene):
    cam, maps = scene
    samples = model_samples(spec=TRUE)
    R = cv2.Rodrigues(np.array([0.09, 0.0, 0.0]))[0] @ cam.R  # tilt about the camera's own horizontal axis
    start = cam.copy(rvec=cv2.Rodrigues(R)[0].reshape(3), tvec=-R @ cam.centre)
    probe = _probe_points(Venue("t", TRUE, cam.f, cam.cx, cam.cy, cam.dist))
    assert motion_px(start, cam, probe) > 40
    coarse = coarse_align(start, maps, samples, TRUE).camera
    fit = refine_joint([coarse], [maps], samples, radii=(8, 5, 3, 2), spec=TRUE)
    assert motion_px(fit.cameras[0], cam, probe) < 0.5


def test_width_scan_finds_a_narrow_court_that_a_wsf_start_cannot(scene):
    cam, maps = scene
    wsf_samples = model_samples(spec=court.WSF.with_(length=TRUE.length))
    start = refine_joint([cam], [maps], wsf_samples, radii=(6, 4, 3), spec=court.WSF.with_(length=TRUE.length)).cameras[0]
    frames = [VenueFrame("synthetic", 0.0, maps, start)]
    width, support = scan_width(frames, court.WSF.with_(length=TRUE.length), np.round(np.arange(5.95, 6.55, 0.05), 2))
    assert width == pytest.approx(TRUE.width, abs=0.05)
    assert support[width] > support[6.4]


def test_joint_fit_recovers_court_dimensions_and_focal(scene):
    cam, maps = scene
    second_cam = nudge(cam, yaw=-0.06)
    maps2 = response_maps(render(second_cam, TRUE))
    start_spec = TRUE.with_(width=6.12, length=9.75)
    samples = model_samples(spec=start_spec)
    starts = [c.copy(f=470.0) for c in (cam, second_cam)]
    fit = refine_joint(starts, [maps, maps2], samples, radii=(6, 4, 3, 2), free="pose+f", spec=start_spec, free_spec=("width", "length"))
    assert fit.spec.width == pytest.approx(TRUE.width, abs=0.02)
    assert fit.spec.length == pytest.approx(TRUE.length, abs=0.1)
    assert fit.cameras[0].f == pytest.approx(cam.f, rel=0.02)


def test_segmentation_splits_at_a_jump_and_never_smooths_across_it():
    cam = behind_glass_camera()
    venue = Venue("t", TRUE, cam.f, cam.cx, cam.cy, cam.dist)
    rng = np.random.default_rng(0)
    poses = []
    for i in range(60):
        c = nudge(cam, yaw=(0.1 if i >= 30 else 0.0) + rng.normal(0, 0.0004))
        poses.append(FramePose(i, i / 30, c.rvec, c.tvec, 0.4, 0.8, 1000, False))
    df = segment_and_smooth(to_frame(poses), venue)
    assert df.segment.nunique() == 2
    assert (df.segment.iloc[:30] == df.segment.iloc[0]).all() and (df.segment.iloc[30:] != df.segment.iloc[0]).all()
    # Smoothed poses on each side stay with their own segment's pose.
    pre, post = nudge(cam, 0.0), nudge(cam, 0.1)
    for i, ref in ((29, pre), (30, post)):
        row = df.iloc[i]
        c = venue.camera(row[["sr0", "sr1", "sr2"]].to_numpy(float), row[["st0", "st1", "st2"]].to_numpy(float))
        assert motion_px(c, ref, _probe_points(venue)) < 1.0


def test_venue_round_trips_through_json(tmp_path):
    v = Venue("court5", TRUE, 478.0, 512.0, 288.0, np.array([0.05, -0.03, 0, 0, 0]), 0.43, 17, {"k": 1}, (1024, 576))
    back = Venue.load(v.save(tmp_path / "venue.json"))
    assert back.spec == v.spec and back.f == v.f and np.allclose(back.dist, v.dist) and back.notes == v.notes
    assert back.size == (1024, 576) and back.seed is None
    seeded = Venue("other", TRUE, 1160.0, 959.5, 539.5, np.zeros(5), image_size=(1920, 1080), seed=(np.array([1.6, 0.0, 0.02]), np.array([-3.0, 1.2, 2.0])))
    again = Venue.load(seeded.save(tmp_path / "seeded.json"))
    assert np.allclose(again.seed[0], seeded.seed[0]) and np.allclose(again.seed[1], seeded.seed[1])
    assert np.allclose(again.resized(960, 540).seed[1], seeded.seed[1])


def test_venue_without_a_recorded_size_assumes_a_centred_principal_point():
    assert Venue("old", TRUE, 478.0, 512.0, 288.0, np.zeros(5)).size == (1024, 576)


def test_resized_venue_projects_where_the_resized_frame_shows_the_court():
    cam = behind_glass_camera()
    venue = Venue("t", TRUE, cam.f, cam.cx, cam.cy, cam.dist, image_size=(1024, 576))
    hd = venue.resized(1920, 1080)
    pts = np.array(list(court.landmarks(TRUE).values()))
    uv = venue.camera(cam.rvec, cam.tvec).project(pts)
    uv_hd = hd.camera(cam.rvec, cam.tvec).project(pts)
    s = 1920 / 1024
    assert np.allclose(uv_hd, (uv + 0.5) * s - 0.5, atol=1e-6)
    assert hd.pixel_scale == pytest.approx(s)
    with pytest.raises(ValueError):
        venue.resized(1920, 1200)


def test_fit_on_a_larger_frame_finds_the_same_pose():
    """The pixel settings scale with the frame: a 1080p render of the scene refines to the same pose."""
    cam = behind_glass_camera()
    hd_venue = Venue("t", TRUE, cam.f, cam.cx, cam.cy, cam.dist, image_size=(1024, 576)).resized(1920, 1080)
    hd_cam = hd_venue.camera(cam.rvec, cam.tvec)
    maps = response_maps(render(hd_cam, TRUE, shape=(1080, 1920)))
    assert maps.scale == pytest.approx(1920 / 1024)
    samples = model_samples(spec=TRUE)
    start = nudge(hd_cam, yaw=0.01, shift=(0.05, -0.05, 0.03))
    fit = refine_joint([start], [maps], samples, radii=(10, 6, 4, 3, 2), spec=TRUE)
    probe = _probe_points(hd_venue)
    assert motion_px(fit.cameras[0], hd_cam, probe) < 0.3 * maps.scale
    assert fit.frames[0].rms < 0.5 * maps.scale
    # Near lines are wider than the 1024-wide kernel at this size; an unscaled kernel loses them (~0.84).
    assert fit.frames[0].match_frac > 0.9
    # A coarse re-acquisition after a pan works at this size too.
    panned = nudge(hd_cam, yaw=0.12)
    coarse = coarse_align(panned, maps, samples, TRUE).camera
    fit = refine_joint([coarse], [maps], samples, radii=(8, 5, 3, 2), spec=TRUE)
    assert motion_px(fit.cameras[0], hd_cam, probe) < 0.5 * maps.scale


def test_a_wide_line_is_located_at_its_centre_not_its_highest_sample():
    """A line several px wide gives a flat-topped response; noise decides which sample is highest."""
    resp = np.zeros((40, 60), np.float32)
    resp[:, 20:30] = 60.0  # paint spans x = 20..29, centre 24.5
    resp[:, 28] = 61.0  # the highest sample sits near one edge
    uv = np.array([[25.0, 20.0]])
    off, peak, ok = match_lines(resp, uv, np.array([[1.0, 0.0]]), radius=5.0)
    assert ok[0] and peak[0] == pytest.approx(61.0)
    assert 25.0 + off[0] == pytest.approx(24.5, abs=0.1)


def test_a_line_cut_off_by_the_search_window_is_not_matched():
    resp = np.zeros((40, 60), np.float32)
    resp[:, 10:50] = 60.0  # wider than the ±2·radius profile: no centre can be measured
    off, peak, ok = match_lines(resp, np.array([[30.0, 20.0]]), np.array([[1.0, 0.0]]), radius=3.0)
    assert not ok[0]
