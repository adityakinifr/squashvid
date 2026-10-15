"""Foot-point ladder, court covariance and court-space smoothing, without video."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from squashvid.geom import court
from squashvid.geom.camera import Camera
from squashvid.players.footpoint import FootInputs, choose, court_point
from squashvid.players.geometry import ANKLE_Z, floor_xy
from squashvid.players.pose import L_ANK, L_HIP, R_ANK, R_HIP
from squashvid.players.smooth import rts_smooth
from squashvid.players.tracklets import is_part

SPEC = court.WSF.with_(width=6.147, length=9.94)
BOX = np.array([400.0, 200.0, 460.0, 380.0])


def behind_glass_camera() -> Camera:
    """A camera like the pilot footage: ~1.8 m behind the back wall at chest height, looking down the court."""
    centre, target = np.array([2.9, -1.8, 1.4]), np.array([3.0, 6.0, 0.4])
    z = (target - centre) / np.linalg.norm(target - centre)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    return Camera(478.0, 512.0, 288.0, cv2.Rodrigues(R)[0].reshape(3), -R @ centre)


def pose(ankle_score: float = 0.9, hip_score: float = 0.9, ankles=((418.0, 372.0), (444.0, 374.0))) -> tuple[np.ndarray, np.ndarray]:
    kp, sc = np.zeros((17, 2)), np.full(17, 0.9)
    kp[L_ANK], kp[R_ANK] = ankles
    kp[L_HIP], kp[R_HIP] = (420.0, 290.0), (440.0, 290.0)
    sc[[L_ANK, R_ANK]] = ankle_score
    sc[[L_HIP, R_HIP]] = hip_score
    return kp, sc


def test_confident_ankles_win():
    kp, sc = pose()
    src, uv, z, truncated = choose(FootInputs(BOX, kp, sc))
    assert src == "ankle" and z == ANKLE_Z and not truncated
    assert uv == pytest.approx([431.0, 373.0])


@pytest.mark.parametrize(
    "inputs",
    [
        FootInputs(BOX, *pose(ankle_score=0.2)),  # ankles not confident
        FootInputs(BOX, *pose(ankles=((418.0, 250.0), (444.0, 255.0)))),  # "ankles" in the upper body
        FootInputs(BOX, *pose(ankles=((600.0, 372.0), (620.0, 374.0)))),  # pose locked onto someone else
        FootInputs(BOX, *pose(), ankle_on_hardware=True),  # ankle keypoints painted on a hinge plate
        FootInputs(BOX, None, None),  # no pose at all
    ],
)
def test_untrusted_ankles_fall_back_to_the_box_bottom(inputs):
    src, uv, z, truncated = choose(inputs)
    assert src == "box" and z == 0.0 and not truncated
    assert uv == pytest.approx([430.0, 380.0])


@pytest.mark.parametrize(
    "box, bottom_on_hardware",
    [(np.array([400.0, 400.0, 460.0, 575.0]), 0.0), (BOX, 0.6)],  # frame bottom; stops on the handle
)
def test_truncated_box_falls_to_the_hip_plane(box, bottom_on_hardware):
    kp, sc = pose(ankle_score=0.1)
    src, uv, z, truncated = choose(FootInputs(box, kp, sc, bottom_on_hardware=bottom_on_hardware), hip_z=0.64)
    assert truncated and src == "hip" and z == 0.64
    assert uv == pytest.approx([430.0, 290.0])


def test_nothing_usable_means_no_point():
    kp, sc = pose(ankle_score=0.1, hip_score=0.1)
    src, uv, z, truncated = choose(FootInputs(BOX, kp, sc, bottom_on_hardware=0.6))
    assert src == "none" and uv is None and np.isnan(z) and truncated


def test_ankle_plane_puts_the_foot_nearer_the_camera_than_the_floor_reading():
    """Reading an ankle pixel on the floor would place the player too far up the court."""
    cam = behind_glass_camera()
    uv = cam.project(np.array([[3.0, 5.0, ANKLE_Z]]))
    on_ankle_plane = floor_xy(cam, uv, SPEC, ANKLE_Z)[0]
    on_floor = floor_xy(cam, uv, SPEC, 0.0)[0]
    assert on_ankle_plane == pytest.approx([3.0, 5.0], abs=1e-6)
    assert on_floor[1] > on_ankle_plane[1] + 0.1


def test_along_court_uncertainty_grows_toward_the_front_wall():
    cam = behind_glass_camera()
    sd_y = []
    for y in (2.0, 5.0, 8.5):
        uv = cam.project(np.array([[3.0, y, ANKLE_Z]]))[0]
        xy, R = court_point(cam, SPEC, "ankle", uv, ANKLE_Z, 40.0)
        assert xy == pytest.approx([3.0, y], abs=1e-6)
        sd_y.append(np.sqrt(R[1, 1]))
    assert sd_y[0] < sd_y[1] < sd_y[2]
    assert sd_y[2] > 3 * sd_y[0]


def test_smoother_bridges_a_gap_and_gates_a_jump():
    t = np.arange(90) / 30
    truth = np.column_stack([1.0 + 1.5 * t, np.full(90, 4.0)])
    z = truth + np.random.default_rng(1).normal(0, 0.05, truth.shape)
    z[40:50] = np.nan  # player hidden behind the other
    z[70] = [5.5, 8.0]  # one wrong foot point, 4 m off
    R = np.tile(np.eye(2) * 0.05**2, (90, 1, 1))
    out = rts_smooth(t, z, R)
    assert out.gated[70] and not out.used[70]
    assert not out.used[40:50].any()
    assert np.abs(out.xy - truth).max() < 0.15
    assert out.vel[45, 0] == pytest.approx(1.5, abs=0.3)


def test_partial_box_is_dropped_only_when_cut_by_hardware_or_deeply_nested():
    near = np.array([300.0, 150.0, 400.0, 450.0])
    torso = np.array([330.0, 150.0, 430.0, 300.0])  # 70 % inside, ends on a hinge plate
    nested = np.array([310.0, 160.0, 390.0, 320.0])  # wholly inside: a duplicate piece
    far_player = np.array([350.0, 200.0, 440.0, 330.0])  # behind the near one, half his box beside him
    assert is_part(torso, near, on_hardware=True)
    assert not is_part(torso, near, on_hardware=False)
    assert is_part(nested, near, on_hardware=False)
    assert not is_part(far_player, near, on_hardware=False)


@pytest.mark.parametrize("y, dropped", [(-0.8, True), (0.3, False)])
def test_person_cut_by_the_frame_bottom_is_dropped_only_behind_the_back_wall(y, dropped):
    """Someone passing right in front of the phone and a player close to the back wall both
    lose their feet below the frame; the hip plane tells them apart."""
    import pandas as pd

    from squashvid.players.run import behind_back_wall

    cam, hip_z = behind_glass_camera(), 0.64
    hip = cam.project(np.array([[3.0, y, hip_z]]))[0]
    assert 0 < hip[1] < 576
    kp, sc = pose(ankle_score=0.1)
    kp[L_HIP], kp[R_HIP] = hip - [10.0, 0.0], hip + [10.0, 0.0]
    box = np.array([hip[0] - 60.0, max(hip[1] - 250.0, 0.0), hip[0] + 60.0, 575.0])
    src, uv, z, truncated = choose(FootInputs(box, kp, sc), hip_z)
    assert src == "hip" and truncated
    xy, _ = court_point(cam, SPEC, src, uv, z, 120.0)
    assert xy[1] == pytest.approx(y, abs=1e-6)
    assert behind_back_wall(pd.DataFrame({"y": [xy[1]], "src": [src]}))[0] == dropped


# Clip 3 (IMG_4090) player B, 20.42-21.58 s: court x, y, sd x, sd y, cov xy per frame (None: no point).
# At 20.85 s a box bottom on a partial box reads 0.5 m short with a tighter covariance than the ankles.
RECORDED = [
    (3.287, 3.149, 0.037, 0.175, 0.0030), (3.325, 3.073, 0.037, 0.171, 0.0032), (3.357, 2.913, 0.037, 0.162, 0.0031),
    (3.397, 2.923, 0.038, 0.163, 0.0034), (3.419, 2.881, 0.038, 0.161, 0.0034), (3.431, 2.822, 0.038, 0.158, 0.0034),
    (3.436, 2.746, 0.037, 0.154, 0.0034), (3.441, 2.689, 0.037, 0.151, 0.0033), (3.453, 2.607, 0.037, 0.147, 0.0032),
    (3.491, 2.589, 0.037, 0.147, 0.0034), (3.556, 2.623, 0.039, 0.148, 0.0037), (3.629, 2.635, 0.040, 0.149, 0.0041),
    (3.705, 2.544, 0.041, 0.145, 0.0043), (3.580, 1.993, 0.081, 0.098, 0.0021), None, None,
    (4.124, 2.887, 0.054, 0.161, 0.0072), (4.217, 3.116, 0.059, 0.173, 0.0085), (4.261, 3.104, 0.060, 0.172, 0.0087),
    (4.359, 3.212, 0.063, 0.178, 0.0096), (4.433, 3.248, 0.066, 0.180, 0.0102), (4.546, 3.388, 0.070, 0.187, 0.0115),
    (4.675, 3.544, 0.076, 0.196, 0.0131), (4.744, 3.557, 0.078, 0.196, 0.0137), (4.760, 3.438, 0.077, 0.190, 0.0132),
    (4.758, 3.309, 0.076, 0.183, 0.0125), (4.763, 3.233, 0.076, 0.179, 0.0122), (4.803, 3.258, 0.077, 0.180, 0.0125),
    (4.895, 3.385, 0.081, 0.187, 0.0138), (4.935, 3.451, 0.083, 0.190, 0.0144), (5.017, 3.547, 0.086, 0.196, 0.0155),
    (5.075, 3.587, 0.089, 0.198, 0.0161), (5.423, 3.693, 0.101, 0.204, 0.0193), (5.447, 3.680, 0.102, 0.203, 0.0193),
    (5.214, 3.304, 0.109, 0.172, 0.0137), (5.287, 3.392, 0.107, 0.177, 0.0147),
]


def test_one_misleading_point_does_not_send_the_track_away():
    """Gating on the forward prediction took the short box bottom, rejected the next seven clean points
    and ran 2 m the wrong way before re-anchoring with a 2.3 m jump. The track must follow the clean points."""
    t = 20.417 + np.arange(len(RECORDED)) / 30
    z = np.array([r[:2] if r else (np.nan, np.nan) for r in RECORDED])
    R = np.array([[[r[2] ** 2, r[4]], [r[4], r[3] ** 2]] if r else np.eye(2) for r in RECORDED])
    out = rts_smooth(t, z, R)
    assert np.linalg.norm(np.diff(out.xy, axis=0), axis=1).max() < 0.25  # no re-anchoring jump (8 m/s)
    assert out.used[16:].sum() >= len(RECORDED) - 16 - 1
    assert out.xy[16:23, 1].min() > 2.6  # the clean points after the gap are at y 2.9-3.5


@pytest.mark.parametrize("gap_px, cut", [(6.0, True), (30.0, False)])
def test_box_resting_on_a_plate_counts_as_cut_despite_the_mask_sitting_low(gap_px, cut):
    """A window's mask can start a few pixels below the plate's real top edge; a box whose bottom
    stops gap_px above the mask is cut by the plate only when that gap is within the slack."""
    from squashvid.players.occluders import OccluderTrack

    mask = np.zeros((1, 288, 512), bool)
    mask[0, 160:200, 200:260] = True  # plate from row 160 (320 px in a 1024 frame)
    occ = OccluderTrack(np.array([0.0]), mask, 2.0)
    box = np.array([420.0, 320.0 - gap_px - 60.0, 450.0, 320.0 - gap_px])  # a short, far box
    assert (occ.under_box(0.0, box) > 0.2) == cut
