"""Bodies: COCO -> H36M mapping, gap bridging, scale/placement through a camera, floor contact, export arrays.

Everything is synthetic and built in memory or under tmp_path; nothing reads or writes work/."""

import cv2
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from squashvid.bodies import place as P  # noqa: E402
from squashvid.bodies import run as R  # noqa: E402
from squashvid.bodies.skeleton import COCO_OF_H36M, PARENT, bones, coco_to_h36m  # noqa: E402
from squashvid.geom.camera import Camera  # noqa: E402
from squashvid.players.geometry import ANKLE_Z  # noqa: E402
from squashvid.viewer.export import bodies_arrays, dequantise  # noqa: E402

DIST = np.zeros(5)


def camera(n=1) -> np.ndarray:
    """A camera 2.5 m behind the back wall and 2.4 m up, looking down the court; rows (f, cx, cy, rvec, tvec)."""
    c = np.array([3.0, -2.5, 2.4])
    fwd = np.array([0.0, 1.0, -0.15])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rm = np.stack([right, down, fwd])
    rvec = cv2.Rodrigues(Rm)[0].ravel()
    row = np.concatenate([[900.0, 960.0, 540.0], rvec, -Rm @ c])
    return np.tile(row, (n, 1))


def standing(lean=0.0) -> np.ndarray:
    """H36M-17 pelvis-relative joints (court axes, metres) of a 1.5 m person facing +y, optionally leaning toward +y."""
    j = np.zeros((17, 3))
    j[1], j[4] = (-0.08, 0, 0), (0.08, 0, 0)  # R hip is at -x when facing +y, L at +x? facing +y: left = -x; kept symmetric here
    j[2], j[5] = j[1] + (0, 0, -0.38), j[4] + (0, 0, -0.38)
    j[3], j[6] = j[2] + (0, 0, -0.36), j[5] + (0, 0, -0.36)
    j[7], j[8], j[9], j[10] = (0, 0, 0.22), (0, 0, 0.44), (0, 0.04, 0.56), (0, 0, 0.68)
    j[11], j[14] = (0.17, 0, 0.42), (-0.17, 0, 0.42)
    j[12], j[15] = j[11] + (0, 0, -0.26), j[14] + (0, 0, -0.26)
    j[13], j[16] = j[12] + (0, 0, -0.24), j[15] + (0, 0, -0.24)
    j[:, 1] += lean * j[:, 2]
    return j - j[0]


def to_coco(h: np.ndarray) -> np.ndarray:
    """H36M court joints -> COCO-17 court points (face points collapse onto the nose / head)."""
    c = np.zeros((17, 3))
    for h_i, c_i in COCO_OF_H36M.items():
        c[c_i] = h[h_i]
    c[1] = c[2] = h[9]
    c[3] = c[4] = h[10]
    return c


def test_mapping_uses_the_expected_coco_joints():
    rng = np.random.default_rng(0)
    kp = rng.uniform(100, 900, (4, 17, 2))
    sc = rng.uniform(0.5, 1, (4, 17))
    h, conf = coco_to_h36m(kp, sc)
    assert np.allclose(h[:, 1], kp[:, 12]) and np.allclose(h[:, 4], kp[:, 11])  # R hip is COCO 12, L hip 11
    assert np.allclose(h[:, 3], kp[:, 16]) and np.allclose(h[:, 6], kp[:, 15])
    assert np.allclose(h[:, 0], (kp[:, 11] + kp[:, 12]) / 2)
    assert np.allclose(h[:, 8], (kp[:, 5] + kp[:, 6]) / 2)
    assert np.allclose(conf[:, 0], np.minimum(sc[:, 11], sc[:, 12]))
    # the head top lies beyond the head along thorax -> head, i.e. higher in the image than the nose when upright
    up = kp.copy()
    up[:, 5:7, 1] = 500
    up[:, 0, 1] = 300
    up[:, 3:5, 1] = 320
    hh, _ = coco_to_h36m(up, sc)
    assert (hh[:, 10, 1] < up[:, 3, 1]).all()


def test_skeleton_tree_is_consistent():
    assert len(bones()) == 16 and all(p < j for p, j in bones())
    assert PARENT[0] == -1


def test_segments_bridge_short_gaps_but_not_long_ones():
    fps = 30
    valid = np.ones(200, bool)
    valid[50:60] = False  # 10 frames, 0.33 s: bridged
    valid[100:140] = False  # 40 frames, 1.3 s: splits
    segs = R.segments(valid, gap=int(0.5 * fps), min_len=9)
    assert segs == [(0, 100), (140, 200)]
    valid[0:100] = False
    valid[3:6] = True  # a 3 frame blip is dropped
    assert R.segments(valid, 15, 9) == [(140, 200)]


def test_build_keeps_bone_lengths_and_place_grounds_the_lowest_foot():
    pose = standing()
    dirs = P.directions(pose[None])
    props = P.bone_proportions(pose[None])
    lengths = props * 3.4
    rel = P.build(np.repeat(dirs, 5, axis=0), lengths)
    for p, j in bones():
        assert np.allclose(np.linalg.norm(rel[:, j] - rel[:, p], axis=1), lengths[j])
    rel[:, 3, 2] += np.linspace(0, 0.1, 5)  # raise one ankle progressively
    pos = np.tile([3.0, 4.0], (5, 1))
    out = P.place(rel, pos, np.zeros(5, bool))
    low = np.minimum(out[:, 3, 2], out[:, 6, 2])
    assert np.allclose(low, ANKLE_Z)
    assert np.allclose((out[:, 3, :2] + out[:, 6, :2]) / 2, pos)
    assert (out[:, :, 2] >= ANKLE_Z - 1e-9).all()  # nothing below the floor plane


def test_pelvis_anchor_when_foot_point_came_from_the_hips():
    rel = P.build(P.directions(standing()[None]), P.bone_proportions(standing()[None]) * 3.4)
    out = P.place(rel, np.array([[2.0, 5.0]]), np.array([True]))
    assert np.allclose(out[0, 0, :2], [2.0, 5.0])


def test_model_to_court_is_a_rotation_and_alpha_one_is_the_camera():
    cam = camera(3)
    Rm = P.rotations(cam)
    for alpha in (0.0, 0.5, 1.0):
        M = P.model_to_court(Rm, alpha)
        for m in M:
            assert np.allclose(m.T @ m, np.eye(3), atol=1e-9) and np.linalg.det(m) > 0
    M1 = P.model_to_court(Rm, 1.0)
    assert np.allclose(M1, np.transpose(Rm, (0, 2, 1)), atol=1e-9)
    M0 = P.model_to_court(Rm, 0.0)
    assert np.allclose(M0[:, 2, 1], -1)  # level camera: model y (down) is court -z


def test_vectorised_projection_matches_opencv_with_distortion():
    cam = camera(2)
    dist = np.array([0.05, -0.02, 0.001, 0.0005, 0.0])
    X = np.array([[[3.0, 5.0, 1.0], [2.0, 3.0, 0.2], [4.0, 6.0, 1.7]]] * 2)
    ours = P.project(cam, dist, X)
    c = Camera(cam[0, 0], cam[0, 1], cam[0, 2], cam[0, 3:6], cam[0, 6:9], dist)
    assert np.allclose(ours[0], c.project(X[0]), atol=1e-6)


def test_fit_scale_recovers_a_known_size_through_the_camera():
    pose = standing(lean=0.25)
    n = 40
    cam = camera(n)
    pos = np.tile([3.2, 4.5], (n, 1))
    total_true = 4.1
    props = P.bone_proportions(pose[None])
    dirs = np.repeat(P.directions(pose[None]), n, axis=0)
    X = P.place(P.build(dirs, props * total_true), pos, np.zeros(n, bool))
    Rm = P.rotations(cam)
    uv = P.project(cam, DIST, X, Rm)
    kp = np.zeros((n, 17, 2))
    ks = np.ones((n, 17))
    for h, c in COCO_OF_H36M.items():
        kp[:, c] = uv[:, h]
    total, err = P.fit_scale(dirs, props, pos, np.zeros(n, bool), cam, DIST, kp, ks, Rm)
    assert abs(total - total_true) < 0.05
    assert err < 1.0  # pixels


class StubLifter:
    """Stands in for MotionBERT: always 'sees' the same upright person, in lifter axes (x right, y down, z away)."""

    def __init__(self):
        j = standing()
        self.pose = np.stack([j[:, 0], -j[:, 2], j[:, 1]], axis=1).astype(np.float32)

    def __call__(self, x):
        t = x.shape[1]
        return torch.from_numpy(np.broadcast_to(self.pose, (1, t, 17, 3)).copy())


def synthetic_player(n, fps=30.0, missing=()):
    cam = camera(n)
    pos = np.tile([3.1, 4.2], (n, 1))
    h = standing() + [pos[0, 0], pos[0, 1], ANKLE_Z + 0.36 * 2 - 0.0]
    h[:, 2] = h[:, 2] - h[[3, 6], 2].min() + ANKLE_Z
    X = np.repeat(h[None], n, axis=0)
    uv = P.project(cam, DIST, X)
    kp = np.zeros((n, 17, 2))
    for hi, ci in COCO_OF_H36M.items():
        kp[:, ci] = uv[:, hi]
    kp[:, 1] = kp[:, 2] = kp[:, 0]
    kp[:, 3] = kp[:, 4] = kp[:, 0] + [0, -10]
    ks = np.full((n, 17), 0.9)
    det = np.ones(n, bool)
    for a, b in missing:
        det[a:b] = False
        ks[a:b] = 0
    return {"kp": kp, "ks": ks, "pos": pos, "det": det, "src": np.array(["ankle"] * n, dtype=object)}, cam


def test_lift_player_bridges_short_gap_and_leaves_long_gap_empty():
    n, fps = 300, 30.0
    arr, cam = synthetic_player(n, missing=[(60, 70), (150, 200)])  # 10 frames (0.33 s), 50 frames (1.7 s)
    tr = R.lift_player(arr, cam, DIST, fps, StubLifter())
    assert (tr.state[60:70] == R.STATE_BRIDGED).all()
    assert (tr.conf[60:70] > 0).all() and (tr.conf[60:70] < tr.conf[:50].min() + 1e-9).all()  # bridged frames are less trusted
    assert (tr.state[150:200] == R.STATE_NONE).all()
    assert (tr.conf[150:200] == 0).all() and np.isnan(tr.joints[150:200]).all()
    assert (tr.state[:60] == R.STATE_SEEN).all() and (tr.state[200:] == R.STATE_SEEN).all()
    seen = tr.joints[tr.state > 0]
    low = np.minimum(seen[:, 3, 2], seen[:, 6, 2])
    assert (low >= ANKLE_Z - 1e-6).all() and low.max() < ANKLE_Z + 0.02  # grounded
    assert np.allclose((seen[:, 3, :2] + seen[:, 6, :2]) / 2, [3.1, 4.2], atol=0.02)  # over the court foot point
    ln = np.linalg.norm(seen[:, 2] - seen[:, 1], axis=1)
    assert ln.std() / ln.mean() < 1e-6  # constant bone length
    assert tr.info["stature_m_p90"] > 1.0


def test_bodies_arrays_round_trip_and_missing_player(tmp_path):
    joints = np.arange(51, dtype=np.float32).reshape(17, 3) / 10
    df = pd.DataFrame({"frame": [1, 2], "t": [0.1, 0.2], "player": ["A", "A"], "state": [1, 2], "conf": [0.8, 0.4], "joints": [joints.ravel(), joints.ravel()]})
    out = bodies_arrays(df, 5)
    body = dequantise(out["A_body"], 1000)
    assert np.isnan(body[0]).all() and np.allclose(body[1], joints, atol=6e-4)
    assert out["A_bconf"].tolist() == [0, 204, 102, 0, 0]
    assert np.isnan(dequantise(out["B_body"], 1000)).all() and not out["B_bconf"].any()
    assert bodies_arrays(None, 5) == {}  # no bodies.parquet: no arrays, the viewer keeps its fallback
