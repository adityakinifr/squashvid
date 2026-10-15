"""Turn lifted bone directions into a metric, court-frame skeleton rooted at the player's court foot point.

Steps: rotate the lifter's camera-axis offsets into the court frame with the calibrated camera heading; rebuild the
skeleton with constant per-player bone lengths (only the lifter's bone *directions* are used); fit one metric scale per
player by reprojecting through the court camera onto the detected keypoints; put the ankle midpoint over the court
foot point and the lowest ankle on the floor.
"""

from __future__ import annotations

import cv2
import numpy as np

from squashvid.bodies.skeleton import COCO_OF_H36M, L_ANKLE, MIRROR, PARENT, R_ANKLE
from squashvid.players.geometry import ANKLE_Z

N_J = 17


def rotations(cam: np.ndarray) -> np.ndarray:
    """(T,9) camera rows (f,cx,cy,rvec,tvec) -> (T,3,3) world-to-camera rotations."""
    return np.stack([cv2.Rodrigues(r.astype(np.float64))[0] for r in cam[:, 3:6]])


def model_to_court(R: np.ndarray, alpha: float) -> np.ndarray:
    """(T,3,3) matrices taking lifter axes (x right, y down, z away) to court axes.

    alpha = 1 uses the real camera axes (the lifter's frame is the camera's); alpha = 0 uses a level camera with the
    same heading (the lifter saw an upright person). Values between tilt by that fraction of the camera pitch."""
    up = np.array([0.0, 0.0, 1.0])
    right = R[:, 0, :]  # camera x axis in court coordinates (rows of R are the camera axes)
    fwd = R[:, 2, :]
    pitch = np.arcsin(np.clip(-fwd[:, 2], -1, 1))  # looking down is positive
    rh = right - (right @ up)[:, None] * up
    rh /= np.linalg.norm(rh, axis=1, keepdims=True)
    fh = np.cross(up, rh)  # right x down = fwd, with down = -up
    ang = alpha * pitch
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    f_t = c * fh - s * up  # forward tilted down by ang
    d_t = -(c * up + s * fh)  # down axis tilted likewise (perpendicular to f_t)
    return np.stack([rh, d_t, f_t], axis=2)  # columns: images of lifter x, y, z


def bone_proportions(lifted: np.ndarray) -> np.ndarray:
    """(T,17,3) lifted -> (17,) fraction of the total bone length each joint's incoming bone takes (median over frames,
    left/right averaged)."""
    par = np.array(PARENT[1:])
    ln = np.linalg.norm(lifted[:, 1:] - lifted[:, par], axis=2)
    frac = np.median(ln / ln.sum(axis=1, keepdims=True), axis=0)
    out = np.zeros(N_J)
    out[1:] = frac
    return (out + out[list(MIRROR)]) / 2


def directions(lifted: np.ndarray) -> np.ndarray:
    par = np.array(PARENT[1:])
    d = lifted[:, 1:] - lifted[:, par]
    return d / np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)


def build(dirs_w: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """(T,16,3) court-frame unit bone directions + (17,) lengths -> (T,17,3) pelvis-relative joints."""
    out = np.zeros((len(dirs_w), N_J, 3))
    for j in range(1, N_J):
        out[:, j] = out[:, PARENT[j]] + lengths[j] * dirs_w[:, j - 1]
    return out


def project(cam: np.ndarray, dist: np.ndarray, X: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """Vectorised OpenCV projection. cam (T,9), X (T,K,3) court points -> (T,K,2) pixels (NaN behind the camera)."""
    R = rotations(cam) if R is None else R
    xc = np.einsum("tij,tkj->tki", R, X) + cam[:, None, 6:9]
    z = xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = xc[..., 0] / z, xc[..., 1] / z
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv = np.stack([cam[:, None, 0] * xd + cam[:, None, 1], cam[:, None, 0] * yd + cam[:, None, 2]], axis=-1)
    uv[z <= 0.05] = np.nan
    return uv


def place(rel: np.ndarray, pos: np.ndarray, anchor_pelvis: np.ndarray, ankle_z: float = ANKLE_Z) -> np.ndarray:
    """(T,17,3) pelvis-relative joints -> court coordinates.

    The ankle midpoint (or the pelvis where the foot point itself came from the hips) goes over `pos` (T,2) and the
    lowest ankle sits at `ankle_z`, the height of the ankle keypoint above the floor."""
    ank = (rel[:, R_ANKLE] + rel[:, L_ANKLE]) / 2
    ref = np.where(anchor_pelvis[:, None], rel[:, 0], ank)
    out = rel.copy()
    out[..., 0] += (pos[:, 0] - ref[:, 0])[:, None]
    out[..., 1] += (pos[:, 1] - ref[:, 1])[:, None]
    out[..., 2] += (ankle_z - np.minimum(rel[:, R_ANKLE, 2], rel[:, L_ANKLE, 2]))[:, None]
    return out


def reproj_error(uv_pred: np.ndarray, uv_det: np.ndarray, score: np.ndarray, min_score: float = 0.3) -> np.ndarray:
    """(T,K) pixel distances on joints both predicted and detected with score >= min_score (NaN elsewhere)."""
    e = np.linalg.norm(uv_pred - uv_det, axis=-1)
    return np.where(score >= min_score, e, np.nan)


def coco_pairs() -> tuple[list[int], list[int]]:
    h = list(COCO_OF_H36M)
    return h, [COCO_OF_H36M[j] for j in h]


def fit_scale(dirs_w: np.ndarray, props: np.ndarray, pos: np.ndarray, anchor_pelvis: np.ndarray, cam: np.ndarray,
              dist: np.ndarray, kp: np.ndarray, ks: np.ndarray, R: np.ndarray, grid=np.linspace(3.0, 8.0, 51)) -> tuple[float, float]:
    """Metric size of the player's bone chain (sum of bone lengths, metres) minimising the median reprojection error."""
    hj, cj = coco_pairs()
    best = (np.inf, float(grid[0]))
    sub = np.linspace(0, len(dirs_w) - 1, min(len(dirs_w), 1500)).astype(int)
    for total in grid:
        X = place(build(dirs_w[sub], props * total), pos[sub], anchor_pelvis[sub])
        e = reproj_error(project(cam[sub], dist, X[:, hj], R[sub]), kp[sub][:, cj], ks[sub][:, cj])
        m = np.nanmedian(e) if np.isfinite(e).any() else np.inf
        if m < best[0]:
            best = (m, float(total))
    lo, hi = best[1] - 0.12, best[1] + 0.12
    for total in np.linspace(lo, hi, 13):
        X = place(build(dirs_w[sub], props * total), pos[sub], anchor_pelvis[sub])
        e = reproj_error(project(cam[sub], dist, X[:, hj], R[sub]), kp[sub][:, cj], ks[sub][:, cj])
        m = np.nanmedian(e) if np.isfinite(e).any() else np.inf
        if m < best[0]:
            best = (m, float(total))
    return best[1], best[0]
