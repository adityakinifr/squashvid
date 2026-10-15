"""Court-space smoothing: constant-velocity Kalman filter with an RTS backward pass, made robust by
re-weighting against the smoothed track.

Each measurement brings its own 2x2 covariance from the foot-point ladder, so
along-court positions near the front wall (5-17 cm/px) are trusted far less
than positions near the back. Time steps come from presentation timestamps,
which keeps clip 2's dropped frames honest.

Outliers are judged against the smoothed track, which sees the data on both
sides, not against the forward prediction. Gating on the forward prediction
alone failed in a characteristic way: one misleading point (a box bottom on a
partial box, or a few frames of the other player's box) bent the predicted
velocity, the correct points that followed were all rejected, and the track
ran away for a fifth of a second before re-anchoring with a 1-3 m jump. That
added metres to rally distances. Here the track is smoothed with every point,
points implausible against it are dropped, and the track is smoothed again,
with the gate tightened over the first passes so gross outliers go before
they can pull their neighbours out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q_ACCEL = 12.0  # white-acceleration spectral density, m^2/s^3: squash changes direction hard
GATE_CHI2 = 16.0  # 2-dof Mahalanobis gate (~99.97 %) on the residual against the smoothed track
GATE_START = 8.0  # the first pass gates at GATE_CHI2 * GATE_START, halving each pass down to GATE_CHI2
MAX_PASSES = 10
V_MAX = 8.0


@dataclass
class Smoothed:
    xy: np.ndarray  # (T, 2)
    vel: np.ndarray  # (T, 2)
    cov: np.ndarray  # (T, 2, 2) positional covariance
    used: np.ndarray  # (T,) bool: measurement accepted
    gated: np.ndarray  # (T,) bool: measurement present but rejected


def _qmat(dt: float, q: float) -> np.ndarray:
    a = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = a
    Q[np.ix_([1, 3], [1, 3])] = a
    return Q


def _fmat(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def _kalman_rts(t: np.ndarray, z: np.ndarray, R: np.ndarray, active: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Plain (ungated) Kalman filter plus RTS pass over the active measurements; returns states and covariances."""
    T = len(t)
    xs_f, Ps_f, xs_p, Ps_p = np.zeros((T, 4)), np.zeros((T, 4, 4)), np.zeros((T, 4)), np.zeros((T, 4, 4))
    x = np.zeros(4)
    P = np.diag([25.0, 25.0, (V_MAX / 2) ** 2, (V_MAX / 2) ** 2])
    if active.any():
        first = int(np.argmax(active))
        x[:2] = z[first]
        P[:2, :2] = R[first]
    for i in range(T):
        if i > 0:
            dt = max(t[i] - t[i - 1], 1e-3)
            F = _fmat(dt)
            x = F @ x
            P = F @ P @ F.T + _qmat(dt, q)
        xs_p[i], Ps_p[i] = x, P
        if active[i]:
            S = H @ P @ H.T + R[i]
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z[i] - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xs_f[i], Ps_f[i] = x, P
    xs, Ps = xs_f.copy(), Ps_f.copy()
    for i in range(T - 2, -1, -1):
        F = _fmat(max(t[i + 1] - t[i], 1e-3))
        C = Ps_f[i] @ F.T @ np.linalg.inv(Ps_p[i + 1])
        xs[i] = xs_f[i] + C @ (xs[i + 1] - xs_p[i + 1])
        Ps[i] = Ps_f[i] + C @ (Ps[i + 1] - Ps_p[i + 1]) @ C.T
    return xs, Ps


def rts_smooth(t: np.ndarray, z: np.ndarray, R: np.ndarray, q: float = Q_ACCEL, gate: float = GATE_CHI2) -> Smoothed:
    """Smooth measurements z (T, 2) with covariances R (T, 2, 2); NaN rows are missing.

    State is (x, y, vx, vy). The track starts at the first measurement with a
    velocity prior of V_MAX / 2 per axis. A measurement is kept when its
    residual against the smoothed track, with covariance R + P_smoothed, passes
    the gate; passes repeat until the kept set stops changing.
    """
    t = np.asarray(t, np.float64)
    z = np.asarray(z, np.float64)
    has = np.isfinite(z).all(1)
    zf = np.where(has[:, None], z, 0.0)
    active = has.copy()
    xs, Ps = _kalman_rts(t, zf, R, active, q)
    for k in range(MAX_PASSES):
        g = gate * max(GATE_START / 2**k, 1.0)
        r = zf - xs[:, :2]
        d2 = np.einsum("ti,tij,tj->t", r, np.linalg.inv(R + Ps[:, :2, :2]), r)
        keep = has & (d2 <= g)
        if not keep.any() or (g == gate and np.array_equal(keep, active)):
            break
        active = keep
        xs, Ps = _kalman_rts(t, zf, R, active, q)
    return Smoothed(xs[:, :2], xs[:, 2:], Ps[:, :2, :2], active, has & ~active)
