"""From image points to court coordinates: ray-plane intersection, uncertainty, sound travel.

Court frame (geom/court.py): x across from the left wall, y from the back wall to the front wall, z up.
Each impact kind lives on one plane, and its pixel's ray meets that plane at the impact:

  front_wall       y = L      axes (x, z)
  side_wall_left   x = 0      axes (y, z)
  side_wall_right  x = W      axes (y, z)
  floor            z = ball radius     axes (x, y)
  back_wall        y = 0      axes (x, z)
  racket           y = the striker's court y (the ball is at the player), axes (x, z); depth is uncertain
"""

from __future__ import annotations

import numpy as np

from squashvid.geom import court
from squashvid.geom.camera import Camera

BALL_RADIUS = 0.02  # m: a squash ball is 40 mm across, so its centre is 20 mm above the floor at a bounce
SOUND_SPEED = 343.0  # m/s
WALL_TOP = 6.0  # m: rays meeting a wall above this are not impacts

# kind -> (plane id for camera.to_plane-style maths, axis names)
AXES = {
    "front_wall": ("x", "z"),
    "side_wall_left": ("y", "z"),
    "side_wall_right": ("y", "z"),
    "floor": ("x", "y"),
    "back_wall": ("x", "z"),
    "racket": ("x", "z"),
}


def ray_plane(centre: np.ndarray, rays: np.ndarray, axis: int, value: float) -> np.ndarray:
    """Points where rays from `centre` meet the plane  P[axis] = value; NaN when parallel or behind the camera."""
    d = np.asarray(rays, np.float64).reshape(-1, 3)
    c = np.asarray(centre, np.float64).reshape(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (value - c[axis]) / d[:, axis]
    t = np.where(np.isfinite(t) & (t > 0), t, np.nan)
    return c + d * t[:, None]


def plane_of(kind: str, spec: court.CourtSpec, player_y: float | None = None) -> tuple[int, float]:
    """(axis index, coordinate) of the plane an event kind sits on."""
    if kind == "front_wall":
        return 1, spec.length
    if kind == "side_wall_left":
        return 0, 0.0
    if kind == "side_wall_right":
        return 0, spec.width
    if kind == "floor":
        return 2, BALL_RADIUS
    if kind == "back_wall":
        return 1, 0.0
    if kind == "racket":
        if player_y is None or not np.isfinite(player_y):
            return 1, np.nan
        return 1, float(player_y)
    raise ValueError(f"unknown event kind {kind!r}")


def locate(cam: Camera, uv: np.ndarray, kind: str, spec: court.CourtSpec, player_y: float | None = None) -> np.ndarray:
    """World (x, y, z) of image points `uv` on the plane of `kind`; NaN rows when the ray misses it."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    axis, value = plane_of(kind, spec, player_y)
    if not np.isfinite(value):
        return np.full((len(uv), 3), np.nan)
    return ray_plane(cam.centre, cam.rays(uv), axis, value)


def inside(p: np.ndarray, kind: str, spec: court.CourtSpec, margin: float = 0.25) -> np.ndarray:
    """Whether located points fall on the physical surface of their plane (with a margin for pixel noise)."""
    p = np.asarray(p, np.float64).reshape(-1, 3)
    W, L = spec.width, spec.length
    with np.errstate(invalid="ignore"):
        if kind in ("front_wall", "back_wall"):
            return (p[:, 0] > -margin) & (p[:, 0] < W + margin) & (p[:, 2] > -margin) & (p[:, 2] < WALL_TOP)
        if kind in ("side_wall_left", "side_wall_right"):
            return (p[:, 1] > -margin) & (p[:, 1] < L + margin) & (p[:, 2] > -margin) & (p[:, 2] < WALL_TOP)
        if kind == "floor":
            return (p[:, 0] > -margin) & (p[:, 0] < W + margin) & (p[:, 1] > -margin) & (p[:, 1] < L + margin)
        return np.isfinite(p).all(axis=1)


def exit_surface(cam: Camera, uv: np.ndarray, spec: court.CourtSpec) -> tuple[str, np.ndarray]:
    """The court surface a pixel's ray meets first inside the court box, and where.

    A point of the ball track that lies ON a surface is at the end of its ray's stay in the box,
    so the exit surface names the only plane an impact at that pixel can be on (apart from the back
    glass, which every ray crosses on its way in). Returns ('none', NaN) for rays that leave through the roof.
    """
    uv = np.asarray(uv, np.float64).reshape(1, 2)
    c, d = cam.centre, cam.rays(uv)
    best, name, hit = np.inf, "none", np.full(3, np.nan)
    for kind in ("front_wall", "side_wall_left", "side_wall_right", "floor"):
        axis, value = plane_of(kind, spec)
        p = ray_plane(c, d, axis, value)[0]
        if not np.isfinite(p).all() or not inside(p[None], kind, spec, margin=0.0)[0]:
            continue
        t = float(np.linalg.norm(p - c))
        if t < best:
            best, name, hit = t, kind, p
    return name, hit


def plane_sigma(cam: Camera, uv: np.ndarray, kind: str, spec: court.CourtSpec, sigma_px: float, player_y: float | None = None,
                depth_sigma: float = 0.0) -> tuple[float, float]:
    """(sigma_a, sigma_b) in metres along the plane's two axes for an isotropic pixel error `sigma_px`.

    `depth_sigma` adds the uncertainty of the plane itself along y (the racket case, where the ball is only
    known to be near the striker): it moves the located point along the ray.
    """
    uv = np.asarray(uv, np.float64).reshape(2)
    ax = AXES[kind]
    idx = {"x": 0, "y": 1, "z": 2}
    base = locate(cam, uv, kind, spec, player_y)[0]
    if not np.isfinite(base).all():
        return float("nan"), float("nan")
    jac = []
    for du, dv in ((1.0, 0.0), (0.0, 1.0)):
        a = locate(cam, uv + [du, dv], kind, spec, player_y)[0]
        b = locate(cam, uv - [du, dv], kind, spec, player_y)[0]
        jac.append((a - b) / 2.0)
    jac = np.array(jac) * sigma_px  # (2 pixel axes, 3 world axes)
    var = (jac**2).sum(axis=0)
    out = [float(np.sqrt(var[idx[a]])) for a in ax]
    if depth_sigma > 0 and np.isfinite(player_y if player_y is not None else np.nan):
        shifted = locate(cam, uv, "racket", spec, float(player_y) + depth_sigma)[0]
        move = np.abs(shifted - base)
        out = [float(np.hypot(o, move[idx[a]])) for o, a in zip(out, ax)]
    return out[0], out[1]


def sound_delay(cam: Camera, point: np.ndarray) -> float:
    """Seconds for a sound at world `point` to reach the camera's microphone."""
    return float(np.linalg.norm(np.asarray(point, np.float64) - cam.centre) / SOUND_SPEED)


NORMALS = {
    "front_wall": np.array([0.0, 1.0, 0.0]),
    "back_wall": np.array([0.0, 1.0, 0.0]),
    "side_wall_left": np.array([1.0, 0.0, 0.0]),
    "side_wall_right": np.array([1.0, 0.0, 0.0]),
    "floor": np.array([0.0, 0.0, 1.0]),
}
# Direction of travel into each plane: the ball must be heading this way before an impact.
INTO = {
    "front_wall": np.array([0.0, 1.0, 0.0]),
    "back_wall": np.array([0.0, -1.0, 0.0]),
    "side_wall_left": np.array([-1.0, 0.0, 0.0]),
    "side_wall_right": np.array([1.0, 0.0, 0.0]),
    "floor": np.array([0.0, 0.0, -1.0]),
}
# Restitution grid for the bounce test: normal speed kept, tangential speed kept.
E_GRID = (0.3, 0.5, 0.7, 0.9)
T_GRID = (0.5, 0.75, 1.0)


def image_jacobian(cam: Camera, point: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """d(u, v)/d(x, y, z) at a world point: (2, 3), pixels per metre."""
    p = np.asarray(point, np.float64).reshape(3)
    cols = []
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        cols.append((cam.project(p + d)[0] - cam.project(p - d)[0]) / (2 * eps))
    return np.array(cols).T


def bounce_residual(cam: Camera, point: np.ndarray, kind: str, v_before: np.ndarray, v_after: np.ndarray) -> tuple[float, float]:
    """How badly image velocities before/after an instant fail to be a bounce off the plane of `kind` at `point`.

    The ball's 3D velocity V before the impact is unknown (depth is not seen), but a bounce ties the
    velocity after to it: the plane-normal part is reversed and shrunk (restitution e), the tangential
    part shrunk (t). The four image-velocity components (before and after) are then explained by the
    three unknowns of V, which must point into the plane; the least-squares residual, in px per frame, over a grid of e and t, measures how
    well a bounce explains the kink. A kink that is not an impact on this plane leaves a large residual.
    Returns (residual in px per frame, implied speed before the impact in metres per frame at the best fit).
    """
    n = NORMALS[kind]
    J = image_jacobian(cam, point)
    vb, va = np.asarray(v_before, np.float64), np.asarray(v_after, np.float64)
    y = np.concatenate([vb, va])
    best, speed = np.inf, np.nan
    P = np.outer(n, n)
    for e in E_GRID:
        for t in T_GRID:
            A = t * (np.eye(3) - P) - e * P
            M = np.vstack([J, J @ A])
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
            if coef @ INTO[kind] <= 0:
                continue  # the fit has the ball leaving the plane before it arrives
            r = float(np.linalg.norm(M @ coef - y))
            if r < best:
                best, speed = r, float(np.linalg.norm(coef))
    return best, speed
