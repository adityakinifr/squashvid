"""Pinhole camera with radial distortion, posed in the court frame."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from squashvid.geom import court


def plane(name: str, spec: court.CourtSpec = court.WSF) -> tuple[np.ndarray, float]:
    """A court surface as (unit normal, offset): points P on it satisfy n . P = d."""
    if name == "floor":
        return np.array([0.0, 0.0, 1.0]), 0.0
    if name == "front":
        return np.array([0.0, 1.0, 0.0]), spec.length
    if name == "left":
        return np.array([1.0, 0.0, 0.0]), 0.0
    if name == "right":
        return np.array([1.0, 0.0, 0.0]), spec.width
    raise ValueError(f"unknown plane {name!r}")


@dataclass
class Camera:
    f: float
    cx: float
    cy: float
    rvec: np.ndarray
    tvec: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64))[0]

    @property
    def centre(self) -> np.ndarray:
        return (-self.R.T @ np.asarray(self.tvec, dtype=np.float64).reshape(3)).reshape(3)

    def copy(self, **changes) -> "Camera":
        base = replace(
            self,
            rvec=np.array(self.rvec, dtype=np.float64).reshape(3),
            tvec=np.array(self.tvec, dtype=np.float64).reshape(3),
            dist=np.array(self.dist, dtype=np.float64).reshape(-1),
        )
        return replace(base, **changes) if changes else base

    def project(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        uv, _ = cv2.projectPoints(pts, np.asarray(self.rvec, np.float64), np.asarray(self.tvec, np.float64), self.K, self.dist)
        return uv.reshape(-1, 2)

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Distance of each point along the optical axis; negative means behind the camera."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (pts @ self.R.T + np.asarray(self.tvec, np.float64).reshape(1, 3))[:, 2]

    def in_field(self, points: np.ndarray, width: int, height: int, min_depth: float = 0.3) -> np.ndarray:
        """True for points in front of the camera whose direction lies within the image's own field.

        The radial distortion polynomial folds back beyond the frame, so a point
        well outside the field can project to a plausible pixel. Callers must
        drop such points before drawing or matching them.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cam_pts = pts @ self.R.T + np.asarray(self.tvec, np.float64).reshape(1, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_norm = np.hypot(cam_pts[:, 0], cam_pts[:, 1]) / cam_pts[:, 2]
        r_max = 1.05 * np.hypot(max(self.cx, width - self.cx), max(self.cy, height - self.cy)) / self.f
        return (cam_pts[:, 2] > min_depth) & np.isfinite(r_norm) & (r_norm < r_max)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """World-frame unit direction of the ray through each (distorted) pixel."""
        pix = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pix, self.K, self.dist).reshape(-1, 2)
        d_cam = np.column_stack([norm, np.ones(len(norm))])
        d = d_cam @ self.R  # R.T applied to each row
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_plane(self, uv: np.ndarray, surface: str = "floor", offset: float = 0.0, spec: court.CourtSpec = court.WSF) -> np.ndarray:
        """Intersect pixel rays with a court surface shifted by `offset` along its normal.

        Rays parallel to the plane or meeting it behind the camera return NaN.
        """
        n, d = plane(surface, spec)
        d = d + offset
        c = self.centre
        rays = self.rays(uv)
        denom = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - c @ n) / denom
        t = np.where((np.abs(denom) > 1e-9) & (t > 0), t, np.nan)
        return c + rays * t[:, None]

    def plane_homography(self, surface: str, spec: court.CourtSpec = court.WSF) -> np.ndarray:
        """Undistorted-image homography mapping 2D plane coordinates to pixels.

        Plane coordinates: floor (x, y); front (x, z); left and right (y, z).
        Apply cv2.undistortPoints(..., P=K) to pixels before using its inverse.
        """
        P = self.K @ np.column_stack([self.R, np.asarray(self.tvec, np.float64).reshape(3)])
        if surface == "floor":
            return P[:, [0, 1, 3]]
        if surface == "front":
            return np.column_stack([P[:, 0], P[:, 2], spec.length * P[:, 1] + P[:, 3]])
        if surface == "left":
            return P[:, [1, 2, 3]]
        if surface == "right":
            return np.column_stack([P[:, 1], P[:, 2], spec.width * P[:, 0] + P[:, 3]])
        raise ValueError(f"unknown plane {surface!r}")

    def to_dict(self) -> dict:
        return {
            "f": float(self.f),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "rvec": [float(v) for v in np.asarray(self.rvec).reshape(3)],
            "tvec": [float(v) for v in np.asarray(self.tvec).reshape(3)],
            "dist": [float(v) for v in np.asarray(self.dist).reshape(-1)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            f=data["f"],
            cx=data["cx"],
            cy=data["cy"],
            rvec=np.array(data["rvec"], dtype=np.float64),
            tvec=np.array(data["tvec"], dtype=np.float64),
            dist=np.array(data.get("dist", [0.0] * 5), dtype=np.float64),
        )


def solve_pnp(world: np.ndarray, image: np.ndarray, f: float, cx: float, cy: float, dist: np.ndarray | None = None) -> tuple[Camera, float]:
    """Pose from point correspondences at a known focal length; returns the camera and RMS error in px."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
    obj = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image, dtype=np.float64).reshape(-1, 2)
    _, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    _, rv, tv = cv2.solvePnP(obj, img, K, dist, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
    cam = Camera(f, cx, cy, rv.reshape(3), tv.reshape(3), dist)
    err = np.linalg.norm(cam.project(obj) - img, axis=1)
    return cam, float(np.sqrt(np.mean(err**2)))


def solve_pnp_focal(world: np.ndarray, image: np.ndarray, cx: float, cy: float, f_range=(250.0, 1200.0), step: float = 2.0) -> tuple[Camera, float]:
    """Grid-search the focal length, keeping the pose with the lowest reprojection error."""
    best: tuple[Camera, float] | None = None
    for f in np.arange(f_range[0], f_range[1], step):
        cam, rms = solve_pnp(world, image, float(f), cx, cy)
        if best is None or rms < best[1]:
            best = (cam, rms)
    assert best is not None
    return best
