"""Torso colour descriptors: the near-binary cue (light-blue shirt versus dark shirt) behind identity.

The torso comes from the shoulder and hip keypoints when pose is confident and
from a fixed box fraction otherwise. Near-black pixels are dropped before any
statistic, because the door's hinge plates and handle block are black and
would otherwise turn the light-blue player dark wherever they overlap.
"""

from __future__ import annotations

import cv2
import numpy as np

from squashvid.players.pose import L_HIP, L_SHO, R_HIP, R_SHO

KPT_MIN = 0.35
HARDWARE_V = 32  # hinge plates sit at V < 30; the dark shirt is V 45-85
H_BINS, S_BINS = 12, 4
FEATURES = ["L", "a", "b", "L_sd", "b_sd", "frac_blue", "frac_dark", "frac_light", "n_px"]
N_FEAT = len(FEATURES) + H_BINS * S_BINS


def torso_polygon(box: np.ndarray, kpts: np.ndarray | None, scores: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Torso rectangle in pixels; True when it came from keypoints.

    From keypoints: between the shoulder and hip midpoints, with a minimum
    width so a side-on player (coincident shoulders) still yields shirt pixels.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if kpts is not None and scores is not None and min(scores[[L_SHO, R_SHO, L_HIP, R_HIP]]) > KPT_MIN:
        sho, hip = kpts[[L_SHO, R_SHO]].astype(np.float64), kpts[[L_HIP, R_HIP]].astype(np.float64)
        top, bot = sho.mean(0), hip.mean(0)
        height = bot[1] - top[1]
        if height > 6:
            half = max(0.4 * abs(sho[0, 0] - sho[1, 0]), 0.22 * height)
            cx = (top[0] + bot[0]) / 2
            ya, yb = top[1] + 0.1 * height, bot[1] - 0.1 * height
            return np.array([[cx - half, ya], [cx + half, ya], [cx + half, yb], [cx - half, yb]]), True
    w, h = x2 - x1, y2 - y1
    quad = np.array([[x1 + 0.3 * w, y1 + 0.2 * h], [x1 + 0.7 * w, y1 + 0.2 * h], [x1 + 0.7 * w, y1 + 0.45 * h], [x1 + 0.3 * w, y1 + 0.45 * h]])
    return quad, False


def torso_features(image: np.ndarray, box: np.ndarray, kpts: np.ndarray | None = None, scores: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Descriptor vector (N_FEAT,) for one person and whether the torso came from keypoints.

    Scalars first (see FEATURES), then a hue-saturation histogram of the kept pixels.
    """
    quad, from_kpts = torso_polygon(box, kpts, scores)
    H, W = image.shape[:2]
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    out = np.zeros(N_FEAT, np.float32)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return out, from_kpts
    crop = image[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad - [x0, y0]).astype(np.int32), 1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = (mask.reshape(-1) > 0) & (hsv[:, 2] >= HARDWARE_V)
    n = int(keep.sum())
    if n < 6:
        return out, from_kpts
    hsv, lab = hsv[keep], lab[keep]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    blue = (h >= 95) & (h <= 128) & (s >= 35) & (v >= 60)
    dark = (v < 85) & ~blue
    light = (v >= 150) & (s < 40)
    out[: len(FEATURES)] = [
        lab[:, 0].mean(),
        lab[:, 1].mean() - 128,
        lab[:, 2].mean() - 128,
        lab[:, 0].std(),
        lab[:, 2].std(),
        blue.mean(),
        dark.mean(),
        light.mean(),
        n,
    ]
    hist, _, _ = np.histogram2d(h, s, bins=[H_BINS, S_BINS], range=[[0, 180], [0, 256]])
    out[len(FEATURES) :] = (hist / n).reshape(-1)
    return out, from_kpts
