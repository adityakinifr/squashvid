"""Skeleton layouts and the COCO-17 -> H36M-17 mapping the lifter expects.

H36M-17 order (MotionBERT / VideoPose3D): 0 pelvis, 1 R hip, 2 R knee, 3 R ankle, 4 L hip, 5 L knee, 6 L ankle,
7 spine, 8 thorax, 9 neck/nose, 10 head top, 11 L shoulder, 12 L elbow, 13 L wrist, 14 R shoulder, 15 R elbow, 16 R wrist.
"""

from __future__ import annotations

import numpy as np

H36M_NAMES = ("pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine", "thorax", "nose",
              "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist")
PARENT = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
# Left/right counterpart of each joint (itself for the midline); mirrored bones share one length.
MIRROR = (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13)
R_ANKLE, L_ANKLE = 3, 6
# (H36M joint, COCO index) for the joints that exist in both layouts, used to compare against the detector.
COCO_OF_H36M = {9: 0, 11: 5, 14: 6, 12: 7, 15: 8, 13: 9, 16: 10, 4: 11, 1: 12, 5: 13, 2: 14, 6: 15, 3: 16}
# Head top is not a COCO keypoint: this far beyond the ear midpoint (or nose) along thorax -> ear/nose.
HEAD_EXTRA_EARS, HEAD_EXTRA_NOSE = 0.45, 0.6


def bones() -> list[tuple[int, int]]:
    """(parent, child) pairs, parents before children."""
    return [(PARENT[j], j) for j in range(1, 17)]


def coco_to_h36m(kp: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,17,2) pixels and (T,17) scores in COCO order -> the same in H36M order; derived joints take the weaker score."""
    kp = np.asarray(kp, np.float64)
    sc = np.asarray(sc, np.float64)
    out = np.full(kp.shape, np.nan)
    conf = np.zeros(sc.shape)
    out[:, 1], conf[:, 1] = kp[:, 12], sc[:, 12]
    out[:, 2], conf[:, 2] = kp[:, 14], sc[:, 14]
    out[:, 3], conf[:, 3] = kp[:, 16], sc[:, 16]
    out[:, 4], conf[:, 4] = kp[:, 11], sc[:, 11]
    out[:, 5], conf[:, 5] = kp[:, 13], sc[:, 13]
    out[:, 6], conf[:, 6] = kp[:, 15], sc[:, 15]
    out[:, 11], conf[:, 11] = kp[:, 5], sc[:, 5]
    out[:, 12], conf[:, 12] = kp[:, 7], sc[:, 7]
    out[:, 13], conf[:, 13] = kp[:, 9], sc[:, 9]
    out[:, 14], conf[:, 14] = kp[:, 6], sc[:, 6]
    out[:, 15], conf[:, 15] = kp[:, 8], sc[:, 8]
    out[:, 16], conf[:, 16] = kp[:, 10], sc[:, 10]
    out[:, 9], conf[:, 9] = kp[:, 0], sc[:, 0]
    out[:, 0] = (kp[:, 11] + kp[:, 12]) / 2
    conf[:, 0] = np.minimum(sc[:, 11], sc[:, 12])
    out[:, 8] = (kp[:, 5] + kp[:, 6]) / 2
    conf[:, 8] = np.minimum(sc[:, 5], sc[:, 6])
    out[:, 7] = (out[:, 0] + out[:, 8]) / 2
    conf[:, 7] = np.minimum(conf[:, 0], conf[:, 8])
    ears = (kp[:, 3] + kp[:, 4]) / 2
    ears_ok = np.minimum(sc[:, 3], sc[:, 4]) >= 0.3
    ears_ok &= np.isfinite(ears).all(axis=1)
    top_e = ears + HEAD_EXTRA_EARS * (ears - out[:, 8])
    top_n = kp[:, 0] + HEAD_EXTRA_NOSE * (kp[:, 0] - out[:, 8])
    out[:, 10] = np.where(ears_ok[:, None], top_e, top_n)
    conf[:, 10] = np.where(ears_ok, np.minimum(np.minimum(sc[:, 3], sc[:, 4]), conf[:, 8]), np.minimum(sc[:, 0], conf[:, 8]))
    return out, np.where(np.isfinite(out).all(axis=2), conf, 0.0)
