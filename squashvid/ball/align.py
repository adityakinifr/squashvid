"""Match WhatsApp-copy frames to the original frames that hold the same picture.

The copy was resampled to a constant 29.97 fps, so its frame k shows an original
frame whose PTS lies 0-33 ms earlier, with a lag that drifts across the clip. A
one-frame slip moves a fast ball 10-35 px, far beyond the 4 px tolerance, so the
match is made on content: among the original frames within one frame period of
the copy's PTS, take the one whose downscaled image differs least from the copy,
looking only at pixels that change between those candidates (players, ball),
since the static court cannot tell them apart.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from squashvid.ball.clips import COPIES, frame_range, video

THUMB = (256, 144)


def _thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def _normalise(g: np.ndarray) -> np.ndarray:
    # The original's SDR conversion and Apple's copy differ in exposure and contrast; compare standardised images.
    return (g - g.mean()) / (g.std() + 1e-6)


def best_match(copy_thumb: np.ndarray, cand_thumbs: list[np.ndarray]) -> tuple[int, float]:
    """Index of the candidate closest to the copy over pixels that vary among candidates, and its margin."""
    c = _normalise(copy_thumb)
    cands = [_normalise(t) for t in cand_thumbs]
    stack = np.stack(cands)
    mask = stack.std(axis=0) > 0.05 if len(cands) > 1 else np.ones(c.shape, bool)
    if mask.sum() < 20:
        mask = np.ones(c.shape, bool)
    err = np.array([float(np.mean((t[mask] - c[mask]) ** 2)) for t in cands])
    order = np.argsort(err)
    margin = float(err[order[1]] - err[order[0]]) if len(err) > 1 else float("inf")
    return int(order[0]), margin


def match_window(original: str, start: float, end: float, slack: float = 0.04) -> pd.DataFrame:
    """For each copy frame with PTS in [start, end]: the matching original frame, its PTS and the margin."""
    copy = COPIES[original]
    vo, vc = video(original), video(copy)
    to = vo.timestamps
    orig = {idx: _thumb(img) for idx, _, img in vo.frames(max(0.0, start - 2 * slack), end + slack)}
    rows = []
    for k, tc, img in vc.frames(start, end):
        lo, hi = frame_range(to, tc - 0.034 - slack / 4, tc + slack / 4)
        cand = [i for i in range(lo, hi + 1) if i in orig]
        if not cand:
            continue
        j, margin = best_match(_thumb(img), [orig[i] for i in cand])
        rows.append({"copy_frame": k, "copy_t": tc, "frame": cand[j], "t": float(to[cand[j]]), "lag": tc - float(to[cand[j]]), "margin": margin, "n_cand": len(cand)})
    return pd.DataFrame(rows)
