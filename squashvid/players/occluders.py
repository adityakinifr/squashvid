"""The door's black hardware (hinge plates, handle block) as a time-varying image mask.

The hardware belongs to the open door between the phone and the court, so it
moves in the image whenever the phone pans. A median over ~30 s of sparse
frames removes the moving players and leaves the hardware, found as large
near-black blobs. The handle's silver plate is too close to the floor's colour to
segment, so the black block found beside it is widened over the plate. Glass
edges are not masked: players stay visible through them.

Masks are not shifted per frame: a rotation-only warp from the registered
camera moved them further from the hardware than leaving them alone (the phone
also translates, and the door is only ~1 m from it), so users of the mask
allow a few pixels of slack instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MASK_WIDTH = 512  # masks are kept at this width whatever the video size, so the blob sizes below hold
DARK_V, DARK_S = 62, 80
MIN_AREA = 40  # mask pixels
WINDOW_S, STEP_S = 30.0, 5.0
HANDLE_PLATE = 1.5  # silver plate width as a multiple of the black block's width (clips 1-3)
# How far past a box edge to look for hardware, in 1024-px frame pixels. A window's mask can sit
# ~6 px (1024 scale) inside the hardware's true edge, because the phone drifts within the 30 s
# median and the dark threshold misses the plate's blended rim. Clip 1 at 227 s: the handle's top
# edge at 584 px (1080p), its mask from 596 px; with 8 px of slack a player cut off by the plate read
# as standing at the front wall, 4 m away.
EDGE_SLACK_PX = 16.0
VERSION = 2


class BackgroundSampler:
    """Collects MASK_WIDTH-wide frames about once a second during the main decode pass."""

    def __init__(self, every_s: float = 1.0):
        self.every_s = every_s
        self.times: list[float] = []
        self.frames: list[np.ndarray] = []
        self.scale = 2.0  # full-resolution pixels per mask pixel, set by the first frame

    def offer(self, t: float, image: np.ndarray) -> None:
        if not self.times or t - self.times[-1] >= self.every_s:
            self.scale = image.shape[1] / MASK_WIDTH
            h = int(round(image.shape[0] / self.scale))
            self.times.append(t)
            self.frames.append(cv2.resize(image, (MASK_WIDTH, h), interpolation=cv2.INTER_AREA))


def widen_handle(mask: np.ndarray) -> np.ndarray:
    """Extend the handle's black block (door strip, mid-height) rightward over its silver plate."""
    h, w = mask.shape
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = mask.copy()
    for k in range(1, n):
        cx, cy = cent[k, 0] / w, cent[k, 1] / h
        bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        if 0.33 <= cx < 0.56 and 0.45 <= cy < 0.72 and bw < 3 * bh:
            x0, y0 = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            out[y0 : y0 + bh, x0 : min(x0 + int(round(bw * (1 + HANDLE_PLATE))), w)] = True
    return out


def hardware_mask(background: np.ndarray) -> np.ndarray:
    """Near-black, low-saturation blobs of a MASK_WIDTH background image, handle widened."""
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    dark = ((hsv[..., 2] < DARK_V) & (hsv[..., 1] < DARK_S)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA
    mask = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return widen_handle(mask)


@dataclass
class OccluderTrack:
    times: np.ndarray  # (K,) window centres, seconds
    masks: np.ndarray  # (K, h, MASK_WIDTH) bool
    scale: float = 2.0  # full-resolution pixels per mask pixel

    @classmethod
    def build(cls, sampler: BackgroundSampler, window_s: float = WINDOW_S, step_s: float = STEP_S) -> "OccluderTrack":
        ts = np.asarray(sampler.times)
        stack = np.stack(sampler.frames)
        centres = np.arange(ts[0], ts[-1] + 1e-6, step_s) if len(ts) else np.zeros(0)
        masks = []
        for c in centres:
            sel = np.abs(ts - c) <= window_s / 2
            masks.append(hardware_mask(np.median(stack[sel], axis=0).astype(np.uint8)))
        return cls(centres, np.stack(masks) if masks else np.zeros((0, 1, 1), bool), sampler.scale)

    def save(self, path: str | Path) -> Path:
        np.savez_compressed(path, times=self.times, masks=np.packbits(self.masks, axis=-1), width=self.masks.shape[-1], scale=self.scale, version=VERSION)
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "OccluderTrack":
        """Load, upgrading masks written before the handle was widened."""
        d = np.load(path)
        masks = np.unpackbits(d["masks"], axis=-1)[..., : int(d["width"])].astype(bool)
        version = int(d["version"]) if "version" in d else 1
        if version < VERSION and len(masks):
            masks = np.stack([widen_handle(m) for m in masks])
        return cls(d["times"], masks, float(d["scale"]) if "scale" in d else 2.0)

    def at(self, t: float) -> np.ndarray:
        """Mask nearest in time."""
        return self.masks[int(np.argmin(np.abs(self.times - t)))]

    def fraction(self, t: float, x0: float, y0: float, x1: float, y1: float) -> float:
        """Share of a full-resolution rectangle covered by hardware."""
        m, s = self.at(t), self.scale
        a, b = int(max(x0, 0) // s), int(np.ceil(max(x1, 0) / s))
        c, d = int(max(y0, 0) // s), int(np.ceil(max(y1, 0) / s))
        patch = m[c : max(d, c + 1), a : max(b, a + 1)]
        return float(patch.mean()) if patch.size else 0.0

    def under_box(self, t: float, box: np.ndarray, px: float = 1.0) -> float:
        """Share of the strip just under a box's bottom edge (middle 60 % of its width) covered by hardware.

        The strip runs from 4 px above the edge to EDGE_SLACK_PX below it (or 12 % of the box height,
        if larger); px scales those 1024-frame pixels to the video, see geometry.px_scale.
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        return self.fraction(t, x1 + 0.2 * w, y2 - 4.0 * px, x2 - 0.2 * w, y2 + max(EDGE_SLACK_PX * px, 0.12 * h))

    def contains(self, t: float, uv: np.ndarray) -> np.ndarray:
        """Whether each full-resolution pixel lies on hardware."""
        m = self.at(t)
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        i = np.clip((uv[:, 1] // self.scale).astype(int), 0, m.shape[0] - 1)
        j = np.clip((uv[:, 0] // self.scale).astype(int), 0, m.shape[1] - 1)
        return m[i, j]


def hardware_bands(track: OccluderTrack, width: int = 1024) -> dict[str, tuple[float, float]]:
    """Image x-ranges (full resolution) of the door strip (handle block plus the glass edge left of it) and
    the hinge column, from hardware present in most of the clip's masks.

    Named by position: the door handle sits at ~0.40-0.50 of the width, the
    hinge plates at ~0.62-0.71; brackets on the outer panels are ignored.
    """
    if not len(track.masks):
        return {}
    common = track.masks.mean(0) > 0.5
    n, _, stats, cent = cv2.connectedComponentsWithStats(common.astype(np.uint8))
    bands: dict[str, list[float]] = {}
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        cx = cent[k, 0] * track.scale / width
        name = "door_strip" if 0.33 <= cx < 0.56 else "hinges" if 0.56 <= cx < 0.78 else None
        if name is None:
            continue
        x0, x1 = stats[k, cv2.CC_STAT_LEFT] * track.scale, (stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH]) * track.scale
        lo, hi = bands.get(name, [x0, x1])
        bands[name] = [min(lo, x0), max(hi, x1)]
    if "door_strip" in bands:
        bands["door_strip"][0] -= 10 * track.scale  # the glass edge just left of the handle block
    return {k: (float(a), float(b)) for k, (a, b) in bands.items()}
