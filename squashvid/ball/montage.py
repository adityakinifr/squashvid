"""Zoomed crop montages for checking detections and labels by eye.

A 2 px ball is invisible in a full frame, so review happens on nearest-neighbour
zoomed crops with a pixel grid, small ring markers that never cover the ball,
and the frame index burned in.
"""

from __future__ import annotations

import cv2
import numpy as np

COLOURS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]


def crop_zoom(img: np.ndarray, cx: float, cy: float, half: int = 24, zoom: int = 6, grid: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a (2*half)^2 window centred on (cx, cy), zoomed with visible pixel blocks.

    Returns the tile and the crop's top-left frame pixel, so callers can map tile
    pixels back with x = x0 + u / zoom - 0.5.
    """
    H, W = img.shape[:2]
    x0 = int(np.clip(round(cx) - half, 0, W - 2 * half))
    y0 = int(np.clip(round(cy) - half, 0, H - 2 * half))
    tile = cv2.resize(img[y0 : y0 + 2 * half, x0 : x0 + 2 * half], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    if grid:
        for k in range(0, 2 * half + 1, grid):
            c = (200, 200, 200) if (x0 + k) % (4 * grid) else (120, 120, 120)
            cv2.line(tile, (k * zoom, 0), (k * zoom, tile.shape[0]), c, 1)
            c = (200, 200, 200) if (y0 + k) % (4 * grid) else (120, 120, 120)
            cv2.line(tile, (0, k * zoom), (tile.shape[1], k * zoom), c, 1)
    return tile, (x0, y0)


def mark(tile: np.ndarray, origin: tuple[int, int], zoom: int, x: float, y: float, colour, radius_px: float = 3.5, label: str | None = None) -> None:
    """Draw a ring of radius_px frame pixels around (x, y) so the ball stays visible inside it."""
    u = int(round((x - origin[0] + 0.5) * zoom))
    v = int(round((y - origin[1] + 0.5) * zoom))
    cv2.circle(tile, (u, v), int(radius_px * zoom), colour, 1, cv2.LINE_AA)
    if label:
        cv2.putText(tile, label, (u + int(radius_px * zoom) + 2, v - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


def caption(tile: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((18, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    pad = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 4, 0, w - t.shape[1] + 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    rows = [np.hstack(pad[i : i + cols] + [np.full_like(pad[0], 255)] * (cols - len(pad[i : i + cols]))) for i in range(0, len(pad), cols)]
    return np.vstack(rows)
