"""Contact sheets for checking tracks and events by eye: a strip of zoomed frames around each event.

Each tile is a nearest-neighbour crop around the event pixel with the frame index and the track point
of that frame ringed; the event's own pixel is marked with a cross on the frames it falls in.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squashvid.ball import montage
from squashvid.ball.clips import video


def _tiles(stem: str, centre: tuple[float, float], frames: list[int], track: pd.DataFrame | None, mark_at: float | None, half: int, zoom: int):
    v = video(stem)
    ts = v.timestamps
    t0, t1 = float(ts[max(frames[0], 0)]), float(ts[min(frames[-1], len(ts) - 1)])
    want = set(frames)
    got = {}
    for idx, _, img in v.frames(t0 - 1e-4, t1 + 1e-4):
        if idx in want:
            got[idx] = img
        if idx >= frames[-1]:
            break
    tiles = []
    by_frame = track.set_index("frame") if track is not None and len(track) else None
    for f in frames:
        if f not in got:
            continue
        tile, org = montage.crop_zoom(got[f], centre[0], centre[1], half=half, zoom=zoom, grid=0)
        if by_frame is not None and f in by_frame.index and np.isfinite(by_frame.at[f, "u"]):
            row = by_frame.loc[f]
            col = (0, 0, 255) if row.state != "interpolated" else (0, 200, 255)
            montage.mark(tile, org, zoom, float(row.u), float(row.v), col, radius_px=4.0)
        if mark_at is not None and abs(f - mark_at) < 0.5:
            u = int(round((centre[0] - org[0] + 0.5) * zoom))
            v_ = int(round((centre[1] - org[1] + 0.5) * zoom))
            cv2.drawMarker(tile, (u, v_), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        tiles.append(montage.caption(tile, f"{stem} f{f}"))
    return tiles


def event_sheet(stem: str, event: pd.Series, track: pd.DataFrame | None, out: Path, before: int = 4, after: int = 5, half: int = 32, zoom: int = 3, cols: int = 5) -> Path:
    f0 = int(np.floor(event["frame"]))
    frames = list(range(f0 - before, f0 + after + 1))
    tiles = _tiles(stem, (float(event["u"]), float(event["v"])), frames, track, float(event["frame"]), half, zoom)
    sheet = montage.grid(tiles, cols)
    head = np.full((22, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{event['kind']} f{event['frame']:.2f} ev={event.get('evidence', '')} conf={event.get('confidence', float('nan')):.2f}",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, sheet]))
    return out
