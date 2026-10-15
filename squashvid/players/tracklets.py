"""Short, conservative tracklets from stored detections.

BoT-SORT (motion + IoU, no ReID, no global motion compensation: the camera
pose already absorbs the phone's drift) links boxes frame to frame. Its output
is then cut wherever an identity swap could hide inside a tracklet: where two
boxes touch, where the box height jumps (a near and a far player swapping),
and where the court position jumps faster than a player can run. Identity is
decided later over these pure pieces.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

NMS_IOU = 0.6
PART_HEIGHT = 0.8
CONTACT_GAP_PX = 8.0
HEIGHT_JUMP = 0.25
JUMP_SPEED = 12.0  # m/s between consecutive frames, above any player (gate for splitting only)
JUMP_SLACK = 0.6  # m of foot-point noise allowed on top of that speed

TRACKER_ARGS = dict(
    tracker_type="botsort",
    track_high_thresh=0.3,
    track_low_thresh=0.1,
    new_track_thresh=0.35,
    track_buffer=8,
    match_thresh=0.7,
    fuse_score=True,
    gmc_method="none",
    proximity_thresh=0.5,
    appearance_thresh=0.8,
    with_reid=False,
    model="auto",
)


class _Boxes:
    """The minimal results-like object ultralytics trackers consume."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x = self.xyxy
        return np.column_stack([(x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2, x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]])

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, m) -> "_Boxes":
        return _Boxes(self.xyxy[m], self.conf[m])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def box_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel gap between two boxes: negative when they overlap, inf when they share no rows."""
    if min(a[3], b[3]) <= max(a[1], b[1]):
        return float("inf")
    return float(max(a[0], b[0]) - min(a[2], b[2]))


def abuts_hardware(occ, t: float, box: np.ndarray, px: float = 1.0) -> bool:
    """The box's top or bottom edge lies on door hardware: the detector cut a person at a plate or handle."""
    x1, y1, x2, y2 = box
    a, b = x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)
    return occ.fraction(t, a, y2 - 4 * px, b, y2 + 8 * px) > 0.2 or occ.fraction(t, a, y1 - 8 * px, b, y1 + 4 * px) > 0.2


def is_part(small: np.ndarray, big: np.ndarray, on_hardware: bool) -> bool:
    """small is a piece of the person in big: mostly inside it, clearly shorter, and either
    cut by hardware or almost wholly contained. A far player standing behind a near one keeps
    his own feet, so his box is rarely this deeply nested."""
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    area = max((small[2] - small[0]) * (small[3] - small[1]), 1e-9)
    inside = ix * iy / area
    shorter = (small[3] - small[1]) < PART_HEIGHT * (big[3] - big[1])
    return shorter and ((on_hardware and inside > 0.6) or inside > 0.9)


def candidates(dets: pd.DataFrame, occ=None, px: float = 1.0) -> pd.DataFrame:
    """On-court, plausibly sized boxes with posed keypoints, without YOLO26's near-duplicates
    or the partial boxes the door hardware cuts out of a player. px: see geometry.px_scale."""
    d = dets[dets.on_court & dets.size_ok].copy()
    if "kpts" in d:
        d = d[d.kpts.notna()]
    keep = []
    for _, g in d.groupby("frame", sort=True):
        g = g.sort_values("conf", ascending=False)
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        alive = np.ones(len(g), bool)
        iou = iou_matrix(boxes, boxes)
        for i in range(len(g)):
            if alive[i]:
                alive[i + 1 :] &= iou[i, i + 1 :] < NMS_IOU
        if alive.sum() > 1:
            t = float(g.t.iat[0])
            for i in np.flatnonzero(alive):
                hw = occ is not None and abuts_hardware(occ, t, boxes[i], px)
                if any(is_part(boxes[i], boxes[j], hw) for j in np.flatnonzero(alive) if j != i):
                    alive[i] = False
        keep.extend(g.index[alive])
    return d.loc[sorted(keep)].sort_values(["frame", "conf"], ascending=[True, False]).reset_index(drop=True)


def run_tracker(cand: pd.DataFrame, frames: np.ndarray, args: dict | None = None) -> np.ndarray:
    """BoT-SORT track id per candidate row (-1 when the tracker left it unconfirmed)."""
    from ultralytics.trackers.bot_sort import BOTSORT

    tracker = BOTSORT(SimpleNamespace(**{**TRACKER_ARGS, **(args or {})}))
    ids = np.full(len(cand), -1, np.int64)
    by_frame = {f: g.index.to_numpy() for f, g in cand.groupby("frame")}
    for f in frames:
        rows = by_frame.get(f, np.zeros(0, np.int64))
        boxes = _Boxes(cand.loc[rows, ["x1", "y1", "x2", "y2"]].to_numpy(), cand.loc[rows, "conf"].to_numpy())
        out = tracker.update(boxes, None)
        for r in out:
            ids[rows[int(r[-1])]] = int(r[4])
    return ids


def contact_flags(cand: pd.DataFrame, gap_px: float = CONTACT_GAP_PX) -> np.ndarray:
    """True for boxes overlapping or within gap_px of another candidate in the same frame."""
    flag = np.zeros(len(cand), bool)
    for _, g in cand.groupby("frame"):
        if len(g) < 2:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if box_gap(b[i], b[j]) < gap_px:
                    flag[g.index[i]] = flag[g.index[j]] = True
    return flag


def split_tracklets(cand: pd.DataFrame, track_id: np.ndarray, contact: np.ndarray, court_xy: np.ndarray) -> np.ndarray:
    """Tracklet id per row: tracker ids cut where the tracker skipped frames, at contact
    changes, height jumps, court-position jumps and wherever the number of boxes in the
    frame changes.

    The last cut lets the labeller drop a stray box's overlap without discarding
    the whole tracklet it overlaps. Rows the tracker left unconfirmed become
    single-row tracklets.
    """
    out = np.full(len(cand), -1, np.int64)
    next_id = 0
    h = (cand.y2 - cand.y1).to_numpy()
    t = cand.t.to_numpy()
    count = cand.groupby("frame").frame.transform("size").to_numpy()
    frame = cand.frame.to_numpy()
    for tid in np.unique(track_id):
        rows = np.flatnonzero(track_id == tid)
        if tid < 0:
            out[rows] = np.arange(next_id, next_id + len(rows))
            next_id += len(rows)
            continue
        rows = rows[np.argsort(frame[rows])]
        out[rows[0]] = next_id
        for p, q in zip(rows[:-1], rows[1:]):
            dt = max(t[q] - t[p], 1 / 30)
            jump = np.linalg.norm(court_xy[q] - court_xy[p])
            cut = (
                frame[q] - frame[p] > 1
                or contact[p] != contact[q]
                or count[p] != count[q]
                or abs(h[q] - h[p]) / max(h[p], 1.0) > HEIGHT_JUMP
                or not np.isfinite(jump)
                or jump > JUMP_SPEED * dt + JUMP_SLACK
            )
            if cut:
                next_id += 1
            out[q] = next_id
        next_id += 1
    return out
