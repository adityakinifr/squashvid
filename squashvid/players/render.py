"""Pictures for checking the player stream by eye: overlays, crossing contact sheets, top views.

Player A is drawn in categorical slot 1 (blue, matching the light-blue shirt)
and player B in slot 2 (orange); the pair passes the colour-vision checks of
the shared chart palette. Every picture says it is a pilot, unvalidated.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squashvid.geom.court import CourtSpec, court_lines
from squashvid.media.video import Video
from squashvid.players.metrics import Event

HEX = {"A": "#2a78d6", "B": "#eb6834"}
BGR = {p: tuple(int(h[i : i + 2], 16) for i in (5, 3, 1)) for p, h in HEX.items()}
OUT_WIDTH = 1024  # overlay videos and check images never exceed this width
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
NAMES = {"A": "A (light-blue shirt)", "B": "B (dark shirt)"}


def _dashed_rect(img: np.ndarray, box: np.ndarray, colour: tuple, dash: int = 6) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash), 1)
        for k in range(0, n, 2):
            p = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            q = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, p, q, colour, 1, cv2.LINE_AA)


def draw_players(img: np.ndarray, rows: dict[str, dict], scale: float = 1.0, offset: tuple[float, float] = (0, 0)) -> None:
    """Boxes, IDs and foot points. rows[p] has box (4,), detected, foot (2,) or NaN, src."""
    ox, oy = offset
    for p, r in rows.items():
        box = (np.asarray(r["box"], np.float64) - [ox, oy, ox, oy]) * scale
        if not np.isfinite(box).all():
            continue
        c = BGR[p]
        if r["detected"]:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), c, 2)
        else:
            _dashed_rect(img, box, c)
        label = p if r["detected"] else f"{p}?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = int(box[0]), int(max(box[1] - 3, th + 2))
        cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + 2), c, -1)
        cv2.putText(img, label, (x + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        foot = np.asarray(r.get("foot", (np.nan, np.nan)), np.float64)
        if np.isfinite(foot).all():
            f = ((foot - [ox, oy]) * scale).astype(int)
            cv2.circle(img, tuple(f), 3, c, -1, cv2.LINE_AA)


def frame_rows(w: pd.DataFrame, frame: int, pred: dict[str, np.ndarray], pos: dict[int, int]) -> dict[str, dict]:
    i = pos[frame]
    out = {}
    for p in ("A", "B"):
        det = bool(w[f"{p}_detected"].iat[i]) if pd.notna(w[f"{p}_detected"].iat[i]) else False
        out[p] = {"box": pred[p][i], "detected": det, "foot": (w[f"{p}_fu"].iat[i], w[f"{p}_fv"].iat[i])}
    return out


class ContactSheet:
    """Seven crops of one event (1 s and 0.5 s before, start, middle, end, 0.5 s and 1 s after), IDs drawn.

    Fed frame by frame from one decode pass (see decode_pass), never by seeking.
    """

    def __init__(self, video: Video, w: pd.DataFrame, pred: dict[str, np.ndarray], event: Event, title: str, tile_h: int = 220):
        self.w, self.pred, self.event, self.title, self.tile_h = w, pred, event, title, tile_h
        self.pos = {f: i for i, f in enumerate(w.index.to_numpy())}
        ts = video.timestamps
        times = [event.t0 - 1.0, event.t0 - 0.5, event.t0, (event.t0 + event.t1) / 2, event.t1, event.t1 + 0.5, event.t1 + 1.0]
        want = sorted({int(min(np.searchsorted(ts, t - 1e-6), len(ts) - 1)) for t in times})
        self.want = [f for f in want if f in self.pos]
        self.ts = ts
        lo, hi = self.pos[self.want[0]], self.pos[self.want[-1]]
        boxes = np.concatenate([pred["A"][lo : hi + 1], pred["B"][lo : hi + 1]])
        boxes = boxes[np.isfinite(boxes).all(1)]
        H, W = video.info.height, video.info.width
        m = W / 1024  # margins were set on the 1024-px copies
        if len(boxes):
            self.rect = (max(boxes[:, 0].min() - 25 * m, 0), max(boxes[:, 1].min() - 20 * m, 0), min(boxes[:, 2].max() + 25 * m, W), min(boxes[:, 3].max() + 15 * m, H))
        else:
            self.rect = (0, 0, W, H)
        self.tiles: dict[int, np.ndarray] = {}

    def frames(self) -> set[int]:
        return set(self.want)

    def add(self, idx: int, img: np.ndarray) -> None:
        x0, y0, x1, y1 = (int(v) for v in self.rect)
        scale = self.tile_h / max(y1 - y0, 1)
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        draw_players(crop, frame_rows(self.w, idx, self.pred, self.pos), scale, (x0, y0))
        rel = self.ts[idx] - self.event.t0
        cv2.putText(crop, f"{rel:+.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(crop, f"{rel:+.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.tiles[idx] = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(252, 252, 251))

    def image(self) -> np.ndarray:
        tiles = [self.tiles[f] for f in self.want if f in self.tiles]
        strip = np.hstack(tiles) if tiles else np.full((self.tile_h, 400, 3), 252, np.uint8)
        head = np.full((24, strip.shape[1], 3), 252, np.uint8)
        cv2.putText(head, self.title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
        return np.vstack([head, strip])


def decode_pass(video: Video, consumers: list) -> None:
    """Feed every consumer the frames it asks for (consumer.frames() -> set of indices; consumer.add(idx, img)),
    from one sequential decode that runs to the end of the clip.

    Running to the end is deliberate: closing a Video.frames generator early (a
    bounded window or frame_at) intermittently deadlocks in PyAV's codec teardown
    with frame-threaded decoding; on the HEVC originals it hung within a few calls.
    """
    need: dict[int, list] = {}
    for c in consumers:
        for f in c.frames():
            need.setdefault(int(f), []).append(c)
    if not need:
        return
    first = min(need)
    for idx, _, img in video.frames(max(video.timestamps[first] - 1e-3, 0.0)):
        for c in need.get(idx, ()):
            c.add(idx, img.copy() if len(need[idx]) > 1 else img)


def stack_sheets(sheets: list[np.ndarray]) -> np.ndarray:
    width = max(s.shape[1] for s in sheets)
    return np.vstack([cv2.copyMakeBorder(s, 0, 4, 0, width - s.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 251)) for s in sheets])


def _inset(spec: CourtSpec, trails: dict[str, np.ndarray], size: tuple[int, int] = (120, 190)) -> np.ndarray:
    """A small top view, front wall at the top, with recent trails."""
    w, h = size
    pad = 8
    img = np.full((h + 2 * pad, w + 2 * pad, 3), 252, np.uint8)
    sx, sy = w / spec.width, h / spec.length

    def px(xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return np.column_stack([pad + xy[:, 0] * sx, pad + (spec.length - xy[:, 1]) * sy]).round().astype(np.int32)

    cv2.rectangle(img, (pad, pad), (pad + w, pad + h), (183, 194, 195), 1)
    for ln in court_lines(spec):
        if ln.surface == "floor":
            a, b = px([ln.a[:2], ln.b[:2]])
            cv2.line(img, tuple(a), tuple(b), (183, 194, 195), 1)
    for p, tr in trails.items():
        tr = tr[np.isfinite(tr).all(1)]
        if len(tr) > 1:
            cv2.polylines(img, [px(tr)], False, BGR[p], 1, cv2.LINE_AA)
        if len(tr):
            cv2.circle(img, tuple(px(tr[-1])[0]), 4, BGR[p], -1, cv2.LINE_AA)
    return img


class OverlayVideo:
    """MP4 of a time window with boxes, IDs, foot points and a top-view inset (H.264 when ffmpeg is present).

    Fed by decode_pass; call close() afterwards to finish the file.
    """

    def __init__(self, video: Video, w: pd.DataFrame, pred: dict[str, np.ndarray], spec: CourtSpec, t0: float, t1: float, path: Path, trail_s: float = 1.5):
        self.w, self.pred, self.spec, self.path, self.trail_s = w, pred, spec, path, trail_s
        self.pos = {f: i for i, f in enumerate(w.index.to_numpy())}
        ts = video.timestamps
        f0, f1 = int(np.searchsorted(ts, t0 - 1e-6)), int(np.searchsorted(ts, t1 + 1e-6))
        self.want = [f for f in range(f0, f1) if f in self.pos]
        self.ts = ts
        self.tmp = path.with_suffix(".mp4v.mp4")
        self.s = min(1.0, OUT_WIDTH / video.info.width)  # 1080p originals are drawn at the copies' size so marks stay legible
        self.size = (int(round(video.info.width * self.s)), int(round(video.info.height * self.s)))
        self.out = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, self.size)
        self.t = w.t.to_numpy()
        self.xy = {p: w[[f"{p}_sx", f"{p}_sy"]].to_numpy() for p in ("A", "B")}

    def frames(self) -> set[int]:
        return set(self.want)

    def add(self, idx: int, img: np.ndarray) -> None:
        i = self.pos[idx]
        ti = self.ts[idx]
        if self.s < 1.0:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        draw_players(img, frame_rows(self.w, idx, self.pred, self.pos), self.s)
        j = int(np.searchsorted(self.t, ti - self.trail_s))
        ins = _inset(self.spec, {p: self.xy[p][j : i + 1] for p in ("A", "B")})
        img[6 : 6 + ins.shape[0], 6 : 6 + ins.shape[1]] = ins
        cv2.putText(img, f"t={ti:6.2f}s  pilot, unvalidated", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.out.write(img)

    def close(self) -> Path:
        self.out.release()
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(self.path)], check=True)
            self.tmp.unlink()
        except (OSError, subprocess.CalledProcessError):
            self.tmp.rename(self.path)
        return self.path


def topview_png(table: pd.DataFrame, spec: CourtSpec, t0: float, t1: float, path: Path, title: str, stats: dict | None = None) -> Path:
    """Both players' smoothed trajectories over one window on a to-scale court, front wall at the top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 8.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.add_patch(plt.Rectangle((0, 0), spec.width, spec.length, fill=False, lw=1.0, ec="#c3c2b7"))
    for ln in court_lines(spec):
        if ln.surface == "floor":
            ax.plot([ln.a[0], ln.b[0]], [ln.a[1], ln.b[1]], color="#c3c2b7", lw=1.0, zorder=1)
    T = spec.t_point
    ax.add_patch(plt.Circle((T[0], T[1]), 1.5, fill=False, ls=(0, (3, 3)), lw=0.8, ec=MUTED, zorder=1))
    ax.annotate("T, 1.5 m", (T[0] + 1.06, T[1] + 1.06), color=MUTED, fontsize=7, ha="left", va="bottom")
    for p in ("A", "B"):
        tp = table[(table.player == p) & (table.t >= t0) & (table.t <= t1)]
        xy = tp[["sx", "sy"]].to_numpy()
        ok = np.isfinite(xy).all(1)
        if not ok.any():
            continue
        xy = xy[ok]
        ax.plot(xy[:, 0], xy[:, 1], color=HEX[p], lw=2.0, solid_capstyle="round", zorder=3, label=NAMES[p])
        ax.plot(*xy[0], marker="o", ms=8, mfc=SURFACE, mec=HEX[p], mew=2, zorder=4)
        ax.plot(*xy[-1], marker="s", ms=8, mfc=HEX[p], mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{p} end", xy[-1], xytext=(6, -10 if p == "A" else 6), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlim(-0.3, spec.width + 0.3)
    ax.set_ylim(-0.3, spec.length + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("across court (m, from left wall)", color=INK2, fontsize=8)
    ax.set_ylabel("from back wall (m)", color=INK2, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(spec.width / 2, spec.length + 0.12, "front wall", ha="center", va="bottom", color=MUTED, fontsize=7)
    ax.text(spec.width / 2, -0.12, "back wall (camera side)", ha="center", va="top", color=MUTED, fontsize=7)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=2, frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.6)
    sub = f"{t0:.1f}-{t1:.1f} s · circle = start, square = end · pilot, unvalidated"
    if stats:
        sub += "\n" + "   ".join(f"{p}: {s['distance_m']:.0f} m, {100 * s['share_within_1.5m_of_T']:.0f}% within 1.5 m of T" for p, s in stats.items())
    ax.set_title(f"{title}\n", color=INK, fontsize=10, loc="left", pad=14)
    ax.text(0, 1.012, sub, transform=ax.transAxes, color=INK2, fontsize=7, va="bottom")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
