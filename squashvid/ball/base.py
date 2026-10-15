"""Shared machinery for temporal heatmap ball detectors.

Every model here maps a short stack of frames to one heatmap per output frame.
They differ in stack length, which frames they predict, how overlapping
predictions are merged and how a heatmap becomes a point, so a detector only
declares those pieces (`plan`, `forward`, `decode`, `select`) and this module
streams the video through it. Frames are decoded once, kept only while a
pending stack needs them, and every detection is mapped back to full-frame
pixel coordinates (pixel centres at integers) whatever crop or input size the
network ran at.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import torch

from squashvid.ball.clips import frame_range

COLUMNS = ["frame", "t", "x", "y", "score", "visible"]
DEFAULT_SIZE = (288, 512)  # (H, W) every paper trains at


@dataclass(frozen=True)
class Geometry:
    """A crop of the frame (roi, end-exclusive) resized to the network input (H, W)."""

    roi: tuple[int, int, int, int]
    size: tuple[int, int]

    @property
    def scale(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.roi
        return (x1 - x0) / self.size[1], (y1 - y0) / self.size[0]

    def warp(self, img: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.roi
        crop = img[y0:y1, x0:x1]
        h, w = self.size
        if crop.shape[:2] == (h, w):
            return crop
        # Area averaging keeps a 2 px ball's ink when shrinking; cubic when enlarging.
        interp = cv2.INTER_AREA if w < crop.shape[1] else cv2.INTER_CUBIC
        return cv2.resize(crop, (w, h), interpolation=interp)

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        sx, sy = self.scale
        return self.roi[0] + (x + 0.5) * sx - 0.5, self.roi[1] + (y + 0.5) * sy - 0.5


def resolve_geometry(frame_hw: tuple[int, int], size: str | tuple[int, int], roi: tuple[int, int, int, int] | None, multiple: int = 32) -> Geometry:
    """'default' = the papers' 288x512, 'native' = 1:1 pixels, 'x2' = 2x upscale, or an explicit (H, W) / "HxW"."""
    H, W = frame_hw
    roi = roi or (0, 0, W, H)
    rh, rw = roi[3] - roi[1], roi[2] - roi[0]

    def snap(v: float) -> int:
        return max(multiple, int(round(v / multiple)) * multiple)

    if size == "default":
        hw = DEFAULT_SIZE
    elif size == "native":
        hw = (snap(rh), snap(rw))
    elif size == "x2":
        hw = (snap(2 * rh), snap(2 * rw))
    elif isinstance(size, str):
        hw = tuple(int(v) for v in size.split("x"))
    else:
        hw = tuple(int(v) for v in size)
    return Geometry(tuple(int(v) for v in roi), hw)


@dataclass
class Clip:
    """One network call: the frame index feeding each input slot, and which outputs to keep."""

    frames: list[int]
    outputs: list[tuple[int, int, float]]  # (output channel, frame index, ensemble weight)


def plan_blocks(i0: int, i1: int, n: int, step: int, lo: int, hi: int, weights: np.ndarray | None = None) -> list[Clip]:
    """Stacks of n consecutive frames every `step` frames, covering [i0, i1] at least once.

    step == n tiles the range (the papers' fast mode); step == 1 gives every frame n
    predictions to ensemble. Indices are clamped to [lo, hi], repeating edge frames
    exactly as the official loaders pad.
    """
    starts = list(range(i0 - (n - 1 if step == 1 else 0), i1 + 1, step))
    if starts and starts[-1] + n - 1 < i1:
        starts.append(i1 - n + 1)
    w = np.ones(n) if weights is None else weights
    clips = []
    for s in starts:
        frames = [min(max(s + k, lo), hi) for k in range(n)]
        outs, seen = [], set()
        for k, f in enumerate(frames):
            if f in seen or not (i0 <= f <= i1) or s + k != f:
                continue
            seen.add(f)
            outs.append((k, f, float(w[k])))
        if outs:
            clips.append(Clip(frames, outs))
    return clips


def plan_causal(i0: int, i1: int, n: int, lo: int) -> list[Clip]:
    """One stack per frame ending at that frame (TOTNet predicts the last of its n frames)."""
    return [Clip([max(f - n + 1 + k, lo) for k in range(n)], [(0, f, 1.0)]) for f in range(i0, i1 + 1)]


def blobs(hm: np.ndarray, thr: float) -> list[tuple[float, float, float, float, int]]:
    """Connected components above thr: (x, y, peak, mass, bbox area), centroid weighted by the heatmap.

    This is the WASB/BlurBall post-processor; TrackNetV3 instead keeps the blob with
    the largest bounding box and uses its centre.
    """
    mask = (hm > thr).astype(np.uint8)
    if not mask.any():
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for m in range(1, n):
        x, y, w, h, _ = stats[m]
        sub = hm[y : y + h, x : x + w] * (labels[y : y + h, x : x + w] == m)
        mass = float(sub.sum())
        ys, xs = np.mgrid[y : y + h, x : x + w]
        out.append((float((xs * sub).sum() / mass), float((ys * sub).sum() / mass), float(sub.max()), mass, int(w * h)))
    return out


def track_online(cands: list[list[tuple[float, float, float, float]]], max_disp: float = 300.0) -> list[tuple[float, float, float, bool]]:
    """WASB's online tracker: drop candidates too far from the last visible point, keep the highest mass.

    Candidates are (x, y, peak, mass) in frame pixels; returns (x, y, peak, visible).
    """
    out, last = [], None
    for frame_cands in cands:
        if last is not None:
            frame_cands = [c for c in frame_cands if np.hypot(c[0] - last[0], c[1] - last[1]) < max_disp]
        if frame_cands:
            best = max(frame_cands, key=lambda c: c[3])
            out.append((best[0], best[1], best[2], True))
            last = best
        else:
            out.append((np.nan, np.nan, 0.0, False))
            last = None
    return out


@dataclass
class RunStats:
    frames: int = 0
    forwards: int = 0
    decode_s: float = 0.0
    infer_s: float = 0.0
    post_s: float = 0.0
    extra: dict = field(default_factory=dict)


class Detector:
    """Base class: subclasses set name/frames_in and implement build, tensor, forward, plan, decode, select."""

    name = "detector"
    frames_in = 3
    merge = "pool"  # "pool": union of per-heatmap candidates; "mean": average heatmaps first
    threshold = 0.5
    batch_default = {"default": 8, "native": 4}

    def __init__(self, weights: str, size: str | tuple[int, int] = "default", roi: tuple[int, int, int, int] | None = None, device: str | None = None, half: bool = False, frame_hw: tuple[int, int] = (576, 1024)):
        self.weights = weights
        self.size_mode = size
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.dtype = torch.float16 if half else torch.float32
        self.frame_hw = frame_hw
        self.geom = resolve_geometry(frame_hw, size, roi)
        self.model = self.build().to(self.device, self.dtype).eval()
        self.stats = RunStats()

    @property
    def variant(self) -> str:
        """Stable id used for output file names, e.g. wasb-tennis@native or blurball@x2-front."""
        tag = self.size_mode if isinstance(self.size_mode, str) else "x".join(map(str, self.size_mode))
        full = self.geom.roi == (0, 0, self.frame_hw[1], self.frame_hw[0])
        return f"{self.name}@{tag}" + ("" if full else f"-roi{'_'.join(map(str, self.geom.roi))}")

    # --- subclass hooks -------------------------------------------------
    def build(self) -> torch.nn.Module:
        raise NotImplementedError

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """One warped RGB uint8 frame to a CHW float32 array."""
        return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0

    def tensor(self, frames: list[np.ndarray]) -> np.ndarray:
        """Stack one clip's preprocessed frames into the model's input layout (no batch axis)."""
        return np.concatenate(frames, axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch of inputs to per-output-channel probability maps (B, C, H, W)."""
        raise NotImplementedError

    def plan(self, i0: int, i1: int, lo: int, hi: int) -> list[Clip]:
        raise NotImplementedError

    def prepare(self, video, lo: int, hi: int) -> None:
        """Called once per run before streaming (TrackNetV3 builds its median background here)."""

    def decode(self, hm: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Heatmap (input pixels) to candidates (x, y, peak, mass) in input pixels."""
        return [(x, y, p, m) for x, y, p, m, _ in blobs(hm, self.threshold)]

    def select(self, cands: list[list[tuple[float, float, float, float]]]) -> list[tuple[float, float, float, bool]]:
        return track_online(cands)

    # --- driver ----------------------------------------------------------
    def run(self, video, start: float, end: float, batch_size: int | None = None) -> pd.DataFrame:
        """Detect the ball in every frame whose PTS lies in [start, end]."""
        ts = video.timestamps
        i0, i1 = frame_range(ts, start, end)
        clips = self.plan(i0, i1, 0, len(ts) - 1)
        lo = min(min(c.frames) for c in clips)
        hi = max(max(c.frames) for c in clips)
        self.prepare(video, lo, hi)
        bs = batch_size or self.batch_default.get(self.size_mode if isinstance(self.size_mode, str) else "native", 4)
        expected = Counter(f for c in clips for _, f, _ in c.outputs)
        acc: dict[int, list] = {}
        cands: dict[int, list] = {}
        cache: dict[int, np.ndarray] = {}
        pending: list[Clip] = []
        st = RunStats()
        ci = 0
        t_dec = time.perf_counter()
        for idx, _, bgr in video.frames(float(ts[lo]), float(ts[hi])):
            if idx > hi:
                break
            cache[idx] = self.preprocess(cv2.cvtColor(self.geom.warp(bgr), cv2.COLOR_BGR2RGB))
            st.frames += 1
            while ci < len(clips) and max(clips[ci].frames) <= idx:
                pending.append(clips[ci])
                ci += 1
                if len(pending) == bs:
                    st.decode_s += time.perf_counter() - t_dec
                    self._flush(pending, cache, acc, cands, expected, st)
                    t_dec = time.perf_counter()
                    pending = []
            need = min([min(c.frames) for c in pending] + ([min(clips[ci].frames)] if ci < len(clips) else [hi + 1]))
            for k in [k for k in cache if k < need]:
                del cache[k]
        st.decode_s += time.perf_counter() - t_dec
        pending += clips[ci:]
        for k in range(0, len(pending), bs):
            self._flush(pending[k : k + bs], cache, acc, cands, expected, st)
        t0 = time.perf_counter()
        frames = list(range(i0, i1 + 1))
        picked = self.select([cands.get(f, []) for f in frames])
        st.post_s += time.perf_counter() - t0
        self.stats = st
        self.candidates = {f: cands.get(f, []) for f in frames}  # every heatmap blob, for offline selection studies
        return pd.DataFrame(
            {
                "frame": frames,
                "t": ts[i0 : i1 + 1],
                "x": [p[0] for p in picked],
                "y": [p[1] for p in picked],
                "score": [p[2] for p in picked],
                "visible": [bool(p[3]) for p in picked],
            },
            columns=COLUMNS,
        )

    def _frame(self, cache: dict[int, np.ndarray], f: int) -> np.ndarray:
        if f in cache:
            return cache[f]
        # A frame the decoder skipped (variable frame rate): reuse the nearest earlier one.
        keys = [k for k in cache if k <= f] or list(cache)
        return cache[max(keys) if keys[0] <= f else min(keys)]

    def _flush(self, clips: list[Clip], cache, acc, cands, expected, st: RunStats) -> None:
        if not clips:
            return
        x = torch.from_numpy(np.stack([self.tensor([self._frame(cache, f) for f in c.frames]) for c in clips]))
        t0 = time.perf_counter()
        with torch.inference_mode():
            y = self.forward(x.to(self.device, self.dtype)).float().cpu().numpy()
        st.infer_s += time.perf_counter() - t0
        st.forwards += len(clips)
        t0 = time.perf_counter()
        for b, c in enumerate(clips):
            for ch, f, w in c.outputs:
                hm = y[b, ch]
                if self.merge == "mean":
                    s = acc.setdefault(f, [np.zeros_like(hm), 0.0, 0])
                    s[0] += w * hm
                    s[1] += w
                    s[2] += 1
                    done = s[2] == expected[f]
                    if done:
                        hm = s[0] / s[1]
                        del acc[f]
                else:
                    s = acc.setdefault(f, [0])
                    s[0] += 1
                    done = s[0] == expected[f]
                if self.merge == "pool" or done:
                    found = [(*self.geom.to_frame(cx, cy), p, m) for cx, cy, p, m in self.decode(hm)]
                    cands.setdefault(f, []).extend(found)
        st.post_s += time.perf_counter() - t0
