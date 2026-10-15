"""The one heavy pass over a clip: decode, detect, court-filter, pose and torso colour for every frame.

Everything downstream (tracklets, identity, foot point, smoothing) is cheap and
reruns from the stored detections, so this pass keeps every person box with
its flags, plus keypoints and colour for the boxes that stand on the court.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from squashvid.calib.venue import Venue
from squashvid.media.video import Video
from squashvid.players.appearance import torso_features
from squashvid.players.detect import court_filter, detect_batch, load_detector
from squashvid.players.geometry import CameraTrack
from squashvid.players.occluders import BackgroundSampler, OccluderTrack
from squashvid.players.pose import PoseModel

BATCH = 8
MAX_POSE = 4  # on-court candidates per frame given pose; more means reflections or clutter
PREFETCH = 48  # decoded frames buffered ahead of the detector


@dataclass
class Timer:
    totals: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds


def frame_rows(idx: int, t: float, img: np.ndarray, d, cams: CameraTrack, venue: Venue, pose: PoseModel, timer: Timer) -> list[dict]:
    """Court filter, pose and torso colour for one frame's detections, as table rows."""
    t0 = time.perf_counter()
    on, size_ok, xy = court_filter(cams(idx), d.xyxy, venue.spec)
    cand = np.flatnonzero(on & size_ok)
    cand = cand[np.argsort(-d.conf[cand])][:MAX_POSE]
    timer.add("filter", time.perf_counter() - t0)
    t0 = time.perf_counter()
    kp, ks = pose(img, d.xyxy[cand])
    timer.add("pose", time.perf_counter() - t0)
    t0 = time.perf_counter()
    posed = {int(c): (kp[i], ks[i]) for i, c in enumerate(cand)}
    rows = []
    for j in range(len(d)):
        row = dict(frame=idx, t=t, det=j, x1=d.xyxy[j, 0], y1=d.xyxy[j, 1], x2=d.xyxy[j, 2], y2=d.xyxy[j, 3], conf=d.conf[j], on_court=bool(on[j]), size_ok=bool(size_ok[j]), bx=xy[j, 0], by=xy[j, 1])
        if j in posed:
            k, s = posed[j]
            feat, from_k = torso_features(img, d.xyxy[j], k, s)
            row.update(kpts=k.reshape(-1).round(1).tolist(), kscore=s.round(3).tolist(), feat=feat.tolist(), torso_kpts=from_k)
        rows.append(row)
    timer.add("features", time.perf_counter() - t0)
    return rows


def prefetched(video: Video, start: float, end: float | None, timer: Timer) -> Iterator[tuple[int, float, np.ndarray]]:
    """Video.frames on a background thread, so decoding overlaps detection and pose.

    The 1080p HEVC Main10 originals take ~70 ms a frame to decode and convert,
    as long as detection itself; PyAV drops the GIL while it decodes. The
    thread's busy time is reported as "decode" and the main thread's wait as
    "decode_wait", which is the decode cost the pass actually pays.
    """
    q: queue.Queue = queue.Queue(maxsize=PREFETCH)
    done = object()
    stop = threading.Event()

    def work() -> None:
        try:
            t0 = time.perf_counter()
            for item in video.frames(start, end):
                timer.add("decode", time.perf_counter() - t0)
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
                t0 = time.perf_counter()
        except BaseException as exc:  # surfaced in the main thread
            q.put(exc)
        q.put(done)

    th = threading.Thread(target=work, name="decode", daemon=True)
    th.start()
    try:
        while True:
            t0 = time.perf_counter()
            item = q.get()
            timer.add("decode_wait", time.perf_counter() - t0)
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        th.join(timeout=5)


def detect_with_retry(model, images: list[np.ndarray], timer: Timer, imgsz: int = 1024) -> list:
    """Batched detection, re-running any frame that came back empty on its own.

    Under heavy GPU contention the batched MPS pass twice returned no boxes for
    runs of 5 and 7 frames in clip 1 that detect both players when rerun. The
    other 16 empty frames there are real misses (both players overlapping behind
    the door strip), so the retry costs a few single-frame passes per clip.
    """
    t0 = time.perf_counter()
    dets = detect_batch(model, images, imgsz=imgsz)
    for i, d in enumerate(dets):
        if not len(d):
            dets[i] = detect_batch(model, [images[i]], imgsz=imgsz)[0]
    timer.add("detect", time.perf_counter() - t0)
    return dets


def extract(
    video: Video, venue: Venue, cams: CameraTrack, out_dir: Path, start: float = 0.0, end: float | None = None, progress=None, imgsz: int = 1024
) -> pd.DataFrame:
    """Run the pass and write detections.parquet, occluders.npz and extract_timing.json into out_dir.

    imgsz is the detector's input width; 1024 is native for the WhatsApp copies.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    t_load = time.perf_counter()
    model = load_detector()
    pose = PoseModel()
    timer.add("load", time.perf_counter() - t_load)
    sampler = BackgroundSampler()
    rows: list[dict] = []
    n_frames = 0
    wall = time.perf_counter()

    def flush(batch: list[tuple[int, float, np.ndarray]]) -> None:
        dets = detect_with_retry(model, [b[2] for b in batch], timer, imgsz)
        for (idx, t, img), d in zip(batch, dets):
            rows.extend(frame_rows(idx, t, img, d, cams, venue, pose, timer))
            sampler.offer(t, img)

    batch: list[tuple[int, float, np.ndarray]] = []
    for item in prefetched(video, start, end, timer):
        batch.append(item)
        n_frames += 1
        if len(batch) == BATCH:
            flush(batch)
            batch = []
            if progress:
                progress(n_frames, item[1])
    if batch:
        flush(batch)
    wall = time.perf_counter() - wall

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "detections.parquet", index=False)
    OccluderTrack.build(sampler).save(out_dir / "occluders.npz")
    timing = {k: round(v, 2) for k, v in timer.totals.items()}
    timing.update(n_frames=n_frames, wall_s=round(wall, 1), imgsz=imgsz, frame_size=[video.info.width, video.info.height])
    timing["ms_per_frame"] = {k: round(1000 * v / max(n_frames, 1), 2) for k, v in timer.totals.items() if k != "load"}
    timing["ms_per_frame"]["wall"] = round(1000 * wall / max(n_frames, 1), 2)
    (out_dir / "extract_timing.json").write_text(json.dumps(timing, indent=2))
    return df
