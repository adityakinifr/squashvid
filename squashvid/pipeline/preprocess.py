from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from squashvid.pipeline.models import Segment


def read_video_metadata(video_path: str) -> dict[str, float | int]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_sec = (frame_count / fps) if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def _frame_motion_ratio(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    diff = cv2.absdiff(previous_gray, current_gray)
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(mask) / mask.size)


def detect_active_segments(
    video_path: str,
    motion_threshold: float = 0.018,
    min_rally_sec: float = 4.0,
    idle_gap_sec: float = 1.2,
    downscale_width: int = 480,
    frame_step: int = 2,
    max_rallies: int | None = None,
) -> list[Segment]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    previous_gray: np.ndarray | None = None
    current_start: float | None = None
    last_active: float | None = None
    segments: list[Segment] = []

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        if width > downscale_width:
            scale = downscale_width / width
            frame = cv2.resize(frame, (downscale_width, int(height * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_ratio = 0.0
        if previous_gray is not None:
            motion_ratio = _frame_motion_ratio(previous_gray, gray)
        previous_gray = gray

        timestamp = index / fps
        is_active = motion_ratio >= motion_threshold
        if is_active:
            if current_start is None:
                current_start = timestamp
            last_active = timestamp
        elif current_start is not None and last_active is not None:
            if (timestamp - last_active) >= idle_gap_sec:
                if (last_active - current_start) >= min_rally_sec:
                    segments.append(Segment(start_sec=current_start, end_sec=last_active))
                    if max_rallies is not None and len(segments) >= max_rallies:
                        break
                current_start = None
                last_active = None

        index += 1
        if frame_step > 1:
            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
                index += 1

    cap.release()

    total_duration = (frame_count / fps) if fps else 0.0
    if current_start is not None and last_active is not None:
        end_sec = max(last_active, min(total_duration, last_active + 0.1))
        if (end_sec - current_start) >= min_rally_sec:
            segments.append(Segment(start_sec=current_start, end_sec=end_sec))

    if not segments and total_duration >= min_rally_sec:
        segments.append(Segment(start_sec=0.0, end_sec=total_duration))

    return segments[:max_rallies] if max_rallies is not None else segments


def clip_windows_from_segments(
    segments: list[Segment],
    pad_sec: float,
    duration_sec: float,
) -> list[Segment]:
    windows: list[Segment] = []
    for seg in segments:
        start = max(0.0, seg.start_sec - pad_sec)
        end = min(duration_sec, seg.end_sec + pad_sec)
        windows.append(Segment(start_sec=start, end_sec=end))
    return windows
