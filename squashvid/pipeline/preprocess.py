from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from squashvid.pipeline.models import Segment

SEGMENTER_VERSION = "adaptive-v3-window-stable"


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


def _segments_from_motion_samples(
    motion_samples: list[tuple[float, float]],
    threshold: float,
    min_rally_sec: float,
    idle_gap_sec: float,
    total_duration: float,
    max_rallies: int | None,
) -> list[Segment]:
    current_start: float | None = None
    last_active: float | None = None
    segments: list[Segment] = []

    for timestamp, motion_ratio in motion_samples:
        is_active = motion_ratio >= threshold
        if is_active:
            if current_start is None:
                current_start = timestamp
            last_active = timestamp
            continue

        if current_start is None or last_active is None:
            continue

        if (timestamp - last_active) >= idle_gap_sec:
            if (last_active - current_start) >= min_rally_sec:
                segments.append(Segment(start_sec=current_start, end_sec=last_active))
                if max_rallies is not None and len(segments) >= max_rallies:
                    return segments
            current_start = None
            last_active = None

    if current_start is not None and last_active is not None:
        end_sec = max(last_active, min(total_duration, last_active + 0.1))
        if (end_sec - current_start) >= min_rally_sec:
            segments.append(Segment(start_sec=current_start, end_sec=end_sec))

    return segments[:max_rallies] if max_rallies is not None else segments


def _merge_close_segments(
    segments: list[Segment],
    merge_gap_sec: float,
    max_rallies: int | None = None,
) -> list[Segment]:
    if not segments:
        return []
    if merge_gap_sec <= 0:
        return segments[:max_rallies] if max_rallies is not None else segments

    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_sec - prev.end_sec
        if gap <= merge_gap_sec:
            merged[-1] = Segment(start_sec=prev.start_sec, end_sec=max(prev.end_sec, seg.end_sec))
        else:
            merged.append(seg)

    return merged[:max_rallies] if max_rallies is not None else merged


def _extend_segment_tails(
    segments: list[Segment],
    tail_pad_sec: float,
    total_duration: float,
    max_rallies: int | None = None,
) -> list[Segment]:
    if not segments:
        return []
    if tail_pad_sec <= 0:
        return segments[:max_rallies] if max_rallies is not None else segments

    padded: list[Segment] = []
    for idx, seg in enumerate(segments):
        next_start = segments[idx + 1].start_sec if idx + 1 < len(segments) else total_duration
        max_end = min(total_duration, next_start)
        end = min(max_end, seg.end_sec + tail_pad_sec)
        if end < seg.start_sec:
            end = seg.end_sec
        padded.append(Segment(start_sec=seg.start_sec, end_sec=end))

    return padded[:max_rallies] if max_rallies is not None else padded


def _adaptive_segments_from_samples(
    motion_samples: list[tuple[float, float]],
    base_threshold: float,
    min_rally_sec: float,
    idle_gap_sec: float,
    total_duration: float,
    max_rallies: int | None,
) -> list[Segment]:
    adaptive_threshold = _select_adaptive_threshold(
        motion_samples=motion_samples,
        base_threshold=base_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=idle_gap_sec,
        total_duration=total_duration,
        max_rallies=max_rallies,
    )
    if adaptive_threshold is None:
        return []
    return _segments_from_motion_samples(
        motion_samples=motion_samples,
        threshold=adaptive_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=idle_gap_sec,
        total_duration=total_duration,
        max_rallies=max_rallies,
    )


def _select_adaptive_threshold(
    motion_samples: list[tuple[float, float]],
    base_threshold: float,
    min_rally_sec: float,
    idle_gap_sec: float,
    total_duration: float,
    max_rallies: int | None,
) -> float | None:
    if len(motion_samples) < 20:
        return None

    ratios = np.array([sample[1] for sample in motion_samples], dtype=float)
    if ratios.size == 0:
        return None

    quantile_thresholds = [
        float(np.percentile(ratios, p))
        for p in (95, 92, 90, 88, 85, 82, 80, 78, 75, 72, 70, 68, 65)
    ]
    scaled_thresholds = [base_threshold * f for f in (0.85, 0.75, 0.65, 0.55, 0.5)]
    raw_candidates = [*quantile_thresholds, *scaled_thresholds]

    threshold_candidates: list[float] = []
    seen: set[float] = set()
    for candidate in raw_candidates:
        thr = max(0.0015, min(0.3, float(candidate)))
        key = round(thr, 6)
        if key in seen:
            continue
        seen.add(key)
        threshold_candidates.append(thr)

    threshold_candidates.sort(reverse=True)

    best_threshold: float | None = None
    best_score = float("-inf")
    long_limit = max(120.0, min(total_duration * 0.8, 240.0))

    for threshold in threshold_candidates:
        segments = _segments_from_motion_samples(
            motion_samples=motion_samples,
            threshold=threshold,
            min_rally_sec=min_rally_sec,
            idle_gap_sec=idle_gap_sec,
            total_duration=total_duration,
            max_rallies=max_rallies,
        )
        if not segments:
            continue

        durations = np.array([seg.duration_sec for seg in segments], dtype=float)
        count = int(durations.size)
        max_duration = float(durations.max()) if count else 0.0
        median_duration = float(np.median(durations)) if count else 0.0
        avg_duration = float(durations.mean()) if count else 0.0

        score = float(count)
        if count < 2:
            score -= 4.0
        if median_duration < (min_rally_sec * 1.05):
            score -= 3.0
        if avg_duration > 55.0:
            score -= 2.0
        if max_duration > long_limit:
            score -= 8.0
        elif max_duration > 90.0:
            score -= 2.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    if best_threshold is None:
        return None

    chosen_segments = _segments_from_motion_samples(
        motion_samples=motion_samples,
        threshold=best_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=idle_gap_sec,
        total_duration=total_duration,
        max_rallies=max_rallies,
    )
    if len(chosen_segments) >= 2:
        return best_threshold
    return None


def detect_active_segments(
    video_path: str,
    motion_threshold: float = 0.018,
    min_rally_sec: float = 4.0,
    idle_gap_sec: float = 1.2,
    downscale_width: int = 480,
    frame_step: int = 2,
    max_rallies: int | None = None,
    max_duration_sec: float | None = None,
    start_offset_sec: float = 0.0,
) -> list[Segment]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Seek to start offset if specified
    start_frame = 0
    if start_offset_sec > 0.0:
        start_frame = int(start_offset_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    previous_gray: np.ndarray | None = None
    motion_samples: list[tuple[float, float]] = []

    max_duration = float(max_duration_sec) if max_duration_sec is not None else None
    if max_duration is not None and max_duration <= 0.0:
        max_duration = None

    index = 0
    while True:
        if max_duration is not None and (index / fps) >= max_duration:
            break

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

        # Timestamp is absolute (includes start offset)
        timestamp = start_offset_sec + (index / fps)
        motion_samples.append((timestamp, motion_ratio))

        index += 1
        if frame_step > 1:
            for _ in range(frame_step - 1):
                if max_duration is not None and ((index + 1) / fps) >= max_duration:
                    break
                if not cap.grab():
                    break
                index += 1

    cap.release()

    observed_duration = (index / fps) if fps else 0.0
    total_duration = (frame_count / fps) if fps and frame_count > 0 else 0.0
    total_duration = max(total_duration, observed_duration)
    if max_duration is not None:
        total_duration = min(total_duration, max_duration)

    segments = _segments_from_motion_samples(
        motion_samples=motion_samples,
        threshold=motion_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=idle_gap_sec,
        total_duration=total_duration,
        max_rallies=None,
    )

    needs_adaptive = False
    if not segments:
        needs_adaptive = True
    elif len(segments) == 1 and total_duration > 120.0:
        needs_adaptive = segments[0].duration_sec > max(100.0, total_duration * 0.72)

    if needs_adaptive:
        adaptive_calibration_sec = min(total_duration, 8.0 * 60.0)
        calibration_samples = [
            (ts, ratio) for ts, ratio in motion_samples if ts <= adaptive_calibration_sec
        ]
        if not calibration_samples:
            calibration_samples = motion_samples

        adaptive_threshold = _select_adaptive_threshold(
            motion_samples=calibration_samples,
            base_threshold=motion_threshold,
            min_rally_sec=min_rally_sec,
            idle_gap_sec=idle_gap_sec,
            total_duration=adaptive_calibration_sec,
            max_rallies=None,
        )
        if adaptive_threshold is not None:
            adaptive_segments = _segments_from_motion_samples(
                motion_samples=motion_samples,
                threshold=adaptive_threshold,
                min_rally_sec=min_rally_sec,
                idle_gap_sec=idle_gap_sec,
                total_duration=total_duration,
                max_rallies=None,
            )
            if len(adaptive_segments) >= 2:
                segments = adaptive_segments

    merge_gap_sec = min(2.4, max(idle_gap_sec * 1.5, idle_gap_sec + 0.5))
    segments = _merge_close_segments(
        segments=segments,
        merge_gap_sec=merge_gap_sec,
        max_rallies=None,
    )
    segments = _extend_segment_tails(
        segments=segments,
        tail_pad_sec=min(1.4, idle_gap_sec + 0.2),
        total_duration=total_duration,
        max_rallies=None,
    )

    min_full_segment_sec = min(min_rally_sec, 1.0)
    if not segments and total_duration >= min_full_segment_sec:
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
