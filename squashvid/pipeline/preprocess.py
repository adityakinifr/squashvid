from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from squashvid.pipeline.models import Segment

SEGMENTER_VERSION = "adaptive-v7-scene-guard"


@dataclass(slots=True)
class SegmentationResult:
    segments: list[Segment]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class SceneChange:
    timestamp_sec: float
    kind: str
    diff_score: float
    motion_ratio: float
    brightness: float
    bright_ratio: float


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


def _frame_scene_diff(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    diff = cv2.absdiff(previous_gray, current_gray)
    return float(np.mean(diff) / 255.0)


def _classify_scene_change(
    *,
    has_previous: bool,
    scene_diff: float,
    motion_ratio: float,
    brightness: float,
    brightness_std: float,
    bright_ratio: float,
) -> str | None:
    is_white_flash = (
        (brightness >= 220.0 and bright_ratio >= 0.62)
        or (brightness >= 245.0 and brightness_std <= 12.0)
    )
    if is_white_flash:
        return "flash"

    if not has_previous:
        return None

    is_hard_cut = scene_diff >= 0.28 and motion_ratio >= 0.82
    if is_hard_cut:
        return "cut"

    return None


def _scene_exclusion_intervals(
    scene_changes: list[SceneChange],
    start_sec: float,
    end_sec: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for change in scene_changes:
        if change.kind == "flash":
            pre_pad_sec = 0.45
            post_pad_sec = 0.90
        else:
            pre_pad_sec = 0.25
            post_pad_sec = 0.65

        start = max(start_sec, change.timestamp_sec - pre_pad_sec)
        end = min(end_sec, change.timestamp_sec + post_pad_sec)
        if end > start:
            intervals.append((start, end))

    if not intervals:
        return []

    intervals.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _suppress_motion_samples(
    motion_samples: list[tuple[float, float]],
    intervals: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], int]:
    if not intervals:
        return motion_samples, 0

    suppressed: list[tuple[float, float]] = []
    interval_idx = 0
    suppressed_count = 0

    for timestamp, ratio in motion_samples:
        while interval_idx < len(intervals) and timestamp > intervals[interval_idx][1]:
            interval_idx += 1

        in_transition = (
            interval_idx < len(intervals)
            and intervals[interval_idx][0] <= timestamp <= intervals[interval_idx][1]
        )
        if in_transition:
            suppressed.append((timestamp, 0.0))
            if ratio > 0.0:
                suppressed_count += 1
        else:
            suppressed.append((timestamp, ratio))

    return suppressed, suppressed_count


def _split_segments_on_scene_intervals(
    segments: list[Segment],
    intervals: list[tuple[float, float]],
    min_piece_sec: float,
) -> list[Segment]:
    if not segments or not intervals:
        return segments

    split_segments: list[Segment] = []
    for segment in segments:
        pieces = [(segment.start_sec, segment.end_sec)]
        for cut_start, cut_end in intervals:
            if cut_end <= segment.start_sec or cut_start >= segment.end_sec:
                continue

            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < cut_start:
                    next_pieces.append((piece_start, min(cut_start, piece_end)))
                if cut_end < piece_end:
                    next_pieces.append((max(cut_end, piece_start), piece_end))
            pieces = next_pieces

        for piece_start, piece_end in pieces:
            if piece_end - piece_start >= min_piece_sec:
                split_segments.append(Segment(start_sec=piece_start, end_sec=piece_end))

    return split_segments


def _scene_changes_payload(
    scene_changes: list[SceneChange],
    limit: int = 80,
) -> list[dict[str, float | str]]:
    return [
        {
            "timestamp_sec": round(float(change.timestamp_sec), 3),
            "kind": change.kind,
            "diff_score": round(float(change.diff_score), 4),
            "motion_ratio": round(float(change.motion_ratio), 4),
            "brightness": round(float(change.brightness), 2),
            "bright_ratio": round(float(change.bright_ratio), 4),
        }
        for change in scene_changes[:limit]
    ]


def _intervals_payload(
    intervals: list[tuple[float, float]],
    limit: int = 80,
) -> list[dict[str, float]]:
    return [
        {
            "start_sec": round(float(start), 3),
            "end_sec": round(float(end), 3),
            "duration_sec": round(float(end - start), 3),
        }
        for start, end in intervals[:limit]
    ]


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


def _bridge_fragmented_segments(
    segments: list[Segment],
    bridge_gap_sec: float,
    fragment_sec: float = 14.0,
    max_merged_sec: float = 75.0,
) -> list[Segment]:
    if not segments:
        return []
    if bridge_gap_sec <= 0:
        return segments

    bridged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = bridged[-1]
        gap = seg.start_sec - prev.end_sec
        merged_duration = seg.end_sec - prev.start_sec
        is_likely_fragment = prev.duration_sec <= fragment_sec or seg.duration_sec <= fragment_sec
        if gap <= bridge_gap_sec and is_likely_fragment and merged_duration <= max_merged_sec:
            bridged[-1] = Segment(start_sec=prev.start_sec, end_sec=max(prev.end_sec, seg.end_sec))
        else:
            bridged.append(seg)
    return bridged


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


def _extend_first_rally_start(
    segments: list[Segment],
    start_offset_sec: float,
    max_leading_gap_sec: float,
) -> tuple[list[Segment], bool]:
    if not segments:
        return [], False

    first = segments[0]
    leading_gap = first.start_sec - start_offset_sec
    if leading_gap <= 0.0 or leading_gap > max_leading_gap_sec:
        return segments, False

    extended = [Segment(start_sec=start_offset_sec, end_sec=first.end_sec), *segments[1:]]
    return extended, True


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
    analysis_duration_sec: float | None = None,
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
    scoring_duration = (
        float(analysis_duration_sec)
        if analysis_duration_sec is not None and analysis_duration_sec > 0.0
        else total_duration
    )
    long_limit = max(120.0, min(scoring_duration * 0.8, 240.0))

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


def _segment_confidence(
    segment: Segment,
    motion_samples: list[tuple[float, float]],
    threshold: float,
) -> tuple[float | None, float | None]:
    samples = [
        ratio
        for ts, ratio in motion_samples
        if segment.start_sec <= ts <= segment.end_sec
    ]
    if not samples:
        return None, None

    ratios = np.array(samples, dtype=float)
    active_rate = float(np.count_nonzero(ratios >= threshold) / ratios.size)
    threshold_strength = float(min(1.0, ratios.mean() / max(threshold * 1.75, 0.0001)))
    duration_strength = float(min(1.0, segment.duration_sec / 18.0))
    confidence = 0.5 * active_rate + 0.32 * threshold_strength + 0.18 * duration_strength
    return round(float(confidence), 3), round(float(active_rate), 3)


def _segments_diagnostic_payload(
    segments: list[Segment],
    limit: int = 80,
    motion_samples: list[tuple[float, float]] | None = None,
    threshold: float | None = None,
) -> list[dict[str, float]]:
    payload: list[dict[str, float]] = []
    for seg in segments[:limit]:
        row = {
            "start_sec": round(float(seg.start_sec), 3),
            "end_sec": round(float(seg.end_sec), 3),
            "duration_sec": round(float(seg.duration_sec), 3),
        }
        if motion_samples is not None and threshold is not None:
            confidence, active_rate = _segment_confidence(seg, motion_samples, threshold)
            if confidence is not None:
                row["confidence"] = confidence
            if active_rate is not None:
                row["active_sample_rate"] = active_rate
        payload.append(row)
    return payload


def _smooth_motion_samples(
    motion_samples: list[tuple[float, float]],
    window_sec: float,
) -> list[tuple[float, float]]:
    if len(motion_samples) < 4 or window_sec <= 0:
        return motion_samples

    timestamps = np.array([sample[0] for sample in motion_samples], dtype=float)
    ratios = np.array([sample[1] for sample in motion_samples], dtype=float)
    deltas = np.diff(timestamps)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        return motion_samples

    sample_step = float(np.median(positive_deltas))
    window_size = max(3, int(round(window_sec / max(sample_step, 0.001))))
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(ratios, kernel, mode="same")
    return [(float(ts), float(ratio)) for ts, ratio in zip(timestamps, smoothed)]


def _motion_preview_payload(
    motion_samples: list[tuple[float, float]],
    threshold: float,
    limit: int = 220,
) -> list[dict[str, float]]:
    if not motion_samples:
        return []

    stride = max(1, int(np.ceil(len(motion_samples) / limit)))
    preview = []
    for timestamp, ratio in motion_samples[::stride][:limit]:
        preview.append(
            {
                "timestamp_sec": round(float(timestamp), 3),
                "motion": round(float(ratio), 6),
                "active": 1.0 if ratio >= threshold else 0.0,
            }
        )
    return preview


def _motion_diagnostic_payload(
    motion_samples: list[tuple[float, float]],
    threshold: float,
) -> dict[str, float | int | None]:
    if not motion_samples:
        return {
            "sample_count": 0,
            "first_sample_sec": None,
            "last_sample_sec": None,
            "avg_motion": 0.0,
            "median_motion": 0.0,
            "p90_motion": 0.0,
            "p95_motion": 0.0,
            "max_motion": 0.0,
            "active_sample_rate": 0.0,
        }

    ratios = np.array([sample[1] for sample in motion_samples], dtype=float)
    return {
        "sample_count": int(ratios.size),
        "first_sample_sec": round(float(motion_samples[0][0]), 3),
        "last_sample_sec": round(float(motion_samples[-1][0]), 3),
        "avg_motion": round(float(ratios.mean()), 6),
        "median_motion": round(float(np.median(ratios)), 6),
        "p90_motion": round(float(np.percentile(ratios, 90)), 6),
        "p95_motion": round(float(np.percentile(ratios, 95)), 6),
        "max_motion": round(float(ratios.max()), 6),
        "active_sample_rate": round(float(np.count_nonzero(ratios >= threshold) / ratios.size), 4),
    }


def detect_active_segments_with_diagnostics(
    video_path: str,
    motion_threshold: float = 0.018,
    min_rally_sec: float = 4.0,
    idle_gap_sec: float = 1.2,
    downscale_width: int = 480,
    frame_step: int = 2,
    max_rallies: int | None = None,
    max_duration_sec: float | None = None,
    start_offset_sec: float = 0.0,
) -> SegmentationResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = 0
    start_offset_sec = max(0.0, float(start_offset_sec))
    if start_offset_sec > 0.0:
        start_frame = int(start_offset_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    previous_gray: np.ndarray | None = None
    motion_samples: list[tuple[float, float]] = []
    scene_changes: list[SceneChange] = []

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

        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray_raw, (5, 5), 0)

        motion_ratio = 0.0
        scene_diff = 0.0
        has_previous = previous_gray is not None
        if previous_gray is not None:
            motion_ratio = _frame_motion_ratio(previous_gray, gray)
            scene_diff = _frame_scene_diff(previous_gray, gray)
        brightness = float(np.mean(gray_raw))
        brightness_std = float(np.std(gray_raw))
        bright_ratio = float(np.count_nonzero(gray_raw >= 235) / gray_raw.size)
        previous_gray = gray

        # Timestamp is absolute (includes start offset)
        timestamp = start_offset_sec + (index / fps)
        scene_kind = _classify_scene_change(
            has_previous=has_previous,
            scene_diff=scene_diff,
            motion_ratio=motion_ratio,
            brightness=brightness,
            brightness_std=brightness_std,
            bright_ratio=bright_ratio,
        )
        if scene_kind is not None:
            scene_changes.append(
                SceneChange(
                    timestamp_sec=timestamp,
                    kind=scene_kind,
                    diff_score=scene_diff,
                    motion_ratio=motion_ratio,
                    brightness=brightness,
                    bright_ratio=bright_ratio,
                )
            )
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
    source_duration = (frame_count / fps) if fps and frame_count > 0 else 0.0
    observed_end_sec = start_offset_sec + observed_duration
    total_duration = source_duration if source_duration > 0.0 else observed_end_sec
    total_duration = max(total_duration, observed_end_sec)
    if max_duration is not None:
        total_duration = min(total_duration, start_offset_sec + max_duration)
    total_duration = max(start_offset_sec, total_duration)
    analysis_duration = max(0.0, total_duration - start_offset_sec)

    raw_motion_samples = motion_samples
    scene_intervals = _scene_exclusion_intervals(
        scene_changes=scene_changes,
        start_sec=start_offset_sec,
        end_sec=total_duration,
    )
    motion_samples, suppressed_sample_count = _suppress_motion_samples(
        motion_samples=raw_motion_samples,
        intervals=scene_intervals,
    )

    smoothing_window_sec = max(0.6, min(1.2, idle_gap_sec * 0.7))
    smoothed_motion_samples = _smooth_motion_samples(
        motion_samples=motion_samples,
        window_sec=smoothing_window_sec,
    )
    smoothed_threshold = max(0.0015, motion_threshold * 0.82)

    base_segments = _segments_from_motion_samples(
        motion_samples=motion_samples,
        threshold=motion_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=idle_gap_sec,
        total_duration=total_duration,
        max_rallies=None,
    )
    smoothed_segments = _segments_from_motion_samples(
        motion_samples=smoothed_motion_samples,
        threshold=smoothed_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=max(idle_gap_sec, smoothing_window_sec),
        total_duration=total_duration,
        max_rallies=None,
    )
    segments = base_segments
    selected_threshold = motion_threshold
    segment_signal = "raw"
    adaptive_threshold: float | None = None
    adaptive_used = False

    if smoothed_segments:
        base_fragments = sum(1 for seg in base_segments if seg.duration_sec <= min_rally_sec * 2.8)
        smoothed_fragments = sum(
            1 for seg in smoothed_segments if seg.duration_sec <= min_rally_sec * 2.8
        )
        if (
            not base_segments
            or (len(base_segments) > 3 and len(smoothed_segments) < len(base_segments))
            or (smoothed_fragments < base_fragments and len(smoothed_segments) <= len(base_segments))
        ):
            segments = smoothed_segments
            selected_threshold = smoothed_threshold
            segment_signal = "smoothed"

    needs_adaptive = False
    if not segments:
        needs_adaptive = True
    elif len(segments) == 1 and analysis_duration > 120.0:
        needs_adaptive = segments[0].duration_sec > max(100.0, analysis_duration * 0.72)

    if needs_adaptive:
        adaptive_calibration_sec = min(analysis_duration, 8.0 * 60.0)
        calibration_end_sec = start_offset_sec + adaptive_calibration_sec
        calibration_samples = [
            (ts, ratio)
            for ts, ratio in motion_samples
            if start_offset_sec <= ts <= calibration_end_sec
        ]
        if not calibration_samples:
            calibration_samples = motion_samples

        adaptive_threshold = _select_adaptive_threshold(
            motion_samples=calibration_samples,
            base_threshold=motion_threshold,
            min_rally_sec=min_rally_sec,
            idle_gap_sec=idle_gap_sec,
            total_duration=calibration_end_sec,
            max_rallies=None,
            analysis_duration_sec=adaptive_calibration_sec,
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
                selected_threshold = adaptive_threshold
                segment_signal = "adaptive"
                adaptive_used = True

    merge_gap_sec = min(3.2, max(idle_gap_sec * 1.8, idle_gap_sec + 0.8))
    pre_merge_segments = segments
    segments = _merge_close_segments(
        segments=segments,
        merge_gap_sec=merge_gap_sec,
        max_rallies=None,
    )
    merged_segments = segments
    bridge_gap_sec = min(6.5, max(idle_gap_sec * 4.2, idle_gap_sec + 4.0))
    pre_bridge_segments = segments
    segments = _bridge_fragmented_segments(
        segments=segments,
        bridge_gap_sec=bridge_gap_sec,
        fragment_sec=max(12.0, min_rally_sec * 3.0),
        max_merged_sec=max(45.0, min(90.0, analysis_duration * 0.55 if analysis_duration else 75.0)),
    )
    bridged_segments = segments
    tail_pad_sec = min(3.2, max(idle_gap_sec + 1.8, idle_gap_sec * 2.5))
    segments = _extend_segment_tails(
        segments=segments,
        tail_pad_sec=tail_pad_sec,
        total_duration=total_duration,
        max_rallies=None,
    )
    pre_scene_split_segments = segments
    scene_min_piece_sec = max(1.0, min_rally_sec * 0.5)
    segments = _split_segments_on_scene_intervals(
        segments=segments,
        intervals=scene_intervals,
        min_piece_sec=scene_min_piece_sec,
    )
    scene_split_segments = segments

    leading_context_sec = min(40.0, max(24.0, min_rally_sec * 8.5))
    leading_extension_blocked_by_scene = False
    if segments:
        first_segment = segments[0]
        leading_extension_blocked_by_scene = any(
            interval_end > start_offset_sec and interval_start < first_segment.start_sec
            for interval_start, interval_end in scene_intervals
        )
    if leading_extension_blocked_by_scene:
        leading_extension_used = False
    else:
        segments, leading_extension_used = _extend_first_rally_start(
            segments=segments,
            start_offset_sec=start_offset_sec,
            max_leading_gap_sec=leading_context_sec,
        )

    min_full_segment_sec = min(min_rally_sec, 1.0)
    fallback_used = False
    if not segments and analysis_duration >= min_full_segment_sec:
        segments.append(Segment(start_sec=start_offset_sec, end_sec=total_duration))
        fallback_used = True

    final_segments = segments[:max_rallies] if max_rallies is not None else segments
    diagnostics = {
        "version": SEGMENTER_VERSION,
        "fps": round(float(fps), 3),
        "frame_count": int(frame_count),
        "frame_step": int(frame_step),
        "source_duration_sec": round(float(source_duration), 3),
        "window_start_sec": round(float(start_offset_sec), 3),
        "window_end_sec": round(float(total_duration), 3),
        "window_duration_sec": round(float(analysis_duration), 3),
        "max_duration_sec": round(float(max_duration), 3) if max_duration is not None else None,
        "requested_max_rallies": int(max_rallies) if max_rallies is not None else None,
        "base_threshold": round(float(motion_threshold), 6),
        "selected_threshold": round(float(selected_threshold), 6),
        "smoothed_threshold": round(float(smoothed_threshold), 6),
        "segment_signal": segment_signal,
        "smoothing_window_sec": round(float(smoothing_window_sec), 3),
        "adaptive_threshold": round(float(adaptive_threshold), 6)
        if adaptive_threshold is not None
        else None,
        "adaptive_used": adaptive_used,
        "fallback_full_window_used": fallback_used,
        "merge_gap_sec": round(float(merge_gap_sec), 3),
        "bridge_gap_sec": round(float(bridge_gap_sec), 3),
        "tail_pad_sec": round(float(tail_pad_sec), 3),
        "scene_guard_enabled": True,
        "scene_change_count": len(scene_changes),
        "scene_exclusion_interval_count": len(scene_intervals),
        "scene_suppressed_sample_count": suppressed_sample_count,
        "scene_min_piece_sec": round(float(scene_min_piece_sec), 3),
        "leading_context_sec": round(float(leading_context_sec), 3),
        "leading_extension_used": leading_extension_used,
        "leading_extension_blocked_by_scene": leading_extension_blocked_by_scene,
        "base_segment_count": len(base_segments),
        "smoothed_segment_count": len(smoothed_segments),
        "pre_merge_segment_count": len(pre_merge_segments),
        "merged_segment_count": len(merged_segments),
        "pre_bridge_segment_count": len(pre_bridge_segments),
        "bridged_segment_count": len(bridged_segments),
        "pre_scene_split_segment_count": len(pre_scene_split_segments),
        "scene_split_segment_count": len(scene_split_segments),
        "final_segment_count": len(final_segments),
        "motion": _motion_diagnostic_payload(motion_samples, selected_threshold),
        "raw_motion": _motion_diagnostic_payload(raw_motion_samples, selected_threshold),
        "motion_preview": _motion_preview_payload(motion_samples, selected_threshold),
        "scene_changes": _scene_changes_payload(scene_changes),
        "scene_exclusion_intervals": _intervals_payload(scene_intervals),
        "base_segments": _segments_diagnostic_payload(
            base_segments,
            motion_samples=motion_samples,
            threshold=motion_threshold,
        ),
        "smoothed_segments": _segments_diagnostic_payload(
            smoothed_segments,
            motion_samples=smoothed_motion_samples,
            threshold=smoothed_threshold,
        ),
        "pre_merge_segments": _segments_diagnostic_payload(
            pre_merge_segments,
            motion_samples=motion_samples,
            threshold=selected_threshold,
        ),
        "merged_segments": _segments_diagnostic_payload(
            merged_segments,
            motion_samples=motion_samples,
            threshold=selected_threshold,
        ),
        "bridged_segments": _segments_diagnostic_payload(
            bridged_segments,
            motion_samples=motion_samples,
            threshold=selected_threshold,
        ),
        "pre_scene_split_segments": _segments_diagnostic_payload(
            pre_scene_split_segments,
            motion_samples=motion_samples,
            threshold=selected_threshold,
        ),
        "scene_split_segments": _segments_diagnostic_payload(
            scene_split_segments,
            motion_samples=motion_samples,
            threshold=selected_threshold,
        ),
        "final_segments": _segments_diagnostic_payload(
            final_segments,
            motion_samples=motion_samples,
            threshold=selected_threshold,
        ),
    }
    return SegmentationResult(segments=final_segments, diagnostics=diagnostics)


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
    return detect_active_segments_with_diagnostics(
        video_path=video_path,
        motion_threshold=motion_threshold,
        min_rally_sec=min_rally_sec,
        idle_gap_sec=idle_gap_sec,
        downscale_width=downscale_width,
        frame_step=frame_step,
        max_rallies=max_rallies,
        max_duration_sec=max_duration_sec,
        start_offset_sec=start_offset_sec,
    ).segments


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
