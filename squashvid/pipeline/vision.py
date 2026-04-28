from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import cv2
import numpy as np

from squashvid.pipeline.models import FrameObservation, Segment, SegmentTrack


@dataclass(slots=True)
class _Candidate:
    x: float
    y: float
    area: float


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _extract_candidates(mask: np.ndarray, min_area: float, max_area: float) -> list[_Candidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        candidates.append(_Candidate(x=x, y=y, area=area))

    candidates.sort(key=lambda c: c.area, reverse=True)
    return candidates


def _detect_t_position(frame: np.ndarray) -> tuple[float, float]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(40, int(0.1 * w)),
        maxLineGap=20,
    )

    if lines is None:
        return (w / 2.0, h * 0.62)

    xs: list[float] = []
    ys: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line.tolist()
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 8 and min(y1, y2) > h * 0.2:
            xs.append((x1 + x2) / 2.0)
        if abs(dy) < 8 and h * 0.3 < y1 < h * 0.8:
            ys.append((y1 + y2) / 2.0)

    x_val = float(np.median(xs)) if xs else (w / 2.0)
    y_val = float(np.median(ys)) if ys else (h * 0.62)
    return (x_val, y_val)


def _assign_players(
    previous: dict[str, tuple[float, float] | None],
    candidates: list[_Candidate],
    max_jump: float = 160.0,
) -> dict[str, tuple[float, float] | None]:
    points = [(c.x, c.y) for c in candidates[:4]]

    if not points:
        return {"A": None, "B": None}

    if previous.get("A") is None and previous.get("B") is None:
        points.sort(key=lambda p: p[0])
        if len(points) == 1:
            return {"A": points[0], "B": None}
        return {"A": points[0], "B": points[-1]}

    best: tuple[float, tuple[float, float] | None, tuple[float, float] | None] | None = None
    candidates_for_pair: list[tuple[float, float] | None] = [*points, None]
    for point_a, point_b in permutations(candidates_for_pair, 2):
        if point_a is None and point_b is None:
            continue
        if point_a is not None and point_b is not None and point_a == point_b:
            continue

        cost = 0.0
        valid = True
        for label, point in (("A", point_a), ("B", point_b)):
            prev_point = previous.get(label)
            if point is None:
                cost += max_jump * 0.9
                continue
            if prev_point is None:
                cost += 20.0
                continue
            dist = _distance(prev_point, point)
            if dist > max_jump:
                valid = False
                break
            cost += dist

        if not valid:
            continue
        if best is None or cost < best[0]:
            best = (cost, point_a, point_b)

    if best is not None:
        return {"A": best[1], "B": best[2]}

    points.sort(key=lambda p: p[0])
    if len(points) == 1:
        prev_a = previous.get("A")
        prev_b = previous.get("B")
        if prev_b is not None and (prev_a is None or _distance(prev_b, points[0]) < _distance(prev_a, points[0])):
            return {"A": None, "B": points[0]}
        return {"A": points[0], "B": None}
    return {"A": points[0], "B": points[-1]}


def _assign_ball(
    previous_ball: tuple[float, float] | None,
    candidates: list[_Candidate],
    max_jump: float = 120.0,
) -> tuple[float, float] | None:
    if not candidates:
        return None

    points = [(c.x, c.y) for c in candidates[:12]]
    if previous_ball is None:
        return points[0]

    nearest = min(points, key=lambda p: _distance(previous_ball, p))
    return nearest if _distance(previous_ball, nearest) <= max_jump else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_court_metadata(
    frame_size: tuple[int, int],
    detected_t: tuple[float, float] | None,
    court_calibration: dict[str, Any] | None,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    frame_w, frame_h = frame_size
    if frame_w <= 0 or frame_h <= 0:
        return detected_t, {"mode": "unavailable"}

    if court_calibration:
        x = _clamp01(float(court_calibration.get("x", 0.0)))
        y = _clamp01(float(court_calibration.get("y", 0.0)))
        w = _clamp01(float(court_calibration.get("w", 1.0)))
        h = _clamp01(float(court_calibration.get("h", 1.0)))
        if x + w > 1.0:
            w = max(0.01, 1.0 - x)
        if y + h > 1.0:
            h = max(0.01, 1.0 - y)
        t_x = _clamp01(float(court_calibration.get("t_x", 0.5)))
        t_y = _clamp01(float(court_calibration.get("t_y", 0.62)))
        t_position = (t_x * frame_w, t_y * frame_h)
        return t_position, {
            "mode": "manual",
            "rect_norm": {"x": x, "y": y, "w": w, "h": h},
            "t_norm": {"x": t_x, "y": t_y},
        }

    t_position = detected_t
    if t_position is None:
        t_position = (frame_w / 2.0, frame_h * 0.62)
    return t_position, {
        "mode": "auto",
        "rect_norm": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "t_norm": {
            "x": round(float(t_position[0] / frame_w), 5),
            "y": round(float(t_position[1] / frame_h), 5),
        },
    }


def _interpolate_short_gaps(
    values: list[tuple[float, float] | None],
    max_gap: int,
) -> tuple[list[tuple[float, float] | None], int]:
    filled = values[:]
    fill_count = 0
    idx = 0
    while idx < len(filled):
        if filled[idx] is not None:
            idx += 1
            continue
        start = idx
        while idx < len(filled) and filled[idx] is None:
            idx += 1
        end = idx
        gap_len = end - start
        left = filled[start - 1] if start > 0 else None
        right = filled[end] if end < len(filled) else None
        if left is None or right is None or gap_len > max_gap:
            continue
        for offset in range(gap_len):
            alpha = float(offset + 1) / float(gap_len + 1)
            filled[start + offset] = (
                left[0] + (right[0] - left[0]) * alpha,
                left[1] + (right[1] - left[1]) * alpha,
            )
            fill_count += 1
    return filled, fill_count


def _smooth_positions(
    values: list[tuple[float, float] | None],
    window: int,
) -> list[tuple[float, float] | None]:
    if window <= 1:
        return values

    radius = max(1, window // 2)
    smoothed: list[tuple[float, float] | None] = []
    for idx, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue
        neighbors = [
            values[j]
            for j in range(max(0, idx - radius), min(len(values), idx + radius + 1))
            if values[j] is not None
        ]
        if not neighbors:
            smoothed.append(value)
            continue
        smoothed.append(
            (
                float(np.mean([p[0] for p in neighbors])),
                float(np.mean([p[1] for p in neighbors])),
            )
        )
    return smoothed


def _smooth_player_tracks(
    observations: list[FrameObservation],
    max_gap: int = 3,
    window: int = 3,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "gap_fills": {"A": 0, "B": 0},
        "raw_samples": {"A": 0, "B": 0},
        "smoothed_samples": {"A": 0, "B": 0},
        "max_gap_frames": max_gap,
        "smoothing_window": window,
    }
    if not observations:
        return diagnostics

    for label in ("A", "B"):
        values = [obs.player_positions.get(label) for obs in observations]
        diagnostics["raw_samples"][label] = sum(1 for item in values if item is not None)
        filled, fill_count = _interpolate_short_gaps(values, max_gap=max_gap)
        smoothed = _smooth_positions(filled, window=window)
        diagnostics["gap_fills"][label] = fill_count
        diagnostics["smoothed_samples"][label] = sum(1 for item in smoothed if item is not None)
        for obs, point in zip(observations, smoothed):
            obs.player_positions[label] = point

    return diagnostics


def track_segment(
    video_path: str,
    segment: Segment,
    frame_step: int = 4,
    downscale_width: int = 640,
    court_calibration: dict[str, Any] | None = None,
) -> SegmentTrack:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    start_frame = int(segment.start_sec * fps)
    end_frame = int(segment.end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    previous_gray: np.ndarray | None = None
    player_state: dict[str, tuple[float, float] | None] = {"A": None, "B": None}
    ball_state: tuple[float, float] | None = None
    observations: list[FrameObservation] = []
    detected_t_position: tuple[float, float] | None = None
    t_position: tuple[float, float] | None = None
    court_metadata: dict[str, Any] = {"mode": "pending"}
    frame_size = (0, 0)

    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if w > downscale_width:
            scale = downscale_width / w
            frame = cv2.resize(frame, (downscale_width, int(h * scale)))
            h, w = frame.shape[:2]

        frame_size = (w, h)
        if detected_t_position is None:
            detected_t_position = _detect_t_position(frame)
            t_position, court_metadata = _resolve_court_metadata(
                frame_size=frame_size,
                detected_t=detected_t_position,
                court_calibration=court_calibration,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_ratio = 0.0
        player_candidates: list[_Candidate] = []
        ball_candidates: list[_Candidate] = []

        if previous_gray is not None:
            diff = cv2.absdiff(previous_gray, gray)
            _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            motion_ratio = float(np.count_nonzero(mask) / mask.size)
            player_candidates = _extract_candidates(mask, min_area=280.0, max_area=15000.0)
            ball_candidates = _extract_candidates(mask, min_area=4.0, max_area=120.0)

        previous_gray = gray

        player_state = _assign_players(player_state, player_candidates)
        ball_state = _assign_ball(ball_state, ball_candidates)

        observations.append(
            FrameObservation(
                timestamp=frame_index / fps,
                player_positions={"A": player_state.get("A"), "B": player_state.get("B")},
                ball_position=ball_state,
                motion_ratio=motion_ratio,
            )
        )
        skipped = 0
        if frame_step > 1:
            for _ in range(frame_step - 1):
                if (frame_index + skipped + 1) > end_frame:
                    break
                if not cap.grab():
                    break
                skipped += 1

        frame_index += 1 + skipped

    cap.release()
    tracking_diagnostics = _smooth_player_tracks(observations)
    if detected_t_position is not None and frame_size[0] and frame_size[1]:
        court_metadata["detected_t_norm"] = {
            "x": round(float(detected_t_position[0] / frame_size[0]), 5),
            "y": round(float(detected_t_position[1] / frame_size[1]), 5),
        }

    return SegmentTrack(
        segment=segment,
        t_position=t_position,
        frame_size=frame_size,
        observations=observations,
        metadata={
            "court_calibration": court_metadata,
            "player_tracking": tracking_diagnostics,
        },
    )
