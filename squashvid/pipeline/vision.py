from __future__ import annotations

from dataclasses import dataclass

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
    updated = {"A": previous.get("A"), "B": previous.get("B")}
    points = [(c.x, c.y) for c in candidates[:4]]

    if not points:
        return {"A": None, "B": None}

    if previous.get("A") is None and previous.get("B") is None:
        points.sort(key=lambda p: p[0])
        if len(points) == 1:
            return {"A": points[0], "B": None}
        return {"A": points[0], "B": points[-1]}

    available = points.copy()
    for label in ("A", "B"):
        prev_point = previous.get(label)
        if prev_point is None or not available:
            continue
        nearest = min(available, key=lambda p: _distance(prev_point, p))
        if _distance(prev_point, nearest) <= max_jump:
            updated[label] = nearest
            available.remove(nearest)
        else:
            updated[label] = None

    for label in ("A", "B"):
        if updated[label] is None and available:
            updated[label] = available.pop(0)

    if updated["A"] is None and updated["B"] is not None:
        updated["A"] = updated["B"]
        updated["B"] = None

    return updated


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


def track_segment(
    video_path: str,
    segment: Segment,
    frame_step: int = 4,
    downscale_width: int = 640,
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
    t_position: tuple[float, float] | None = None
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
        if t_position is None:
            t_position = _detect_t_position(frame)

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

    return SegmentTrack(
        segment=segment,
        t_position=t_position,
        frame_size=frame_size,
        observations=observations,
    )
