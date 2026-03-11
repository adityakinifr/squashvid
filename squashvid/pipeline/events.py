from __future__ import annotations

from collections import defaultdict
from statistics import mean

import numpy as np

from squashvid.pipeline.models import FrameObservation, SegmentTrack
from squashvid.schemas import MatchTimeline, PlayerLabel, RallyPositions, RallySummary, ShotEvent


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _nearest_player(
    obs: FrameObservation,
    ball: tuple[float, float],
) -> PlayerLabel:
    distances: list[tuple[PlayerLabel, float]] = []
    for label in ("A", "B"):
        player_pos = obs.player_positions.get(label)
        if player_pos is None:
            continue
        distances.append((PlayerLabel(label), _distance(player_pos, ball)))

    if not distances:
        return PlayerLabel.UNKNOWN
    return min(distances, key=lambda item: item[1])[0]


def _classify_shot(
    prev_ball: tuple[float, float],
    next_ball: tuple[float, float],
    center_x: float,
) -> tuple[str, str, str, str]:
    dx = next_ball[0] - prev_ball[0]
    dy = next_ball[1] - prev_ball[1]

    if abs(dx) > (1.25 * abs(dy)):
        shot_type = "crosscourt"
    elif dy < -8:
        shot_type = "volley drop"
    elif abs(dx) < 6 and dy > 0:
        shot_type = "boast"
    else:
        shot_type = "drive"

    side = "backhand" if prev_ball[0] < center_x else "forehand"

    speed = float(np.hypot(dx, dy))
    quality = "tight" if speed > 15 else "loose"
    length = "short" if prev_ball[1] < 0.45 * next_ball[1] else "deep"
    return shot_type, side, quality, length


def _find_observation_at_or_after(
    observations: list[FrameObservation],
    timestamp: float,
) -> FrameObservation | None:
    for obs in observations:
        if obs.timestamp >= timestamp:
            return obs
    return observations[-1] if observations else None


def infer_shots(track: SegmentTrack) -> list[ShotEvent]:
    observations = [obs for obs in track.observations if obs.ball_position is not None]
    if len(observations) < 4:
        return []

    points = [(obs.timestamp, obs.ball_position[0], obs.ball_position[1]) for obs in observations]  # type: ignore[index]
    center_x = (track.frame_size[0] / 2.0) if track.frame_size[0] else 0.0

    shots: list[ShotEvent] = []
    last_shot_ts = -999.0
    for idx in range(1, len(points) - 1):
        t_prev, x_prev, y_prev = points[idx - 1]
        t_curr, x_curr, y_curr = points[idx]
        t_next, x_next, y_next = points[idx + 1]

        dt1 = max(1e-6, t_curr - t_prev)
        dt2 = max(1e-6, t_next - t_curr)
        v1 = np.array([(x_curr - x_prev) / dt1, (y_curr - y_prev) / dt1], dtype=float)
        v2 = np.array([(x_next - x_curr) / dt2, (y_next - y_curr) / dt2], dtype=float)

        mag1 = float(np.linalg.norm(v1))
        mag2 = float(np.linalg.norm(v2))
        if mag1 < 6.0 and mag2 < 6.0:
            continue

        cos_theta = float(np.dot(v1, v2) / (max(1e-6, mag1) * max(1e-6, mag2)))
        if cos_theta > -0.2:
            continue

        if (t_curr - last_shot_ts) < 0.35:
            continue

        obs = _find_observation_at_or_after(track.observations, t_curr)
        if obs is None or obs.ball_position is None:
            continue

        player = _nearest_player(obs, obs.ball_position)
        shot_type, side, quality, length = _classify_shot(
            prev_ball=(x_prev, y_prev),
            next_ball=(x_next, y_next),
            center_x=center_x,
        )

        shots.append(
            ShotEvent(
                timestamp=t_curr,
                player=player,
                type=shot_type,
                side=side,
                quality=quality,
                length=length,
            )
        )
        last_shot_ts = t_curr

    return shots


def _compute_t_metrics(track: SegmentTrack, shots: list[ShotEvent]) -> RallyPositions:
    if not track.observations or track.t_position is None:
        return RallyPositions()

    t_pos = track.t_position
    frame_w, frame_h = track.frame_size
    radius = min(frame_w, frame_h) * 0.12 if frame_w and frame_h else 40.0

    occupancy_counts = {"A": 0, "B": 0}
    available_counts = {"A": 0, "B": 0}
    coverage_points: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for obs in track.observations:
        for label in ("A", "B"):
            p = obs.player_positions.get(label)
            if p is None:
                continue
            available_counts[label] += 1
            coverage_points[label].append(p)
            if _distance(p, t_pos) <= radius:
                occupancy_counts[label] += 1

    recovery_times: dict[str, list[float]] = defaultdict(list)
    for shot in shots:
        player = shot.player.value
        if player not in ("A", "B"):
            continue
        for obs in track.observations:
            if obs.timestamp <= shot.timestamp:
                continue
            p = obs.player_positions.get(player)
            if p is None:
                continue
            if _distance(p, t_pos) <= radius:
                recovery_times[player].append(obs.timestamp - shot.timestamp)
                break

    def _coverage(points: list[tuple[float, float]]) -> float | None:
        if len(points) < 2 or frame_w == 0 or frame_h == 0:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        court_area = float(frame_w * frame_h)
        return float(bbox_area / court_area) if court_area else None

    a_occ = (occupancy_counts["A"] / available_counts["A"]) if available_counts["A"] else None
    b_occ = (occupancy_counts["B"] / available_counts["B"]) if available_counts["B"] else None

    return RallyPositions(
        A_avg_T_recovery_sec=mean(recovery_times["A"]) if recovery_times["A"] else None,
        B_avg_T_recovery_sec=mean(recovery_times["B"]) if recovery_times["B"] else None,
        A_T_occupancy=a_occ,
        B_T_occupancy=b_occ,
        A_court_coverage=_coverage(coverage_points["A"]),
        B_court_coverage=_coverage(coverage_points["B"]),
    )


def _infer_outcome(track: SegmentTrack, shots: list[ShotEvent]) -> str:
    if not shots:
        return "unknown"

    last_shot = shots[-1]
    if last_shot.player == PlayerLabel.UNKNOWN:
        return "unknown"

    opponent = "B" if last_shot.player == PlayerLabel.A else "A"
    last_obs = next((obs for obs in reversed(track.observations) if obs.ball_position is not None), None)
    if last_obs is None or last_obs.ball_position is None:
        return f"{last_shot.player.value} pressure"

    opp_pos = last_obs.player_positions.get(opponent)
    if opp_pos is None:
        return f"{last_shot.player.value} winner"

    dist = _distance(opp_pos, last_obs.ball_position)
    if dist > min(track.frame_size) * 0.18:
        return f"{last_shot.player.value} winner"

    return f"{opponent} forced error"


def _focus_crop_metadata(track: SegmentTrack) -> dict[str, float]:
    frame_w, frame_h = track.frame_size
    if frame_w <= 0 or frame_h <= 0:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    points: list[tuple[float, float]] = []
    for obs in track.observations:
        for label in ("A", "B"):
            pos = obs.player_positions.get(label)
            if pos is not None:
                points.append(pos)
        if obs.ball_position is not None:
            points.append(obs.ball_position)

    if len(points) < 3:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    pad_x = 0.12 * frame_w
    pad_y = 0.14 * frame_h

    x1 = max(0.0, min_x - pad_x)
    y1 = max(0.0, min_y - pad_y)
    x2 = min(float(frame_w), max_x + pad_x)
    y2 = min(float(frame_h), max_y + pad_y)

    min_w = 0.42 * frame_w
    min_h = 0.55 * frame_h

    current_w = x2 - x1
    current_h = y2 - y1
    if current_w < min_w:
        center_x = (x1 + x2) / 2.0
        x1 = center_x - (min_w / 2.0)
        x2 = center_x + (min_w / 2.0)
    if current_h < min_h:
        center_y = (y1 + y2) / 2.0
        y1 = center_y - (min_h / 2.0)
        y2 = center_y + (min_h / 2.0)

    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > frame_w:
        x1 -= (x2 - frame_w)
        x2 = float(frame_w)
    if y2 > frame_h:
        y1 -= (y2 - frame_h)
        y2 = float(frame_h)

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(frame_w), x2)
    y2 = min(float(frame_h), y2)

    if x2 <= x1 or y2 <= y1:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    norm_x = x1 / frame_w
    norm_y = y1 / frame_h
    norm_w = (x2 - x1) / frame_w
    norm_h = (y2 - y1) / frame_h

    return {
        "x": round(float(norm_x), 5),
        "y": round(float(norm_y), 5),
        "w": round(float(norm_w), 5),
        "h": round(float(norm_h), 5),
    }


def rally_from_track(track: SegmentTrack, rally_id: int) -> RallySummary:
    shots = infer_shots(track)
    positions = _compute_t_metrics(track, shots)
    focus_crop = _focus_crop_metadata(track)
    avg_motion = (
        mean(obs.motion_ratio for obs in track.observations) if track.observations else 0.0
    )

    return RallySummary(
        rally_id=rally_id,
        start_time=track.segment.start_sec,
        end_time=track.segment.end_sec,
        duration_sec=track.segment.duration_sec,
        shots=shots,
        positions=positions,
        outcome=_infer_outcome(track, shots),
        metadata={
            "avg_motion": round(float(avg_motion), 5),
            "shot_count": len(shots),
            "frame_samples": len(track.observations),
            "focus_crop_norm": focus_crop,
        },
    )


def aggregate_match(
    video_path: str,
    fps: float,
    frame_count: int,
    rallies: list[RallySummary],
) -> MatchTimeline:
    shot_total = sum(len(r.shots) for r in rallies)
    shot_types: dict[str, int] = defaultdict(int)
    backhand_count = 0
    short_attacks = 0
    winners = 0

    a_t_recovery: list[float] = []
    b_t_recovery: list[float] = []
    a_occ: list[float] = []
    b_occ: list[float] = []

    for rally in rallies:
        for shot in rally.shots:
            shot_types[shot.type] += 1
            if shot.side == "backhand":
                backhand_count += 1
            if shot.type in {"volley drop", "drop", "boast"}:
                short_attacks += 1
        if "winner" in rally.outcome:
            winners += 1

        if rally.positions.A_avg_T_recovery_sec is not None:
            a_t_recovery.append(rally.positions.A_avg_T_recovery_sec)
        if rally.positions.B_avg_T_recovery_sec is not None:
            b_t_recovery.append(rally.positions.B_avg_T_recovery_sec)
        if rally.positions.A_T_occupancy is not None:
            a_occ.append(rally.positions.A_T_occupancy)
        if rally.positions.B_T_occupancy is not None:
            b_occ.append(rally.positions.B_T_occupancy)

    tactical = {
        "backhand_pressure_rate": (backhand_count / shot_total) if shot_total else 0.0,
        "boast_usage": (shot_types.get("boast", 0) / shot_total) if shot_total else 0.0,
        "crosscourt_frequency": (shot_types.get("crosscourt", 0) / shot_total) if shot_total else 0.0,
        "short_ball_punish_rate": (short_attacks / shot_total) if shot_total else 0.0,
        "winner_rate": (winners / len(rallies)) if rallies else 0.0,
        "avg_shots_per_rally": (shot_total / len(rallies)) if rallies else 0.0,
    }

    movement = {
        "A_avg_T_recovery_sec": mean(a_t_recovery) if a_t_recovery else 0.0,
        "B_avg_T_recovery_sec": mean(b_t_recovery) if b_t_recovery else 0.0,
        "A_T_occupancy": mean(a_occ) if a_occ else 0.0,
        "B_T_occupancy": mean(b_occ) if b_occ else 0.0,
    }

    notes = [
        "CV output is heuristic; use for trend spotting before coach-level review.",
        "For production use, replace the ball tracker with a squash-specific detector.",
    ]

    return MatchTimeline(
        video_path=video_path,
        fps=fps,
        frame_count=frame_count,
        rallies=rallies,
        tactical_patterns={k: round(v, 4) for k, v in tactical.items()},
        movement_summary={k: round(v, 4) for k, v in movement.items()},
        notes=notes,
    )
