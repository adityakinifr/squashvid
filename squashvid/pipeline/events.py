from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

import numpy as np

from squashvid.pipeline.models import FrameObservation, SegmentTrack
from squashvid.schemas import MatchTimeline, PlayerLabel, RallyPositions, RallySummary, ShotEvent


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _court_rect(track: SegmentTrack) -> tuple[float, float, float, float]:
    frame_w, frame_h = track.frame_size
    calibration = track.metadata.get("court_calibration", {}) if track.metadata else {}
    rect = calibration.get("rect_norm", {}) if isinstance(calibration, dict) else {}
    x = float(rect.get("x", 0.0)) * frame_w
    y = float(rect.get("y", 0.0)) * frame_h
    w = max(1.0, float(rect.get("w", 1.0)) * frame_w)
    h = max(1.0, float(rect.get("h", 1.0)) * frame_h)
    return x, y, w, h


def _normalize_to_court(
    track: SegmentTrack,
    point: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point is None or track.frame_size[0] <= 0 or track.frame_size[1] <= 0:
        return None
    rect_x, rect_y, rect_w, rect_h = _court_rect(track)
    return (
        max(0.0, min(1.0, (point[0] - rect_x) / rect_w)),
        max(0.0, min(1.0, (point[1] - rect_y) / rect_h)),
    )


def _zone_for_point(point: tuple[float, float]) -> str:
    x, y = point
    depth = "front" if y < 0.34 else "back" if y > 0.66 else "mid"
    side = "left" if x < 0.42 else "right" if x > 0.58 else "center"
    return f"{depth}_{side}"


def _court_t_position(track: SegmentTrack) -> tuple[float, float] | None:
    if track.t_position is None:
        return None
    return _normalize_to_court(track, track.t_position)


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

    t_pos = _court_t_position(track)
    if t_pos is None:
        return RallyPositions()
    radius = 0.13

    occupancy_counts = {"A": 0, "B": 0}
    available_counts = {"A": 0, "B": 0}
    coverage_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    speed_samples: dict[str, list[float]] = defaultdict(list)
    late_retrievals = {"A": 0, "B": 0}
    previous_points: dict[str, tuple[float, tuple[float, float]] | None] = {"A": None, "B": None}

    for obs in track.observations:
        for label in ("A", "B"):
            p = _normalize_to_court(track, obs.player_positions.get(label))
            if p is None:
                continue
            available_counts[label] += 1
            coverage_points[label].append(p)
            if _distance(p, t_pos) <= radius:
                occupancy_counts[label] += 1

            prev = previous_points.get(label)
            if prev is not None:
                prev_t, prev_p = prev
                dt = max(0.001, obs.timestamp - prev_t)
                speed = _distance(prev_p, p) / dt
                if speed < 5.0:
                    speed_samples[label].append(speed)
                    if speed > 0.42 and _distance(p, t_pos) > 0.28:
                        late_retrievals[label] += 1
            previous_points[label] = (obs.timestamp, p)

    recovery_times: dict[str, list[float]] = defaultdict(list)
    for shot in shots:
        player = shot.player.value
        if player not in ("A", "B"):
            continue
        for obs in track.observations:
            if obs.timestamp <= shot.timestamp:
                continue
            p = _normalize_to_court(track, obs.player_positions.get(player))
            if p is None:
                continue
            if _distance(p, t_pos) <= radius:
                recovery_times[player].append(obs.timestamp - shot.timestamp)
                break

    def _coverage(points: list[tuple[float, float]]) -> float | None:
        if len(points) < 2:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        return float(max(0.0, min(1.0, bbox_area)))

    a_occ = (occupancy_counts["A"] / available_counts["A"]) if available_counts["A"] else None
    b_occ = (occupancy_counts["B"] / available_counts["B"]) if available_counts["B"] else None

    return RallyPositions(
        A_avg_T_recovery_sec=mean(recovery_times["A"]) if recovery_times["A"] else None,
        B_avg_T_recovery_sec=mean(recovery_times["B"]) if recovery_times["B"] else None,
        A_T_occupancy=a_occ,
        B_T_occupancy=b_occ,
        A_court_coverage=_coverage(coverage_points["A"]),
        B_court_coverage=_coverage(coverage_points["B"]),
        A_avg_speed=mean(speed_samples["A"]) if speed_samples["A"] else None,
        B_avg_speed=mean(speed_samples["B"]) if speed_samples["B"] else None,
        A_max_speed=max(speed_samples["A"]) if speed_samples["A"] else None,
        B_max_speed=max(speed_samples["B"]) if speed_samples["B"] else None,
        A_late_retrievals=late_retrievals["A"],
        B_late_retrievals=late_retrievals["B"],
    )


def _movement_metadata(track: SegmentTrack) -> dict[str, object]:
    zone_counts: dict[str, dict[str, int]] = {"A": defaultdict(int), "B": defaultdict(int)}
    path_preview: dict[str, list[dict[str, float]]] = {"A": [], "B": []}
    stride = max(1, len(track.observations) // 40) if track.observations else 1

    for idx, obs in enumerate(track.observations):
        for label in ("A", "B"):
            point = _normalize_to_court(track, obs.player_positions.get(label))
            if point is None:
                continue
            zone_counts[label][_zone_for_point(point)] += 1
            if idx % stride == 0:
                path_preview[label].append(
                    {
                        "t": round(float(obs.timestamp), 3),
                        "x": round(float(point[0]), 4),
                        "y": round(float(point[1]), 4),
                    }
                )

    def _sorted_zones(label: str) -> dict[str, int]:
        return dict(sorted(zone_counts[label].items(), key=lambda item: item[0]))

    return {
        "court_calibration": track.metadata.get("court_calibration", {}),
        "player_tracking": track.metadata.get("player_tracking", {}),
        "court_zones": {"A": _sorted_zones("A"), "B": _sorted_zones("B")},
        "movement_path_preview": path_preview,
    }


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


def _outcome_model(outcome: str) -> dict[str, object]:
    raw = str(outcome or "unknown")
    model: dict[str, object] = {
        "winner": None,
        "loser": None,
        "category": "unknown",
        "confidence": "low",
        "reason": "Outcome could not be inferred from terminal ball/player geometry.",
    }
    if raw.startswith("A winner"):
        model.update(
            {
                "winner": "A",
                "loser": "B",
                "category": "winner",
                "confidence": "medium",
                "reason": "Terminal ball position was far from the opponent.",
            }
        )
    elif raw.startswith("B winner"):
        model.update(
            {
                "winner": "B",
                "loser": "A",
                "category": "winner",
                "confidence": "medium",
                "reason": "Terminal ball position was far from the opponent.",
            }
        )
    elif raw.startswith("A forced error"):
        model.update(
            {
                "winner": "B",
                "loser": "A",
                "category": "forced_error",
                "confidence": "low",
                "reason": "Terminal exchange stayed near the opponent but ended after pressure.",
            }
        )
    elif raw.startswith("B forced error"):
        model.update(
            {
                "winner": "A",
                "loser": "B",
                "category": "forced_error",
                "confidence": "low",
                "reason": "Terminal exchange stayed near the opponent but ended after pressure.",
            }
        )
    elif raw.startswith("A pressure"):
        model.update(
            {
                "winner": "A",
                "loser": "B",
                "category": "pressure",
                "confidence": "low",
                "reason": "Last classified shot belonged to A but terminal geometry was ambiguous.",
            }
        )
    elif raw.startswith("B pressure"):
        model.update(
            {
                "winner": "B",
                "loser": "A",
                "category": "pressure",
                "confidence": "low",
                "reason": "Last classified shot belonged to B but terminal geometry was ambiguous.",
            }
        )
    return model


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


def _rally_pressure_features(
    shots: list[ShotEvent],
    positions: RallyPositions,
    duration_sec: float,
) -> dict[str, object]:
    shot_count = len(shots)
    duration_score = min(1.0, duration_sec / 35.0)
    shot_score = min(1.0, shot_count / 18.0)
    late_total = float((positions.A_late_retrievals or 0) + (positions.B_late_retrievals or 0))
    coverage_total = float((positions.A_court_coverage or 0.0) + (positions.B_court_coverage or 0.0))
    intensity = min(1.0, (duration_score * 0.34) + (shot_score * 0.34) + min(1.0, late_total / 8.0) * 0.2 + min(1.0, coverage_total) * 0.12)

    flags: list[dict[str, object]] = []
    if duration_sec >= 25.0:
        flags.append({"label": "Long Rally", "tone": "stamina", "detail": f"{duration_sec:.1f}s rally duration."})
    if shot_count >= 12:
        flags.append({"label": "Extended Exchange", "tone": "tactical", "detail": f"{shot_count} classified shots."})
    if (positions.A_late_retrievals or 0) >= 3:
        flags.append({"label": "A Late Retrieval Cluster", "tone": "movement", "detail": "A had repeated fast recoveries away from the T."})
    if (positions.B_late_retrievals or 0) >= 3:
        flags.append({"label": "B Late Retrieval Cluster", "tone": "movement", "detail": "B had repeated fast recoveries away from the T."})
    if any(shot.side == "backhand" for shot in shots[-4:]):
        flags.append({"label": "Late Backhand Pressure", "tone": "pressure", "detail": "Backhand-side shots appeared in the final exchange."})
    if any(shot.type in {"boast", "volley drop", "drop"} for shot in shots[-4:]):
        flags.append({"label": "Short-Ball Finish", "tone": "attack", "detail": "Short-court shot appeared late in the rally."})

    return {
        "intensity_score": round(float(intensity), 4),
        "duration_score": round(float(duration_score), 4),
        "shot_score": round(float(shot_score), 4),
        "flags": flags,
    }


def rally_from_track(track: SegmentTrack, rally_id: int) -> RallySummary:
    shots = infer_shots(track)
    positions = _compute_t_metrics(track, shots)
    focus_crop = _focus_crop_metadata(track)
    outcome = _infer_outcome(track, shots)
    avg_motion = (
        mean(obs.motion_ratio for obs in track.observations) if track.observations else 0.0
    )
    pressure = _rally_pressure_features(shots, positions, track.segment.duration_sec)

    return RallySummary(
        rally_id=rally_id,
        start_time=track.segment.start_sec,
        end_time=track.segment.end_sec,
        duration_sec=track.segment.duration_sec,
        shots=shots,
        positions=positions,
        outcome=outcome,
        metadata={
            "avg_motion": round(float(avg_motion), 5),
            "shot_count": len(shots),
            "frame_samples": len(track.observations),
            "focus_crop_norm": focus_crop,
            "movement": _movement_metadata(track),
            "outcome_model": _outcome_model(outcome),
            "pressure": pressure,
            "tactical_flags": pressure["flags"],
        },
    )


def _rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def _shot_mix(shots: list[ShotEvent], attr: str) -> dict[str, float]:
    values = [str(getattr(shot, attr) or "unknown") for shot in shots]
    counts = Counter(values)
    total = sum(counts.values())
    return {key: round(_rate(value, total), 4) for key, value in sorted(counts.items())}


def _rally_winner(rally: RallySummary) -> str | None:
    model = rally.metadata.get("outcome_model", {}) if rally.metadata else {}
    winner = model.get("winner") if isinstance(model, dict) else None
    return str(winner) if winner in {"A", "B"} else None


def _phase_label(index: int, total: int) -> str:
    if total <= 1:
        return "single"
    ratio = index / max(1, total - 1)
    if ratio < 0.34:
        return "early"
    if ratio < 0.67:
        return "middle"
    return "late"


def _phase_summary(rallies: list[RallySummary]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[RallySummary]] = defaultdict(list)
    for idx, rally in enumerate(rallies):
        phase = _phase_label(idx, len(rallies))
        rally.metadata["phase"] = phase
        buckets[phase].append(rally)

    summary: dict[str, dict[str, float]] = {}
    for phase, rows in buckets.items():
        shot_total = sum(len(rally.shots) for rally in rows)
        short_total = sum(
            1
            for rally in rows
            for shot in rally.shots
            if shot.type in {"boast", "volley drop", "drop"}
        )
        backhand_total = sum(
            1
            for rally in rows
            for shot in rally.shots
            if shot.side == "backhand"
        )
        summary[phase] = {
            "rallies": float(len(rows)),
            "avg_duration_sec": round(mean(rally.duration_sec for rally in rows), 4) if rows else 0.0,
            "avg_shots": round(mean(len(rally.shots) for rally in rows), 4) if rows else 0.0,
            "backhand_rate": round(_rate(backhand_total, shot_total), 4),
            "short_attack_rate": round(_rate(short_total, shot_total), 4),
        }
    return summary


def _player_profiles(rallies: list[RallySummary]) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for label in ("A", "B"):
        won = [rally for rally in rallies if _rally_winner(rally) == label]
        lost = [rally for rally in rallies if _rally_winner(rally) not in {None, label}]
        player_shots = [shot for rally in rallies for shot in rally.shots if shot.player.value == label]
        t_values = [
            getattr(rally.positions, f"{label}_T_occupancy")
            for rally in rallies
            if getattr(rally.positions, f"{label}_T_occupancy") is not None
        ]
        coverage_values = [
            getattr(rally.positions, f"{label}_court_coverage")
            for rally in rallies
            if getattr(rally.positions, f"{label}_court_coverage") is not None
        ]
        late_retrievals = sum(
            int(getattr(rally.positions, f"{label}_late_retrievals") or 0)
            for rally in rallies
        )
        avg_speed_values = [
            getattr(rally.positions, f"{label}_avg_speed")
            for rally in rallies
            if getattr(rally.positions, f"{label}_avg_speed") is not None
        ]
        pressure_index = min(
            1.0,
            (1.0 - (mean(t_values) if t_values else 0.0)) * 0.36
            + min(1.0, late_retrievals / max(1, len(rallies) * 3)) * 0.34
            + (mean(coverage_values) if coverage_values else 0.0) * 0.3,
        )

        profiles[label] = {
            "rallies_won": len(won),
            "rallies_lost": len(lost),
            "winner_rate": round(_rate(len(won), len(won) + len(lost)), 4),
            "avg_shots_in_wins": round(mean(len(rally.shots) for rally in won), 4) if won else 0.0,
            "avg_shots_in_losses": round(mean(len(rally.shots) for rally in lost), 4) if lost else 0.0,
            "shot_mix": _shot_mix(player_shots, "type"),
            "side_mix": _shot_mix(player_shots, "side"),
            "avg_t_occupancy": round(mean(t_values), 4) if t_values else 0.0,
            "avg_court_coverage": round(mean(coverage_values), 4) if coverage_values else 0.0,
            "avg_speed": round(mean(avg_speed_values), 4) if avg_speed_values else 0.0,
            "late_retrievals": late_retrievals,
            "pressure_index": round(float(pressure_index), 4),
        }
    return profiles


def _sequence_patterns(rallies: list[RallySummary], window: int = 3) -> list[dict[str, object]]:
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    player_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    outcome_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for rally in rallies:
        shot_types = [shot.type for shot in rally.shots if shot.type and shot.type != "unknown"]
        if len(shot_types) < 2:
            continue
        max_window = min(window, len(shot_types))
        for size in range(2, max_window + 1):
            for idx in range(0, len(shot_types) - size + 1):
                sequence = tuple(shot_types[idx : idx + size])
                sequence_counts[sequence] += 1
                player = rally.shots[min(idx + size - 1, len(rally.shots) - 1)].player.value
                player_counts[sequence][player] += 1
                outcome_counts[sequence][str(rally.outcome or "unknown")] += 1

    total_rallies = max(1, len(rallies))
    patterns = []
    for sequence, count in sequence_counts.most_common(8):
        if count < 2:
            continue
        patterns.append(
            {
                "sequence": " -> ".join(sequence),
                "count": count,
                "rally_rate": round(count / total_rallies, 4),
                "players": dict(sorted(player_counts[sequence].items())),
                "outcomes": dict(outcome_counts[sequence].most_common(3)),
            }
        )
    return patterns


def _risk_flags(profiles: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for label, profile in profiles.items():
        pressure_index = float(profile.get("pressure_index", 0.0) or 0.0)
        if pressure_index >= 0.52:
            flags.append(
                {
                    "player": label,
                    "severity": "high",
                    "label": "Movement Pressure",
                    "detail": "High pressure index from lower T occupancy, court coverage, and late retrieval load.",
                }
            )
        if float(profile.get("winner_rate", 0.0) or 0.0) <= 0.35 and (
            int(profile.get("rallies_won", 0)) + int(profile.get("rallies_lost", 0))
        ) >= 3:
            flags.append(
                {
                    "player": label,
                    "severity": "medium",
                    "label": "Low Rally Conversion",
                    "detail": "Outcome proxy shows this player converting fewer than 35% of resolved rallies.",
                }
            )
        side_mix = profile.get("side_mix", {})
        if isinstance(side_mix, dict) and float(side_mix.get("backhand", 0.0) or 0.0) >= 0.58:
            flags.append(
                {
                    "player": label,
                    "severity": "medium",
                    "label": "Backhand Load",
                    "detail": "Shot proxy shows a heavy share of backhand-side contacts.",
                }
            )
    return flags


def _review_pack(rallies: list[RallySummary], duration_sec: float) -> dict[str, object]:
    scored: dict[int, dict[str, object]] = {}
    if not rallies:
        return {"version": "review-pack-v1", "clip_pad_sec": 1.5, "highlights": [], "multimodal_prompt": ""}

    max_duration = max(1.0, max(rally.duration_sec for rally in rallies))
    max_shots = max(1, max(len(rally.shots) for rally in rallies))

    for rally in rallies:
        pressure = rally.metadata.get("pressure", {}) if rally.metadata else {}
        intensity = float(pressure.get("intensity_score", 0.0) or 0.0) if isinstance(pressure, dict) else 0.0
        score = (
            min(1.0, rally.duration_sec / max_duration) * 0.32
            + min(1.0, len(rally.shots) / max_shots) * 0.26
            + intensity * 0.3
            + (0.12 if "winner" in rally.outcome else 0.0)
        )
        reasons = []
        if rally.duration_sec == max_duration:
            reasons.append("Longest rally")
        if len(rally.shots) == max_shots:
            reasons.append("Most classified shots")
        if intensity >= 0.6:
            reasons.append("High pressure/intensity")
        if "winner" in rally.outcome:
            reasons.append("Terminal winner")
        if not reasons:
            reasons.append("Representative tactical sample")

        scored[rally.rally_id] = {
            "rally_id": rally.rally_id,
            "score": round(float(score), 4),
            "reasons": reasons,
            "start_sec": round(float(rally.start_time), 3),
            "end_sec": round(float(rally.end_time), 3),
            "clip_start_sec": round(max(0.0, float(rally.start_time) - 1.5), 3),
            "clip_end_sec": round(min(duration_sec or rally.end_time, float(rally.end_time) + 1.5), 3),
            "focus": _review_focus(rally),
        }

    highlights = sorted(scored.values(), key=lambda item: item["score"], reverse=True)[:8]
    for idx, item in enumerate(highlights, start=1):
        item["rank"] = idx

    prompt_lines = [
        "You are a high-level squash coach reviewing selected rally clips plus structured CV data.",
        "For each highlighted rally, verify the CV event sequence, identify the cause of advantage/loss, and give one concrete correction.",
        "Pay special attention to T recovery, backhand pressure, short-ball decisions, and fatigue signals.",
        "Highlighted rallies:",
    ]
    for item in highlights:
        prompt_lines.append(
            f"- Rally {item['rally_id']} ({item['clip_start_sec']}s-{item['clip_end_sec']}s): {item['focus']}"
        )

    return {
        "version": "review-pack-v1",
        "clip_pad_sec": 1.5,
        "highlights": highlights,
        "multimodal_prompt": "\n".join(prompt_lines),
    }


def _review_focus(rally: RallySummary) -> str:
    pressure = rally.metadata.get("pressure", {}) if rally.metadata else {}
    flags = pressure.get("flags", []) if isinstance(pressure, dict) else []
    labels = [
        str(flag.get("label"))
        for flag in flags
        if isinstance(flag, dict) and flag.get("label")
    ]
    if labels:
        return f"{rally.outcome}; review {', '.join(labels[:3]).lower()}."
    return f"{rally.outcome}; review shot selection and recovery shape."


def _match_intelligence(rallies: list[RallySummary], duration_sec: float) -> dict[str, object]:
    profiles = _player_profiles(rallies)
    outcome_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    for rally in rallies:
        model = rally.metadata.get("outcome_model", {}) if rally.metadata else {}
        if isinstance(model, dict):
            outcome_counts[str(model.get("category") or "unknown")] += 1
            winner = model.get("winner")
            if winner in {"A", "B"}:
                winner_counts[str(winner)] += 1
            else:
                winner_counts["unknown"] += 1

    return {
        "version": "match-intelligence-v1",
        "outcome_summary": {
            "A_wins": winner_counts.get("A", 0),
            "B_wins": winner_counts.get("B", 0),
            "unknown": winner_counts.get("unknown", 0),
            "categories": dict(sorted(outcome_counts.items())),
        },
        "player_profiles": profiles,
        "phase_splits": _phase_summary(rallies),
        "sequence_patterns": _sequence_patterns(rallies),
        "risk_flags": _risk_flags(profiles),
        "review_pack": _review_pack(rallies, duration_sec),
    }


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
    a_coverage: list[float] = []
    b_coverage: list[float] = []
    a_speed: list[float] = []
    b_speed: list[float] = []
    a_late = 0
    b_late = 0

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
        if rally.positions.A_court_coverage is not None:
            a_coverage.append(rally.positions.A_court_coverage)
        if rally.positions.B_court_coverage is not None:
            b_coverage.append(rally.positions.B_court_coverage)
        if rally.positions.A_avg_speed is not None:
            a_speed.append(rally.positions.A_avg_speed)
        if rally.positions.B_avg_speed is not None:
            b_speed.append(rally.positions.B_avg_speed)
        if rally.positions.A_late_retrievals is not None:
            a_late += rally.positions.A_late_retrievals
        if rally.positions.B_late_retrievals is not None:
            b_late += rally.positions.B_late_retrievals

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
        "A_court_coverage": mean(a_coverage) if a_coverage else 0.0,
        "B_court_coverage": mean(b_coverage) if b_coverage else 0.0,
        "A_avg_speed": mean(a_speed) if a_speed else 0.0,
        "B_avg_speed": mean(b_speed) if b_speed else 0.0,
        "A_late_retrievals": float(a_late),
        "B_late_retrievals": float(b_late),
    }

    notes = [
        "CV output is heuristic; use for trend spotting before coach-level review.",
        "For production use, replace the ball tracker with a squash-specific detector.",
    ]
    inferred_duration = (frame_count / fps) if fps > 0 else 0.0
    duration_sec = max(inferred_duration, max((r.end_time for r in rallies), default=0.0))
    intelligence = _match_intelligence(rallies, duration_sec=duration_sec)

    return MatchTimeline(
        video_path=video_path,
        fps=fps,
        frame_count=frame_count,
        rallies=rallies,
        tactical_patterns={k: round(v, 4) for k, v in tactical.items()},
        movement_summary={k: round(v, 4) for k, v in movement.items()},
        diagnostics={
            "match_intelligence": intelligence,
            "review_pack": intelligence["review_pack"],
        },
        notes=notes,
    )
