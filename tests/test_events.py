from __future__ import annotations

import numpy as np

from squashvid.pipeline.events import aggregate_match, infer_shots, rally_from_track
from squashvid.pipeline.models import FrameObservation, Segment, SegmentTrack


def _make_track() -> SegmentTrack:
    segment = Segment(start_sec=0.0, end_sec=6.0)
    observations: list[FrameObservation] = []

    timestamps = np.linspace(0.0, 6.0, 31)
    for idx, ts in enumerate(timestamps):
        if idx < 15:
            ball = (20.0 + idx * 2.0, 75.0 - idx * 1.2)
        else:
            j = idx - 15
            ball = (50.0 - j * 2.3, 57.0 + j * 1.5)

        obs = FrameObservation(
            timestamp=float(ts),
            player_positions={
                "A": (ball[0] - 5.0, 78.0),
                "B": (ball[0] + 8.0, 40.0),
            },
            ball_position=ball,
            motion_ratio=0.03,
        )
        observations.append(obs)

    return SegmentTrack(
        segment=segment,
        t_position=(50.0, 55.0),
        frame_size=(100, 100),
        observations=observations,
    )


def test_infer_shots_detects_direction_change() -> None:
    track = _make_track()
    shots = infer_shots(track)
    assert shots
    assert shots[0].player.value in {"A", "B", "Unknown"}


def test_rally_and_match_aggregation() -> None:
    track = _make_track()
    rally = rally_from_track(track, rally_id=1)
    timeline = aggregate_match(
        video_path="/tmp/demo.mp4",
        fps=30.0,
        frame_count=180,
        rallies=[rally],
    )

    assert timeline.tactical_patterns["avg_shots_per_rally"] >= 0
    assert "A_avg_T_recovery_sec" in timeline.movement_summary
