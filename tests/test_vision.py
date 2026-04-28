from __future__ import annotations

from squashvid.pipeline.models import FrameObservation
from squashvid.pipeline.vision import _resolve_court_metadata, _smooth_player_tracks


def test_resolve_court_metadata_uses_manual_frame_coordinates() -> None:
    t_position, metadata = _resolve_court_metadata(
        frame_size=(200, 100),
        detected_t=(100.0, 62.0),
        court_calibration={
            "x": 0.1,
            "y": 0.05,
            "w": 0.8,
            "h": 0.9,
            "t_x": 0.55,
            "t_y": 0.6,
        },
    )

    assert t_position is not None
    assert round(t_position[0], 3) == 110.0
    assert round(t_position[1], 3) == 60.0
    assert metadata["mode"] == "manual"
    assert metadata["rect_norm"]["w"] == 0.8
    assert metadata["t_norm"]["x"] == 0.55


def test_smooth_player_tracks_fills_short_gaps() -> None:
    observations = [
        FrameObservation(timestamp=0.0, player_positions={"A": (10.0, 10.0), "B": (50.0, 50.0)}),
        FrameObservation(timestamp=0.1, player_positions={"A": None, "B": (52.0, 50.0)}),
        FrameObservation(timestamp=0.2, player_positions={"A": (30.0, 30.0), "B": (54.0, 50.0)}),
    ]

    diagnostics = _smooth_player_tracks(observations, max_gap=2, window=1)

    assert observations[1].player_positions["A"] == (20.0, 20.0)
    assert diagnostics["gap_fills"]["A"] == 1
    assert diagnostics["smoothed_samples"]["A"] == 3
