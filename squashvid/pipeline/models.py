from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Segment:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class FrameObservation:
    timestamp: float
    player_positions: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    ball_position: tuple[float, float] | None = None
    motion_ratio: float = 0.0


@dataclass(slots=True)
class SegmentTrack:
    segment: Segment
    t_position: tuple[float, float] | None
    frame_size: tuple[int, int]
    observations: list[FrameObservation] = field(default_factory=list)
