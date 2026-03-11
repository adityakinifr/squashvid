from __future__ import annotations

import squashvid.pipeline.orchestrator as orchestrator
from squashvid.pipeline.models import FrameObservation, Segment, SegmentTrack


def test_resolve_cv_workers_auto_and_bounds() -> None:
    assert orchestrator._resolve_cv_workers(None, 0) == 1
    assert orchestrator._resolve_cv_workers(None, 1) == 1
    assert orchestrator._resolve_cv_workers(8, 3) == 3
    assert orchestrator._resolve_cv_workers(2, 9) == 2


def test_build_rallies_falls_back_to_sequential_when_process_pool_blocked(monkeypatch) -> None:
    segments = [Segment(start_sec=0.0, end_sec=4.0), Segment(start_sec=6.0, end_sec=11.0)]

    class _BlockedPool:
        def __init__(self, *args, **kwargs) -> None:
            raise PermissionError("blocked")

    def _fake_track_segment(video_path: str, segment: Segment, frame_step: int = 4) -> SegmentTrack:
        return SegmentTrack(
            segment=segment,
            t_position=(50.0, 50.0),
            frame_size=(100, 100),
            observations=[
                FrameObservation(
                    timestamp=segment.start_sec,
                    player_positions={"A": (40.0, 60.0), "B": (60.0, 40.0)},
                    ball_position=(50.0, 50.0),
                    motion_ratio=0.02,
                )
            ],
        )

    def _fake_rally_from_track(track: SegmentTrack, rally_id: int):
        return {"rally_id": rally_id, "start_time": track.segment.start_sec, "end_time": track.segment.end_sec}

    monkeypatch.setattr(orchestrator, "ProcessPoolExecutor", _BlockedPool)
    monkeypatch.setattr(orchestrator, "track_segment", _fake_track_segment)
    monkeypatch.setattr(orchestrator, "rally_from_track", _fake_rally_from_track)

    rallies, workers = orchestrator._build_rallies(
        local_video_path="/tmp/fake.mp4",
        segments=segments,
        tracking_frame_step=4,
        cv_workers=4,
    )
    assert workers == 1
    assert len(rallies) == 2
