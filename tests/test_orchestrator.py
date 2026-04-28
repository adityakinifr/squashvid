from __future__ import annotations

import squashvid.pipeline.orchestrator as orchestrator
from squashvid.pipeline.models import FrameObservation, Segment, SegmentTrack
from squashvid.schemas import AnalyzeOptions, ManualRallySegment, RallySummary


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

    def _fake_track_segment(
        video_path: str,
        segment: Segment,
        frame_step: int = 4,
        court_calibration: dict | None = None,
    ) -> SegmentTrack:
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


def test_analyze_video_execution_uses_manual_segments(monkeypatch) -> None:
    class _PreparedSource:
        local_path = "/tmp/fake.mp4"
        downloaded = False
        title = None

    captured: dict[str, object] = {}

    def _fail_auto_segmentation(*args, **kwargs):
        raise AssertionError("automatic segmentation should not run")

    def _fake_build_rallies(
        local_video_path: str,
        segments: list[Segment],
        tracking_frame_step: int,
        cv_workers: int | None,
        court_calibration: dict | None = None,
    ):
        captured["segments"] = segments
        captured["court_calibration"] = court_calibration
        rallies = [
            RallySummary(
                rally_id=idx,
                start_time=segment.start_sec,
                end_time=segment.end_sec,
                duration_sec=segment.duration_sec,
                outcome="unknown",
            )
            for idx, segment in enumerate(segments, start=1)
        ]
        return rallies, 1

    monkeypatch.setattr(orchestrator, "prepare_video_source", lambda *args, **kwargs: _PreparedSource())
    monkeypatch.setattr(
        orchestrator,
        "read_video_metadata",
        lambda _: {"fps": 30.0, "frame_count": 3000, "duration_sec": 100.0},
    )
    monkeypatch.setattr(orchestrator, "detect_active_segments_with_diagnostics", _fail_auto_segmentation)
    monkeypatch.setattr(orchestrator, "_build_rallies", _fake_build_rallies)

    execution = orchestrator.analyze_video_execution(
        "/tmp/fake.mp4",
        AnalyzeOptions(
            include_llm=False,
            manual_segments=[
                ManualRallySegment(
                    rally_id=1,
                    start_sec=0.4,
                    end_sec=34.0,
                    corrected=True,
                )
            ],
            court_calibration={
                "x": 0.08,
                "y": 0.04,
                "w": 0.84,
                "h": 0.9,
                "t_x": 0.5,
                "t_y": 0.62,
            },
        ),
    )

    assert captured["segments"][0].start_sec == 0.4
    assert captured["segments"][0].end_sec == 34.0
    assert captured["court_calibration"]["w"] == 0.84
    diagnostics = execution.result.timeline.diagnostics["segmentation"]
    assert diagnostics["mode"] == "manual"
    assert diagnostics["manual_override_used"] is True
    assert diagnostics["corrected_segment_count"] == 1
