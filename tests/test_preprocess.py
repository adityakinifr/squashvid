from __future__ import annotations

import numpy as np

import squashvid.pipeline.preprocess as preprocess


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float, frame_count: int) -> None:
        self._frames = frames
        self._fps = fps
        self._frame_count = frame_count
        self._idx = 0

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV-like API.
        return True

    def get(self, prop: int) -> float:  # noqa: N802 - OpenCV-like API.
        if prop == preprocess.cv2.CAP_PROP_FPS:
            return self._fps
        if prop == preprocess.cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return 0.0

    def set(self, prop: int, value: float) -> bool:  # noqa: N802 - OpenCV-like API.
        if prop == preprocess.cv2.CAP_PROP_POS_FRAMES:
            self._idx = max(0, min(len(self._frames), int(value)))
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:  # noqa: N802 - OpenCV-like API.
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame.copy()

    def grab(self) -> bool:  # noqa: N802 - OpenCV-like API.
        if self._idx >= len(self._frames):
            return False
        self._idx += 1
        return True

    def release(self) -> None:  # noqa: N802 - OpenCV-like API.
        return None


def _stable_frame(value: int = 90) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


def _active_frame(index: int) -> np.ndarray:
    frame = _stable_frame()
    x = 8 + (index % 9) * 12
    frame[38:86, x : x + 34] = 220
    return frame


def test_detect_active_segments_uses_observed_duration_when_frame_count_missing(
    monkeypatch,
) -> None:
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(120)]
    fake_cap = _FakeCapture(frames=frames, fps=30.0, frame_count=0)
    monkeypatch.setattr(preprocess.cv2, "VideoCapture", lambda _: fake_cap)

    segments = preprocess.detect_active_segments(
        video_path="/tmp/fake.mp4",
        motion_threshold=0.99,
        min_rally_sec=4.0,
        idle_gap_sec=1.2,
        frame_step=2,
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec >= 3.9


def test_detect_active_segments_respects_max_duration(monkeypatch) -> None:
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(300)]  # 10s @ 30fps
    fake_cap = _FakeCapture(frames=frames, fps=30.0, frame_count=300)
    monkeypatch.setattr(preprocess.cv2, "VideoCapture", lambda _: fake_cap)

    segments = preprocess.detect_active_segments(
        video_path="/tmp/fake.mp4",
        motion_threshold=0.99,
        min_rally_sec=1.0,
        idle_gap_sec=0.8,
        frame_step=1,
        max_duration_sec=2.5,
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec <= 2.5 + 0.05


def test_detect_active_segments_respects_start_offset_window(monkeypatch) -> None:
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(600)]  # 60s @ 10fps
    fake_cap = _FakeCapture(frames=frames, fps=10.0, frame_count=600)
    monkeypatch.setattr(preprocess.cv2, "VideoCapture", lambda _: fake_cap)

    result = preprocess.detect_active_segments_with_diagnostics(
        video_path="/tmp/fake.mp4",
        motion_threshold=0.99,
        min_rally_sec=1.0,
        idle_gap_sec=0.8,
        frame_step=1,
        start_offset_sec=30.0,
        max_duration_sec=12.0,
    )

    assert len(result.segments) == 1
    assert result.segments[0].start_sec == 30.0
    assert result.segments[0].end_sec <= 42.0
    assert result.diagnostics["window_start_sec"] == 30.0
    assert result.diagnostics["window_end_sec"] == 42.0
    assert result.diagnostics["fallback_full_window_used"] is True


def test_adaptive_segments_can_recover_when_base_threshold_is_too_high() -> None:
    samples: list[tuple[float, float]] = []
    t = 0.0
    dt = 0.2
    for _ in range(6):
        for _ in range(30):  # 6.0s active-ish
            samples.append((t, 0.009))
            t += dt
        for _ in range(10):  # 2.0s idle gap
            samples.append((t, 0.002))
            t += dt

    base = preprocess._segments_from_motion_samples(
        motion_samples=samples,
        threshold=0.018,
        min_rally_sec=4.0,
        idle_gap_sec=1.2,
        total_duration=t,
        max_rallies=None,
    )
    assert not base

    adaptive = preprocess._adaptive_segments_from_samples(
        motion_samples=samples,
        base_threshold=0.018,
        min_rally_sec=4.0,
        idle_gap_sec=1.2,
        total_duration=t,
        max_rallies=None,
    )
    assert len(adaptive) >= 2


def test_merge_close_segments_and_tail_extension() -> None:
    base = [
        preprocess.Segment(start_sec=0.4, end_sec=23.0),
        preprocess.Segment(start_sec=24.6, end_sec=32.6),
        preprocess.Segment(start_sec=40.7, end_sec=62.3),
    ]
    merged = preprocess._merge_close_segments(base, merge_gap_sec=1.8)
    assert len(merged) == 2
    assert merged[0].start_sec == 0.4
    assert merged[0].end_sec == 32.6

    padded = preprocess._extend_segment_tails(
        merged,
        tail_pad_sec=1.4,
        total_duration=120.0,
    )
    assert round(padded[0].end_sec, 2) == 34.0
    # Extension must not cross into next segment.
    assert padded[0].end_sec <= padded[1].start_sec


def test_bridge_fragmented_segments_merges_short_rally_splits() -> None:
    fragments = [
        preprocess.Segment(start_sec=8.0, end_sec=12.2),
        preprocess.Segment(start_sec=15.6, end_sec=22.2),
        preprocess.Segment(start_sec=24.6, end_sec=31.0),
        preprocess.Segment(start_sec=49.0, end_sec=63.0),
    ]

    bridged = preprocess._bridge_fragmented_segments(
        fragments,
        bridge_gap_sec=4.0,
        fragment_sec=12.0,
        max_merged_sec=60.0,
    )

    assert len(bridged) == 2
    assert bridged[0].start_sec == 8.0
    assert bridged[0].end_sec == 31.0
    assert bridged[1].start_sec == 49.0


def test_first_rally_can_extend_back_to_analysis_start() -> None:
    segments = [
        preprocess.Segment(start_sec=24.8, end_sec=31.0),
        preprocess.Segment(start_sec=60.0, end_sec=74.0),
    ]

    extended, used = preprocess._extend_first_rally_start(
        segments=segments,
        start_offset_sec=0.0,
        max_leading_gap_sec=35.0,
    )

    assert used is True
    assert extended[0].start_sec == 0.0
    assert extended[0].end_sec == 31.0
    assert extended[1].start_sec == 60.0


def test_first_rally_does_not_extend_over_long_intro() -> None:
    segments = [preprocess.Segment(start_sec=55.0, end_sec=70.0)]

    extended, used = preprocess._extend_first_rally_start(
        segments=segments,
        start_offset_sec=0.0,
        max_leading_gap_sec=35.0,
    )

    assert used is False
    assert extended[0].start_sec == 55.0


def test_scene_guard_removes_flash_from_next_rally_start(monkeypatch) -> None:
    fps = 10.0
    frames: list[np.ndarray] = []
    for idx in range(220):
        if idx <= 55:
            frames.append(_active_frame(idx))
        elif 110 <= idx <= 112:
            frames.append(np.full((120, 160, 3), 255, dtype=np.uint8))
        elif 120 <= idx <= 180:
            frames.append(_active_frame(idx))
        else:
            frames.append(_stable_frame())

    fake_cap = _FakeCapture(frames=frames, fps=fps, frame_count=len(frames))
    monkeypatch.setattr(preprocess.cv2, "VideoCapture", lambda _: fake_cap)

    result = preprocess.detect_active_segments_with_diagnostics(
        video_path="/tmp/fake.mp4",
        motion_threshold=0.02,
        min_rally_sec=3.0,
        idle_gap_sec=1.2,
        frame_step=1,
    )

    assert len(result.segments) >= 2
    assert result.segments[1].start_sec >= 12.0
    assert result.diagnostics["scene_change_count"] >= 1
    assert result.diagnostics["scene_suppressed_sample_count"] >= 1


def test_motion_preview_payload_is_downsampled() -> None:
    samples = [(float(i), 0.02 if i % 2 == 0 else 0.001) for i in range(500)]

    preview = preprocess._motion_preview_payload(samples, threshold=0.018, limit=50)

    assert len(preview) <= 50
    assert preview[0]["timestamp_sec"] == 0.0
    assert preview[0]["active"] == 1.0
