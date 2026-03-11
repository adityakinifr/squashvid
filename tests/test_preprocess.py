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
