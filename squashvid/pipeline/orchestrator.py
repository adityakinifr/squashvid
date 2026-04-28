from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from squashvid.pipeline.events import aggregate_match, rally_from_track
from squashvid.pipeline.llm import generate_coaching_insight
from squashvid.pipeline.models import Segment
from squashvid.pipeline.preprocess import (
    SEGMENTER_VERSION,
    SegmentationResult,
    detect_active_segments_with_diagnostics,
    read_video_metadata,
)
from squashvid.pipeline.video_source import prepare_video_source
from squashvid.pipeline.vision import track_segment
from squashvid.schemas import (
    AnalysisResult,
    AnalyzeOptions,
    CoachingInsight,
    ManualRallySegment,
    RallySummary,
)


@dataclass(slots=True)
class AnalysisExecution:
    result: AnalysisResult
    local_video_path: str
    source_downloaded: bool
    source_title: str | None = None


def _segments_payload(segments: list[Segment]) -> list[dict[str, float]]:
    return [
        {
            "start_sec": round(float(seg.start_sec), 3),
            "end_sec": round(float(seg.end_sec), 3),
            "duration_sec": round(float(seg.duration_sec), 3),
        }
        for seg in segments
    ]


def _manual_segmentation_result(
    manual_segments: list[ManualRallySegment],
    source_duration_sec: float,
) -> SegmentationResult:
    segments: list[Segment] = []
    corrected_count = 0

    for idx, manual in enumerate(manual_segments, start=1):
        start = float(manual.start_sec)
        end = float(manual.end_sec)
        if end <= start:
            label = manual.rally_id if manual.rally_id is not None else idx
            raise ValueError(f"Manual rally segment {label} must end after it starts.")

        if source_duration_sec > 0.0:
            if start >= source_duration_sec:
                label = manual.rally_id if manual.rally_id is not None else idx
                raise ValueError(f"Manual rally segment {label} starts after the video ends.")
            end = min(end, source_duration_sec)
            if end <= start:
                label = manual.rally_id if manual.rally_id is not None else idx
                raise ValueError(f"Manual rally segment {label} is outside the video duration.")

        if manual.corrected:
            corrected_count += 1
        segments.append(Segment(start_sec=start, end_sec=end))

    segments.sort(key=lambda seg: (seg.start_sec, seg.end_sec))
    window_start = segments[0].start_sec if segments else 0.0
    window_end = segments[-1].end_sec if segments else 0.0
    diagnostics = {
        "version": f"{SEGMENTER_VERSION}-manual-overrides",
        "mode": "manual",
        "manual_override_used": True,
        "manual_segment_count": len(segments),
        "corrected_segment_count": corrected_count,
        "source_duration_sec": round(float(source_duration_sec), 3),
        "window_start_sec": round(float(window_start), 3),
        "window_end_sec": round(float(window_end), 3),
        "window_duration_sec": round(float(max(0.0, window_end - window_start)), 3),
        "final_segment_count": len(segments),
        "final_segments": _segments_payload(segments),
    }
    return SegmentationResult(segments=segments, diagnostics=diagnostics)


def _resolve_cv_workers(requested_workers: int | None, segment_count: int) -> int:
    if segment_count <= 1:
        return 1
    if requested_workers is None:
        return max(1, min(os.cpu_count() or 1, segment_count))
    return max(1, min(int(requested_workers), segment_count))


def _can_use_process_pool() -> bool:
    try:
        import __main__

        main_file = str(getattr(__main__, "__file__", "") or "")
        if not main_file or main_file.startswith("<"):
            return False
        return True
    except Exception:
        return False


def _track_rally_payload(
    task: tuple[int, str, Segment, int, dict | None],
) -> tuple[int, dict | None]:
    rally_id, local_video_path, segment, tracking_frame_step, court_calibration = task

    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    track = track_segment(
        local_video_path,
        segment,
        frame_step=tracking_frame_step,
        court_calibration=court_calibration,
    )
    if not track.observations:
        return rally_id, None

    rally = rally_from_track(track, rally_id=rally_id)
    return rally_id, rally.model_dump(mode="python")


def _build_rallies(
    local_video_path: str,
    segments: list[Segment],
    tracking_frame_step: int,
    cv_workers: int | None,
    court_calibration: dict | None = None,
) -> tuple[list[RallySummary], int]:
    if not segments:
        return [], 1

    worker_count = _resolve_cv_workers(cv_workers, len(segments))
    if worker_count <= 1 or not _can_use_process_pool():
        rallies: list[RallySummary] = []
        for idx, segment in enumerate(segments, start=1):
            track = track_segment(
                local_video_path,
                segment,
                frame_step=tracking_frame_step,
                court_calibration=court_calibration,
            )
            if not track.observations:
                continue
            rallies.append(rally_from_track(track, rally_id=idx))
        return rallies, 1

    tasks = [
        (idx, local_video_path, segment, tracking_frame_step, court_calibration)
        for idx, segment in enumerate(segments, start=1)
    ]
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            payloads = list(executor.map(_track_rally_payload, tasks))
    except (BrokenProcessPool, PermissionError, OSError, RuntimeError):
        # Some restricted runtimes disallow process semaphores/forking.
        rallies: list[RallySummary] = []
        for idx, segment in enumerate(segments, start=1):
            track = track_segment(
                local_video_path,
                segment,
                frame_step=tracking_frame_step,
                court_calibration=court_calibration,
            )
            if not track.observations:
                continue
            rallies.append(rally_from_track(track, rally_id=idx))
        return rallies, 1

    rallies = [
        RallySummary.model_validate(payload)
        for _idx, payload in sorted(payloads, key=lambda item: item[0])
        if payload is not None
    ]
    return rallies, worker_count


def analyze_video(
    video_path: str,
    options: AnalyzeOptions | None = None,
) -> AnalysisResult:
    return analyze_video_execution(video_path, options).result


def analyze_video_execution(
    video_path: str,
    options: AnalyzeOptions | None = None,
) -> AnalysisExecution:
    opts = options or AnalyzeOptions()
    player_a = opts.player_a_name.strip() or "Player A"
    player_b = opts.player_b_name.strip() or "Player B"
    video_source = prepare_video_source(
        video_path,
        cache_dir=opts.youtube_cache_dir,
        cookies_file=opts.youtube_cookies_file,
        use_oauth2=opts.youtube_oauth2,
    )
    local_video_path = str(Path(video_source.local_path).expanduser().resolve())

    meta = read_video_metadata(local_video_path)
    start_offset_sec = float(opts.analysis_start_minute) * 60.0
    max_duration_sec = (
        float(opts.max_video_minutes) * 60.0
        if opts.max_video_minutes is not None
        else None
    )
    if opts.manual_segments:
        segmentation = _manual_segmentation_result(
            manual_segments=opts.manual_segments,
            source_duration_sec=float(meta.get("duration_sec", 0.0) or 0.0),
        )
    else:
        segmentation = detect_active_segments_with_diagnostics(
            video_path=local_video_path,
            motion_threshold=opts.motion_threshold,
            min_rally_sec=opts.min_rally_sec,
            idle_gap_sec=opts.idle_gap_sec,
            frame_step=opts.segment_frame_step,
            max_rallies=opts.max_rallies,
            max_duration_sec=max_duration_sec,
            start_offset_sec=start_offset_sec,
        )
    segments = segmentation.segments
    court_calibration = (
        opts.court_calibration.model_dump(mode="python")
        if opts.court_calibration is not None
        else None
    )

    rallies, worker_count = _build_rallies(
        local_video_path=local_video_path,
        segments=segments,
        tracking_frame_step=opts.tracking_frame_step,
        cv_workers=opts.cv_workers,
        court_calibration=court_calibration,
    )

    fps = float(meta["fps"])
    source_duration_sec = float(meta.get("duration_sec", 0.0) or 0.0)
    if source_duration_sec > 0.0:
        source_window_end_sec = source_duration_sec
    elif max_duration_sec is not None:
        source_window_end_sec = start_offset_sec + max_duration_sec
    else:
        source_window_end_sec = float(segmentation.diagnostics.get("window_end_sec", 0.0) or 0.0)

    effective_start_sec = min(start_offset_sec, source_window_end_sec)
    effective_end_sec = source_window_end_sec
    if max_duration_sec is not None:
        effective_end_sec = min(source_window_end_sec, start_offset_sec + max_duration_sec)
    effective_duration_sec = max(0.0, effective_end_sec - effective_start_sec)

    if fps > 0 and effective_duration_sec > 0:
        effective_frame_count = max(1, int(round(effective_duration_sec * fps)))
    else:
        effective_frame_count = int(meta["frame_count"])

    timeline = aggregate_match(
        video_path=video_path,
        fps=fps,
        frame_count=effective_frame_count,
        rallies=rallies,
    )
    timeline.diagnostics["segmentation"] = segmentation.diagnostics
    timeline.diagnostics["analysis_window"] = {
        "start_sec": round(float(effective_start_sec), 3),
        "end_sec": round(float(effective_end_sec), 3),
        "duration_sec": round(float(effective_duration_sec), 3),
        "max_video_minutes": round(float(opts.max_video_minutes), 3)
        if opts.max_video_minutes is not None
        else None,
    }
    timeline.notes.append(f"Segmentation engine: {SEGMENTER_VERSION}.")
    if video_source.downloaded:
        title_info = f" ({video_source.title})" if video_source.title else ""
        timeline.notes.append(
            f"Source downloaded from YouTube{title_info} to: {local_video_path}"
        )
    timeline.notes.append(f"CV workers used: {worker_count}.")
    if court_calibration is not None:
        timeline.notes.append("Manual court calibration was applied to player movement metrics.")
    end_label = f"{effective_end_sec:.1f}s" if effective_end_sec > 0.0 else "source end"
    max_minutes_label = (
        f"{opts.max_video_minutes:.1f} minutes"
        if opts.max_video_minutes is not None
        else "full available video"
    )
    timeline.notes.append(
        f"Analysis window: {start_offset_sec:.1f}s to {end_label} "
        f"(start {opts.analysis_start_minute:.1f} minutes, duration {max_minutes_label})."
    )
    if segmentation.diagnostics.get("adaptive_used"):
        timeline.notes.append(
            "Adaptive segmentation threshold was used because the base threshold produced weak boundaries."
        )
    if segmentation.diagnostics.get("fallback_full_window_used"):
        timeline.notes.append(
            "No confident rally breaks were detected, so the analyzed window was kept as one fallback segment."
        )
    if segmentation.diagnostics.get("manual_override_used"):
        timeline.notes.append(
            f"Manual rally boundaries were used for {segmentation.diagnostics.get('manual_segment_count', 0)} segments "
            f"({segmentation.diagnostics.get('corrected_segment_count', 0)} corrected)."
        )
    timeline.notes.append(f"Player label mapping: A={player_a}, B={player_b}.")

    insight: CoachingInsight | None = None
    if opts.include_llm:
        insight = generate_coaching_insight(
            timeline=timeline,
            llm_model=opts.llm_model,
            openai_api_key=opts.openai_api_key,
            player_a_name=player_a,
            player_b_name=player_b,
        )

    return AnalysisExecution(
        result=AnalysisResult(timeline=timeline, insight=insight),
        local_video_path=local_video_path,
        source_downloaded=video_source.downloaded,
        source_title=video_source.title,
    )
