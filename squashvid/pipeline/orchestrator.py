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
    detect_active_segments,
    read_video_metadata,
)
from squashvid.pipeline.video_source import prepare_video_source
from squashvid.pipeline.vision import track_segment
from squashvid.schemas import AnalysisResult, AnalyzeOptions, CoachingInsight, RallySummary


@dataclass(slots=True)
class AnalysisExecution:
    result: AnalysisResult
    local_video_path: str
    source_downloaded: bool
    source_title: str | None = None


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
    task: tuple[int, str, Segment, int],
) -> tuple[int, dict | None]:
    rally_id, local_video_path, segment, tracking_frame_step = task

    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    track = track_segment(
        local_video_path,
        segment,
        frame_step=tracking_frame_step,
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
            )
            if not track.observations:
                continue
            rallies.append(rally_from_track(track, rally_id=idx))
        return rallies, 1

    tasks = [
        (idx, local_video_path, segment, tracking_frame_step)
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
    max_duration_sec = float(opts.max_video_minutes) * 60.0
    segments = detect_active_segments(
        video_path=local_video_path,
        motion_threshold=opts.motion_threshold,
        min_rally_sec=opts.min_rally_sec,
        idle_gap_sec=opts.idle_gap_sec,
        frame_step=opts.segment_frame_step,
        max_rallies=opts.max_rallies,
        max_duration_sec=max_duration_sec,
        start_offset_sec=start_offset_sec,
    )

    rallies, worker_count = _build_rallies(
        local_video_path=local_video_path,
        segments=segments,
        tracking_frame_step=opts.tracking_frame_step,
        cv_workers=opts.cv_workers,
    )

    fps = float(meta["fps"])
    source_duration_sec = float(meta.get("duration_sec", 0.0) or 0.0)
    effective_duration_sec = source_duration_sec
    if max_duration_sec is not None:
        if effective_duration_sec > 0.0:
            effective_duration_sec = min(effective_duration_sec, max_duration_sec)
        else:
            effective_duration_sec = max_duration_sec

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
    timeline.notes.append(f"Segmentation engine: {SEGMENTER_VERSION}.")
    if video_source.downloaded:
        title_info = f" ({video_source.title})" if video_source.title else ""
        timeline.notes.append(
            f"Source downloaded from YouTube{title_info} to: {local_video_path}"
        )
    timeline.notes.append(f"CV workers used: {worker_count}.")
    end_offset_sec = start_offset_sec + max_duration_sec
    timeline.notes.append(
        f"Analysis window: {start_offset_sec:.1f}s to {end_offset_sec:.1f}s "
        f"({opts.analysis_start_minute:.1f} to {opts.analysis_start_minute + opts.max_video_minutes:.1f} minutes)."
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
