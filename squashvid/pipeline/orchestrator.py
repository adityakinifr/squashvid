from __future__ import annotations

from pathlib import Path

from squashvid.pipeline.events import aggregate_match, rally_from_track
from squashvid.pipeline.llm import generate_coaching_insight
from squashvid.pipeline.preprocess import detect_active_segments, read_video_metadata
from squashvid.pipeline.video_source import prepare_video_source
from squashvid.pipeline.vision import track_segment
from squashvid.schemas import AnalysisResult, AnalyzeOptions, CoachingInsight


def analyze_video(
    video_path: str,
    options: AnalyzeOptions | None = None,
) -> AnalysisResult:
    opts = options or AnalyzeOptions()
    video_source = prepare_video_source(video_path, cache_dir=opts.youtube_cache_dir)
    local_video_path = str(Path(video_source.local_path).expanduser().resolve())

    meta = read_video_metadata(local_video_path)
    segments = detect_active_segments(
        video_path=local_video_path,
        motion_threshold=opts.motion_threshold,
        min_rally_sec=opts.min_rally_sec,
        idle_gap_sec=opts.idle_gap_sec,
        frame_step=opts.segment_frame_step,
        max_rallies=opts.max_rallies,
    )

    rallies = []
    for idx, segment in enumerate(segments, start=1):
        track = track_segment(
            local_video_path,
            segment,
            frame_step=opts.tracking_frame_step,
        )
        if not track.observations:
            continue
        rallies.append(rally_from_track(track, rally_id=idx))

    timeline = aggregate_match(
        video_path=video_path,
        fps=float(meta["fps"]),
        frame_count=int(meta["frame_count"]),
        rallies=rallies,
    )
    if video_source.downloaded:
        title_info = f" ({video_source.title})" if video_source.title else ""
        timeline.notes.append(
            f"Source downloaded from YouTube{title_info} to: {local_video_path}"
        )

    insight: CoachingInsight | None = None
    if opts.include_llm:
        insight = generate_coaching_insight(
            timeline=timeline,
            llm_model=opts.llm_model,
            openai_api_key=opts.openai_api_key,
        )

    return AnalysisResult(timeline=timeline, insight=insight)
