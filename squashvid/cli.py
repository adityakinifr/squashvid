from __future__ import annotations

import argparse
import json
from pathlib import Path

from squashvid.pipeline.orchestrator import analyze_video
from squashvid.schemas import AnalyzeOptions


def _analyze(args: argparse.Namespace) -> int:
    options = AnalyzeOptions(
        include_llm=not args.no_llm,
        llm_model=args.llm_model,
        motion_threshold=args.motion_threshold,
        min_rally_sec=args.min_rally_sec,
        idle_gap_sec=args.idle_gap_sec,
        max_rallies=args.max_rallies,
        segment_frame_step=args.segment_frame_step,
        tracking_frame_step=args.tracking_frame_step,
        youtube_cache_dir=args.youtube_cache_dir,
    )

    result = analyze_video(args.video, options)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if result.insight is not None:
        insight_path = output_path.with_suffix(".insight.md")
        insight_path.write_text(result.insight.report_markdown, encoding="utf-8")
        print(f"Wrote insight report: {insight_path}")

    print(f"Wrote analysis JSON: {output_path}")
    print(
        f"Rallies detected: {len(result.timeline.rallies)} | "
        f"Avg shots/rally: {result.timeline.tactical_patterns.get('avg_shots_per_rally', 0.0)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Squash video analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a squash video")
    analyze_parser.add_argument(
        "--video",
        required=True,
        help="Path to input video or YouTube URL",
    )
    analyze_parser.add_argument(
        "--output",
        default="analysis.json",
        help="Path to output JSON analysis file",
    )
    analyze_parser.add_argument("--no-llm", action="store_true", help="Skip LLM step")
    analyze_parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="Model used by OpenAI Responses API",
    )
    analyze_parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.018,
        help="Motion threshold for rally segmentation",
    )
    analyze_parser.add_argument(
        "--min-rally-sec",
        type=float,
        default=4.0,
        help="Minimum rally duration in seconds",
    )
    analyze_parser.add_argument(
        "--idle-gap-sec",
        type=float,
        default=1.2,
        help="Idle gap before ending rally segment",
    )
    analyze_parser.add_argument(
        "--max-rallies",
        type=int,
        default=None,
        help="Optional cap on number of rallies to process",
    )
    analyze_parser.add_argument(
        "--segment-frame-step",
        type=int,
        default=2,
        help="Frame stride for rally segmentation pass (higher is faster)",
    )
    analyze_parser.add_argument(
        "--tracking-frame-step",
        type=int,
        default=4,
        help="Frame stride for tracking pass (higher is faster)",
    )
    analyze_parser.add_argument(
        "--youtube-cache-dir",
        default=None,
        help="Optional cache directory for downloaded YouTube videos",
    )
    analyze_parser.set_defaults(func=_analyze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
