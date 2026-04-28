from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlayerLabel(str, Enum):
    A = "A"
    B = "B"
    UNKNOWN = "Unknown"


class ShotEvent(BaseModel):
    timestamp: float = Field(..., ge=0)
    player: PlayerLabel = PlayerLabel.UNKNOWN
    type: str = "unknown"
    side: str | None = None
    quality: str | None = None
    length: str | None = None
    result: str | None = None


class RallyPositions(BaseModel):
    A_avg_T_recovery_sec: float | None = None
    B_avg_T_recovery_sec: float | None = None
    A_T_occupancy: float | None = None
    B_T_occupancy: float | None = None
    A_court_coverage: float | None = None
    B_court_coverage: float | None = None
    A_avg_speed: float | None = None
    B_avg_speed: float | None = None
    A_max_speed: float | None = None
    B_max_speed: float | None = None
    A_late_retrievals: int | None = None
    B_late_retrievals: int | None = None


class RallySummary(BaseModel):
    rally_id: int
    start_time: float
    end_time: float
    duration_sec: float
    shots: list[ShotEvent] = Field(default_factory=list)
    positions: RallyPositions = Field(default_factory=RallyPositions)
    outcome: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchTimeline(BaseModel):
    video_path: str
    fps: float
    frame_count: int
    rallies: list[RallySummary] = Field(default_factory=list)
    tactical_patterns: dict[str, float] = Field(default_factory=dict)
    movement_summary: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CoachingInsight(BaseModel):
    model: str
    report_markdown: str
    key_patterns: list[str] = Field(default_factory=list)
    drills: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class AnalysisResult(BaseModel):
    timeline: MatchTimeline
    insight: CoachingInsight | None = None
    source_video_url: str | None = None


class ManualRallySegment(BaseModel):
    rally_id: int | None = Field(default=None, ge=1)
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)
    corrected: bool = False


class CourtCalibration(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.01, le=1.0)
    h: float = Field(..., ge=0.01, le=1.0)
    t_x: float = Field(..., ge=0.0, le=1.0)
    t_y: float = Field(..., ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    video_path: str
    include_llm: bool = True
    llm_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    player_a_name: str = "Player A"
    player_b_name: str = "Player B"
    motion_threshold: float = Field(default=0.018, ge=0.001, le=0.5)
    min_rally_sec: float = Field(default=4.0, ge=1.0, le=120.0)
    idle_gap_sec: float = Field(default=1.2, ge=0.1, le=15.0)
    max_rallies: int | None = Field(default=None, ge=1)
    segment_frame_step: int = Field(default=2, ge=1, le=12)
    tracking_frame_step: int = Field(default=4, ge=1, le=12)
    cv_workers: int | None = Field(default=None, ge=1, le=128)
    analysis_start_minute: float = Field(default=0.0, ge=0.0, le=240.0)
    max_video_minutes: float | None = Field(default=None, gt=0.05, le=240.0)
    manual_segments: list[ManualRallySegment] | None = None
    court_calibration: CourtCalibration | None = None
    youtube_cache_dir: str | None = None
    youtube_cookies_file: str | None = None
    youtube_oauth2: bool = False


class AnalyzeOptions(BaseModel):
    include_llm: bool = True
    llm_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None
    player_a_name: str = "Player A"
    player_b_name: str = "Player B"
    motion_threshold: float = Field(default=0.018, ge=0.001, le=0.5)
    min_rally_sec: float = Field(default=4.0, ge=1.0, le=120.0)
    idle_gap_sec: float = Field(default=1.2, ge=0.1, le=15.0)
    max_rallies: int | None = Field(default=None, ge=1)
    segment_frame_step: int = Field(default=2, ge=1, le=12)
    tracking_frame_step: int = Field(default=4, ge=1, le=12)
    cv_workers: int | None = Field(default=None, ge=1, le=128)
    analysis_start_minute: float = Field(default=0.0, ge=0.0, le=240.0)
    max_video_minutes: float | None = Field(default=None, gt=0.05, le=240.0)
    manual_segments: list[ManualRallySegment] | None = None
    court_calibration: CourtCalibration | None = None
    youtube_cache_dir: str | None = None
    youtube_cookies_file: str | None = None
    youtube_oauth2: bool = False
