# SquashVid

Hybrid **computer vision + LLM** squash video analyzer.

This project implements the architecture you described:

1. **Preprocess video**: detect active rally segments from motion.
2. **Sports CV layer**: track players/ball and estimate T position using OpenCV heuristics.
3. **Structured event timeline**: convert trajectories into rally/shot schema.
4. **LLM reasoning layer**: generate coaching insight from the structured timeline.

## What This MVP Does

- Splits a full match video into rally candidates.
- Extracts per-rally shot proxies (drive, crosscourt, volley drop, boast).
- Computes movement proxies (T recovery, T occupancy, court coverage).
- Aggregates tactical counters:
  - `backhand_pressure_rate`
  - `boast_usage`
  - `crosscourt_frequency`
  - `short_ball_punish_rate`
- Produces:
  - `analysis.json` structured timeline
  - optional coaching report markdown

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI Usage

```bash
squashvid analyze --video /absolute/path/to/match.mp4 --output analysis.json
```

Skip LLM and only run CV/event extraction:

```bash
squashvid analyze --video /absolute/path/to/match.mp4 --output analysis.json --no-llm
```

Control segmentation sensitivity:

```bash
squashvid analyze \
  --video /absolute/path/to/match.mp4 \
  --output analysis.json \
  --motion-threshold 0.02 \
  --min-rally-sec 4.5 \
  --idle-gap-sec 1.3
```

Speed-focused run for long YouTube videos:

```bash
squashvid analyze \
  --video 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --output analysis.json \
  --max-rallies 24 \
  --segment-frame-step 2 \
  --tracking-frame-step 4 \
  --cv-workers 10 \
  --max-video-minutes 12 \
  --player-a-name "Aditya" \
  --player-b-name "Opponent"
```

If `OPENAI_API_KEY` is available, the app calls OpenAI Responses API for coaching insight. Otherwise it returns a local heuristic insight report.
In the web UI, you can also paste an OpenAI API key per request in the **OpenAI API Key (optional)** field.

You can also pass a YouTube URL directly:

```bash
squashvid analyze \
  --video 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --output analysis.json
```

Optional download cache location:

```bash
squashvid analyze \
  --video 'https://youtu.be/VIDEO_ID' \
  --output analysis.json \
  --youtube-cache-dir /absolute/path/to/cache
```

## API Usage

Run server:

```bash
squashvid-api
```

Then open the UI at: `http://localhost:8000/`

UI includes a **Rally Review Scrubber**:
- scrubber/video sits directly under rally filters in the Rally Timeline section
- click any rally card to load that rally
- video view is cropped to the inferred rally focus area
- shot markers appear on the scrubber timeline and are clickable
- selected rally panel shows full shot-by-shot details
- preview controls let you adjust selected rally start/end bounds locally
- saved boundary corrections can be re-run so shots, metrics, and coaching output are recomputed
- correction controls support merging the next rally, deleting false rally windows, and adding a missing rally window
- rally grid supports sorting by match order, longest, shortest, or most shots
- rally picker lets you jump directly to a specific rally number
- filters include winner, rally-length range, and shot-count range
- player names are configurable (A/B labels become custom names across UI + insights)
- coaching insights are rendered as visual cards for patterns, drills, and full written report sections
- advanced controls include CV worker-process count, start minute, and optional duration
- segmentation diagnostics show analyzed window, threshold choice, signal source, bridge behavior, motion preview, candidate segments, and fallback usage

Analyze by file path:

```bash
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"/absolute/path/to/match.mp4","include_llm":true}'
```

Analyze directly from YouTube:

```bash
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":true}'

# with explicit cache dir
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":true,"youtube_cache_dir":"/absolute/path/to/cache","motion_threshold":0.018,"min_rally_sec":4.0,"idle_gap_sec":1.2,"max_rallies":30,"segment_frame_step":2,"tracking_frame_step":4}'

# with CPU-parallel CV and first-N-minutes window
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":true,"cv_workers":10,"max_video_minutes":12}'

# with explicit start/end analysis window
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":false,"analysis_start_minute":4,"max_video_minutes":8}'

# with manual rally boundaries, bypassing automatic segmentation
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":false,"manual_segments":[{"rally_id":1,"start_sec":0.4,"end_sec":34.0,"corrected":true}]}'

# with per-request API key override
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":true,"openai_api_key":"sk-..."}'

# with player naming
curl -X POST http://localhost:8000/analyze/path \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"https://www.youtube.com/watch?v=VIDEO_ID","include_llm":true,"player_a_name":"Aditya","player_b_name":"Opponent"}'
```

Analyze by upload:

```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F file=@/absolute/path/to/match.mp4 \
  -F include_llm=true
```

## Output Schema (per rally)

```json
{
  "rally_id": 18,
  "start_time": 532.1,
  "end_time": 548.7,
  "duration_sec": 16.6,
  "shots": [
    {"player": "A", "type": "drive", "side": "backhand", "quality": "tight"},
    {"player": "B", "type": "crosscourt", "side": "forehand", "length": "short"},
    {"player": "A", "type": "volley drop", "result": "winner"}
  ],
  "positions": {
    "A_avg_T_recovery_sec": 0.82,
    "B_avg_T_recovery_sec": 1.14
  },
  "outcome": "A winner"
}
```

## Important Limitations

- Ball tracking in squash is difficult (speed + occlusion). The current tracker is **heuristic**, not production-grade.
- Shot classification is trajectory-based and should be replaced with a dedicated event model.
- This is a strong scaffold for iterative upgrades, not final competitive analytics.

## Recommended Next Upgrades

1. Replace heuristic ball tracking with a squash-specific detector/tracker.
2. Add pose estimation and footwork labels (lunge, split-step timing).
3. Train a rally outcome classifier (winner / forced / unforced error).
4. Feed selected clips + structured timeline to a multimodal model for higher-fidelity coaching.
