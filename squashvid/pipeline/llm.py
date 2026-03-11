from __future__ import annotations

import json
import os

from squashvid.schemas import CoachingInsight, MatchTimeline

SYSTEM_PROMPT = """
You are a high-level squash coach and match analyst.
Analyze the structured rally dataset and return coaching-grade feedback.
Focus on:
1) tactical patterns,
2) movement inefficiencies,
3) shot selection under pressure,
4) differences between early and late rallies,
5) five concrete coaching adjustments.
Do not only summarize points. Identify recurring causes of lost rallies.
""".strip()


def _local_fallback_insight(timeline: MatchTimeline) -> CoachingInsight:
    tactical = timeline.tactical_patterns
    movement = timeline.movement_summary
    rally_count = len(timeline.rallies)

    key_patterns = [
        f"Crosscourt frequency is {tactical.get('crosscourt_frequency', 0.0):.2f}; monitor if it becomes predictable.",
        f"Backhand pressure rate is {tactical.get('backhand_pressure_rate', 0.0):.2f}; track if depth is maintained.",
        f"Average shots per rally is {tactical.get('avg_shots_per_rally', 0.0):.1f} across {rally_count} rallies.",
    ]

    a_recovery = movement.get("A_avg_T_recovery_sec", 0.0)
    b_recovery = movement.get("B_avg_T_recovery_sec", 0.0)

    if a_recovery > 0 and b_recovery > 0:
        if a_recovery > b_recovery:
            key_patterns.append("Player A recovers to T slower than Player B after shots.")
        else:
            key_patterns.append("Player B recovers to T slower than Player A after shots.")

    drills = [
        "Ghosting with split-step timing: 6x90 seconds, focus on immediate T recovery.",
        "Backhand length ladder: 40 reps aiming deep to back wall target zones.",
        "Pressure pattern drill (drive-drive-boast-volley): 8 sets with quality targets.",
        "Short-ball punish sequence from mid-court feeds, alternating forehand/backhand.",
        "Late-rally decision drill: condition shot selection after 8+ shot exchanges.",
    ]

    report_lines = [
        "## Match Coaching Insight (Local Heuristic)",
        "",
        "### Tactical Patterns",
        f"- Crosscourt frequency: **{tactical.get('crosscourt_frequency', 0.0):.2f}**",
        f"- Backhand pressure rate: **{tactical.get('backhand_pressure_rate', 0.0):.2f}**",
        f"- Short-ball usage/punish proxy: **{tactical.get('short_ball_punish_rate', 0.0):.2f}**",
        "",
        "### Movement",
        f"- A avg T recovery: **{a_recovery:.2f}s**",
        f"- B avg T recovery: **{b_recovery:.2f}s**",
        f"- A T occupancy: **{movement.get('A_T_occupancy', 0.0):.2f}**",
        f"- B T occupancy: **{movement.get('B_T_occupancy', 0.0):.2f}**",
        "",
        "### Coaching Adjustments",
        *[f"- {item}" for item in drills],
    ]

    return CoachingInsight(
        model="local-heuristic",
        report_markdown="\n".join(report_lines),
        key_patterns=key_patterns,
        drills=drills,
        confidence="medium",
    )


def _extract_json_block(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


def generate_coaching_insight(
    timeline: MatchTimeline,
    llm_model: str = "gpt-4.1-mini",
    openai_api_key: str | None = None,
) -> CoachingInsight:
    api_key = (openai_api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _local_fallback_insight(timeline)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user_payload = {
            "instruction": "Return strict JSON with keys: report_markdown, key_patterns, drills, confidence.",
            "timeline": timeline.model_dump(),
        }
        response = client.responses.create(
            model=llm_model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(user_payload, ensure_ascii=True),
                        }
                    ],
                },
            ],
            temperature=0.2,
        )
        raw_text = response.output_text
        payload = _extract_json_block(raw_text)

        return CoachingInsight(
            model=llm_model,
            report_markdown=str(payload.get("report_markdown", "")).strip(),
            key_patterns=[str(x) for x in payload.get("key_patterns", [])][:12],
            drills=[str(x) for x in payload.get("drills", [])][:10],
            confidence=str(payload.get("confidence", "medium")),
        )
    except Exception:
        return _local_fallback_insight(timeline)
