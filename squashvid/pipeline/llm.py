from __future__ import annotations

import ast
import json
import os
from typing import Any

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
Use match_intelligence/review_pack diagnostics when present to prioritize high-leverage rallies.
""".strip()


def _local_fallback_insight(timeline: MatchTimeline) -> CoachingInsight:
    tactical = timeline.tactical_patterns
    movement = timeline.movement_summary
    intelligence = timeline.diagnostics.get("match_intelligence", {})
    review_pack = timeline.diagnostics.get("review_pack", {})
    rally_count = len(timeline.rallies)

    key_patterns = [
        f"Crosscourt frequency is {tactical.get('crosscourt_frequency', 0.0):.2f}; monitor if it becomes predictable.",
        f"Backhand pressure rate is {tactical.get('backhand_pressure_rate', 0.0):.2f}; track if depth is maintained.",
        f"Average shots per rally is {tactical.get('avg_shots_per_rally', 0.0):.1f} across {rally_count} rallies.",
    ]

    a_recovery = movement.get("A_avg_T_recovery_sec", 0.0)
    b_recovery = movement.get("B_avg_T_recovery_sec", 0.0)
    a_coverage = movement.get("A_court_coverage", 0.0)
    b_coverage = movement.get("B_court_coverage", 0.0)
    a_speed = movement.get("A_avg_speed", 0.0)
    b_speed = movement.get("B_avg_speed", 0.0)
    a_late = movement.get("A_late_retrievals", 0.0)
    b_late = movement.get("B_late_retrievals", 0.0)

    if a_recovery > 0 and b_recovery > 0:
        if a_recovery > b_recovery:
            key_patterns.append("Player A recovers to T slower than Player B after shots.")
        else:
            key_patterns.append("Player B recovers to T slower than Player A after shots.")
    if a_coverage > 0 or b_coverage > 0:
        key_patterns.append(
            f"Court coverage proxy is Player A {a_coverage:.2f} vs Player B {b_coverage:.2f}; use it to spot who is being stretched."
        )
    if a_late or b_late:
        key_patterns.append(
            f"Late retrieval proxy counts are Player A {a_late:.0f} vs Player B {b_late:.0f}; review the rallies with clustered late recoveries."
        )
    if isinstance(intelligence, dict):
        sequences = intelligence.get("sequence_patterns", [])
        if isinstance(sequences, list) and sequences:
            first = sequences[0]
            if isinstance(first, dict):
                key_patterns.append(
                    f"Most repeated shot sequence proxy: {first.get('sequence', 'n/a')} ({first.get('count', 0)} occurrences)."
                )
        risks = intelligence.get("risk_flags", [])
        if isinstance(risks, list) and risks:
            risk = risks[0]
            if isinstance(risk, dict):
                key_patterns.append(
                    f"Primary risk flag: Player {risk.get('player', '?')} {risk.get('label', 'risk')} - {risk.get('detail', '')}"
                )

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
        f"- A court coverage proxy: **{a_coverage:.2f}**",
        f"- B court coverage proxy: **{b_coverage:.2f}**",
        f"- A average movement speed: **{a_speed:.2f} court/s**",
        f"- B average movement speed: **{b_speed:.2f} court/s**",
        f"- A late retrievals: **{a_late:.0f}**",
        f"- B late retrievals: **{b_late:.0f}**",
        "",
        "### Coaching Adjustments",
        *[f"- {item}" for item in drills],
    ]
    if isinstance(review_pack, dict) and review_pack.get("highlights"):
        report_lines.extend(
            [
                "",
                "### Review Pack",
                *[
                    f"- Rally {item.get('rally_id')}: {item.get('focus')}"
                    for item in review_pack.get("highlights", [])[:5]
                    if isinstance(item, dict)
                ],
            ]
        )

    return CoachingInsight(
        model="local-heuristic",
        report_markdown="\n".join(report_lines),
        key_patterns=key_patterns,
        drills=drills,
        confidence="medium",
    )


def _replace_player_tokens(text: str, player_a_name: str, player_b_name: str) -> str:
    out = str(text)
    out = out.replace("Player A", player_a_name)
    out = out.replace("Player B", player_b_name)
    out = out.replace(" A ", f" {player_a_name} ")
    out = out.replace(" B ", f" {player_b_name} ")
    out = out.replace("A avg", f"{player_a_name} avg")
    out = out.replace("B avg", f"{player_b_name} avg")
    out = out.replace("A T", f"{player_a_name} T")
    out = out.replace("B T", f"{player_b_name} T")
    return out


def _extract_json_block(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


def _coerce_item_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()

        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        ):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    return _coerce_item_text(parsed)
                except Exception:
                    continue
        return text

    if isinstance(value, dict):
        preferred_keys = (
            "drill",
            "title",
            "description",
            "focus",
            "summary",
            "text",
            "name",
            "objective",
            "cue",
        )
        for key in preferred_keys:
            if key in value:
                resolved = _coerce_item_text(value.get(key))
                if resolved:
                    return resolved

        parts: list[str] = []
        for key, raw in value.items():
            resolved = _coerce_item_text(raw)
            if not resolved:
                continue
            parts.append(f"{key}: {resolved}")
        return " | ".join(parts).strip()

    if isinstance(value, (list, tuple, set)):
        parts = [_coerce_item_text(item) for item in value]
        return " | ".join([part for part in parts if part]).strip()

    return str(value).strip()


def _coerce_text_list(values: Any, limit: int) -> list[str]:
    if values is None:
        return []

    if isinstance(values, (list, tuple, set)):
        iterable = list(values)
    else:
        iterable = [values]

    coerced: list[str] = []
    for item in iterable:
        text = _coerce_item_text(item)
        if not text:
            continue
        coerced.append(text)
        if len(coerced) >= limit:
            break
    return coerced


def generate_coaching_insight(
    timeline: MatchTimeline,
    llm_model: str = "gpt-4.1-mini",
    openai_api_key: str | None = None,
    player_a_name: str = "Player A",
    player_b_name: str = "Player B",
) -> CoachingInsight:
    player_a = player_a_name.strip() or "Player A"
    player_b = player_b_name.strip() or "Player B"
    api_key = (openai_api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        local = _local_fallback_insight(timeline)
        local.report_markdown = _replace_player_tokens(local.report_markdown, player_a, player_b)
        local.key_patterns = [_replace_player_tokens(x, player_a, player_b) for x in local.key_patterns]
        local.drills = [_replace_player_tokens(x, player_a, player_b) for x in local.drills]
        return local

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user_payload = {
            "instruction": "Return strict JSON with keys: report_markdown, key_patterns, drills, confidence. Use player names in text.",
            "player_names": {"A": player_a, "B": player_b},
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

        insight = CoachingInsight(
            model=llm_model,
            report_markdown=_coerce_item_text(payload.get("report_markdown", "")).strip(),
            key_patterns=_coerce_text_list(payload.get("key_patterns", []), limit=12),
            drills=_coerce_text_list(payload.get("drills", []), limit=10),
            confidence=str(payload.get("confidence", "medium")),
        )
        insight.report_markdown = _replace_player_tokens(insight.report_markdown, player_a, player_b)
        insight.key_patterns = [_replace_player_tokens(x, player_a, player_b) for x in insight.key_patterns]
        insight.drills = [_replace_player_tokens(x, player_a, player_b) for x in insight.drills]
        return insight
    except Exception:
        local = _local_fallback_insight(timeline)
        local.report_markdown = _replace_player_tokens(local.report_markdown, player_a, player_b)
        local.key_patterns = [_replace_player_tokens(x, player_a, player_b) for x in local.key_patterns]
        local.drills = [_replace_player_tokens(x, player_a, player_b) for x in local.drills]
        return local
