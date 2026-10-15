"""A plain, kind match report (Markdown) from the stats and the verified insights."""

from __future__ import annotations

import json
import re

from squashvid.analysis.insights import clock
from squashvid.analysis.io import PLAYERS, WORK


def names() -> dict[str, str]:
    """Player names from work/settings.json ({"players": {"A": "Name", "B": "Name"}}) when it exists."""
    f = WORK / "settings.json"
    out = {"A": "Player A", "B": "Player B"}
    try:
        out.update({k: v for k, v in json.load(open(f)).get("players", {}).items() if k in out and isinstance(v, str) and v.strip()})
    except (OSError, ValueError):
        pass
    return out


def _name_text(text: str, nm: dict[str, str]) -> str:
    return re.sub(r"(?<![\w'])([AB])(?![\w])", lambda mo: nm[mo.group(1)] if nm[mo.group(1)] != f"Player {mo.group(1)}" else mo.group(1), text)


def fmt(d: dict | None, dec: int = 1, scale: float = 1.0, unit: str = "") -> str:
    if not d or d.get("value") is None:
        return "n/a"
    v = f"{d['value'] * scale:.{dec}f}{unit}"
    if d.get("ci"):
        v += f" ({d['ci'][0] * scale:.{dec}f} to {d['ci'][1] * scale:.{dec}f})"
    return f"{v}, n={d['n']}"


def stats_table(stats: dict) -> list[str]:
    P = stats["players"]
    rows = [
        ("Shots hit", lambda p: str(p["shots"])),
        ("Average distance from the T during rallies (m)", lambda p: fmt(p["mean_dist_from_T_m"], 1)),
        ("Distance from the T when the opponent hits (m)", lambda p: fmt(p["mean_dist_from_T_at_opponent_strike_m"], 1)),
        (f"Time within {stats['params']['t_radius_m']:g} m of the T", lambda p: fmt(p["share_within_T_radius"], 0, 100, "%")),
        ("Time in front of the short line", lambda p: fmt(p["share_in_front_of_short_line"], 0, 100, "%")),
        ("Time available for their shots (median, s)", lambda p: f"{p['available_time_s']['median']:.1f}, n={p['available_time_s']['n']}" if p["available_time_s"]["median"] is not None else "n/a"),
        ("Shots with under 1.1 s to play them", lambda p: fmt(p["available_time_s"]["share_under_1.1s"], 0, 100, "%")),
        ("Back near the T before the next shot", lambda p: fmt(p["recovery_to_T"]["share_back_on_T_before_next_hit"], 0, 100, "%")),
        ("Distance run in rallies (m)", lambda p: f"{p['distance_covered_in_play_m']['value']}, {p['distance_covered_in_play_m']['n_rallies']} rallies"),
    ]
    out = ["| | A | B |", "|---|---|---|"]
    for label, fn in rows:
        out.append(f"| {label} | {fn(P['A'])} | {fn(P['B'])} |")
    return out


def render(stem: str, stats: dict, ins: dict, title: str | None = None) -> str:
    nm = names()
    mt = stats["match"]
    lines = [
        f"# Match report: {title or stem}",
        "",
        f"We found **{mt['rallies']} rallies** and **{mt['shots_total']} shots** in {mt['clip_seconds']:g} seconds of video. "
        + (f"A typical rally was {mt['rally_shots']['median']:g} shots and {mt['rally_seconds']['median']:g} seconds." if mt["rally_shots"]["median"] is not None else ""),
        "",
        "This report looks at where each player stood and how much time they had. It does not score anyone, and it does not say who won.",
        "",
        "## The numbers",
        "",
        "Each cell shows the value, then the range we are fairly sure it lies in, then how many things it is based on (n). "
        "Shorter ranges mean more trust. Distance run is shown only for interest: running more or less does not tell us who plays better.",
        "",
        *[_name_text(x, nm) for x in stats_table(stats)],
        "",
    ]
    if stats.get("ball"):
        lines += ["Ball measurements were also available for this clip; see `stats.json`.", ""]
    for p in PLAYERS:
        who = nm[p]
        lines += [f"## {who}", ""]
        for kind, head, empty in (
            ("strengths", "What went well", "Nothing stood out clearly enough to be sure of in this clip. That is normal with only a few rallies."),
            ("improvements", "What to work on", "Nothing stood out clearly enough to be sure of in this clip."),
        ):
            lines += [f"### {head}", ""]
            items = ins["players"][p][kind]
            if not items:
                lines += [empty, ""]
            for it in items:
                lines += [f"**{it['behaviour'][0].upper() + it['behaviour'][1:]}**", ""]
                for s in it["sentences"]:
                    lines += [f"- {_name_text(s['text'], nm)}"]
                if it["evidence"]:
                    lines += ["", "Clips: " + "; ".join(f"{e['clip']} {clock(e['t0'])}-{clock(e['t1'])} ({_name_text(e['label'], nm)})" for e in it["evidence"])]
                lines += [""]
    lines += [
        "## How sure are we?",
        "",
        "- Positions come from one phone behind the back wall. Distances near the front wall are rougher than near the back.",
        "- Shot times and who hit each shot are estimated from the players' movement, the sound of the ball on the front wall and the rule that players take turns. A few shots will be missing or wrong.",
        f"- A finding only appears when it holds up across at least eight rallies. {ins['n_rallies']} rallies were available here.",
        "- Everything is about this match only. It says how things went today, not what anyone is like as a player.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["render", "names", "stats_table"]
