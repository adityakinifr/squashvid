"""A plain, kind match report (Markdown) from the stats and the verified insights."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from squashvid.analysis.coach import fill
from squashvid.analysis.insights import clock
from squashvid.analysis.io import PLAYERS, WORK
from squashvid.analysis.stats import NEAR_TIN_M

log = logging.getLogger(__name__)


def names(work: Path = WORK) -> dict[str, str]:
    """Player names saved by the viewer's Names button (work/viewer_settings.json, {"names": {"A": ..., "B": ...}})."""
    out = {"A": "Player A", "B": "Player B"}
    try:
        saved = json.loads((work / "viewer_settings.json").read_text()).get("names", {})
    except (OSError, ValueError):
        return out
    out.update({k: v.strip() for k, v in saved.items() if k in out and isinstance(v, str) and v.strip()})
    return out


def fmt(d: dict | None, dec: int = 1, scale: float = 1.0, unit: str = "", per_rally: bool = False) -> str:
    """Value (range), then what it rests on. Per-frame measures cite rallies: thousands of frames are not thousands of observations."""
    if not d or d.get("value") is None:
        return "n/a"
    v = f"{d['value'] * scale:.{dec}f}{unit}"
    if d.get("ci"):
        v += f" ({d['ci'][0] * scale:.{dec}f} to {d['ci'][1] * scale:.{dec}f})"
    return f"{v}, {d['n_rallies']} rallies" if per_rally and d.get("n_rallies") else f"{v}, n={d['n']}"


def stats_table(stats: dict) -> list[str]:
    P = stats["players"]
    rows = [
        ("Shots hit", lambda p: str(p["shots"])),
        ("Average distance from the T during rallies (m)", lambda p: fmt(p["mean_dist_from_T_m"], 1, per_rally=True)),
        ("Distance from the T when the opponent hits (m)", lambda p: fmt(p["mean_dist_from_T_at_opponent_strike_m"], 1)),
        (f"Time within {stats['params']['t_radius_m']:g} m of the T", lambda p: fmt(p["share_within_T_radius"], 0, 100, "%", per_rally=True)),
        ("Time in front of the short line", lambda p: fmt(p["share_in_front_of_short_line"], 0, 100, "%", per_rally=True)),
        ("Time available for their shots (median, s)", lambda p: f"{p['available_time_s']['median']:.1f}, n={p['available_time_s']['n']}" if p["available_time_s"]["median"] is not None else "n/a"),
        ("Shots with under 1.1 s to play them", lambda p: fmt(p["available_time_s"]["share_under_1.1s"], 0, 100, "%")),
        ("Back near the T before the next shot", lambda p: fmt(p["recovery_to_T"]["share_back_on_T_before_next_hit"], 0, 100, "%")),
        ("Distance run in rallies (m)", lambda p: f"{p['distance_covered_in_play_m']['value']}, {p['distance_covered_in_play_m']['n_rallies']} rallies"),
    ]
    out = ["| | {A} | {B} |", "|---|---|---|"]
    for label, fn in rows:
        out.append(f"| {label} | {fn(P['A'])} | {fn(P['B'])} |")
    return out


def ball_table(stats: dict) -> list[str]:
    """Where each player's shots met the front wall, for the shots whose impact was measured."""
    B, P = stats["ball"], stats["players"]

    def pct(d):
        return fmt(d, 0, 100, "%")

    rows = [
        ("Shots with a measured front-wall impact", lambda p: f"{B[p]['n_shots_with_front_wall']} of {P[p]['shots']}"),
        ("Below the top of the tin (a tin, or a mis-measured impact)", lambda p: pct(B[p]["band"].get("below_tin"))),
        ("Between the tin and the service line", lambda p: pct(B[p]["band"].get("tin_to_service_line"))),
        ("Between the service line and 2.5 m", lambda p: pct(B[p]["band"].get("service_line_to_2.5m"))),
        ("Above 2.5 m", lambda p: pct(B[p]["band"].get("above_2.5m"))),
        (f"Less than {NEAR_TIN_M * 100:.0f} cm above the tin", lambda p: pct(B[p].get("near_tin_share"))),
        ("Left third of the wall", lambda p: pct(B[p]["third"].get("left"))),
        ("Centre third", lambda p: pct(B[p]["third"].get("centre"))),
        ("Right third", lambda p: pct(B[p]["third"].get("right"))),
    ]
    out = ["| | {A} | {B} |", "|---|---|---|"]
    for label, fn in rows:
        out.append(f"| {label} | {fn('A')} | {fn('B')} |")
    return out


def outcomes_section(out: dict | None) -> list[str]:
    """How rallies ended, from the winners inferred from who served next (serves.outcomes); empty without them."""
    if not out or not out.get("n_decided"):
        return []
    P, k = out["players"], out["long_from_shots"]

    def of(pair):
        return f"{pair[0]} of {pair[1]}" if pair[1] else "n/a"

    rows = [
        (f"Rallies won (of {out['n_decided']} with a known winner)", lambda p: str(P[p]["won"])),
        ("Won when serving", lambda p: of(P[p]["serving"])),
        ("Won when receiving", lambda p: of(P[p]["receiving"])),
        (f"Won in shorter rallies (under {k} shots)", lambda p: of(P[p]["short"])),
        (f"Won in longer rallies ({k} shots or more)", lambda p: of(P[p]["long"])),
        ("Rallies that ended on their shot, won", lambda p: of(P[p]["ended_on_own_shot"])),
    ]
    lines = [
        "## How rallies ended (inferred)",
        "",
        "Nobody recorded the score, so who won each rally is inferred from the rule that the winner serves the next rally. "
        f"{out['n_rallies'] - out['n_decided']} of {out['n_rallies']} rallies have no winner: the last rally of each clip, "
        "and rallies after which the same player served again from the same side (a let, or a rally the tracking missed). "
        "A rally that ended on a player's shot and went to the other player usually means that last shot was a mistake "
        "(into the tin, out, or bouncing before the front wall), but it can also mean the tracking missed the other player's last shot.",
        "",
        "| | {A} | {B} |",
        "|---|---|---|",
    ]
    return lines + [f"| {label} | {fn('A')} | {fn('B')} |" for label, fn in rows] + [""]


def placement_section(pl: dict | None) -> list[str]:
    """Where the shots went (shots.placement): counts of the shots whose facts the ball tracking showed."""
    if not pl or not pl.get("direction_known"):
        return []
    P = pl["players"]

    def cell(d: dict) -> str:
        return " / ".join(str(v) for v in d.values()) if sum(d.values()) else "n/a"

    zone_name = {"back": "the back (behind the service boxes)", "mid": "mid-court", "front": "the front (in front of the short line)"}
    rows = []
    for z in ("back", "mid", "front"):
        rows.append((f"From {zone_name[z]}: straight / cross-court", lambda p, z=z: cell(P[p]["zones"][z]["direction"])))
        if z != "front":
            rows.append((f"From {zone_name[z]}: deep / middle / short", lambda p, z=z: cell(P[p]["zones"][z]["length"])))
    rows += [
        ("Volleys / off the bounce", lambda p: cell(P[p]["volley"])),
        ("Close to a side wall: yes / no", lambda p: cell(P[p]["tight"])),
        ("Drives / boasts / drops / lobs / kills", lambda p: cell(P[p]["shot_type"])),
    ]
    lines = [
        "## Where the shots went",
        "",
        f"Counts of shots (serves left out) where the ball tracking showed the answer; the direction is known for {pl['direction_known']} of "
        f"{pl['n_shots']} shots, the rest are not counted. Deep means the first bounce was behind the service box, short in front of the "
        "short line, middle between. Straight or cross-court is which side of the middle line the ball went to. These describe the shots; "
        "none of them is coached, because how a choice of shot worked out is not measured here.",
        "",
        "| | {A} | {B} |",
        "|---|---|---|",
    ]
    return lines + [f"| {label} | {fn('A')} | {fn('B')} |" for label, fn in rows] + [""]


def coaching(ins: dict, nm: dict[str, str], stem: str | None = None) -> list[str]:
    """The layered coaching: a short list per player first, then the evidence for a coach or parent.

    stem: the clip the report is about, so its own clip name is left off the timestamps (None when pooled)."""
    from squashvid.players.identity import LOOKS

    def f(text: str) -> str:
        return fill(text, nm)

    lines: list[str] = []
    for p in PLAYERS:
        mine = ins["players"][p]
        lines += [f"## {nm[p]} ({LOOKS[p]})", ""]
        if not mine["try"] and not mine["keep"]:
            lines += ["Nothing stood out clearly enough to be sure of here. That is normal with only a few rallies.", ""]
        if mine["keep"]:
            lines += ["**Doing well**", ""] + [f"- ✓ {f(it.get('plain', it['headline']))}" for it in mine["keep"]] + [""]
        if mine["try"]:
            lines += ["**Work on**", ""] + [f"- ✗ {f(it.get('plain', it['headline']))}" for it in mine["try"]] + [""]
    detail = [(p, it) for p in PLAYERS for kind in ("try", "keep") for it in ins["players"][p][kind]]
    if detail:
        lines += ["## For the coach or parent", ""]
    for p, it in detail:
        lines += [f"### {nm[p]}: {it['title']} ({'to try' if it['kind'] == 'try' else 'to keep'})", "", f"*{f(it['headline'])}*", ""]
        lines += [f(r) for r in it["why"]] + [""]
        if it["evidence"]:
            when = ", ".join(f"{'' if e['clip'] == stem else e['clip'] + ' '}{clock(e['t0'])} ({f(e['label'])})" for e in it["evidence"])
            lines += [f"- Watch {when}. In each, look at {f(it['look_for'])}."]
        if it["drill"]:
            lines += [f"- Drill: {it['drill']}"]
        sure = "Clear" if it["confidence"] == "clear" else "Likely, but the gap could be small"
        lines += [f"- How sure: {sure}. " + " ".join(f(d) for d in it["detail"]), ""]
    return lines


def render(stem: str, stats: dict, ins: dict, title: str | None = None, nm: dict[str, str] | None = None) -> str:
    nm = nm or names()
    mt = stats["match"]
    lines = [
        f"# Match report: {title or stem}",
        "",
        f"We found **{mt['rallies']} rallies** and **{mt['shots_total']} shots** in {mt['clip_seconds']:g} seconds of video. "
        + (f"A typical rally was {mt['rally_shots']['median']:g} shots and {mt['rally_seconds']['median']:g} seconds." if mt["rally_shots"]["median"] is not None else ""),
        "",
        *coaching(ins, nm, stem),
        "## The numbers",
        "",
        "Each cell shows the value, then the range we are fairly sure it lies in, then what it is based on (n shots, or rallies). "
        "Shorter ranges mean more trust. Distance run is shown only for interest: running more or less does not tell us who plays better.",
        "",
        *[fill(x, nm) for x in stats_table(stats)],
        "",
        *[fill(x, nm) for x in outcomes_section(ins.get("outcomes"))],
        *[fill(x, nm) for x in placement_section(ins.get("placement"))],
    ]
    if stats.get("ball") and all(p in stats["ball"] for p in PLAYERS):
        lines += [
            "## Where the ball met the front wall",
            "",
            "Only shots whose front-wall impact the ball tracker measured are counted, so these shares describe those shots. "
            "Some impacts are hidden behind the glass door frame, so not every shot is measured.",
            "",
            *[fill(x, nm) for x in ball_table(stats)],
            "",
        ]
    lines += [
        "## How sure are we?",
        "",
        "- Positions come from one phone behind the back wall. Distances near the front wall are rougher than near the back.",
        "- Shot times and who hit each shot are estimated from the players' movement, the sound of the ball on the front wall and the rule that players take turns. A few shots will be missing or wrong.",
        f"- A finding only appears when it holds up across at least eight rallies and the difference is big enough to matter. {ins['n_rallies']} rallies were available here.",
        "- Everything is about this match only. It says how things went today, not what anyone is like as a player.",
        "",
    ]
    return "\n".join(lines)


def render_pooled(ins: dict, nm: dict[str, str] | None = None) -> str:
    """The all-clips report: pooled coaching only, since positions live in per-clip matches."""
    nm = nm or names()
    stems = ins["clip"].split("+")
    head = ["# Match report: all clips together", "", f"{ins['n_rallies']} rallies across {len(stems)} clips ({', '.join(stems)}). The coaching below pools every rally; see each clip's report for its numbers.", ""]
    tables = outcomes_section(ins.get("outcomes")) + placement_section(ins.get("placement"))
    return "\n".join(head + coaching(ins, nm) + [fill(x, nm) for x in tables])


def clip_report(work: Path, stem: str, nm: dict[str, str] | None = None) -> str | None:
    """A clip's report rendered now from its stats and insights (so it carries the current names); None if not analysed."""
    a = work / stem / "analysis"
    try:
        stats, ins = json.loads((a / "stats.json").read_text()), json.loads((a / "insights.json").read_text())
    except (OSError, ValueError):
        return None
    return render(stem, stats, ins, nm=nm or names(work))


def rewrite_all(work: Path = WORK) -> list[Path]:
    """Re-render every report.md under work/ with the saved names; the numbers are not recomputed."""
    nm, out = names(work), []
    for d in sorted(p for p in work.iterdir() if (p / "analysis" / "insights.json").exists() and (p / "analysis" / "stats.json").exists()):
        md = clip_report(work, d.name, nm)
        if md is not None:
            (d / "analysis" / "report.md").write_text(md)
            out.append(d / "analysis" / "report.md")
    pooled = work / "analysis" / "insights.json"
    if pooled.exists():
        try:
            (work / "analysis" / "report.md").write_text(render_pooled(json.loads(pooled.read_text()), nm))
            out.append(work / "analysis" / "report.md")
        except (OSError, ValueError, KeyError) as e:
            log.warning("pooled report not rewritten: %s", e)
    return out


__all__ = ["render", "render_pooled", "coaching", "clip_report", "rewrite_all", "names", "stats_table", "ball_table", "outcomes_section", "placement_section"]
