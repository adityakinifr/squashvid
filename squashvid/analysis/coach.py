"""Coaching: findings grouped into themes and told twice, a short line for the player and the evidence for a coach or parent.

Each theme (getting back to the T, taking time away, length, ...) keeps its clearest finding for each player.
A player gets at most three things to try and two to keep doing. A theme appears as both only when one of the
two is about a narrower situation (say, holding the T well overall but not after shots from the back). A thing
to try carries why it matters (its numbers and, for position, what being away from the T cost), the moments to
watch with what to look for in them, and one drill. Every sentence with a number cites the candidates it rests
on and passes the insights verifier; the headlines, cues and drills are fixed coaching text with no numbers.

Findings that rest on who won each rally (inferred from who served next) may say won and lost, and carry a caveat
in their detail; for position, what holding the T was worth in rallies won is one of the reasons.

Text names players as {A} and {B}; fill() swaps in the saved names when a report or the viewer shows it.
"""

from __future__ import annotations

from squashvid.analysis.insights import METRICS, tok, verify_sentence
from squashvid.analysis.io import PLAYERS, other

MAX_TRY = 3
MAX_KEEP = 2

# Title, and weight: how much a finding in this theme matters for a junior's next session (ranks themes against
# each other; within a theme the clearest finding wins).
THEMES = {
    "position": ("Getting back to the T", 1.0),
    "length": ("Length", 0.9),
    "errors": ("Fewer mistakes at the end of rallies", 0.85),
    "pressure": ("Taking time away from the opponent", 0.8),
    "stamina": ("Keeping it up to the end", 0.6),
}
THEME_OF = {
    "t_at_strike": "position", "t_overall": "position", "t_share": "position", "recovered": "position",
    "recover_from_back": "position", "off_T_cost": "position", "time_given": "pressure", "deep_bounce": "length",
    "won_on_T": "position", "last_shot_error": "errors",
}
# Plays about one situation rather than a whole match; a narrow thing to try may sit beside a broad one to keep.
NARROW = {"recover_from_back", "short_time"}

# The fixed coaching text for each play: the line for the player (as a thing to try, or to keep), what to look for
# in the clips, and a drill. {p} is the player, {o} the opponent.
PLAYBOOK = {
    "t_share": {
        "try": "After every shot, move back to the T before {o} hits.",
        "keep": "You spend a lot of each rally on the T. Keep making it your home.",
        "look": "where {p} is standing at the moment {o} hits the ball",
        "drill": "Ghosting to six corners: from the T, move to a corner, play a shadow shot, and be back on the T before you call the next corner. 30 seconds on, 30 off, five times.",
    },
    "t_overall": {
        "try": "Stay closer to the T between shots; most balls can be reached from there.",
        "keep": "You stay close to the T between shots.",
        "look": "how far {p} drifts from the T between shots",
        "drill": "Hit and touch: a partner feeds to any corner; after each shot, touch the T with your foot before the next feed. 3 sets of 12.",
    },
    "t_at_strike": {
        "try": "Be on the T, not behind it, at the moment {o} hits.",
        "keep": "You are usually on the T when {o} hits.",
        "look": "where {p} is at the moment {o} strikes the ball",
        "drill": "Shadow your partner: they play solo shots around the court while you follow from the T and stop on the T, ready, every time they hit. 2 minutes, three times.",
    },
    "recovered": {
        "try": "Hit, then straight back to the T before the next shot.",
        "keep": "You get back to the T before the next shot.",
        "look": "whether {p} is back on the T before {o}'s reply",
        "drill": "Pressure ghosting: a partner calls corners quickly and you finish every movement on the T. 45 seconds on, 45 off, four times.",
    },
    "recover_from_back": {
        "try": "After a shot from the back of the court, push back up to the T before {o} hits.",
        "keep": "After shots from the back, you get back up to the T in time.",
        "look": "how far {p} has come out of the back corner when {o} hits",
        "drill": "Drive and recover: a partner feeds to a back corner; play a straight drive, then move up and touch the T before the next feed. 3 sets of 10 each side.",
    },
    "short_time": {
        "try": "Even after a rushed shot, get back to the T before {o} hits.",
        "keep": "Even when rushed, you still get back to the T.",
        "look": "what {p} does straight after a rushed shot",
        "drill": "Rushed-shot recovery: a partner feeds quick balls to the corners; after each one, reset to the T before the next feed. 3 sets of 10.",
    },
    "time_given": {
        "try": "Take the ball earlier so {o} has less time: step in and volley when you can.",
        "keep": "Your shots keep {o} rushed. Keep taking the ball early.",
        "look": "how long {o} has between {p}'s shot and the reply",
        "drill": "Volley drives: stand just behind the short line while a partner feeds straight drives; volley them back to length. 3 sets of 20 each side.",
    },
    "deep_bounce": {
        "try": "Hit your drives to bounce behind the service box.",
        "keep": "Your drives get to the back of the court.",
        "look": "where the ball first bounces after {p}'s shot",
        "drill": "Length targets: put a towel just behind each service box and count how many of 20 drives bounce on or behind it. Try to beat the number each session.",
    },
    "last_shot_error": {
        "try": "When you are under pressure, hit a safe straight drive to the back instead of going for a winner.",
        "keep": "Your rallies rarely end with your own mistake. Keep choosing the safe shot when you are under pressure.",
        "look": "{p}'s last shot of the rally: where {p} was, and whether a safer shot was on",
        "drill": "Safe-shot games: play first to 7, where a ball into the tin or out costs two points and a straight drive past the service box is always allowed. Then play the same game where short shots are allowed only from in front of the short line.",
    },
    "early_late:t_at_strike": {
        "try": "Keep getting back to the T in the last rallies, not just the first ones.",
        "keep": "You keep getting back to the T right to the end.",
        "look": "where {p} is when {o} hits, in the later rallies",
        "drill": "Finish each session with three 1-minute ghosting sets at full speed, as if it were the end of a close game.",
    },
    "early_late:recovered": {
        "try": "Keep getting back to the T in the last rallies, not just the first ones.",
        "keep": "You keep getting back to the T right to the end.",
        "look": "whether {p} gets back to the T in the later rallies",
        "drill": "Finish each session with three 1-minute ghosting sets at full speed, as if it were the end of a close game.",
    },
}

# Why it matters, from the candidate's display numbers: {v} the player's value, {c} the comparison.
WHY = {
    "t_share": "{p} spent {v}% of rally time within about a metre of the T; {o} spent {c}%.",
    "t_overall": "{p} was on average {v} m from the T during rallies; {o} was {c} m.",
    "t_at_strike": "When {o} hit, {p} was on average {v} m from the T; when {p} hit, {o} was {c} m away.",
    "recovered": "{p} was back near the T before the next shot {v}% of the time; {o} was {c}%.",
    "recover_from_back": "After a shot from the back of the court, {p} was back near the T before {o}'s reply {v}% of the time; {o} managed {c}%.",
    "off_T_cost": "When {p} was more than {far_m} m from the T as {o} hit, {p}'s next shot was rushed {v}% of the time, against {c}% when nearer.",
    "time_given": "{p}'s shots gave {o} {v} s on average to reply; {o}'s gave {p} {c} s.",
    "deep_bounce": "{v}% of {p}'s measured shots first bounced behind the service box; {c}% of {o}'s did.",
    "short_time": "After a rushed shot (under {split} s), {p} got back near the T {v}% of the time, against {c}% after shots with more time.",
    "last_shot_error": "Of the {n_own} rallies that ended on {p}'s shot, {o} won {lost_own} ({v}%); of the {o_n_own} that ended on {o}'s shot, {p} won {o_lost_own} ({c}%).",
    "won_on_T": "{p} won {won_more} of the {n_more} rallies in which {p} spent more time near the T than {o}, and {won_less} of the other {n_less}.",
    "early_late:t_at_strike": "In the later rallies {p} was on average {v} m from the T when {o} hit, against {c} m in the earlier ones.",
    "early_late:recovered": "In the later rallies {p} was back near the T before the next shot {v}% of the time, against {c}% in the earlier ones.",
}
UNIT_GAP = {"%": " percentage points", "m": " m", "s": " s"}
# Said once in the detail of any item that rests on the inferred rally winners.
WINNER_CAVEAT = "Who won each rally is inferred from who served the next one; a let, or a last shot the tracking missed, can put a rally on the wrong side."


def theme(c: dict) -> str:
    return "stamina" if c["generator"] == "early_vs_late" else THEME_OF[c["metric"]]


def play(c: dict) -> str:
    if c["generator"] == "early_vs_late":
        return f"early_late:{c['metric']}"
    return "short_time" if c["generator"] == "short_vs_long" else c["metric"]


def fill(text: str, names: dict[str, str]) -> str:
    """Generated text with {A} and {B} replaced by the players' names."""
    for p in PLAYERS:
        text = text.replace(tok(p), names[p])
    return text


def clarity(c: dict) -> float:
    """The gap in units of its interval's half-width: how sure the direction is, capped so a huge n does not dominate."""
    half = (c["ci"][1] - c["ci"][0]) / 2
    return min(abs(c["diff"]) / max(half, 1e-9), 4.0)


def confidence(c: dict) -> str:
    """'clear' when even the near end of the interval is a difference worth coaching, else 'likely'."""
    near = min(abs(c["ci"][0]), abs(c["ci"][1]))
    return "clear" if near >= METRICS[c["metric"]].get("min_effect", 0) else "likely"


def why(c: dict) -> str:
    d, p = c["display"], c["player"]
    return WHY[play(c)].format(**{**d, "p": tok(p), "o": tok(other(p)), "v": d["value"], "c": d.get("comparison", "")})


def basis(c: dict) -> str:
    """What the finding rests on and how big the gap probably is."""
    d, gap = c["display"], UNIT_GAP[METRICS[c["metric"]]["unit"]]
    return f"Based on {d['n_rallies']} rallies ({d['n']} {c.get('n_label', 'shots')}); the gap is about {d['diff']}{gap}, probably between {d['diff_lo']} and {d['diff_hi']}{gap}."


def choose(cands: list[dict], p: str) -> dict[str, list[dict]]:
    """The headline candidate of each item for player p: at most MAX_TRY to try and MAX_KEEP to keep."""
    mine = [c for c in cands if c["player"] == p and c.get("role") != "why"]
    picked: dict[str, list[dict]] = {"try": [], "keep": []}
    cap = {"try": MAX_TRY, "keep": MAX_KEEP}
    for c in sorted(mine, key=lambda c: THEMES[theme(c)][1] * clarity(c), reverse=True):
        kind = "keep" if c["favourable"] else "try"
        if len(picked[kind]) == cap[kind] or any(theme(x) == theme(c) for x in picked[kind]):
            continue
        # Never praise and criticise the same thing. The one exception reads naturally: holding the T well overall
        # but not after shots from the back (a narrow thing to try beside a broad one to keep), never the reverse.
        try_, keep_ = (c, None) if kind == "try" else (None, c)
        clash = next((x for x in picked["keep" if kind == "try" else "try"] if theme(x) == theme(c)), None)
        if clash is not None:
            try_, keep_ = try_ or clash, keep_ or clash
            if not (play(try_) in NARROW and play(keep_) not in NARROW):
                continue
        picked[kind].append(c)
    return picked


def item(c: dict, kind: str, cands: list[dict]) -> dict:
    p = c["player"]
    o = other(p)
    book = PLAYBOOK[play(c)]
    fmt = {"p": tok(p), "o": tok(o)}
    support = [x for x in cands if x is not c and x["player"] == p and x.get("role") != "why" and theme(x) == theme(c) and x["favourable"] == c["favourable"]][:2]
    reasons = [why(c)]
    cites = [c["id"]]
    evidence = list(c["evidence"])
    if kind == "try" and theme(c) == "position":
        cost = next((x for x in cands if x["player"] == p and x["metric"] == "off_T_cost" and not x["favourable"]), None)
        if cost is not None:
            reasons.append(why(cost))
            cites.append(cost["id"])
            evidence = evidence[:2] + cost["evidence"][:1]
    if theme(c) == "position" and play(c) in ("t_share", "t_overall", "t_at_strike"):
        # What holding the T was worth, in rallies: told for both a thing to try and one to keep.
        worth = next((x for x in cands if x["player"] == p and x["metric"] == "won_on_T" and x["favourable"]), None)
        if worth is not None:
            reasons.append(why(worth))
            cites.append(worth["id"])
    rests_on_winners = any(x.get("allows") for x in [c, *support, *(x for x in cands if x["id"] in cites)])
    return {
        "theme": theme(c),
        "title": THEMES[theme(c)][0],
        "kind": kind,
        "play": play(c),
        "headline": book[kind].format(**fmt),
        "why": reasons,
        "look_for": book["look"].format(**fmt),
        "drill": book["drill"] if kind == "try" else None,
        "detail": [basis(c)] + [f"{why(x)} {basis(x)}" for x in support] + ([WINNER_CAVEAT] if rests_on_winners else []),
        "evidence": evidence,
        "cites": cites + [x["id"] for x in support],
        "confidence": confidence(c),
    }


def _verified(it: dict, by_id: dict[str, dict], rejected: list[dict], p: str) -> dict | None:
    """The item with failing sentences removed; None if none of its reasons survive."""
    def keep(texts):
        good = []
        for t in texts:
            ok, reason = verify_sentence(t, it["cites"], by_id)
            if ok:
                good.append(t)
            else:
                rejected.append({"player": p, "text": t, "reason": reason})
        return good

    reasons = keep(it["why"])
    return {**it, "why": reasons, "detail": keep(it["detail"])} if reasons else None


def write(cands: list[dict]) -> tuple[dict, list[dict]]:
    """Coaching for both players from the unsuppressed candidates: ({player: {"try": [...], "keep": [...]}}, rejections)."""
    by_id = {c["id"]: c for c in cands}
    rejected: list[dict] = []
    out = {}
    for p in PLAYERS:
        picked = choose(cands, p)
        out[p] = {kind: [it for c in picked[kind] if (it := _verified(item(c, kind, cands), by_id, rejected, p))] for kind in ("try", "keep")}
    return out, rejected


__all__ = ["write", "fill", "choose", "item", "PLAYBOOK", "THEMES"]
