"""Who served each rally, and who won it, from the squash rule that the winner of a rally serves the next one.

The server is the hitter of a confirmed serve among a rally's first two hits (see rallies.serves). When the
serve itself was missed, the setup gives it away: in the second or so before the first detected hit, one player
stands in a service box with the other waiting behind on the other half (rallies.serve_shaped, without the
stillness test, scanned back to two seconds before the rally). The side is the half the server stood on.

The winner of rally k is the server of rally k+1. A server who wins must change sides, so the same server on
the same side means the rally was a let (replayed) or the next rally went undetected; it is left unclear,
except across a game break, when the server of the first rally of a game picks either side. A clip's last
rally is unclear (the clips were filmed minutes apart), as is any rally whose next server is unknown.

Checked by eye on one frame per rally (servers and sides): all 65 servers found across the three clips were
right; one rally (IMG_4091 at 165 s, the camera knocked) has none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from squashvid.analysis.io import PLAYERS, Match
from squashvid.analysis.rallies import SERVE_LOOK_S, _median_in, serve_shaped

SETUP_BACK_S = 2.0  # how far before the rally's start to look for a serve setup
SETUP_STEP_S = 0.25
GAME_BREAK_S = 30.0  # between rallies; longer gaps are taken as a break between games
SERVER_COLUMNS = ["server", "server_side", "server_source", "winner", "winner_basis"]


def server_of(m: Match, g: pd.DataFrame, start_t: float) -> tuple[str | None, str | None, str]:
    """(server, side "L"/"R", source "serve"/"setup"/"unknown") of the rally with sorted hits g."""
    who, when, how = None, np.nan, "unknown"
    s = g.iloc[:2]
    s = s[s.serve] if "serve" in s else s.iloc[:0]
    if len(s):
        who, when, how = str(s.hitter.iloc[0]), float(s.t.iloc[0]), "serve"
    else:
        for t in np.arange(float(g.t.iloc[0]) + 0.05, start_t - SETUP_BACK_S, -SETUP_STEP_S):
            ok = [p for p in PLAYERS if serve_shaped(m, float(t), p, still=False)]
            if len(ok) == 1:
                who, when, how = ok[0], float(t), "setup"
                break
    if who is None:
        return None, None, how
    x, _, _ = _median_in(m, who, when + SERVE_LOOK_S[0], when + SERVE_LOOK_S[1])
    return who, ("L" if x < m.spec.width / 2 else "R"), how


def winners(r: pd.DataFrame) -> pd.DataFrame:
    """Winner and its basis per rally of one clip, from columns server, server_side, start_t, end_t (sorted)."""
    win, basis = [], []
    for k in range(len(r)):
        if k + 1 == len(r):
            win.append(None), basis.append("last_in_clip")
            continue
        a, b = r.iloc[k], r.iloc[k + 1]
        if b.server is None or pd.isna(b.server):
            win.append(None), basis.append("next_server_unknown")
        elif b.server == a.server and b.server_side == a.server_side:
            if b.start_t - a.end_t > GAME_BREAK_S:
                win.append(b.server), basis.append("served_again_after_break")
            else:
                win.append(None), basis.append("same_server_same_side")
        else:
            win.append(b.server), basis.append("served_next")
    return r.assign(winner=win, winner_basis=basis)


def add_servers(m: Match, rallies: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """The rallies table with SERVER_COLUMNS added."""
    if len(rallies) == 0:
        return rallies.assign(**{c: pd.Series(dtype=object) for c in SERVER_COLUMNS})
    rows = [server_of(m, hits[hits.rally == x.rally].sort_values("t"), float(x.start_t)) for x in rallies.itertuples()]
    r = rallies.sort_values("start_t").assign(server=[w for w, _, _ in rows], server_side=[s for _, s, _ in rows], server_source=[h for _, _, h in rows])
    return winners(r)


def last_hitters(hits: pd.DataFrame) -> pd.Series:
    """Who hit each rally's last detected shot, by rally id."""
    return hits.sort_values("t").groupby("rally").hitter.last()


def outcomes(rallies: pd.DataFrame, hits: pd.DataFrame) -> dict:
    """Counts behind "How rallies ended": per player, rallies won overall, serving and receiving, in short and long
    rallies (split at the median number of shots), and how the rallies that ended on their own shot went."""
    if "winner" not in rallies:
        return {"n_rallies": int(len(rallies)), "n_decided": 0}
    d = rallies[rallies.winner.notna()].copy()
    d["last"] = d.rally.map(last_hitters(hits))
    split = float(d.n_shots.median()) if len(d) else 0.0
    out: dict = {"n_rallies": int(len(rallies)), "n_decided": int(len(d)), "long_from_shots": int(np.ceil(split)), "players": {}}
    for p in PLAYERS:
        won = d.winner == p
        long_ = d.n_shots >= np.ceil(split)
        mine = d["last"] == p
        out["players"][p] = {
            "won": int(won.sum()),
            "serving": [int((won & (d.server == p)).sum()), int((d.server == p).sum())],
            "receiving": [int((won & d.server.notna() & (d.server != p)).sum()), int((d.server.notna() & (d.server != p)).sum())],
            "short": [int((won & ~long_).sum()), int((~long_).sum())],
            "long": [int((won & long_).sum()), int(long_.sum())],
            "ended_on_own_shot": [int((mine & won).sum()), int(mine.sum())],
        }
    return out


__all__ = ["add_servers", "winners", "server_of", "outcomes", "last_hitters", "SERVER_COLUMNS"]
