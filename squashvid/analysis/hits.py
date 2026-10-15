"""Racket contacts: when each shot was hit and by whom.

Evidence per candidate contact (a wrist-speed peak of one player):
  * the swing itself: how far the faster wrist's speed rises above walking;
  * a front-wall impact heard (or, with ball events, seen) shortly afterwards,
    because every shot reaches the front wall once;
  * a faint racket onset in the audio;
  * with ball events: a racket contact near this player.
Front-wall impacts no swing explains become audio-only candidates for both
players. A dynamic programme then picks the best sequence: shots at least
MIN_GAP_S apart, hitters alternating (singles), a penalty for a repeated hitter
because that means the other player's shot went missing.

Position-only results are the fallback; ball events upgrade the evidence and
time when they exist (`Match.ball_events`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from squashvid.analysis.io import PLAYERS, Match

MIN_GAP_S = 0.45
CHAIN_GAP_S = 3.5
SWING_MIN = 4.5  # box heights per second; walking arms stay near 2
SWING_CAP = 40.0
SWING_TO_CONTACT_S = 0.04  # the wrist peaks a little before the ball is struck (pooled over the labels)
FW_WINDOW_S = (0.06, 0.75)


@dataclass(frozen=True)
class Params:
    w_swing: float = 0.7
    w_front: float = 2.0
    w_racket: float = 0.5
    w_ball: float = 2.5
    w_rival: float = 0.6  # this swing against the other player's swing at the same moment
    w_still: float = 1.2  # hitters are nearly still around contact; runners are not
    w_audio_only: float = 0.0  # extra reward for a front-wall impact with no swing under it
    tau: float = 1.8
    alt_bonus: float = 0.5
    same_penalty: float = 0.8
    fast_penalty: float = 0.3  # gap under 0.7 s: possible but rare
    snap_to_ball: bool = True  # time a contact by the ball event when one backs it


def travel_time(hitter_y: float, length: float) -> float:
    """Expected racket-to-front-wall flight time: nearer shots arrive sooner (fitted on 38 labelled shots)."""
    return float(np.clip(0.07 + 0.02 * (length - hitter_y), 0.1, 0.4))


def swing_candidates(m: Match) -> pd.DataFrame:
    rows = []
    for k in PLAYERS:
        w = np.minimum(np.nan_to_num(m.tracks[k].wrist), SWING_CAP)
        ws = pd.Series(w).rolling(3, center=True, min_periods=1).max().to_numpy()
        pk, _ = find_peaks(ws, height=SWING_MIN, distance=6)
        for i in pk:
            rows.append((float(m.t[i]) + SWING_TO_CONTACT_S, k, float(ws[i]), "pose"))
    return pd.DataFrame(rows, columns=["t", "player", "swing", "kind"])


def _support(t: float, times: np.ndarray, expect: float, sigma: float = 0.09, lo: float = FW_WINDOW_S[0], hi: float = FW_WINDOW_S[1]) -> float:
    """Best gaussian match of an event time after t to the expected delay, 0 if none in range."""
    if len(times) == 0:
        return 0.0
    d = times[(times > t + lo) & (times < t + hi)] - t
    return float(np.exp(-0.5 * ((d - expect) / sigma) ** 2).max()) if len(d) else 0.0


def build_candidates(m: Match, params: Params = Params()) -> pd.DataFrame:
    """Candidate contacts with a log-odds-like reward, before alternation."""
    sw = swing_candidates(m)
    L = m.spec.length
    ev = m.ball_events
    fw_audio = m.onsets[m.onsets.kind == "front_wall"].t.to_numpy()
    fw_ball = np.array([])
    ball_rk = pd.DataFrame(columns=["t", "player", "confidence", "player_dist_px"])
    if ev is not None:
        fw_ball = ev[ev.kind == "front_wall"].t.to_numpy()
        ball_rk = ev[(ev.kind == "racket") & ev.player.isin(PLAYERS)][["t", "player", "confidence", "player_dist_px"]].reset_index(drop=True)
    fw_all = np.sort(np.r_[fw_audio, fw_ball])
    rk_aud = m.onsets[m.onsets.kind == "racket"].t.to_numpy()

    rows = []
    for r in sw.itertuples():
        rival = sw[(sw.player != r.player) & (abs(sw.t - r.t) < 0.5)].swing
        rel = float(np.clip(np.log(r.swing / (rival.max() + 1.0 if len(rival) else 1.0)), -1.5, 2.5))
        i0 = m.frame_at(r.t)
        spd = float(m.tracks[r.player].speed[max(i0 - 6, 0) : i0 + 7].mean())
        still = float(np.clip(1.5 - spd, -1.5, 1.0))
        y = m.pos(r.player, r.t)[1]
        tt = travel_time(y if np.isfinite(y) else 4.0, L)
        f_a, f_b = _support(r.t, fw_audio, tt), _support(r.t, fw_ball, tt, sigma=0.06)
        fw = max(f_a, f_b)
        rk = float(np.exp(-0.5 * ((rk_aud - r.t) / 0.04) ** 2).max()) if len(rk_aud) else 0.0
        bs, bt = 0.0, np.nan
        if len(ball_rk):
            b = ball_rk[(ball_rk.player == r.player) & (abs(ball_rk.t - r.t) < 0.15)]
            if len(b):
                bs = float(b.confidence.max())
                bt = float(b.t[b.confidence.idxmax()])
        src = [s for s, v in (("pose", 1), ("audio", f_a > 0.3), ("ball", bs > 0 or f_b > 0.3), ("racket_onset", rk > 0.3)) if v]
        sw_term = np.clip(np.log2(r.swing / SWING_MIN) + 0.3, 0, 3)
        reward = params.w_swing * sw_term + params.w_front * fw + params.w_racket * rk + params.w_ball * bs + params.w_rival * rel + params.w_still * still - params.tau
        rows.append((r.t, r.player, reward, "+".join(src), r.swing, fw, bs, bt))
    cands = pd.DataFrame(rows, columns=["t", "player", "reward", "source", "swing", "front_support", "ball_support", "ball_t"])

    # Ball racket contacts with no pose peak (hidden wrists) are candidates too.
    for b in ball_rk.itertuples():
        near = cands[(cands.player == b.player) & (abs(cands.t - b.t) < 0.15)] if len(cands) else cands
        if len(near) == 0:
            fw = max(_support(b.t, fw_all, 0.2), 0.0)
            reward = params.w_ball * float(b.confidence) + params.w_front * fw - params.tau
            cands.loc[len(cands)] = (float(b.t), b.player, reward, "ball", 0.0, fw, float(b.confidence), float(b.t))

    # Front-wall impacts that no swing candidate explains: the shot happened, the hitter is unknown.
    taken = cands[cands.reward > -1e9]
    pseudo = []
    for f in fw_all:
        explained = False
        for k in PLAYERS:
            tk = taken[taken.player == k]
            y = m.pos(k, f - 0.2)[1]
            tt = travel_time(y if np.isfinite(y) else 4.0, L)
            if len(tk) and (abs(tk.t.to_numpy() + tt - f) < 0.2).any():
                explained = True
        if not explained:
            for k in PLAYERS:
                y = m.pos(k, f - 0.2)[1]
                tt = travel_time(y if np.isfinite(y) else 4.0, L)
                pseudo.append((f - tt, k, params.w_front + params.w_audio_only - params.tau - 0.6, "audio", 0.0, 1.0, 0.0, np.nan))
    if pseudo:
        cands = pd.concat([cands, pd.DataFrame(pseudo, columns=cands.columns)], ignore_index=True)
    return cands.sort_values("t").reset_index(drop=True)


def select_hits(t: np.ndarray, player: np.ndarray, reward: np.ndarray, params: Params = Params()) -> list[int]:
    """Best-scoring subsequence under the alternation and spacing rules.

    Maximises the summed node rewards plus an alternation bonus for each pair of
    consecutive selected shots that are a plausible rally gap apart, minus a
    penalty when the same player hits twice running. Selected shots are at least
    MIN_GAP_S apart whoever hit them. Returns indices into the inputs, in time order.
    """
    n = len(t)
    order = np.argsort(t, kind="stable")
    t, player, reward = np.asarray(t)[order], np.asarray(player)[order], np.asarray(reward)[order]
    best = np.full(n, -np.inf)
    prev = np.full(n, -1)
    far_best, far_arg = -np.inf, -1  # best path ending at least CHAIN_GAP_S before the current node
    far_ptr = 0
    for i in range(n):
        # Advance the pointer of nodes that are now far enough back to break the chain.
        while far_ptr < i and t[i] - t[far_ptr] > CHAIN_GAP_S:
            if best[far_ptr] > far_best:
                far_best, far_arg = best[far_ptr], far_ptr
            far_ptr += 1
        cand_val, cand_arg = 0.0, -1
        if far_best > cand_val:
            cand_val, cand_arg = far_best, far_arg
        for j in range(far_ptr, i):
            gap = t[i] - t[j]
            if gap < MIN_GAP_S:
                continue
            e = params.alt_bonus if player[i] != player[j] else -params.same_penalty
            if gap < 0.7:
                e -= params.fast_penalty
            v = best[j] + e
            if v > cand_val:
                cand_val, cand_arg = v, j
        best[i] = reward[i] + cand_val
        prev[i] = cand_arg
    if n == 0:
        return []
    i = int(np.argmax(best))
    if best[i] <= 0:
        return []
    out = []
    while i >= 0:
        out.append(int(order[i]))
        i = int(prev[i])
    out.reverse()
    return out


def confidence_from(reward: float, alternates: bool) -> float:
    z = 0.9 * (reward + 1.4) / 2.0 + (0.3 if alternates else -0.3) - 0.6
    return float(1 / (1 + np.exp(-z)))


def detect_hits(m: Match, params: Params = Params()) -> pd.DataFrame:
    """Selected contacts with time, hitter, confidence and source. Rally ids come later."""
    cands = build_candidates(m, params)
    if len(cands) == 0:
        return pd.DataFrame(columns=["t", "frame", "hitter", "confidence", "source"])
    keep = select_hits(cands.t.to_numpy(), cands.player.to_numpy(), cands.reward.to_numpy(), params)
    h = cands.iloc[keep].reset_index(drop=True)
    snap = params.snap_to_ball
    prevp = h.player.shift(1)
    nextp = h.player.shift(-1)
    alt = ((prevp != h.player) & prevp.notna()) | ((nextp != h.player) & nextp.notna())
    conf = [confidence_from(r, bool(a)) for r, a in zip(h.reward, alt)]
    out = pd.DataFrame(
        {
            "t": np.where(h.ball_t.notna() & snap, h.ball_t, h.t),
            "frame": [m.frame_at(x) for x in np.where(h.ball_t.notna() & snap, h.ball_t, h.t)],
            "hitter": h.player.to_numpy(),
            "confidence": np.round(conf, 3),
            "source": h.source.to_numpy(),
            "swing": h.swing.to_numpy(),
        }
    )
    # A selected hit with no evidence beyond a lone pose peak and no neighbour is not a shot.
    return out[(out.confidence > 0.25)].reset_index(drop=True)




# --- context per hit -------------------------------------------------------------------------

T_RADIUS_M = 1.0  # "on the T" for shares of time; recovery uses RECOVERY_RADIUS_M
RECOVERY_RADIUS_M = 1.5
STEP_S = 0.1  # distance is summed on a 10 Hz track so foot jitter does not add up
MAX_SPEED_MPS = 8.0


def path_length(m: Match, player: str, t0: float, t1: float) -> float:
    tr = m.tracks[player]
    i0, i1 = m.frame_at(t0), m.frame_at(t1)
    step = max(int(round(STEP_S * m.fps)), 1)
    idx = np.arange(i0, i1 + 1, step)
    if len(idx) < 2:
        return 0.0
    xy = np.c_[tr.x[idx], tr.y[idx]]
    return float(np.nansum(np.hypot(*np.diff(xy, axis=0).T)))


def peak_speed(m: Match, player: str, t0: float, t1: float) -> float:
    s = m.tracks[player].speed[m.frame_at(t0) : m.frame_at(t1) + 1]
    return float(min(np.nanmax(s), MAX_SPEED_MPS)) if len(s) else float("nan")


def zone_cells(x: np.ndarray, y: np.ndarray, m: Match) -> np.ndarray:
    """Murray-style court cell: back / mid / front (by service-box back line and just past the short line) by left / centre / right."""
    from squashvid.players.metrics import court_zone

    return court_zone(np.asarray(x, float), np.asarray(y, float), m.spec)


def add_context(h: pd.DataFrame, m: Match) -> pd.DataFrame:
    """Add both players' positions at each hit and the Murray-style variables of each shot.

    Per hit i (hitter is the returner of shot i-1): available time = t_i - t_(i-1)
    when the previous shot was the opponent's; the returner's distance moved and
    peak speed between the two hits; the hitter's and opponent's distance from the T
    at the hit; the cell the shot was hit from and the cell it was returned from (the
    next hitter's cell). Rows are expected in rally and time order.
    """
    h = h.sort_values(["rally", "shot_index"]).reset_index(drop=True).copy()
    n = len(h)
    pos = {k: np.array([m.pos(k, t) for t in h.t]).reshape(-1, 2) for k in PLAYERS}
    for k in PLAYERS:
        h[f"{k}_x"], h[f"{k}_y"] = pos[k][:, 0], pos[k][:, 1]
    is_a = (h.hitter == "A").to_numpy()[:, None]
    hit_xy = np.where(is_a, pos["A"], pos["B"])
    opp_xy = np.where(is_a, pos["B"], pos["A"])
    h["hitter_x"], h["hitter_y"] = hit_xy[:, 0], hit_xy[:, 1]
    h["opp_x"], h["opp_y"] = opp_xy[:, 0], opp_xy[:, 1]
    h["hitter_T_dist"] = np.hypot(hit_xy[:, 0] - m.T[0], hit_xy[:, 1] - m.T[1])
    h["opp_T_dist"] = np.hypot(opp_xy[:, 0] - m.T[0], opp_xy[:, 1] - m.T[1])
    prev_t = h.groupby("rally").t.shift(1)
    prev_h = h.groupby("rally").hitter.shift(1)
    valid = prev_t.notna() & (prev_h != h.hitter)
    h["available_time"] = np.where(valid, h.t - prev_t, np.nan)
    moved, peak = np.full(n, np.nan), np.full(n, np.nan)
    for i in np.flatnonzero(valid.to_numpy()):
        moved[i] = path_length(m, h.hitter[i], prev_t[i], h.t[i])
        peak[i] = peak_speed(m, h.hitter[i], prev_t[i], h.t[i])
    h["returner_moved"], h["returner_peak_speed"] = moved, peak
    h["cell_hit_from"] = zone_cells(h.hitter_x, h.hitter_y, m) if n else pd.Series(dtype=object)
    nxt = h.groupby("rally").cell_hit_from.shift(-1)
    nxt_h = h.groupby("rally").hitter.shift(-1)
    h["cell_returned_from"] = nxt.where(nxt_h.notna() & (nxt_h != h.hitter))
    # Recovery: time from this hit until the hitter is back within RECOVERY_RADIUS_M of the T, censored at the next hit.
    rec, recd = np.full(n, np.nan), np.zeros(n, bool)
    nxt_t = h.groupby("rally").t.shift(-1)
    for i in range(n):
        end = nxt_t[i] if np.isfinite(nxt_t[i]) else h.t[i] + 3.0
        a, b = m.frame_at(h.t[i]), m.frame_at(end)
        tr = m.tracks[h.hitter[i]]
        d = np.hypot(tr.x[a : b + 1] - m.T[0], tr.y[a : b + 1] - m.T[1])
        ok = np.flatnonzero(d <= RECOVERY_RADIUS_M)
        rec[i] = (ok[0] / m.fps) if len(ok) else end - h.t[i]
        recd[i] = len(ok) > 0
    h["recovery_time"], h["recovered"] = rec, recd
    h["situation"] = [classify_situation(a, mv, d) for a, mv, d in zip(h.available_time, h.returner_moved, h.hitter_T_dist)]
    return h


# Situation clusters approximate Murray's six (defence, maintain stability, pressing, pressure,
# attack, attempted winner), which he found by clustering four variables on elite tracking data:
# time between shots, returner's distance moved, peak speed and distance from the T at the hit.
# His centroids are not in hand, so these are rules of thumb on the same variables, checked only
# for being plausible on this footage. Time is the main axis (Vuckovic 2013: under about 1.1 s is
# "little time"); distance from the T and distance moved say how stretched the returner was.
SITUATION_RULES = {
    "defence": "under 1.1 s available and either more than 2.5 m from the T or more than 3 m run",
    "pressure": "under 1.1 s available otherwise, or 1.1-1.8 s available while stretched (over 2.5 m from the T or 3 m run)",
    "maintain": "1.1-1.8 s available, not stretched, more than 1.5 m from the T; or over 1.8 s available but more than 2.5 m from the T",
    "pressing": "1.1-1.8 s available, within 1.5 m of the T",
    "attack": "over 1.8 s available and within 2.5 m of the T",
    "attempted_winner": "over 1.8 s available, within 1.5 m of the T and under 1 m run",
}


def classify_situation(avail: float, moved: float, dT: float) -> str | None:
    if not (np.isfinite(avail) and np.isfinite(moved) and np.isfinite(dT)):
        return None
    stretched = dT > 2.5 or moved > 3.0
    if avail < 1.1:
        return "defence" if stretched else "pressure"
    if avail <= 1.8:
        if stretched:
            return "pressure"
        return "pressing" if dT <= 1.5 else "maintain"
    if dT <= 1.5 and moved < 1.0:
        return "attempted_winner"
    return "attack" if dT <= 2.5 else "maintain"
