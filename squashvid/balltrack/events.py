"""Court events from the ball track's kinks, fused with the audio onsets.

For every kink (an instant the ball's image velocity jumps) the stage asks which impact it could be:

* A plane impact (floor, front wall, left or right wall, back glass). The kink's pixel ray meets the
  plane at a court point; the before/after image velocities must then be explained by a bounce off
  that plane (`world.bounce_residual`: ball arriving and leaving, normal part reversed and reduced,
  tangential part kept), at a plausible 3D speed, on the physical surface.
* A racket contact: no plane explains the kink and a player is within reach of the ball.

Audio onsets, shifted by the clip's audio-video offset and the sound's flight time to the camera,
add evidence for any kink they fall on. Front-wall onsets with no kink are kept as audio-only
impacts, since the front wall is the audio stage's dependable event.

Positions use the camera ray to the plane (the floor plane sits one ball radius up). Uncertainty is
the pixel error of the kink propagated to the plane's two axes, inflated when the kink was only seen
from one side, bridged across a gap, or came from audio alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from squashvid.balltrack import world
from squashvid.geom import court
from squashvid.geom.camera import Camera

KINDS = ("front_wall", "side_wall_left", "side_wall_right", "floor", "back_wall", "racket")
COLUMNS = ["t", "frame", "kind", "u", "v", "x", "y", "z", "sigma_a", "sigma_b", "evidence", "confidence", "player", "player_dist_px"]
PLANES = ("floor", "front_wall", "side_wall_left", "side_wall_right", "back_wall")
AUDIO_KIND = {"front_wall": "front_wall", "floor": "floor", "side_wall_left": "side_wall", "side_wall_right": "side_wall", "racket": "racket"}
L_WRI, R_WRI = 9, 10  # COCO wrist keypoints


@dataclass(frozen=True)
class EventParams:
    rho_abs: float = 2.0  # px per frame of velocity residual always allowed ...
    rho_rel: float = 0.10  # ... plus this fraction of the ball's image speed
    accept: float = 1.0  # a plane hypothesis needs residual / allowance <= this
    # Kinks are detected sensitively: the front wall is seen end-on, so its impacts jump the image velocity by only a
    # few px per frame. Any other kind needs a clear kink; the front wall's physics test does the filtering instead.
    strong_jump: float = 6.0  # px per frame ...
    strong_rel: float = 0.5  # ... and this fraction of the image speed
    strong_gain: float = 12.0
    front_accept: float = 0.8  # tighter residual allowance for a front-wall hypothesis that is not a strong kink
    speed_range: tuple[float, float] = (2.0, 45.0)  # m/s, implied 3D speed before the impact
    back_max_z: float = 2.4  # m, the glass wall
    back_clear: float = 1.0  # a back-glass impact needs no player within this many player heights of the ball
    racket_min_jump: float = 15.0  # px per frame: a racket changes the ball's velocity by more than this (unless audio hears it) ...
    racket_min_gain: float = 60.0  # ... and the kink must be this clear (bridged kinks have no gain and need a jump of racket_bridge_jump)
    racket_bridge_jump: float = 25.0
    reach: float = 0.7  # a racket contact needs the ball within this many player heights of the player's box
    audio_tol: float = 0.05  # s, between a kink and an audio onset after offset and flight-time correction
    min_score: float = 0.3
    pixel_sigma: float = 2.0  # px, kink position error for a clean two-sided kink
    pixel_sigma_speed: float = 0.08  # plus this fraction of the image speed (timing error of the kink)
    one_sided_gap: float = 3.0  # frames: a kink whose nearest sample on a side is farther than this is one-sided
    sigma_floor: float = 0.02  # m
    depth_sigma: float = 0.6  # m, how far from the striker's court y the ball can be at a racket contact
    audio_only_sigma_px: float = 6.0


def _fps(T: np.ndarray) -> float:
    return float(1.0 / np.median(np.diff(T)))


def _player_table(players: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if players is None:
        return {}
    return {name: g.set_index("frame") for name, g in players.groupby("player")}


def nearest_player(tables: dict[str, pd.DataFrame], frame: int, uv: np.ndarray) -> tuple[str | None, float, float, float, float]:
    """(player, distance px, box distance px, player height px, court y) of the player nearest the ball pixel.

    The distance is to the closer wrist where a wrist keypoint is usable, else to the player's box.
    """
    best = (None, np.inf, np.inf, np.nan, np.nan)
    for name, tab in tables.items():
        if frame not in tab.index:
            continue
        r = tab.loc[frame]
        if not np.isfinite(r["x1"]):
            continue
        box_d = float(np.hypot(max(r["x1"] - uv[0], 0.0, uv[0] - r["x2"]), max(r["y1"] - uv[1], 0.0, uv[1] - r["y2"])))
        d = box_d
        kp, sc = r.get("kpts"), r.get("kscore")
        if kp is not None and sc is not None and len(kp) == 34:
            k = np.asarray(kp, float).reshape(17, 2)
            s = np.asarray(sc, float)
            w = [np.hypot(*(k[j] - uv)) for j in (L_WRI, R_WRI) if s[j] > 0.2]
            if w:
                d = float(min(w))
        if d < best[1]:
            best = (name, d, box_d, float(r["y2"] - r["y1"]), float(r["sy"]) if np.isfinite(r["sy"]) else np.nan)
    return best


def plane_hypotheses(cam: Camera, uv: np.ndarray, v_before: np.ndarray, v_after: np.ndarray, spec: court.CourtSpec, fps: float, p: EventParams, strong: bool = True) -> list[dict]:
    """Every plane the kink could be a bounce on, with its fit quality: kind, world point, rho (<= accept passes), speed m/s."""
    speed_px = 0.5 * (np.hypot(*v_before) + np.hypot(*v_after))
    allow = p.rho_abs + p.rho_rel * speed_px
    out = []
    for kind in PLANES:
        X = world.locate(cam, uv, kind, spec)[0]
        if not np.isfinite(X).all() or not world.inside(X[None], kind, spec)[0]:
            continue
        res, v_m = world.bounce_residual(cam, X, kind, v_before, v_after)
        if not np.isfinite(res):
            continue
        v_ms = v_m * fps
        rho = res / allow
        if kind != "front_wall" and not strong:
            continue
        ok = rho <= (p.accept if strong or kind != "front_wall" else p.front_accept) and p.speed_range[0] <= v_ms <= p.speed_range[1]
        if kind == "back_wall":
            ok = ok and X[2] <= p.back_max_z and rho <= 0.6
        out.append(dict(kind=kind, X=X, rho=float(rho), speed=float(v_ms), ok=bool(ok)))
    # Walls seen from behind: the back glass explains any pixel at some speed, so it only counts when the front wall's
    # reading of the same kink is physically impossible.
    front = next((h for h in out if h["kind"] == "front_wall"), None)
    if front is not None and front["ok"]:
        for h in out:
            if h["kind"] == "back_wall":
                h["ok"] = False
    return out


def _audio_match(onsets: pd.DataFrame, t_video: float, kind: str, X: np.ndarray | None, cam: Camera, offset: float, tol: float) -> tuple[int, float]:
    """Index (into onsets) of the audio onset of this kind that falls on video time t_video, and its signed error in s; (-1, nan) if none."""
    if onsets.empty:
        return -1, float("nan")
    want = AUDIO_KIND.get(kind)
    sel = onsets[onsets.kind == want]
    if sel.empty:
        return -1, float("nan")
    delay = world.sound_delay(cam, X) if X is not None and np.isfinite(X).all() else 0.03
    ta = sel.t.to_numpy() - offset - delay
    d = ta - t_video
    j = int(np.argmin(np.abs(d)))
    return (int(sel.index[j]), float(d[j])) if abs(d[j]) <= tol else (-1, float("nan"))


def detect_events(kinks: pd.DataFrame, cam_of, spec: court.CourtSpec, T: np.ndarray, onsets: pd.DataFrame | None = None, offset: float = 0.0,
                  players: pd.DataFrame | None = None, track: pd.DataFrame | None = None, p: EventParams = EventParams(), audio_only: bool = True) -> pd.DataFrame:
    """Events of one clip (or window) from its kinks; `cam_of(frame)` gives the camera, `T` the frame timestamps."""
    onsets = onsets if onsets is not None else pd.DataFrame(columns=["t", "kind"])
    tables = _player_table(players)
    fps = _fps(T)
    rows: list[dict] = []
    used_audio: set[int] = set()
    frames_idx = np.arange(len(T), dtype=float)
    for k in kinks.itertuples():
        f = float(k.frame)
        fi = int(round(f))
        cam = cam_of(fi)
        uv = np.array([k.u, k.v], float)
        vb, va = np.array([k.vb_u, k.vb_v], float), np.array([k.va_u, k.va_v], float)
        t_k = float(np.interp(f, frames_idx, T))
        speed_px = 0.5 * (np.hypot(*vb) + np.hypot(*va))
        bridged = not np.isfinite(k.gain)
        strong = k.jump >= max(p.strong_jump, p.strong_rel * speed_px) and (bridged or k.gain >= p.strong_gain)
        hyps = plane_hypotheses(cam, uv, vb, va, spec, fps, p, strong)
        pl, pl_d, box_d, pl_h, pl_y = nearest_player(tables, fi, uv)
        reach_ok = pl is not None and np.isfinite(box_d) and np.isfinite(pl_h) and box_d <= p.reach * pl_h
        cands = []
        for h in hyps:
            if not h["ok"]:
                continue
            j, dt = _audio_match(onsets, t_k, h["kind"], h["X"], cam, offset, p.audio_tol)
            if h["kind"] == "back_wall" and pl is not None and np.isfinite(box_d) and np.isfinite(pl_h) and box_d <= p.back_clear * pl_h:
                continue  # the glass explains any pixel at some speed; a player within reach is the likelier cause
            if h["kind"] == "front_wall" and not strong and j < 0:
                continue  # a faint front-wall kink is only believed when the front-wall onset agrees
            score = 0.6 * (1.0 - h["rho"] / p.accept) + 0.4 + (0.5 if j >= 0 else 0.0)
            cands.append((score, h["kind"], h["X"], j, dt, h))
        # A racket contact: needs a player in reach, or a racket onset right on the kink.
        X_r = None
        if pl is not None and np.isfinite(pl_y):
            X_r = world.locate(cam, uv, "racket", spec, pl_y)[0]
        j, dt = _audio_match(onsets, t_k, "racket", X_r, cam, offset, p.audio_tol)
        hard = k.jump >= p.racket_min_jump and ((k.gain >= p.racket_min_gain) if np.isfinite(k.gain) else (k.jump >= p.racket_bridge_jump))
        if (reach_ok and hard) or j >= 0:
            closeness = 1.0 - min(box_d / (p.reach * pl_h), 1.0) if reach_ok and np.isfinite(box_d) else 0.0
            cands.append((0.25 + 0.3 * closeness + (0.5 if j >= 0 else 0.0), "racket", X_r, j, dt, None))
        if not cands:
            continue
        score, kind, X, j, dt, h = max(cands, key=lambda c: c[0])
        if score < p.min_score:
            continue
        if X is None or not np.isfinite(X).all():
            X = np.full(3, np.nan)
        gl, gr = float(k.gap_left), float(k.gap_right)
        sig_px = p.pixel_sigma + p.pixel_sigma_speed * speed_px
        one_sided = max(gl, gr) > p.one_sided_gap
        infl = (1.5 if one_sided else 1.0) * (1.5 if k.source == "bridge" else 1.0)
        sa, sb = world.plane_sigma(cam, uv, kind, spec, sig_px * infl, pl_y if kind == "racket" else None, p.depth_sigma if kind == "racket" else 0.0)
        sa, sb = (max(sa, p.sigma_floor), max(sb, p.sigma_floor)) if np.isfinite(sa) else (np.nan, np.nan)
        evidence = "both" if j >= 0 else "track"
        t_ev, f_ev = t_k, f
        if j >= 0:
            used_audio.add(j)
            if one_sided or k.source == "bridge":  # the kink time is weak; the onset's is sharper
                t_ev = t_k + dt
                f_ev = float(np.interp(t_ev, T, frames_idx))
        quality = float(np.clip(score / 1.5, 0.0, 1.0))
        gain = 0.0 if not np.isfinite(k.gain) else float(np.clip(k.gain / 150.0, 0.0, 1.0))
        conf = float(np.clip(0.15 + 0.5 * quality + 0.2 * gain + (0.15 if j >= 0 else 0.0) - (0.1 if one_sided else 0.0), 0.05, 0.99))
        rows.append(dict(t=t_ev, frame=f_ev, kind=kind, u=float(k.u), v=float(k.v), x=X[0], y=X[1], z=X[2], sigma_a=sa, sigma_b=sb, evidence=evidence,
                         confidence=conf, player=pl if kind == "racket" else None, player_dist_px=pl_d if kind == "racket" else np.nan))

    if audio_only and not onsets.empty:
        rows += _audio_only(onsets, used_audio, cam_of, spec, T, offset, track, p)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values("t").reset_index(drop=True)


def _audio_only(onsets: pd.DataFrame, used: set[int], cam_of, spec: court.CourtSpec, T: np.ndarray, offset: float, track: pd.DataFrame | None, p: EventParams) -> list[dict]:
    """Front-wall onsets no kink explained: timed from the audio, placed by the ball track nearby when there is one."""
    rows = []
    fi_all = np.arange(len(T), dtype=float)
    fw = onsets[(onsets.kind == "front_wall") & ~onsets.index.isin(list(used))]
    lo, hi = float(T[0]), float(T[-1])
    for idx, o in fw.iterrows():
        cam0 = cam_of(0)
        delay = world.sound_delay(cam0, np.array([spec.width / 2, spec.length, 1.5]))
        t_ev = float(o.t - offset - delay)
        if not lo <= t_ev <= hi:
            continue
        f_ev = float(np.interp(t_ev, T, fi_all))
        fi = int(round(f_ev))
        cam = cam_of(fi)
        u = v = np.nan
        X = np.full(3, np.nan)
        sa = sb = np.nan
        if track is not None and len(track):
            near = track[(track.frame >= fi - 3) & (track.frame <= fi + 3) & track.u.notna()]
            if len(near):
                u, v = np.interp(f_ev, near.frame.to_numpy(float), near.u.to_numpy()), np.interp(f_ev, near.frame.to_numpy(float), near.v.to_numpy())
                X = world.locate(cam, np.array([u, v]), "front_wall", spec)[0]
                if np.isfinite(X).all() and world.inside(X[None], "front_wall", spec)[0]:
                    sa, sb = world.plane_sigma(cam, np.array([u, v]), "front_wall", spec, p.audio_only_sigma_px)
                    sa, sb = max(sa, p.sigma_floor), max(sb, p.sigma_floor)
                else:
                    X = np.full(3, np.nan)
                    u = v = np.nan
        db = float(o.strength_db) if "strength_db" in o and np.isfinite(o.get("strength_db", np.nan)) else 10.0
        conf = float(np.clip(0.2 + 0.2 * np.clip(db / 20.0, 0, 1) + (0.1 if np.isfinite(X[0]) else 0.0), 0.05, 0.6))
        rows.append(dict(t=t_ev, frame=f_ev, kind="front_wall", u=u, v=v, x=X[0], y=X[1] if np.isfinite(X[1]) else spec.length, z=X[2], sigma_a=sa, sigma_b=sb,
                         evidence="audio", confidence=conf, player=None, player_dist_px=np.nan))
    return rows
