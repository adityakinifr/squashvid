"""What each player looks like, for painting the 3D figures: the colours of their shirt and shorts, read from the
proxy video at the pose keypoints.

Frames are taken spread over the clip where the player was detected (not interpolated, not truncated) with the
torso keypoints sure and the other player's box not overlapping. In each frame:

- shirt: the torso quadrilateral (shoulders and hips) shrunk toward its centre, so arms and background stay out;
- shorts: a strip down the middle of each thigh, from 35% to 75% of the way from the hips to the knees (a loose
  shirt hangs over the top of them);

Skin and hair are not read. They are small and in shade, so on the test clips they came out grey (forearms about
[66, 62, 60] where white shoes read [130, 126, 118] beside walls at [202, 200, 192]); every figure gets one neutral
skin tone and the model's own dark hair, neither a measurement of the player.

The camera mostly sees the players' backs, away from the ceiling lights, so a part's pixels are its colour in shade.
The shirt's colour is the median of its brighter pixels (the top LIT_SHARE by brightness: the side the light
reaches); the shorts are too narrow for that (their brightest pixels are the wall behind), so they take the median.
Each is then scaled per channel in linear light so that the frames' white (the 95th percentile of each frame: the
walls) becomes WHITE, which corrects exposure and white balance together. Colours are sRGB 0-255. A part seen in
fewer than MIN_FRAMES frames is left out: the viewer then paints the shirt in the player's UI colour and leaves the
model's own shorts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_FRAMES = 48
MIN_FRAMES = 6
KP_MIN = 0.5
SHRINK = 0.3  # share of the way each corner moves toward the region's centre
LIT_SHARE = 0.4
WHITE = 235.0  # what the walls' white becomes (a white wall's albedo, about 0.83)
PARTS = ("shirt", "shorts")
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN = 5, 6, 9, 10, 11, 12, 13, 14


def _shrunk(pts: np.ndarray, k: float = SHRINK) -> np.ndarray:
    c = pts.mean(axis=0)
    return pts + k * (c - pts)


def _poly_mask(shape: tuple[int, int], poly: np.ndarray) -> np.ndarray:
    import cv2

    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def regions(k: np.ndarray, s: np.ndarray, shape: tuple[int, int]) -> dict[str, list[np.ndarray]]:
    """Pixel masks per part for one player in one frame. k: (17, 2) COCO keypoints in pixels, s: (17,) scores."""
    ok = lambda *i: all(s[j] >= KP_MIN for j in i)  # noqa: E731
    out: dict[str, list[np.ndarray]] = {p: [] for p in PARTS}
    if not ok(L_SH, R_SH, L_HIP, R_HIP):
        return out
    out["shirt"].append(_poly_mask(shape, _shrunk(k[[L_SH, R_SH, R_HIP, L_HIP]])))
    width = float(np.linalg.norm(k[L_SH] - k[R_SH]))
    if ok(L_KN, R_KN):
        for h, kn in ((L_HIP, L_KN), (R_HIP, R_KN)):  # one strip down each thigh, never the gap between the legs
            a, b = k[h] + 0.35 * (k[kn] - k[h]), k[h] + 0.75 * (k[kn] - k[h])
            d = b - a
            n = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6) * 0.1 * width
            out["shorts"].append(_poly_mask(shape, np.array([a - n, a + n, b + n, b - n])))
    return out


def _pick_frames(pl: pd.DataFrame, player: str, n: int) -> list[int]:
    me = pl[(pl.player == player) & pl.detected & ~pl.interpolated & ~pl.truncated & pl.kpts.notna()]
    if not len(me):
        return []
    sc = np.stack(me.kscore.to_numpy())
    me = me[(sc[:, [L_SH, R_SH, L_HIP, R_HIP]] >= KP_MIN).all(axis=1)]
    other = pl[pl.player != player].set_index("frame")
    keep = []
    for r in me.itertuples():
        if r.frame in other.index:
            o = other.loc[r.frame]
            o = o.iloc[0] if isinstance(o, pd.DataFrame) else o
            if np.isfinite(o.x1) and not (o.x2 < r.x1 or o.x1 > r.x2 or o.y2 < r.y1 or o.y1 > r.y2):
                continue  # boxes overlap: the other player may cover this one
        keep.append(int(r.frame))
    if len(keep) <= n:
        return keep
    return [keep[i] for i in np.linspace(0, len(keep) - 1, n).round().astype(int)]


def _srgb_to_lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return 255 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _lit(px: np.ndarray) -> np.ndarray:
    """Median colour of the brightest LIT_SHARE of the pixels."""
    lum = px.astype(float) @ [0.2126, 0.7152, 0.0722]
    return np.median(px[lum >= np.quantile(lum, 1 - LIT_SHARE)], axis=0)


def estimate(frames: dict[int, np.ndarray], pl: pd.DataFrame, player: str) -> dict[str, list[int]]:
    """Part colours of one player from the given frames (index -> BGR image)."""
    pix: dict[str, list[np.ndarray]] = {p: [] for p in PARTS}
    seen = {p: 0 for p in PARTS}
    whites = []
    rows = pl[pl.player == player].set_index("frame")
    for f, img in frames.items():
        if f not in rows.index:
            continue
        r = rows.loc[f]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        k, s = np.asarray(r.kpts, float).reshape(17, 2), np.asarray(r.kscore, float)
        rgb = img[..., ::-1]
        whites.append(np.percentile(rgb.reshape(-1, 3), 95, axis=0))
        for part, masks in regions(k, s, img.shape[:2]).items():
            got = [rgb[m] for m in masks if m.any()]
            if got:
                pix[part].append(np.concatenate(got))
                seen[part] += 1
    found: dict[str, np.ndarray] = {}
    for part in PARTS:
        if seen[part] >= MIN_FRAMES:  # only the torso is wide enough that its brightest pixels are still the shirt
            px = np.concatenate(pix[part])
            found[part] = _lit(px) if part == "shirt" else np.median(px, axis=0)
    if not found:
        return {}
    gain = _srgb_to_lin(WHITE) / np.maximum(_srgb_to_lin(np.median(whites, axis=0)), 1e-3)
    return {part: _lin_to_srgb(_srgb_to_lin(c) * gain).round().astype(int).tolist() for part, c in found.items()}


def read_frames(video: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    import cv2

    cap, out = cv2.VideoCapture(str(video)), {}
    try:
        for f in sorted(set(wanted)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                out[f] = img
    finally:
        cap.release()
    return out


def appearance(work: Path) -> dict[str, dict[str, list[int]]]:
    """{player: {part: [r, g, b]}} for the clip in `work`; empty without the proxy or the player keypoints."""
    pp, video = work / "players.parquet", work / "proxy.mp4"
    if not pp.exists() or not video.exists():
        return {}
    pl = pd.read_parquet(pp, columns=["frame", "player", "detected", "interpolated", "truncated", "x1", "y1", "x2", "y2", "kpts", "kscore"])
    picks = {p: _pick_frames(pl, p, N_FRAMES) for p in ("A", "B")}
    frames = read_frames(video, picks["A"] + picks["B"])
    return {p: est for p in ("A", "B") if (est := estimate({f: frames[f] for f in picks[p] if f in frames}, pl, p))}


MIN_VOTES = 10
HAND_MARGIN_PX = 15.0


def racket_hand(work: Path) -> dict[str, str]:
    """Which hand each player holds the racket in: at each located racket impact, the hitter's wrist nearer the ball
    (by more than HAND_MARGIN_PX) gets a vote. "right" without MIN_VOTES votes, as most players are right-handed.
    On the three clips both players vote right, 59 to 18 and 67 to 31."""
    ev, pp = work / "ball_events.parquet", work / "players.parquet"
    out = {"A": "right", "B": "right"}
    if not ev.exists() or not pp.exists():
        return out
    e = pd.read_parquet(ev)
    if not {"kind", "u", "v", "player"} <= set(e.columns):
        return out
    e = e[(e.kind == "racket") & e.u.notna() & e.v.notna() & e.player.isin(list(out))]
    pl = pd.read_parquet(pp, columns=["frame", "player", "kpts", "kscore"]).dropna(subset=["kpts"])
    rows = {(r.player, int(r.frame)): r for r in pl.itertuples()}
    votes = {p: [0, 0] for p in out}  # right, left
    for r in e.itertuples():
        hit = rows.get((r.player, int(round(r.frame))))
        if hit is None:
            continue
        k, sc = np.asarray(hit.kpts, float).reshape(17, 2), np.asarray(hit.kscore, float)
        if sc[L_WR] < 0.3 or sc[R_WR] < 0.3:
            continue
        dl, dr = (float(np.hypot(*(k[w] - (r.u, r.v)))) for w in (L_WR, R_WR))
        if abs(dl - dr) > HAND_MARGIN_PX:
            votes[r.player][0 if dr < dl else 1] += 1
    for p, (right, left) in votes.items():
        if right + left >= MIN_VOTES and left > right:
            out[p] = "left"
    return out


__all__ = ["appearance", "estimate", "racket_hand", "regions"]
