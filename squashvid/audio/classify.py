"""First-pass typing of onsets: racket, walls, floor, squeak, voice, other courts.

A decision list over the onset features, not a trained model: there are only a
few dozen labelled events. What the labelled rallies showed (iPhone AAC track,
camera on the walkway behind the back glass):

- Front-wall impacts are the loudest court-5 events. The ball stops dead on
  plaster, so the sound starts within a millisecond and most of its energy is
  in the first 15 ms (el_ratio >= 4 dB). The wall is straight ahead, so the two
  channels hear it about equally (ILD near 0). It comes 0.1-0.6 s after the stroke.
- Racket contacts are quieter and softer-edged: strings and frame take a few
  milliseconds to ring up (median rise 2.9 ms against 0.9 ms for other onsets in
  the same rallies, the best single separator), and their ILD sits around -4 dB.
  Their strength is no different from the background clatter.
- Impacts louder on the left channel (ILD > +1 dB) with a sharp front come from
  the neighbouring court: five times as frequent in pauses as in rallies.
- Plenty of racket-like impacts also occur while both players stand still
  (other courts, handling). Features alone cannot tell them apart, so a racket
  candidate is promoted only when a front-wall impact follows it at a plausible
  flight time, and of several candidates the slowest-rising wins. Hits whose
  wall impact was missed stay "other". On two labelled clips (64 contacts) this
  types 13-16% of the visible racket contacts at a precision of 0.44-0.55; the
  false ones are mostly foot plants. Front-wall typing is better: 15 of 19
  front_wall onsets in labelled rallies lie within 0.1 s of a visual front-wall
  label.

Shoe squeaks are tonal (a narrow peak, low spectral flatness, few bins rising);
voices rise slowly with their energy below 1 kHz; floor bounces are dull thuds
with little energy above 4 kHz. Once rallies are known, impacts outside all of
them become "other_court" (mark_other_court).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KINDS = ("racket", "front_wall", "side_wall", "floor", "squeak", "voice", "other_court", "other")

# tonal events
SQUEAK_PEAKINESS_DB = 32.0
SQUEAK_FLATNESS = 0.03
SQUEAK_BROADBAND = 0.25
VOICE_CENTROID_HZ = 900.0
VOICE_RISE_MS = 6.0
# impacts
IMPULSE_RISE_MS = 12.0
HIT_MIN_STRENGTH_DB = 5.0
RACKET_MIN_RISE_MS = 2.0
WALL_MIN_STRENGTH_DB = 9.0
WALL_EL_RATIO_DB = 4.0  # front-wall impacts: energy concentrated in the first 15 ms
WALL_RISE_MS = 2.0
WALL_ILD_DB = (-3.5, 1.0)  # straight ahead
OTHER_COURT_ILD_DB = 1.0  # louder on the left channel: the neighbouring court
OTHER_COURT_EL_RATIO_DB = 3.0
FLOOR_HIGH_RATIO = 0.03
FLOOR_CENTROID_HZ = 1000.0
FLIGHT_S = (0.1, 0.6)  # stroke to front wall
# rally rhythm: strong sharp impacts that are not from the left-hand court
RALLY_MIN_STRENGTH_DB = 14.0
RALLY_MIN_EL_RATIO_DB = 3.0
SIDE_WALL_GAP_S = 0.1  # a second sharp impact this soon after a wall impact is a side/back wall


def _base_kind(r) -> str:
    if r.peakiness_db > SQUEAK_PEAKINESS_DB and r.flatness < SQUEAK_FLATNESS and r.broadband < SQUEAK_BROADBAND:
        return "squeak"
    if r.centroid_hz < VOICE_CENTROID_HZ and r.rise_ms > VOICE_RISE_MS and r.broadband < SQUEAK_BROADBAND:
        return "voice"
    if r.rise_ms > IMPULSE_RISE_MS or r.strength_db < HIT_MIN_STRENGTH_DB:
        return "other"
    ild = getattr(r, "ild_db", 0.0)
    ild = 0.0 if not np.isfinite(ild) else ild  # mono input: no direction cue
    if ild > OTHER_COURT_ILD_DB and r.el_ratio_db >= OTHER_COURT_EL_RATIO_DB:
        return "other_court"
    if (r.el_ratio_db >= WALL_EL_RATIO_DB and r.rise_ms <= WALL_RISE_MS and WALL_ILD_DB[0] < ild <= WALL_ILD_DB[1]
            and r.strength_db >= WALL_MIN_STRENGTH_DB):
        return "wall"
    if r.high_ratio < FLOOR_HIGH_RATIO and r.centroid_hz < FLOOR_CENTROID_HZ and r.strength_db >= WALL_MIN_STRENGTH_DB:
        return "floor"
    return "near" if r.rise_ms >= RACKET_MIN_RISE_MS else "other"


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add `kind` (one of KINDS) to an onsets frame from onsets.detect."""
    df = df.copy()
    if df.empty:
        df["kind"] = pd.Series(dtype=str)
        return df
    kind = np.array([_base_kind(r) for r in df.itertuples()], dtype=object)
    t = df["t"].to_numpy()
    rise = df["rise_ms"].to_numpy()
    walls = np.flatnonzero(kind == "wall")
    near = np.flatnonzero(kind == "near")
    racket = set()
    for w in walls:
        cand = near[(t[near] >= t[w] - FLIGHT_S[1]) & (t[near] <= t[w] - FLIGHT_S[0])]
        if len(cand):
            racket.add(int(cand[np.argmax(rise[cand])]))
    last_wall = -np.inf
    for i in range(len(kind)):
        if kind[i] == "wall":
            kind[i] = "side_wall" if t[i] - last_wall < SIDE_WALL_GAP_S else "front_wall"
            last_wall = t[i]
        elif kind[i] == "near":
            kind[i] = "racket" if i in racket else "other"
    df["kind"] = kind
    return df


def rally_impacts(df: pd.DataFrame) -> np.ndarray:
    """Mask of the onsets that carry a rally's rhythm, for rallies.segment.

    Front-wall impacts are the most reliably heard event of each shot, so the
    chain is built from strong, sharp impacts whatever their tentative type,
    minus the left-hand court (ILD > +1 dB). On IMG_4091 this matched 14 of 24
    visually counted rallies with 4 spurious (mean IoU 0.51).
    """
    ild = df["ild_db"].fillna(0.0) if "ild_db" in df else 0.0
    return ((df["rise_ms"] < IMPULSE_RISE_MS) & (df["strength_db"] >= RALLY_MIN_STRENGTH_DB) & (df["el_ratio_db"] >= RALLY_MIN_EL_RATIO_DB)
            & (ild <= OTHER_COURT_ILD_DB) & ~df["kind"].isin(["squeak", "voice"])).to_numpy()


def mark_other_court(df: pd.DataFrame) -> pd.DataFrame:
    """After rally segmentation: strong impacts outside every rally are probably from neighbouring courts."""
    df = df.copy()
    out = (df["rally"] < 0) & df["kind"].isin(["racket", "front_wall", "side_wall"])
    df.loc[out, "kind"] = "other_court"
    return df
