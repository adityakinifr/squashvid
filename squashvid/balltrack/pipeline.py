"""Candidates -> linked track -> events, for a labelled window or a whole clip."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from squashvid.balltrack import data, events, kinks
from squashvid.balltrack import link as linker
from squashvid.ball.clips import work_dir
from squashvid.ball.regions import venue
from squashvid.players.geometry import CameraTrack


@dataclass(frozen=True)
class Params:
    link: linker.LinkParams = field(default_factory=linker.LinkParams)
    fill: linker.FillParams = field(default_factory=linker.FillParams)
    kink: kinks.KinkParams = field(default_factory=kinks.KinkParams)
    events: events.EventParams = field(default_factory=events.EventParams)


@lru_cache(maxsize=None)
def camera_track(stem: str) -> CameraTrack:
    return CameraTrack.load(venue(stem), work_dir(stem) / "camera.parquet")


def build_track(cands: pd.DataFrame, T: np.ndarray, n_frames: int, params: Params = Params()) -> linker.Track:
    linked = linker.link(cands, T, params.link)
    return linker.fill_gaps(linked, n_frames, params.fill, params.kink)


def build_events(stem: str, track: linker.Track, T: np.ndarray, params: Params = Params(), lo: int | None = None, hi: int | None = None,
                 audio: bool = True, players: bool = True, audio_only: bool = True) -> pd.DataFrame:
    """Events of the frames [lo, hi] (all by default)."""
    spec = venue(stem).spec
    ct = camera_track(stem)
    kk = track.kinks
    fr = track.frames
    if lo is not None:
        kk = kk[(kk.frame >= lo) & (kk.frame <= hi)]
        fr = fr[(fr.frame >= lo) & (fr.frame <= hi)]
    onsets, offset = None, 0.0
    if audio:
        onsets = data.audio_onsets(stem)
        offset = data.audio_offset(stem)
        if lo is not None:
            onsets = onsets[(onsets.t - offset >= T[lo] - 0.2) & (onsets.t - offset <= T[hi] + 0.2)]
    pl = None
    if players:
        pl = data.players(stem)
        if lo is not None:
            pl = pl[(pl.frame >= lo - 2) & (pl.frame <= hi + 2)]
    return events.detect_events(kk, ct, spec, T, onsets, offset, pl, fr, params.events, audio_only=audio_only)
