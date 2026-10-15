"""Matching two encodes of one recording by presentation time."""

from __future__ import annotations

import numpy as np

from squashvid.players.compare import match_times


def test_frames_match_by_time_not_index():
    """A re-encode with a dropped frame and slightly shifted timestamps still pairs each frame with its twin."""
    a = np.arange(10) / 30
    b = np.delete(a, 4) + 0.004
    k = match_times(a, b, tol=1 / 30)
    assert (k == np.delete(np.arange(10), 4)).all()


def test_frames_beyond_the_tolerance_have_no_match():
    a = np.arange(10) / 30
    a = np.delete(a, [5, 6, 7])  # a stall in the first encode
    b = np.array([0.0, 0.14, 6 / 30, 0.4])  # the stall's middle and a time past the end have no twin
    k = match_times(a, b, tol=1 / 30)
    assert k.tolist() == [0, 4, -1, -1]
