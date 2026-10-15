"""Audio hit-timing stream on synthetic signals: clicks in noise, a shoe squeak, a voice, rally chains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from squashvid.audio import classify, evaluate, onsets, rallies

SR = 44100


def _noise(seconds: float, dbfs: float = -50.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(seconds * SR)) * 10 ** (dbfs / 20)


def _click(y: np.ndarray, t: float, amp: float, decay_s: float = 0.15, seed: int = 1, attack_s: float = 3e-4) -> None:
    """Broadband impact: a rise over attack_s into an exponentially decaying noise tail (a reverberant room)."""
    rng = np.random.default_rng(seed)
    i = int(round(t * SR))
    n = min(len(y) - i, int(1.0 * SR))
    tt = np.arange(n) / SR
    y[i : i + n] += amp * rng.standard_normal(n) * np.exp(-tt / decay_s) * np.minimum(1.0, tt / attack_s)


def _squeak(y: np.ndarray, t: float, dur: float, amp: float) -> None:
    """Shoe squeak: a tonal glide 4.6 -> 5.4 kHz with a slow 25 ms attack."""
    i = int(t * SR)
    tt = np.arange(int(dur * SR)) / SR
    f = 4600 + 800 * tt / dur
    y[i : i + len(tt)] += amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1.0, tt / 0.025) * np.minimum(1.0, (dur - tt) / 0.02)


def _voice(y: np.ndarray, t: float, dur: float, amp: float) -> None:
    """Shout: harmonics of 220 Hz, energy below 2 kHz, 60 ms attack."""
    i = int(t * SR)
    tt = np.arange(int(dur * SR)) / SR
    s = sum(np.sin(2 * np.pi * 220 * k * tt) / k for k in range(1, 9))
    y[i : i + len(tt)] += amp * s * np.minimum(1.0, tt / 0.06) * np.minimum(1.0, (dur - tt) / 0.05)


@pytest.fixture(scope="module")
def clicks():
    y = _noise(12.0)
    true = np.array([0.73, 1.61, 2.05, 3.30, 4.12, 4.90, 6.02, 6.47, 7.85, 9.10, 10.31])
    amps = np.array([0.05, 0.2, 0.05, 0.1, 0.3, 0.03, 0.08, 0.15, 0.04, 0.25, 0.06])
    for k, (t, a) in enumerate(zip(true, amps)):
        _click(y, t, a, seed=k)
    return y, true


def test_clicks_found_with_sub_frame_timing(clicks):
    y, true = clicks
    df = onsets.detect(y, SR)
    m = evaluate.match(df.t.to_numpy(), true, tol=0.002)
    assert len(m) == len(true)
    errs = np.array([df.t.iloc[i] - true[j] for i, j in m])
    assert np.max(np.abs(errs)) < 1e-3  # well inside one video frame (33 ms) and one STFT hop (1.5 ms)
    assert len(df) <= len(true) + 1  # reverberant tails do not retrigger


def test_click_features_measure_strength_and_decay(clicks):
    y, true = clicks
    df = onsets.detect(y, SR)
    loud, quiet = df.iloc[np.argmin(np.abs(df.t - 4.12))], df.iloc[np.argmin(np.abs(df.t - 4.90))]
    assert loud.strength_db > quiet.strength_db + 15
    assert loud.rise_ms < 3 and loud.broadband > 0.3
    assert 10 < loud.decay10_ms < 600


def test_squeak_and_voice_are_not_impacts():
    y = _noise(6.0)
    _squeak(y, 1.0, 0.25, 0.05)
    _voice(y, 3.0, 0.4, 0.05)
    _click(y, 5.0, 0.1)
    df = classify.classify(onsets.detect(y, SR))
    near = lambda t: df[(df.t > t - 0.03) & (df.t < t + 0.45)]  # noqa: E731
    assert set(near(1.0).kind) <= {"squeak", "other"}
    assert set(near(3.0).kind) <= {"voice", "other"}
    assert near(5.0).kind.iloc[0] in {"front_wall", "racket", "other"}  # an impact, typed by its context


def test_stroke_then_wall_types_racket_and_front_wall():
    y = _noise(8.0)
    for k, t in enumerate([1.0, 3.0, 5.0]):
        _click(y, t, 0.03, decay_s=0.2, attack_s=0.005, seed=k)  # racket: strings ring up, long room tail
        _click(y, t + 0.3, 0.1, decay_s=0.008, seed=10 + k)  # front wall: dead stop, energy up front
    _click(y, 7.0, 0.03, decay_s=0.2, attack_s=0.005, seed=20)  # racket-like with no wall after it
    df = classify.classify(onsets.detect(y, SR))
    kind_at = lambda t: df.iloc[np.argmin(np.abs(df.t - t))].kind  # noqa: E731
    assert [kind_at(t) for t in (1.0, 3.0, 5.0)] == ["racket"] * 3
    assert [kind_at(t + 0.3) for t in (1.0, 3.0, 5.0)] == ["front_wall"] * 3
    assert kind_at(7.0) == "other"


def test_segment_chains_and_rejects_bounce_metronome():
    rally1 = 10 + np.cumsum([0, 1.1, 0.9, 1.4, 1.2, 0.8, 1.6])
    bounces = 25 + 0.55 * np.arange(8)  # server bouncing the ball
    rally2 = 40 + np.cumsum([0, 1.3, 1.0, 2.1, 1.1])
    lone = np.array([60.0, 75.0])
    rl = rallies.segment(np.concatenate([rally1, bounces, rally2, lone]))
    assert len(rl) == 2
    assert rl.first_hit.tolist() == [rally1[0], rally2[0]]
    assert rl.n_hits.tolist() == [len(rally1), len(rally2)]
    on = pd.DataFrame({"t": [10.5, 30.0, 41.0]})
    assert rallies.assign(on, rl).tolist() == [0, -1, 1]


def test_match_is_one_to_one():
    pred = np.array([1.000, 1.010, 2.030, 5.0])
    true = np.array([1.005, 2.000, 3.000])
    m = evaluate.match(pred, true, tol=0.034)
    assert m == [(0, 0), (2, 1)]
    pr = evaluate.precision_recall(pred, true, tol=0.034)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (2, 2, 1)


def test_offset_estimate_recovers_shift():
    rng = np.random.default_rng(3)
    true = np.sort(rng.uniform(0, 100, 60))
    pred = np.sort(np.concatenate([true + 0.025 + rng.normal(0, 0.004, 60), rng.uniform(0, 100, 20)]))
    est = evaluate.estimate_offset(pred, true, window=0.1)
    assert abs(est["median_ms"] - 25) < 3 and est["n"] >= 55


def test_travel_time_to_camera():
    assert evaluate.travel_s(3.075, 9.94) == pytest.approx(11.94 / 343, abs=2e-4)  # front wall centre, ~35 ms
    assert evaluate.travel_s(3.075, 0.0) < evaluate.travel_s(0.0, 5.0)


def test_band_follows_codec_bandwidth():
    rng = np.random.default_rng(5)
    full = rng.standard_normal(10 * 48000) * 0.01
    F = np.fft.rfft(full)
    F[np.fft.rfftfreq(len(full), 1 / 48000) > 12500] = 0  # a 64 kbps AAC copy cuts off like a brick wall
    cut = np.fft.irfft(F, len(full))
    assert onsets.band_for(full, 48000).band[1] == 16000.0
    assert onsets.band_for(cut, 48000).band[1] == pytest.approx(10000.0, abs=500)


def test_evaluate_clip_recovers_offset_and_scores():
    rng = np.random.default_rng(7)
    hits = np.sort(rng.uniform(10, 60, 30))
    lab = pd.DataFrame({"rally": "A", "event": "racket", "t": hits, "confidence": "high", "travel_s": 0.02})
    audio = hits + 0.02 - 0.015 + rng.normal(0, 0.003, len(hits))  # travel 20 ms, audio leads video by 15 ms
    junk = rng.uniform(10, 60, 10)
    on = pd.DataFrame({"t": np.concatenate([audio, junk]), "kind": ["racket"] * len(hits) + ["squeak"] * len(junk)}).sort_values("t")
    rep = evaluate.evaluate_clip(on, pd.DataFrame(), lab)
    assert abs(rep["offset_used_ms"] + 15) < 3
    s = rep["scores"]["racket_class/all/1f"]
    assert s["recall"] == 1.0 and s["precision"] == 1.0
    assert rep["scores"]["any_onset_vs_impacts/all/1f"]["fp"] <= len(junk)
