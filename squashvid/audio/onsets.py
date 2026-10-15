"""Impact onsets from the soundtrack, with the features needed to tell hits from everything else.

Racket and wall impacts are broadband clicks: in a few milliseconds energy
rises across the whole 2-10 kHz band at once. Shoe squeaks are tonal glides
that light up a handful of bins, and voices rise slowly with their energy below
2 kHz. Detection therefore runs on a band-limited log spectral flux aggregated
by a high quantile over bins, which a click lifts and a squeak barely moves. The
upper band edge follows what the codec kept (band_for): 10 kHz for the 64 kbps
WhatsApp copies, which carry nothing above about 12.5 kHz, and 15-16 kHz for the
iPhone originals, where clicks stand 15-25 dB above the floor at 10-18 kHz.

Timing comes from the sample-rate band envelope, not the STFT grid: each onset
is placed where its rise crosses 10% of the way from the local floor to the
peak, which is the closest observable proxy for the instant of impact.
Strength is read from a 2 ms RMS envelope instead, because the raw Hilbert
envelope of background noise alone swings by 10 dB, and it is measured against
the 30 ms just before the event rather than a longer floor: in play a hit often
lands in the tail of the previous impact.

The courts are glass boxes in a hard hall, so every impact leaves a reverberant
tail of up to a second. The early-to-late energy ratio (first 15 ms against
40-200 ms) was meant as a distance cue but measures sharpness instead: the ball
stopping dead on the front wall puts most of its energy in the first 15 ms
(el >= 4 dB) despite the 12 m distance, while racket contacts and foot plants
ring up more slowly and score lower.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import fft, signal
from scipy.ndimage import maximum_filter1d, median_filter

EPS = 1e-10
MAD_FLOOR_DB = 0.3  # flux can sit at exactly 0 dB for seconds where AAC zeroes the band; keep z finite


@dataclass(frozen=True)
class OnsetParams:
    band: tuple[float, float] = (2000.0, 10000.0)
    n_fft: int = 512  # 11.6 ms at 44.1 kHz: short enough to resolve hits 50 ms apart
    hop: int = 64  # 1.45 ms
    flux_lag: int = 3  # frames back for the reference spectrum (superflux), ~4 ms
    freq_max_bins: int = 3  # max filter across frequency, so gliding squeaks cancel themselves
    flux_quantile: float = 0.75  # aggregate across bins: a click lifts most bins, a squeak few
    thresh_window: float = 2.0  # s, window of the adaptive threshold
    thresh_k: float = 2.0  # robust z (local MADs above the local median) needed to be a candidate
    min_strength_db: float = 5.0  # 2 ms RMS peak above the level just before the event; racket contacts are often only 5-8 dB
    base_window_ms: tuple[float, float] = (45.0, 12.0)  # before the peak: the pre-event level, so a tail's own ripples do not retrigger
    peak_frac: float = 0.7  # the event peak is the first envelope sample within 3 dB of the window maximum
    min_run_ms: float = 0.3  # the onset is the end of the last sustained stretch below the 10% level
    min_gap: float = 0.04  # s, two impacts closer than this are merged
    env_smooth_ms: float = 0.5
    rms_ms: float = 2.0
    floor_window: tuple[float, float] = (0.25, 0.015)  # s before the peak: the background level (floor_dbfs)
    rise_frac: float = 0.1  # onset = where the rise crosses this fraction of floor→peak (linear amplitude)
    max_rise_ms: float = 40.0
    spec_window_ms: float = 24.0  # spectrum of the first part of the event


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase band-pass, so the filter adds no delay to onset times."""
    sos = signal.butter(order, [lo, min(hi, 0.49 * sr)], "bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y, axis=0)


def rms_envelope(x: np.ndarray, sr: int, window_ms: float = 2.0) -> np.ndarray:
    """Centred moving RMS: steadier than the Hilbert envelope, so strengths are not noise spikes."""
    n = max(1, int(round(window_ms * 1e-3 * sr)))
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    idx = np.clip(np.arange(len(x)) - n // 2, 0, max(0, len(x) - n))
    return np.sqrt(np.maximum((c[idx + n] - c[idx]) / n, 0.0))


def envelope(x: np.ndarray, sr: int, smooth_ms: float) -> np.ndarray:
    """Amplitude envelope at the sample rate: Hilbert magnitude, lightly smoothed, centred."""
    env = np.abs(signal.hilbert(x, N=fft.next_fast_len(len(x)))[: len(x)])
    n = max(1, int(round(smooth_ms * 1e-3 * sr)))
    if n > 1:
        env = np.convolve(env, np.ones(n) / n, mode="same")
    return env


def spectral_flux(y: np.ndarray, sr: int, p: OnsetParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Band-limited superflux. Returns frame times, quantile flux (dB) and the fraction of bins rising by > 6 dB."""
    f, t, Z = signal.stft(y, sr, window="hann", nperseg=p.n_fft, noverlap=p.n_fft - p.hop, boundary=None, padded=False)
    band = (f >= p.band[0]) & (f <= p.band[1])
    logS = 20 * np.log10(np.abs(Z[band]) + EPS).astype(np.float32)
    ref = maximum_filter1d(logS, size=p.freq_max_bins, axis=0)
    ref = np.concatenate([np.repeat(ref[:, :1], p.flux_lag, axis=1), ref[:, : -p.flux_lag]], axis=1)
    rise = np.maximum(logS - ref, 0.0)
    k = int(round(p.flux_quantile * (rise.shape[0] - 1)))
    flux = np.partition(rise, k, axis=0)[k]
    frac = (rise > 6.0).mean(axis=0)
    return t, flux.astype(np.float64), frac


def robust_z(odf: np.ndarray, frame_rate: float, window_s: float) -> np.ndarray:
    """(x - running median) / running MAD, so one threshold works through loud and quiet stretches."""
    w = max(3, int(window_s * frame_rate) | 1)
    stride = max(1, w // 64)  # a decimated running median is plenty for a 2 s window
    sub = odf[::stride]
    ws = max(3, (w // stride) | 1)
    med_s = median_filter(sub, size=ws, mode="nearest")
    mad_s = median_filter(np.abs(sub - med_s), size=ws, mode="nearest")
    med = np.repeat(med_s, stride)[: len(odf)]
    mad = np.repeat(mad_s, stride)[: len(odf)]
    return (odf - med) / (1.4826 * np.maximum(mad, MAD_FLOOR_DB))


def _crossing(env: np.ndarray, lo: int, pk: int, level: float, min_run: int = 1) -> int:
    """First sample after the last run of at least min_run samples below level before pk (no further back than lo).

    A reverberant impact is noise-like, so its envelope has brief deep dips
    after the true onset; requiring a sustained run below the level skips them.
    """
    below = env[lo : pk + 1] < level
    if min_run > 1:
        runs = np.convolve(below.astype(np.int32), np.ones(min_run, np.int32), mode="valid") == min_run
        ends = np.flatnonzero(runs) + min_run - 1
        return lo + (int(ends[-1]) + 1 if len(ends) else 0)
    idx = np.flatnonzero(below)
    return lo + (int(idx[-1]) + 1 if len(idx) else 0)


def _psd(x: np.ndarray, n_fft: int) -> np.ndarray:
    w = signal.windows.tukey(len(x), 0.25)
    return np.abs(np.fft.rfft(x * w, n_fft)) ** 2 / np.sum(w**2)


def _gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_lag_s: float = 6e-4, interp: int = 8) -> float:
    """Time delay of b relative to a in ms (positive: the sound reached channel a first)."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    R = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    R /= np.abs(R) + EPS
    cc = np.fft.irfft(R, n * interp)
    m = int(max_lag_s * sr * interp)
    cc = np.concatenate([cc[-m:], cc[: m + 1]])
    return 1e3 * (np.argmax(cc) - m) / (sr * interp)


class _Spectra:
    """Event and background spectra around an onset, and the summary numbers drawn from them."""

    def __init__(self, sr: int, window_ms: float, top_hz: float = 11000.0):
        self.n = int(window_ms * 1e-3 * sr)
        self.n_fft = 1 << int(np.ceil(np.log2(self.n)) + 1)
        f = np.fft.rfftfreq(self.n_fft, 1 / sr)
        self.f = f
        self.low, self.mid, self.high = (f >= 200) & (f < 1500), (f >= 1500) & (f < 4000), (f >= 4000) & (f < 10000)
        self.all, self.tonal = (f >= 200) & (f < max(top_hz, 11000)), (f >= 1000) & (f < 10000)
        self.air = (f >= 10000) & (f < top_hz) if top_hz > 11000 else None  # only where the codec kept it

    def __call__(self, y: np.ndarray, i_on: int, sr: int) -> dict | None:
        s0 = max(0, i_on - int(0.001 * sr))
        seg = y[s0 : s0 + self.n]
        bg_end = max(0, i_on - int(0.02 * sr))
        bg_start = max(0, bg_end - 4 * self.n)
        if len(seg) < self.n or bg_end - bg_start < self.n:
            return None
        P = _psd(seg, self.n_fft)
        Pb = np.mean([_psd(y[s : s + self.n], self.n_fft) for s in range(bg_start, bg_end - self.n + 1, self.n // 2)], axis=0)
        X = np.maximum(P - Pb, 0.0)  # what the event added to the background
        tot = X[self.all].sum() + EPS
        Pt = P[self.tonal]
        sm = np.convolve(10 * np.log10(Pt + EPS), np.ones(5) / 5, mode="same")
        return dict(
            centroid_hz=float((self.f[self.all] * X[self.all]).sum() / tot),
            low_ratio=float(X[self.low].sum() / tot),
            mid_ratio=float(X[self.mid].sum() / tot),
            high_ratio=float(X[self.high].sum() / tot),
            air_ratio=float(X[self.air].sum() / tot) if self.air is not None else float("nan"),
            flatness=float(np.exp(np.mean(np.log(Pt + EPS))) / (np.mean(Pt) + EPS)),
            peakiness_db=float(np.max(sm) - np.median(sm)),
            spec_snr_db=float(10 * np.log10((P[self.all].sum() + EPS) / (Pb[self.all].sum() + EPS))),
        )


def onset_features(y: np.ndarray, sr: int, hb: np.ndarray, cands_s: np.ndarray, p: OnsetParams, stereo: np.ndarray | None = None) -> pd.DataFrame:
    """Measure each candidate: timing on the Hilbert envelope, level and decay on the RMS envelope, spectra on the raw signal."""
    env = envelope(hb, sr, p.env_smooth_ms)
    rms_db = 20 * np.log10(rms_envelope(hb, sr, p.rms_ms) + EPS)
    n = len(y)
    ms = lambda v: int(v * 1e-3 * sr)  # noqa: E731
    spectra = _Spectra(sr, p.spec_window_ms, top_hz=p.band[1] * 1.05)
    stereo_band = bandpass(stereo, sr, *p.band) if stereo is not None else None
    rows = []
    for t in cands_s:
        c = int(t * sr)
        a, b = max(0, c - ms(15)), min(n, c + ms(30))
        if b - a < 8:
            continue
        w = env[a:b]
        pk = a + int(np.argmax(w >= p.peak_frac * w.max()))  # first near-maximum, not a later spike in the tail
        fa, fb = max(0, pk - int(p.floor_window[0] * sr)), max(1, pk - int(p.floor_window[1] * sr))
        floor_db = float(np.median(rms_db[fa:fb]))
        peak_db = float(rms_db[max(0, pk - ms(3)) : pk + ms(3) + 1].max())
        # the level just before the event: in a reverberant tail it sits well below the 250 ms median
        ba, bb = max(0, pk - ms(p.base_window_ms[0])), max(1, pk - ms(p.base_window_ms[1]))
        base_db = float(np.median(rms_db[ba:bb]))

        # rise on the Hilbert envelope, against the same pre-event level on that scale
        lin_floor = float(np.median(env[ba:bb]))
        lo = max(0, pk - ms(p.max_rise_ms))
        span = env[pk] - lin_floor
        run = max(1, ms(p.min_run_ms))
        i_on = _crossing(env, lo, pk, lin_floor + p.rise_frac * span, run)
        rise_ms = 1e3 * (_crossing(env, lo, pk, lin_floor + 0.9 * span) - i_on) / sr

        tail = rms_db[pk : min(n, pk + ms(500))]
        d10 = np.flatnonzero(tail < peak_db - 10)
        dfl = np.flatnonzero(tail < floor_db + 6)
        decay10_ms = 1e3 * (d10[0] if len(d10) else len(tail)) / sr
        dur_ms = 1e3 * ((dfl[0] if len(dfl) else len(tail)) + pk - i_on) / sr

        e = env[i_on:] ** 2  # direct sound against reverberation, per unit time
        if len(e) < ms(200):
            continue
        el_ratio_db = float(10 * np.log10((e[: ms(15)].mean() + EPS) / (e[ms(40) : ms(200)].mean() + EPS)))

        spec = spectra(y, i_on, sr)
        if spec is None:
            continue
        row = dict(t=i_on / sr, t_peak=pk / sr, t_cand=float(t), strength_db=peak_db - base_db, peak_dbfs=peak_db, floor_dbfs=floor_db,
                   rise_ms=rise_ms, decay10_ms=decay10_ms, dur_ms=dur_ms, el_ratio_db=el_ratio_db, **spec)
        if stereo_band is not None:
            w0, w1 = max(0, i_on - ms(2)), min(n, i_on + ms(12))
            L, R = stereo_band[w0:w1, 0], stereo_band[w0:w1, 1]
            row["itd_ms"] = _gcc_phat(L, R, sr)
            row["ild_db"] = float(10 * np.log10((np.sum(L**2) + EPS) / (np.sum(R**2) + EPS)))
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_close(df: pd.DataFrame, min_gap: float) -> pd.DataFrame:
    """Two flux peaks can refine onto the same rise; of any pair closer than min_gap keep the sharper onset.

    The later of two duplicates was measured from a window in the tail, so its
    rise and flux are wrong; the higher flux z picks the candidate that sat on
    the rise itself.
    """
    keep = np.ones(len(df), bool)
    t, s = df["t"].to_numpy(), df["flux_z"].to_numpy()
    last = 0
    for i in range(1, len(df)):
        if t[i] - t[last] < min_gap:
            loser = last if s[i] > s[last] else i
            keep[loser] = False
            last = i if loser == last else last
        else:
            last = i
    return df[keep].reset_index(drop=True)


def usable_top_hz(y: np.ndarray, sr: int, seconds: float = 60.0) -> float:
    """Highest frequency the codec kept: where the long-term spectrum falls 45 dB below its 4-6 kHz level.

    The WhatsApp copies stop near 12.5 kHz, the iPhone originals near 19 kHz.
    """
    seg = np.asarray(y[: int(seconds * sr)], dtype=np.float64)
    f, P = signal.welch(seg, sr, nperseg=4096)
    ref = np.median(P[(f >= 4000) & (f < 6000)])
    ok = np.flatnonzero(P > ref * 10 ** (-45 / 10))
    return float(f[ok[-1]]) if len(ok) else 0.5 * sr


def band_for(y: np.ndarray, sr: int, p: OnsetParams = OnsetParams(), max_hi: float = 16000.0) -> OnsetParams:
    """Detection band widened to what the recording carries: clicks stand furthest above the floor at high frequencies."""
    from dataclasses import replace

    return replace(p, band=(p.band[0], float(np.clip(0.8 * usable_top_hz(y, sr), p.band[1], max_hi))))


def detect(y: np.ndarray, sr: int, p: OnsetParams = OnsetParams(), stereo: np.ndarray | None = None) -> pd.DataFrame:
    """All impact-like onsets in a mono signal, one row per onset, sorted by time.

    stereo, if given (n × 2, same timing), adds the inter-channel delay and level
    difference, a direction cue that may help flag hits on neighbouring courts.
    """
    y = np.asarray(y, dtype=np.float64)
    t_frames, flux, frac = spectral_flux(y, sr, p)
    z = robust_z(flux, sr / p.hop, p.thresh_window)
    peaks, _ = signal.find_peaks(z, height=p.thresh_k, distance=max(1, int(p.min_gap * sr / p.hop)))
    df = onset_features(y, sr, bandpass(y, sr, *p.band), t_frames[peaks], p, stereo=stereo)
    if df.empty:
        return df
    idx = np.clip(np.searchsorted(t_frames, df["t_cand"].to_numpy()), 0, len(flux) - 1)
    df["flux_db"] = flux[idx]
    df["flux_z"] = z[idx]
    df["broadband"] = maximum_filter1d(frac, 5)[idx]
    df = df.drop(columns="t_cand")
    df = df[df["strength_db"] >= p.min_strength_db].sort_values("t").reset_index(drop=True)
    return _merge_close(df, p.min_gap)
