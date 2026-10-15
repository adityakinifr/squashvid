"""Review plot: band envelope with onsets by tentative class, rallies and (optionally) visual labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from squashvid.audio.onsets import OnsetParams, bandpass, rms_envelope  # noqa: E402

COLOURS = {"racket": "tab:red", "front_wall": "tab:blue", "side_wall": "tab:cyan", "floor": "tab:brown",
           "squeak": "tab:purple", "voice": "tab:pink", "other_court": "tab:gray", "other": "tab:olive"}


def busiest_window(rallies: pd.DataFrame, duration: float, length: float = 30.0) -> float:
    """Start of the length-second window holding the most rally time."""
    if rallies.empty:
        return 0.0
    best, best_t = -1.0, 0.0
    for t0 in np.arange(0.0, max(0.0, duration - length) + 1e-9, 1.0):
        cover = sum(max(0.0, min(r.end_t, t0 + length) - max(r.start_t, t0)) for r in rallies.itertuples())
        if cover > best:
            best, best_t = cover, t0
    return float(best_t)


def plot_window(y: np.ndarray, sr: int, onsets: pd.DataFrame, rallies: pd.DataFrame, t0: float, out: str | Path, length: float = 30.0,
                labels: pd.DataFrame | None = None, title: str = "", p: OnsetParams = OnsetParams()) -> Path:
    """Two stacked panels of length/2 s each: 2 ms RMS dB of the detection band, onset markers coloured by class, rallies shaded."""
    a, b = int(max(0.0, t0 - 0.5) * sr), int(min(len(y) / sr, t0 + length + 0.5) * sr)
    env = 20 * np.log10(rms_envelope(bandpass(y[a:b], sr, *p.band), sr, p.rms_ms) + 1e-10)
    dec = max(1, sr // 2000)
    te = (a + np.arange(0, len(env), dec)) / sr
    env = env[::dec]
    half = length / 2
    fig, axes = plt.subplots(2, 1, figsize=(18, 7.5), sharey=True)
    for k, ax in enumerate(axes):
        w0, w1 = t0 + k * half, t0 + (k + 1) * half
        sel = (te >= w0) & (te <= w1)
        ax.plot(te[sel], env[sel], lw=0.4, color="0.35")
        for r in rallies.itertuples():
            if r.end_t > w0 and r.start_t < w1:
                ax.axvspan(max(r.start_t, w0), min(r.end_t, w1), color="gold", alpha=0.25, lw=0)
                ax.text(max(r.start_t, w0), 1.0, f" rally {r.rally}: {r.n_hits} hits", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
        o = onsets[(onsets.t >= w0) & (onsets.t <= w1)]
        for kind, g in o.groupby("kind"):
            ax.scatter(g.t, g.peak_dbfs + 2, s=10 + 2 * g.strength_db, color=COLOURS.get(kind, "k"), label=kind, zorder=3, edgecolor="k", lw=0.3)
        if labels is not None and len(labels):
            lab = labels[(labels.t_audio >= w0) & (labels.t_audio <= w1)]
            for ev, g in lab.groupby("event"):
                ax.vlines(g.t_audio, 0, 0.08, transform=ax.get_xaxis_transform(), color=COLOURS.get(ev, "k"), lw=1.5)
        ax.set_xlim(w0, w1)
        ax.set_ylabel(f"{p.band[0] / 1e3:.0f}-{p.band[1] / 1e3:.1f} kHz RMS (dBFS)")
        ax.grid(alpha=0.3, lw=0.3)
    h, lab_ = axes[0].get_legend_handles_labels()
    h2, lab2 = axes[1].get_legend_handles_labels()
    leg = dict(zip(lab_ + lab2, h + h2))
    axes[0].legend(leg.values(), leg.keys(), loc="upper right", fontsize=8, ncol=4)
    axes[1].set_xlabel("time (s); ticks at the bottom = visual labels on the audio clock" if labels is not None else "time (s)")
    fig.suptitle(title or f"{t0:.0f}-{t0 + length:.0f} s", fontsize=10)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
