"""Review plots: a timeline per clip that puts every signal and the detected rallies and shots on one axis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from squashvid.analysis.io import REVIEW_DIR, Match  # noqa: E402


def timeline(m: Match, rallies: pd.DataFrame, hits: pd.DataFrame, ev: pd.DataFrame, tag: str = "") -> str:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(5, 1, figsize=(26, 12), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1, 1, 0.8]})
    colour = {"A": "#1f77b4", "B": "#ff7f0e"}
    for k in "AB":
        tr = m.tracks[k]
        ax[0].plot(m.t, pd.Series(tr.speed).rolling(15, center=True, min_periods=1).mean(), color=colour[k], lw=0.6, label=f"{k} speed")
        ax[1].plot(m.t, np.hypot(tr.x - m.T[0], tr.y - m.T[1]), color=colour[k], lw=0.6, label=f"{k} distance from T")
    ax[0].plot(ev.t, ev.play_p * 4, color="k", lw=1.2, label="play score x4")
    ax[0].legend(loc="upper right", ncol=3)
    ax[0].set_ylabel("m/s")
    ax[1].set_ylabel("m from T")
    ax[1].legend(loc="upper right", ncol=2)
    for kind, c in (("front_wall", "r"), ("racket", "b"), ("floor", "g")):
        q = m.onsets[m.onsets.kind == kind]
        ax[2].vlines(q.t, 0, q.strength_db, color=c, lw=0.8, label=kind)
    ax[2].legend(loc="upper right", ncol=3)
    ax[2].set_ylabel("audio onsets (dB)")
    d = m.ball_detections
    if d is not None and len(d):
        ax[3].plot(ev.t, ev.ball_presence, color="purple", lw=0.8, label="ball detections per frame")
    if m.ball_events is not None:
        for kind, c in (("front_wall", "r"), ("racket", "b"), ("floor", "g")):
            q = m.ball_events[m.ball_events.kind == kind]
            ax[3].scatter(q.t, np.full(len(q), 0.9), s=8, color=c, marker="|")
    ax[3].legend(loc="upper right")
    ax[3].set_ylabel("ball")
    for k, y in (("A", 1), ("B", 0)):
        q = hits[hits.hitter == k]
        ax[4].scatter(q.t, np.full(len(q), y), s=14, color=colour[k], label=f"{k} hits")
    ax[4].set_yticks([0, 1], ["B", "A"])
    for a in ax:
        for r in rallies.itertuples():
            a.axvspan(r.start_t, r.end_t, color="green", alpha=0.07)
    for r in rallies.itertuples():
        ax[0].text(r.start_t, ax[0].get_ylim()[1] * 0.9, str(r.rally), fontsize=7)
    ax[4].set_xlabel("seconds")
    fig.suptitle(f"{m.stem}: {len(rallies)} rallies, {len(hits)} shots" + (" (ball events used)" if m.ball_events is not None else " (no ball events)"))
    fig.tight_layout()
    path = REVIEW_DIR / f"timeline_{m.stem}{('_' + tag) if tag else ''}.png"
    fig.savefig(path, dpi=60)
    plt.close(fig)
    return str(path)


def boundary_sheet(stem: str, boundaries: list[tuple[str, float]], path=None, offsets=(-2.0, -0.7, 0.7, 2.0), width: int = 420) -> str:
    """Contact sheet of decoded frames around each (label, time): one row per boundary, one column per offset."""
    import cv2

    from squashvid.ball.clips import video

    v = video(stem, sdr=True)
    rows = []
    for label, t in boundaries:
        cells = []
        for off in offsets:
            _, ts, img = v.frame_at(max(t + off, 0.0))
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
            cv2.putText(img, f"{label} {off:+.1f}s (t={ts:.1f})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cells.append(img)
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = path or REVIEW_DIR / f"boundaries_{stem}.jpg"
    cv2.imwrite(str(path), sheet)
    return str(path)
