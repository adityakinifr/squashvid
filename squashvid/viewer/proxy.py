"""Browser proxy of an HLG original: H.264 SDR with the original presentation timestamps.

Frames come from `squashvid.media.video.Video(sdr=True)` (the project's own HLG to SDR conversion), are encoded
with their original PTS (VFR preserved, millisecond-exact on the 1/600 s clock the iPhone uses), and the first
audio stream is stream-copied, so proxy time equals original time. `check_pts` compares the proxy's frame PTS
list with `Video.timestamps`.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

TIME_BASE = Fraction(1, 600)
GOP = 15


def proxy_path(stem: str) -> Path:
    from squashvid.ball.clips import work_dir

    return work_dir(stem) / "proxy.mp4"


def pts_of(path: str | Path) -> np.ndarray:
    """Presentation time (seconds, sorted) of every video packet in a file, read without decoding."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        pts = [p.pts for p in c.demux(s) if p.pts is not None and p.size > 0]
    return np.sort(np.asarray(pts, np.float64) * tb)


def compare_pts(proxy: np.ndarray, original: np.ndarray, tol: float = 1.0 / 600 + 1e-6) -> dict:
    """Does a proxy carry its original's presentation times?

    The proxy has one frame per frame the decoder returns; the iPhone files also hold trailing packets that never
    decode (IMG_4091: 32, after the last picture), so the original list may be longer. The check passes when the
    proxy's PTS equal the original's first len(proxy) PTS within one tick (tol) of the 1/600 s clock and every
    leftover original PTS lies after the proxy's last frame.
    """
    n = min(len(proxy), len(original))
    err = np.abs(proxy[:n] - original[:n]) if n else np.zeros(0)
    worst = float(err.max()) if n else 0.0
    tail = original[n:]
    tail_after = bool(len(proxy) >= len(original) or (len(proxy) > 0 and tail.min() > proxy[-1] - tol))
    return {
        "n_proxy": int(len(proxy)),
        "n_original": int(len(original)),
        "n_original_tail_not_in_proxy": int(max(0, len(original) - len(proxy))),
        "max_abs_err_s": worst,
        "ok": bool(len(proxy) > 0 and len(proxy) <= len(original) and worst <= tol and tail_after),
    }


def check_pts(stem: str, path: Path | None = None) -> dict:
    from squashvid.ball.clips import video

    return compare_pts(pts_of(path or proxy_path(stem)), video(stem).timestamps)


def make_proxy(stem: str, encoder: str = "h264_videotoolbox", bitrate: int = 6_000_000, out: Path | None = None) -> dict:
    from squashvid.ball.clips import clip_path, video

    out = out or proxy_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    v = video(stem)
    info = v.info
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        vonly = Path(tmp) / "video.mp4"
        with av.open(str(vonly), "w") as oc:
            st = oc.add_stream(encoder, rate=30)
            st.time_base = TIME_BASE
            st.codec_context.time_base = TIME_BASE
            st.width, st.height = info.width, info.height
            st.pix_fmt = "yuv420p"
            st.codec_context.gop_size = GOP
            st.codec_context.max_b_frames = 0
            opts = {"allow_sw": "1", "realtime": "0"} if "videotoolbox" in encoder else {"preset": "veryfast", "crf": "22"}
            if "videotoolbox" in encoder:
                st.bit_rate = bitrate
            st.codec_context.options = opts
            for attr, val in (("colorspace", 1), ("color_primaries", 1), ("color_trc", 1), ("color_range", 1)):
                try:
                    setattr(st.codec_context, attr, val)
                except Exception:  # older PyAV builds lack some tags
                    pass
            n = 0
            for _, t, bgr in v.frames():
                f = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p", dst_colorspace=1, interpolation="BILINEAR")
                f.pts = int(round(t / float(TIME_BASE)))
                f.time_base = TIME_BASE
                for p in st.encode(f):
                    oc.mux(p)
                n += 1
            for p in st.encode(None):
                oc.mux(p)
        tmp_out = Path(tmp) / "final.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(vonly), "-i", str(clip_path(stem)), "-map", "0:v:0", "-map", "1:a:0",
             "-c", "copy", "-movflags", "+faststart", str(tmp_out)],
            check=True,
        )
        tmp_out.replace(out)
    res = {"stem": stem, "frames": n, "seconds": round(time.time() - t0, 1), "bytes": out.stat().st_size, **check_pts(stem, out)}
    return res
