"""Video ingest keyed to presentation timestamps, with the display rotation applied.

iPhone footage is 10-bit HLG with BT.2020 primaries. Decoded straight to bgr24
it comes out bright, flat and half as saturated as the same clip exported to
SDR, because swscale converts the YUV matrix but ignores the transfer curve and
the primaries. Frames from an HLG source are therefore mapped to SDR BT.709
(see hlg_to_sdr) unless the caller asks for the raw code values.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

HLG = "arib-std-b67"


def _rotate(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply a display-matrix rotation (degrees, counter-clockwise as ffprobe reports it)."""
    r = rotation % 360
    if r == 0:
        return img
    if r == 180:
        return np.ascontiguousarray(img[::-1, ::-1])
    if r == 90:  # display matrix +90 means the frame must turn counter-clockwise
        return np.ascontiguousarray(np.rot90(img, 1))
    if r == 270:
        return np.ascontiguousarray(np.rot90(img, 3))
    raise ValueError(f"unsupported rotation {rotation}")


@dataclass
class VideoInfo:
    path: Path
    width: int  # after rotation
    height: int
    duration: float
    rotation: int
    avg_fps: float
    has_audio: bool
    audio_rate: int | None
    transfer: str | None = None  # ffprobe color_transfer, e.g. "arib-std-b67" (HLG)
    primaries: str | None = None  # ffprobe color_primaries, e.g. "bt2020"


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if "rotate" in video.get("tags", {}):
        rotation = -int(video["tags"]["rotate"])
    w, h = int(video["width"]), int(video["height"])
    if rotation % 180:
        w, h = h, w
    num, den = (int(x) for x in video["avg_frame_rate"].split("/"))
    return VideoInfo(
        path=path,
        width=w,
        height=h,
        duration=float(data["format"]["duration"]),
        rotation=rotation,
        avg_fps=num / den if den else 0.0,
        has_audio=audio is not None,
        audio_rate=int(audio["sample_rate"]) if audio else None,
        transfer=video.get("color_transfer"),
        primaries=video.get("color_primaries"),
    )


# ITU-R BT.2100 HLG inverse OETF constants.
_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991073
# Linear BT.2020 RGB -> linear BT.709 RGB (ITU-R BT.2087).
BT2020_TO_BT709 = np.array([[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]])
# Scene-light gain onto SDR, fitted to iOS's own SDR export of the court 5 clips
# (best fit 1.182-1.194 at 14 frames across all three; 4 grey levels mean error).
HLG_SDR_GAIN = 1.187
_KNEE = 0.8  # linear SDR level above which highlights roll off instead of clipping
_IN_BITS = 12  # the source is 10-bit, so a 12-bit index loses nothing
_OUT_MAX, _OUT_N = 2.0, 16384


def hlg_inverse_oetf(e: np.ndarray) -> np.ndarray:
    """HLG signal (0..1) -> normalised scene linear light (0..1)."""
    e = np.asarray(e, np.float64)
    return np.where(e <= 0.5, e * e / 3.0, (np.exp((e - _HLG_C) / _HLG_A) + _HLG_B) / 12.0)


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def soft_knee(lin: np.ndarray, knee: float = _KNEE) -> np.ndarray:
    """Identity below knee, then a tanh roll-off that approaches 1.0 instead of clipping."""
    return np.where(lin > knee, knee + (1 - knee) * np.tanh((lin - knee) / (1 - knee)), lin)


class _HlgToSdr:
    """Lookup tables for hlg_to_sdr, built on first use."""

    def __init__(self, bt2020: bool):
        self.lut_in = hlg_inverse_oetf((np.arange(2**_IN_BITS) + 0.5) / 2**_IN_BITS).astype(np.float32)
        m = BT2020_TO_BT709 if bt2020 else np.eye(3)
        self.matrix = (m[::-1] * HLG_SDR_GAIN).astype(np.float32)  # RGB in, BGR out
        x = np.linspace(0.0, _OUT_MAX, _OUT_N)
        self.lut_out = np.round(srgb_encode(soft_knee(x)) * 255).astype(np.uint8)

    def __call__(self, rgb48: np.ndarray) -> np.ndarray:
        lin = self.lut_in[rgb48 >> (16 - _IN_BITS)]
        lin = cv2.transform(lin, self.matrix)
        q = np.clip(lin * ((_OUT_N - 1) / _OUT_MAX), 0, _OUT_N - 1).astype(np.uint16)
        return self.lut_out[q]


_CONVERTERS: dict[bool, _HlgToSdr] = {}


def hlg_to_sdr(rgb48: np.ndarray, bt2020: bool = True) -> np.ndarray:
    """Map full-range 16-bit HLG R'G'B' to 8-bit SDR BGR (sRGB, BT.709 primaries).

    Scene-light conversion: undo the HLG curve, convert BT.2020 primaries to
    BT.709 in linear light, apply the fitted gain, roll off highlights and
    encode as sRGB. Tables keep it to ~25 ms per 1080p frame; the result is
    within one grey level of evaluating the formulas directly.
    """
    conv = _CONVERTERS.get(bt2020)
    if conv is None:
        conv = _CONVERTERS[bt2020] = _HlgToSdr(bt2020)
    return conv(rgb48)


class Video:
    """Sequential and random access to upright frames, each tagged with its PTS in seconds.

    HLG sources are converted to SDR BT.709 unless sdr=False, which returns the
    code values as swscale decodes them.
    """

    def __init__(self, path: str | Path, sdr: bool = True):
        self.path = Path(path)
        self.info = probe(self.path)
        self.tone_map = sdr and self.info.transfer == HLG

    def _to_bgr(self, frame: av.VideoFrame) -> np.ndarray:
        if self.tone_map:
            # swscale's 8-bit output from 10-bit input is 2-3 levels dark; the 16-bit path is exact.
            return hlg_to_sdr(frame.to_ndarray(format="rgb48le"), bt2020=self.info.primaries == "bt2020")
        return frame.to_ndarray(format="bgr24")

    @cached_property
    def timestamps(self) -> np.ndarray:
        """Presentation time of every frame, sorted, read from packets without decoding."""
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            pts = [p.pts for p in container.demux(stream) if p.pts is not None and p.size > 0]
        start = None
        with av.open(str(self.path)) as container:
            start = container.streams.video[0].start_time
        offset = (start or 0) * tb
        return np.sort(np.asarray(pts, dtype=np.float64) * tb - offset)

    def frames(self, start: float = 0.0, end: float | None = None, step: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame index, pts seconds, BGR image) from start to end."""
        ts = self.timestamps
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            offset = (stream.start_time or 0) * tb
            if start > 0:
                container.seek(int((start + offset) / tb), stream=stream, backward=True, any_frame=False)
            try:
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    t = frame.pts * tb - offset
                    if t < start - 1e-6:
                        continue
                    if end is not None and t > end + 1e-6:
                        break
                    idx = int(np.searchsorted(ts, t - 1e-6))
                    if idx % step:
                        continue
                    yield idx, t, _rotate(self._to_bgr(frame), self.info.rotation)
            finally:
                # Freeing the codec waits for its decoder threads while holding the
                # GIL. Once PyAV's Python log callback is installed (`import
                # torchvision` does it), those threads need the GIL to log, so a
                # generator closed mid-decode can deadlock under CPU load. Flushing
                # parks the threads with the GIL released first.
                stream.codec_context.flush_buffers()

    def frame_at(self, t: float) -> tuple[int, float, np.ndarray]:
        """The first frame whose PTS is at or after t."""
        for item in self.frames(start=t):
            return item
        raise IndexError(f"no frame at or after {t:.3f}s")


def extract_audio(path: str | Path, out: str | Path, rate: int | None = 44100, mono: bool = True, stream: int = 0) -> Path:
    """PCM WAV of one audio stream; rate=None keeps the native rate.

    The stream is chosen explicitly: left to itself ffmpeg picks the 4-channel
    Apple spatial track on iPhone .MOV files, which it cannot decode.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", f"0:a:{stream}", "-vn", "-ac", "1" if mono else "2"]
    if rate is not None:
        cmd += ["-ar", str(rate)]
    subprocess.run(cmd + ["-c:a", "pcm_s16le", str(out)], check=True)
    return out
