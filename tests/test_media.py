from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from squashvid.media.video import BT2020_TO_BT709, HLG, HLG_SDR_GAIN, Video, extract_audio, hlg_inverse_oetf, hlg_to_sdr, probe, soft_knee, srgb_encode

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

W, H = 64, 48


def _make(tmp_path, rotation: int | None):
    """An asymmetric test clip, optionally tagged with a display-matrix rotation (no re-encode)."""
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=10:duration=0.5", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-qp", "0", str(src)],
        check=True,
    )
    if rotation is None:
        return src
    out = tmp_path / f"rot{rotation}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(src), "-c", "copy", str(out)], check=True)
    return out


def _ffmpeg_upright(path, w: int, h: int) -> np.ndarray:
    """First frame as ffmpeg displays it (ffmpeg applies the display matrix by default)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_frames_are_upright_like_ffmpeg_displays_them(tmp_path, rotation):
    path = _make(tmp_path, rotation)
    info = probe(path)
    assert (info.width, info.height) == ((H, W) if rotation % 180 else (W, H))
    _, _, ours = next(Video(path).frames())
    ref = _ffmpeg_upright(path, info.width, info.height)
    assert ours.shape == ref.shape
    assert np.mean(np.abs(ours.astype(int) - ref.astype(int))) < 3.0


def test_timestamps_start_at_zero_and_match_frames(tmp_path):
    path = _make(tmp_path, None)
    v = Video(path)
    ts = v.timestamps
    assert ts[0] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(np.diff(ts), 0.1, atol=1e-3)
    got = [(i, t) for i, t, _ in v.frames()]
    assert [i for i, _ in got] == list(range(len(ts)))
    assert np.allclose([t for _, t in got], ts)
    idx, t, _ = v.frame_at(0.25)
    assert t >= 0.25 - 1e-6 and idx == int(np.searchsorted(ts, 0.25 - 1e-6))


def _hlg_encode(lin: np.ndarray) -> np.ndarray:
    """BT.2100 HLG OETF: scene linear (0..1) -> signal (0..1)."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin = np.asarray(lin, np.float64)
    return np.where(lin <= 1 / 12, np.sqrt(3 * lin), a * np.log(np.maximum(12 * lin - b, 1e-12)) + c)


def _to_rgb48(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, 0, 1) * 65535).astype(np.uint16)


def test_hlg_inverse_oetf_undoes_the_oetf():
    lin = np.linspace(0, 1, 1001)
    assert np.allclose(hlg_inverse_oetf(_hlg_encode(lin)), lin, atol=1e-9)


def test_hlg_to_sdr_matches_the_formulas():
    rgb48 = np.random.default_rng(0).integers(0, 65536, size=(64, 64, 3), dtype=np.uint16)
    lin = hlg_inverse_oetf(rgb48 / 65535.0) @ (BT2020_TO_BT709.T * HLG_SDR_GAIN)
    ref = np.clip(srgb_encode(soft_knee(lin)) * 255, 0, 255)[..., ::-1]
    assert np.abs(hlg_to_sdr(rgb48).astype(float) - ref).max() <= 1.0


def test_hlg_greys_stay_neutral_and_ordered():
    ramp = _to_rgb48(np.repeat(np.linspace(0, 1, 256)[None, :, None], 3, axis=2))
    out = hlg_to_sdr(ramp)[0].astype(int)
    assert out[0].tolist() == [0, 0, 0]
    assert np.abs(out - out[:, 1:2]).max() <= 1  # B = G = R
    assert (np.diff(out[:, 1]) >= 0).all()


def test_hlg_to_sdr_converts_bt2020_primaries():
    # A BT.709 primary written in BT.2020 coordinates must come back as that primary,
    # where ignoring the primaries (the raw decode) leaves it desaturated.
    for channel in range(3):
        want = np.zeros(3)
        want[channel] = 0.5 / HLG_SDR_GAIN
        lin2020 = np.linalg.solve(BT2020_TO_BT709, want)
        out = hlg_to_sdr(_to_rgb48(_hlg_encode(lin2020))[None, None])[0, 0][::-1].astype(int)  # RGB
        others = np.delete(out, channel)
        assert out[channel] == pytest.approx(255 * srgb_encode(0.5), abs=1.5)
        assert others.max() <= 1


def _make_hlg(tmp_path):
    path = tmp_path / "hlg.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"gradients=size={W}x{H}:rate=10:duration=0.3:c0=0x6878a0:c1=0xa08060:x0=0:y0=0:x1={W - 1}:y1={H - 1}",
            "-c:v", "libx265", "-x265-params", "log-level=error:lossless=1:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
            "-pix_fmt", "yuv420p10le", str(path),
        ],
        check=True,
    )
    return path


def _bt2020_limited_to_rgb48(yuv10: np.ndarray) -> np.ndarray:
    """R'G'B' from 10-bit limited-range BT.2020 YUV by the textbook formula (independent of swscale)."""
    y, cb, cr = (yuv10[..., 0] - 64) / 876, (yuv10[..., 1] - 512) / 896, (yuv10[..., 2] - 512) / 896
    kr, kb = 0.2627, 0.0593
    r, b = y + 2 * (1 - kr) * cr, y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / (1 - kr - kb)
    return _to_rgb48(np.stack([r, g, b], -1))


def test_hlg_video_frames_are_mapped_to_sdr(tmp_path):
    path = _make_hlg(tmp_path)
    info = probe(path)
    assert (info.transfer, info.primaries) == (HLG, "bt2020")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-"], check=True, capture_output=True).stdout
    yuv10 = np.frombuffer(raw, np.uint16).reshape(3, H, W).transpose(1, 2, 0) / 64.0
    ref = hlg_to_sdr(_bt2020_limited_to_rgb48(yuv10)).astype(int)
    _, _, sdr = next(Video(path).frames())
    _, _, code_values = next(Video(path, sdr=False).frames())
    # Mild colours: on saturated ones the BT.709 conversion magnifies swscale's ~1 level YUV->RGB rounding.
    assert np.abs(sdr.astype(int) - ref).mean() < 1.5
    assert np.abs(code_values.astype(int) - ref).mean() > 10  # sdr=False really is the untouched decode
    assert not Video(_make(tmp_path, None)).tone_map  # SDR sources pass through


def _dominant_hz(path) -> tuple[float, int, int]:
    import wave

    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())[:, 0]
        rate, channels = w.getframerate(), w.getnchannels()
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.abs(np.fft.rfft(x)).argmax()]), rate, channels


def test_extract_audio_takes_the_requested_stream_not_ffmpegs_pick(tmp_path):
    # Like an iPhone .MOV: ffmpeg's own choice is a later, 4-channel track.
    src = tmp_path / "two.mov"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.5",
            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.5",
            "-filter_complex", "[0:a]pan=stereo|c0=c0|c1=c0[s];[1:a]pan=4c|c0=c0|c1=c0|c2=c0|c3=c0[q]",
            "-map", "[s]", "-map", "[q]", "-c:a", "pcm_s16le", "-disposition:a:0", "0", "-disposition:a:1", "default", str(src),
        ],
        check=True,
    )
    hz, rate, channels = _dominant_hz(extract_audio(src, tmp_path / "a.wav", rate=None, mono=False))
    assert (hz, rate, channels) == (pytest.approx(440, abs=3), 48000, 2)
    hz, rate, channels = _dominant_hz(extract_audio(src, tmp_path / "b.wav", stream=1))
    assert (hz, rate, channels) == (pytest.approx(1000, abs=3), 44100, 1)
