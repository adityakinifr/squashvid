"""The static copy of the viewer: the page reads files instead of the API, and each video fits a host's file limit."""

import json
import os
import re
import shutil
import subprocess

import pytest

from squashvid.viewer import static_build as S


def test_the_video_plan_fills_the_budget_and_narrows_for_long_clips():
    plans = {d: S.video_plan(d) for d in (60.0, 291.0, 601.7, 3600.0)}
    for d, (kbps, _) in plans.items():
        assert (kbps + S.AUDIO_KBPS) * 1000 / 8 * d <= S.MAX_VIDEO_BYTES * S.SIZE_MARGIN + 1
    widths = [w for _, w in plans.values()]
    assert widths == sorted(widths, reverse=True) and widths[0] > widths[-1]


def test_the_page_is_marked_static_before_any_script_and_has_a_fragment_form():
    html = (S.STATIC / "index.html").read_text()
    marked = S.mark_static(html)
    assert marked.index("window.SV_STATIC") < marked.index('<script src="')
    frag = S.page_fragment(marked)
    assert frag.startswith("<title>") and "window.SV_STATIC" in frag
    assert not re.search(r"<!doctype|<html[\s>]|<head[\s>]|<body[\s>]|</body>", frag, re.I)


def clip(work, stem, proxy=b"proxy", web=b"web"):
    v = work / stem / "viewer"
    v.mkdir(parents=True)
    (v / "meta.json").write_text(json.dumps({"stem": stem, "duration": 10.0}))
    (v / "t.bin").write_bytes(b"\0" * 16)
    (work / stem / "proxy.mp4").write_bytes(proxy)
    (work / stem / "web.mp4").write_bytes(web)
    os.utime(work / stem / "proxy.mp4", (1, 1))  # the cached encode is newer than the proxy


def test_build_writes_the_page_the_data_and_the_saved_names(tmp_path):
    work, out = tmp_path / "work", tmp_path / "out"
    clip(work, "IMG_1")
    (work / "no_proxy" / "viewer").mkdir(parents=True)
    (work / "no_proxy" / "viewer" / "meta.json").write_text("{}")
    (work / "viewer_settings.json").write_text(json.dumps({"names": {"A": "Asha", "B": "Ben"}}))
    res = S.build(out, work)
    assert res["clips"] == ["IMG_1"] and res["encoded"] == {}
    assert "window.SV_STATIC" in (out / "index.html").read_text() and (out / "page.html").exists()
    assert (out / "static" / "app.js").exists() and not (out / "static" / "index.html").exists()
    assert json.loads((out / "data" / "clips.json").read_text())["clips"] == [{"stem": "IMG_1", "proxy": True}]
    assert json.loads((out / "data" / "settings.json").read_text())["names"] == {"A": "Asha", "B": "Ben"}
    d = out / "data" / "IMG_1"
    assert (d / "t.bin").read_bytes() == b"\0" * 16 and (d / "video.mp4").read_bytes() == b"web"
    assert json.loads((d / "meta.json").read_text())["stem"] == "IMG_1"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_the_reencode_keeps_every_frame_time(tmp_path):
    src, dst = tmp_path / "src.mp4", tmp_path / "web.mp4"
    # Variable frame rate, like the iPhone's: every third frame 10 ms late, on a 1/600 s clock.
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2", "-f", "lavfi",
                    "-i", "sine=duration=2", "-vf", "settb=1/600,setpts=N/30/TB+eq(mod(N\\,3)\\,1)*0.01/TB", "-fps_mode", "passthrough",
                    "-enc_time_base", "1:600", "-video_track_timescale", "600", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)], check=True)
    from squashvid.viewer.proxy import pts_of

    assert len(set((pts_of(src) * 600).round() % 20)) > 1  # really off the 1/30 s grid
    res = S.encode_video(src, dst, 2.0)
    assert res["ok"] and res["n_proxy"] == res["n_original"] and dst.stat().st_size <= S.MAX_VIDEO_BYTES
    assert not list(tmp_path.glob("web_x264*"))
