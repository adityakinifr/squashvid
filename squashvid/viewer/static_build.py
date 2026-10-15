"""A static copy of the viewer: the page, its scripts and models, and every exported clip's data as plain files, so it
can be opened from any static file host (or published as a claude.ai artifact) without the API server.

    python -I -m squashvid.viewer build-static --out DIR      # then e.g. `python -m http.server -d DIR`

DIR gets index.html (the page, which sets window.SV_STATIC so the scripts read files instead of /api), page.html (the
same page as a fragment, without the document tags, for hosts that wrap pages in their own skeleton), static/ (the
viewer's scripts, styles and vendored libraries), and data/:

- clips.json, settings.json: the clip list and the names saved now (a static copy cannot save names, so the Names
  button is hidden);
- <stem>/meta.json and <stem>/<array>.bin: the export, unchanged;
- <stem>/report.json, pooled/insights.json, pooled/report.json: the reports rendered with those names;
- <stem>/video.mp4: the proxy re-encoded to fit MAX_VIDEO_BYTES (a host's per-file limit): H.264 at the bitrate that
  fills it, at the largest width that still gets MIN_BPP bits per pixel, with mono audio, keeping the proxy's frame
  timestamps (checked, as the viewer maps time to frames by them). The encode is cached as work/<stem>/web.mp4.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from squashvid.ball.clips import WORK

STATIC = Path(__file__).with_name("static")
MAX_VIDEO_BYTES = 14_500_000
AUDIO_KBPS = 24
MIN_BPP = 0.022
WIDTHS = (1280, 960, 854, 640, 512)
SIZE_MARGIN = 0.94  # two-pass x264 lands within a few percent of its target


def video_plan(duration_s: float, fps: float = 30.0, max_bytes: int = MAX_VIDEO_BYTES, aspect: float = 16 / 9) -> tuple[int, int]:
    """(video kbps, width) for a clip of this length: the bitrate that fills max_bytes with the audio, and the
    largest width in WIDTHS that still gets MIN_BPP bits per pixel (the smallest otherwise)."""
    kbps = int(max_bytes * 8 * SIZE_MARGIN / duration_s / 1000) - AUDIO_KBPS
    for w in WIDTHS:
        if kbps * 1000 / (w * (w / aspect) * fps) >= MIN_BPP:
            return kbps, w
    return kbps, WIDTHS[-1]


def encode_video(src: Path, dst: Path, duration_s: float) -> dict:
    """Re-encode `src` into `dst` (see video_plan), keeping its frame timestamps; returns size and the PTS check."""
    from squashvid.viewer.proxy import compare_pts, pts_of

    kbps, width = video_plan(duration_s)
    tmp = dst.with_suffix(".tmp.mp4")
    # The encoder's own clock must be the iPhone's 1/600 s too; by default it is 1/frame rate and snaps VFR times.
    common = ["-fps_mode", "passthrough", "-enc_time_base", "1:600", "-video_track_timescale", "600", "-vf", f"scale={width}:-2", "-c:v", "libx264",
              "-preset", "slow", "-b:v", f"{kbps}k", "-g", "60", "-pix_fmt", "yuv420p"]
    log = dst.parent / "web_x264"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *common, "-pass", "1", "-passlogfile", str(log), "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *common, "-pass", "2", "-passlogfile", str(log),
                    "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-ac", "1", "-movflags", "+faststart", str(tmp)], check=True)
    for f in dst.parent.glob("web_x264*"):
        f.unlink()
    check = compare_pts(pts_of(tmp), pts_of(src))
    if not check["ok"]:
        raise RuntimeError(f"{dst}: frame timestamps changed in the re-encode: {check}")
    tmp.replace(dst)
    return {"bytes": dst.stat().st_size, "kbps": kbps, "width": width, **check}


def page_fragment(html: str) -> str:
    """The page without its document tags: the title and stylesheet, then the body's content."""
    title = re.search(r"<title>.*?</title>", html, re.S).group(0)
    links = "\n".join(re.findall(r"<link [^>]*>", html))
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.S).group(1)
    return f"{title}\n{links}\n{body.strip()}\n"


def mark_static(html: str) -> str:
    """The page with window.SV_STATIC set before its first script."""
    i = html.index("<script")
    return html[:i] + "<script>window.SV_STATIC = true;</script>\n" + html[i:]


def build(out: Path, work: Path = WORK, stems: list[str] | None = None) -> dict:
    from squashvid.analysis.report import clip_report, render_pooled
    from squashvid.players.identity import LOOKS
    from squashvid.viewer.server import load_settings

    out = Path(out)
    stems = stems or sorted(d.name for d in work.iterdir() if (d / "viewer" / "meta.json").exists() and (d / "proxy.mp4").exists())
    if out.exists():
        shutil.rmtree(out / "static", ignore_errors=True)
        shutil.rmtree(out / "data", ignore_errors=True)
    shutil.copytree(STATIC, out / "static", ignore=shutil.ignore_patterns("index.html", ".*"))
    html = mark_static((STATIC / "index.html").read_text())
    (out / "index.html").write_text(html)
    (out / "page.html").write_text(page_fragment(html))
    data = out / "data"
    data.mkdir(parents=True)
    settings = load_settings(work / "viewer_settings.json")
    names = settings["names"]
    (data / "settings.json").write_text(json.dumps({**settings, "looks": LOOKS}))
    (data / "clips.json").write_text(json.dumps({"clips": [{"stem": s, "proxy": True} for s in stems]}))
    ins = work / "analysis" / "insights.json"
    if ins.exists():
        (data / "pooled").mkdir()
        shutil.copyfile(ins, data / "pooled" / "insights.json")
        (data / "pooled" / "report.json").write_text(json.dumps({"markdown": render_pooled(json.loads(ins.read_text()), names)}))
    videos = {}
    for stem in stems:
        src, d = work / stem, data / stem
        shutil.copytree(src / "viewer", d)
        md = clip_report(work, stem, names)
        if md is not None:
            (d / "report.json").write_text(json.dumps({"markdown": md}))
        web, proxy = src / "web.mp4", src / "proxy.mp4"
        if not web.exists() or web.stat().st_mtime < proxy.stat().st_mtime:
            videos[stem] = encode_video(proxy, web, json.loads((d / "meta.json").read_text())["duration"])
        shutil.copyfile(web, d / "video.mp4")
    files = [p for p in out.rglob("*") if p.is_file()]
    return {"out": str(out), "clips": stems, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "largest": max((p.stat().st_size, str(p.relative_to(out))) for p in files), "encoded": videos}


__all__ = ["build", "encode_video", "mark_static", "page_fragment", "video_plan"]
