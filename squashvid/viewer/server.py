"""Viewer server: proxy video with HTTP Range, exported data, static UI and a small settings file.

    python -I -m squashvid.viewer serve [--port 8771]

Local use only (bound to 127.0.0.1 by default); nothing here is meant to be served publicly.
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
DEFAULT_SETTINGS = {"names": {"A": "Player A", "B": "Player B"}}
CHUNK = 1 << 20
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Settings(BaseModel):
    names: dict[str, str]


def parse_range(header, size):
    """(start, end) inclusive for a single-range header; None when absent. Raises ValueError if unsatisfiable."""
    if not header:
        return None
    m = _RANGE.match(header.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        raise ValueError("malformed range")
    if m.group(1) == "":  # suffix: last N bytes
        n = int(m.group(2))
        if n == 0:
            raise ValueError("empty suffix")
        return max(0, size - n), size - 1
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable")
    return start, min(end, size - 1)


def load_settings(path):
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    names = {**DEFAULT_SETTINGS["names"], **{k: v for k, v in (data.get("names") or {}).items() if isinstance(v, str) and v.strip()}}
    return {**data, "names": names}


def create_app(work=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

    if work is None:
        from squashvid.ball.clips import WORK

        work = WORK
    work = Path(work)
    settings_path = work / "viewer_settings.json"
    app = FastAPI(title="squashvid viewer")

    def stem_dir(stem: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", stem) or not (work / stem).is_dir():
            raise HTTPException(404, f"no clip {stem}")
        return work / stem

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/clips")
    def clips():
        out = []
        for d in sorted(work.iterdir()):
            if (d / "viewer" / "meta.json").exists():
                out.append({"stem": d.name, "proxy": (d / "proxy.mp4").exists()})
        return {"clips": out}

    @app.get("/api/{stem}/meta")
    def meta(stem: str):
        p = stem_dir(stem) / "viewer" / "meta.json"
        if not p.exists():
            raise HTTPException(404, "not exported: run `python -I -m squashvid.viewer export " + stem + "`")
        return Response(p.read_bytes(), media_type="application/json", headers={"Cache-Control": "no-cache"})

    @app.get("/api/{stem}/array/{name}")
    def array(stem: str, name: str):
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise HTTPException(404)
        p = stem_dir(stem) / "viewer" / f"{name}.bin"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="application/octet-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/{stem}/proxy.mp4")
    def proxy(stem: str, request: Request):
        p = stem_dir(stem) / "proxy.mp4"
        if not p.exists():
            raise HTTPException(404, "no proxy: run `python -I -m squashvid.viewer proxy " + stem + "`")
        size = p.stat().st_size
        try:
            rng = parse_range(request.headers.get("range"), size)
        except ValueError:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        base = {"Accept-Ranges": "bytes", "Content-Type": "video/mp4"}
        if rng is None:
            start, end, status = 0, size - 1, 200
        else:
            (start, end), status = rng, 206
        length = end - start + 1
        headers = {**base, "Content-Length": str(length)}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        if request.method == "HEAD":
            return Response(status_code=status, headers=headers)

        def body():
            with p.open("rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        break
                    left -= len(chunk)
                    yield chunk

        return StreamingResponse(body(), status_code=status, headers=headers)

    app.add_api_route("/api/{stem}/proxy.mp4", proxy, methods=["HEAD"], include_in_schema=False)

    @app.get("/api/settings")
    def get_settings():
        from squashvid.players.identity import LOOKS

        return JSONResponse({**load_settings(settings_path), "looks": LOOKS})

    @app.get("/api/{stem}/report")
    def report(stem: str):
        """The clip's report rendered now, so it carries the names saved since the analysis ran."""
        from squashvid.analysis.report import clip_report

        md = clip_report(work, stem_dir(stem).name, load_settings(settings_path)["names"])
        if md is None:
            raise HTTPException(404, "no analysis yet: run `python -I -m squashvid.analysis " + stem + "`")
        return JSONResponse({"markdown": md}, headers={"Cache-Control": "no-cache"})

    @app.post("/api/settings")
    def post_settings(s: Settings):
        data = load_settings(settings_path)
        data["names"] = {**data["names"], **{k: v.strip() or DEFAULT_SETTINGS["names"].get(k, k) for k, v in s.names.items() if k in ("A", "B")}}
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(settings_path)
        from squashvid.analysis.report import rewrite_all

        try:  # keep the report files on disk in step with the names; the viewer renders its own copy regardless
            rewrite_all(work)
        except Exception as e:  # noqa: BLE001 - a stale report file must not fail the save
            log.warning("reports not rewritten after a name change: %s", e)
        return data

    from fastapi.staticfiles import StaticFiles

    class FreshStatic(StaticFiles):
        """The UI is edited in place, so the browser revalidates instead of running a stale script."""

        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    app.mount("/static", FreshStatic(directory=STATIC), name="static")
    return app
