from __future__ import annotations

import argparse
import json

STEMS = ("IMG_4089", "IMG_4090", "IMG_4091")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -I -m squashvid.viewer")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("proxy", "export"):
        s = sub.add_parser(name)
        s.add_argument("stem", nargs="?")
        s.add_argument("--all", action="store_true")
        if name == "proxy":
            s.add_argument("--encoder", default="h264_videotoolbox")
            s.add_argument("--check-only", action="store_true")
    s = sub.add_parser("build-static", help="a copy of the viewer and its data as plain files (static_build.py)")
    s.add_argument("--out", required=True)
    s.add_argument("stems", nargs="*", help="clips to include (default: the same clips as --all)")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8771)
    s.add_argument("--host", default="127.0.0.1")
    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from squashvid.viewer.server import create_app

        uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
        return 0
    if a.cmd == "build-static":
        from pathlib import Path

        from squashvid.viewer.static_build import build

        print(json.dumps(build(Path(a.out), stems=a.stems or list(STEMS)), default=str), flush=True)
        return 0
    stems = STEMS if a.all else ([a.stem] if a.stem else [])
    if not stems:
        p.error("give a stem or --all")
    for stem in stems:
        if a.cmd == "proxy":
            from squashvid.viewer import proxy

            res = proxy.check_pts(stem) if a.check_only else proxy.make_proxy(stem, encoder=a.encoder)
        else:
            from squashvid.viewer.export import export

            res = export(stem)
        print(json.dumps(res, default=str), flush=True)
    return 0
