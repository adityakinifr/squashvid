from __future__ import annotations

import shutil
import os
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from squashvid.pipeline.video_source import is_url
from squashvid.pipeline.preprocess import SEGMENTER_VERSION
from squashvid.schemas import AnalysisResult, AnalyzeOptions, AnalyzeRequest

app = FastAPI(title="SquashVid Analyzer", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
MEDIA_ROOT = (Path.cwd() / ".squashvid_ui_media").resolve()
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
MAX_MEDIA_ENTRIES = 40
MEDIA_REGISTRY: OrderedDict[str, tuple[Path, bool]] = OrderedDict()


def _register_media_file(local_video_path: str, persist_copy: bool) -> str:
    source_path = Path(local_video_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Media source not found: {source_path}")

    token = uuid.uuid4().hex
    stored_path = source_path
    is_persisted_copy = False

    if persist_copy:
        suffix = source_path.suffix or ".mp4"
        stored_path = MEDIA_ROOT / f"{token}{suffix}"
        shutil.copy2(source_path, stored_path)
        is_persisted_copy = True

    MEDIA_REGISTRY[token] = (stored_path, is_persisted_copy)
    while len(MEDIA_REGISTRY) > MAX_MEDIA_ENTRIES:
        _, (old_path, was_persisted) = MEDIA_REGISTRY.popitem(last=False)
        if was_persisted:
            try:
                old_path.unlink()
            except OSError:
                pass

    return f"/media/{token}"


@app.get("/", include_in_schema=False)
def ui_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/media/{token}", include_in_schema=False)
def media_file(token: str) -> FileResponse:
    entry = MEDIA_REGISTRY.get(token)
    if entry is None:
        raise HTTPException(status_code=404, detail="Media token not found.")

    media_path, _ = entry
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file is no longer available.")

    return FileResponse(media_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "segmenter_version": SEGMENTER_VERSION}


@app.post("/analyze/path", response_model=AnalysisResult)
async def analyze_path(request: AnalyzeRequest) -> AnalysisResult:
    from squashvid.pipeline.orchestrator import analyze_video_execution

    source = request.video_path
    if not is_url(source):
        input_path = Path(source).expanduser()
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")
        source = str(input_path)

    options = AnalyzeOptions(
        include_llm=request.include_llm,
        llm_model=request.llm_model,
        openai_api_key=request.openai_api_key,
        player_a_name=request.player_a_name,
        player_b_name=request.player_b_name,
        motion_threshold=request.motion_threshold,
        min_rally_sec=request.min_rally_sec,
        idle_gap_sec=request.idle_gap_sec,
        max_rallies=request.max_rallies,
        segment_frame_step=request.segment_frame_step,
        tracking_frame_step=request.tracking_frame_step,
        cv_workers=request.cv_workers,
        max_video_minutes=request.max_video_minutes,
        youtube_cache_dir=request.youtube_cache_dir,
        youtube_cookies_file=request.youtube_cookies_file,
        youtube_oauth2=request.youtube_oauth2,
    )
    try:
        execution = await run_in_threadpool(analyze_video_execution, source, options)
        execution.result.source_video_url = _register_media_file(
            execution.local_video_path,
            persist_copy=False,
        )
        return execution.result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze/upload", response_model=AnalysisResult)
async def analyze_upload(
    file: UploadFile = File(...),
    include_llm: bool = Form(True),
    llm_model: str = Form("gpt-4.1-mini"),
    openai_api_key: str | None = Form(None),
    player_a_name: str = Form("Player A"),
    player_b_name: str = Form("Player B"),
    motion_threshold: float = Form(0.018),
    min_rally_sec: float = Form(4.0),
    idle_gap_sec: float = Form(1.2),
    max_rallies: int | None = Form(None),
    segment_frame_step: int = Form(2),
    tracking_frame_step: int = Form(4),
    cv_workers: int | None = Form(None, ge=1, le=128),
    max_video_minutes: float | None = Form(None, gt=0.05, le=240.0),
    youtube_cache_dir: str | None = Form(None),
    youtube_cookies_file: str | None = Form(None),
    youtube_oauth2: bool = Form(False),
) -> AnalysisResult:
    from squashvid.pipeline.orchestrator import analyze_video_execution

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(prefix="squashvid_", suffix=suffix, delete=False)
    tmp_path = Path(tmp_file.name)

    try:
        tmp_file.write(await file.read())
        tmp_file.close()

        options = AnalyzeOptions(
            include_llm=include_llm,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
            player_a_name=player_a_name,
            player_b_name=player_b_name,
            motion_threshold=motion_threshold,
            min_rally_sec=min_rally_sec,
            idle_gap_sec=idle_gap_sec,
            max_rallies=max_rallies,
            segment_frame_step=segment_frame_step,
            tracking_frame_step=tracking_frame_step,
            cv_workers=cv_workers,
            max_video_minutes=max_video_minutes,
            youtube_cache_dir=youtube_cache_dir,
            youtube_cookies_file=youtube_cookies_file,
            youtube_oauth2=youtube_oauth2,
        )
        execution = await run_in_threadpool(analyze_video_execution, str(tmp_path), options)
        execution.result.source_video_url = _register_media_file(
            execution.local_video_path,
            persist_copy=True,
        )
        return execution.result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main() -> None:
    uvicorn.run("squashvid.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
