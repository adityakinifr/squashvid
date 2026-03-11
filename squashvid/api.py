from __future__ import annotations

import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from squashvid.pipeline.video_source import is_url
from squashvid.schemas import AnalysisResult, AnalyzeOptions, AnalyzeRequest

app = FastAPI(title="SquashVid Analyzer", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def ui_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze/path", response_model=AnalysisResult)
async def analyze_path(request: AnalyzeRequest) -> AnalysisResult:
    from squashvid.pipeline.orchestrator import analyze_video

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
        motion_threshold=request.motion_threshold,
        min_rally_sec=request.min_rally_sec,
        idle_gap_sec=request.idle_gap_sec,
        max_rallies=request.max_rallies,
        segment_frame_step=request.segment_frame_step,
        tracking_frame_step=request.tracking_frame_step,
        youtube_cache_dir=request.youtube_cache_dir,
    )
    try:
        return await run_in_threadpool(analyze_video, source, options)
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
    motion_threshold: float = Form(0.018),
    min_rally_sec: float = Form(4.0),
    idle_gap_sec: float = Form(1.2),
    max_rallies: int | None = Form(None),
    segment_frame_step: int = Form(2),
    tracking_frame_step: int = Form(4),
    youtube_cache_dir: str | None = Form(None),
) -> AnalysisResult:
    from squashvid.pipeline.orchestrator import analyze_video

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
            motion_threshold=motion_threshold,
            min_rally_sec=min_rally_sec,
            idle_gap_sec=idle_gap_sec,
            max_rallies=max_rallies,
            segment_frame_step=segment_frame_step,
            tracking_frame_step=tracking_frame_step,
            youtube_cache_dir=youtube_cache_dir,
        )
        return await run_in_threadpool(analyze_video, str(tmp_path), options)
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
