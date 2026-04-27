from __future__ import annotations

import asyncio
import shutil
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from squashvid.pipeline.video_source import is_url
from squashvid.pipeline.preprocess import SEGMENTER_VERSION
from squashvid.schemas import AnalysisResult, AnalyzeOptions, AnalyzeRequest

# Supabase client for async updates
_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            from supabase import create_client
            _supabase_client = create_client(url, key)
    return _supabase_client


class AsyncAnalyzeRequest(BaseModel):
    analysis_id: str
    video_url: str
    analysis_start_minute: float = 0.0
    max_video_minutes: float | None = None


# Simple task queue for sequential processing
from dataclasses import dataclass
from queue import Queue
import time

@dataclass
class AnalysisTask:
    analysis_id: str
    video_url: str
    start_minute: float
    max_minutes: float | None
    queued_at: float

_task_queue: Queue[AnalysisTask] = Queue()
_current_task: AnalysisTask | None = None
_queue_lock = threading.Lock()
_worker_started = False


def _get_queue_status(analysis_id: str) -> dict:
    """Get the queue position for an analysis."""
    with _queue_lock:
        if _current_task and _current_task.analysis_id == analysis_id:
            return {"status": "processing", "position": 0}

        # Check queue position
        position = 1  # Start at 1 (after current task)
        for task in list(_task_queue.queue):
            if task.analysis_id == analysis_id:
                return {"status": "queued", "position": position}
            position += 1

        return {"status": "not_found", "position": -1}


def _queue_worker():
    """Background worker that processes tasks from the queue one at a time."""
    global _current_task

    while True:
        try:
            task = _task_queue.get(block=True)

            with _queue_lock:
                _current_task = task

            # Update status to processing
            db = get_supabase()
            if db:
                try:
                    db.table("mac_video_analyses").update({
                        "status": "processing",
                        "updated_at": datetime.utcnow().isoformat(),
                    }).eq("id", task.analysis_id).execute()
                except Exception as e:
                    print(f"[queue] Failed to update status to processing: {e}")

            # Process the video
            _process_video_sync(task.analysis_id, task.video_url, task.start_minute, task.max_minutes)

            with _queue_lock:
                _current_task = None

            _task_queue.task_done()

        except Exception as e:
            print(f"[queue] Worker error: {e}")
            with _queue_lock:
                _current_task = None


def _start_worker():
    """Start the background worker thread if not already started."""
    global _worker_started
    if not _worker_started:
        _worker_started = True
        worker = threading.Thread(target=_queue_worker, daemon=True)
        worker.start()
        print("[queue] Worker thread started")


def _process_video_sync(
    analysis_id: str,
    video_url: str,
    start_minute: float,
    max_minutes: float | None,
):
    """Synchronously process video and update Supabase."""
    from squashvid.pipeline.orchestrator import analyze_video_execution

    db = get_supabase()
    if not db:
        print(f"[sync] No Supabase client for analysis {analysis_id}")
        return

    try:
        print(f"[sync] Starting analysis {analysis_id}: {video_url}")

        options = AnalyzeOptions(
            include_llm=False,
            analysis_start_minute=start_minute,
            max_video_minutes=max_minutes,
        )

        execution = analyze_video_execution(video_url, options)
        result_dict = execution.result.model_dump()

        # Update Supabase with success
        db.table("mac_video_analyses").update({
            "status": "done",
            "result_json": result_dict,
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", analysis_id).execute()

        print(f"[sync] Completed analysis {analysis_id}")

    except Exception as exc:
        error_msg = str(exc)
        print(f"[sync] Failed analysis {analysis_id}: {error_msg}")

        # Update Supabase with failure
        try:
            db.table("mac_video_analyses").update({
                "status": "failed",
                "admin_notes": f"Processing error: {error_msg[:500]}",
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", analysis_id).execute()
        except Exception as db_err:
            print(f"[sync] Failed to update DB for {analysis_id}: {db_err}")

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


@app.post("/analyze/async")
async def analyze_async(request: AsyncAnalyzeRequest):
    """
    Queue video analysis. Returns immediately with queue position.
    The result will be written to Supabase when done.
    """
    # Ensure worker is started
    _start_worker()

    # Create task and add to queue
    task = AnalysisTask(
        analysis_id=request.analysis_id,
        video_url=request.video_url,
        start_minute=request.analysis_start_minute,
        max_minutes=request.max_video_minutes,
        queued_at=time.time(),
    )

    # Check if already in queue
    existing = _get_queue_status(request.analysis_id)
    if existing["status"] != "not_found":
        return {
            "status": existing["status"],
            "analysis_id": request.analysis_id,
            "position": existing["position"],
            "message": f"Already {'processing' if existing['status'] == 'processing' else 'in queue'}.",
        }

    _task_queue.put(task)
    queue_status = _get_queue_status(request.analysis_id)

    return {
        "status": queue_status["status"],
        "analysis_id": request.analysis_id,
        "position": queue_status["position"],
        "message": "Added to processing queue.",
    }


@app.get("/analyze/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get the queue status for an analysis."""
    status = _get_queue_status(analysis_id)
    return {
        "analysis_id": analysis_id,
        "status": status["status"],
        "position": status["position"],
    }


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
        analysis_start_minute=request.analysis_start_minute,
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
    analysis_start_minute: float = Form(0.0, ge=0.0, le=240.0),
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
            analysis_start_minute=analysis_start_minute,
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
