"""
api.py  –  FastAPI front‑door for the Avatar Renderer Pod
=========================================================
* POST /render         → returns {jobId, statusUrl}
* GET  /status/{id}    → returns either {"state": "..."} or the MP4 file
* GET  /health/live    → liveness probe  (200 OK)
* GET  /health/ready   → readiness probe (checks Celery broker if present)
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

from settings import Settings

settings = Settings()  # pydantic‑based env loader

# ───────────────────────────  Celery optional  ──────────────────────────── #
celery_available = False
try:
    from celery import Celery
    from celery.result import AsyncResult

    celery_app = Celery(
        "avatar_renderer",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_BACKEND_URL or settings.CELERY_BROKER_URL,
    )
    celery_available = bool(settings.CELERY_BROKER_URL)
except ImportError:
    celery_app = None  # type: ignore[assignment]

# import here to avoid GPU initialise for health probes
from pipeline import render_pipeline  # noqa: E402

app = FastAPI(
    title="avatar‑renderer‑svc",
    version="0.1.0",
    description="Generate a lip‑synced avatar video (REST façade)",
)

WORK_ROOT = Path("/tmp/avatar-jobs")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ────────────────────────────  tasks  ───────────────────────────────────── #
if celery_available:

    @celery_app.task(name="render_video_task")
    def _render_video_task(payload: dict):
        render_pipeline(
            face_image=payload["avatar_path"],
            audio=payload["audio_path"],
            reference_video=payload.get("driver_video"),
            viseme_json=payload.get("viseme_json"),
            out_path=payload["out_path"],
        )


else:
    # poor‑man's async fallback so /render still works without Celery
    def _render_video_thread(payload: dict):
        render_pipeline(
            face_image=payload["avatar_path"],
            audio=payload["audio_path"],
            reference_video=payload.get("driver_video"),
            viseme_json=payload.get("viseme_json"),
            out_path=payload["out_path"],
        )
        # mark success
        (WORK_ROOT / payload["job_id"] / "done").touch()


# ──────────────────────────────  REST  ──────────────────────────────────── #
@app.post("/render")
def render_job(body: dict, bg: BackgroundTasks):
    """
    Body example:
      {
        "avatarPath": "/mnt/models/alice.png",
        "audioPath":  "/tmp/hello.wav",
        "driverVideo": null
      }
    """
    avatar_p = body.get("avatarPath")
    audio_p = body.get("audioPath")
    if not avatar_p or not audio_p:
        raise HTTPException(400, "avatarPath and audioPath are required")

    job_id = str(uuid.uuid4())
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = job_dir / "out.mp4"

    payload = {
        "job_id": job_id,
        "avatar_path": avatar_p,
        "audio_path": audio_p,
        "driver_video": body.get("driverVideo"),
        "viseme_json": body.get("visemeJson"),
        "out_path": str(out_mp4),
    }

    # record the request
    (job_dir / "request.json").write_text(json.dumps(body, indent=2))

    if celery_available:
        task: AsyncResult = _render_video_task.delay(payload)  # type: ignore
        (job_dir / "celery_id").write_text(task.id)
    else:
        bg.add_task(_render_video_thread, payload)

    return {
        "jobId": job_id,
        "statusUrl": f"/status/{job_id}",
        "async": celery_available,
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job_dir = WORK_ROOT / job_id
    if not job_dir.exists():
        raise HTTPException(404, "job not found")

    out_mp4 = job_dir / "out.mp4"
    if out_mp4.exists():
        return FileResponse(out_mp4, media_type="video/mp4")

    if celery_available:
        celery_id_path = job_dir / "celery_id"
        if not celery_id_path.exists():
            raise HTTPException(500, "job metadata missing")
        task_id = celery_id_path.read_text()
        task = AsyncResult(task_id, app=celery_app)
        return {"state": task.state}
    else:
        done_marker = job_dir / "done"
        return {"state": "finished" if done_marker.exists() else "processing"}


# ───────────────────── health / readiness  ─────────────────────────────── #
@app.get("/health/live")
def liveness():
    return JSONResponse({"status": "alive"})


@app.get("/health/ready")
def readiness():
    if celery_available:
        try:
            celery_app.control.ping(timeout=1)
        except Exception as err:
            raise HTTPException(503, f"celery ping failed: {err}") from err
    return JSONResponse({"status": "ready"})
