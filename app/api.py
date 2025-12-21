"""
api.py  –  FastAPI front‑door for the Avatar Renderer Pod
==========================================================
* POST /render          → returns {jobId, statusUrl, async} (expects server-side file paths)
* POST /render-upload   → upload avatar + audio, returns {jobId, statusUrl, async} (browser-friendly)
* GET  /status/{id}     → returns either {"state": "..."} or the MP4 file
* GET  /avatars         → list available models and system capabilities
* GET  /health/live     → liveness probe  (200 OK)
* GET  /health/ready    → readiness probe (checks Celery broker if present)
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .settings import Settings  # pydantic‑based env loader

settings = Settings()

# ─────────────────── Celery optional ────────────────────────────────────── #
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
    celery_app = None  # type: ignore

# import pipeline after Celery to avoid GPU init on health checks
from .pipeline import render_pipeline  # noqa: E402

# ───────────────────────── FastAPI setup ────────────────────────────────── #
app = FastAPI(
    title="avatar‑renderer‑svc",
    version="0.1.0",
    description="Generate a lip‑synced avatar video (REST façade)",
)

# ───────────────────────────── CORS setup ────────────────────────────────── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_ROOT = Path("/tmp/avatar-jobs")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Pydantic models ────────────────────────────── #
class RenderBody(BaseModel):
    avatarPath: str = Field(..., alias="avatarPath", description="Path to PNG/JPG portrait image")
    audioPath: str = Field(..., alias="audioPath", description="Path to WAV/MP3 audio file")
    driverVideo: Optional[str] = Field(None, alias="driverVideo", description="Optional MP4 for head pose")
    visemeJson: Optional[str] = Field(None, alias="visemeJson", description="Optional phoneme alignment JSON")
    qualityMode: str = Field(
        default="auto",
        alias="qualityMode",
        description=(
            "Rendering quality mode: 'real_time' for fast streaming (SadTalker+Wav2Lip), "
            "'high_quality' for best results (FOMM+Diff2Lip+GFPGAN), or 'auto' to choose automatically"
        )
    )


# ───────────────────────── Celery vs Thread task ─────────────────────────── #
if celery_available:

    @celery_app.task(name="render_video_task")
    def _render_video_task(payload: dict):
        render_pipeline(
            face_image=payload["avatar_path"],
            audio=payload["audio_path"],
            reference_video=payload.get("driver_video"),
            viseme_json=payload.get("viseme_json"),
            quality_mode=payload.get("quality_mode", "auto"),
            out_path=payload["out_path"],
        )

else:
    def _render_video_thread(payload: dict):
        render_pipeline(
            face_image=payload["avatar_path"],
            audio=payload["audio_path"],
            reference_video=payload.get("driver_video"),
            viseme_json=payload.get("viseme_json"),
            quality_mode=payload.get("quality_mode", "auto"),
            out_path=payload["out_path"],
        )
        # mark success for readiness
        (WORK_ROOT / payload["job_id"] / "done").touch()


# ───────────────────────────── REST endpoints ────────────────────────────── #
@app.post("/render")
def render_job(body: RenderBody, bg: BackgroundTasks):
    """Start a render job and return jobId + status URL."""
    job_id = str(uuid.uuid4())
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = job_dir / "out.mp4"

    payload = {
        "job_id": job_id,
        "avatar_path": body.avatarPath,
        "audio_path": body.audioPath,
        "driver_video": body.driverVideo,
        "viseme_json": body.visemeJson,
        "quality_mode": body.qualityMode,
        "out_path": str(out_mp4),
    }

    # save original request
    (job_dir / "request.json").write_text(json.dumps(body.dict(by_alias=True), indent=2))

    if celery_available:
        task = _render_video_task.delay(payload)  # type: ignore
        (job_dir / "celery_id").write_text(task.id)
        async_mode = True
    else:
        bg.add_task(_render_video_thread, payload)
        async_mode = False

    return {
        "jobId": job_id,
        "statusUrl": f"/status/{job_id}",
        "async": async_mode,
    }


@app.post("/render-upload")
async def render_upload(
    bg: BackgroundTasks,
    avatar: UploadFile = File(...),
    audio: UploadFile = File(...),
    qualityMode: str = Form("auto"),
):
    """Upload avatar image + audio, start render job, return jobId + status URL."""
    job_id = str(uuid.uuid4())
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    avatar_path = job_dir / f"avatar_{avatar.filename}"
    audio_path = job_dir / f"audio_{audio.filename}"
    out_mp4 = job_dir / "out.mp4"

    with avatar_path.open("wb") as f:
        shutil.copyfileobj(avatar.file, f)
    with audio_path.open("wb") as f:
        shutil.copyfileobj(audio.file, f)

    payload = {
        "job_id": job_id,
        "avatar_path": str(avatar_path),
        "audio_path": str(audio_path),
        "quality_mode": qualityMode,
        "out_path": str(out_mp4),
    }

    # save upload metadata
    (job_dir / "upload.json").write_text(
        json.dumps(
            {
                "avatar_filename": avatar.filename,
                "audio_filename": audio.filename,
                "quality_mode": qualityMode,
            },
            indent=2,
        )
    )

    if celery_available:
        task = _render_video_task.delay(payload)  # type: ignore
        (job_dir / "celery_id").write_text(task.id)
        async_mode = True
    else:
        bg.add_task(_render_video_thread, payload)
        async_mode = False

    return {
        "jobId": job_id,
        "statusUrl": f"/status/{job_id}",
        "async": async_mode,
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Fetch job state or return the completed MP4."""
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
        state = "finished" if done_marker.exists() else "processing"
        return {"state": state}


# ────────────────────── Health & Readiness probes ────────────────────────── #
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


@app.get("/avatars")
def list_avatars():
    """List available avatar models and their status.

    This endpoint provides health status for all avatar rendering models,
    indicating which models are available and ready for use.
    """
    import torch
    from pathlib import Path

    # Model checkpoint paths from settings
    model_checks = {
        "fomm": {
            "name": "First Order Motion Model",
            "path": settings.FOMM_CKPT_DIR / "vox-cpk.pth.tar",
            "purpose": "Head pose and expression generation",
            "required_for": "high_quality"
        },
        "diff2lip": {
            "name": "Diff2Lip",
            "path": settings.DIFF2LIP_CKPT_DIR / "Diff2Lip.pth",
            "purpose": "Photorealistic lip synchronization",
            "required_for": "high_quality"
        },
        "sadtalker": {
            "name": "SadTalker",
            "path": settings.SADTALKER_CKPT_DIR / "sadtalker.pth",
            "purpose": "Talking head generation (fallback)",
            "required_for": "real_time"
        },
        "wav2lip": {
            "name": "Wav2Lip",
            "path": settings.WAV2LIP_CKPT,
            "purpose": "Lip synchronization GAN (fallback)",
            "required_for": "real_time"
        },
        "gfpgan": {
            "name": "GFPGAN",
            "path": settings.GFPGAN_CKPT,
            "purpose": "Face enhancement",
            "required_for": "both"
        }
    }

    models_status = {}
    for model_id, info in model_checks.items():
        path = Path(info["path"])
        models_status[model_id] = {
            "name": info["name"],
            "purpose": info["purpose"],
            "required_for": info["required_for"],
            "available": path.exists(),
            "path": str(path)
        }

    # System capabilities
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_info = []
    if cuda_available:
        for i in range(gpu_count):
            try:
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                })
            except Exception:
                pass

    # Determine rendering modes available
    high_quality_ready = (
        models_status["fomm"]["available"] and
        models_status["diff2lip"]["available"] and
        cuda_available
    )
    real_time_ready = (
        models_status["sadtalker"]["available"] and
        models_status["wav2lip"]["available"]
    )

    return JSONResponse({
        "status": "ready",
        "models": models_status,
        "system": {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpus": gpu_info,
            "celery_enabled": celery_available
        },
        "rendering_modes": {
            "high_quality": {
                "available": high_quality_ready,
                "description": "FOMM + Diff2Lip pipeline for best quality (requires GPU)",
                "models": ["fomm", "diff2lip", "gfpgan"]
            },
            "real_time": {
                "available": real_time_ready,
                "description": "SadTalker + Wav2Lip pipeline for faster processing",
                "models": ["sadtalker", "wav2lip", "gfpgan"]
            }
        }
    })
