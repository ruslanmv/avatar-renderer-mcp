"""
api.py – FastAPI front-door for the Avatar Renderer Pod
==========================================================
* POST /render          → returns {jobId, statusUrl, async} (expects server-side file paths)
* POST /render-upload   → upload avatar + audio, returns {jobId, statusUrl, async} (browser-friendly)
* GET  /status/{id}     → returns either {"state": "..."} or the MP4 file
* GET  /avatars         → list available models and system capabilities
* GET  /health/live     → liveness probe (200 OK)
* GET  /health/ready    → readiness probe (checks Celery broker if present)
* POST /text-to-audio   → synthesize text to speech via Chatterbox
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .settings import Settings  # pydantic-based env loader

settings = Settings()
log = logging.getLogger("avatar-renderer.api")

# ─────────────────── TTS imports ───────────────────────────────────────── #
try:
    from .tts.chatterbox_client import (
        ChatterboxTtsError,
        tts_wav_bytes_async,
        tts_wav_base64_async,
        chatterbox_health_async,
    )
    tts_available = True
except ImportError:
    tts_available = False

# ─────────────────── Celery optional ────────────────────────────────────── #
# The Celery app and render task live in app.worker (single source of truth) so
# the API enqueues exactly the task the worker process consumes. If Celery (or a
# heavy transitive import) is unavailable, we transparently fall back to running
# the pipeline in a FastAPI BackgroundTask.
celery_available = False
try:
    from celery.result import AsyncResult

    from .worker import celery_app, render_task

    celery_available = bool(settings.CELERY_BROKER_URL)
except ImportError:
    celery_app = None  # type: ignore
    render_task = None  # type: ignore

# import pipeline after Celery to avoid GPU init on health checks
from .pipeline import render_pipeline  # noqa: E402

# ───────────────────────── FastAPI setup ────────────────────────────────── #
app = FastAPI(
    title="avatar-renderer-svc",
    version="0.1.0",
    description="Generate a lip-synced avatar video (REST façade)",
)

# ───────────────────────────── CORS setup ────────────────────────────────── #
# CORS_ALLOW_ORIGINS may be a comma-separated list of explicit origins, or "*".
# Per the CORS spec a wildcard origin cannot be combined with credentials, so we
# only enable credentials when explicit origins are configured.
_cors_origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────── Static frontend (HF Spaces / Docker) ─────────────────── #
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def serve_frontend():
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Avatar Renderer MCP</h1><p>API is running. Visit <a href='/docs'>/docs</a></p>")

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
            "Rendering quality mode: 'real_time' (Wav2Lip/MuseTalk), "
            "'high_quality' (FOMM+Diff2Lip/LatentSync+GFPGAN), "
            "'cinematic' (Hallo3 DiT), '3d' (Gaussian Splatting), or 'auto'"
        )
    )
    enhancements: Optional[list] = Field(
        default=None,
        description=(
            "List of enhancement names to apply. Options: "
            "'emotion_expressions', 'musetalk_lipsync', 'eye_gaze_blink', "
            "'liveportrait_driver', 'latentsync_lipsync', 'hallo3_cinematic', "
            "'cosyvoice_tts', 'viseme_guided', 'gesture_animation', "
            "'gaussian_splatting'. Use ['all'] for all available."
        )
    )
    transcript: Optional[str] = Field(
        None,
        description="Optional text transcript of the audio (enables emotion detection and gesture sync)"
    )


class TextToAudioRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    voice: Optional[str] = Field(None, description="Voice profile: 'female', 'male', or 'neutral'")
    language: Optional[str] = Field(None, description="Language code (ISO 639-1, e.g., 'en', 'it', 'fr')")
    temperature: Optional[float] = Field(0.7, description="Temperature for TTS generation (0.0-1.0)", ge=0.0, le=1.0)
    cfg_weight: Optional[float] = Field(0.4, description="CFG weight for TTS generation (0.0-1.0)", ge=0.0, le=1.0)
    exaggeration: Optional[float] = Field(0.3, description="Exaggeration for TTS generation (0.0-1.0)", ge=0.0, le=1.0)
    speed: Optional[float] = Field(1.0, description="Speed for TTS generation (0.5-2.0)", ge=0.5, le=2.0)
    output_format: Optional[str] = Field("file", description="Output format: 'file' (WAV file) or 'base64' (base64-encoded WAV)")


class TextToAudioResponse(BaseModel):
    status: str = Field(..., description="Status of the request ('success' or 'error')")
    audio_path: Optional[str] = Field(None, description="Path to the generated WAV file (if output_format='file')")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV audio (if output_format='base64')")
    error: Optional[str] = Field(None, description="Error message (if status='error')")


# ───────────────────────── Thread fallback task ──────────────────────────── #
# When Celery is unavailable the pipeline runs in a FastAPI BackgroundTask.
# Completion and failure are recorded as marker files so /status can report a
# terminal state instead of hanging on "processing" forever.
def _render_video_thread(payload: dict):
    job_dir = WORK_ROOT / payload["job_id"]
    try:
        render_pipeline(
            face_image=payload["avatar_path"],
            audio=payload["audio_path"],
            reference_video=payload.get("driver_video"),
            viseme_json=payload.get("viseme_json"),
            quality_mode=payload.get("quality_mode", "auto"),
            out_path=payload["out_path"],
            enhancements=payload.get("enhancements"),
            transcript=payload.get("transcript"),
        )
        (job_dir / "done").touch()
    except Exception as exc:  # surface failures via /status instead of silently hanging
        log.exception("Render job %s failed", payload["job_id"])
        (job_dir / "error").write_text(str(exc))


# ───────────────────────────── REST endpoints ────────────────────────────── #
@app.post("/render")
def render_job(body: RenderBody, bg: BackgroundTasks):
    """Start a render job and return jobId + status URL."""
    job_id = str(uuid.uuid4())
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = job_dir / "out.mp4"

    # Resolve enhancements: per-request or default from settings
    active_enhancements = body.enhancements
    if active_enhancements is None and settings.DEFAULT_ENHANCEMENTS:
        active_enhancements = [e.strip() for e in settings.DEFAULT_ENHANCEMENTS.split(",") if e.strip()]

    payload = {
        "job_id": job_id,
        "avatar_path": body.avatarPath,
        "audio_path": body.audioPath,
        "driver_video": body.driverVideo,
        "viseme_json": body.visemeJson,
        "quality_mode": body.qualityMode,
        "out_path": str(out_mp4),
        "enhancements": active_enhancements,
        "transcript": body.transcript,
    }

    # save original request
    (job_dir / "request.json").write_text(json.dumps(body.model_dump(by_alias=True), indent=2))

    if celery_available:
        task = render_task.delay(payload)  # type: ignore[union-attr]
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
    enhancements: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
):
    """Upload avatar image + audio, start render job, return jobId + status URL.

    enhancements: Comma-separated list of enhancement names (e.g., 'emotion_expressions,eye_gaze_blink')
    transcript: Optional text transcript of the audio
    """
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

    # Parse enhancements from comma-separated form field
    active_enhancements = None
    if enhancements:
        active_enhancements = [e.strip() for e in enhancements.split(",") if e.strip()]
    elif settings.DEFAULT_ENHANCEMENTS:
        active_enhancements = [e.strip() for e in settings.DEFAULT_ENHANCEMENTS.split(",") if e.strip()]

    payload = {
        "job_id": job_id,
        "avatar_path": str(avatar_path),
        "audio_path": str(audio_path),
        "quality_mode": qualityMode,
        "out_path": str(out_mp4),
        "enhancements": active_enhancements,
        "transcript": transcript,
    }

    # save upload metadata
    (job_dir / "upload.json").write_text(
        json.dumps(
            {
                "avatar_filename": avatar.filename,
                "audio_filename": audio.filename,
                "quality_mode": qualityMode,
                "enhancements": active_enhancements,
            },
            indent=2,
        )
    )

    if celery_available:
        task = render_task.delay(payload)  # type: ignore[union-attr]
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

    # Surface a failed render rather than reporting "processing" forever.
    error_marker = job_dir / "error"
    if error_marker.exists():
        return JSONResponse({"state": "error", "detail": error_marker.read_text()}, status_code=500)

    if celery_available:
        celery_id_path = job_dir / "celery_id"
        if not celery_id_path.exists():
            raise HTTPException(500, "job metadata missing")
        task_id = celery_id_path.read_text()
        task = AsyncResult(task_id, app=celery_app)
        return {"state": task.state}

    done_marker = job_dir / "done"
    state = "finished" if done_marker.exists() else "processing"
    return {"state": state}


# ─────────────────────── Text-to-Audio endpoint ────────────────────────────── #
@app.post("/text-to-audio", response_model=TextToAudioResponse)
async def text_to_audio(body: TextToAudioRequest):
    """
    Convert text to speech using the Chatterbox TTS service.

    This endpoint generates audio from text and returns either:
    - A WAV file path (if output_format='file')
    - Base64-encoded WAV audio (if output_format='base64')

    The generated audio can be used with the avatar rendering pipeline.
    """
    if not tts_available:
        raise HTTPException(
            503,
            "TTS service is not available. Please check the Chatterbox TTS server configuration."
        )

    try:
        # Determine output format
        output_format = body.output_format or "file"

        # Get voice and language (use defaults from settings if not specified)
        voice = body.voice or settings.CHATTERBOX_DEFAULT_VOICE
        language = body.language or settings.CHATTERBOX_DEFAULT_LANGUAGE

        if output_format == "base64":
            # Return base64-encoded WAV
            audio_base64 = await tts_wav_base64_async(
                body.text,
                voice=voice,
                language=language,
                temperature=body.temperature or 0.7,
                cfg_weight=body.cfg_weight or 0.4,
                exaggeration=body.exaggeration or 0.3,
                speed=body.speed or 1.0,
            )

            return TextToAudioResponse(
                status="success",
                audio_base64=audio_base64,
            )
        else:
            # Generate WAV and save to file
            wav_bytes = await tts_wav_bytes_async(
                body.text,
                voice=voice,
                language=language,
                temperature=body.temperature or 0.7,
                cfg_weight=body.cfg_weight or 0.4,
                exaggeration=body.exaggeration or 0.3,
                speed=body.speed or 1.0,
            )

            # Save to a file in the work directory
            audio_id = str(uuid.uuid4())
            audio_dir = WORK_ROOT / f"tts-{audio_id}"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / "audio.wav"

            audio_path.write_bytes(wav_bytes)

            return TextToAudioResponse(
                status="success",
                audio_path=str(audio_path),
            )

    except ChatterboxTtsError as exc:
        return TextToAudioResponse(
            status="error",
            error=f"TTS generation failed: {str(exc)}",
        )
    except Exception as exc:
        return TextToAudioResponse(
            status="error",
            error=f"Unexpected error: {str(exc)}",
        )


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


@app.get("/health/tts")
async def tts_health():
    """Check the health of the Chatterbox TTS service."""
    if not tts_available:
        raise HTTPException(
            503,
            "TTS service is not available. TTS module could not be imported."
        )

    try:
        health_status = await chatterbox_health_async()
        return JSONResponse({
            "status": "healthy",
            "tts_server": settings.CHATTERBOX_URL,
            "details": health_status,
        })
    except ChatterboxTtsError as exc:
        raise HTTPException(
            503,
            f"TTS health check failed: {str(exc)}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            503,
            f"Unexpected TTS health check error: {str(exc)}"
        ) from exc


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

    # Enhancements status
    enhancements_info = []
    try:
        from .enhancements import registry as enhancement_registry
        enhancements_info = enhancement_registry.get_info_all()
    except ImportError:
        pass

    return JSONResponse({
        "status": "ready",
        "models": models_status,
        "system": {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpus": gpu_info,
            "celery_enabled": celery_available,
            "tts_enabled": tts_available
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
            },
            "cinematic": {
                "available": any(e["name"] == "hallo3_cinematic" and e["available"] for e in enhancements_info),
                "description": "Hallo3 Diffusion Transformer for cinematic quality (slow, GPU intensive)",
                "models": ["hallo3"]
            },
            "3d": {
                "available": any(e["name"] == "gaussian_splatting" and e["available"] for e in enhancements_info),
                "description": "InsTaG 3D Gaussian Splatting for real-time 3D avatars",
                "models": ["instag"]
            }
        },
        "enhancements": enhancements_info,
        "tts": {
            "available": tts_available,
            "server_url": settings.CHATTERBOX_URL if tts_available else None,
            "default_voice": settings.CHATTERBOX_DEFAULT_VOICE if tts_available else None,
            "default_language": settings.CHATTERBOX_DEFAULT_LANGUAGE if tts_available else None,
            "description": "Chatterbox TTS for text-to-speech synthesis"
        }
    })