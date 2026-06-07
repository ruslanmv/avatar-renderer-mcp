"""
worker.py  – Celery worker for the Avatar Renderer
==================================================

* Listens on the *render* queue (broker URL taken from Settings).
* Consumes JSON payloads produced by the FastAPI API or the MCP server.
* Executes the FOMM + Diff2Lip (or Wav2Lip) pipeline (+ optional enhancements).
* Optionally uploads the resulting MP4 to S3-compatible Cloud Object Storage.

This module is the **single source of truth** for the Celery app and task:
``app.api`` imports ``celery_app`` and ``render_task`` from here so that the API
and the worker process agree on the task name, queue, and result backend.

The module is **import-safe**: importing it does *not* spin up a worker.
Only ``celery -A app.worker worker …`` or ``python -m app.worker`` does.
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

from celery import Celery, signals

from .render import render as run_render
from .settings import Settings

logger = logging.getLogger("avatar-renderer.worker")

# --------------------------------------------------------------------------- #
# Celery setup                                                                #
# --------------------------------------------------------------------------- #

settings = Settings()  # loads env vars once

celery_app = Celery(
    "avatar_renderer",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_BACKEND_URL or settings.CELERY_BROKER_URL or "rpc://",
)

celery_app.conf.update(
    task_default_queue="render",
    worker_prefetch_multiplier=1,  # GPU → one task per worker process
    task_acks_late=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_concurrency=settings.CELERY_CONCURRENCY,
)

# Backwards-compatible alias (older imports referenced ``app``).
app = celery_app


# --------------------------------------------------------------------------- #
# Celery task                                                                 #
# --------------------------------------------------------------------------- #

@celery_app.task(name="render_video_task", queue="render")
def render_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Render an avatar video from a job payload.

    Expected payload (same shape produced by the API and MCP server)::

        {
          "job_id":       "<uuid>",                 # optional
          "avatar_path":  "/path/alice.png",
          "audio_path":   "/path/hello.wav",
          "driver_video": "/path/ref.mp4",          # optional
          "viseme_json":  "/path/visemes.json",     # optional
          "quality_mode": "auto",
          "enhancements": ["emotion_expressions"],  # optional
          "transcript":   "hello world",            # optional
          "out_path":     "/tmp/<uuid>.mp4",
        }
    """
    job_id = payload.get("job_id") or str(uuid.uuid4())
    logger.info("Job %s – start render", job_id)

    out_mp4 = payload.get("out_path") or payload.get("output") or f"/tmp/{job_id}.mp4"
    viseme_json = payload.get("viseme_json")

    # If a transcript was supplied without explicit visemes, derive them.
    transcript = payload.get("transcript")
    if viseme_json is None and transcript:
        viseme_json = _maybe_build_visemes(payload["audio_path"], transcript)

    run_render(
        face_image=payload["avatar_path"],
        audio=payload["audio_path"],
        reference_video=payload.get("driver_video"),
        viseme_json=viseme_json,
        quality_mode=payload.get("quality_mode", "auto"),
        out_path=out_mp4,
        enhancements=payload.get("enhancements"),
        transcript=transcript,
    )

    result: Dict[str, Any] = {"job_id": job_id, "output": out_mp4}

    # Optional Cloud Object Storage upload (only when a bucket is configured).
    if settings.COS_BUCKET:
        try:
            try:
                from scripts.cos_utils import upload_to_cos
            except ImportError:  # repo-root not on path → load by file location
                import importlib.util

                cos_path = Path(__file__).resolve().parent.parent / "scripts" / "cos_utils.py"
                spec = importlib.util.spec_from_file_location("cos_utils", cos_path)
                module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                upload_to_cos = module.upload_to_cos

            cos_url = upload_to_cos(out_mp4, settings, key=f"{job_id}.mp4")
            result["cos_url"] = cos_url
            logger.info("Job %s – uploaded to %s", job_id, cos_url)
        except Exception as exc:  # upload failure must not lose the local result
            logger.warning("Job %s – COS upload failed: %s", job_id, exc)

    logger.info("Job %s – DONE", job_id)
    return result


def _maybe_build_visemes(audio_path: str, transcript: str) -> str | None:
    """Best-effort viseme alignment serialized to a temp JSON file (path)."""
    try:
        from .viseme_align import get_viseme_json  # lazy: pulls librosa

        visemes = get_viseme_json(audio_path, transcript)
        tmp = Path(tempfile.mkdtemp(prefix="visemes_")) / "visemes.json"
        tmp.write_text(json.dumps(visemes))
        return str(tmp)
    except Exception as exc:  # alignment is optional; never block rendering
        logger.warning("Viseme alignment skipped: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Graceful GPU configuration                                                  #
# --------------------------------------------------------------------------- #

@signals.worker_process_init.connect
def _configure_cuda(**_: Any) -> None:
    """Log GPU visibility restrictions applied by the orchestrator (k8s, etc.)."""
    import os

    gpu = os.getenv("NVIDIA_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES")
    if gpu:
        logger.info("CUDA restricted to device(s): %s", gpu)


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Equivalent to: celery -A app.worker worker -l info -Q render
    celery_app.worker_main(
        argv=[
            "worker",
            "--loglevel=INFO",
            "--queues=render",
            f"--concurrency={settings.CELERY_CONCURRENCY}",
            "--hostname=renderer@%h",
        ]
    )
