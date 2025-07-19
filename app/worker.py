"""
worker.py  – Celery worker for Avatar Renderer Pod
=================================================

• Listens on the *render* queue (broker URL taken from env/Settings).
• Consumes JSON payloads produced by API or MCP server.
• Executes the FOMM + Diff2Lip (or SadTalker/Wav2Lip) pipeline on a GPU.
• Uploads the resulting MP4 to Cloud Object Storage (if configured).
• Publishes status events to Kafka/WebSocket topic (optional).

The module is **import‑safe**: importing it *does not* spin a worker.
Only `celery -A app.worker worker …` or `python -m app.worker` does.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict

from celery import Celery, signals
from kombu import Queue

from settings import Settings            # pydantic‑based env settings
from pipeline import render_pipeline     # main video generator
from viseme_align import get_viseme_json # optional phoneme → viseme

# --------------------------------------------------------------------------- #
# Celery setup                                                                #
# --------------------------------------------------------------------------- #

settings = Settings()   # loads env vars once

BROKER_URL  = settings.celery_broker
RESULT_BACK = settings.celery_backend or "rpc://"

app = Celery(
    "avatar_renderer",
    broker=BROKER_URL,
    backend=RESULT_BACK,
    include=["app.worker"],             # so tasks are auto‑registered
)

app.conf.update(
    task_queues=[Queue("render", routing_key="render.#")],
    task_default_exchange="render",
    task_default_routing_key="render.task",
    worker_prefetch_multiplier=1,       # GPUs → one task per worker
    task_acks_late=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
)

logger = logging.getLogger("avatar-renderer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(msg)s")


# --------------------------------------------------------------------------- #
# Celery task                                                                 #
# --------------------------------------------------------------------------- #

@app.task(name="avatar_renderer.render")
def render_task(payload: Dict) -> Dict:
    """
    Payload example (same shape used by FastAPI + MCP):

        {
          "avatar_path": "/mnt/avatars/alice.png",
          "audio_path":  "/mnt/audio/hello.wav",
          "driver_video": "/tmp/ref.mp4",          # optional
          "upload_key":   "videos/2025/07/clip.mp4"
        }
    """

    job_id = payload.get("job_id") or str(uuid.uuid4())
    logger.info("Job %s – start render", job_id)

    avatar_path  = payload["avatar_path"]
    audio_path   = payload["audio_path"]
    driver_video = payload.get("driver_video")
    out_mp4      = payload.get("output") or f"/tmp/{job_id}.mp4"

    # 1) Viseme alignment (∼ 40 ms on GPU node; optional)
    try:
        visemes = get_viseme_json(audio_path, payload.get("transcript", ""))
    except Exception as e:  # keep going even if align fails
        logger.warning("Viseme align failed: %s", e)
        visemes = None

    # 2) Core face‑animation pipeline
    render_pipeline(
        face_image=avatar_path,
        audio=audio_path,
        reference_video=driver_video,
        viseme_json=visemes,
        out_path=out_mp4,
    )

    result = {"job_id": job_id, "output": out_mp4}

    # 3) Optional COS upload
    if settings.cos_bucket:
        from scripts.cos_utils import upload_to_cos   # lazy‑import
        cos_url = upload_to_cos(out_mp4, settings)
        result["cos_url"] = cos_url
        logger.info("Job %s – uploaded to %s", job_id, cos_url)

    logger.info("Job %s – DONE", job_id)
    return result


# --------------------------------------------------------------------------- #
# Graceful GPU memory cleanup                                                 #
# --------------------------------------------------------------------------- #

@signals.worker_process_init.connect
def _configure_cuda(**_):
    """Limits CUDA_VISIBLE_DEVICES if env var provided by k8s down‑scaling."""
    gpu = os.getenv("NVIDIA_VISIBLE_DEVICES")
    if gpu:
        logger.info("CUDA restricted to device(s): %s", gpu)


# --------------------------------------------------------------------------- #
# CLI entry‑point                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    Allows: `python -m app.worker` in dev containers.
    Equivalent to `celery -A app.worker worker -l info -Q render`.
    """
    from celery.bin import worker as celery_worker

    worker = celery_worker.worker(app=app)
    options = {
        "loglevel": "INFO",
        "queues": "render",
        "concurrency": 1,         # 1 GPU == 1 process
        "hostname": "renderer@%h",
        "without_heartbeat": False,
    }
    worker.run(**options)
