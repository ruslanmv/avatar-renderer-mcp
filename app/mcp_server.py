#!/usr/bin/env python3
"""
mcp_server.py  –  STDIO server for MCP Gateway
==============================================

Exposes one tool:  render_avatar
  params:
    avatar_path   – PNG/JPG still image
    audio_path    – WAV/MP3/OGG speech
    driver_video  – optional MP4 for head‑pose (FOMM)  [default: None]
    viseme_json   – optional phoneme/viseme alignment JSON
    quality_mode  – "real_time" | "high_quality" | "auto" [default: "auto"]

Quality Modes:
  real_time     – Fast processing for streaming (SadTalker + Wav2Lip)
  high_quality  – Best quality for content (FOMM + Diff2Lip + GFPGAN)
  auto          – Automatically select based on GPU and models

The server reads line‑delimited JSON on **stdin** and writes replies on **stdout**.

Successful reply:
  { "jobId": "<uuid>", "output": "/tmp/<uuid>.mp4", "qualityMode": "real_time" }

Error reply:
  { "error": "<string>" }

You can run it locally:
  echo '{"tool":"render_avatar","params":{"avatar_path":"test.png","audio_path":"test.wav","quality_mode":"real_time"}}' | python -m app.mcp_server
"""
from __future__ import annotations

import json
import logging
import os
import queue
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from .pipeline import render_pipeline

# ───────────────────────  optional Kafka progress  ───────────────────────── #
try:
    from kafka import KafkaProducer  # type: ignore
except ModuleNotFoundError:          # Kafka is *optional*
    KafkaProducer = None  # type: ignore


# ─────────────────────────────  logging  ─────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("mcp-server")

# ─────────────────────────────  progress  ────────────────────────────────── #
def _init_progress() -> Optional[KafkaProducer]:
    brokers = os.getenv("KAFKA_BROKERS")
    if not (KafkaProducer and brokers):
        return None
    try:
        prod = KafkaProducer(
            bootstrap_servers=brokers.split(","),
            value_serializer=lambda v: json.dumps(v).encode(),
            linger_ms=500,
        )
        log.info("Kafka progress enabled -> %s", brokers)
        return prod
    except Exception as e:
        log.warning("Kafka disabled: %s", e)
        return None


progress_producer = _init_progress()
PROGRESS_TOPIC = os.getenv("PROGRESS_TOPIC", "videoStatus")


def _emit_progress(job_id: str, state: str, extra: Dict | None = None):
    if not progress_producer:
        return
    payload = {"jobId": job_id, "state": state, **(extra or {})}
    progress_producer.send(PROGRESS_TOPIC, payload)


# ───────────────────────────  worker thread  ─────────────────────────────── #
_job_q: queue.Queue[Dict] = queue.Queue()
_stop_flag = threading.Event()


def _worker_loop() -> None:
    """Consumes jobs from queue so main thread keeps STDIO responsive."""
    while not _stop_flag.is_set():
        try:
            job = _job_q.get(timeout=0.2)
        except queue.Empty:
            continue

        job_id = job["job_id"]
        quality_mode = job.get("quality_mode", "auto")
        try:
            _emit_progress(job_id, "started", {"qualityMode": quality_mode})
            result_path = render_pipeline(
                face_image=job["avatar_path"],
                audio=job["audio_path"],
                reference_video=job.get("driver_video"),
                viseme_json=job.get("viseme_json"),
                quality_mode=quality_mode,
                out_path=job["out_path"],
            )
            _emit_progress(job_id, "finished", {"output": result_path, "qualityMode": quality_mode})
            job["on_done"](
                {
                    "jobId": job_id,
                    "output": result_path,
                    "qualityMode": quality_mode,
                },
                error=None,
            )
        except Exception as exc:
            log.exception("Job %s failed", job_id)
            _emit_progress(job_id, "error", {"detail": str(exc)})
            job["on_done"](None, error=str(exc))
        finally:
            _job_q.task_done()


threading.Thread(target=_worker_loop, daemon=True).start()

# ────────────────────────  graceful shutdown  ───────────────────────────── #
def _graceful_exit(signum, _frame):
    log.info("Signal %s received – shutting down", signum)
    _stop_flag.set()


signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)

# ─────────────────────────────  helpers  ─────────────────────────────────── #
def _reply(obj: Dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle_request(req: Dict):
    if req.get("tool") != "render_avatar":
        return _reply({"error": "unknown_tool"})

    # Validate params
    for key in ("avatar_path", "audio_path"):
        if key not in req["params"]:
            return _reply({"error": f"missing_{key}"})

    job_id = str(uuid.uuid4())
    out_mp4 = req["params"].get("out_path") or f"/tmp/{job_id}.mp4"

    # Enqueue heavy work
    def _done(resp_success: Dict | None, error: str | None):
        _reply(resp_success if error is None else {"error": error})

    _job_q.put(
        {
            "job_id": job_id,
            "avatar_path": req["params"]["avatar_path"],
            "audio_path": req["params"]["audio_path"],
            "driver_video": req["params"].get("driver_video"),
            "viseme_json": req["params"].get("viseme_json"),
            "quality_mode": req["params"].get("quality_mode", "auto"),
            "out_path": out_mp4,
            "on_done": _done,
        }
    )


# ─────────────────────────────  main IO  ─────────────────────────────────── #
log.info("MCP STDIO server ready – listening for JSON on stdin")
for line in sys.stdin:
    if _stop_flag.is_set():
        break
    line = line.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
        _handle_request(request)
    except Exception as e:
        log.warning("Bad request: %s", e)
        _reply({"error": "invalid_json"})

log.info("MCP server stopped – bye")
