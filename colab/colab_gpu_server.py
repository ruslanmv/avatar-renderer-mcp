"""
colab_gpu_server.py — Temporary, token-authenticated GPU job server for Colab.

Design goals (see docs/COLAB_GPU_TESTING.md):
  • NO SSH, NO arbitrary shell. Only the allowlisted actions below.
  • Every request must carry the X-Colab-Token header (rotated per session).
  • Long actions (setup / render) run as background JOBS; clients poll /jobs/{id}
    so the HTTP request returns instantly and never trips a tunnel timeout.
  • Renders go through the production app.render.orchestrate via colab/render_one.py
    — so what passes here is what ships. Premium tiers never silently downgrade.

Endpoints
  GET  /ping                 (no auth) liveness for the tunnel
  GET  /health               server + GPU + repo summary
  GET  /engines              engine-registry availability + licenses
  POST /git/pull             {branch}                       → job
  POST /gpu/check                                           → job (nvidia-smi/torch)
  POST /setup                {engines:[musetalk,diff2lip,...]} → job (clone + weights)
  POST /render/sample        {engine,quality_mode,text|use_sample,...} → job
  GET  /jobs/{id}            status + log tail (+ artifact url when done)
  GET  /jobs                 recent jobs
  GET  /artifact/{name}      download a produced .mp4 (sanitized name)
  GET  /logs/{name}          tail a log file (sanitized name)
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# python-multipart is only needed for the optional file-upload endpoint. FastAPI
# raises at import time if you register a File()/Form() route without it, so we
# register /upload conditionally — the server still boots if it's missing.
try:
    import multipart  # noqa: F401  (python-multipart)
    _HAS_MULTIPART = True
except Exception:
    _HAS_MULTIPART = False

# ── Config (from env, set by the notebook) ───────────────────────────────────
REPO_DIR = Path(os.environ.get("REPO_DIR", "/content/work/avatar-renderer-mcp"))
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/content/artifacts"))
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/content/inputs"))
API_TOKEN = os.environ.get("COLAB_GPU_TOKEN", "")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Validation allowlists (defence-in-depth; engines also re-checked at run time).
_QUALITY = {"preview", "real_time", "standard", "high_quality", "premium", "cinematic", "auto"}
_KNOWN_ENGINES = {"auto", "simple", "wav2lip_fast", "wav2lip_raw", "wav2lip_band",
                  "fullface", "wav2lip", "diff2lip", "musetalk", "latentsync"}
_SETUP_ALLOWED = {"musetalk", "diff2lip", "latentsync", "liveportrait",
                  "emotion", "eye_gaze", "hallo3", "cosyvoice", "viseme", "gesture"}

app = FastAPI(title="Avatar Renderer — Temporary Colab GPU Runner")

# ── Job registry (in-memory; this server is ephemeral) ───────────────────────
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _auth(token: Optional[str]) -> None:
    if not API_TOKEN:
        raise HTTPException(500, "COLAB_GPU_TOKEN not configured on the server")
    if not token or not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(401, "invalid or missing X-Colab-Token")


def _safe_name(name: str) -> str:
    """Reject path traversal — only a bare filename within our dirs is allowed."""
    n = Path(name).name
    if not n or n != name:
        raise HTTPException(400, "invalid name")
    return n


def _new_job(kind: str) -> dict:
    jid = uuid.uuid4().hex
    log = ARTIFACT_DIR / f"{kind}_{jid}.log"
    job = {"job_id": jid, "kind": kind, "status": "running", "returncode": None,
           "started": time.time(), "ended": None, "log": str(log),
           "artifact": None, "artifact_url": None}
    with _JOBS_LOCK:
        _JOBS[jid] = job
    return job


def _run_argv_async(job: dict, argv: list[str], *, cwd: Path = REPO_DIR,
                    timeout: int = 1800, env: Optional[dict] = None,
                    on_success=None) -> None:
    """Run a FIXED argv (no shell) in a background thread, streaming to job log."""
    def _target() -> None:
        log_path = Path(job["log"])
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(argv) + "\n\n")
                log.flush()
                proc = subprocess.run(
                    argv, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout,
                    env={**os.environ, **(env or {})},
                )
            job["returncode"] = proc.returncode
            if proc.returncode == 0 and on_success:
                on_success(job)
        except subprocess.TimeoutExpired:
            job["returncode"] = -1
            with log_path.open("a") as log:
                log.write(f"\n[timeout after {timeout}s]\n")
        except Exception as exc:  # pragma: no cover - defensive
            job["returncode"] = -2
            with log_path.open("a") as log:
                log.write(f"\n[server error] {exc}\n")
        finally:
            job["ended"] = time.time()
            job["status"] = "done" if job["returncode"] == 0 else "error"

    threading.Thread(target=_target, daemon=True).start()


def _job_view(job: dict, tail: int = 4000) -> dict:
    out = {k: job[k] for k in ("job_id", "kind", "status", "returncode",
                               "artifact", "artifact_url")}
    out["seconds"] = round((job["ended"] or time.time()) - job["started"], 1)
    log_path = Path(job["log"])
    out["log_tail"] = (log_path.read_text(errors="replace")[-tail:]
                       if log_path.exists() else "")
    return out


# ── Request models ───────────────────────────────────────────────────────────
class PullReq(BaseModel):
    branch: str = "main"


class SetupReq(BaseModel):
    engines: list[str] = ["musetalk", "diff2lip"]
    core_weights: bool = True  # FOMM/Wav2Lip/Diff2Lip/GFPGAN from the model repo


class RenderReq(BaseModel):
    engine: str = "auto"
    quality_mode: str = "standard"
    text: str = "Hello! This avatar is rendering on a Colab GPU."
    voice: str = "en-US-AriaNeural"
    use_sample: bool = True            # use the bundled portrait
    image_path: Optional[str] = None   # else a file already inside INPUT_DIR
    audio_path: Optional[str] = None   # else TTS from `text`
    commercial: bool = False
    timeout_sec: int = 1500


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"pong": True, "service": "avatar-renderer-colab-gpu"}


@app.get("/health")
def health(x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        vram = (round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
                if cuda else None)
        tv = torch.__version__
    except Exception as exc:
        cuda, gpu, vram, tv = False, None, None, f"torch import failed: {exc}"
    return {"ok": True, "repo_dir": str(REPO_DIR), "repo_exists": REPO_DIR.exists(),
            "artifact_dir": str(ARTIFACT_DIR), "cuda": cuda, "gpu": gpu,
            "vram_gb": vram, "torch": tv, "jobs": len(_JOBS)}


@app.get("/engines")
def engines(x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    # Import lazily with the engine paths set so probes are accurate.
    sys.path.insert(0, str(REPO_DIR))
    os.environ.setdefault("MODEL_ROOT", str(REPO_DIR / "models"))
    os.environ.setdefault("EXT_DEPS_DIR", str(REPO_DIR / "external_deps"))
    try:
        import importlib
        import app.compliance as _c
        import app.engines as _e
        importlib.reload(_c)
        importlib.reload(_e)
        rows = _e.registry.info_all()
    except Exception as exc:
        raise HTTPException(500, f"engine registry import failed: {exc}")
    return {"available": [r["name"] for r in rows if r["available"]], "engines": rows}


@app.post("/git/pull")
def git_pull(req: PullReq, x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    if not (REPO_DIR / ".git").exists():
        raise HTTPException(500, f"repo not found at {REPO_DIR}")
    job = _new_job("git_pull")
    # Fixed argv chain via bash -c is avoided; run a tiny git script with set args.
    argv = ["git", "-C", str(REPO_DIR), "fetch", "--all"]
    # Chain checkout+pull on success by sequencing in the thread.

    def _seq(job: dict) -> None:
        for cmd in (["git", "-C", str(REPO_DIR), "checkout", req.branch],
                    ["git", "-C", str(REPO_DIR), "pull", "origin", req.branch]):
            with Path(job["log"]).open("a") as log:
                log.write("\n$ " + " ".join(cmd) + "\n")
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True).returncode
            if rc != 0:
                job["returncode"] = rc
                return
        job["returncode"] = 0

    _run_argv_async(job, argv, timeout=300, on_success=_seq)
    return {"job_id": job["job_id"], "status": "running"}


@app.post("/gpu/check")
def gpu_check(x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    job = _new_job("gpu_check")
    code = ("import torch,subprocess;"
            "print('CUDA:',torch.cuda.is_available());"
            "print('torch:',torch.__version__,'cuda',torch.version.cuda);"
            "print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);"
            "subprocess.run(['nvidia-smi'])")
    _run_argv_async(job, [sys.executable, "-c", code], timeout=120)
    return {"job_id": job["job_id"], "status": "running"}


@app.post("/setup")
def setup(req: SetupReq, x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    bad = [e for e in req.engines if e not in _SETUP_ALLOWED]
    if bad:
        raise HTTPException(400, f"engines not allowed for setup: {bad}")
    job = _new_job("setup")
    script = REPO_DIR / "scripts" / "colab_gpu_setup.sh"
    if not script.exists():
        raise HTTPException(500, "scripts/colab_gpu_setup.sh missing — pull the branch first")
    argv = ["bash", str(script), "1" if req.core_weights else "0", *req.engines]
    _run_argv_async(job, argv, timeout=2400)
    return {"job_id": job["job_id"], "status": "running"}


def _tts(text: str, voice: str) -> str:
    import asyncio
    import edge_tts
    out = str(INPUT_DIR / f"tts_{uuid.uuid4().hex[:8]}.mp3")

    async def _go():
        await edge_tts.Communicate(text, voice).save(out)

    asyncio.new_event_loop().run_until_complete(_go())
    return out


def _resolve_inputs(req: RenderReq) -> tuple[str, str]:
    # Image
    if req.image_path:
        img = INPUT_DIR / _safe_name(req.image_path)
        if not img.exists():
            raise HTTPException(400, f"image not in INPUT_DIR: {img.name}")
        image = str(img)
    elif req.use_sample:
        cands = [REPO_DIR / "zerogpu/assets/alice.png",
                 REPO_DIR / "tests/assets/alice.png"]
        image = next((str(c) for c in cands if c.exists()), None)
        if not image:
            raise HTTPException(500, "no bundled sample portrait found")
    else:
        raise HTTPException(400, "provide image_path (in INPUT_DIR) or set use_sample=true")
    # Audio
    if req.audio_path:
        aud = INPUT_DIR / _safe_name(req.audio_path)
        if not aud.exists():
            raise HTTPException(400, f"audio not in INPUT_DIR: {aud.name}")
        audio = str(aud)
    else:
        audio = _tts(req.text, req.voice)
    return image, audio


@app.post("/render/sample")
def render_sample(req: RenderReq, x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    if req.engine not in _KNOWN_ENGINES:
        raise HTTPException(400, f"unknown engine '{req.engine}'")
    if req.quality_mode not in _QUALITY:
        raise HTTPException(400, f"unknown quality_mode '{req.quality_mode}'")

    image, audio = _resolve_inputs(req)
    out_name = f"render_{req.engine}_{uuid.uuid4().hex[:8]}.mp4"
    out_path = ARTIFACT_DIR / out_name

    job = _new_job("render")
    argv = [sys.executable, str(REPO_DIR / "colab" / "render_one.py"),
            "--image", image, "--audio", audio, "--out", str(out_path),
            "--engine", req.engine, "--quality", req.quality_mode]
    if req.commercial:
        argv.append("--commercial")

    def _on_ok(job: dict) -> None:
        if out_path.exists():
            job["artifact"] = out_name
            job["artifact_url"] = f"/artifact/{out_name}"

    _run_argv_async(job, argv, timeout=req.timeout_sec, on_success=_on_ok)
    return {"job_id": job["job_id"], "status": "running", "expected_artifact": out_name}


if _HAS_MULTIPART:
    from fastapi import File, UploadFile

    @app.post("/upload")
    async def upload(file: UploadFile = File(...),
                     x_colab_token: Optional[str] = Header(None)):
        _auth(x_colab_token)
        name = _safe_name(file.filename or f"upload_{uuid.uuid4().hex[:6]}")
        dst = INPUT_DIR / name
        dst.write_bytes(await file.read())
        return {"ok": True, "name": name, "bytes": dst.stat().st_size}


@app.get("/jobs/{jid}")
def job_status(jid: str, x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    job = _JOBS.get(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return _job_view(job)


@app.get("/jobs")
def jobs(x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    items = sorted(_JOBS.values(), key=lambda j: j["started"], reverse=True)[:20]
    return {"jobs": [{k: j[k] for k in ("job_id", "kind", "status", "returncode")}
                     for j in items]}


@app.get("/artifact/{name}")
def artifact(name: str, x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    path = ARTIFACT_DIR / _safe_name(name)
    if not path.exists():
        raise HTTPException(404, "artifact not found")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


@app.get("/logs/{name}")
def logs(name: str, x_colab_token: Optional[str] = Header(None)):
    _auth(x_colab_token)
    path = ARTIFACT_DIR / _safe_name(name)
    if not path.exists() and not path.suffix:
        path = ARTIFACT_DIR / f"{_safe_name(name)}.log"
    if not path.exists():
        raise HTTPException(404, "log not found")
    return JSONResponse({"name": path.name, "text": path.read_text(errors="replace")[-12000:]})
