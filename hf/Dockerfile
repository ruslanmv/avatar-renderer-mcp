# =============================================================================
# Avatar Renderer MCP — Hugging Face GPU Docker Space (dev-v0.1.25 architecture)
# =============================================================================
# Full pipeline: FOMM → Diff2Lip/MuseTalk/LatentSync → Wav2Lip fallback → GFPGAN.
# Requires a GPU runtime (attach T4/A10G in the Space hardware settings). On CPU
# it builds and serves but the heavy engines are slow/limited.
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for OpenCV / audio / ffmpeg / pyaudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git wget curl ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libsndfile1 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# HF Spaces run as uid 1000.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /data /tmp/avatar-jobs /tmp/avatar-cache && \
    chown -R appuser:appuser /app /data /tmp/avatar-jobs /tmp/avatar-cache

WORKDIR /app

# Install the package (full ML stack) from pyproject. Cache layer: deps first.
COPY --chown=appuser:appuser pyproject.toml README.md LICENSE ./
COPY --chown=appuser:appuser app/ app/
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install .

# Runtime assets + startup script.
COPY --chown=appuser:appuser start.sh /app/start.sh
COPY --chown=appuser:appuser static /app/static
COPY --chown=appuser:appuser scripts/download_models.sh /app/scripts/download_models.sh
RUN chmod +x /app/start.sh && \
    mkdir -p /app/models /app/external_deps && \
    chown -R appuser:appuser /app

# RENDER_BACKEND=auto → start.sh clones engine repos, downloads weights, and the
# tier-driven renderer prefers the full pipeline (Diff2Lip primary, Wav2Lip fallback).
ENV MODEL_ROOT=/app/models \
    EXT_DEPS_DIR=/app/external_deps \
    RENDER_BACKEND=auto \
    PORT=7860 \
    HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=1200s --retries=5 \
    CMD curl -f http://localhost:7860/health/live || exit 1

CMD ["bash", "/app/start.sh"]
