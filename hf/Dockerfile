# =============================================================================
# Avatar Renderer MCP — Hugging Face Spaces Dockerfile
# =============================================================================
# Ultra-lean build following ollabridge-cloud pattern exactly.
# ALL heavy deps (PyTorch, ML libs) installed at STARTUP, not build.
# Build only installs FastAPI + lightweight deps (~30s).
# =============================================================================

FROM python:3.11-slim

# -- Minimal system deps (identical to ollabridge) ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        procps \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -- Create app user (HF Spaces runs as uid 1000) ----------------------------
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /tmp/avatar-jobs /tmp/avatar-cache && \
    chown -R appuser:appuser /app /tmp/avatar-jobs /tmp/avatar-cache

WORKDIR /app

# -- Copy dependency spec first (cache layer) ---------------------------------
COPY pyproject.toml ./

# -- Install ONLY lightweight deps at build time ------------------------------
# Heavy packages (torch, transformers, opencv, etc.) are installed at startup.
# This keeps the Docker build under 3 minutes (like ollabridge).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "fastapi>=0.116.1,<0.117.0" \
        "uvicorn[standard]>=0.23.2,<0.24.0" \
        "python-multipart>=0.0.5,<0.1.0" \
        "pydantic>=2.11.1,<3.0.0" \
        "pydantic-settings>=2.10.1,<3.0.0" \
        "python-dotenv>=1.0.0,<2.0.0" \
        "PyYAML>=6.0.0,<7.0.0" \
        "tqdm>=4.65.0,<5.0.0" \
        "click>=8.1.0,<9.0.0" \
        "rich>=13.0.0,<14.0.0" \
        "requests>=2.32.0" \
        "huggingface-hub>=0.20.0"

# -- Copy application code ----------------------------------------------------
COPY . .

# -- Startup script -----------------------------------------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# -- Model directories --------------------------------------------------------
RUN mkdir -p /app/models/fomm /app/models/wav2lip /app/models/diff2lip \
             /app/models/gfpgan /app/models/sadtalker

# -- Environment defaults for HF Spaces --------------------------------------
ENV MODEL_ROOT=/app/models \
    EXT_DEPS_DIR=/app/external_deps \
    PYTHONPATH="/app" \
    PORT=7860 \
    HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

RUN chown -R appuser:appuser /app /tmp/avatar-jobs /tmp/avatar-cache

# Switch to non-root user
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7860/health/live || exit 1

CMD ["/app/start.sh"]
