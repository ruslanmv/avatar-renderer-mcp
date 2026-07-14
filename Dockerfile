# Dockerfile
# Avatar Renderer Pod: FOMM + Diff2Lip pipeline with MCP & REST interfaces

# Base image with CUDA 12.4 runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create venv
RUN python3.11 -m venv .venv && \
    .venv/bin/pip install --upgrade pip setuptools wheel

# Copy project metadata + source, then install from pyproject.toml (no
# requirements.txt in this repo). README/LICENSE are referenced by the metadata.
COPY pyproject.toml README.md LICENSE ./
COPY app/ app/
RUN .venv/bin/pip install --no-cache-dir .

# Remaining runtime assets
COPY scripts/download_models.sh scripts/download_models.sh
COPY charts charts
COPY k8s k8s
COPY mcp-tool.json .

# Ensure venv is used by default
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Expose REST port
EXPOSE 8080

# Default command: start both REST API and MCP STDIO server under a process manager
# For simplicity, we launch REST by default; MCP can be enabled by env VAR
CMD ["bash", "-lc", "\
  if [ \"$MCP_ENABLE\" = \"true\" ]; then \
    exec python -m app.mcp_server; \
  else \
    exec uvicorn app.api:app --host 0.0.0.0 --port 8080; \
  fi\
"]
