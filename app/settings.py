"""Application settings and configuration management.

This module provides centralized runtime configuration for the Avatar Renderer MCP
using Pydantic Settings for robust environment variable management.

Configuration sources (in order of precedence):
    1. Environment variables
    2. .env file (auto-loaded if present)
    3. Default values specified in field definitions

Author:
    Ruslan Magana Vsevolodovna

Website:
    https://ruslanmv.com

License:
    Apache-2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module-level logger
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dynamic Path Calculation
# -----------------------------------------------------------------------------
# Resolves to the project root (e.g., /mnt/c/blog/avatar-renderer-mcp)
# __file__ = app/settings.py -> parent = app -> parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide configuration settings.

    Attributes:
        LOG_LEVEL: Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        TMP_DIR: Temporary directory for intermediate processing files.
        CUDA_VISIBLE_DEVICES: GPU device IDs to use (e.g., "0" or "0,1").
        TORCH_USE_INT8: Whether to enable INT8 quantization for inference.
        CELERY_BROKER_URL: Celery message broker URL (Redis/RabbitMQ).
        CELERY_BACKEND_URL: Celery result backend URL.
        CELERY_CONCURRENCY: Number of concurrent Celery worker processes.
        MODEL_ROOT: Root directory containing all model checkpoints.
        FFMPEG_BIN: Path to FFmpeg binary for video encoding.
        MCP_ENABLE: Whether to enable the MCP STDIO server.
        MCP_TOOL_NAME: Tool name advertised to MCP Gateway.
        CHATTERBOX_URL: URL of the Chatterbox TTS server.
        CHATTERBOX_TIMEOUT: Timeout for Chatterbox TTS requests in seconds.
        CHATTERBOX_DEFAULT_VOICE: Default voice for TTS (female, male, neutral).
        CHATTERBOX_DEFAULT_LANGUAGE: Default language for TTS (ISO 639-1 code).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # General Settings
    # ─────────────────────────────────────────────────────────────────────────

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    TMP_DIR: Path = Field(
        default=Path("/tmp"),
        description="Scratch directory for job processing and temporary files",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # GPU / CUDA Configuration
    # ─────────────────────────────────────────────────────────────────────────

    CUDA_VISIBLE_DEVICES: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated GPU device IDs (e.g., '0' or '0,1'). "
            "If set, restricts GPU visibility to specified devices."
        ),
    )

    TORCH_USE_INT8: bool = Field(
        default=False,
        description="Enable INT8 quantization for supported models to reduce memory usage",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Celery Task Queue Configuration
    # ─────────────────────────────────────────────────────────────────────────

    CELERY_BROKER_URL: Optional[str] = Field(
        default=None,
        description=(
            "Celery message broker URL (e.g., 'redis://redis:6379/0'). "
            "If empty, the system operates in local thread mode without Celery."
        ),
    )

    CELERY_BACKEND_URL: Optional[str] = Field(
        default=None,
        description="Celery result backend URL. Defaults to broker URL if not specified.",
    )

    CELERY_CONCURRENCY: int = Field(
        default=1,
        description=(
            "Number of concurrent Celery worker processes. "
            "For GPU workloads, typically set to 1 per GPU."
        ),
        ge=1,
        le=128,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Model Checkpoint Paths
    # ─────────────────────────────────────────────────────────────────────────

    # FIX: Use dynamic project root instead of hardcoded absolute path
    MODEL_ROOT: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "models",
        description="Root directory containing all model checkpoints",
    )

    # Properties are used for sub-paths to ensure they always follow MODEL_ROOT
    @property
    def FOMM_CKPT_DIR(self) -> Path:
        """Directory for FOMM (First Order Motion Model) checkpoints."""
        return self.MODEL_ROOT / "fomm"

    @property
    def DIFF2LIP_CKPT_DIR(self) -> Path:
        """Directory for Diff2Lip diffusion model checkpoints."""
        return self.MODEL_ROOT / "diff2lip"

    @property
    def SADTALKER_CKPT_DIR(self) -> Path:
        """Directory for SadTalker model checkpoints."""
        return self.MODEL_ROOT / "sadtalker"

    @property
    def WAV2LIP_CKPT(self) -> Path:
        """Path to Wav2Lip GAN checkpoint file."""
        return self.MODEL_ROOT / "wav2lip" / "wav2lip_gan.pth"

    @property
    def GFPGAN_CKPT(self) -> Path:
        """Path to GFPGAN face enhancement checkpoint file."""
        return self.MODEL_ROOT / "gfpgan" / "GFPGANv1.3.pth"

    # ─────────────────────────────────────────────────────────────────────────
    # FFmpeg Configuration
    # ─────────────────────────────────────────────────────────────────────────

    FFMPEG_BIN: str = Field(
        default="ffmpeg",
        description=(
            "Path or command name for the FFmpeg binary. "
            "Must support NVENC for GPU-accelerated video encoding."
        ),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MCP (Model Context Protocol) Integration
    # ─────────────────────────────────────────────────────────────────────────

    MCP_ENABLE: bool = Field(
        default=True,
        description="Whether to enable the MCP STDIO server for AI agent integration",
    )

    MCP_TOOL_NAME: str = Field(
        default="avatar_renderer",
        description="Tool name advertised to the MCP Gateway for service discovery",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TTS (Text-to-Speech) Configuration
    # ─────────────────────────────────────────────────────────────────────────

    CHATTERBOX_URL: str = Field(
        default="http://localhost:4123",
        description="URL of the Chatterbox TTS server for text-to-speech synthesis",
    )

    CHATTERBOX_TIMEOUT: float = Field(
        default=30.0,
        description="Timeout in seconds for Chatterbox TTS requests",
        ge=1.0,
        le=300.0,
    )

    CHATTERBOX_DEFAULT_VOICE: str = Field(
        default="female",
        description="Default voice for TTS synthesis (female, male, or neutral)",
    )

    CHATTERBOX_DEFAULT_LANGUAGE: str = Field(
        default="en",
        description="Default language for TTS synthesis (ISO 639-1 code, e.g., 'en', 'it', 'fr')",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Field Validators
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("LOG_LEVEL", mode="after")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate and normalize the log level."""
        value = value.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in valid_levels:
            msg = (
                f"Invalid log level: {value}. "
                f"Must be one of: {', '.join(sorted(valid_levels))}"
            )
            raise ValueError(msg)
        return value

    # ─────────────────────────────────────────────────────────────────────────
    # Pydantic Settings Configuration
    # ─────────────────────────────────────────────────────────────────────────

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown environment variables
        validate_default=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.configure_logging()
        self._validate_model_paths()

    def configure_logging(self) -> None:
        """Configure application-wide logging based on LOG_LEVEL setting."""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Prevent double logging if re-initialized
        logger.setLevel(getattr(logging, self.LOG_LEVEL))

    def _validate_model_paths(self):
        """Validate that model checkpoint paths exist at runtime."""
        checks = [
            ("FOMM_CKPT_DIR", self.FOMM_CKPT_DIR),
            ("DIFF2LIP_CKPT_DIR", self.DIFF2LIP_CKPT_DIR),
            ("SADTALKER_CKPT_DIR", self.SADTALKER_CKPT_DIR),
            ("WAV2LIP_CKPT", self.WAV2LIP_CKPT),
            ("GFPGAN_CKPT", self.GFPGAN_CKPT),
        ]

        for name, path in checks:
            if not path.exists():
                logger.warning(
                    "Model path '%s' (%s) does not exist at initialization. "
                    "The path will be checked again at runtime.",
                    name,
                    path,
                )
            else:
                logger.info("Model found: %s -> %s", name, path)


# ─────────────────────────────────────────────────────────────────────────────
# Global Settings Instance
# ─────────────────────────────────────────────────────────────────────────────

# Singleton settings instance used throughout the application
settings = Settings()