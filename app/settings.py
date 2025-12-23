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

Examples:
    >>> # Load settings from environment
    >>> settings = Settings()
    >>> print(settings.MODEL_ROOT)
    PosixPath('/models')

    >>> # Override settings programmatically (useful for testing)
    >>> test_settings = Settings(CELERY_BROKER_URL="redis://localhost:6379/1")
    >>> print(test_settings.CELERY_BROKER_URL)
    redis://localhost:6379/1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module-level logger
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application-wide configuration settings.

    This class defines all configurable parameters for the avatar rendering system,
    including paths to model checkpoints, GPU settings, Celery configuration,
    and MCP integration options.

    All settings can be overridden via environment variables. The naming convention
    is case-insensitive, so both LOG_LEVEL and log_level will work.

    Attributes:
        LOG_LEVEL: Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        TMP_DIR: Temporary directory for intermediate processing files.
        CUDA_VISIBLE_DEVICES: GPU device IDs to use (e.g., "0" or "0,1").
        TORCH_USE_INT8: Whether to enable INT8 quantization for inference.
        CELERY_BROKER_URL: Celery message broker URL (Redis/RabbitMQ).
        CELERY_BACKEND_URL: Celery result backend URL.
        CELERY_CONCURRENCY: Number of concurrent Celery worker processes.
        MODEL_ROOT: Root directory containing all model checkpoints.
        FOMM_CKPT_DIR: Directory for FOMM model checkpoints.
        DIFF2LIP_CKPT_DIR: Directory for Diff2Lip model checkpoints.
        SADTALKER_CKPT_DIR: Directory for SadTalker model checkpoints.
        WAV2LIP_CKPT: Path to Wav2Lip model checkpoint file.
        GFPGAN_CKPT: Path to GFPGAN face enhancement model checkpoint.
        FFMPEG_BIN: Path to FFmpeg binary for video encoding.
        MCP_ENABLE: Whether to enable the MCP STDIO server.
        MCP_TOOL_NAME: Tool name advertised to MCP Gateway.
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

    MODEL_ROOT: Path = Field(
        default=Path("/models"),
        description="Root directory containing all model checkpoints (read-only mount in K8s)",
    )

    FOMM_CKPT_DIR: Path = Field(
        default=Path("/models/fomm"),
        description="Directory containing FOMM (First Order Motion Model) checkpoints",
    )

    DIFF2LIP_CKPT_DIR: Path = Field(
        default=Path("/models/diff2lip"),
        description="Directory containing Diff2Lip diffusion model checkpoints",
    )

    SADTALKER_CKPT_DIR: Path = Field(
        default=Path("/models/sadtalker"),
        description="Directory containing SadTalker model checkpoints",
    )

    WAV2LIP_CKPT: Path = Field(
        default=Path("/models/wav2lip/wav2lip_gan.pth"),
        description="Full path to Wav2Lip GAN checkpoint file",
    )

    GFPGAN_CKPT: Path = Field(
        default=Path("/models/gfpgan/GFPGANv1.3.pth"),
        description="Full path to GFPGAN face enhancement checkpoint file",
    )

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
    # Field Validators
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator(
        "FOMM_CKPT_DIR",
        "DIFF2LIP_CKPT_DIR",
        "SADTALKER_CKPT_DIR",
        "WAV2LIP_CKPT",
        "GFPGAN_CKPT",
        mode="after",
    )
    @classmethod
    def validate_model_paths(cls, value: Path, info: FieldInfo) -> Path:
        """Validate that model checkpoint paths exist.

        This validator logs a warning if a model path doesn't exist at initialization,
        but doesn't fail validation. This allows the application to start even if
        some models are missing, with runtime errors occurring only when those
        specific models are actually needed.

        Args:
            value: The path to validate.
            info: Field metadata containing the field name.

        Returns:
            The validated Path object (unchanged).

        Note:
            This is a soft validation - warnings are logged but the application
            continues to initialize.
        """
        if not value.exists():
            logger.warning(
                "Model path '%s' (%s) does not exist at initialization. "
                "The path will be checked again at runtime.",
                info.field_name,
                value,
            )
        return value

    @field_validator("LOG_LEVEL", mode="after")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate and normalize the log level.

        Args:
            value: The log level string to validate.

        Returns:
            The uppercase log level string.

        Raises:
            ValueError: If the log level is not a valid Python logging level.
        """
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

    def configure_logging(self) -> None:
        """Configure application-wide logging based on LOG_LEVEL setting.

        This method should be called early in the application lifecycle to set up
        consistent logging across all modules.

        Example:
            >>> settings = Settings()
            >>> settings.configure_logging()
            >>> logging.info("Logging is now configured")
        """
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.info("Logging configured with level: %s", self.LOG_LEVEL)


# ─────────────────────────────────────────────────────────────────────────────
# Global Settings Instance
# ─────────────────────────────────────────────────────────────────────────────

# Singleton settings instance used throughout the application
settings = Settings()

# Configure logging immediately upon import
settings.configure_logging()
