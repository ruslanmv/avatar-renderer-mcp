"""
settings.py – centralised runtime configuration for Avatar Renderer Pod
======================================================================

All options can be supplied via:

1. Environment variables   e.g. CELERY_BROKER_URL=redis://…
2. A “.env” file (auto‑loaded if present)
3. Direct keyword overrides in tests, e.g. Settings(celery_broker_url="…")
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, FieldValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------#
    # General                                                           #
    # ------------------------------------------------------------------#
    LOG_LEVEL: str = Field("INFO", description="Python logging level")
    TMP_DIR: Path = Field(Path("/tmp"), description="Scratch directory for jobs")

    # ------------------------------------------------------------------#
    # GPU / CUDA                                                        #
    # ------------------------------------------------------------------#
    CUDA_VISIBLE_DEVICES: str | None = Field(
        None,
        alias="CUDA_VISIBLE_DEVICES",
        description="If set, pins the container to a specific GPU id",
    )
    TORCH_USE_INT8: bool = Field(False, description="Enable INT8 inference where supported")

    # ------------------------------------------------------------------#
    # Celery (optional)                                                 #
    # ------------------------------------------------------------------#
    CELERY_BROKER_URL: str | None = Field(
        None,
        description="e.g. redis://redis:6379/0 – leave empty for ‘local thread’ mode",
    )
    CELERY_BACKEND_URL: str | None = Field(
        None, description="Result backend (defaults to broker if empty)"
    )
    CELERY_CONCURRENCY: int = Field(
        1, description="Number of Celery worker processes inside the pod"
    )

    # ------------------------------------------------------------------#
    # Model paths (all must resolve at run‑time)                        #
    # ------------------------------------------------------------------#
    MODEL_ROOT: Path = Field(Path("/models"), description="Parent directory for checkpoints")

    # Sub‑folders – override independently if you mount them elsewhere
    FOMM_CKPT_DIR: Path = Field(default_factory=lambda: Path("/models/fomm"))
    DIFF2LIP_CKPT_DIR: Path = Field(default_factory=lambda: Path("/models/diff2lip"))
    SADTALKER_CKPT_DIR: Path = Field(default_factory=lambda: Path("/models/sadtalker"))
    WAV2LIP_CKPT: Path = Field(default_factory=lambda: Path("/models/wav2lip/wav2lip_gan.pth"))
    GFPGAN_CKPT: Path = Field(default_factory=lambda: Path("/models/gfpgan/GFPGANv1.4.pth"))

    # ------------------------------------------------------------------#
    # FFmpeg                                                            #
    # ------------------------------------------------------------------#
    FFMPEG_BIN: str = Field("ffmpeg", description="Path or name of the ffmpeg binary")

    # ------------------------------------------------------------------#
    # MCP integration                                                   #
    # ------------------------------------------------------------------#
    MCP_ENABLE: bool = Field(True, description="Whether to start the MCP server")
    MCP_TOOL_NAME: str = Field("avatar_renderer", description="Name advertised to the gateway")

    # ------------------------------------------------------------------#
    # Validation hooks                                                  #
    # ------------------------------------------------------------------#
    @field_validator(
        "FOMM_CKPT_DIR",
        "DIFF2LIP_CKPT_DIR",
        "SADTALKER_CKPT_DIR",
        "WAV2LIP_CKPT",
        "GFPGAN_CKPT",
        mode="before",
    )
    @classmethod
    def _warn_if_path_missing(cls, value: Path, info: FieldValidationInfo) -> Path:
        """Log a warning (but don’t crash) if a model path is missing at boot‑time."""
        p = Path(value)
        if not p.exists():
            logging.getLogger("settings").warning(
                "⚠️  %s path %s does not exist at boot‑time", info.field_name, p
            )
        return p

    # ------------------------------------------------------------------#
    # Pydantic‑Settings configuration                                   #
    # ------------------------------------------------------------------#
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# A singleton shared across the project
settings = Settings()
