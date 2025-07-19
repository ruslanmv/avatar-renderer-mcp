"""
settings.py  –  Centralised runtime configuration for Avatar Renderer Pod
========================================================================
Uses *Pydantic‑v2* `BaseSettings` so every option can be supplied via:

1. Environment variables   e.g.  CELERY_BROKER_URL=redis://…
2. A `.env` file mounted into the container (if `load_dotenv=True`)
3. Keyword overrides inside unit‑tests  ⇒  `Settings(celery_broker_url="…")`

Add new fields here instead of scattering `os.getenv()` across the code‑base.
"""

from pathlib import Path
from pydantic import Field, field_validator, FieldValidationInfo
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------#
    # General                                                           #
    # ------------------------------------------------------------------#
    LOG_LEVEL: str = Field("INFO", description="python-logging level")
    TMP_DIR: Path = Field(Path("/tmp"), description="scratch dir for jobs")

    # ------------------------------------------------------------------#
    # GPU / CUDA                                                        #
    # ------------------------------------------------------------------#
    CUDA_VISIBLE_DEVICES: str | None = Field(
        None,
        description="Override to pin the container to a specific GPU id.",
        alias="CUDA_VISIBLE_DEVICES",
    )
    TORCH_USE_INT8: bool = Field(
        False, description="Enable INT8 inference where supported"
    )

    # ------------------------------------------------------------------#
    # Celery (optional)                                                 #
    # ------------------------------------------------------------------#
    CELERY_BROKER_URL: str | None = Field(
        None,
        description="e.g. redis://redis:6379/0 - leave empty for ‘local thread’ mode",
    )
    CELERY_BACKEND_URL: str | None = Field(
        None, description="Result backend – default to broker if empty"
    )
    CELERY_CONCURRENCY: int = Field(
        1, description="Number of Celery worker processes inside the pod"
    )

    # ------------------------------------------------------------------#
    # Model paths (all must resolve at run‑time)                        #
    # ------------------------------------------------------------------#
    MODEL_ROOT: Path = Field(
        Path("/models"), description="Parent directory for all checkpoints"
    )
    # Sub‑folders – can be overridden independently if you mount them elsewhere
    FOMM_CKPT_DIR: Path = Field(default_factory=lambda: Path("/models/fomm"))
    DIFF2LIP_CKPT_DIR: Path = Field(default_factory=lambda: Path("/models/diff2lip"))
    SADTALKER_CKPT_DIR: Path = Field(default_factory=lambda: Path("/models/sadtalker"))
    WAV2LIP_CKPT: Path = Field(default_factory=lambda: Path("/models/wav2lip/wav2lip_gan.pth"))
    GFPGAN_CKPT: Path = Field(default_factory=lambda: Path("/models/gfpgan/GFPGANv1.4.pth"))

    # ------------------------------------------------------------------#
    # FFmpeg                                                            #
    # ------------------------------------------------------------------#
    FFMPEG_BIN: str = Field("ffmpeg", description="Path or name of ffmpeg binary")

    # ------------------------------------------------------------------#
    # MCP integration                                                   #
    # ------------------------------------------------------------------#
    MCP_ENABLE: bool = Field(True, description="Whether to start MCP server")
    MCP_TOOL_NAME: str = Field("avatar_renderer", description="Name exposed to gateway")

    # ------------------------------------------------------------------#
    # Validation hooks                                                  #
    # ------------------------------------------------------------------#
    @field_validator("FOMM_CKPT_DIR", "DIFF2LIP_CKPT_DIR", "SADTALKER_CKPT_DIR",
                     "WAV2LIP_CKPT", "GFPGAN_CKPT", mode='before')
    @classmethod
    def _path_exists(cls, value: Path, info: FieldValidationInfo):
        # Only warn – don’t fail pod start‑up; pipeline will raise if required model missing
        # We ensure the value is resolved to a Path object before checking existence
        p = Path(value)
        if not p.exists():
            import logging

            logging.getLogger("settings").warning(
                "⚠️  %s path %s does not exist at boot‑time", info.field_name, value
            )
        return value

    # ------------------------------------------------------------------#
    # Pydantic-Settings Configuration                                   #
    # ------------------------------------------------------------------#
    model_config = {
        "env_file": ".env",             # auto‑load if present
        "env_file_encoding": "utf-8",
        "case_sensitive": False,        # so `celery_broker_url` and `CELERY_BROKER_URL` both work
    }


# singleton instance used across the project
settings = Settings()