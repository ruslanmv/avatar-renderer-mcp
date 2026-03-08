"""
enhancements – Additive enhancement registry for the Avatar Renderer pipeline.

This module provides a plugin-like system where each enhancement is:
  - Self-contained (own model loading, inference, cleanup)
  - Non-destructive (original pipeline untouched; enhancements wrap around it)
  - Independently toggleable via settings or per-request parameters
  - Gracefully degrading (if a model is missing, the enhancement is skipped)

Architecture:
    Each enhancement implements the Enhancement base class with:
      - name/id/priority for ordering
      - is_available() to check if models/deps are present
      - apply() to process frames/video/audio in the pipeline context

    Enhancements are applied in priority order AFTER the core pipeline stages.
    They receive an EnhancementContext containing all intermediate artifacts.

Usage:
    from app.enhancements import registry, EnhancementContext

    ctx = EnhancementContext(frames_dir=..., audio_path=..., ...)
    ctx = registry.apply_all(ctx, enabled=["emotion_expressions", "eye_gaze_blink"])
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enhancement Context – carries all artifacts through the enhancement chain
# --------------------------------------------------------------------------- #

@dataclass
class EnhancementContext:
    """Mutable context object passed through the enhancement chain.

    Each enhancement reads what it needs and writes back updated artifacts.
    The original values are never lost (stored in `original_*` fields).
    """

    # --- Input artifacts (set by pipeline before enhancements) ---
    face_image: str = ""
    audio_path: str = ""
    reference_video: Optional[str] = None
    viseme_json: Optional[str] = None
    transcript: Optional[str] = None
    quality_mode: str = "auto"

    # --- Intermediate artifacts (mutated by enhancements) ---
    frames_dir: Optional[Path] = None
    video_path: Optional[Path] = None
    tmp_dir: Optional[Path] = None

    # --- Emotion / expression state (set by emotion analyzer) ---
    detected_emotion: Optional[str] = None
    emotion_scores: Optional[Dict[str, float]] = None
    expression_params: Optional[Dict[str, float]] = None

    # --- Gesture state (set by gesture generator) ---
    gesture_keypoints: Optional[Any] = None

    # --- Eye gaze state (set by gaze modeler) ---
    gaze_schedule: Optional[List[Dict]] = None
    blink_schedule: Optional[List[float]] = None

    # --- Original artifacts (preserved for fallback) ---
    original_frames_dir: Optional[Path] = None
    original_video_path: Optional[Path] = None

    # --- Metadata ---
    applied_enhancements: List[str] = field(default_factory=list)
    enhancement_logs: Dict[str, str] = field(default_factory=dict)
    fps: int = 25
    device: str = "cpu"

    def snapshot_originals(self):
        """Save current state as originals before any enhancement modifies it."""
        if self.original_frames_dir is None:
            self.original_frames_dir = self.frames_dir
        if self.original_video_path is None:
            self.original_video_path = self.video_path


# --------------------------------------------------------------------------- #
# Enhancement Base Class
# --------------------------------------------------------------------------- #

class Enhancement(ABC):
    """Base class for all pipeline enhancements.

    Subclasses must implement:
        - name: unique string identifier
        - stage: when this enhancement runs (pre_motion, lip_sync, post_process, tts)
        - priority: integer for ordering (lower = earlier)
        - is_available(): whether required models/deps are present
        - apply(ctx): process the context and return updated context
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this enhancement (e.g., 'emotion_expressions')."""

    @property
    @abstractmethod
    def stage(self) -> str:
        """Pipeline stage: 'pre_motion', 'motion_driver', 'lip_sync', 'post_process', 'tts'."""

    @property
    @abstractmethod
    def priority(self) -> int:
        """Execution order within stage (lower = earlier)."""

    @property
    def description(self) -> str:
        """Human-readable description."""
        return ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all required models and dependencies are present."""

    @abstractmethod
    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        """Apply this enhancement to the context. Must return the (possibly modified) context."""

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about this enhancement for the /avatars endpoint."""
        return {
            "name": self.name,
            "stage": self.stage,
            "priority": self.priority,
            "description": self.description,
            "available": self.is_available(),
        }


# --------------------------------------------------------------------------- #
# Enhancement Registry
# --------------------------------------------------------------------------- #

class EnhancementRegistry:
    """Registry of all available enhancements.

    Enhancements register themselves at import time via `registry.register(enhancement)`.
    The pipeline calls `registry.apply_all(ctx, enabled=[...])` to run them.
    """

    def __init__(self):
        self._enhancements: Dict[str, Enhancement] = {}

    def register(self, enhancement: Enhancement) -> None:
        """Register an enhancement instance."""
        if enhancement.name in self._enhancements:
            logger.warning("Enhancement '%s' already registered, overwriting.", enhancement.name)
        self._enhancements[enhancement.name] = enhancement
        logger.debug("Registered enhancement: %s (stage=%s, priority=%d)",
                      enhancement.name, enhancement.stage, enhancement.priority)

    def get(self, name: str) -> Optional[Enhancement]:
        """Get an enhancement by name."""
        return self._enhancements.get(name)

    def list_all(self) -> List[Enhancement]:
        """Return all registered enhancements sorted by (stage, priority)."""
        stage_order = {"tts": 0, "pre_motion": 1, "motion_driver": 2, "lip_sync": 3, "post_process": 4}
        return sorted(
            self._enhancements.values(),
            key=lambda e: (stage_order.get(e.stage, 99), e.priority),
        )

    def list_available(self) -> List[Enhancement]:
        """Return only available enhancements."""
        return [e for e in self.list_all() if e.is_available()]

    def list_names(self) -> List[str]:
        """Return all enhancement names."""
        return [e.name for e in self.list_all()]

    def get_info_all(self) -> List[Dict[str, Any]]:
        """Return metadata for all enhancements (for /avatars endpoint)."""
        return [e.get_info() for e in self.list_all()]

    def apply_stage(
        self,
        ctx: EnhancementContext,
        stage: str,
        enabled: Optional[Set[str]] = None,
    ) -> EnhancementContext:
        """Apply all enhancements for a specific stage."""
        for enhancement in self.list_all():
            if enhancement.stage != stage:
                continue
            if enabled is not None and enhancement.name not in enabled:
                continue
            if not enhancement.is_available():
                logger.info("Enhancement '%s' not available, skipping.", enhancement.name)
                continue

            try:
                logger.info("Applying enhancement: %s", enhancement.name)
                ctx = enhancement.apply(ctx)
                ctx.applied_enhancements.append(enhancement.name)
                ctx.enhancement_logs[enhancement.name] = "success"
            except Exception as e:
                logger.error("Enhancement '%s' failed: %s", enhancement.name, e, exc_info=True)
                ctx.enhancement_logs[enhancement.name] = f"error: {e}"
                # Non-destructive: failure just means we skip this enhancement

        return ctx

    def apply_all(
        self,
        ctx: EnhancementContext,
        enabled: Optional[Set[str]] = None,
    ) -> EnhancementContext:
        """Apply all enabled enhancements in stage/priority order."""
        ctx.snapshot_originals()
        for stage in ("tts", "pre_motion", "motion_driver", "lip_sync", "post_process"):
            ctx = self.apply_stage(ctx, stage, enabled)
        return ctx


# --------------------------------------------------------------------------- #
# Global registry instance
# --------------------------------------------------------------------------- #

registry = EnhancementRegistry()


# --------------------------------------------------------------------------- #
# Auto-discover and register all enhancement modules
# --------------------------------------------------------------------------- #

def _auto_register():
    """Import all enhancement modules so they self-register."""
    import importlib

    modules = [
        "emotion_expressions",
        "musetalk_lipsync",
        "eye_gaze_blink",
        "liveportrait_driver",
        "latentsync_lipsync",
        "hallo3_cinematic",
        "cosyvoice_tts",
        "viseme_guided",
        "gesture_animation",
        "gaussian_splatting",
    ]
    for mod_name in modules:
        try:
            importlib.import_module(f".{mod_name}", package=__name__)
        except Exception as e:
            logger.debug("Could not load enhancement module '%s': %s", mod_name, e)


_auto_register()
