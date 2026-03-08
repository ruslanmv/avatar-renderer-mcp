"""
Enhancement #1: Emotion-Aware Expressions
==========================================

Detects emotion from audio (via NVIDIA Audio2Emotion or fallback text sentiment)
and drives facial expression parameters that downstream enhancements (LivePortrait,
eye gaze) can consume.

Pipeline stage: pre_motion (runs before motion drivers so they can use emotion data)

Models required (any one of):
  - NVIDIA Audio2Emotion (models/audio2emotion/)
  - Transformers sentiment classifier (auto-downloaded)

This enhancement SETS emotion state on the context but does NOT modify frames.
Downstream enhancements (LivePortrait, eye gaze) read ctx.detected_emotion
and ctx.expression_params to adjust their behavior.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from . import Enhancement, EnhancementContext, registry

logger = logging.getLogger(__name__)

# Model paths
_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "models"))
_A2E_MODEL_DIR = _MODEL_ROOT / "audio2emotion"

# Expression parameter mappings for detected emotions
# Values are blendshape-like weights: positive = activate, 0 = neutral
EMOTION_EXPRESSION_MAP = {
    "happy": {
        "smile": 0.7, "eyebrow_raise": 0.2, "eye_squint": 0.3,
        "mouth_open": 0.1, "cheek_raise": 0.4,
    },
    "sad": {
        "smile": -0.3, "eyebrow_raise": -0.2, "eye_squint": 0.0,
        "mouth_corners_down": 0.5, "inner_brow_raise": 0.4,
    },
    "angry": {
        "smile": -0.4, "eyebrow_lower": 0.6, "eye_squint": 0.2,
        "jaw_clench": 0.3, "nostril_flare": 0.2,
    },
    "surprised": {
        "smile": 0.1, "eyebrow_raise": 0.8, "eye_wide": 0.7,
        "mouth_open": 0.5, "jaw_drop": 0.3,
    },
    "fearful": {
        "smile": -0.1, "eyebrow_raise": 0.5, "eye_wide": 0.6,
        "mouth_open": 0.2, "inner_brow_raise": 0.6,
    },
    "disgusted": {
        "smile": -0.3, "nose_wrinkle": 0.6, "upper_lip_raise": 0.4,
        "eyebrow_lower": 0.3, "eye_squint": 0.2,
    },
    "neutral": {
        "smile": 0.0, "eyebrow_raise": 0.0, "eye_squint": 0.0,
        "mouth_open": 0.0,
    },
}


def _detect_emotion_from_audio_a2e(audio_path: str) -> Optional[Dict[str, Any]]:
    """Detect emotion using NVIDIA Audio2Emotion model."""
    try:
        # Audio2Emotion uses ONNX runtime or torch for inference
        import numpy as np

        a2e_model_path = _A2E_MODEL_DIR / "audio2emotion.onnx"
        if not a2e_model_path.exists():
            return None

        try:
            import onnxruntime as ort
        except ImportError:
            logger.debug("onnxruntime not installed, skipping Audio2Emotion")
            return None

        import librosa

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Extract mel spectrogram features (Audio2Emotion input format)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=160)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Chunk into windows (A2E processes 1-second windows)
        window_size = 100  # ~1 second at 16kHz with hop=160
        emotions_over_time = []

        session = ort.InferenceSession(str(a2e_model_path))
        input_name = session.get_inputs()[0].name

        for i in range(0, mel_db.shape[1], window_size):
            chunk = mel_db[:, i:i + window_size]
            if chunk.shape[1] < window_size:
                chunk = np.pad(chunk, ((0, 0), (0, window_size - chunk.shape[1])))
            chunk = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)

            outputs = session.run(None, {input_name: chunk})
            emotion_probs = outputs[0][0]
            emotions_over_time.append(emotion_probs)

        if not emotions_over_time:
            return None

        # Average emotion across all windows
        avg_emotions = np.mean(emotions_over_time, axis=0)
        emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
        scores = {label: float(score) for label, score in zip(emotion_labels, avg_emotions)}
        dominant = max(scores, key=scores.get)

        return {"emotion": dominant, "scores": scores}

    except Exception as e:
        logger.warning("Audio2Emotion inference failed: %s", e)
        return None


def _detect_emotion_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Fallback: detect emotion from transcript text using transformers."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.debug("transformers not installed, skipping text emotion detection")
        return None

    try:
        classifier = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,  # CPU to avoid GPU memory conflicts
        )
        results = classifier(text[:512])  # Truncate to model max length
        if results and results[0]:
            scores = {r["label"]: r["score"] for r in results[0]}
            dominant = max(scores, key=scores.get)
            return {"emotion": dominant, "scores": scores}
    except Exception as e:
        logger.warning("Text emotion detection failed: %s", e)

    return None


def _detect_emotion_from_audio_prosody(audio_path: str) -> Optional[Dict[str, Any]]:
    """Lightweight fallback: use audio prosody features (pitch, energy, tempo)."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # Extract prosody features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 200.0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 50.0

        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        tempo = librosa.beat.tempo(y=y, sr=sr)[0]

        # Simple heuristic mapping
        # High pitch + high energy + fast tempo → happy/excited
        # Low pitch + low energy + slow tempo → sad
        # High energy + low pitch variation → angry
        scores = {
            "neutral": 0.3,
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "fearful": 0.0,
            "disgusted": 0.0,
        }

        if pitch_mean > 250 and energy_mean > 0.05 and tempo > 120:
            scores["happy"] = 0.6
        elif pitch_mean < 150 and energy_mean < 0.03 and tempo < 90:
            scores["sad"] = 0.6
        elif energy_mean > 0.08 and pitch_std < 30:
            scores["angry"] = 0.5
        elif pitch_std > 80:
            scores["surprised"] = 0.4
        else:
            scores["neutral"] = 0.6

        dominant = max(scores, key=scores.get)
        return {"emotion": dominant, "scores": scores}

    except Exception as e:
        logger.warning("Prosody analysis failed: %s", e)
        return None


class EmotionExpressionsEnhancement(Enhancement):
    """Detects emotion from audio/text and sets expression parameters on context."""

    @property
    def name(self) -> str:
        return "emotion_expressions"

    @property
    def stage(self) -> str:
        return "pre_motion"

    @property
    def priority(self) -> int:
        return 10

    @property
    def description(self) -> str:
        return (
            "Analyzes audio prosody and/or transcript text to detect emotion "
            "(happy, sad, angry, surprised, etc.) and sets facial expression "
            "parameters for downstream enhancements."
        )

    def is_available(self) -> bool:
        # Always available via prosody fallback; upgraded if A2E or transformers present
        return True

    def apply(self, ctx: EnhancementContext) -> EnhancementContext:
        result = None

        # Try NVIDIA Audio2Emotion first (best quality)
        if _A2E_MODEL_DIR.exists():
            result = _detect_emotion_from_audio_a2e(ctx.audio_path)
            if result:
                logger.info("Emotion detected via Audio2Emotion: %s", result["emotion"])

        # Fallback to text sentiment if transcript available
        if result is None and ctx.transcript:
            result = _detect_emotion_from_text(ctx.transcript)
            if result:
                logger.info("Emotion detected via text sentiment: %s", result["emotion"])

        # Final fallback: audio prosody heuristics
        if result is None and ctx.audio_path:
            result = _detect_emotion_from_audio_prosody(ctx.audio_path)
            if result:
                logger.info("Emotion detected via prosody: %s", result["emotion"])

        if result:
            ctx.detected_emotion = result["emotion"]
            ctx.emotion_scores = result["scores"]
            ctx.expression_params = EMOTION_EXPRESSION_MAP.get(
                result["emotion"], EMOTION_EXPRESSION_MAP["neutral"]
            )
        else:
            ctx.detected_emotion = "neutral"
            ctx.expression_params = EMOTION_EXPRESSION_MAP["neutral"]

        return ctx


# Auto-register
registry.register(EmotionExpressionsEnhancement())
