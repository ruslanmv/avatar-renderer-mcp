"""
Unit tests for the enhancements module.

All tests are mock-based and require NO GPU, models, or heavy ML dependencies.
Compatible with GitHub CI workflows (only needs pytest + standard lib).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ===========================================================================
# 1. EnhancementContext tests
# ===========================================================================

class TestEnhancementContext:

    def test_defaults(self):
        from app.enhancements import EnhancementContext

        ctx = EnhancementContext()
        assert ctx.face_image == ""
        assert ctx.audio_path == ""
        assert ctx.quality_mode == "auto"
        assert ctx.fps == 25
        assert ctx.device == "cpu"
        assert ctx.applied_enhancements == []
        assert ctx.enhancement_logs == {}

    def test_snapshot_originals_preserves_first_value(self, tmp_path):
        from app.enhancements import EnhancementContext

        frames = tmp_path / "frames"
        frames.mkdir()
        ctx = EnhancementContext(frames_dir=frames)

        ctx.snapshot_originals()
        assert ctx.original_frames_dir == frames

        # Second snapshot should NOT overwrite
        new_frames = tmp_path / "new_frames"
        new_frames.mkdir()
        ctx.frames_dir = new_frames
        ctx.snapshot_originals()
        assert ctx.original_frames_dir == frames  # still the first one

    def test_mutable_lists_are_independent(self):
        from app.enhancements import EnhancementContext

        ctx1 = EnhancementContext()
        ctx2 = EnhancementContext()
        ctx1.applied_enhancements.append("foo")
        assert "foo" not in ctx2.applied_enhancements


# ===========================================================================
# 2. Enhancement base class tests
# ===========================================================================

class TestEnhancementBaseClass:

    def test_get_info_returns_correct_keys(self):
        from app.enhancements import Enhancement, EnhancementContext

        class DummyEnhancement(Enhancement):
            @property
            def name(self):
                return "dummy"

            @property
            def stage(self):
                return "post_process"

            @property
            def priority(self):
                return 50

            def is_available(self):
                return True

            def apply(self, ctx):
                return ctx

        enh = DummyEnhancement()
        info = enh.get_info()
        assert info["name"] == "dummy"
        assert info["stage"] == "post_process"
        assert info["priority"] == 50
        assert info["available"] is True
        assert "description" in info


# ===========================================================================
# 3. EnhancementRegistry tests
# ===========================================================================

class TestEnhancementRegistry:

    def _make_enhancement(self, name, stage, priority, available=True):
        from app.enhancements import Enhancement, EnhancementContext

        class E(Enhancement):
            @property
            def name(self_inner):
                return name

            @property
            def stage(self_inner):
                return stage

            @property
            def priority(self_inner):
                return priority

            def is_available(self_inner):
                return available

            def apply(self_inner, ctx):
                return ctx

        return E()

    def test_register_and_get(self):
        from app.enhancements import EnhancementRegistry

        reg = EnhancementRegistry()
        enh = self._make_enhancement("test_enh", "post_process", 10)
        reg.register(enh)
        assert reg.get("test_enh") is enh
        assert reg.get("nonexistent") is None

    def test_list_all_sorted_by_stage_then_priority(self):
        from app.enhancements import EnhancementRegistry

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("pp_late", "post_process", 30))
        reg.register(self._make_enhancement("pre", "pre_motion", 10))
        reg.register(self._make_enhancement("pp_early", "post_process", 5))
        reg.register(self._make_enhancement("tts", "tts", 10))

        names = [e.name for e in reg.list_all()]
        assert names == ["tts", "pre", "pp_early", "pp_late"]

    def test_list_available_filters_unavailable(self):
        from app.enhancements import EnhancementRegistry

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("yes", "post_process", 10, available=True))
        reg.register(self._make_enhancement("no", "post_process", 20, available=False))

        available = reg.list_available()
        assert len(available) == 1
        assert available[0].name == "yes"

    def test_list_names(self):
        from app.enhancements import EnhancementRegistry

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("a", "tts", 1))
        reg.register(self._make_enhancement("b", "post_process", 1))
        assert reg.list_names() == ["a", "b"]

    def test_get_info_all(self):
        from app.enhancements import EnhancementRegistry

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("x", "lip_sync", 10))
        infos = reg.get_info_all()
        assert len(infos) == 1
        assert infos[0]["name"] == "x"

    def test_apply_stage_respects_enabled_filter(self):
        from app.enhancements import EnhancementRegistry, EnhancementContext

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("a", "post_process", 10))
        reg.register(self._make_enhancement("b", "post_process", 20))

        ctx = EnhancementContext()
        ctx = reg.apply_stage(ctx, "post_process", enabled={"a"})
        assert "a" in ctx.applied_enhancements
        assert "b" not in ctx.applied_enhancements

    def test_apply_stage_skips_unavailable(self):
        from app.enhancements import EnhancementRegistry, EnhancementContext

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("avail", "pre_motion", 10, available=True))
        reg.register(self._make_enhancement("not_avail", "pre_motion", 20, available=False))

        ctx = EnhancementContext()
        ctx = reg.apply_stage(ctx, "pre_motion")
        assert "avail" in ctx.applied_enhancements
        assert "not_avail" not in ctx.applied_enhancements

    def test_apply_stage_catches_exceptions(self):
        """An enhancement that raises should NOT crash the registry."""
        from app.enhancements import Enhancement, EnhancementRegistry, EnhancementContext

        class Broken(Enhancement):
            @property
            def name(self):
                return "broken"

            @property
            def stage(self):
                return "post_process"

            @property
            def priority(self):
                return 1

            def is_available(self):
                return True

            def apply(self, ctx):
                raise RuntimeError("boom")

        reg = EnhancementRegistry()
        reg.register(Broken())
        reg.register(self._make_enhancement("ok", "post_process", 10))

        ctx = EnhancementContext()
        ctx = reg.apply_stage(ctx, "post_process")
        # broken should have logged an error, not crashed
        assert "broken" in ctx.enhancement_logs
        assert "error" in ctx.enhancement_logs["broken"]
        # the second enhancement should still run
        assert "ok" in ctx.applied_enhancements

    def test_apply_all_runs_stages_in_order(self):
        from app.enhancements import EnhancementRegistry, EnhancementContext

        reg = EnhancementRegistry()
        reg.register(self._make_enhancement("pp", "post_process", 1))
        reg.register(self._make_enhancement("pre", "pre_motion", 1))
        reg.register(self._make_enhancement("tts_e", "tts", 1))

        ctx = EnhancementContext()
        ctx = reg.apply_all(ctx)
        # tts → pre_motion → post_process
        assert ctx.applied_enhancements == ["tts_e", "pre", "pp"]

    def test_apply_all_snapshots_originals(self, tmp_path):
        from app.enhancements import EnhancementRegistry, EnhancementContext

        reg = EnhancementRegistry()
        frames = tmp_path / "f"
        frames.mkdir()
        ctx = EnhancementContext(frames_dir=frames)
        ctx = reg.apply_all(ctx)
        assert ctx.original_frames_dir == frames


# ===========================================================================
# 4. Global registry has all 10 enhancements registered
# ===========================================================================

class TestGlobalRegistry:

    def test_all_10_enhancements_registered(self):
        from app.enhancements import registry

        expected = {
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
        }
        registered = set(registry.list_names())
        assert expected.issubset(registered), f"Missing: {expected - registered}"

    def test_each_enhancement_has_valid_stage(self):
        from app.enhancements import registry

        valid_stages = {"tts", "pre_motion", "motion_driver", "lip_sync", "post_process"}
        for enh in registry.list_all():
            assert enh.stage in valid_stages, f"{enh.name} has invalid stage '{enh.stage}'"

    def test_each_enhancement_has_name_and_priority(self):
        from app.enhancements import registry

        for enh in registry.list_all():
            assert isinstance(enh.name, str) and len(enh.name) > 0
            assert isinstance(enh.priority, int)

    def test_get_info_all_returns_list_of_dicts(self):
        from app.enhancements import registry

        infos = registry.get_info_all()
        assert isinstance(infos, list)
        assert len(infos) >= 10
        for info in infos:
            assert "name" in info
            assert "stage" in info
            assert "available" in info

    @pytest.mark.parametrize("name", [
        "emotion_expressions",
        "gesture_animation",
    ])
    def test_always_available_enhancements(self, name):
        """Enhancements with procedural fallbacks should always be available."""
        from app.enhancements import registry

        enh = registry.get(name)
        assert enh is not None
        assert enh.is_available() is True

    @pytest.mark.parametrize("name", [
        "musetalk_lipsync",
        "liveportrait_driver",
        "latentsync_lipsync",
        "hallo3_cinematic",
        "gaussian_splatting",
    ])
    def test_model_dependent_enhancements_unavailable_without_models(self, name):
        """Enhancements requiring external models should be unavailable in CI."""
        from app.enhancements import registry

        enh = registry.get(name)
        assert enh is not None
        # These should be unavailable since no models/repos exist in CI
        assert enh.is_available() is False


# ===========================================================================
# 5. Emotion Expressions tests
# ===========================================================================

class TestEmotionExpressions:

    def test_emotion_expression_map_has_all_emotions(self):
        from app.enhancements.emotion_expressions import EMOTION_EXPRESSION_MAP

        expected = {"happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"}
        assert expected == set(EMOTION_EXPRESSION_MAP.keys())

    def test_all_emotions_have_smile_key(self):
        from app.enhancements.emotion_expressions import EMOTION_EXPRESSION_MAP

        for emotion, params in EMOTION_EXPRESSION_MAP.items():
            assert "smile" in params, f"Emotion '{emotion}' missing 'smile' param"

    def test_apply_sets_neutral_when_no_audio(self):
        from app.enhancements import EnhancementContext
        from app.enhancements.emotion_expressions import EmotionExpressionsEnhancement

        enh = EmotionExpressionsEnhancement()
        ctx = EnhancementContext(audio_path="")
        ctx = enh.apply(ctx)
        assert ctx.detected_emotion == "neutral"
        assert ctx.expression_params is not None

    def test_apply_with_prosody_fallback(self, tmp_path):
        """When audio exists but no models, prosody fallback should set emotion."""
        from app.enhancements import EnhancementContext
        from app.enhancements.emotion_expressions import EmotionExpressionsEnhancement

        enh = EmotionExpressionsEnhancement()

        # Mock the prosody function to return a known result
        with patch(
            "app.enhancements.emotion_expressions._detect_emotion_from_audio_prosody"
        ) as mock_prosody:
            mock_prosody.return_value = {
                "emotion": "happy",
                "scores": {"happy": 0.6, "neutral": 0.3, "sad": 0.0,
                           "angry": 0.0, "surprised": 0.0, "fearful": 0.0, "disgusted": 0.0},
            }
            ctx = EnhancementContext(audio_path="/fake/audio.wav")
            ctx = enh.apply(ctx)

            assert ctx.detected_emotion == "happy"
            assert ctx.expression_params["smile"] > 0

    def test_apply_text_fallback(self):
        """When transcript is provided and A2E is absent, text detection runs."""
        from app.enhancements import EnhancementContext
        from app.enhancements.emotion_expressions import EmotionExpressionsEnhancement

        enh = EmotionExpressionsEnhancement()

        with patch(
            "app.enhancements.emotion_expressions._detect_emotion_from_text"
        ) as mock_text:
            mock_text.return_value = {
                "emotion": "sad",
                "scores": {"sad": 0.7, "neutral": 0.2, "happy": 0.1,
                           "angry": 0.0, "surprised": 0.0, "fearful": 0.0, "disgusted": 0.0},
            }
            ctx = EnhancementContext(audio_path="", transcript="I feel terrible today")
            ctx = enh.apply(ctx)

            assert ctx.detected_emotion == "sad"

    def test_properties(self):
        from app.enhancements.emotion_expressions import EmotionExpressionsEnhancement

        enh = EmotionExpressionsEnhancement()
        assert enh.name == "emotion_expressions"
        assert enh.stage == "pre_motion"
        assert enh.priority == 10
        assert enh.is_available() is True
        assert len(enh.description) > 0


# ===========================================================================
# 6. Eye Gaze / Blink tests
# ===========================================================================

class TestEyeGazeBlink:

    def test_generate_blink_schedule_returns_blinks(self):
        from app.enhancements.eye_gaze_blink import generate_blink_schedule

        schedule = generate_blink_schedule(10.0, fps=25)
        assert len(schedule) > 0
        for blink in schedule:
            assert "frame" in blink
            assert "duration_frames" in blink
            assert "intensity" in blink
            assert 0 < blink["intensity"] <= 1.0
            assert blink["duration_frames"] >= 1

    def test_blink_count_roughly_matches_rate(self):
        from app.enhancements.eye_gaze_blink import generate_blink_schedule

        # 60 seconds at 17 blinks/min => ~17 blinks
        schedule = generate_blink_schedule(60.0, fps=25, blinks_per_minute=17.0)
        assert 8 <= len(schedule) <= 30  # generous range for randomness

    def test_generate_gaze_schedule_returns_events(self):
        from app.enhancements.eye_gaze_blink import generate_gaze_schedule

        schedule = generate_gaze_schedule(5.0, fps=25)
        assert len(schedule) > 0
        for event in schedule:
            assert "frame" in event
            assert "dx" in event
            assert "dy" in event
            assert "hold_frames" in event
            assert -3.0 <= event["dx"] <= 3.0
            assert -2.0 <= event["dy"] <= 2.0

    def test_properties(self):
        from app.enhancements.eye_gaze_blink import EyeGazeBlinkEnhancement

        enh = EyeGazeBlinkEnhancement()
        assert enh.name == "eye_gaze_blink"
        assert enh.stage == "post_process"
        assert enh.priority == 10

    def test_apply_no_frames_returns_ctx_unchanged(self):
        """When no frames exist and cv2 is available, apply should return ctx unchanged."""
        from app.enhancements import EnhancementContext
        from app.enhancements.eye_gaze_blink import EyeGazeBlinkEnhancement

        enh = EyeGazeBlinkEnhancement()
        ctx = EnhancementContext()

        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = enh.apply(ctx)
        assert result.frames_dir is None  # unchanged


# ===========================================================================
# 7. Gesture Animation tests
# ===========================================================================

class TestGestureAnimation:

    def test_procedural_gestures_returns_keypoints(self):
        from app.enhancements.gesture_animation import _generate_procedural_gestures

        kps = _generate_procedural_gestures(5.0, fps=25)
        assert len(kps) == 125  # 5s * 25fps
        for kp in kps:
            assert "frame" in kp
            assert "shoulder_dy" in kp
            assert "torso_dx" in kp
            assert "torso_dy" in kp

    def test_procedural_gestures_are_smooth(self):
        from app.enhancements.gesture_animation import _generate_procedural_gestures

        kps = _generate_procedural_gestures(2.0, fps=25)
        # Values should be small (subtle motion)
        for kp in kps:
            assert abs(kp["shoulder_dy"]) <= 2.0
            assert abs(kp["torso_dx"]) <= 3.0

    def test_gesturelsm_unavailable_in_ci(self):
        from app.enhancements.gesture_animation import _gesturelsm_available

        assert _gesturelsm_available() is False

    def test_properties(self):
        from app.enhancements.gesture_animation import GestureAnimationEnhancement

        enh = GestureAnimationEnhancement()
        assert enh.name == "gesture_animation"
        assert enh.stage == "post_process"
        assert enh.priority == 30
        assert enh.is_available() is True

    def test_apply_no_frames_returns_ctx_unchanged(self):
        from app.enhancements import EnhancementContext
        from app.enhancements.gesture_animation import GestureAnimationEnhancement

        enh = GestureAnimationEnhancement()
        ctx = EnhancementContext()

        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = enh.apply(ctx)
        assert result.frames_dir is None


# ===========================================================================
# 8. Viseme Guided tests
# ===========================================================================

class TestVisemeGuided:

    def test_viseme_mouth_params_has_expected_keys(self):
        from app.enhancements.viseme_guided import VISEME_MOUTH_PARAMS

        assert "A" in VISEME_MOUTH_PARAMS
        assert "BMP" in VISEME_MOUTH_PARAMS
        assert "rest" in VISEME_MOUTH_PARAMS
        for key, params in VISEME_MOUTH_PARAMS.items():
            assert "openness" in params
            assert "width" in params
            assert "roundness" in params

    def test_properties(self):
        from app.enhancements.viseme_guided import VisemeGuidedEnhancement

        enh = VisemeGuidedEnhancement()
        assert enh.name == "viseme_guided"
        assert enh.stage == "post_process"


# ===========================================================================
# 9. CosyVoice TTS tests
# ===========================================================================

class TestCosyVoiceTTS:

    def test_emotion_style_map(self):
        from app.enhancements.cosyvoice_tts import EMOTION_STYLE_MAP

        assert "happy" in EMOTION_STYLE_MAP
        assert "neutral" in EMOTION_STYLE_MAP
        assert isinstance(EMOTION_STYLE_MAP["happy"], str)

    def test_properties(self):
        from app.enhancements.cosyvoice_tts import CosyVoiceTTSEnhancement

        enh = CosyVoiceTTSEnhancement()
        assert enh.name == "cosyvoice_tts"
        assert enh.stage == "tts"


# ===========================================================================
# 10. MuseTalk Lip-Sync tests
# ===========================================================================

class TestMuseTalkLipSync:

    def test_properties(self):
        from app.enhancements.musetalk_lipsync import MuseTalkLipSyncEnhancement

        enh = MuseTalkLipSyncEnhancement()
        assert enh.name == "musetalk_lipsync"
        assert enh.stage == "lip_sync"
        assert enh.is_available() is False  # no models in CI

    def test_unavailable_without_models(self):
        from app.enhancements.musetalk_lipsync import _musetalk_models_present

        assert _musetalk_models_present() is False


# ===========================================================================
# 11. LivePortrait tests
# ===========================================================================

class TestLivePortrait:

    def test_properties(self):
        from app.enhancements.liveportrait_driver import LivePortraitDriverEnhancement

        enh = LivePortraitDriverEnhancement()
        assert enh.name == "liveportrait_driver"
        assert enh.stage == "motion_driver"
        assert enh.is_available() is False

    def test_unavailable_without_repo(self):
        from app.enhancements.liveportrait_driver import _liveportrait_available

        assert _liveportrait_available() is False


# ===========================================================================
# 12. LatentSync tests
# ===========================================================================

class TestLatentSync:

    def test_properties(self):
        from app.enhancements.latentsync_lipsync import LatentSyncLipSyncEnhancement

        enh = LatentSyncLipSyncEnhancement()
        assert enh.name == "latentsync_lipsync"
        assert enh.stage == "lip_sync"
        assert enh.is_available() is False


# ===========================================================================
# 13. Hallo3 tests
# ===========================================================================

class TestHallo3:

    def test_properties(self):
        from app.enhancements.hallo3_cinematic import Hallo3CinematicEnhancement

        enh = Hallo3CinematicEnhancement()
        assert enh.name == "hallo3_cinematic"
        assert enh.stage == "motion_driver"
        assert enh.is_available() is False


# ===========================================================================
# 14. Gaussian Splatting tests
# ===========================================================================

class TestGaussianSplatting:

    def test_properties(self):
        from app.enhancements.gaussian_splatting import GaussianSplattingEnhancement

        enh = GaussianSplattingEnhancement()
        assert enh.name == "gaussian_splatting"
        assert enh.stage == "motion_driver"
        assert enh.is_available() is False


# ===========================================================================
# 15. Integration: apply_all with mocked enhancements
# ===========================================================================

class TestIntegration:

    def test_apply_all_with_only_available(self):
        """Simulates a CI run: only procedural enhancements are available."""
        from app.enhancements import EnhancementContext, EnhancementRegistry

        # Build a fresh registry with mocked enhancements
        reg = EnhancementRegistry()

        from app.enhancements.emotion_expressions import EmotionExpressionsEnhancement
        reg.register(EmotionExpressionsEnhancement())

        ctx = EnhancementContext(audio_path="")
        ctx = reg.apply_all(ctx, enabled={"emotion_expressions"})

        assert "emotion_expressions" in ctx.applied_enhancements
        assert ctx.detected_emotion is not None

    def test_unavailable_enhancements_logged_and_skipped(self):
        """Unavailable enhancements should be silently skipped."""
        from app.enhancements import EnhancementContext, EnhancementRegistry
        from app.enhancements.hallo3_cinematic import Hallo3CinematicEnhancement

        reg = EnhancementRegistry()
        reg.register(Hallo3CinematicEnhancement())

        ctx = EnhancementContext()
        ctx = reg.apply_all(ctx, enabled={"hallo3_cinematic"})

        assert "hallo3_cinematic" not in ctx.applied_enhancements

    def test_full_registry_apply_all_no_crash(self):
        """The global registry's apply_all should not crash even with no models."""
        from app.enhancements import registry, EnhancementContext

        ctx = EnhancementContext(audio_path="", face_image="")
        # Enable only the always-available ones
        ctx = registry.apply_all(ctx, enabled={"emotion_expressions"})
        assert "emotion_expressions" in ctx.applied_enhancements
