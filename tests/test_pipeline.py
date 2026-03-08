"""
test_pipeline.py – Integration tests for app/pipeline.py

These tests exercise the real render_pipeline() function with all external
dependencies (models, ffmpeg, external repos) mocked out. They verify:

  1. Pipeline flow: FOMM → lip-sync → GFPGAN → enhancements → encode
  2. Quality mode routing (auto, real_time, high_quality)
  3. Enhancement integration within the pipeline
  4. Error handling and fallback paths
  5. PyTorch tensor operations work correctly

Requires: pytest, torch, pydantic, pydantic-settings
No GPU, no model files, no ffmpeg needed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a workspace with fake face image and audio file."""
    face = tmp_path / "face.png"
    audio = tmp_path / "speech.wav"
    output = tmp_path / "output.mp4"

    # Create minimal files so FileNotFoundError checks pass
    face.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    return {"face": str(face), "audio": str(audio), "output": str(output), "tmp": tmp_path}


@pytest.fixture
def fake_frames_dir(tmp_path):
    """Create a directory with fake PNG frames."""
    frames = tmp_path / "fomm_frames"
    frames.mkdir()
    for i in range(5):
        (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
    return frames


@pytest.fixture
def fake_video(tmp_path):
    """Create a fake MP4 file."""
    vid = tmp_path / "result.mp4"
    vid.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
    return vid


# ---------------------------------------------------------------------------
# Test: PyTorch is functional
# ---------------------------------------------------------------------------

class TestPyTorchBasics:
    """Verify PyTorch works in CI — the foundation for pipeline testing."""

    def test_torch_import(self):
        assert hasattr(torch, "__version__")

    def test_tensor_creation(self):
        t = torch.zeros(3, 256, 256)
        assert t.shape == (3, 256, 256)
        assert t.sum().item() == 0.0

    def test_tensor_operations(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = a @ b  # matmul
        assert c.shape == (4, 4)

    def test_device_detection(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor([1.0, 2.0, 3.0], device=device)
        assert t.device.type in ("cpu", "cuda")

    def test_simple_model(self):
        """A minimal nn.Module — proves model construction works."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        x = torch.randn(1, 10)
        y = model(x)
        assert y.shape == (1, 1)


# ---------------------------------------------------------------------------
# Test: Pipeline module imports
# ---------------------------------------------------------------------------

class TestPipelineImports:
    """Verify pipeline module loads and exposes expected API."""

    def test_import_render_pipeline(self):
        from app.pipeline import render_pipeline
        assert callable(render_pipeline)

    def test_import_helpers(self):
        from app.pipeline import (
            run_fomm, run_diff2lip, run_wav2lip,
            enhance_with_gfpgan, encode_mp4,
        )
        for fn in [run_fomm, run_diff2lip, run_wav2lip, enhance_with_gfpgan, encode_mp4]:
            assert callable(fn)

    def test_model_paths_defined(self):
        from app.pipeline import FOMM_CKPT, D2L_PAPER_CKPT, W2L_CKPT, GFPGAN_CKPT
        for p in [FOMM_CKPT, D2L_PAPER_CKPT, W2L_CKPT, GFPGAN_CKPT]:
            assert isinstance(p, Path)

    def test_render_pipeline_signature(self):
        import inspect
        from app.pipeline import render_pipeline
        sig = inspect.signature(render_pipeline)
        params = list(sig.parameters.keys())
        assert "face_image" in params
        assert "audio" in params
        assert "out_path" in params
        assert "quality_mode" in params
        assert "enhancements" in params
        assert sig.parameters["quality_mode"].default == "auto"


# ---------------------------------------------------------------------------
# Test: render_pipeline with mocked stages
# ---------------------------------------------------------------------------

class TestRenderPipelineCoreFlow:
    """Test the main render_pipeline() by mocking all external calls."""

    def _mock_run_fomm(self, face_img, audio_wav, ref_video, tmp_dir):
        """Simulate FOMM producing frames."""
        frames = tmp_dir / "fomm_frames"
        frames.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
        return frames

    def _mock_run_wav2lip(self, frames_dir, audio_wav, tmp_dir):
        """Simulate Wav2Lip producing a video."""
        out = tmp_dir / "wav2lip_out.mp4"
        out.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
        return out

    def _mock_encode_mp4(self, frames_dir, audio_wav, out_mp4, fps=25):
        """Simulate ffmpeg encoding."""
        Path(out_mp4).write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)

    def _mock_remux(self, video_in, audio_in, out_mp4):
        """Simulate ffmpeg remux."""
        out_mp4.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_basic_pipeline_no_enhancements(
        self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace
    ):
        """Pipeline runs FOMM → Wav2Lip → encode without enhancements."""
        from app.pipeline import render_pipeline

        mock_fomm.side_effect = lambda *a, **kw: self._mock_run_fomm(*a, **kw)
        mock_wav2lip.side_effect = lambda *a, **kw: self._mock_run_wav2lip(*a, **kw)
        mock_remux.side_effect = lambda v, a, o: self._mock_remux(v, a, o)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
        )

        mock_fomm.assert_called_once()
        mock_wav2lip.assert_called_once()
        assert result is not None

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_pipeline_with_force_wav2lip(
        self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace
    ):
        """force_wav2lip=True always routes to Wav2Lip."""
        from app.pipeline import render_pipeline

        mock_fomm.side_effect = lambda *a, **kw: self._mock_run_fomm(*a, **kw)
        mock_wav2lip.side_effect = lambda *a, **kw: self._mock_run_wav2lip(*a, **kw)
        mock_remux.side_effect = lambda v, a, o: self._mock_remux(v, a, o)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            force_wav2lip=True,
        )

        mock_wav2lip.assert_called_once()
        assert result is not None

    def test_pipeline_rejects_missing_face(self, tmp_workspace):
        """Pipeline raises FileNotFoundError for missing face image."""
        from app.pipeline import render_pipeline
        with pytest.raises(FileNotFoundError, match="Face image not found"):
            render_pipeline(
                face_image="/nonexistent/face.png",
                audio=tmp_workspace["audio"],
                out_path=tmp_workspace["output"],
            )

    def test_pipeline_rejects_missing_audio(self, tmp_workspace):
        """Pipeline raises FileNotFoundError for missing audio."""
        from app.pipeline import render_pipeline
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            render_pipeline(
                face_image=tmp_workspace["face"],
                audio="/nonexistent/audio.wav",
                out_path=tmp_workspace["output"],
            )


# ---------------------------------------------------------------------------
# Test: Pipeline with enhancements enabled
# ---------------------------------------------------------------------------

class TestPipelineWithEnhancements:
    """Test enhancement integration within render_pipeline."""

    def _mock_run_fomm(self, face_img, audio_wav, ref_video, tmp_dir):
        frames = tmp_dir / "fomm_frames"
        frames.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
        return frames

    def _mock_run_wav2lip(self, frames_dir, audio_wav, tmp_dir):
        out = tmp_dir / "wav2lip_out.mp4"
        out.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
        return out

    def _mock_remux(self, video_in, audio_in, out_mp4):
        out_mp4.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_pipeline_with_emotion_enhancement(
        self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace
    ):
        """Pipeline runs with emotion_expressions enhancement enabled."""
        from app.pipeline import render_pipeline

        mock_fomm.side_effect = lambda *a, **kw: self._mock_run_fomm(*a, **kw)
        mock_wav2lip.side_effect = lambda *a, **kw: self._mock_run_wav2lip(*a, **kw)
        mock_remux.side_effect = lambda v, a, o: self._mock_remux(v, a, o)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            enhancements=["emotion_expressions"],
        )

        assert result is not None
        mock_fomm.assert_called_once()

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_pipeline_with_multiple_enhancements(
        self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace
    ):
        """Pipeline runs with multiple zero-config enhancements."""
        from app.pipeline import render_pipeline

        mock_fomm.side_effect = lambda *a, **kw: self._mock_run_fomm(*a, **kw)
        mock_wav2lip.side_effect = lambda *a, **kw: self._mock_run_wav2lip(*a, **kw)
        mock_remux.side_effect = lambda v, a, o: self._mock_remux(v, a, o)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            enhancements=["emotion_expressions", "eye_gaze_blink", "gesture_animation"],
        )

        assert result is not None

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_pipeline_with_all_enhancements(
        self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace
    ):
        """Pipeline runs with enhancements=["all"]."""
        from app.pipeline import render_pipeline

        mock_fomm.side_effect = lambda *a, **kw: self._mock_run_fomm(*a, **kw)
        mock_wav2lip.side_effect = lambda *a, **kw: self._mock_run_wav2lip(*a, **kw)
        mock_remux.side_effect = lambda v, a, o: self._mock_remux(v, a, o)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            enhancements=["all"],
        )

        assert result is not None


# ---------------------------------------------------------------------------
# Test: Individual pipeline stages
# ---------------------------------------------------------------------------

class TestPipelineStages:
    """Test helper functions in isolation."""

    def test_ffmpeg_binary_check(self):
        from app.pipeline import _ffmpeg_binary_available
        # Just verify it returns a bool without crashing
        result = _ffmpeg_binary_available()
        assert isinstance(result, bool)

    def test_ffprobe_binary_check(self):
        from app.pipeline import _ffprobe_binary_available
        result = _ffprobe_binary_available()
        assert isinstance(result, bool)

    def test_resolve_model_root(self):
        from app.pipeline import _resolve_model_root
        root = _resolve_model_root()
        assert isinstance(root, Path)

    def test_resolve_model_root_env_override(self):
        with patch.dict(os.environ, {"MODEL_ROOT": "/custom/models"}):
            from app.pipeline import _resolve_model_root
            result = _resolve_model_root()
            assert result == Path("/custom/models")

    def test_run_fomm_missing_external(self, tmp_workspace):
        """run_fomm raises when FOMM repo is not present."""
        from app.pipeline import run_fomm
        tmp = Path(tmp_workspace["tmp"])
        with patch("app.pipeline.EXT_DEPS_DIR", tmp / "nonexistent"):
            with pytest.raises(RuntimeError, match="FOMM not found"):
                run_fomm(tmp_workspace["face"], tmp_workspace["audio"], None, tmp)

    def test_encode_mp4_no_ffmpeg(self, fake_frames_dir, tmp_workspace):
        """encode_mp4 raises when ffmpeg is not available."""
        from app.pipeline import encode_mp4
        with patch("app.pipeline._ffmpeg_binary_available", return_value=False):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                encode_mp4(fake_frames_dir, tmp_workspace["audio"], tmp_workspace["output"])

    def test_encode_mp4_no_frames(self, tmp_path, tmp_workspace):
        """encode_mp4 raises when frame directory is empty."""
        from app.pipeline import encode_mp4
        empty_dir = tmp_path / "empty_frames"
        empty_dir.mkdir()
        with patch("app.pipeline._ffmpeg_binary_available", return_value=True):
            with pytest.raises(RuntimeError, match="No PNG frames"):
                encode_mp4(empty_dir, tmp_workspace["audio"], tmp_workspace["output"])

    def test_enhance_with_gfpgan_no_module(self, fake_frames_dir, tmp_path):
        """GFPGAN skips gracefully when not installed."""
        from app.pipeline import enhance_with_gfpgan
        result = enhance_with_gfpgan(fake_frames_dir, tmp_path)
        # Should return input unchanged since gfpgan module is not installed
        assert result == fake_frames_dir

    def test_wav2lip_missing_script(self, fake_frames_dir, tmp_workspace):
        """Wav2Lip returns frames unchanged when script is missing."""
        from app.pipeline import run_wav2lip
        tmp = Path(tmp_workspace["tmp"])
        with patch("app.pipeline.EXT_DEPS_DIR", tmp / "nonexistent"):
            result = run_wav2lip(fake_frames_dir, tmp_workspace["audio"], tmp)
        assert result == fake_frames_dir

    def test_verify_has_audio_no_ffprobe(self):
        """_verify_has_audio returns True when ffprobe is unavailable."""
        from app.pipeline import _verify_has_audio
        with patch("app.pipeline._ffprobe_binary_available", return_value=False):
            assert _verify_has_audio(Path("/any/file.mp4")) is True


# ---------------------------------------------------------------------------
# Test: Quality mode routing
# ---------------------------------------------------------------------------

class TestQualityModes:
    """Verify quality_mode affects pipeline behavior."""

    def _setup_mocks(self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux):
        def fake_fomm(face_img, audio_wav, ref_video, tmp_dir):
            frames = tmp_dir / "fomm_frames"
            frames.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
            return frames

        def fake_wav2lip(frames_dir, audio_wav, tmp_dir):
            out = tmp_dir / "wav2lip_out.mp4"
            out.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
            return out

        mock_fomm.side_effect = fake_fomm
        mock_wav2lip.side_effect = fake_wav2lip
        mock_remux.side_effect = lambda v, a, o: o.write_bytes(b"\x00" * 50)
        mock_gfpgan.exists.return_value = False

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_auto_mode(self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace):
        from app.pipeline import render_pipeline
        self._setup_mocks(mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux)
        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            quality_mode="auto",
        )
        assert result is not None

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_real_time_mode(self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace):
        from app.pipeline import render_pipeline
        self._setup_mocks(mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux)
        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            quality_mode="real_time",
        )
        assert result is not None

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_high_quality_mode(self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace):
        from app.pipeline import render_pipeline
        self._setup_mocks(mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux)
        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
            quality_mode="high_quality",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Test: Pipeline output verification
# ---------------------------------------------------------------------------

class TestPipelineOutput:
    """Verify pipeline produces correct output artifacts."""

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_output_file_created(self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace):
        from app.pipeline import render_pipeline

        def fake_fomm(face_img, audio_wav, ref_video, tmp_dir):
            frames = tmp_dir / "fomm_frames"
            frames.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
            return frames

        def fake_wav2lip(frames_dir, audio_wav, tmp_dir):
            out = tmp_dir / "wav2lip_out.mp4"
            out.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
            return out

        mock_fomm.side_effect = fake_fomm
        mock_wav2lip.side_effect = fake_wav2lip
        mock_remux.side_effect = lambda v, a, o: o.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 100)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
        )

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    @patch("app.pipeline._verify_has_audio", return_value=True)
    @patch("app.pipeline._remux_video_with_audio")
    @patch("app.pipeline.GFPGAN_CKPT")
    @patch("app.pipeline.run_wav2lip")
    @patch("app.pipeline.run_fomm")
    def test_output_is_absolute_path(self, mock_fomm, mock_wav2lip, mock_gfpgan, mock_remux, mock_verify, tmp_workspace):
        from app.pipeline import render_pipeline

        def fake_fomm(face_img, audio_wav, ref_video, tmp_dir):
            frames = tmp_dir / "fomm_frames"
            frames.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
            return frames

        def fake_wav2lip(frames_dir, audio_wav, tmp_dir):
            out = tmp_dir / "wav2lip_out.mp4"
            out.write_bytes(b"\x00" * 100)
            return out

        mock_fomm.side_effect = fake_fomm
        mock_wav2lip.side_effect = fake_wav2lip
        mock_remux.side_effect = lambda v, a, o: o.write_bytes(b"\x00" * 100)
        mock_gfpgan.exists.return_value = False

        result = render_pipeline(
            face_image=tmp_workspace["face"],
            audio=tmp_workspace["audio"],
            out_path=tmp_workspace["output"],
        )

        assert Path(result).is_absolute()


# ---------------------------------------------------------------------------
# Test: Simulated model checkpoint validation with PyTorch
# ---------------------------------------------------------------------------

class TestModelCheckpointSimulation:
    """Simulate loading model checkpoints the way the pipeline would."""

    def test_save_and_load_fomm_style_checkpoint(self, tmp_path):
        """Simulate FOMM checkpoint: a dict with 'generator' state_dict."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 7, padding=3),
        )
        ckpt = {"generator": model.state_dict(), "epoch": 100}
        ckpt_path = tmp_path / "vox-cpk.pth"
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "generator" in loaded
        assert loaded["epoch"] == 100

        # Verify we can load state_dict back
        model2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 7, padding=3),
        )
        model2.load_state_dict(loaded["generator"])

    def test_save_and_load_wav2lip_style_checkpoint(self, tmp_path):
        """Simulate Wav2Lip checkpoint: {'state_dict': ..., 'optimizer': ...}."""
        model = torch.nn.Linear(256, 128)
        ckpt = {
            "state_dict": model.state_dict(),
            "optimizer": {"lr": 1e-4},
            "global_step": 50000,
        }
        ckpt_path = tmp_path / "wav2lip_gan.pth"
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "state_dict" in loaded
        assert loaded["global_step"] == 50000

    def test_save_and_load_gfpgan_style_checkpoint(self, tmp_path):
        """Simulate GFPGAN checkpoint with nested params."""
        params = {
            "params_ema": torch.nn.Linear(512, 512).state_dict(),
            "params": torch.nn.Linear(512, 512).state_dict(),
        }
        ckpt_path = tmp_path / "GFPGANv1.3.pth"
        torch.save(params, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "params_ema" in loaded

    def test_checkpoint_device_mapping(self, tmp_path):
        """Verify map_location='cpu' works for GPU-saved checkpoints."""
        model = torch.nn.Linear(100, 50)
        ckpt_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), ckpt_path)

        # Simulate loading a GPU checkpoint on CPU
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model2 = torch.nn.Linear(100, 50)
        model2.load_state_dict(state)

        # Verify inference works
        x = torch.randn(1, 100)
        y = model2(x)
        assert y.shape == (1, 50)


# ---------------------------------------------------------------------------
# Test: Enhancement context creation within pipeline
# ---------------------------------------------------------------------------

class TestEnhancementContextInPipeline:
    """Test that enhancement context is properly created inside the pipeline."""

    def test_enhancement_context_dataclass(self):
        from app.enhancements import EnhancementContext
        ctx = EnhancementContext(
            face_image="test.png",
            audio_path="test.wav",
            quality_mode="auto",
        )
        assert ctx.face_image == "test.png"
        assert ctx.detected_emotion is None
        assert ctx.applied_enhancements == []

    def test_enhancement_registry_available(self):
        from app.enhancements import registry
        names = registry.list_names()
        assert isinstance(names, list)
        assert len(names) >= 3  # At least emotion, eye_gaze, gesture

    def test_enhancement_registry_lists_stages(self):
        from app.enhancements import registry
        all_enh = registry.list_all()
        stages = {e.stage for e in all_enh}
        # Verify all 5 pipeline stages have at least one enhancement
        expected = {"tts", "pre_motion", "motion_driver", "lip_sync", "post_process"}
        assert stages == expected


# ---------------------------------------------------------------------------
# Test: Edge cases and error recovery
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    """Test pipeline robustness and error handling."""

    def test_output_directory_auto_created(self, tmp_workspace):
        """Pipeline creates output directory if it doesn't exist."""
        deep_output = str(Path(tmp_workspace["tmp"]) / "a" / "b" / "c" / "output.mp4")

        with patch("app.pipeline.run_fomm") as mock_fomm, \
             patch("app.pipeline.run_wav2lip") as mock_wav2lip, \
             patch("app.pipeline.GFPGAN_CKPT") as mock_gfpgan, \
             patch("app.pipeline._remux_video_with_audio") as mock_remux, \
             patch("app.pipeline._verify_has_audio", return_value=True):

            def fake_fomm(face_img, audio_wav, ref_video, tmp_dir):
                frames = tmp_dir / "fomm_frames"
                frames.mkdir(parents=True, exist_ok=True)
                for i in range(3):
                    (frames / f"{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
                return frames

            def fake_wav2lip(frames_dir, audio_wav, tmp_dir):
                out = tmp_dir / "wav2lip_out.mp4"
                out.write_bytes(b"\x00" * 100)
                return out

            mock_fomm.side_effect = fake_fomm
            mock_wav2lip.side_effect = fake_wav2lip
            mock_remux.side_effect = lambda v, a, o: o.write_bytes(b"\x00" * 100)
            mock_gfpgan.exists.return_value = False

            from app.pipeline import render_pipeline
            result = render_pipeline(
                face_image=tmp_workspace["face"],
                audio=tmp_workspace["audio"],
                out_path=deep_output,
            )

            assert Path(result).parent.exists()

    def test_mpi4py_mock_creation(self, tmp_path):
        """Verify the mpi4py mock helper creates a valid package."""
        from app.pipeline import _create_mpi4py_mock
        _create_mpi4py_mock(tmp_path)

        pkg = tmp_path / "mpi4py" / "__init__.py"
        assert pkg.exists()

        # Verify it's importable
        import importlib.util
        spec = importlib.util.spec_from_file_location("mpi4py", str(pkg))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "MPI")
        assert mod.MPI.COMM_WORLD.Get_rank() == 0
        assert mod.MPI.COMM_WORLD.Get_size() == 1
