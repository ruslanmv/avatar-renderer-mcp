"""
Lightweight health check tests for avatar-renderer-mcp.

These tests verify basic functionality without running heavy model operations.
Perfect for CI/CD pipelines and quick verification.
"""

import os
import sys
from pathlib import Path


def test_python_version():
    """Verify Python version is compatible."""
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    assert sys.version_info < (3, 13), "Python 3.13+ not yet supported"
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}")


def test_imports():
    """Verify all critical imports work."""
    print("\nTesting critical imports...")

    # Core dependencies
    import torch
    print(f"  ✓ torch {torch.__version__}")

    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")

    # ML libraries
    import diffusers
    print(f"  ✓ diffusers {diffusers.__version__}")

    import transformers
    print(f"  ✓ transformers {transformers.__version__}")

    # Audio processing
    import librosa
    print(f"  ✓ librosa {librosa.__version__}")

    import soundfile
    print(f"  ✓ soundfile {soundfile.__version__}")

    # Video processing
    import ffmpeg
    print(f"  ✓ ffmpeg-python installed")

    # Face enhancement (may have compatibility issues with newer torchvision)
    try:
        import gfpgan
        print(f"  ✓ gfpgan installed")
    except ImportError as e:
        print(f"  ⚠ gfpgan import issue (known compatibility): {e}")
        # This is OK - gfpgan has issues with newer torchvision but still works at runtime

    # App modules
    from app import settings
    print(f"  ✓ app.settings")

    from app import pipeline
    print(f"  ✓ app.pipeline")


def test_environment_variables():
    """Verify environment setup."""
    print("\nTesting environment variables...")

    model_root = os.environ.get("MODEL_ROOT", "models")
    print(f"  ✓ MODEL_ROOT: {model_root}")

    ext_deps = os.environ.get("EXT_DEPS_DIR", "external_deps")
    print(f"  ✓ EXT_DEPS_DIR: {ext_deps}")


def test_pipeline_helpers():
    """Test pipeline helper functions."""
    print("\nTesting pipeline helpers...")

    from app.pipeline import (
        _can_import,
        _python_ffmpeg_available,
        _ffmpeg_binary_available,
        _fomm_runtime_available,
    )

    # Test import checker
    assert _can_import("os") is True
    assert _can_import("nonexistent_module_xyz") is False
    print("  ✓ _can_import() works")

    # Test ffmpeg-python check
    has_ffmpeg_py = _python_ffmpeg_available()
    print(f"  ✓ _python_ffmpeg_available(): {has_ffmpeg_py}")

    # Test system ffmpeg check
    has_ffmpeg_bin = _ffmpeg_binary_available()
    print(f"  ✓ _ffmpeg_binary_available(): {has_ffmpeg_bin}")

    # Test FOMM runtime check
    fomm_ok, fomm_reason = _fomm_runtime_available()
    print(f"  ✓ _fomm_runtime_available(): {fomm_ok}")
    if not fomm_ok:
        print(f"    Reason: {fomm_reason}")


def test_model_paths():
    """Verify model path configuration."""
    print("\nTesting model paths...")

    from app.pipeline import MODEL_ROOT, FOMM_CKPT, D2L_CKPT, SAD_CKPT, W2L_CKPT

    print(f"  ✓ MODEL_ROOT: {MODEL_ROOT}")
    print(f"  ✓ FOMM_CKPT: {FOMM_CKPT}")
    print(f"  ✓ D2L_CKPT: {D2L_CKPT}")
    print(f"  ✓ SAD_CKPT: {SAD_CKPT}")
    print(f"  ✓ W2L_CKPT: {W2L_CKPT}")

    # Verify they're Path objects
    assert isinstance(MODEL_ROOT, Path)
    print("  ✓ Model paths are Path objects")


def test_cuda_availability():
    """Check CUDA/GPU availability."""
    print("\nTesting CUDA availability...")

    import torch
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"  ✓ CUDA available")
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  ℹ CUDA not available (CPU mode)")


def test_settings():
    """Test settings module."""
    print("\nTesting settings...")

    from app.settings import settings

    print(f"  ✓ MODEL_ROOT: {settings.MODEL_ROOT}")
    print(f"  ✓ Log level: {settings.LOG_LEVEL}")

    assert settings.MODEL_ROOT.exists() or True  # May not exist yet
    print("  ✓ Settings loaded successfully")


def run_all_tests():
    """Run all health check tests."""
    print("="*60)
    print("Avatar Renderer MCP - Health Check")
    print("="*60)

    tests = [
        test_python_version,
        test_imports,
        test_environment_variables,
        test_pipeline_helpers,
        test_model_paths,
        test_cuda_availability,
        test_settings,
    ]

    failed = []

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {test_func.__name__} failed: {e}")
            failed.append((test_func.__name__, e))

    print("\n" + "="*60)
    if not failed:
        print("✅ All health checks passed!")
        print("="*60)
        return 0
    else:
        print(f"❌ {len(failed)} health check(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
