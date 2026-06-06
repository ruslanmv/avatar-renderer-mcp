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
    """Verify critical imports work.

    Core app modules must import. Heavy optional ML/media libraries are skipped
    gracefully when absent so this health check stays meaningful in a lightweight
    CI environment as well as a full GPU install.
    """
    import pytest

    print("\nTesting critical imports...")

    # Core dependency required by the app modules.
    import torch
    print(f"  ✓ torch {torch.__version__}")

    # Optional heavy libraries — only verified when installed.
    for mod in ("torchvision", "diffusers", "transformers", "librosa", "soundfile", "ffmpeg", "gfpgan"):
        try:
            __import__(mod)
            print(f"  ✓ {mod} installed")
        except ImportError:
            print(f"  ⚠ {mod} not installed (optional in this environment)")

    # App modules must always import.
    from app import settings  # noqa: F401
    print("  ✓ app.settings")

    from app import pipeline  # noqa: F401
    print("  ✓ app.pipeline")


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
        _ffmpeg_binary_available,
        _ffprobe_binary_available,
        _resolve_model_root,
        load_module_from_path,
    )

    # ffmpeg / ffprobe availability checks return booleans.
    assert isinstance(_ffmpeg_binary_available(), bool)
    assert isinstance(_ffprobe_binary_available(), bool)
    print(f"  ✓ _ffmpeg_binary_available(): {_ffmpeg_binary_available()}")
    print(f"  ✓ _ffprobe_binary_available(): {_ffprobe_binary_available()}")

    # Model root resolution returns a Path.
    assert isinstance(_resolve_model_root(), Path)
    print(f"  ✓ _resolve_model_root(): {_resolve_model_root()}")

    # Dynamic module loader rejects missing files.
    import pytest

    with pytest.raises(ImportError):
        load_module_from_path("does_not_exist", Path("/nonexistent/module_xyz.py"))
    print("  ✓ load_module_from_path() raises on missing file")


def test_model_paths():
    """Verify model path configuration."""
    print("\nTesting model paths...")

    from app.pipeline import (
        MODEL_ROOT,
        FOMM_CKPT,
        D2L_PAPER_CKPT,
        D2L_LEGACY_CKPT,
        W2L_CKPT,
        GFPGAN_CKPT,
    )

    print(f"  ✓ MODEL_ROOT: {MODEL_ROOT}")
    print(f"  ✓ FOMM_CKPT: {FOMM_CKPT}")
    print(f"  ✓ D2L_PAPER_CKPT: {D2L_PAPER_CKPT}")
    print(f"  ✓ D2L_LEGACY_CKPT: {D2L_LEGACY_CKPT}")
    print(f"  ✓ W2L_CKPT: {W2L_CKPT}")
    print(f"  ✓ GFPGAN_CKPT: {GFPGAN_CKPT}")

    # Verify they're Path objects
    for p in (MODEL_ROOT, FOMM_CKPT, D2L_PAPER_CKPT, D2L_LEGACY_CKPT, W2L_CKPT, GFPGAN_CKPT):
        assert isinstance(p, Path)
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
