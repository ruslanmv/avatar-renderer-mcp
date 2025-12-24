#!/usr/bin/env python3
"""
Quick verification script for Avatar Renderer MCP setup.
Run this to check if everything is configured correctly.
"""

import os
import sys
from pathlib import Path

def check_setup():
    """Verify Avatar Renderer MCP setup."""
    print("=" * 60)
    print("Avatar Renderer MCP - Setup Verification")
    print("=" * 60)
    print()

    # Get repository root
    repo_root = Path.cwd()
    print(f"📁 Repository root: {repo_root}")
    print()

    # Check environment variables
    print("🔧 Environment Variables:")
    model_root = os.environ.get("MODEL_ROOT")
    ext_deps_dir = os.environ.get("EXT_DEPS_DIR")

    if model_root:
        print(f"  ✓ MODEL_ROOT   = {model_root}")
    else:
        model_root = repo_root / "models"
        print(f"  ⚠ MODEL_ROOT not set, using default: {model_root}")

    if ext_deps_dir:
        print(f"  ✓ EXT_DEPS_DIR = {ext_deps_dir}")
    else:
        ext_deps_dir = repo_root / "external_deps"
        print(f"  ⚠ EXT_DEPS_DIR not set, using default: {ext_deps_dir}")

    model_root = Path(model_root)
    ext_deps_dir = Path(ext_deps_dir)
    print()

    # Check model checkpoints
    print("📦 Model Checkpoints:")
    checkpoints = {
        "FOMM": model_root / "fomm" / "vox-cpk.pth",
        "Diff2Lip": model_root / "diff2lip" / "Diff2Lip.pth",
        "SadTalker": model_root / "sadtalker" / "sadtalker.pth",
        "Wav2Lip": model_root / "wav2lip" / "wav2lip_gan.pth",
        "GFPGAN": model_root / "gfpgan" / "GFPGANv1.3.pth",
    }

    all_models_exist = True
    hq_models_exist = True
    rt_models_exist = True

    for name, path in checkpoints.items():
        exists = path.exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {name:15} {path}")

        if not exists:
            all_models_exist = False
            if name in ["FOMM", "Diff2Lip", "GFPGAN"]:
                hq_models_exist = False
            if name in ["SadTalker", "Wav2Lip"]:
                rt_models_exist = False
    print()

    # Check external dependencies
    print("🔗 External Dependencies:")
    ext_deps = {
        "first-order-model": ext_deps_dir / "first-order-model",
        "FOMM wrapper": ext_deps_dir / "first-order-model" / "fomm_wrapper.py",
        "FOMM config": ext_deps_dir / "first-order-model" / "config" / "vox-256.yaml",
        "SadTalker": ext_deps_dir / "SadTalker",
        "Wav2Lip": ext_deps_dir / "Wav2Lip",
    }

    all_deps_exist = True
    for name, path in ext_deps.items():
        exists = path.exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {name:20} {path}")
        if not exists:
            all_deps_exist = False
    print()

    # Check Python dependencies
    print("🐍 Python Dependencies:")
    python_deps = [
        ("torch", "PyTorch"),
        ("ffmpeg", "ffmpeg-python"),
        ("yacs", "YACS"),
        ("imageio", "imageio"),
        ("skimage", "scikit-image"),
    ]

    all_python_deps = True
    for module, name in python_deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (install with: pip install {name.lower().replace(' ', '-')})")
            all_python_deps = False
    print()

    # Check CUDA
    print("🖥️  GPU/CUDA:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"  ⚠ CUDA not available (CPU mode will be used)")
            print(f"    Note: High-quality mode requires GPU")
    except ImportError:
        print(f"  ✗ PyTorch not installed")
    print()

    # Summary
    print("=" * 60)
    print("📊 Summary:")
    print("=" * 60)

    if all_deps_exist and hq_models_exist and all_python_deps:
        print("✅ High-quality mode: READY")
    elif not all_deps_exist:
        print("❌ High-quality mode: Missing external dependencies")
        print("   Fix: make install-git-deps")
    elif not hq_models_exist:
        print("❌ High-quality mode: Missing HQ model checkpoints")
        print("   Fix: make download-models")
    else:
        print("❌ High-quality mode: Missing Python dependencies")
        print("   Fix: pip install -e .")

    if all_deps_exist and rt_models_exist and all_python_deps:
        print("✅ Real-time mode: READY")
    elif not all_deps_exist:
        print("❌ Real-time mode: Missing external dependencies")
        print("   Fix: make install-git-deps")
    elif not rt_models_exist:
        print("❌ Real-time mode: Missing RT model checkpoints")
        print("   Fix: make download-models")
    else:
        print("❌ Real-time mode: Missing Python dependencies")
        print("   Fix: pip install -e .")

    print()

    # Recommendations
    if not all_models_exist or not all_deps_exist or not all_python_deps:
        print("🔧 Recommended fixes:")
        if not all_deps_exist:
            print("  1. Run: make install-git-deps")
        if not all_models_exist:
            print("  2. Run: make download-models")
        if not all_python_deps:
            print("  3. Run: pip install -e .")
        print()

    return all_models_exist and all_deps_exist and all_python_deps


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
