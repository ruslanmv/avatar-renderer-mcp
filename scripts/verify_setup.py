#!/usr/bin/env python3
"""
Quick verification script for Avatar Renderer MCP setup.
Run this to check if everything is configured correctly.

Usage:
    python scripts/verify_setup.py
    make verify  # Alternative using Makefile
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def check_python_dependency(module_name: str, display_name: str, 
                            required: bool = True) -> bool:
    """Check if a Python module is installed."""
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print_success(f"{display_name:25} (version: {version})")
        return True
    except ImportError:
        if required:
            print_error(f"{display_name:25} - MISSING (required)")
        else:
            print_warning(f"{display_name:25} - MISSING (optional)")
        return not required


def check_numpy_version() -> bool:
    """Check if numpy version is compatible (<2.0.0)."""
    try:
        import numpy as np
        version = np.__version__
        major = int(version.split('.')[0])
        
        if major < 2:
            print_success(f"numpy                     (version: {version}) - Compatible")
            return True
        else:
            print_error(f"numpy                     (version: {version}) - INCOMPATIBLE")
            print(f"  {Colors.YELLOW}→ Requires numpy <2.0.0 for Wav2Lip/SadTalker compatibility{Colors.RESET}")
            return False
    except ImportError:
        print_error(f"numpy                     - MISSING (required)")
        return False


def check_librosa_version() -> bool:
    """Check if librosa version is compatible (==0.9.2)."""
    try:
        import librosa
        version = librosa.__version__
        
        if version.startswith('0.9'):
            print_success(f"librosa                   (version: {version}) - Compatible")
            return True
        else:
            print_warning(f"librosa                   (version: {version}) - May cause issues")
            print(f"  {Colors.YELLOW}→ Recommended: librosa==0.9.2 for Wav2Lip compatibility{Colors.RESET}")
            return True  # Warning, not error
    except ImportError:
        print_error(f"librosa                   - MISSING (required)")
        return False


def check_guided_diffusion() -> bool:
    """Check if guided_diffusion is installed (critical for Diff2Lip)."""
    try:
        import guided_diffusion
        print_success(f"guided_diffusion          - Installed")
        return True
    except ImportError:
        print_error(f"guided_diffusion          - MISSING (required for Diff2Lip)")
        print(f"  {Colors.YELLOW}→ Install: pip install git+https://github.com/openai/guided-diffusion.git{Colors.RESET}")
        return False


def check_file_exists(path: Path, description: str, required: bool = True) -> bool:
    """Check if a file or directory exists."""
    exists = path.exists()
    
    if exists:
        size_info = ""
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            size_info = f" ({size_mb:.2f} MB)"
        print_success(f"{description:25} {size_info}")
    else:
        if required:
            print_error(f"{description:25} - MISSING")
        else:
            print_warning(f"{description:25} - MISSING (optional)")
    
    return exists or not required


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path:
        print_success(f"ffmpeg                    - Available at {ffmpeg_path}")
        return True
    else:
        print_error(f"ffmpeg                    - NOT FOUND in PATH")
        print(f"  {Colors.YELLOW}→ Install ffmpeg for video processing{Colors.RESET}")
        return False


def check_cuda() -> Tuple[bool, Optional[str]]:
    """Check CUDA availability and GPU info."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_success(f"CUDA available")
            print(f"  GPU: {device_name}")
            print(f"  Memory: {memory_gb:.2f} GB")
            return True, device_name
        else:
            print_warning(f"CUDA not available - CPU mode will be used")
            print(f"  {Colors.YELLOW}Note: High-quality mode is much slower on CPU{Colors.RESET}")
            return False, None
    except ImportError:
        print_error(f"PyTorch not installed - Cannot check CUDA")
        return False, None


def check_setup() -> Dict[str, bool]:
    """Verify Avatar Renderer MCP setup."""
    print("=" * 70)
    print(f"{Colors.BOLD}Avatar Renderer MCP - Setup Verification{Colors.RESET}")
    print("=" * 70)

    results = {}

    # Get repository root
    repo_root = Path.cwd()
    print(f"\n📁 Repository root: {repo_root}")

    # Check environment variables
    print_header("🔧 Environment Variables")
    
    model_root = os.environ.get("MODEL_ROOT")
    ext_deps_dir = os.environ.get("EXT_DEPS_DIR")

    if model_root:
        print_success(f"MODEL_ROOT   = {model_root}")
    else:
        model_root = str(repo_root / "models")
        print_warning(f"MODEL_ROOT not set, using default: {model_root}")

    if ext_deps_dir:
        print_success(f"EXT_DEPS_DIR = {ext_deps_dir}")
    else:
        ext_deps_dir = str(repo_root / "external_deps")
        print_warning(f"EXT_DEPS_DIR not set, using default: {ext_deps_dir}")

    model_root = Path(model_root)
    ext_deps_dir = Path(ext_deps_dir)

    # Check Python dependencies - CRITICAL
    print_header("🐍 Critical Python Dependencies")
    
    critical_deps = [
        ("torch", "torch (PyTorch)"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("soundfile", "soundfile"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
        ("imageio", "imageio"),
        ("ffmpeg", "ffmpeg-python"),
    ]
    
    critical_ok = all(check_python_dependency(mod, name) for mod, name in critical_deps)
    
    # Check version-specific dependencies
    print_header("🔍 Version-Specific Dependencies")
    numpy_ok = check_numpy_version()
    librosa_ok = check_librosa_version()
    guided_diff_ok = check_guided_diffusion()
    
    version_ok = numpy_ok and librosa_ok and guided_diff_ok
    
    # Check optional Python dependencies
    print_header("🔌 Optional Python Dependencies")
    
    optional_deps = [
        ("gfpgan", "gfpgan"),
        ("facexlib", "facexlib"),
        ("basicsr", "basicsr"),
        ("face_alignment", "face-alignment"),
        ("kornia", "kornia"),
        ("mediapipe", "mediapipe"),
        ("yacs", "yacs"),
        ("tqdm", "tqdm"),
    ]
    
    for mod, name in optional_deps:
        check_python_dependency(mod, name, required=False)

    # Check model checkpoints
    print_header("📦 Model Checkpoints")
    
    checkpoints = {
        "FOMM": (model_root / "fomm" / "vox-cpk.pth", True),
        "Diff2Lip": (model_root / "diff2lip" / "Diff2Lip.pth", False),
        "Wav2Lip": (model_root / "wav2lip" / "wav2lip_gan.pth", True),
        "SadTalker": (model_root / "sadtalker" / "sadtalker.pth", False),
        "GFPGAN": (model_root / "gfpgan" / "GFPGANv1.3.pth", False),
    }

    models_ok = True
    hq_models_ok = True
    for name, (path, required) in checkpoints.items():
        exists = check_file_exists(path, name, required)
        if not exists and required:
            models_ok = False
        if name in ["FOMM", "Diff2Lip", "GFPGAN"] and not exists:
            hq_models_ok = False

    results['models'] = models_ok

    # Check external dependencies
    print_header("🔗 External Git Dependencies")
    
    ext_deps_checks = {
        "first-order-model": (ext_deps_dir / "first-order-model", True),
        "FOMM wrapper": (ext_deps_dir / "first-order-model" / "fomm_wrapper.py", False),
        "FOMM config": (ext_deps_dir / "first-order-model" / "config" / "vox-256.yaml", True),
        "SadTalker": (ext_deps_dir / "SadTalker", False),
        "Wav2Lip": (ext_deps_dir / "Wav2Lip", True),
        "Wav2Lip inference": (ext_deps_dir / "Wav2Lip" / "inference.py", True),
        "Diff2Lip": (ext_deps_dir / "Diff2Lip", False),
        "Diff2Lip generate": (ext_deps_dir / "Diff2Lip" / "generate.py", False),
    }

    git_deps_ok = True
    diff2lip_available = False
    for name, (path, required) in ext_deps_checks.items():
        exists = check_file_exists(path, name, required)
        if not exists and required:
            git_deps_ok = False
        if "Diff2Lip" in name and exists:
            diff2lip_available = True

    results['git_deps'] = git_deps_ok

    # Check system dependencies
    print_header("🖥️  System Dependencies")
    ffmpeg_ok = check_ffmpeg()
    
    # Check CUDA
    print_header("🎮 GPU/CUDA")
    cuda_ok, gpu_name = check_cuda()

    # Summary
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}📊 Summary{Colors.RESET}")
    print("=" * 70)

    all_critical_ok = critical_ok and version_ok and git_deps_ok and models_ok and ffmpeg_ok

    # Mode availability
    print(f"\n{Colors.BOLD}Pipeline Modes:{Colors.RESET}")
    
    if all_critical_ok and hq_models_ok and diff2lip_available and guided_diff_ok:
        print_success("High-Quality Mode (FOMM + Diff2Lip):  READY")
    elif not guided_diff_ok:
        print_error("High-Quality Mode (FOMM + Diff2Lip):  BLOCKED - guided_diffusion missing")
        print(f"  {Colors.YELLOW}→ Install: pip install git+https://github.com/openai/guided-diffusion.git{Colors.RESET}")
    elif not diff2lip_available:
        print_warning("High-Quality Mode (FOMM + Diff2Lip):  UNAVAILABLE - Diff2Lip not cloned")
        print(f"  {Colors.YELLOW}→ Run: make install-git-deps{Colors.RESET}")
    else:
        print_error("High-Quality Mode (FOMM + Diff2Lip):  NOT READY")

    if all_critical_ok:
        print_success("Standard Mode (FOMM + Wav2Lip):        READY")
    else:
        print_error("Standard Mode (FOMM + Wav2Lip):        NOT READY")

    # Recommendations
    if not all_critical_ok:
        print(f"\n{Colors.BOLD}🔧 Recommended Fixes:{Colors.RESET}")
        
        if not critical_ok:
            print(f"  1. Install Python dependencies:")
            print(f"     {Colors.BLUE}make install{Colors.RESET}")
            print(f"     or: {Colors.BLUE}uv pip install -e .{Colors.RESET}")
        
        if not version_ok:
            if not numpy_ok:
                print(f"  2. Fix numpy version:")
                print(f"     {Colors.BLUE}uv pip install 'numpy<2.0.0'{Colors.RESET}")
            if not librosa_ok:
                print(f"  3. Fix librosa version:")
                print(f"     {Colors.BLUE}uv pip install 'librosa==0.9.2'{Colors.RESET}")
            if not guided_diff_ok:
                print(f"  4. Install guided_diffusion:")
                print(f"     {Colors.BLUE}uv pip install git+https://github.com/openai/guided-diffusion.git{Colors.RESET}")
        
        if not git_deps_ok:
            print(f"  5. Clone external dependencies:")
            print(f"     {Colors.BLUE}make install-git-deps{Colors.RESET}")
        
        if not models_ok:
            print(f"  6. Download model checkpoints:")
            print(f"     {Colors.BLUE}make download-models{Colors.RESET}")
        
        if not ffmpeg_ok:
            print(f"  7. Install ffmpeg:")
            print(f"     Ubuntu/Debian: {Colors.BLUE}sudo apt-get install ffmpeg{Colors.RESET}")
            print(f"     macOS:         {Colors.BLUE}brew install ffmpeg{Colors.RESET}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All critical dependencies are ready!{Colors.RESET}")
        
        if cuda_ok:
            print(f"{Colors.GREEN}✅ GPU acceleration available{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠  GPU not available - will run on CPU (slower){Colors.RESET}")
    
    # Quick start
    if all_critical_ok:
        print(f"\n{Colors.BOLD}🚀 Quick Start:{Colors.RESET}")
        print(f"  Run demo:    {Colors.BLUE}make demo{Colors.RESET}")
        print(f"  Start API:   {Colors.BLUE}make run{Colors.RESET}")
        print(f"  Start MCP:   {Colors.BLUE}make run-stdio{Colors.RESET}")

    print()
    return all_critical_ok


def main() -> int:
    """Main entry point."""
    try:
        success = check_setup()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}Error during verification: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())