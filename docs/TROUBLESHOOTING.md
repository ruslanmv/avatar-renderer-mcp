# Troubleshooting Guide

This guide helps you resolve common issues when setting up and using Avatar Renderer MCP.

## Table of Contents

1. [FOMM "main function not found" Error](#fomm-main-function-not-found-error)
2. [External Dependencies Missing](#external-dependencies-missing)
3. [GPU/CUDA Issues](#gpucuda-issues)
4. [Model Download Issues](#model-download-issues)
5. [Installation Verification](#installation-verification)

---

## FOMM "main function not found" Error

### Symptom

```
RuntimeError: FOMM module loaded but 'main' function not found
```

### Root Cause

The first-order-model repository's `demo.py` doesn't have a `main()` function that the pipeline expects. The code is in the `if __name__ == "__main__":` block.

### Solution

This has been fixed in the latest version. The fix includes:

1. **FOMM Wrapper** (`external_deps/first-order-model/fomm_wrapper.py`):
   - Provides a `main()` function interface
   - Handles both video-driven and audio-driven modes
   - Manages frame export to the pipeline

2. **Updated Pipeline** (`app/pipeline.py`):
   - Modified `run_fomm()` to use the wrapper
   - Added fallback to demo.py for compatibility
   - Enhanced error messages

### Verification Steps

1. **Check external dependencies exist:**
   ```bash
   ls external_deps/first-order-model/
   # Should show: demo.py, fomm_wrapper.py, config/, modules/, etc.
   ```

2. **Verify wrapper is being used:**
   ```bash
   ls external_deps/first-order-model/fomm_wrapper.py
   # Should exist
   ```

3. **Test the pipeline:**
   ```python
   from app.pipeline import render_pipeline

   result = render_pipeline(
       face_image="tests/assets/alice.png",
       audio="tests/assets/hello.wav",
       out_path="test_output.mp4",
       quality_mode="high_quality"
   )
   ```

---

## External Dependencies Missing

### Symptom

```
Missing first-order-model repo at external_deps/first-order-model
```

or

```
✓ first-order-model repo: ✓  # But still fails
```

### Root Cause

The `external_deps/` directory doesn't exist or is empty. The external repositories (SadTalker, Wav2Lip, first-order-model) were not cloned.

### Solution

#### Method 1: Using Make (Recommended)

```bash
make install-git-deps
```

This will:
- Clone SadTalker, Wav2Lip, and first-order-model
- Install minimal required dependencies (yacs, pyyaml, imageio)
- Set up the FOMM wrapper

#### Method 2: Manual Installation

```bash
# Create external_deps directory
mkdir -p external_deps

# Clone repositories
git clone --depth=1 https://github.com/OpenTalker/SadTalker.git external_deps/SadTalker
git clone --depth=1 https://github.com/AliaksandrSiarohin/first-order-model.git external_deps/first-order-model
git clone --depth=1 https://github.com/Rudrabha/Wav2Lip.git external_deps/Wav2Lip

# Install dependencies
pip install ffmpeg-python yacs pyyaml imageio scikit-image
```

### Verification

```bash
# Check directories exist
ls -la external_deps/
# Should show: SadTalker/, first-order-model/, Wav2Lip/

# Check key files
ls external_deps/first-order-model/demo.py
ls external_deps/first-order-model/fomm_wrapper.py
ls external_deps/first-order-model/config/vox-256.yaml
```

---

## GPU/CUDA Issues

### Symptom

```
CUDA available: ✗
```

or

```
high_quality mode requires GPU (CUDA). Use real_time mode instead.
```

### Solutions

#### Option 1: Use Real-Time Mode (CPU-Compatible)

```python
result = render_pipeline(
    face_image="path/to/image.png",
    audio="path/to/audio.wav",
    out_path="output.mp4",
    quality_mode="real_time"  # Works on CPU
)
```

#### Option 2: Install CUDA Support

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   ```

2. **Install PyTorch with CUDA:**
   ```bash
   # For CUDA 12.4
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Verify CUDA in Python:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

---

## Model Download Issues

### Symptom

```
Model checkpoint not found: models/fomm/vox-cpk.pth
```

### Solution

```bash
# Download all model checkpoints (~3GB)
make download-models
```

This downloads:
- FOMM checkpoint (`models/fomm/vox-cpk.pth`)
- Diff2Lip checkpoint (`models/diff2lip/Diff2Lip.pth`)
- SadTalker checkpoint (`models/sadtalker/sadtalker.pth`)
- Wav2Lip checkpoint (`models/wav2lip/wav2lip_gan.pth`)
- GFPGAN checkpoint (`models/gfpgan/GFPGANv1.3.pth`)

### Manual Download

If `make download-models` fails, check `scripts/download_models.sh` for direct download links.

---

## Installation Verification

### Complete Setup Checklist

Run this checklist to verify your installation:

```bash
# 1. Check Python version
python --version  # Should be 3.11+

# 2. Install dependencies
make install

# 3. Install external Git repos
make install-git-deps

# 4. Download models
make download-models

# 5. Verify installation
python scripts/verify_installation.py
```

### Quick Health Check

```python
from pathlib import Path
import os, sys
import torch

# Set environment
REPO_ROOT = Path.cwd()
os.environ.setdefault("MODEL_ROOT", str((REPO_ROOT / "models").resolve()))
os.environ.setdefault("EXT_DEPS_DIR", str((REPO_ROOT / "external_deps").resolve()))

MODEL_ROOT = Path(os.environ["MODEL_ROOT"])
EXT_DEPS_DIR = Path(os.environ["EXT_DEPS_DIR"])

# Check models
has_realtime = (MODEL_ROOT/"sadtalker"/"sadtalker.pth").exists() and \
               (MODEL_ROOT/"wav2lip"/"wav2lip_gan.pth").exists()
has_hq_models = (MODEL_ROOT/"diff2lip"/"Diff2Lip.pth").exists() and \
                (MODEL_ROOT/"fomm"/"vox-cpk.pth").exists()

# Check external deps
has_fomm_repo = (EXT_DEPS_DIR/"first-order-model").exists()
has_fomm_wrapper = (EXT_DEPS_DIR/"first-order-model"/"fomm_wrapper.py").exists()
has_fomm_config = (EXT_DEPS_DIR/"first-order-model"/"config"/"vox-256.yaml").exists()

print("✅ Installation Status:")
print(f"  Real-time models:    {'✓' if has_realtime else '✗'}")
print(f"  HQ models:           {'✓' if has_hq_models else '✗'}")
print(f"  FOMM repo:           {'✓' if has_fomm_repo else '✗'}")
print(f"  FOMM wrapper:        {'✓' if has_fomm_wrapper else '✗'}")
print(f"  FOMM config:         {'✓' if has_fomm_config else '✗'}")
print(f"  CUDA available:      {'✓' if torch.cuda.is_available() else '✗'}")
```

---

## Common Error Messages

### Error: "No virtual environment found"

**Solution:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Then retry installation
make install
```

### Error: "ffmpeg not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

### Error: "Missing python package 'ffmpeg'"

**Solution:**
```bash
pip install ffmpeg-python
```

### Error: "Cannot import name 'main' from demo"

This is the FOMM wrapper issue. See [FOMM "main function not found" Error](#fomm-main-function-not-found-error) above.

---

## Getting Help

If you're still experiencing issues:

1. **Check logs:** Look for detailed error messages in the console output
2. **Run verification:** `python scripts/verify_installation.py`
3. **Check GitHub Issues:** [https://github.com/ruslanmv/avatar-renderer-mcp/issues](https://github.com/ruslanmv/avatar-renderer-mcp/issues)
4. **Report a bug:** Include:
   - Python version
   - OS and version
   - Full error traceback
   - Output of health check script above

---

## Summary of Fixes Applied

### December 2024 - FOMM Integration Fix

**Files Modified:**
- `external_deps/first-order-model/fomm_wrapper.py` (new)
- `app/pipeline.py` (updated `run_fomm()` and `_fomm_runtime_available()`)

**Changes:**
1. Created FOMM wrapper with `main()` function interface
2. Updated pipeline to use wrapper instead of demo.py directly
3. Added proper CPU/GPU detection
4. Enhanced error messages with specific troubleshooting steps
5. Added config directory validation

**Result:** High-quality mode now works correctly with FOMM!
