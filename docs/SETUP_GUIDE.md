# Complete Setup Guide - Avatar Renderer MCP

This guide walks you through setting up Avatar Renderer MCP in your environment.

##  Critical Fixes Applied

We've fixed two major issues:

### 1. FOMM "main function not found" Error
- **Problem**: first-order-model/demo.py doesn't have a `main()` function
- **Solution**: Created `patches/fomm/fomm_wrapper.py` that provides the interface
- **Status**: ✅ Fixed

### 2. MODEL_ROOT Path Mismatch
- **Problem**: Pipeline imports set MODEL_ROOT before notebook sets environment variables
- **Solution**: Updated `demo.ipynb` to set env vars BEFORE importing pipeline
- **Status**: ✅ Fixed

---

## Setup Instructions for Your Environment

Since you have models at `/mnt/c/blog/avatar-renderer-mcp/`, follow these steps:

### Step 1: Navigate to Your Repository

```bash
cd /mnt/c/blog/avatar-renderer-mcp/
```

### Step 2: Pull Latest Fixes

```bash
git pull origin claude/analyze-project-setup-xSSJs
```

### Step 3: Install Python Dependencies

```bash
# Option A: Using pip (recommended)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python safetensors scipy librosa tqdm
pip install yacs pyyaml imageio scikit-image ffmpeg-python

# Option B: Using uv (if available)
uv pip install torch torchvision opencv-python safetensors scipy librosa tqdm yacs pyyaml imageio scikit-image ffmpeg-python
```

**Important**: Use numpy 1.x (not 2.x) for compatibility with SadTalker:
```bash
pip install "numpy<2.0" --force-reinstall
```

### Step 4: Install System Dependencies

```bash
# Ubuntu/Debian/WSL
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

### Step 5: Verify Setup

```bash
# Set environment variables
export MODEL_ROOT="/mnt/c/blog/avatar-renderer-mcp/models"
export EXT_DEPS_DIR="/mnt/c/blog/avatar-renderer-mcp/external_deps"

# Run verification
python scripts/verify_setup.py
```

You should see:
```
✅ High-quality mode: READY
✅ Real-time mode: READY
```

### Step 6: Test the Pipeline

```bash
# Simple test (real-time mode)
python scripts/test_pipeline.py
```

Or use the Jupyter notebook:
```bash
# IMPORTANT: Restart kernel before running!
jupyter notebook demo.ipynb
```

**Critical**: When running the notebook:
1. **Restart the kernel** (Kernel → Restart Kernel)
2. Run cells **in order** from the top
3. The `imports` cell now sets MODEL_ROOT **before** importing pipeline

---

## Troubleshooting

### Issue: "FOMM checkpoint missing"

**Cause**: MODEL_ROOT environment variable not set before pipeline import

**Solution**:
```bash
# In your shell:
export MODEL_ROOT="/mnt/c/blog/avatar-renderer-mcp/models"
export EXT_DEPS_DIR="/mnt/c/blog/avatar-renderer-mcp/external_deps"

# Or in Python (BEFORE importing pipeline):
import os
os.environ["MODEL_ROOT"] = "/mnt/c/blog/avatar-renderer-mcp/models"
os.environ["EXT_DEPS_DIR"] = "/mnt/c/blog/avatar-renderer-mcp/external_deps"

# THEN import:
from app.pipeline import render_pipeline
```

### Issue: "No module named 'cv2'"

**Solution**:
```bash
pip install opencv-python-headless
```

### Issue: "No module named 'safetensors'"

**Solution**:
```bash
pip install safetensors
```

### Issue: "numpy.VisibleDeprecationWarning not found"

**Cause**: Numpy 2.x is installed, but SadTalker requires Numpy 1.x

**Solution**:
```bash
pip install "numpy<2.0" --force-reinstall
```

### Issue: "FOMM module loaded but 'main' function not found"

**Cause**: FOMM wrapper not installed

**Solution**:
```bash
make install-git-deps
```

This will copy `patches/fomm/fomm_wrapper.py` to `external_deps/first-order-model/`.

---

## Quick Test Script

Save this as `quick_test.py`:

```python
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# MUST set BEFORE any imports
repo_root = Path("/mnt/c/blog/avatar-renderer-mcp")
os.environ["MODEL_ROOT"] = str(repo_root / "models")
os.environ["EXT_DEPS_DIR"] = str(repo_root / "external_deps")
sys.path.insert(0, str(repo_root))

# Now import
from app.pipeline import render_pipeline

# Test
result = render_pipeline(
    face_image=str(repo_root / "tests/assets/alice.png"),
    audio=str(repo_root / "tests/assets/hello.wav"),
    out_path=str(repo_root / "test_output/hello.mp4"),
    quality_mode="real_time",  # or "high_quality" if you have GPU
)

print(f"✅ Success! Video: {result}")
```

Run it:
```bash
python quick_test.py
```

---

## Using Jupyter Notebook

The fixed `demo.ipynb` now sets environment variables in the **imports cell** (before importing pipeline).

**Workflow**:

1. Open Jupyter:
   ```bash
   cd /mnt/c/blog/avatar-renderer-mcp
   jupyter notebook demo.ipynb
   ```

2. **Restart kernel** (Kernel → Restart Kernel)

3. Run cells in order from the top

4. The first cell after markdown should set:
   ```python
   os.environ["MODEL_ROOT"] = str((Path.cwd() / "models").resolve())
   os.environ["EXT_DEPS_DIR"] = str((Path.cwd() / "external_deps").resolve())
   ```

5. Then import pipeline:
   ```python
   from app.pipeline import render_pipeline
   ```

---

## Files Modified by Our Fixes

```
✓ app/pipeline.py               - Updated run_fomm() to use wrapper
✓ Makefile                      - Added wrapper installation step
✓ patches/fomm/fomm_wrapper.py  - New wrapper with main() function
✓ patches/README.md             - Documentation for patches
✓ demo.ipynb                    - Fixed import order
✓ docs/TROUBLESHOOTING.md       - Comprehensive troubleshooting guide
✓ scripts/verify_setup.py       - Setup verification script
✓ scripts/test_pipeline.py      - Simple test script
```

---

## Summary

The core issues are **fixed**. The remaining setup is standard Python dependency installation.

**What we fixed**:
- ✅ FOMM wrapper (provides missing `main()` function)
- ✅ MODEL_ROOT path mismatch (env vars set before import)

**What you need to do**:
- Install Python dependencies (numpy<2.0, opencv, safetensors, etc.)
- Install system ffmpeg
- Restart Jupyter kernel before running notebook
- Set environment variables in your shell or at the top of scripts

After these steps, your notebook will work correctly!
