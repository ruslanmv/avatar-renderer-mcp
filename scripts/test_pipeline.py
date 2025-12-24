#!/usr/bin/env python3
"""
Simple test to verify Avatar Renderer MCP pipeline works correctly.
This tests the real_time mode (CPU-compatible).
"""

import os
import sys
from pathlib import Path

# Set environment variables BEFORE importing pipeline
repo_root = Path(__file__).parent.parent
os.environ["MODEL_ROOT"] = str((repo_root / "models").resolve())
os.environ["EXT_DEPS_DIR"] = str((repo_root / "external_deps").resolve())

# Add repo root to Python path
sys.path.insert(0, str(repo_root))

print("=" * 60)
print("Avatar Renderer MCP - Pipeline Test")
print("=" * 60)
print(f"\nMODEL_ROOT   = {os.environ['MODEL_ROOT']}")
print(f"EXT_DEPS_DIR = {os.environ['EXT_DEPS_DIR']}")
print()

# Now import pipeline
from app.pipeline import render_pipeline
import torch

# Check assets
test_image = repo_root / "tests" / "assets" / "alice.png"
test_audio = repo_root / "tests" / "assets" / "hello.wav"
output_dir = repo_root / "test_output"
output_dir.mkdir(exist_ok=True)

output_video = output_dir / "test_hello.mp4"

print("📋 Test Configuration:")
print(f"  Input image:  {test_image}")
print(f"  Input audio:  {test_audio}")
print(f"  Output video: {output_video}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print()

# Verify assets exist
if not test_image.exists():
    print(f"❌ Test image not found: {test_image}")
    sys.exit(1)

if not test_audio.exists():
    print(f"❌ Test audio not found: {test_audio}")
    sys.exit(1)

# Run pipeline
print("🎬 Running pipeline in real_time mode...")
print("This may take 30-60 seconds...\n")

try:
    result = render_pipeline(
        face_image=str(test_image),
        audio=str(test_audio),
        out_path=str(output_video),
        quality_mode="real_time",  # Use real_time for CPU compatibility
    )

    print(f"\n✅ Success!")
    print(f"Video generated: {result}")
    print(f"File size: {Path(result).stat().st_size / (1024*1024):.2f} MB")
    print()
    print("=" * 60)
    print("✅ Pipeline test PASSED")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    print(f"\n❌ Pipeline test FAILED:")
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("=" * 60)
    print("❌ Pipeline test FAILED")
    print("=" * 60)
    sys.exit(1)
