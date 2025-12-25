#!/usr/bin/env python3
"""
demo.py - Avatar Renderer MCP Demo (Production-Ready)

Generates a talking avatar video from a static portrait image + audio using app.pipeline.render_pipeline.

Examples:
  python demo.py
  python demo.py --image tests/assets/alice.png --audio tests/assets/hello.wav --out demo.mp4 --quality auto
  python demo.py --quality real_time
  python demo.py --quality high_quality

Notes:
  - Download models:        make download-models
  - Install external deps:  make install-git-deps
  - Ensure ffmpeg exists:   ffmpeg -version
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def print_banner() -> None:
    print("=" * 60)
    print("🎬 Avatar Renderer MCP - Demo")
    print("=" * 60)
    print()


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def check_file(label: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{label} is a directory, expected a file: {path}")


def print_asset(label: str, path: Path) -> None:
    size = human_mb(path.stat().st_size)
    print(f"  ✅ {label}: {path} ({size})")


def generate_video(image_path: Path, audio_path: Path, output_path: Path, quality_mode: str) -> Path:
    print("🎬 Generating avatar video...")
    print(f"  Image:  {image_path}")
    print(f"  Audio:  {audio_path}")
    print(f"  Output: {output_path}")
    print(f"  Quality: {quality_mode}")
    print()
    print("⏳ Processing...\n")

    start_time = time.time()

    from app.pipeline import render_pipeline  # noqa: WPS433

    result = render_pipeline(
        face_image=str(image_path.resolve()),
        audio=str(audio_path.resolve()),
        out_path=str(output_path.resolve()),
        quality_mode=quality_mode,
    )

    elapsed = time.time() - start_time
    result_path = Path(result)

    print("✅ Success!")
    print(f"  ⏱️  Time: {elapsed:.1f}s")
    print(f"  📁 Output: {result_path}")

    if result_path.exists():
        print(f"  📊 Size: {human_mb(result_path.stat().st_size)}")

    return result_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Avatar Renderer MCP demo")
    p.add_argument("--image", type=Path, default=Path("tests/assets/alice.png"),
                   help="Path to portrait image (png/jpg)")
    p.add_argument("--audio", type=Path, default=Path("tests/assets/hello.wav"),
                   help="Path to audio file (wav recommended)")
    p.add_argument("--out", type=Path, default=Path("demo.mp4"),
                   help="Output mp4 path")
    p.add_argument("--quality", default="auto", choices=["auto", "real_time", "high_quality"],
                   help="Quality mode")
    return p


def main() -> int:
    print_banner()
    args = build_argparser().parse_args()

    model_root = os.environ.get("MODEL_ROOT", "(auto)")
    ext_deps = os.environ.get("EXT_DEPS_DIR", "external_deps")
    print(f"🔧 MODEL_ROOT: {model_root}")
    print(f"🔧 EXT_DEPS_DIR: {ext_deps}\n")

    print("📁 Checking assets...")
    try:
        check_file("Image", args.image)
        check_file("Audio", args.audio)
        print_asset("Image", args.image)
        print_asset("Audio", args.audio)
    except Exception as e:
        print(f"  ❌ {e}")
        print("\nFix:")
        print("  - Ensure test assets exist or pass --image/--audio with valid paths")
        return 2

    print()

    try:
        result = generate_video(args.image, args.audio, args.out, args.quality)
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Demo Failed")
        print("=" * 60)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()

        print("\nTroubleshooting:")
        print("  1. Download models:")
        print("     make download-models")
        print("  2. Install external dependencies:")
        print("     make install-git-deps")
        print("  3. Verify ffmpeg is installed and on PATH:")
        print("     ffmpeg -version")
        print("  4. Run pipeline CLI smoke test:")
        print("     python -m app.pipeline --help")
        return 1

    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)
    print(f"\nYour avatar video is ready: {result.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
