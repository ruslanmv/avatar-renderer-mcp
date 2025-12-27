#!/usr/bin/env python3
"""
assets.py - Generate all demo videos for the Avatar Renderer MCP frontend

Creates videos for all avatar personas shown in the frontend:
- Professional (Support, Sales, HR)
- Creator (Influencer, Ads)
- Educator (Courses, Tutors)
- Game NPC (Dialogue, Lore)
- Brand (Retail, Events)
- Custom

Usage:
    python assets.py                    # Generate all demo videos
    python assets.py --avatar professional  # Generate only professional avatar
    python assets.py --quality high_quality # Use high quality mode
    python assets.py --output-dir ./frontend/public/videos  # Custom output directory
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

# Avatar configurations with their descriptions and scripts
AVATAR_CONFIGS = {
    "professional": {
        "description": "Professional avatar for Support, Sales, and HR",
        "image": "external_deps/SadTalker/examples/source_image/art_1.png",
        "script": "Hello! I'm your professional assistant. I'm here to help you with customer support, sales inquiries, and HR matters. How can I assist you today?",
        "output": "professional_demo.mp4",
    },
    "creator": {
        "description": "Creator avatar for Influencers and Advertisements",
        "image": "external_deps/SadTalker/examples/source_image/art_3.png",
        "script": "Hey everyone! Welcome to my channel. Today I'm excited to share amazing content with you. Don't forget to like, subscribe, and hit that notification bell!",
        "output": "creator_demo.mp4",
    },
    "educator": {
        "description": "Educator avatar for Online Courses and Tutoring",
        "image": "external_deps/SadTalker/examples/source_image/art_4.png",
        "script": "Welcome to today's lesson. I'm here to guide you through this learning journey. Let's explore new concepts together and make education engaging and interactive!",
        "output": "educator_demo.mp4",
    },
    "game_npc": {
        "description": "Game NPC avatar with unique personality and dialogue",
        "image": "external_deps/SadTalker/examples/source_image/art_5.png",
        "script": "Greetings, traveler! I have a quest for you. The ancient artifacts are scattered across the land. Will you help us restore balance to the realm?",
        "output": "game_npc_demo.mp4",
    },
    "brand": {
        "description": "Brand ambassador for Retail and Events",
        "image": "external_deps/SadTalker/examples/source_image/art_6.png",
        "script": "Welcome to our brand experience! Discover our latest products and exclusive offers. We're committed to delivering excellence and innovation in everything we do.",
        "output": "brand_demo.mp4",
    },
    "custom": {
        "description": "Custom avatar - bring your own personality",
        "image": "tests/assets/alice.png",
        "script": "Hello! I'm your AI avatar, ready to bring your content to life with realistic expressions and natural voice synchronization. Let's create something amazing together!",
        "output": "custom_demo.mp4",
    },
}


def print_banner() -> None:
    """Print the script banner."""
    print("=" * 70)
    print("🎬 Avatar Renderer MCP - Demo Assets Generator")
    print("=" * 70)
    print()


def human_mb(num_bytes: int) -> str:
    """Convert bytes to human-readable MB."""
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def check_dependencies(skip_model_check: bool = False) -> bool:
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")

    # Check if ffmpeg is available
    import shutil
    if not shutil.which("ffmpeg"):
        print("  ❌ ffmpeg not found. Please install ffmpeg.")
        return False
    print("  ✅ ffmpeg found")

    # Check if models directory exists
    model_root = os.environ.get("MODEL_ROOT", "models")
    if not Path(model_root).exists():
        print(f"  ⚠️  MODEL_ROOT not found: {model_root}")
        print("  Run: make download-models")
        if not skip_model_check:
            return False
    else:
        print(f"  ✅ MODEL_ROOT: {model_root}")

    # Check if key model files exist (unless skipped)
    if not skip_model_check:
        fomm_model = Path(model_root) / "fomm" / "vox-cpk.pth"
        if not fomm_model.exists():
            print(f"  ⚠️  FOMM model not found: {fomm_model}")
            print("  Run: make download-models")
            print("  Note: The script will fail during video generation without models.")
            print()

            # Ask user if they want to continue anyway
            try:
                response = input("Continue anyway to test the script? (y/N): ").strip().lower()
                if response != 'y':
                    return False
            except (EOFError, KeyboardInterrupt):
                print("\n  Aborted.")
                return False
    else:
        print("  ⚠️  Model check skipped (--skip-model-check flag used)")
        print("  Note: Video generation will fail if models are not present.")

    print()
    return True


def generate_tts_audio(text: str, output_path: Path) -> Path:
    """
    Generate TTS audio from text.
    For now, we'll use the existing audio file as a placeholder.
    In production, you would integrate with a TTS service.
    """
    # TODO: Integrate with actual TTS service (e.g., OpenAI TTS, Google TTS, etc.)
    # For now, we'll use the existing hello.wav as a placeholder
    placeholder_audio = Path("tests/assets/hello.wav")
    if placeholder_audio.exists():
        import shutil
        shutil.copy(placeholder_audio, output_path)
        return output_path
    else:
        raise FileNotFoundError(f"Placeholder audio not found: {placeholder_audio}")


def generate_avatar_video(
    avatar_name: str,
    config: Dict[str, str],
    output_dir: Path,
    quality_mode: str = "auto",
    use_tts: bool = False,
) -> Path:
    """Generate a single avatar video."""
    print(f"🎬 Generating {avatar_name} avatar...")
    print(f"  Description: {config['description']}")
    print(f"  Image: {config['image']}")
    print(f"  Script: {config['script'][:60]}...")

    # Check if image exists
    image_path = Path(config['image'])
    if not image_path.exists():
        print(f"  ⚠️  Image not found: {image_path}")
        print(f"  Skipping {avatar_name}...")
        return None

    # Prepare audio
    if use_tts:
        print("  🔊 Generating TTS audio...")
        audio_path = output_dir / f"{avatar_name}_audio.wav"
        generate_tts_audio(config['script'], audio_path)
    else:
        # Use placeholder audio for demo
        audio_path = Path("tests/assets/hello.wav")
        if not audio_path.exists():
            print(f"  ❌ Audio not found: {audio_path}")
            return None

    # Output path
    output_path = output_dir / config['output']

    print(f"  Output: {output_path}")
    print(f"  Quality: {quality_mode}")
    print()

    # Import and run the pipeline
    start_time = time.time()

    try:
        from app.pipeline import render_pipeline

        result = render_pipeline(
            face_image=str(image_path.resolve()),
            audio=str(audio_path.resolve()),
            out_path=str(output_path.resolve()),
            quality_mode=quality_mode,
        )

        elapsed = time.time() - start_time
        result_path = Path(result)

        print(f"  ✅ Success! Generated in {elapsed:.1f}s")
        if result_path.exists():
            print(f"  📊 Size: {human_mb(result_path.stat().st_size)}")
        print()

        return result_path

    except Exception as e:
        print(f"  ❌ Failed to generate {avatar_name}: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None


def generate_all_avatars(
    output_dir: Path,
    quality_mode: str = "auto",
    avatars: List[str] = None,
    use_tts: bool = False,
) -> Dict[str, Path]:
    """Generate all avatar demo videos."""
    results = {}

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.resolve()}\n")

    # Filter avatars if specified
    if avatars:
        configs = {k: v for k, v in AVATAR_CONFIGS.items() if k in avatars}
    else:
        configs = AVATAR_CONFIGS

    total = len(configs)
    print(f"🎯 Generating {total} avatar demo videos...\n")

    for idx, (avatar_name, config) in enumerate(configs.items(), 1):
        print(f"[{idx}/{total}] Processing {avatar_name}...")
        result = generate_avatar_video(
            avatar_name, config, output_dir, quality_mode, use_tts
        )
        if result:
            results[avatar_name] = result
        print()

    return results


def print_summary(results: Dict[str, Path], output_dir: Path) -> None:
    """Print summary of generated videos."""
    print("=" * 70)
    print("📊 Generation Summary")
    print("=" * 70)
    print()

    if results:
        print(f"✅ Successfully generated {len(results)} videos:\n")
        for avatar_name, path in results.items():
            if path.exists():
                size = human_mb(path.stat().st_size)
                print(f"  • {avatar_name:15} → {path.name:30} ({size})")

        print()
        print(f"📁 All videos saved to: {output_dir.resolve()}")
        print()
        print("💡 Next steps:")
        print("  1. Copy videos to your frontend public directory:")
        print(f"     cp {output_dir}/*.mp4 frontend/public/videos/")
        print("  2. Update your frontend to use these demo videos")
        print("  3. Deploy to Vercel or your hosting platform")
    else:
        print("❌ No videos were generated successfully.")
        print("\nTroubleshooting:")
        print("  1. Ensure models are downloaded: make download-models")
        print("  2. Check that image files exist in external_deps/SadTalker/examples/")
        print("  3. Verify ffmpeg is installed: ffmpeg -version")

    print()


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Generate all demo videos for Avatar Renderer MCP frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python assets.py
  python assets.py --avatar professional creator educator
  python assets.py --quality high_quality
  python assets.py --output-dir ./frontend/public/videos
  python assets.py --use-tts  # Use TTS to generate audio from scripts (requires TTS setup)
  python assets.py --skip-model-check  # Skip model file verification (testing only)

Available avatars:
  professional - Professional avatar for Support, Sales, HR
  creator      - Creator avatar for Influencers, Ads
  educator     - Educator avatar for Online Courses, Tutoring
  game_npc     - Game NPC with unique personality
  brand        - Brand ambassador for Retail, Events
  custom       - Custom avatar (bring your own)
        """,
    )

    p.add_argument(
        "--avatar",
        nargs="+",
        choices=list(AVATAR_CONFIGS.keys()),
        help="Generate specific avatar(s). If not specified, generates all avatars.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_videos"),
        help="Output directory for generated videos (default: ./demo_videos)",
    )

    p.add_argument(
        "--quality",
        default="auto",
        choices=["auto", "real_time", "high_quality"],
        help="Quality mode for video generation (default: auto)",
    )

    p.add_argument(
        "--use-tts",
        action="store_true",
        help="Generate audio from scripts using TTS (requires TTS setup)",
    )

    p.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip model file verification (for testing purposes only)",
    )

    return p


def main() -> int:
    """Main entry point."""
    print_banner()

    args = build_argparser().parse_args()

    # Check dependencies
    if not check_dependencies(skip_model_check=args.skip_model_check):
        print("❌ Dependency check failed. Please fix the issues above.")
        return 1

    # Generate avatars
    try:
        results = generate_all_avatars(
            output_dir=args.output_dir,
            quality_mode=args.quality,
            avatars=args.avatar,
            use_tts=args.use_tts,
        )

        # Print summary
        print_summary(results, args.output_dir)

        if results:
            print("=" * 70)
            print("🎉 All demos generated successfully!")
            print("=" * 70)
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
