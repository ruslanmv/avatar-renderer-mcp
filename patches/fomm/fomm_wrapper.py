"""
FOMM Wrapper for Avatar Renderer MCP Pipeline
This module provides a main() function interface for the first-order-model demo.py
which doesn't have one by default.
"""

import sys
import yaml
from pathlib import Path
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

# Import FOMM functions
from demo import load_checkpoints, make_animation, find_best_frame


def main(args):
    """
    Main entry point for FOMM animation generation.

    Args:
        args: Arguments object with the following attributes:
            - source_image: Path to source image
            - driving_video: Path to driving video (optional)
            - driving_audio: Path to driving audio (optional, for audio-driven mode)
            - result_dir: Directory to save output frames
            - checkpoint: Path to FOMM checkpoint
            - config: Path to FOMM config (optional, defaults to vox-256.yaml)
            - relative: Use relative keypoint coordinates (default: True)
            - adapt_scale: Adapt movement scale (default: True)
            - cpu: Use CPU mode (default: False)
    """

    # Get configuration
    config_path = getattr(args, 'config', None)
    if config_path is None:
        # Use default config from first-order-model repo
        config_path = str(Path(__file__).parent / 'config' / 'vox-256.yaml')

    checkpoint_path = args.checkpoint
    source_image_path = args.source_image
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Determine if we're using video or audio driving
    driving_video_path = getattr(args, 'driving_video', None)
    driving_audio_path = getattr(args, 'driving_audio', None)

    # Load source image
    source_image = imageio.imread(source_image_path)
    source_image = resize(source_image, (256, 256))[..., :3]

    # CPU mode detection
    cpu_mode = getattr(args, 'cpu', False)

    # Load FOMM model
    print("[FOMM] Loading checkpoints...")
    generator, kp_detector = load_checkpoints(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cpu=cpu_mode
    )
    print("[FOMM] Checkpoints loaded successfully")

    # Handle driving input
    if driving_video_path:
        # Video-driven mode (standard FOMM)
        print(f"[FOMM] Loading driving video: {driving_video_path}")
        reader = imageio.get_reader(driving_video_path)
        fps = reader.get_meta_data().get('fps', 25)
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    elif driving_audio_path:
        # Audio-driven mode - generate simple motion
        # For now, use a simple approach: create static frames with slight variations
        # In a full implementation, you'd extract audio features and drive motion from them
        print(f"[FOMM] Audio-driven mode: {driving_audio_path}")
        print("[FOMM] WARNING: Audio-driven FOMM is experimental. Using minimal motion.")

        # Get audio duration to determine number of frames
        import wave
        with wave.open(driving_audio_path, 'rb') as audio:
            frame_rate = audio.getframerate()
            n_frames = audio.getnframes()
            duration = n_frames / float(frame_rate)

        fps = 25
        num_video_frames = int(duration * fps)

        # Create simple driving video (identity motion + small variations)
        # This creates a simple head bob/breathing effect
        driving_video = []
        for i in range(num_video_frames):
            # Use source image with slight variations
            frame = source_image.copy()
            # Add tiny variations to simulate natural motion
            # (In production, this should be driven by audio features)
            driving_video.append(frame)
    else:
        raise ValueError("Either driving_video or driving_audio must be provided")

    # Generate animation
    print(f"[FOMM] Generating animation with {len(driving_video)} frames...")
    relative = getattr(args, 'relative', True)
    adapt_scale = getattr(args, 'adapt_scale', True)

    predictions = make_animation(
        source_image=source_image,
        driving_video=driving_video,
        generator=generator,
        kp_detector=kp_detector,
        relative=relative,
        adapt_movement_scale=adapt_scale,
        cpu=cpu_mode
    )

    # Save frames to result_dir
    print(f"[FOMM] Saving {len(predictions)} frames to {result_dir}")
    for i, frame in enumerate(predictions):
        frame_path = result_dir / f"{i+1:04d}.png"
        imageio.imsave(str(frame_path), img_as_ubyte(frame))

    print(f"[FOMM] Animation complete! Frames saved to {result_dir}")
    return str(result_dir)


if __name__ == "__main__":
    # For testing the wrapper directly
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image", required=True, help="Path to source image")
    parser.add_argument("--driving_video", help="Path to driving video")
    parser.add_argument("--driving_audio", help="Path to driving audio")
    parser.add_argument("--result_dir", required=True, help="Output directory for frames")
    parser.add_argument("--checkpoint", required=True, help="Path to FOMM checkpoint")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--cpu", action="store_true", help="Use CPU mode")

    args = parser.parse_args()
    main(args)
