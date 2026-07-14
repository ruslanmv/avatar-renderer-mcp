#!/usr/bin/env python3
"""Generate narration audio for the Vercel homepage demo video."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import edge_tts


async def _synthesize(text: str, voice: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate web demo narration with edge-tts")
    parser.add_argument("--text", required=True, help="Narration text to synthesize")
    parser.add_argument("--voice", default="en-US-AriaNeural", help="edge-tts voice name")
    parser.add_argument("--out", type=Path, required=True, help="Output audio path")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    asyncio.run(_synthesize(args.text, args.voice, args.out))
    print(f"Generated narration audio: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
