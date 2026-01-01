#!/usr/bin/env python3
"""
tests/tts_test_cli.py

Production-compatible tester for your FastAPI Chatterbox TTS server.

Fixes the common client-side issue:
- If server streams raw PCM (Content-Type: audio/L16), this script wraps it into a valid WAV.
- Supports *very long inputs* (blog posts / pages) by:
  - reading from --text or --file
  - splitting into paragraphs (or optional max chars)
  - sending multiple requests and concatenating audio into ONE output WAV

Works with:
- /v1/audio/speech (stream=true/false)
- /health

Examples:
  python tests/tts_test_cli.py --text "Hello world"
  python tests/tts_test_cli.py --file blog.txt --voice female --lang en --stream
  python tests/tts_test_cli.py --file blog.txt --stream --out out.wav --max-chars 1800
"""

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import requests

DEFAULT_URL = "http://localhost:4123"
ENDPOINT_HEALTH = "/health"
ENDPOINT_TTS = "/v1/audio/speech"


def parse_args():
    p = argparse.ArgumentParser(description="Chatterbox TTS Server CLI Tester (PCM-safe)")

    p.add_argument("--url", default=DEFAULT_URL, help="Base URL, e.g. http://localhost:4123")
    p.add_argument("--out", default="output.wav", help="Output WAV path")

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--text", default=None, help="Text to synthesize")
    src.add_argument("--file", default=None, help="Text file to synthesize (blog posts/pages)")

    p.add_argument("--lang", default="en", help="Language code (en, es, fr, it, ...)")
    p.add_argument("--voice", default="female", choices=["female", "male", "neutral"])
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--cfg", type=float, default=0.4)
    p.add_argument("--exag", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--rep-pen", type=float, default=1.2)

    p.add_argument("--stream", action="store_true", help="Ask server to stream (recommended)")
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")

    # Long-text splitting
    p.add_argument("--split-paragraphs", action="store_true", default=True,
                   help="Split by blank lines/paragraphs (default: true)")
    p.add_argument("--no-split-paragraphs", action="store_false", dest="split_paragraphs",
                   help="Do not split by paragraphs")
    p.add_argument("--max-chars", type=int, default=0,
                   help="Optional hard max chars per request chunk (0=disabled). Useful for huge pages.")

    p.add_argument("--verbose", action="store_true", help="Print response headers + debug info")

    return p.parse_args()


def check_health(base_url: str, timeout: float) -> Tuple[bool, dict]:
    url = f"{base_url}{ENDPOINT_HEALTH}"
    print(f"🩺 Checking Server Health: {url} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            print(f"❌ HTTP {r.status_code}")
            return False, {}
        data = r.json()
        ready = bool(data.get("model_ready"))
        dev = data.get("device", "unknown")
        print(f"✅ ONLINE ({'Ready' if ready else 'Loading'} on {dev})")
        return True, data
    except Exception as e:
        print(f"❌ FAILED ({e})")
        return False, {}


def read_input_text(args) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    # fallback demo text
    return "Good morning. I’m your AI assistant. I am designed to provide clear, reliable support."


def split_text(text: str, split_paragraphs: bool, max_chars: int) -> List[str]:
    t = text.strip()
    if not t:
        return []

    chunks: List[str] = [t]

    if split_paragraphs:
        # split on blank lines into paragraphs
        parts = [p.strip() for p in t.split("\n\n") if p.strip()]
        if parts:
            chunks = parts

    if max_chars and max_chars > 0:
        # further split any chunk longer than max_chars
        out: List[str] = []
        for c in chunks:
            if len(c) <= max_chars:
                out.append(c)
            else:
                # naive split by sentences-ish; fall back to hard split
                start = 0
                while start < len(c):
                    end = min(len(c), start + max_chars)
                    # try to cut at a sentence boundary if possible
                    cut = c.rfind(".", start, end)
                    if cut == -1:
                        cut = c.rfind("!", start, end)
                    if cut == -1:
                        cut = c.rfind("?", start, end)
                    if cut != -1 and cut > start + 200:
                        end = cut + 1
                    out.append(c[start:end].strip())
                    start = end
        chunks = [x for x in out if x]
    return chunks


def build_payload(args, text: str) -> dict:
    return {
        "input": text,
        "language": args.lang,
        "voice": args.voice,
        "speed": args.speed,
        "temperature": args.temp,
        "cfg_weight": args.cfg,
        "exaggeration": args.exag,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.rep_pen,
        "stream": bool(args.stream),
        "chunk_by_sentences": True,  # let server help too
    }


def save_wav_from_pcm16le(out_path: str, pcm_bytes: bytes, sr: int, channels: int = 1):
    out_path = str(out_path)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)


def request_audio(args, base_url: str, text_chunk: str) -> Tuple[bytes, str, dict]:
    """
    Returns:
      (audio_bytes, content_type, headers)
    audio_bytes:
      - if audio/L16 -> raw PCM16LE bytes
      - if audio/wav -> WAV file bytes
    """
    url = f"{base_url}{ENDPOINT_TTS}"
    payload = build_payload(args, text_chunk)

    with requests.post(url, json=payload, stream=True, timeout=args.timeout) as r:
        if r.status_code != 200:
            # show server error JSON if possible
            try:
                err = r.json()
                raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(err, indent=2)}")
            except Exception:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

        ctype = r.headers.get("Content-Type", "")
        headers = dict(r.headers)

        data = bytearray()
        for b in r.iter_content(chunk_size=64 * 1024):
            if b:
                data.extend(b)

        return bytes(data), ctype, headers


def main():
    args = parse_args()

    ok, health = check_health(args.url, timeout=min(args.timeout, 5.0))
    if not ok:
        sys.exit(1)

    text = read_input_text(args)
    chunks = split_text(text, split_paragraphs=args.split_paragraphs, max_chars=args.max_chars)

    if not chunks:
        print("No text to synthesize.")
        sys.exit(1)

    print("\n📝 Request Configuration:")
    print(f"   Lang:  {args.lang} | Voice: {args.voice} | Speed: {args.speed}")
    print(f"   Mode:  {'STREAMING' if args.stream else 'BATCH'}")
    print(f"   Chunks: {len(chunks)}")
    print("-" * 60)

    out_wav = Path(args.out).resolve()
    tmp_pcm = bytearray()
    sr_fallback = int(health.get("device") and 24000)  # harmless default
    sr_final: Optional[int] = None

    start_all = time.time()

    for i, chunk_text in enumerate(chunks):
        print(f"➡️  [{i+1}/{len(chunks)}] Sending {len(chunk_text)} chars...", end=" ", flush=True)
        t0 = time.time()
        try:
            audio_bytes, ctype, hdrs = request_audio(args, args.url, chunk_text)
        except Exception as e:
            print("\n❌ Request failed:")
            print(e)
            sys.exit(1)

        dt = time.time() - t0
        print(f"✅ {len(audio_bytes)/1024:.1f} KB in {dt:.2f}s ({ctype})")

        if args.verbose:
            print("   Headers (subset):")
            for k in ["Content-Type", "X-Audio-Format", "X-Sample-Rate", "X-Channels", "X-Streaming"]:
                if k in hdrs:
                    print(f"     {k}: {hdrs[k]}")

        # If server returns raw PCM (audio/L16), collect and later wrap into one WAV.
        if "audio/L16" in ctype:
            sr = int(hdrs.get("X-Sample-Rate", "24000"))
            ch = int(hdrs.get("X-Channels", "1"))
            if sr_final is None:
                sr_final = sr
            elif sr_final != sr:
                print(f"⚠️  Sample-rate changed mid-run ({sr_final} -> {sr}). Using first value.")
            if ch != 1:
                print(f"⚠️  Channels={ch} not expected. This script assumes mono output.")
            tmp_pcm.extend(audio_bytes)

        # If server returns WAV, we can either:
        # - for single chunk: just save it
        # - for multiple chunks: extract PCM frames and concatenate
        else:
            # Parse WAV and append frames to tmp_pcm so final output is ONE WAV.
            import io
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

            if sw != 2:
                print(f"⚠️  WAV sampwidth={sw} not supported. Expected 16-bit.")
                sys.exit(1)
            if ch != 1:
                print(f"⚠️  WAV channels={ch} not supported. Expected mono.")
                sys.exit(1)

            if sr_final is None:
                sr_final = sr
            elif sr_final != sr:
                print(f"⚠️  Sample-rate changed mid-run ({sr_final} -> {sr}). Using first value.")

            tmp_pcm.extend(frames)

    sr_final = sr_final or sr_fallback

    # Write final WAV
    save_wav_from_pcm16le(str(out_wav), bytes(tmp_pcm), sr=sr_final, channels=1)

    total = time.time() - start_all
    print("-" * 60)
    print("✅ DONE")
    print(f"   Output: {out_wav}")
    print(f"   Duration: {total:.2f}s")
    print(f"   WAV SR: {sr_final} Hz | Bytes: {len(tmp_pcm)}")


if __name__ == "__main__":
    main()
