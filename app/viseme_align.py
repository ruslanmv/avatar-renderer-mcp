"""
viseme_align.py  – word/phoneme/viseme timing helper
----------------------------------------------------

The high‑level function `get_viseme_json(audio_wav, transcript)` returns a list
like:
    [
        {"phone": "DH", "start": 0.12, "end": 0.18, "viseme": "D"},
        {"phone": "AH", "start": 0.18, "end": 0.30, "viseme": "A"},
        ...
    ]

If Montreal Forced Aligner (MFA) or its acoustic model is unavailable, we fall
back to a uniform segmentation that simply divides the clip length by
character count – still acceptable for demo UX.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

import librosa

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

MFA_MODEL = "/models/mfa/english_acoustic_model.zip"      # e.g. montreal EN
MFA_DICT  = "/models/mfa/cmudict.dict"                    # CMU phones → IPA
VIS_MAP   = {
    # Simplistic CMU‑phone → viseme class, refine as needed
    "AA": "A", "AE": "A", "AH": "A", "AO": "O", "AW": "O",
    "AY": "A", "B": "BMP", "CH": "C", "D": "D", "DH": "D",
    "EH": "E", "ER": "E", "EY": "E", "F": "FV", "G": "G",
    "HH": "rest", "IH": "E", "IY": "E", "JH": "C", "K": "G",
    "L": "L", "M": "BMP", "N": "L", "NG": "L", "OW": "O",
    "OY": "O", "P": "BMP", "R": "rest", "S": "S", "SH": "S",
    "T": "D", "TH": "TH", "UH": "O", "UW": "O",
    "V": "FV", "W": "O", "Y": "E", "Z": "S", "ZH": "S",
}

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def get_viseme_json(audio_wav: str, transcript: str) -> List[Dict]:
    """Main entry called by pipeline.py before Wav2Lip/Diff2Lip."""

    if _mfa_available():
        return _align_with_mfa(audio_wav, transcript)
    else:
        print("[viseme_align] MFA unavailable – using naive timing.")
        return _uniform_visemes(audio_wav, transcript)


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _mfa_available() -> bool:
    """Return True if the `mfa` binary *and* models are present."""
    return shutil.which("mfa") and Path(MFA_MODEL).exists() and Path(MFA_DICT).exists()


def _align_with_mfa(audio_wav: str, transcript: str) -> List[Dict]:
    """
    Runs Montreal Forced Aligner in a temp dir then parses the TextGrid output.
    """
    import tgt  # TextGrid parser (pip install tgt)

    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        (work / "wav").mkdir()
        (work / "txt").mkdir()

        # Prepare files
        shutil.copy(audio_wav, work / "wav" / "in.wav")
        (work / "txt" / "in.txt").write_text(transcript.strip())

        # Run MFA (can take 1‑2 s on GPU node, ~10 s CPU)
        subprocess.check_call(
            [
                "mfa",
                "align",
                str(work / "wav"),
                MFA_DICT,
                MFA_MODEL,
                str(work / "out"),
                "--overwrite",
                "--clean",
                "--quiet",
            ]
        )

        tg_path = next((work / "out").glob("*.TextGrid"))
        tg = tgt.io.read_textgrid(tg_path)
        tier = tg.get_tier_by_name("phones")
        entries = []
        for interval in tier.intervals:
            phone = interval.text.strip().split("_")[0]  # strip stress
            vis = VIS_MAP.get(phone, "rest")
            entries.append(
                {"phone": phone, "start": interval.start_time,
                 "end": interval.end_time, "viseme": vis}
            )
        return entries


def _uniform_visemes(audio_wav: str, transcript: str) -> List[Dict]:
    """Fallback: split duration evenly over characters (very rough)."""
    duration = librosa.get_duration(filename=audio_wav)
    n = max(len(transcript), 1)
    seg = duration / n
    out = []
    t = 0.0
    for ch in transcript:
        phone = ch.upper()
        vis = VIS_MAP.get(phone, "rest")
        out.append({"phone": phone, "start": t, "end": t + seg, "viseme": vis})
        t += seg
    return out


# --------------------------------------------------------------------------- #
# CLI debug helper                                                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, pprint

    p = argparse.ArgumentParser("viseme_align debug")
    p.add_argument("--audio", required=True)
    p.add_argument("--text", required=True)
    args = p.parse_args()

    js = get_viseme_json(args.audio, args.text)
    pprint.pp(js)
    out_file = Path(args.audio).with_suffix(".visemes.json")
    out_file.write_text(json.dumps(js, indent=2))
    print("Wrote", out_file)
