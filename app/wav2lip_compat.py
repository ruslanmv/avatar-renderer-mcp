"""Compatibility helpers for vendored Wav2Lip code."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("avatar-renderer.wav2lip_compat")

_LEGACY_MEL_CALL = "librosa.filters.mel(hp.sample_rate, hp.n_fft,"
_MODERN_MEL_CALL = "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,"


def patch_wav2lip_audio_for_librosa(wav2lip_dir: Path) -> bool:
    """Patch Wav2Lip's legacy librosa mel call for librosa >= 0.10.

    Wav2Lip's upstream ``audio.py`` calls ``librosa.filters.mel`` with the
    sample rate and FFT size as positional arguments. Modern librosa made those
    parameters keyword-only, so subprocess Wav2Lip fails with:
    ``TypeError: mel() takes 0 positional arguments...``.

    Returns ``True`` when a file was changed and ``False`` when the file was
    already compatible, missing, or could not be patched.
    """
    audio_py = wav2lip_dir / "audio.py"
    try:
        text = audio_py.read_text()
    except OSError as exc:
        log.warning("Could not read Wav2Lip audio.py for librosa compatibility patch: %s", exc)
        return False

    if _LEGACY_MEL_CALL not in text:
        return False

    try:
        audio_py.write_text(text.replace(_LEGACY_MEL_CALL, _MODERN_MEL_CALL))
    except OSError as exc:
        log.warning("Could not patch Wav2Lip audio.py for modern librosa: %s", exc)
        return False

    log.info("Patched Wav2Lip audio.py for modern librosa compatibility.")
    return True
