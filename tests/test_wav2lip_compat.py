from app.wav2lip_compat import patch_wav2lip_audio_for_librosa


def test_patch_wav2lip_audio_for_modern_librosa(tmp_path):
    audio_py = tmp_path / "audio.py"
    audio_py.write_text(
        "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,\n"
        "                           fmin=hp.fmin, fmax=hp.fmax)\n"
    )

    assert patch_wav2lip_audio_for_librosa(tmp_path) is True
    patched = audio_py.read_text()
    assert "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft," in patched
    assert "librosa.filters.mel(hp.sample_rate, hp.n_fft," not in patched


def test_patch_wav2lip_audio_is_idempotent(tmp_path):
    audio_py = tmp_path / "audio.py"
    audio_py.write_text(
        "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels)\n"
    )

    assert patch_wav2lip_audio_for_librosa(tmp_path) is False
    assert "sr=hp.sample_rate" in audio_py.read_text()
