"""Covers the soundfile-based audio decode path in data/prep/manifest.py, which exists
specifically to avoid `datasets`' default torchcodec decoder (that needs an FFmpeg install
not present on this machine -- see README.md "Environment notes")."""

import io

import numpy as np
import soundfile as sf

from data.prep.manifest import _decode_audio


def _make_wav_bytes(duration=0.5, sample_rate=16000, freq=440.0) -> bytes:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV")
    return buf.getvalue()


def test_decode_audio_from_bytes():
    wav_bytes = _make_wav_bytes(duration=1.0, sample_rate=16000)
    audio, sr = _decode_audio({"bytes": wav_bytes, "path": None})
    assert sr == 16000
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16000) <= 1  # allow off-by-one from encode/decode rounding


def test_decode_audio_collapses_stereo_to_mono():
    t = np.arange(8000) / 16000
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.2 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, 16000, format="WAV")

    audio, sr = _decode_audio({"bytes": buf.getvalue(), "path": None})
    assert audio.ndim == 1
    assert sr == 16000
