"""Small shared audio helpers for the pretrained backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000

# Extensions soundfile (libsndfile) reads natively and fast. Anything else falls back to
# PyAV, which bundles its own decoder libraries and handles mp3/m4a/aac/etc. without a
# separate system FFmpeg install (confirmed working this session -- see load_audio_file).
_SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 PCM at TARGET_SAMPLE_RATE (16kHz), regardless
    of its original format/sample rate/channel count. Tries soundfile first for common
    formats (fast, no re-decoding needed if already 16kHz mono WAV); falls back to PyAV
    for anything soundfile can't open (mp3, m4a, aac, ...)."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _SOUNDFILE_EXTENSIONS:
        try:
            return _load_with_soundfile(path)
        except Exception:  # noqa: BLE001, S110 -- deliberate format-detection fallback to PyAV
            pass

    return _load_with_pyav(path)


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        data = _resample(data, sample_rate, TARGET_SAMPLE_RATE)
    return data.astype(np.float32), TARGET_SAMPLE_RATE


def _load_with_pyav(path: Path) -> tuple[np.ndarray, int]:
    import av

    container = av.open(str(path))
    audio_streams = container.streams.audio
    if not audio_streams:
        raise ValueError(f"No audio stream found in {path}")

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio_streams[0]):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1))
    container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32), TARGET_SAMPLE_RATE
    return np.concatenate(chunks).astype(np.float32), TARGET_SAMPLE_RATE


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def pad_to_min_length(audio: np.ndarray, sample_rate: int, min_seconds: float = 0.5) -> np.ndarray:
    """Zero-pad very short segments up to `min_seconds`.

    Needed because real VAD output can contain very short segments (confirmed on real
    Indic DiarBench audio: a 34ms first segment) that crash convolutional feature
    extractors expecting a minimum number of time frames -- e.g. ECAPA-TDNN's SincNet
    front-end raised `RuntimeError: Padding size should be less than the corresponding
    input dimension` on a 2-frame mel-spectrogram. Zero-padding (rather than dropping the
    segment) keeps the segment's timing/speaker attribution meaningful even though the
    resulting embedding will be lower-confidence for such short segments.
    """
    min_samples = int(min_seconds * sample_rate)
    if len(audio) >= min_samples:
        return audio
    pad_width = min_samples - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant")
