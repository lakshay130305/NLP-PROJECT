"""Multilingual ASR backend, section 4.5 / 9.1-9.2. Backed by faster-whisper (CTranslate2),
which is CPU-friendly and supports word-level timestamps out of the box.

Kept as ONE fixed backend across all system variants (section 4.5) so that DER/WDER/cpWER
differences between variants can be attributed to overlap handling, not ASR changes.
"""

from __future__ import annotations

import re

import numpy as np

from schemas.types import WordToken


class RegexEnergyASR:
    """Dependency-free fallback ASR for tests/offline mode: does not do real recognition.
    It segments audio by energy into pseudo-"words" and labels each with a placeholder
    token, purely so the timing/fusion/attribution plumbing can be exercised without a
    model download. Swap in `FasterWhisperASR` for real transcription quality."""

    def __init__(self, frame_size: float = 0.02, energy_threshold: float = 1e-4, min_word_gap: float = 0.15):
        self.frame_size = frame_size
        self.energy_threshold = energy_threshold
        self.min_word_gap = min_word_gap

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> list[WordToken]:
        hop = max(1, int(self.frame_size * sample_rate))
        n_frames = len(audio) // hop
        active = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            frame = audio[i * hop:(i + 1) * hop]
            active[i] = float(np.mean(frame.astype(np.float64) ** 2)) > self.energy_threshold

        words: list[WordToken] = []
        start = None
        last_end = -1e9
        idx = 0
        for i, is_active in enumerate(active):
            t = i * self.frame_size
            if is_active and start is None:
                if t - last_end < self.min_word_gap and words:
                    start = words[-1].start
                    words.pop()
                else:
                    start = t
            elif not is_active and start is not None:
                words.append(WordToken(text=f"<w{idx}>", start=start, end=t))
                idx += 1
                last_end = t
                start = None
        if start is not None:
            end = n_frames * self.frame_size
            words.append(WordToken(text=f"<w{idx}>", start=start, end=end))
        return words


class FasterWhisperASR:
    """Wraps faster-whisper (CTranslate2 Whisper) for CPU-friendly multilingual ASR with
    word-level timestamps. Requires network access to download weights on first use.

    Model size matters a lot on CPU: 'tiny'/'base' for fast iteration, 'small'/'medium'
    for quality once you have real compute. IndicWhisper-family checkpoints can be loaded
    the same way via a HuggingFace CTranslate2-converted repo id.
    """

    def __init__(self, model_size: str = "tiny", compute_type: str = "int8", device: str = "cpu"):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None = None) -> list[WordToken]:
        from pipeline.audio_utils import pad_to_min_length

        if sample_rate != 16000:
            audio = _resample_np(audio, sample_rate, 16000)
        # see pipeline/embeddings.py::ECAPAEmbedder.embed for the general reason this guard
        # exists; confirmed separately for faster-whisper's word-level alignment, which
        # raised IndexError ("boolean index did not match indexed array") on very short
        # segments -- its internal DTW alignment needs a minimum number of encoder frames.
        audio = pad_to_min_length(audio, 16000, min_seconds=0.5)

        model = self._ensure_loaded()
        segments, _info = model.transcribe(
            audio.astype(np.float32), language=language, word_timestamps=True
        )

        words: list[WordToken] = []
        for seg in segments:
            for w in (seg.words or []):
                text = re.sub(r"\s+", " ", w.word).strip()
                if not text:
                    continue
                words.append(WordToken(text=text, start=w.start, end=w.end, confidence=getattr(w, "probability", 1.0)))
        return words


def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()
