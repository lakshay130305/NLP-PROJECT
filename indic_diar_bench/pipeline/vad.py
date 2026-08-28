"""Voice Activity Detection, section 4.2. Backed by pyannote's segmentation model,
which jointly estimates VAD and overlap in one forward pass (reused by osd.py too,
so the two modules share the underlying model when constructed via `load_shared_segmentation_model`)."""

from __future__ import annotations

import numpy as np

from schemas.types import SpeechSegment


class EnergyVAD:
    """Dependency-free fallback VAD: simple frame-energy thresholding with
    minimum-duration smoothing. Used for tests and as a last-resort backend
    when no pretrained model is available."""

    def __init__(self, frame_size: float = 0.02, energy_threshold: float = 1e-4, min_duration: float = 0.1):
        self.frame_size = frame_size
        self.energy_threshold = energy_threshold
        self.min_duration = min_duration

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        hop = max(1, int(self.frame_size * sample_rate))
        n_frames = len(audio) // hop
        active = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            frame = audio[i * hop:(i + 1) * hop]
            energy = float(np.mean(frame.astype(np.float64) ** 2))
            active[i] = energy > self.energy_threshold

        segments: list[SpeechSegment] = []
        start = None
        for i, is_active in enumerate(active):
            t = i * self.frame_size
            if is_active and start is None:
                start = t
            elif not is_active and start is not None:
                if t - start >= self.min_duration:
                    segments.append(SpeechSegment(start, t))
                start = None
        if start is not None:
            end = n_frames * self.frame_size
            if end - start >= self.min_duration:
                segments.append(SpeechSegment(start, end))
        return segments


class PyannoteVAD:
    """Wraps pyannote.audio's pretrained segmentation-3.0 model for VAD, via its
    powerset frame classifier (see pipeline/_pyannote_powerset.py for the confirmed
    real API -- this was fixed after testing against the live model; the model call
    convention and "which classes mean overlap" logic were both wrong in an earlier
    untested version of this wrapper).

    Requires a HuggingFace auth token accepted for the gated pyannote model
    (set HF_TOKEN env var or pass hf_token=) and network access to download weights
    on first use."""

    def __init__(
        self, model_name: str = "pyannote/segmentation-3.0", hf_token: str | None = None, speech_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.speech_threshold = speech_threshold
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from pyannote.audio import Model

            self._model = Model.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        return self._model

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        from pipeline._pyannote_powerset import (
            active_speaker_counts_per_class,
            powerset_frame_probs,
        )

        model = self._ensure_loaded()
        times, probs = powerset_frame_probs(model, audio, sample_rate)
        speaker_counts = active_speaker_counts_per_class(model)

        speech_prob = probs[:, speaker_counts >= 1].sum(axis=-1)
        speech = speech_prob > self.speech_threshold

        segments: list[SpeechSegment] = []
        start = None
        hop = times[1] - times[0] if len(times) > 1 else 0.0
        for i, is_active in enumerate(speech):
            t = times[i]
            if is_active and start is None:
                start = t
            elif not is_active and start is not None:
                segments.append(SpeechSegment(start, t))
                start = None
        if start is not None:
            segments.append(SpeechSegment(start, times[-1] + hop if len(times) else 0.0))
        return segments
