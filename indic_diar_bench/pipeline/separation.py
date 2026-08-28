"""Speech separation for overlapping regions, section 7. Backed by speechbrain's SepFormer."""

from __future__ import annotations

import numpy as np


class NullSeparator:
    """Dependency-free fallback for tests/offline mode: returns the same mixed audio twice
    (i.e. "separation" is a no-op). Lets the multi-speaker attribution / fusion logic be
    exercised end-to-end without a model download. Swap in `SepFormerSeparator` for real use."""

    def separate(self, audio: np.ndarray, sample_rate: int, max_speakers: int = 2) -> list[np.ndarray]:
        return [audio.copy() for _ in range(max_speakers)]


class SepFormerSeparator:
    """Wraps speechbrain's pretrained SepFormer 2-speaker separation model
    (sepformer-wsj02mix, 8kHz). Requires network access to download weights on first use."""

    def __init__(self, source: str = "speechbrain/sepformer-wsj02mix", savedir: str | None = None):
        self.source = source
        self.savedir = savedir or ".cache/speechbrain/sepformer"
        self.model_sample_rate = 8000
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from speechbrain.inference.separation import SepformerSeparation
            from speechbrain.utils.fetching import LocalStrategy

            # see pipeline/embeddings.py::ECAPAEmbedder for why COPY (not the SYMLINK
            # default) is needed on Windows without Developer Mode / admin privileges.
            self._model = SepformerSeparation.from_hparams(
                source=self.source, savedir=self.savedir, local_strategy=LocalStrategy.COPY
            )
        return self._model

    def separate(self, audio: np.ndarray, sample_rate: int, max_speakers: int = 2) -> list[np.ndarray]:
        import torch
        import torchaudio

        from pipeline.audio_utils import pad_to_min_length

        if max_speakers != 2:
            raise ValueError("SepFormerSeparator (wsj02mix) only supports 2-speaker separation; "
                              "see section 7.3 for the >2-speaker fallback strategy (iterative separation).")

        # see pipeline/embeddings.py::ECAPAEmbedder.embed for why this guard exists --
        # very short real VAD segments can crash convolutional feature extractors.
        audio = pad_to_min_length(audio, sample_rate, min_seconds=0.5)

        model = self._ensure_loaded()
        if sample_rate != self.model_sample_rate:
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.model_sample_rate)
        else:
            waveform = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.inference_mode():
            est_sources = model.separate_batch(waveform)  # (1, T, n_speakers)

        streams = []
        for i in range(est_sources.shape[-1]):
            stream = est_sources[0, :, i].cpu().numpy()
            if sample_rate != self.model_sample_rate:
                stream_t = torch.from_numpy(stream).float().unsqueeze(0)
                stream_t = torchaudio.functional.resample(stream_t, self.model_sample_rate, sample_rate)
                stream = stream_t.squeeze(0).numpy()
            streams.append(stream)
        return streams


def iterative_multi_speaker_separation(
    audio: np.ndarray, sample_rate: int, separator: SepFormerSeparator, max_streams: int = 3
) -> list[np.ndarray]:
    """Fallback for >2 simultaneous speakers (section 7.3): repeatedly 2-way-separate the
    residual with the highest estimated energy until `max_streams` streams are collected
    or the residual energy falls below a floor. This is a pragmatic approximation, not a
    principled multi-speaker separator -- flagged as a known limitation (section 22)."""
    streams: list[np.ndarray] = []
    residual = audio.copy()
    while len(streams) < max_streams - 1:
        s1, s2 = separator.separate(residual, sample_rate, max_speakers=2)
        streams.append(s1)
        residual = s2
        if float(np.mean(residual.astype(np.float64) ** 2)) < 1e-6:
            break
    streams.append(residual)
    return streams
