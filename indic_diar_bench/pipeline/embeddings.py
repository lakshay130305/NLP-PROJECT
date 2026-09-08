"""Speaker embedding extraction, section 4.3 / 8.1. Backed by speechbrain's ECAPA-TDNN."""

from __future__ import annotations

import numpy as np


class MFCCStatsEmbedder:
    """Dependency-free fallback embedder for tests/offline mode: mean+std of a simple
    log-mel-like spectral summary. Not speaker-discriminative in any real sense --
    swap in `ECAPAEmbedder` for real speaker verification quality."""

    def __init__(self, n_bands: int = 24, sample_rate_hint: int = 16000):
        self.n_bands = n_bands
        self.sample_rate_hint = sample_rate_hint

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = audio.astype(np.float64)
        if len(audio) == 0:
            return np.zeros(self.n_bands * 2, dtype=np.float32)
        spectrum = np.abs(np.fft.rfft(audio))
        bands = np.array_split(spectrum, self.n_bands)
        band_energy = np.array([np.log(np.mean(b) + 1e-8) for b in bands])
        # frame the signal to get a "std across time" component too
        hop = max(1, len(audio) // 20)
        frame_bands = []
        for i in range(0, len(audio) - hop, hop):
            frame_spec = np.abs(np.fft.rfft(audio[i:i + hop]))
            fb = np.array_split(frame_spec, self.n_bands)
            frame_bands.append([np.log(np.mean(b) + 1e-8) for b in fb])
        std = np.std(frame_bands, axis=0) if frame_bands else np.zeros(self.n_bands)
        return np.concatenate([band_energy, std]).astype(np.float32)


class ECAPAEmbedder:
    """Wraps speechbrain's pretrained ECAPA-TDNN speaker encoder (spkrec-ecapa-voxceleb).
    Requires network access to download weights on first use."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", savedir: str | None = None, device: str = "cpu"):
        self.source = source
        self.savedir = savedir or ".cache/speechbrain/ecapa"
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            # LocalStrategy.SYMLINK (speechbrain's default) needs Developer Mode / admin
            # privileges on Windows and fails with OSError: [WinError 1314] otherwise
            # (confirmed this session); COPY works everywhere at the cost of extra disk space.
            self._model = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                local_strategy=LocalStrategy.COPY,
                run_opts={"device": self.device},
            )
        return self._model

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        from pipeline.audio_utils import pad_to_min_length

        model = self._ensure_loaded()
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)
        audio = pad_to_min_length(audio, 16000, min_seconds=0.5)
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()
