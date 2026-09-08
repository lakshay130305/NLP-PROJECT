"""Overlapped Speech Detection module, section 5.2.

p_t = P(O_t = 1 | x_t); O_t = 1 if p_t > tau, else 0 (section 5.2)
Includes boundary smoothing / minimum-duration constraints (section 5.2.3).
"""

from __future__ import annotations

import numpy as np

from schemas.types import OverlapRegion


class EnergyRatioOSD:
    """Dependency-free OSD heuristic for tests/offline mode: estimates overlap probability
    from local spectral flatness + energy variance as a proxy for "more than one voice present".
    This is NOT a substitute for a trained OSD model -- it exists so the routing/fusion logic
    can be exercised end-to-end without a model download. Swap in `PyannoteOSD` for real use."""

    def __init__(self, frame_hop: float = 0.02, frame_size: float = 0.05):
        self.frame_hop = frame_hop
        self.frame_size = frame_size

    def frame_overlap_probs(
        self, audio: np.ndarray, sample_rate: int, frame_hop: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        hop = int((frame_hop or self.frame_hop) * sample_rate)
        win = int(self.frame_size * sample_rate)
        n_frames = max(1, (len(audio) - win) // hop + 1)
        times = np.arange(n_frames) * (frame_hop or self.frame_hop)
        probs = np.zeros(n_frames, dtype=np.float64)

        for i in range(n_frames):
            frame = audio[i * hop: i * hop + win].astype(np.float64)
            if len(frame) == 0:
                continue
            spectrum = np.abs(np.fft.rfft(frame)) + 1e-8
            geo_mean = np.exp(np.mean(np.log(spectrum)))
            arith_mean = np.mean(spectrum)
            flatness = geo_mean / arith_mean  # higher flatness ~ more "noise-like" / mixed sources
            probs[i] = float(np.clip(flatness * 2.0, 0.0, 1.0))

        return times, probs


class PyannoteOSD:
    """Wraps pyannote.audio's segmentation-3.0 model, whose powerset output distinguishes
    0/1/2-active-speaker frame classes directly, giving p_t = P(>=2 active speakers).

    Fixed after testing against the live model (2026-08 session): the model call must be
    `model(waveform_tensor)`, not `model({"waveform": ..., "sample_rate": ...})` (the dict
    form raises AttributeError deep inside SincNet). Also, segmentation-3.0's powerset has
    7 classes = [none, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3] for 3 base speaker slots
    with max 2 simultaneously active -- an earlier untested version of this method summed
    `probs[..., 2:]`, which wrongly counts single-speaker classes 2 and 3 as "overlap".
    The correct overlap classes are wherever `Powerset.mapping` shows >=2 active speakers
    (indices 4,5,6 for this specific checkpoint) -- computed generically via
    pipeline/_pyannote_powerset.py so it stays correct for any base-class/max-class count.
    """

    def __init__(self, model_name: str = "pyannote/segmentation-3.0", hf_token: str | None = None, device: str = "cpu"):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from pyannote.audio import Model

            self._model = Model.from_pretrained(self.model_name, use_auth_token=self.hf_token).to(self.device)
        return self._model

    def frame_overlap_probs(
        self, audio: np.ndarray, sample_rate: int, frame_hop: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray]:
        from pipeline._pyannote_powerset import (
            active_speaker_counts_per_class,
            powerset_frame_probs,
        )

        model = self._ensure_loaded()
        times, probs = powerset_frame_probs(model, audio, sample_rate, device=self.device)
        speaker_counts = active_speaker_counts_per_class(model)

        overlap_prob = probs[:, speaker_counts >= 2].sum(axis=-1)
        return times, overlap_prob


def threshold_overlap(times: np.ndarray, probs: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """O_t = 1 if p_t > tau else 0 -- section 5.2."""
    return (probs > tau).astype(np.int8)


def smooth_overlap_decisions(
    times: np.ndarray, decisions: np.ndarray, min_duration: float = 0.1
) -> list[OverlapRegion]:
    """Merge frame-level binary decisions into regions, dropping any region shorter than
    `min_duration` to avoid fragmentation (section 5.2.3)."""
    if len(times) == 0:
        return []
    hop = times[1] - times[0] if len(times) > 1 else 0.02

    regions: list[OverlapRegion] = []
    start = None
    for i, d in enumerate(decisions):
        t = times[i]
        if d and start is None:
            start = t
        elif not d and start is not None:
            if t - start >= min_duration:
                regions.append(OverlapRegion(start, t))
            start = None
    if start is not None:
        end = times[-1] + hop
        if end - start >= min_duration:
            regions.append(OverlapRegion(start, end))
    return regions
