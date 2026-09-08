"""Shared helper for pyannote/segmentation-3.0's powerset output, used by both
PyannoteVAD (vad.py) and PyannoteOSD (osd.py) since they wrap the same model.

Confirmed against the real model AND real conversational audio (2026-08 session,
pyannote-audio 4.0.7, one real recording from sarvamai/indic-diarbench): VAD found 20
speech segments (vs. 18 ground-truth speaker-turn segments -- close and expected to
differ since VAD detects speech activity, not speaker turns) covering 59.3s of a 75.3s
recording; OSD found ~7.5% of frames with overlap probability > 0.5, a plausible overlap
ratio for real conversational speech. Details below were confirmed via direct model
inspection at the same time:
  - `model(waveform)` where waveform is a (1, num_samples) float tensor -- NOT
    `model({"waveform": ..., "sample_rate": ...})`, which raises
    `AttributeError: 'dict' object has no attribute 'dim'` inside SincNet.
  - Output is raw LOGITS of shape (1, n_frames, n_powerset_classes) -- needs softmax.
  - segmentation-3.0's powerset has 3 base speaker slots, max 2 simultaneously active,
    giving 7 classes: [none, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3]. The ORIGINAL
    implementation here hardcoded "any class index >= 2 means overlap", which is wrong
    (indices 1,2,3 are each a single active speaker, not overlap) -- this module instead
    reads pyannote's own `Powerset.mapping` to find which class indices correspond to
    >=2 simultaneously active speakers, so it stays correct even if a different
    segmentation checkpoint has a different number of base classes / max_classes.
"""

from __future__ import annotations

import numpy as np


def powerset_frame_probs(
    model, audio: np.ndarray, sample_rate: int, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame_times, frame_probs) where frame_probs has shape (n_frames, n_powerset_classes)."""
    import torch

    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(waveform)
    probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

    n_frames = probs.shape[0]
    audio_duration = len(audio) / sample_rate
    frame_duration = audio_duration / n_frames if n_frames > 0 else 0.0
    times = np.arange(n_frames) * frame_duration
    return times, probs


def active_speaker_counts_per_class(model) -> np.ndarray:
    """Returns an array of length n_powerset_classes, where entry i = how many base
    speakers are simultaneously active in powerset class i (0 = silence, 1 = single
    speaker, >=2 = overlap)."""
    from pyannote.audio.utils.powerset import Powerset

    specs = model.specifications
    powerset = Powerset(len(specs.classes), specs.powerset_max_classes)
    return powerset.mapping.sum(dim=-1).cpu().numpy()
