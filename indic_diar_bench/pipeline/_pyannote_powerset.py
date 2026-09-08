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
  - `model.specifications.duration` is 10.0 seconds -- segmentation-3.0 is trained on
    10s chunks. A prior version of this module ran the ENTIRE recording through the
    model in a single forward pass regardless of length; on real multi-minute
    recordings that puts the model 5-10x outside the input length it was trained on,
    which lines up with an observed missed-speech rate (38.5% of reference speech time
    on a 26-recording real-dataset run, dominating the DER breakdown) far above what a
    correctly-windowed segmentation model should produce. `powerset_frame_probs` below
    now chunks audio into `duration`-second windows with 50% overlap (matching
    `model.specifications.duration`, the same convention pyannote's own `Inference`
    class uses) and averages overlapping frame predictions, instead of one giant
    out-of-distribution forward pass.
"""

from __future__ import annotations

import numpy as np


def _forward_chunk(model, chunk: np.ndarray, device: str) -> np.ndarray:
    """Runs one <= `model.specifications.duration`-second chunk through the model.
    Returns frame_probs of shape (n_frames, n_powerset_classes)."""
    import torch

    waveform = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(waveform)
    return logits.softmax(dim=-1).squeeze(0).cpu().numpy()


def powerset_frame_probs(
    model, audio: np.ndarray, sample_rate: int, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame_times, frame_probs) where frame_probs has shape (n_frames, n_powerset_classes).

    Chunks audio into `model.specifications.duration`-second windows (the model's trained
    chunk length) with 50% overlap when the recording is longer than one chunk, and
    averages predictions in overlapping regions -- the same sliding-window convention as
    pyannote's own `Inference` class. Recordings shorter than one chunk take the fast path
    (a single forward pass, unchanged from before) since there's nothing to window."""
    audio = np.asarray(audio, dtype=np.float32)
    audio_duration = len(audio) / sample_rate
    chunk_duration = float(getattr(model.specifications, "duration", 10.0)) or 10.0

    if audio_duration <= chunk_duration or len(audio) == 0:
        probs = _forward_chunk(model, audio, device)
        n_frames = probs.shape[0]
        frame_duration = audio_duration / n_frames if n_frames > 0 else 0.0
        times = np.arange(n_frames) * frame_duration
        return times, probs

    # Frame resolution (frames per second) is fixed by the model's conv-stride, so a
    # reference chunk of exactly `chunk_duration` seconds tells us frame_duration and the
    # per-chunk frame count for every window (all chunks below are padded/clamped to this
    # same length so their frame counts line up exactly).
    chunk_samples = int(round(chunk_duration * sample_rate))
    ref_probs = _forward_chunk(model, audio[:chunk_samples], device)
    frames_per_chunk, n_classes = ref_probs.shape
    frame_duration = chunk_duration / frames_per_chunk

    step_duration = chunk_duration / 2
    step_samples = int(round(step_duration * sample_rate))

    n_global_frames = int(np.ceil(audio_duration / frame_duration))
    sum_probs = np.zeros((n_global_frames, n_classes), dtype=np.float64)
    counts = np.zeros(n_global_frames, dtype=np.float64)

    starts = list(range(0, len(audio) - chunk_samples, step_samples))
    last_start = len(audio) - chunk_samples
    if not starts or starts[-1] != last_start:
        starts.append(max(0, last_start))

    for start_sample in starts:
        chunk_start_time = start_sample / sample_rate
        chunk = audio[start_sample:start_sample + chunk_samples]
        if start_sample == 0:
            chunk_probs = ref_probs  # reuse -- identical input, avoid a redundant forward pass
        else:
            chunk_probs = _forward_chunk(model, chunk, device)

        base_frame = int(round(chunk_start_time / frame_duration))
        for local_i in range(chunk_probs.shape[0]):
            g = base_frame + local_i
            if 0 <= g < n_global_frames:
                sum_probs[g] += chunk_probs[local_i]
                counts[g] += 1.0

    counts = np.maximum(counts, 1.0)
    probs = sum_probs / counts[:, None]
    times = np.arange(n_global_frames) * frame_duration
    return times, probs


def active_speaker_counts_per_class(model) -> np.ndarray:
    """Returns an array of length n_powerset_classes, where entry i = how many base
    speakers are simultaneously active in powerset class i (0 = silence, 1 = single
    speaker, >=2 = overlap)."""
    from pyannote.audio.utils.powerset import Powerset

    specs = model.specifications
    powerset = Powerset(len(specs.classes), specs.powerset_max_classes)
    return powerset.mapping.sum(dim=-1).cpu().numpy()
