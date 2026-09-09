"""Synthetic overlap conversation generator.

Used as a stand-in for Indic DiarBench while the real dataset is being sourced
(see data/prep/manifest.py for the real-dataset loader). Mixes per-speaker tone
bursts (each speaker gets a distinct fundamental frequency, so speaker embedders
that look at spectral content -- even the dependency-free MFCCStatsEmbedder --
can tell speakers apart) into one waveform with known ground-truth segments, so
the full pipeline (VAD/OSD/routing/separation/attribution/ASR/fusion/eval) can
be exercised end-to-end without needing real speech audio or network access.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from schemas.types import (
    AcousticCondition,
    ManifestEntry,
    SpeakerAttributedTranscript,
    SpeechSegment,
    Utterance,
    WordToken,
)

SAMPLE_RATE = 16000
_SPEAKER_FREQS = [220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 880.0, 990.0, 1100.0]


@dataclass
class SyntheticConversation:
    audio: np.ndarray
    sample_rate: int
    segments: list[SpeechSegment]  # ground truth, may overlap in time
    # per-speaker isolated tracks, same length as `audio`, summing to it. These are the
    # ground-truth stems SI-SDR needs (eval/separation_quality.py); the real dataset has no
    # equivalent, so synthetic runs are the only place separation quality is directly scorable.
    source_tracks: dict[str, np.ndarray] = field(default_factory=dict)


def _tone_burst(freq: float, duration: float, sample_rate: int, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    # a few harmonics + light noise so spectral-band embeddings have nontrivial structure
    signal = (
        amplitude * np.sin(2 * np.pi * freq * t)
        + 0.15 * amplitude * np.sin(2 * np.pi * freq * 2 * t)
        + 0.05 * amplitude * np.random.default_rng(int(freq)).standard_normal(len(t))
    )
    # short fade in/out to avoid clicks that would confuse VAD boundaries
    fade = min(len(signal) // 10, int(0.01 * sample_rate)) or 1
    envelope = np.ones_like(signal)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (signal * envelope).astype(np.float32)


def generate_conversation(
    num_speakers: int = 2,
    num_turns: int = 8,
    overlap_ratio: float = 0.15,
    turn_duration_range: tuple[float, float] = (0.6, 1.8),
    gap_range: tuple[float, float] = (0.05, 0.4),
    seed: int = 0,
) -> SyntheticConversation:
    rng = random.Random(seed)
    speakers = [f"SPK{i:02d}" for i in range(num_speakers)]
    freqs = {spk: _SPEAKER_FREQS[i % len(_SPEAKER_FREQS)] for i, spk in enumerate(speakers)}

    segments: list[SpeechSegment] = []
    cursor = 0.0
    for turn_i in range(num_turns):
        speaker = speakers[turn_i % num_speakers]
        duration = rng.uniform(*turn_duration_range)

        if turn_i > 0 and rng.random() < overlap_ratio:
            # start this turn while the previous one is still active -> creates an overlap region
            overlap_amount = rng.uniform(0.15, min(0.6, duration))
            start = max(0.0, cursor - overlap_amount)
        else:
            start = cursor + rng.uniform(*gap_range)

        end = start + duration
        segments.append(SpeechSegment(start=start, end=end, speaker=speaker))
        cursor = max(cursor, end)

    total_duration = cursor + 0.3
    n_samples = int(total_duration * SAMPLE_RATE)
    # build each speaker's isolated track first, then sum -- the mixture is by construction
    # the sum of the stems, which is exactly the assumption SI-SDR scoring makes
    source_tracks = {spk: np.zeros(n_samples, dtype=np.float32) for spk in speakers}
    for seg in segments:
        tone = _tone_burst(freqs[seg.speaker], seg.duration, SAMPLE_RATE)
        start_idx = int(seg.start * SAMPLE_RATE)
        end_idx = min(n_samples, start_idx + len(tone))
        source_tracks[seg.speaker][start_idx:end_idx] += tone[: end_idx - start_idx]

    audio = np.clip(sum(source_tracks.values()), -1.0, 1.0).astype(np.float32)
    return SyntheticConversation(
        audio=audio, sample_rate=SAMPLE_RATE, segments=segments, source_tracks=source_tracks
    )


def synthetic_manifest_entry(
    recording_id: str,
    conversation: SyntheticConversation,
    language: str = "synthetic",
    condition: AcousticCondition = AcousticCondition.NEAR_FIELD,
) -> ManifestEntry:
    duration = len(conversation.audio) / conversation.sample_rate

    # placeholder lexical content (real text isn't derivable from synthetic tones) -- one
    # word per ground-truth segment, purely so WDER/cpWER/WER exercise their full code
    # path in the smoke test rather than short-circuiting on an empty reference.
    utterances = [
        Utterance(
            speaker=seg.speaker,
            start=seg.start,
            end=seg.end,
            words=[WordToken(text=f"seg{i}", start=seg.start, end=seg.end, speaker=seg.speaker)],
        )
        for i, seg in enumerate(conversation.segments)
    ]
    reference_transcript = SpeakerAttributedTranscript(recording_id=recording_id, utterances=utterances)

    return ManifestEntry(
        recording_id=recording_id,
        audio_path="",  # in-memory only; see scripts/run_variant.py --synthetic for direct use
        language=language,
        condition=condition,
        duration=duration,
        reference_segments=conversation.segments,
        reference_transcript=reference_transcript,
        source_tracks=dict(conversation.source_tracks),
        sample_rate=conversation.sample_rate,
    )
