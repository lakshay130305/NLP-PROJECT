"""Loader for the real Indic DiarBench dataset (sarvamai/indic-diarbench on HuggingFace).

Confirmed schema (22 languages, one HF config per language, each with a "test" split;
~1164 samples total, ~108h -- see ALL_LANGUAGES below for the exact config names):
  sample_id: str            recording_id: str  ("<language>_<condition>_<nnn>")
  language: str              audio: 16kHz mono WAV (decoded here via soundfile, not torchcodec)
  dataset_type: str          "Near field" | "Far field" | "In the wild"
  duration_seconds: float
  annotated_transcript: list[{speaker_id, transcript, start_time, end_time}]
  num_speakers: int          num_segments: int

Requires network access + the `datasets` library (already in requirements) to pull from
the HuggingFace Hub. Converts each HF row into our ManifestEntry/SpeechSegment/WordToken
schema so it's a drop-in match for the synthetic generator's output shape.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from schemas.types import (
    AcousticCondition,
    ManifestEntry,
    SpeakerAttributedTranscript,
    SpeechSegment,
    Utterance,
    WordToken,
)

_CONDITION_MAP = {
    "Near field": AcousticCondition.NEAR_FIELD,
    "Far field": AcousticCondition.FAR_FIELD,
    "In the wild": AcousticCondition.IN_THE_WILD,
}

HF_DATASET_ID = "sarvamai/indic-diarbench"

# The dataset is published as one HF "config" per language (confirmed against the live
# repo -- `load_dataset(HF_DATASET_ID)` with no config raises and lists these 22 names).
ALL_LANGUAGES = [
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada", "Kashmiri",
    "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", "Nepali", "Odia", "Punjabi",
    "Sanskrit", "Santali", "Sindhi", "Tamil", "Telugu", "Urdu",
]


def load_indic_diarbench(
    languages: list[str] | None = None,
    conditions: list[AcousticCondition] | None = None,
    limit: int | None = None,
    streaming: bool = True,
) -> Iterator[tuple[ManifestEntry, np.ndarray, int]]:
    """Yields (manifest_entry, audio_waveform, sample_rate) tuples.

    `streaming=True` avoids downloading the full ~108h corpus up front -- recommended for
    the "validate on a small subset first" workflow described in the project plan. Set
    `languages`/`conditions` to restrict to a slice, and `limit` to cap the TOTAL sample
    count across all requested languages combined.
    """
    from datasets import Audio, load_dataset

    target_languages = languages if languages else ALL_LANGUAGES

    count = 0
    for lang in target_languages:
        ds = load_dataset(HF_DATASET_ID, lang, split="test", streaming=streaming)
        # Force the raw-bytes audio path instead of `datasets`' default torchcodec-based
        # decoder: torchcodec needs a matching FFmpeg install (versions 4-7) that isn't
        # present here, so decode=True raises RuntimeError("Could not load libtorchcodec").
        # `soundfile` (already a dependency) decodes the raw WAV bytes directly below,
        # with no FFmpeg/torchcodec dependency at all.
        ds = ds.cast_column("audio", Audio(decode=False))
        for row in ds:
            condition = _CONDITION_MAP.get(row["dataset_type"])
            if condition is None:
                continue
            if conditions and condition not in conditions:
                continue

            entry, audio, sr = _row_to_entry(row, condition)
            yield entry, audio, sr

            count += 1
            if limit is not None and count >= limit:
                return


def _decode_audio(audio_field: dict) -> tuple[np.ndarray, int]:
    """Decode a `datasets.Audio(decode=False)` field ({'bytes': ..., 'path': ...}) via
    soundfile, sidestepping the torchcodec/FFmpeg dependency (see load_indic_diarbench)."""
    import io

    import soundfile as sf

    if audio_field.get("bytes") is not None:
        data, sample_rate = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    else:
        data, sample_rate = sf.read(audio_field["path"], dtype="float32")

    if data.ndim > 1:  # collapse to mono if the source has multiple channels
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def _row_to_entry(row: dict, condition: AcousticCondition):
    audio, sample_rate = _decode_audio(row["audio"])

    segments: list[SpeechSegment] = []
    words: list[WordToken] = []
    for seg in row["annotated_transcript"]:
        speaker = seg["speaker_id"]
        start, end = float(seg["start_time"]), float(seg["end_time"])
        segments.append(SpeechSegment(start=start, end=end, speaker=speaker))
        # transcript is a segment-level string; treat it as one "word" token spanning the
        # segment for WDER/cpWER purposes unless/until forced word-alignment is added
        text = seg.get("transcript", "").strip()
        if text:
            words.append(WordToken(text=text, start=start, end=end, speaker=speaker))

    reference_transcript = SpeakerAttributedTranscript(
        recording_id=row["recording_id"],
        utterances=[
            Utterance(speaker=w.speaker, start=w.start, end=w.end, words=[w]) for w in words
        ],
    )

    entry = ManifestEntry(
        recording_id=row["recording_id"],
        audio_path=f"hf://{HF_DATASET_ID}/{row['sample_id']}",
        language=row["language"],
        condition=condition,
        duration=float(row["duration_seconds"]),
        reference_segments=segments,
        reference_transcript=reference_transcript,
    )
    return entry, audio, sample_rate
