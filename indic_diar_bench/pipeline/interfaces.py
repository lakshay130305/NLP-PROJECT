"""Protocol definitions for every pipeline stage (section 4 baseline + section 5-9 proposed framework).

Each stage is defined as a `typing.Protocol` so that:
  - pretrained-model-backed implementations (pyannote/speechbrain/faster-whisper) and
  - lightweight dummy/deterministic implementations (for fast unit tests, no network/model download)
are interchangeable. `variants.py` wires concrete implementations together per system variant (B1..B5, Proposed).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from schemas.types import SpeechSegment, WordToken


class VADModel(Protocol):
    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        """Return speech-active intervals (no speaker labels yet)."""
        ...


class OSDModel(Protocol):
    def frame_overlap_probs(
        self, audio: np.ndarray, sample_rate: int, frame_hop: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (frame_times, p_t) -- section 5.2, p_t = P(O_t=1 | x_t)."""
        ...


class SpeakerEmbedder(Protocol):
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a fixed-dim embedding vector for one audio segment."""
        ...


class Clusterer(Protocol):
    def cluster(self, embeddings: np.ndarray) -> list[str]:
        """Return a speaker label per embedding row."""
        ...


class SeparationModel(Protocol):
    def separate(
        self, audio: np.ndarray, sample_rate: int, max_speakers: int = 2
    ) -> list[np.ndarray]:
        """Return one waveform per estimated speaker stream -- section 7.2/7.3."""
        ...


class ASRModel(Protocol):
    def transcribe(
        self, audio: np.ndarray, sample_rate: int, language: str | None = None
    ) -> list[WordToken]:
        """Return word-level tokens with local (0-based) timestamps relative to the input audio."""
        ...
