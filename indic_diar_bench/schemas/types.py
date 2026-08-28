"""Core data structures shared across the pipeline.

These are plain dataclasses (not pydantic) so they stay cheap to construct
in hot loops (per-frame OSD scoring, per-segment routing, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AcousticCondition(str, Enum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"
    IN_THE_WILD = "in_the_wild"


@dataclass(frozen=True)
class SpeechSegment:
    """A single speaker-homogeneous time interval, as produced by VAD/diarization."""

    start: float  # seconds
    end: float  # seconds
    speaker: str | None = None  # None until assigned by clustering/attribution

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: SpeechSegment) -> bool:
        return self.start < other.end and other.start < self.end

    def intersection(self, other: SpeechSegment) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class OverlapRegion:
    """A time interval where >=2 ground-truth or predicted speakers are simultaneously active."""

    start: float
    end: float
    speakers: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class WordToken:
    """A single recognized word with timing, speaker attribution, and confidence."""

    text: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float = 1.0


@dataclass
class Utterance:
    """One (speaker, time-range, recognized words) unit -- the atomic output of ASR + attribution."""

    speaker: str
    start: float
    end: float
    words: list[WordToken] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class SpeakerAttributedTranscript:
    """Y = {(s_i, t_i^s, t_i^e, w_i)} -- the final task output, per section 3.6 of the paper plan."""

    recording_id: str
    utterances: list[Utterance] = field(default_factory=list)

    def sorted_by_time(self) -> list[Utterance]:
        return sorted(self.utterances, key=lambda u: u.start)


@dataclass
class ManifestEntry:
    """One recording's worth of metadata pulled from the Indic DiarBench manifest / RTTM."""

    recording_id: str
    audio_path: str
    language: str
    condition: AcousticCondition
    duration: float
    reference_segments: list[SpeechSegment] = field(default_factory=list)
    reference_transcript: SpeakerAttributedTranscript | None = None
