"""Multilingual ASR transcript fusion, section 9.3-9.6.

Combines per-branch (non-overlap / separated-overlap) word streams -- each still
carrying LOCAL timestamps relative to the audio chunk they came from -- into one
chronological, speaker-attributed transcript in the original recording timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.types import SpeakerAttributedTranscript, Utterance, WordToken


@dataclass
class BranchOutput:
    """One processed chunk's worth of words, still in local time, plus the info needed
    to restore it to the original recording timeline (section 9.3) and attribute speaker(s)."""

    chunk_start: float  # offset of this chunk within the original recording
    words: list[WordToken]  # timestamps are LOCAL to the chunk (0-based)
    speaker: str


def restore_timestamps(branch: BranchOutput) -> list[WordToken]:
    """section 9.3: map separated-stream / chunk-local timestamps back to the recording timeline."""
    return [
        WordToken(
            text=w.text,
            start=w.start + branch.chunk_start,
            end=w.end + branch.chunk_start,
            speaker=branch.speaker,
            confidence=w.confidence,
        )
        for w in branch.words
    ]


def resolve_duplicates(words: list[WordToken], boundary_tolerance: float = 0.05) -> list[WordToken]:
    """section 9.5: drop near-duplicate words introduced at segmentation/chunk boundaries --
    same speaker, overlapping/adjacent timing, identical (case-insensitive) text."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.start)
    kept: list[WordToken] = [ordered[0]]
    for w in ordered[1:]:
        prev = kept[-1]
        same_speaker = w.speaker == prev.speaker
        same_text = w.text.strip().lower() == prev.text.strip().lower()
        close_in_time = w.start <= prev.end + boundary_tolerance
        if same_speaker and same_text and close_in_time:
            continue  # duplicate fragment from an adjacent chunk boundary
        kept.append(w)
    return kept


def fuse_transcript(
    recording_id: str,
    branch_outputs: list[BranchOutput],
    boundary_tolerance: float = 0.05,
    turn_gap: float = 0.5,
) -> SpeakerAttributedTranscript:
    """section 9.6: produce the final chronological speaker-attributed transcript,
    Y = {(s_i, t_i^s, t_i^e, w_i)} per section 3.6 -- preserving simultaneous speech
    (multiple speakers' words at overlapping timestamps are kept, not merged into one)."""
    all_words: list[WordToken] = []
    for branch in branch_outputs:
        all_words.extend(restore_timestamps(branch))

    all_words = resolve_duplicates(all_words, boundary_tolerance)

    # group consecutive same-speaker words into utterances (small gaps within one speaker's turn)
    by_speaker: dict[str, list[WordToken]] = {}
    for w in all_words:
        by_speaker.setdefault(w.speaker or "UNK", []).append(w)

    utterances: list[Utterance] = []
    for speaker, words in by_speaker.items():
        words = sorted(words, key=lambda w: w.start)
        current: list[WordToken] = []
        for w in words:
            if current and w.start - current[-1].end > turn_gap:
                utterances.append(_make_utterance(speaker, current))
                current = []
            current.append(w)
        if current:
            utterances.append(_make_utterance(speaker, current))

    return SpeakerAttributedTranscript(recording_id=recording_id, utterances=utterances)


def _make_utterance(speaker: str, words: list[WordToken]) -> Utterance:
    return Utterance(speaker=speaker, start=words[0].start, end=words[-1].end, words=words)
