from pipeline.fuse import (
    BranchOutput,
    fuse_transcript,
    resolve_duplicates,
    restore_timestamps,
)
from schemas.types import WordToken


def test_restore_timestamps_offsets_by_chunk_start():
    branch = BranchOutput(chunk_start=10.0, words=[WordToken("hi", 0.0, 0.5)], speaker="A")
    restored = restore_timestamps(branch)
    assert restored[0].start == 10.0
    assert restored[0].end == 10.5
    assert restored[0].speaker == "A"


def test_resolve_duplicates_drops_boundary_repeat():
    words = [
        WordToken("hello", 0.0, 0.5, speaker="A"),
        WordToken("hello", 0.48, 0.52, speaker="A"),  # near-duplicate at a chunk boundary
        WordToken("world", 0.6, 1.0, speaker="A"),
    ]
    resolved = resolve_duplicates(words)
    assert len(resolved) == 2


def test_fuse_transcript_groups_by_speaker_and_time():
    branches = [
        BranchOutput(chunk_start=0.0, words=[WordToken("hi", 0.0, 0.3)], speaker="A"),
        BranchOutput(chunk_start=0.5, words=[WordToken("there", 0.0, 0.3)], speaker="B"),
    ]
    transcript = fuse_transcript("rec1", branches)
    assert transcript.recording_id == "rec1"
    speakers = {u.speaker for u in transcript.utterances}
    assert speakers == {"A", "B"}


def test_fuse_transcript_preserves_simultaneous_speech():
    branches = [
        BranchOutput(chunk_start=0.0, words=[WordToken("hi", 0.0, 1.0)], speaker="A"),
        BranchOutput(chunk_start=0.0, words=[WordToken("yo", 0.2, 0.8)], speaker="B"),
    ]
    transcript = fuse_transcript("rec1", branches)
    assert len(transcript.utterances) == 2  # both speakers' overlapping words are kept, not merged
