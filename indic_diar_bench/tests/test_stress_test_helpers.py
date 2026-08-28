"""Unit tests for stress_test.py's pure-logic helpers (file I/O, resumability) --
the parts that don't require a real model/dataset connection to exercise."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
from stress_test import (
    append_jsonl,
    load_processed_ids,
    truncate_recording,
    write_summary,
)

from schemas.types import (
    AcousticCondition,
    ManifestEntry,
    SpeakerAttributedTranscript,
    SpeechSegment,
    Utterance,
    WordToken,
)


def _make_entry(duration=10.0):
    segments = [
        SpeechSegment(0.0, 3.0, "A"),
        SpeechSegment(3.5, 7.0, "B"),
        SpeechSegment(7.5, 10.0, "A"),
    ]
    words = [
        WordToken("hi", 0.0, 0.5, "A"),
        WordToken("there", 4.0, 4.5, "B"),
        WordToken("end", 8.0, 8.5, "A"),
    ]
    transcript = SpeakerAttributedTranscript(
        recording_id="r1",
        utterances=[Utterance(w.speaker, w.start, w.end, [w]) for w in words],
    )
    return ManifestEntry(
        recording_id="r1", audio_path="", language="Hindi", condition=AcousticCondition.NEAR_FIELD,
        duration=duration, reference_segments=segments, reference_transcript=transcript,
    )


def test_append_jsonl_appends_one_line_per_call(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"recording_id": "a", "der": 0.1})
    append_jsonl(path, {"recording_id": "b", "der": 0.2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_load_processed_ids_reads_both_files(tmp_path):
    results = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    append_jsonl(results, {"recording_id": "a"})
    append_jsonl(failures, {"recording_id": "b"})

    seen = load_processed_ids(results, failures)
    assert seen == {"a", "b"}


def test_load_processed_ids_handles_missing_files(tmp_path):
    seen = load_processed_ids(tmp_path / "missing1.jsonl", tmp_path / "missing2.jsonl")
    assert seen == set()


def test_load_processed_ids_skips_malformed_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("not valid json\n" + '{"recording_id": "a"}\n', encoding="utf-8")
    seen = load_processed_ids(path, tmp_path / "missing.jsonl")
    assert seen == {"a"}


def test_write_summary_overwrites_previous_content(tmp_path):
    path = tmp_path / "summary.json"
    write_summary(path, {"n_success": 1})
    write_summary(path, {"n_success": 2})

    import json

    assert json.loads(path.read_text(encoding="utf-8"))["n_success"] == 2


def test_jsonl_round_trip_preserves_unicode(tmp_path):
    """Confirms the append_jsonl path doesn't reintroduce the UnicodeEncodeError class of
    bug fixed in scripts/_stdio.py -- non-Latin-1 text must survive a write+read cycle."""
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"recording_id": "a", "note": "हिन्दी नमूना पाठ"})
    seen_ids = load_processed_ids(path, tmp_path / "missing.jsonl")
    assert seen_ids == {"a"}

    import json

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["note"] == "हिन्दी नमूना पाठ"


def test_truncate_recording_shortens_audio_and_segments():
    entry = _make_entry(duration=10.0)
    audio = np.arange(160000, dtype=np.float32)  # 10s at 16kHz
    truncated_entry, truncated_audio = truncate_recording(entry, audio, 16000, max_seconds=5.0)

    assert len(truncated_audio) == 80000
    assert truncated_entry.duration == 5.0
    assert all(s.end <= 5.0 for s in truncated_entry.reference_segments)
    # segment [3.5, 7.0) should be clipped to [3.5, 5.0), not dropped
    assert any(abs(s.start - 3.5) < 1e-9 and abs(s.end - 5.0) < 1e-9 for s in truncated_entry.reference_segments)
    # segment [7.5, 10.0) starts after the cutoff -> dropped entirely
    assert not any(s.start >= 5.0 for s in truncated_entry.reference_segments)


def test_truncate_recording_clips_reference_words():
    entry = _make_entry(duration=10.0)
    audio = np.arange(160000, dtype=np.float32)
    truncated_entry, _audio = truncate_recording(entry, audio, 16000, max_seconds=5.0)

    all_words = [w for u in truncated_entry.reference_transcript.utterances for w in u.words]
    assert all(w.start < 5.0 for w in all_words)
    assert len(all_words) == 2  # "hi" (0.0) and "there" (4.0) survive; "end" (8.0) is dropped


def test_truncate_recording_noop_when_already_short():
    entry = _make_entry(duration=10.0)
    audio = np.arange(160000, dtype=np.float32)
    truncated_entry, truncated_audio = truncate_recording(entry, audio, 16000, max_seconds=20.0)

    assert len(truncated_audio) == len(audio)
    assert truncated_entry.duration == 10.0


def test_truncate_recording_disabled_when_max_seconds_none_or_nonpositive():
    entry = _make_entry(duration=10.0)
    audio = np.arange(160000, dtype=np.float32)

    e1, a1 = truncate_recording(entry, audio, 16000, max_seconds=None)
    assert len(a1) == len(audio) and e1 is entry

    e2, a2 = truncate_recording(entry, audio, 16000, max_seconds=0)
    assert len(a2) == len(audio) and e2 is entry
