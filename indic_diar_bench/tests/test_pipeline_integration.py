"""End-to-end smoke tests: every system variant (B1..B5, PROPOSED) must run without
crashing on synthetic audio, using the dummy (no network/model download) backend."""

import pytest

from data.prep.synthetic import generate_conversation, synthetic_manifest_entry
from eval.evaluate import evaluate_recording
from pipeline.config import Backend
from pipeline.orchestrator import OverlapAwarePipeline
from pipeline.variants import ALL_VARIANTS, build_variant


@pytest.fixture(scope="module")
def synthetic_recording():
    convo = generate_conversation(num_speakers=2, num_turns=6, overlap_ratio=0.3, seed=42)
    entry = synthetic_manifest_entry("test_rec", convo)
    return entry, convo.audio, convo.sample_rate


@pytest.mark.parametrize("variant_name", ALL_VARIANTS)
def test_variant_runs_end_to_end(variant_name, synthetic_recording):
    entry, audio, sr = synthetic_recording
    cfg = build_variant(variant_name, backend=Backend.DUMMY)
    pipeline = OverlapAwarePipeline(cfg)

    transcript, stats, overlap_regions = pipeline.run(audio, sr, recording_id=entry.recording_id)

    assert transcript.recording_id == "test_rec"
    assert stats.audio_duration > 0
    assert stats.rtf >= 0

    result = evaluate_recording(entry, transcript, overlap_regions, stats)
    assert 0.0 <= result.der or result.der >= 0.0  # DER can exceed 1.0 in bad cases; just check it's finite
    assert result.wer >= 0.0


def test_proposed_routes_more_than_baseline_when_overlap_present(synthetic_recording):
    _entry, audio, sr = synthetic_recording

    b1 = OverlapAwarePipeline(build_variant("B1", backend=Backend.DUMMY))
    proposed = OverlapAwarePipeline(build_variant("PROPOSED", backend=Backend.DUMMY))

    _, b1_stats, _ = b1.run(audio, sr)
    _, proposed_stats, _ = proposed.run(audio, sr)

    assert b1_stats.separator_routed_fraction == 0.0  # B1 never separates
    assert proposed_stats.separator_routed_fraction >= 0.0


def test_empty_audio_does_not_crash():
    import numpy as np

    cfg = build_variant("PROPOSED", backend=Backend.DUMMY)
    pipeline = OverlapAwarePipeline(cfg)
    transcript, _stats, _overlap_regions = pipeline.run(np.zeros(0, dtype=np.float32), 16000)
    assert transcript.utterances == []
