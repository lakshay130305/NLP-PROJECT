"""Per-recording result persistence, resume, and pooling arithmetic."""

import json

from eval.evaluate import RecordingResult, aggregate
from eval.results_io import (
    load_completed,
    open_results_file,
    read_results,
    write_result,
)


def _result(rid, der=0.5, **kw):
    defaults = {
        "der": der, "der_missed": 0.1, "der_false_alarm": 0.1, "der_confusion": 0.3,
        "wder": 0.4, "cpwer": 1.0, "wer": 0.9, "rtf": 0.5, "osd_precision": 0.5,
        "osd_recall": 0.5, "osd_f1": 0.5, "routed_fraction": 0.2,
    }
    defaults.update(kw)
    return RecordingResult(recording_id=rid, **defaults)


def test_roundtrip_preserves_every_field(tmp_path):
    path = tmp_path / "r.jsonl"
    original = _result("rec1", language="Hindi", condition="near_field",
                       num_ref_speakers=3, overlap_ratio=0.12, si_sdr=7.5,
                       der_overlap=0.6, der_nonoverlap=0.4)
    with open_results_file(path) as fh:
        write_result(fh, "PROPOSED", original)

    loaded = read_results([path])
    assert list(loaded) == ["PROPOSED"]
    assert loaded["PROPOSED"][0] == original


def test_none_si_sdr_survives_the_roundtrip(tmp_path):
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec1", si_sdr=None))
    assert read_results([path])["B1"][0].si_sdr is None


def test_appends_rather_than_truncates(tmp_path):
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec1"))
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec2"))
    assert len(read_results([path])["B1"]) == 2


def test_pools_across_several_files(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with open_results_file(a) as fh:
        write_result(fh, "B1", _result("hindi_1", language="Hindi"))
    with open_results_file(b) as fh:
        write_result(fh, "B1", _result("santali_1", language="Santali"))
    assert len(read_results([a, b])["B1"]) == 2


def test_duplicate_recording_id_is_deduplicated(tmp_path):
    """Re-running a language after a crash and pooling both logs must not double-count it."""
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec1", der=0.9))
        write_result(fh, "B1", _result("rec1", der=0.4))  # the rerun
    results = read_results([path])["B1"]
    assert len(results) == 1
    assert results[0].der == 0.4  # last write wins


def test_same_recording_under_different_variants_is_kept(tmp_path):
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec1"))
        write_result(fh, "PROPOSED", _result("rec1"))
    loaded = read_results([path])
    assert len(loaded["B1"]) == 1 and len(loaded["PROPOSED"]) == 1


def test_load_completed_keys_by_variant_and_recording(tmp_path):
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec1"))
        write_result(fh, "B1", _result("rec2"))
        write_result(fh, "PROPOSED", _result("rec1"))
    done = load_completed(path)
    assert set(done["B1"]) == {"rec1", "rec2"}
    assert set(done["PROPOSED"]) == {"rec1"}


def test_load_completed_on_missing_file_is_empty_not_an_error(tmp_path):
    assert load_completed(tmp_path / "nope.jsonl") == {}


def test_unknown_fields_are_ignored(tmp_path):
    """A log written by a future version must still load, not crash."""
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({
        "variant": "B1", "recording_id": "rec1", "der": 0.5, "some_future_metric": 1.23,
    }) + "\n", encoding="utf-8")
    assert read_results([path])["B1"][0].der == 0.5


def test_truncated_final_line_does_not_lose_the_completed_work(tmp_path):
    """A process killed mid-write leaves a partial last line. That is exactly the case
    --resume exists for, so the completed records before it must still load."""
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        write_result(fh, "B1", _result("rec1"))
        write_result(fh, "B1", _result("rec2"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"variant": "B1", "recording_id": "rec3", "der": 0.4')  # killed here

    results = read_results([path])["B1"]
    assert {r.recording_id for r in results} == {"rec1", "rec2"}


def test_pooling_means_over_recordings_not_over_languages(tmp_path):
    """The reason this module exists: with unequal per-language n, pooling the raw
    recordings and averaging the per-language means give different answers, and only the
    former is what the results table reports."""
    path = tmp_path / "r.jsonl"
    with open_results_file(path) as fh:
        for i in range(10):
            write_result(fh, "B1", _result(f"hindi_{i}", der=0.50, language="Hindi"))
        for i in range(2):
            write_result(fh, "B1", _result(f"santali_{i}", der=0.80, language="Santali"))

    pooled = aggregate(read_results([path])["B1"])["der"]
    assert abs(pooled - (10 * 0.50 + 2 * 0.80) / 12) < 1e-9  # 0.55
    assert abs(pooled - (0.50 + 0.80) / 2) > 0.09            # NOT the 0.65 macro-average
