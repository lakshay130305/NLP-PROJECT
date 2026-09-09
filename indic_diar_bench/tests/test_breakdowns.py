"""Per-stratum aggregation (language / condition / speaker-count / overlap-intensity),
plus the CLI parsing for the OSD-threshold sweep and condition filter."""

import pytest

from eval.evaluate import (
    RecordingResult,
    aggregate,
    aggregate_by,
    overlap_bucket,
    speaker_count_bucket,
)
from scripts.run_variant import parse_conditions, parse_taus


def _result(rid, **kw):
    defaults = {
        "der": 0.5, "der_missed": 0.1, "der_false_alarm": 0.1, "der_confusion": 0.3,
        "wder": 0.4, "cpwer": 1.0, "wer": 0.9, "rtf": 0.5, "osd_precision": 0.5,
        "osd_recall": 0.5, "osd_f1": 0.5, "routed_fraction": 0.2,
    }
    defaults.update(kw)
    return RecordingResult(recording_id=rid, **defaults)


def test_aggregate_reports_its_own_sample_size():
    assert aggregate([_result("a"), _result("b")])["n"] == 2
    assert aggregate([]) == {}


def test_si_sdr_averages_only_over_recordings_that_have_it():
    results = [_result("a", si_sdr=10.0), _result("b", si_sdr=None), _result("c", si_sdr=20.0)]
    agg = aggregate(results)
    assert agg["si_sdr"] == 15.0
    assert agg["n_si_sdr"] == 2


def test_si_sdr_is_nan_when_no_recording_has_stems():
    agg = aggregate([_result("a"), _result("b")])
    assert agg["si_sdr"] != agg["si_sdr"]  # NaN
    assert agg["n_si_sdr"] == 0


def test_grouping_by_language_keeps_per_group_counts():
    results = [
        _result("a", language="Hindi", der=0.4),
        _result("b", language="Hindi", der=0.6),
        _result("c", language="Telugu", der=0.2),
    ]
    by_lang = aggregate_by(results, "language")
    assert set(by_lang) == {"Hindi", "Telugu"}
    assert by_lang["Hindi"]["n"] == 2
    assert abs(by_lang["Hindi"]["der"] - 0.5) < 1e-9
    assert by_lang["Telugu"]["n"] == 1


def test_speaker_count_buckets_follow_the_papers_strata():
    assert speaker_count_bucket(2) == "2"
    assert speaker_count_bucket(3) == "3-4"
    assert speaker_count_bucket(4) == "3-4"
    assert speaker_count_bucket(5) == "5+"


def test_overlap_buckets_follow_the_papers_strata():
    assert overlap_bucket(0.01) == "very_low"
    assert overlap_bucket(0.07) == "low"
    assert overlap_bucket(0.15) == "moderate"
    assert overlap_bucket(0.30) == "high"


def test_overlap_groups_come_back_in_intensity_order_not_alphabetical():
    results = [
        _result("a", overlap_ratio=0.30),
        _result("b", overlap_ratio=0.01),
        _result("c", overlap_ratio=0.15),
    ]
    assert list(aggregate_by(results, "overlap")) == ["very_low", "moderate", "high"]


def test_speaker_groups_come_back_in_count_order():
    results = [_result("a", num_ref_speakers=6), _result("b", num_ref_speakers=2)]
    assert list(aggregate_by(results, "speakers")) == ["2", "5+"]


def test_unknown_grouping_key_is_rejected():
    with pytest.raises(ValueError):
        aggregate_by([_result("a")], "nonsense")


def test_parse_taus():
    assert parse_taus(None) == [None]
    assert parse_taus("0.5") == [0.5]
    assert parse_taus("0.3,0.5,0.7") == [0.3, 0.5, 0.7]
    with pytest.raises(SystemExit):
        parse_taus("1.5")
    with pytest.raises(SystemExit):
        parse_taus("high")


def test_parse_conditions():
    from schemas.types import AcousticCondition

    assert parse_conditions(None) is None
    assert parse_conditions(["near_field"]) == [AcousticCondition.NEAR_FIELD]
    with pytest.raises(SystemExit):
        parse_conditions(["nearfield"])
