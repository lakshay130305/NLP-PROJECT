"""Region-restricted DER -- the machinery behind the overlap-vs-non-overlap split."""

from eval.der import complement_regions, compute_der
from schemas.types import SpeechSegment


def test_unrestricted_matches_previous_behaviour():
    ref = [SpeechSegment(0, 2, "A"), SpeechSegment(1.5, 3, "B")]
    hyp = [SpeechSegment(0, 2, "A")]
    assert compute_der(ref, hyp).der == compute_der(ref, hyp, score_regions=None).der


def test_scoring_restricted_to_a_region_ignores_error_outside_it():
    ref = [SpeechSegment(0, 4, "A")]
    hyp = [SpeechSegment(0, 2, "A")]  # second half is missed
    # scored only over the first half, where the hypothesis is perfect
    assert compute_der(ref, hyp, score_regions=[(0.0, 2.0)]).der < 1e-9
    # scored only over the second half, where it misses everything
    assert compute_der(ref, hyp, score_regions=[(2.0, 4.0)]).der == 1.0


def test_denominator_is_reference_time_inside_the_regions():
    ref = [SpeechSegment(0, 10, "A")]
    hyp = []
    result = compute_der(ref, hyp, score_regions=[(0.0, 3.0)])
    assert result.total_reference_time == 3.0
    assert result.missed_speech == 3.0


def test_empty_region_list_scores_nothing():
    ref = [SpeechSegment(0, 2, "A")]
    assert compute_der(ref, [], score_regions=[]).der == 0.0


def test_overlapping_score_regions_are_merged_not_double_counted():
    ref = [SpeechSegment(0, 10, "A")]
    merged = compute_der(ref, [], score_regions=[(0.0, 6.0), (4.0, 8.0)])
    assert merged.total_reference_time == 8.0


def test_complement_is_the_rest_of_the_timeline():
    assert complement_regions([(2.0, 4.0)], 0.0, 10.0) == [(0.0, 2.0), (4.0, 10.0)]
    assert complement_regions([], 0.0, 5.0) == [(0.0, 5.0)]
    assert complement_regions([(0.0, 5.0)], 0.0, 5.0) == []


def test_overlap_and_nonoverlap_partition_the_total_error():
    """Error charged inside the overlap regions plus error charged outside them should
    account for all of it -- otherwise the split would be hiding or duplicating error."""
    ref = [SpeechSegment(0, 6, "A"), SpeechSegment(2, 4, "B")]
    hyp = [SpeechSegment(0, 3, "A")]
    overlap = [(2.0, 4.0)]
    inside = compute_der(ref, hyp, score_regions=overlap)
    outside = compute_der(ref, hyp, score_regions=complement_regions(overlap, 0.0, 6.0))
    whole = compute_der(ref, hyp)

    total_err = inside.missed_speech + inside.false_alarm + inside.speaker_confusion
    total_err += outside.missed_speech + outside.false_alarm + outside.speaker_confusion
    whole_err = whole.missed_speech + whole.false_alarm + whole.speaker_confusion
    assert abs(total_err - whole_err) < 1e-6
    assert abs(inside.total_reference_time + outside.total_reference_time - whole.total_reference_time) < 1e-6
