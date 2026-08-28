from schemas.overlap import (
    OverlapCategory,
    categorize_overlap_ratio,
    compute_overlap_stats,
    derive_overlap_regions,
)
from schemas.types import SpeechSegment


def test_no_overlap():
    segs = [SpeechSegment(0, 1, "A"), SpeechSegment(1.5, 2.5, "B")]
    assert derive_overlap_regions(segs) == []


def test_simple_overlap():
    segs = [SpeechSegment(0, 2, "A"), SpeechSegment(1, 3, "B")]
    regions = derive_overlap_regions(segs)
    assert len(regions) == 1
    assert regions[0].start == 1
    assert regions[0].end == 2


def test_touching_segments_not_overlap():
    segs = [SpeechSegment(0, 1, "A"), SpeechSegment(1, 2, "B")]
    assert derive_overlap_regions(segs) == []


def test_three_way_overlap():
    segs = [SpeechSegment(0, 3, "A"), SpeechSegment(1, 4, "B"), SpeechSegment(2, 5, "C")]
    regions = derive_overlap_regions(segs)
    total = sum(r.duration for r in regions)
    assert total > 0


def test_overlap_categories():
    assert categorize_overlap_ratio(0.02) == OverlapCategory.VERY_LOW
    assert categorize_overlap_ratio(0.07) == OverlapCategory.LOW
    assert categorize_overlap_ratio(0.15) == OverlapCategory.MODERATE
    assert categorize_overlap_ratio(0.30) == OverlapCategory.HIGH


def test_overlap_stats():
    segs = [SpeechSegment(0, 2, "A"), SpeechSegment(1, 3, "B")]
    stats = compute_overlap_stats(segs)
    assert stats.speech_duration == 3.0
    assert stats.overlap_duration == 1.0
    assert abs(stats.overlap_ratio - (1 / 3)) < 1e-9
