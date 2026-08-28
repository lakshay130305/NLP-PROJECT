"""Overlap characterization, per section 3.5 of the paper plan.

Overlap is derived purely from an interval list (ground-truth or predicted),
by finding time ranges covered by >=2 simultaneously-active segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from schemas.types import OverlapRegion, SpeechSegment


class OverlapCategory(str, Enum):
    VERY_LOW = "very_low"  # < 5%
    LOW = "low"  # 5-10%
    MODERATE = "moderate"  # 10-20%
    HIGH = "high"  # > 20%


def categorize_overlap_ratio(overlap_ratio: float) -> OverlapCategory:
    if overlap_ratio < 0.05:
        return OverlapCategory.VERY_LOW
    if overlap_ratio < 0.10:
        return OverlapCategory.LOW
    if overlap_ratio < 0.20:
        return OverlapCategory.MODERATE
    return OverlapCategory.HIGH


def _boundary_events(segments: list[SpeechSegment]) -> list[tuple[float, int, str]]:
    events: list[tuple[float, int, str]] = []
    for seg in segments:
        events.append((seg.start, 1, seg.speaker or ""))
        events.append((seg.end, -1, seg.speaker or ""))
    # process ends before starts at identical timestamps so touching (non-overlapping) segments don't count as overlap
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def derive_overlap_regions(segments: list[SpeechSegment]) -> list[OverlapRegion]:
    """Sweep-line over segment boundaries to find intervals with >=2 concurrently active speakers."""
    if not segments:
        return []

    events = _boundary_events(segments)
    active: dict[str, int] = {}
    regions: list[OverlapRegion] = []
    region_start: float | None = None

    prev_t = events[0][0]
    for t, delta, spk in events:
        if t > prev_t:
            n_active = sum(c > 0 for c in active.values())
            if n_active >= 2 and region_start is None:
                region_start = prev_t
            elif n_active < 2 and region_start is not None:
                regions.append(
                    OverlapRegion(start=region_start, end=prev_t, speakers=tuple(sorted(k for k, c in active.items() if c > 0)))
                )
                region_start = None
        active[spk] = active.get(spk, 0) + delta
        prev_t = t

    if region_start is not None:
        regions.append(OverlapRegion(start=region_start, end=prev_t))

    return [r for r in regions if r.duration > 0]


def total_speech_duration(segments: list[SpeechSegment]) -> float:
    """Union of segment durations (does not double-count overlapping time)."""
    if not segments:
        return 0.0
    intervals = sorted((s.start, s.end) for s in segments)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


@dataclass
class OverlapStats:
    overlap_duration: float
    speech_duration: float
    overlap_ratio: float
    category: OverlapCategory
    num_regions: int


def compute_overlap_stats(segments: list[SpeechSegment]) -> OverlapStats:
    regions = derive_overlap_regions(segments)
    overlap_duration = sum(r.duration for r in regions)
    speech_duration = total_speech_duration(segments)
    ratio = (overlap_duration / speech_duration) if speech_duration > 0 else 0.0
    return OverlapStats(
        overlap_duration=overlap_duration,
        speech_duration=speech_duration,
        overlap_ratio=ratio,
        category=categorize_overlap_ratio(ratio),
        num_regions=len(regions),
    )
