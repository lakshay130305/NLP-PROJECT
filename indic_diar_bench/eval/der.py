"""Diarization Error Rate (DER), section 12.1.

DER = (false_alarm + missed_speech + speaker_confusion) / total_reference_speech_time

Implemented directly via a sweep-line over reference/hypothesis segments so this
module has zero heavy dependencies (no pyannote.metrics import needed for the
core arithmetic -- useful for fast unit testing and for environments where
torch/pyannote aren't importable yet).
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass

from schemas.types import SpeechSegment


@dataclass
class DERResult:
    der: float
    missed_speech: float
    false_alarm: float
    speaker_confusion: float
    total_reference_time: float


def _events(segments: list[SpeechSegment], tag: str) -> list[tuple[float, int, str, str]]:
    ev = []
    for seg in segments:
        ev.append((seg.start, 1, tag, seg.speaker or ""))
        ev.append((seg.end, -1, tag, seg.speaker or ""))
    return ev


def _optimal_speaker_mapping(
    reference: list[SpeechSegment], hypothesis: list[SpeechSegment]
) -> dict[str, str]:
    """Map hypothesis speaker labels -> reference speaker labels by maximizing total overlap time
    (Hungarian algorithm over the cost = -overlap matrix)."""
    ref_speakers = sorted({s.speaker for s in reference if s.speaker})
    hyp_speakers = sorted({s.speaker for s in hypothesis if s.speaker})
    if not ref_speakers or not hyp_speakers:
        return {}

    overlap = [[0.0] * len(hyp_speakers) for _ in ref_speakers]
    for i, rspk in enumerate(ref_speakers):
        r_segs = [s for s in reference if s.speaker == rspk]
        for j, hspk in enumerate(hyp_speakers):
            h_segs = [s for s in hypothesis if s.speaker == hspk]
            overlap[i][j] = sum(rs.intersection(hs) for rs in r_segs for hs in h_segs)

    row_ind, col_ind = _hungarian_max(overlap)
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if overlap[i][j] > 0:
            mapping[hyp_speakers[j]] = ref_speakers[i]
    return mapping


def _hungarian_max(matrix: list[list[float]]) -> tuple[list[int], list[int]]:
    """Small dependency-free assignment solver (maximize sum) via scipy if available, else greedy fallback."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = -np.array(matrix)
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(row_ind), list(col_ind)
    except ImportError:
        # greedy fallback: repeatedly pick the largest remaining cell
        n_rows, n_cols = len(matrix), len(matrix[0])
        used_rows, used_cols = set(), set()
        pairs = []
        cells = sorted(
            ((matrix[i][j], i, j) for i in range(n_rows) for j in range(n_cols)),
            reverse=True,
        )
        for val, i, j in cells:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            pairs.append((i, j))
        pairs.sort()
        return [p[0] for p in pairs], [p[1] for p in pairs]


def compute_der(
    reference: list[SpeechSegment],
    hypothesis: list[SpeechSegment],
    collar: float = 0.0,
    score_regions: list[tuple[float, float]] | None = None,
) -> DERResult:
    """Compute DER via a fine-grained frame sweep. `collar` (seconds) excludes a margin
    around reference segment boundaries from scoring, matching common DER conventions.

    `score_regions`, if given, restricts scoring to those time intervals: only slices whose
    midpoint falls inside a region are charged, and the denominator is the reference speech
    time inside those regions. This is what makes an overlap-vs-non-overlap DER split
    possible (pass the reference overlap regions, or their complement) -- an aggregate DER
    cannot say whether a system's error is concentrated in the overlapping regions it was
    designed for or spread across the easy ones."""
    regions = _merge_regions(score_regions) if score_regions is not None else None
    if regions is not None and not regions:
        return DERResult(0.0, 0.0, 0.0, 0.0, 0.0)

    if regions is None:
        total_ref_time = sum(s.duration for s in reference)
    else:
        total_ref_time = sum(_region_intersection(s.start, s.end, regions) for s in reference)
    if total_ref_time == 0:
        return DERResult(0.0, 0.0, 0.0, 0.0, 0.0)

    mapping = _optimal_speaker_mapping(reference, hypothesis)
    mapped_hyp = [
        SpeechSegment(s.start, s.end, mapping.get(s.speaker, s.speaker)) for s in hypothesis
    ]

    boundaries = sorted({s.start for s in reference} | {s.end for s in reference}
                         | {s.start for s in mapped_hyp} | {s.end for s in mapped_hyp}
                         | ({b for r in regions for b in r} if regions else set()))

    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0

    for a, b in itertools.pairwise(boundaries):
        mid = (a + b) / 2
        dur = b - a
        if dur <= 0:
            continue
        if regions is not None and not _in_regions(mid, regions):
            continue
        if collar > 0 and _within_collar(mid, reference, collar):
            continue

        ref_active = {s.speaker for s in reference if s.start <= mid < s.end}
        hyp_active = {s.speaker for s in mapped_hyp if s.start <= mid < s.end}

        n_ref, n_hyp = len(ref_active), len(hyp_active)
        correct = len(ref_active & hyp_active)

        if n_hyp > n_ref:
            false_alarm += (n_hyp - max(n_ref, correct)) * dur
        if n_ref > n_hyp:
            missed += (n_ref - max(n_hyp, correct)) * dur
        mismatched = min(n_ref, n_hyp) - correct
        if mismatched > 0:
            confusion += mismatched * dur

    der = (missed + false_alarm + confusion) / total_ref_time
    return DERResult(der, missed, false_alarm, confusion, total_ref_time)


def _within_collar(t: float, reference: list[SpeechSegment], collar: float) -> bool:
    for seg in reference:
        if abs(t - seg.start) < collar or abs(t - seg.end) < collar:
            return True
    return False


def _merge_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and coalesce touching/overlapping intervals so containment tests can binary-search."""
    ordered = sorted((float(s), float(e)) for s, e in regions if e > s)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def complement_regions(
    regions: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """The parts of [start, end) not covered by `regions` -- i.e. the non-overlap timeline
    when `regions` are the reference overlap regions."""
    merged = _merge_regions(regions)
    out: list[tuple[float, float]] = []
    cursor = start
    for r_start, r_end in merged:
        if r_end <= start or r_start >= end:
            continue
        if r_start > cursor:
            out.append((cursor, min(r_start, end)))
        cursor = max(cursor, r_end)
    if cursor < end:
        out.append((cursor, end))
    return [(s, e) for s, e in out if e > s]


def _in_regions(t: float, regions: list[tuple[float, float]]) -> bool:
    i = bisect.bisect_right(regions, (t, float("inf"))) - 1
    return i >= 0 and regions[i][0] <= t < regions[i][1]


def _region_intersection(start: float, end: float, regions: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, r_end) - max(start, r_start)) for r_start, r_end in regions)
