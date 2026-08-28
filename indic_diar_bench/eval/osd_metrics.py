"""Overlap detection metrics, section 12.5: frame-level precision/recall/F1
between predicted overlap regions and ground-truth overlap regions."""

from __future__ import annotations

from dataclasses import dataclass

from schemas.types import OverlapRegion


@dataclass
class OSDMetrics:
    precision: float
    recall: float
    f1: float
    true_positive_time: float
    false_positive_time: float
    false_negative_time: float


def _covered_duration(regions: list[OverlapRegion]) -> float:
    if not regions:
        return 0.0
    intervals = sorted((r.start, r.end) for r in regions)
    merged: list[list[float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def _intersection_duration(a: list[OverlapRegion], b: list[OverlapRegion]) -> float:
    total = 0.0
    for ra in a:
        for rb in b:
            total += max(0.0, min(ra.end, rb.end) - max(ra.start, rb.start))
    return total


def compute_osd_metrics(
    reference: list[OverlapRegion], predicted: list[OverlapRegion]
) -> OSDMetrics:
    ref_dur = _covered_duration(reference)
    pred_dur = _covered_duration(predicted)
    tp = _intersection_duration(reference, predicted)
    tp = min(tp, ref_dur, pred_dur)  # guard against double counting from overlapping merged regions

    fp = max(0.0, pred_dur - tp)
    fn = max(0.0, ref_dur - tp)

    precision = tp / pred_dur if pred_dur > 0 else 0.0
    recall = tp / ref_dur if ref_dur > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return OSDMetrics(precision=precision, recall=recall, f1=f1,
                       true_positive_time=tp, false_positive_time=fp, false_negative_time=fn)
