"""Per-recording evaluation, wiring together section 13.1's metric table
(DER, WDER, cpWER, WER, RTF) plus OSD precision/recall/F1 (12.5)."""

from __future__ import annotations

from dataclasses import dataclass

from eval.cpwer import compute_cpwer
from eval.der import compute_der
from eval.efficiency import EfficiencyStats
from eval.osd_metrics import compute_osd_metrics
from eval.wder import compute_wder
from eval.wer import compute_wer
from schemas.overlap import derive_overlap_regions
from schemas.types import (
    ManifestEntry,
    OverlapRegion,
    SpeakerAttributedTranscript,
    SpeechSegment,
)


@dataclass
class RecordingResult:
    recording_id: str
    der: float
    der_missed: float
    der_false_alarm: float
    der_confusion: float
    wder: float
    cpwer: float
    wer: float
    rtf: float
    osd_precision: float
    osd_recall: float
    osd_f1: float
    routed_fraction: float


DER_COLLAR = 0.25  # seconds; matches the forgiveness collar used by the external baselines
# this project compares against (README's "beating B1" section, arXiv:2607.23808) -- human
# boundary annotations are noisy at the +-100-200ms level, and scoring with collar=0.0 (the
# previous default here) charges every recording for that annotation noise on top of real
# error, making our DER not apples-to-apples with the numbers we're benchmarking against.


def evaluate_recording(
    entry: ManifestEntry,
    predicted_transcript: SpeakerAttributedTranscript,
    predicted_overlap_regions: list[OverlapRegion],
    stats: EfficiencyStats,
) -> RecordingResult:
    reference_segments = entry.reference_segments
    predicted_segments = [
        SpeechSegment(u.start, u.end, u.speaker) for u in predicted_transcript.utterances
    ]
    der_result = compute_der(reference_segments, predicted_segments, collar=DER_COLLAR)

    ref_words = [w for u in (entry.reference_transcript.utterances if entry.reference_transcript else []) for w in u.words]
    hyp_words = [w for u in predicted_transcript.utterances for w in u.words]

    wder_result = compute_wder(ref_words, hyp_words)
    cpwer_result = compute_cpwer(ref_words, hyp_words)
    wer_result = compute_wer(
        " ".join(w.text for w in sorted(ref_words, key=lambda w: w.start)),
        " ".join(w.text for w in sorted(hyp_words, key=lambda w: w.start)),
    )

    ref_overlap_regions = derive_overlap_regions(reference_segments)
    osd_result = compute_osd_metrics(ref_overlap_regions, predicted_overlap_regions)

    ref_time = der_result.total_reference_time or 1.0
    return RecordingResult(
        recording_id=entry.recording_id,
        der=der_result.der,
        der_missed=der_result.missed_speech / ref_time,
        der_false_alarm=der_result.false_alarm / ref_time,
        der_confusion=der_result.speaker_confusion / ref_time,
        wder=wder_result.wder,
        cpwer=cpwer_result.cpwer,
        wer=wer_result.wer,
        rtf=stats.rtf,
        osd_precision=osd_result.precision,
        osd_recall=osd_result.recall,
        osd_f1=osd_result.f1,
        routed_fraction=stats.separator_routed_fraction,
    )


def aggregate(results: list[RecordingResult]) -> dict[str, float]:
    if not results:
        return {}
    fields = [
        "der", "der_missed", "der_false_alarm", "der_confusion", "wder", "cpwer", "wer", "rtf",
        "osd_precision", "osd_recall", "osd_f1", "routed_fraction",
    ]
    return {f: sum(getattr(r, f) for r in results) / len(results) for f in fields}
