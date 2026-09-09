"""Per-recording evaluation, wiring together section 13.1's metric table
(DER, WDER, cpWER, WER, RTF) plus OSD precision/recall/F1 (12.5)."""

from __future__ import annotations

from dataclasses import dataclass

from eval.cpwer import compute_cpwer
from eval.der import complement_regions, compute_der
from eval.efficiency import EfficiencyStats
from eval.osd_metrics import compute_osd_metrics
from eval.separation_quality import score_separated_segments
from eval.wder import compute_wder
from eval.wer import compute_wer
from schemas.overlap import (
    OverlapCategory,
    categorize_overlap_ratio,
    compute_overlap_stats,
    derive_overlap_regions,
)
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
    # DER restricted to the reference overlap regions vs everything else. The aggregate DER
    # above cannot distinguish "helps exactly where it was designed to" from "shifts error
    # around", which is the whole question an overlap-aware system has to answer.
    der_overlap: float = 0.0
    der_nonoverlap: float = 0.0
    # grouping keys for the per-language / per-condition / per-speaker-count / per-overlap
    # breakdowns -- carried on the result so the CLI can slice without re-reading the manifest
    language: str = ""
    condition: str = ""
    num_ref_speakers: int = 0
    overlap_ratio: float = 0.0
    # separation quality (SI-SDR, dB) against ground-truth isolated sources, when the
    # recording has them; None on real dataset recordings, which ship only the mixture
    si_sdr: float | None = None


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

    overlap_spans = [(r.start, r.end) for r in ref_overlap_regions]
    # the non-overlap timeline has to extend past the last hypothesis segment too, or false
    # alarms after the end of reference speech would fall outside every scoring region
    timeline_end = max(
        [entry.duration] + [s.end for s in reference_segments] + [s.end for s in predicted_segments]
    )
    der_overlap = compute_der(
        reference_segments, predicted_segments, collar=DER_COLLAR, score_regions=overlap_spans
    ).der
    der_nonoverlap = compute_der(
        reference_segments,
        predicted_segments,
        collar=DER_COLLAR,
        score_regions=complement_regions(overlap_spans, 0.0, timeline_end),
    ).der

    overlap_stats = compute_overlap_stats(reference_segments)

    # SI-SDR only when the recording ships ground-truth stems AND the pipeline was asked to
    # keep its separated waveforms; on the real dataset both are absent and this stays None.
    separation_quality = score_separated_segments(
        entry.source_tracks or {},
        entry.sample_rate or 16000,
        stats.separated_segments,
        reference_segments,
    ) if entry.source_tracks and stats.separated_segments else None
    si_sdr = separation_quality.si_sdr if separation_quality else None

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
        der_overlap=der_overlap,
        der_nonoverlap=der_nonoverlap,
        language=entry.language,
        condition=entry.condition.value if hasattr(entry.condition, "value") else str(entry.condition),
        num_ref_speakers=len({s.speaker for s in reference_segments if s.speaker}),
        overlap_ratio=overlap_stats.overlap_ratio,
        si_sdr=si_sdr,
    )


AGGREGATE_FIELDS = (
    "der", "der_missed", "der_false_alarm", "der_confusion", "der_overlap", "der_nonoverlap",
    "wder", "cpwer", "wer", "rtf", "osd_precision", "osd_recall", "osd_f1", "routed_fraction",
    "overlap_ratio",
)


def aggregate(results: list[RecordingResult]) -> dict[str, float]:
    if not results:
        return {}
    agg = {f: sum(getattr(r, f) for r in results) / len(results) for f in AGGREGATE_FIELDS}
    agg["n"] = len(results)
    # si_sdr is None on recordings without ground-truth stems -- average over the ones that
    # have it rather than letting a single None poison the whole column
    scored = [r.si_sdr for r in results if r.si_sdr is not None]
    agg["si_sdr"] = sum(scored) / len(scored) if scored else float("nan")
    agg["n_si_sdr"] = len(scored)
    return agg


def speaker_count_bucket(num_speakers: int) -> str:
    """Section 17.3's speaker-count strata: 2 vs 3-4 vs 5+."""
    if num_speakers <= 2:
        return "2"
    if num_speakers <= 4:
        return "3-4"
    return "5+"


def overlap_bucket(overlap_ratio: float) -> str:
    """Section 3.5's overlap-intensity strata (very low / low / moderate / high)."""
    return categorize_overlap_ratio(overlap_ratio).value


_GROUP_KEYS = {
    "language": lambda r: r.language or "(unknown)",
    "condition": lambda r: r.condition or "(unknown)",
    "speakers": lambda r: speaker_count_bucket(r.num_ref_speakers),
    "overlap": lambda r: overlap_bucket(r.overlap_ratio),
}


def aggregate_by(results: list[RecordingResult], key: str) -> dict[str, dict[str, float]]:
    """Aggregate the same metrics within each stratum of `key`, for the per-language,
    per-condition, per-speaker-count and per-overlap-intensity breakdowns. Each returned
    row carries its own `n`, since a stratum holding one recording cannot support the same
    claims as one holding thirty and the reader has to be able to see which is which."""
    key_fn = _GROUP_KEYS.get(key)
    if key_fn is None:
        raise ValueError(f"Unknown grouping key '{key}'. Known keys: {sorted(_GROUP_KEYS)}")

    groups: dict[str, list[RecordingResult]] = {}
    for r in results:
        groups.setdefault(key_fn(r), []).append(r)

    if key == "overlap":
        order = [c.value for c in OverlapCategory]
        ordered = sorted(groups, key=lambda g: order.index(g) if g in order else len(order))
    elif key == "speakers":
        order = ["2", "3-4", "5+"]
        ordered = sorted(groups, key=lambda g: order.index(g) if g in order else len(order))
    else:
        ordered = sorted(groups)
    return {g: aggregate(groups[g]) for g in ordered}
