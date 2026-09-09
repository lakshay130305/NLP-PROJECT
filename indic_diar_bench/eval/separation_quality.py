"""Separation-quality metrics (SI-SDR / SDR), section 12.6.

Until now separation was only measurable *indirectly*, through its downstream effect on
DER/WDER/cpWER -- which cannot distinguish "the separator produced clean streams and
something later lost them" from "the separator produced artefacts". These metrics score the
separated waveforms directly.

Scoring needs ground-truth isolated sources (stems). Indic DiarBench publishes only the
mixed recording plus an RTTM, so SI-SDR is *not* computable on the real dataset; it is
computable on the synthetic generator (data/prep/synthetic.py), which mixes per-speaker
tracks it can hand back. Callers therefore treat a `None` result as "no stems available"
rather than as a failure.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

_EPS = 1e-8


@dataclass
class SeparationQuality:
    si_sdr: float
    sdr: float
    num_scored_segments: int


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Projects the estimate onto the reference so an overall
    gain difference (which a separator is free to introduce) is not charged as error."""
    reference, estimate = _align(reference, estimate)
    if reference.size == 0:
        return float("nan")
    ref_energy = float(np.dot(reference, reference)) + _EPS
    alpha = float(np.dot(estimate, reference)) / ref_energy
    target = alpha * reference
    noise = estimate - target
    return 10.0 * np.log10((float(np.dot(target, target)) + _EPS) / (float(np.dot(noise, noise)) + _EPS))


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Plain (scale-dependent) SDR in dB."""
    reference, estimate = _align(reference, estimate)
    if reference.size == 0:
        return float("nan")
    noise = estimate - reference
    return 10.0 * np.log10((float(np.dot(reference, reference)) + _EPS) / (float(np.dot(noise, noise)) + _EPS))


def best_permutation_si_sdr(
    references: list[np.ndarray], estimates: list[np.ndarray]
) -> tuple[float, float]:
    """Permutation-invariant SI-SDR/SDR: a separator has no way to know which output stream
    corresponds to which speaker, so it is scored under the assignment most favourable to it
    (standard PIT evaluation). Returns (si_sdr, sdr), both averaged over matched pairs.

    Handles a stream/source count mismatch by scoring min(len) pairs -- an over- or
    under-separated segment is scored on what it did produce, and the count mismatch itself
    shows up in the diarization metrics rather than being double-charged here."""
    n = min(len(references), len(estimates))
    if n == 0:
        return float("nan"), float("nan")

    best = (-np.inf, -np.inf)
    for perm in itertools.permutations(range(len(estimates)), n):
        pair_si = [si_sdr(references[i], estimates[perm[i]]) for i in range(n)]
        if any(np.isnan(v) for v in pair_si):
            continue
        mean_si = float(np.mean(pair_si))
        if mean_si > best[0]:
            mean_sdr = float(np.mean([sdr(references[i], estimates[perm[i]]) for i in range(n)]))
            best = (mean_si, mean_sdr)
    return best if np.isfinite(best[0]) else (float("nan"), float("nan"))


def score_separated_segments(
    source_tracks: dict[str, np.ndarray],
    sample_rate: int,
    separated_segments: list[tuple[float, float, list[np.ndarray]]],
    reference_segments,
) -> SeparationQuality | None:
    """Score every separated segment against the ground-truth stems over the same time span.

    `separated_segments` is (start_s, end_s, streams) as recorded by the pipeline;
    `reference_segments` supplies which speakers are genuinely active in that span, so a
    segment is scored only against the speakers actually there rather than against every
    stem in the recording. Duration-weighted, since a 2-second overlap and a 0.2-second one
    are not equally informative. Returns None when nothing was separable/scoreable."""
    if not source_tracks or not separated_segments:
        return None

    weighted_si, weighted_sdr, total_weight, scored = 0.0, 0.0, 0.0, 0
    for start, end, streams in separated_segments:
        if end <= start or not streams:
            continue
        active = sorted({
            s.speaker for s in reference_segments
            if s.speaker in source_tracks and s.start < end and start < s.end
        })
        if len(active) < 2:
            continue  # nothing to separate here; scoring it would measure the reference, not the separator
        i0, i1 = int(start * sample_rate), int(end * sample_rate)
        refs = [source_tracks[spk][i0:i1] for spk in active]
        seg_si, seg_sdr = best_permutation_si_sdr(refs, [np.asarray(s, dtype=np.float64).ravel() for s in streams])
        if not np.isfinite(seg_si):
            continue
        weight = end - start
        weighted_si += seg_si * weight
        weighted_sdr += seg_sdr * weight
        total_weight += weight
        scored += 1

    if scored == 0 or total_weight == 0:
        return None
    return SeparationQuality(
        si_sdr=weighted_si / total_weight,
        sdr=weighted_sdr / total_weight,
        num_scored_segments=scored,
    )


def _align(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate to the common length and mean-centre, as SI-SDR assumes zero-mean signals.
    A separator's output can be a few samples longer or shorter than the slice it was given
    (framing/padding), which is not an error worth charging."""
    reference = np.asarray(reference, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    if n == 0:
        return reference, estimate
    return reference - reference.mean(), estimate - estimate.mean()
