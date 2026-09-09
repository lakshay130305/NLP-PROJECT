"""SI-SDR / SDR separation-quality metrics."""

import numpy as np

from eval.separation_quality import (
    best_permutation_si_sdr,
    score_separated_segments,
    sdr,
    si_sdr,
)
from schemas.types import SpeechSegment


def _tone(freq, n=16000, sr=16000):
    return np.sin(2 * np.pi * freq * np.arange(n) / sr)


def test_perfect_estimate_scores_very_high():
    s = _tone(220.0)
    assert si_sdr(s, s.copy()) > 60


def test_si_sdr_is_scale_invariant_but_sdr_is_not():
    s = _tone(220.0)
    assert si_sdr(s, s * 0.25) > 60
    assert sdr(s, s * 0.25) < 20


def test_unseparated_mixture_scores_about_zero_db():
    """Handing back the mixture instead of separating it (what NullSeparator does) should
    score ~0 dB against an equal-energy source, not a good number."""
    a, b = _tone(220.0), _tone(440.0)
    assert abs(si_sdr(a, a + b)) < 1.0


def test_permutation_invariance_picks_the_best_assignment():
    a, b = _tone(220.0), _tone(440.0)
    swapped, _ = best_permutation_si_sdr([a, b], [b.copy(), a.copy()])
    matched, _ = best_permutation_si_sdr([a, b], [a.copy(), b.copy()])
    assert swapped > 60 and matched > 60


def test_stream_count_mismatch_scores_what_was_produced():
    a, b = _tone(220.0), _tone(440.0)
    value, _ = best_permutation_si_sdr([a, b], [a.copy()])
    assert value > 60


def test_scoring_needs_stems_and_separated_segments():
    assert score_separated_segments({}, 16000, [(0.0, 1.0, [_tone(220.0)])], []) is None
    assert score_separated_segments({"A": _tone(220.0)}, 16000, [], []) is None


def test_single_speaker_spans_are_not_scored():
    """A span with only one active reference speaker measures the reference, not the
    separator, so it must be skipped rather than inflating the average."""
    tracks = {"A": _tone(220.0), "B": _tone(440.0)}
    ref = [SpeechSegment(0.0, 1.0, "A")]  # B never speaks in this span
    assert score_separated_segments(tracks, 16000, [(0.0, 1.0, [tracks["A"].copy()])], ref) is None


def test_perfect_separation_of_a_real_overlap_span():
    tracks = {"A": _tone(220.0), "B": _tone(440.0)}
    ref = [SpeechSegment(0.0, 1.0, "A"), SpeechSegment(0.0, 1.0, "B")]
    streams = [tracks["A"][:16000].copy(), tracks["B"][:16000].copy()]
    quality = score_separated_segments(tracks, 16000, [(0.0, 1.0, streams)], ref)
    assert quality is not None
    assert quality.num_scored_segments == 1
    assert quality.si_sdr > 60
