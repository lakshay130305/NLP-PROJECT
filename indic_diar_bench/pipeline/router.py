"""Selective overlap routing, section 6.

Splits VAD-active audio into non-overlap and overlap regions using the OSD
output, so speech separation (expensive) is only invoked where actually needed
(section 6.1, 6.5). Hard routing (6.2) makes a binary decision per region;
soft routing (6.3) instead produces a probability-weighted blend weight for
downstream combination. Routing confidence (6.4) flags decisions near tau.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from pipeline.osd import smooth_overlap_decisions, threshold_overlap
from schemas.types import OverlapRegion, SpeechSegment


class RoutingMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class RoutingDecision:
    segment: SpeechSegment
    route: str  # "overlap" or "single"
    overlap_weight: float  # for soft routing: h_t = (1-p_t) h_single + p_t h_overlap (section 6.3)
    is_uncertain: bool  # near tau (section 6.4)


class SelectiveRouter:
    def __init__(
        self,
        tau: float = 0.5,
        mode: RoutingMode = RoutingMode.HARD,
        min_overlap_duration: float = 0.1,
        uncertainty_band: float = 0.1,
    ):
        self.tau = tau
        self.mode = mode
        self.min_overlap_duration = min_overlap_duration
        self.uncertainty_band = uncertainty_band

    def route_segments(
        self, vad_segments: list[SpeechSegment], osd_times: np.ndarray, osd_probs: np.ndarray
    ) -> list[RoutingDecision]:
        overlap_regions = smooth_overlap_decisions(
            osd_times, threshold_overlap(osd_times, osd_probs, self.tau), self.min_overlap_duration
        )

        decisions: list[RoutingDecision] = []
        for seg in vad_segments:
            # Previously one route decision was made for the WHOLE vad segment ("overlap" if
            # ANY overlap region touched it anywhere), so a 10s segment with 0.3s of overlap
            # sent the entire 10s to the separation branch and got transcribed twice (once per
            # separated stream) -- against section 6.1's own stated goal of only invoking
            # separation "where actually needed". Split each segment at overlap-region
            # boundaries so only the actually-overlapping sub-interval is routed to
            # separation; sub-segments that don't touch any overlap region are left whole
            # (this is a no-op change for the common case of a segment that's fully inside or
            # fully outside every overlap region).
            for sub in self._split_at_boundaries(seg, overlap_regions):
                mean_p = self._mean_prob_in_range(osd_times, osd_probs, sub.start, sub.end)
                is_overlap = self._overlaps_any(sub, overlap_regions)
                is_uncertain = abs(mean_p - self.tau) < self.uncertainty_band

                # Branch selection (which work actually runs) is driven by `is_overlap` in
                # BOTH modes: a sub-segment's mean OSD probability dilutes to near-zero once
                # any short overlap sub-interval is averaged across a multi-second window
                # (confirmed on real audio: mean-probability routing sent 0% of a recording
                # to the separation branch despite the OSD correctly finding ~7.5% of frames
                # as overlapping). Hard vs. soft only differs in `overlap_weight`: hard uses a
                # binary 0/1 signal, soft carries the continuous probability for downstream
                # blending per h_t = (1-p_t) h_t^single + p_t h_t^overlap (section 6.3).
                route = "overlap" if is_overlap else "single"
                if self.mode == RoutingMode.HARD:
                    weight = 1.0 if is_overlap else 0.0
                else:
                    weight = float(np.clip(mean_p, 0.0, 1.0))

                decisions.append(RoutingDecision(segment=sub, route=route, overlap_weight=weight, is_uncertain=is_uncertain))
        return decisions

    @staticmethod
    def _split_at_boundaries(seg: SpeechSegment, regions: list[OverlapRegion]) -> list[SpeechSegment]:
        """Cut `seg` at every overlap-region start/end that falls strictly inside it, so each
        piece is either fully inside or fully outside every region. Pieces shorter than 20ms
        are dropped (routing/ASR noise, not a real sub-turn)."""
        cuts = {seg.start, seg.end}
        for r in regions:
            if seg.start < r.start < seg.end:
                cuts.add(r.start)
            if seg.start < r.end < seg.end:
                cuts.add(r.end)
        points = sorted(cuts)
        pieces = [
            SpeechSegment(a, b, seg.speaker)
            for a, b in zip(points, points[1:])
            if b - a >= 0.02
        ]
        return pieces or [seg]

    @staticmethod
    def _mean_prob_in_range(times: np.ndarray, probs: np.ndarray, start: float, end: float) -> float:
        if len(times) == 0:
            return 0.0
        mask = (times >= start) & (times < end)
        if not mask.any():
            return 0.0
        return float(np.mean(probs[mask]))

    @staticmethod
    def _overlaps_any(seg: SpeechSegment, regions: list[OverlapRegion]) -> bool:
        return any(seg.start < r.end and r.start < seg.end for r in regions)


def routed_fraction(decisions: list[RoutingDecision]) -> float:
    """Fraction of total segment duration sent to the overlap/separation branch --
    directly used for section 13.8 / 20.3 routing efficiency reporting."""
    total = sum(d.segment.duration for d in decisions)
    routed = sum(d.segment.duration for d in decisions if d.route == "overlap")
    return routed / total if total > 0 else 0.0
