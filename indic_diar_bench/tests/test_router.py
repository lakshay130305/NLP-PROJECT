import numpy as np

from pipeline.router import RoutingMode, SelectiveRouter, routed_fraction
from schemas.types import SpeechSegment


def _flat_probs(times, value):
    return np.full_like(times, value, dtype=np.float64)


def test_hard_routing_sends_high_overlap_segment_to_overlap_branch():
    times = np.arange(0, 2, 0.02)
    probs = np.where((times >= 0.5) & (times < 1.5), 0.9, 0.1)
    router = SelectiveRouter(tau=0.5, mode=RoutingMode.HARD, min_overlap_duration=0.05)
    segments = [SpeechSegment(0, 0.4), SpeechSegment(0.5, 1.5), SpeechSegment(1.6, 2.0)]
    decisions = router.route_segments(segments, times, probs)

    routes = {d.segment.start: d.route for d in decisions}
    assert routes[0] == "single"
    assert routes[0.5] == "overlap"
    assert routes[1.6] == "single"


def test_routed_fraction():
    times = np.arange(0, 2, 0.02)
    probs = _flat_probs(times, 0.9)
    router = SelectiveRouter(tau=0.5, mode=RoutingMode.HARD)
    segments = [SpeechSegment(0, 1), SpeechSegment(1, 2)]
    decisions = router.route_segments(segments, times, probs)
    assert routed_fraction(decisions) == 1.0


def test_soft_routing_weight_reflects_probability():
    times = np.arange(0, 1, 0.02)
    probs = _flat_probs(times, 0.7)
    router = SelectiveRouter(tau=0.5, mode=RoutingMode.SOFT)
    segments = [SpeechSegment(0, 1)]
    decisions = router.route_segments(segments, times, probs)
    assert abs(decisions[0].overlap_weight - 0.7) < 1e-6
    assert decisions[0].route == "overlap"
