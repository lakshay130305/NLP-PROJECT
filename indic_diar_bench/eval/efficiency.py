"""Computational metrics, section 12.7 / 20: RTF, latency, routing fraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EfficiencyStats:
    audio_duration: float = 0.0
    wall_clock_time: float = 0.0
    separator_calls: int = 0
    separator_audio_seconds: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio duration. <1 means faster than real time."""
        return self.wall_clock_time / self.audio_duration if self.audio_duration > 0 else 0.0

    @property
    def separator_routed_fraction(self) -> float:
        """Fraction of total audio duration that was actually sent to the separation branch --
        directly validates the selective-routing efficiency claim (section 20.3)."""
        return self.separator_audio_seconds / self.audio_duration if self.audio_duration > 0 else 0.0

    def add_stage_time(self, stage: str, seconds: float) -> None:
        self.stage_times[stage] = self.stage_times.get(stage, 0.0) + seconds
        self.wall_clock_time += seconds

    def record_separator_call(self, segment_duration: float) -> None:
        self.separator_calls += 1
        self.separator_audio_seconds += segment_duration
