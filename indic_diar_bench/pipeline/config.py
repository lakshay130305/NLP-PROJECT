"""Pipeline configuration: which backend implements each stage, and which optional
modules (OSD / separation / attribution / routing) are switched on -- this is what
turns the same orchestrator into B1..B5 or the full Proposed system (section 11.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pipeline.router import RoutingMode


class Backend(str, Enum):
    """Selects dummy (dependency-free, deterministic) vs pretrained (real model, needs
    network + weights) implementations for every ML-backed stage. Dummy backends exist so
    the full pipeline can be exercised end-to-end offline -- see README for how/when to
    flip this to PRETRAINED once model downloads and the dataset are available."""

    DUMMY = "dummy"
    PRETRAINED = "pretrained"


@dataclass
class PipelineConfig:
    name: str
    backend: Backend = Backend.DUMMY

    # section 11.3 feature toggles
    use_osd: bool = False
    use_separation: bool = False
    use_attribution: bool = False
    routing_mode: RoutingMode | None = None  # None => no selective routing (always separate on any detected overlap)

    # hyperparameters
    osd_tau: float = 0.5
    osd_min_duration: float = 0.1
    num_speakers: int | None = None
    clustering_distance_threshold: float = 0.7
    max_overlap_speakers: int = 2
    unknown_speaker_threshold: float = 0.25
    asr_language: str | None = None

    asr_model_size: str = "tiny"  # only used when backend == PRETRAINED
    hf_token: str | None = None

    extra: dict = field(default_factory=dict)
