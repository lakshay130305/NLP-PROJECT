"""System variants, section 11.3.

| Variant  | Description                                    | osd | separation | attribution | routing |
|----------|-------------------------------------------------|-----|------------|--------------|---------|
| B1       | Standard diarization + ASR                       | no  | no         | no           | -       |
| B2       | B1 + overlap detection                           | yes | no         | no           | -       |
| B3       | OSD + multi-speaker labeling                     | yes | no         | yes          | -       |
| B4       | OSD + speech separation                          | yes | yes        | no           | hard    |
| B5       | OSD + separation + multi-speaker attribution     | yes | yes        | yes          | hard    |
| Proposed | Full selective-routing overlap-aware framework   | yes | yes        | yes          | soft    |

B2's OSD detections are surfaced as metrics/metadata only (section 12.5) and do not change
the diarization/ASR output relative to B1 -- it isolates "does detecting overlap alone help"
from "does acting on that detection help" (RQ2). B4 separates+transcribes overlap regions but
assigns the resulting streams by the same clustering as the non-overlap branch (no dedicated
prototype-based attribution), isolating the separation contribution (RQ3) from the attribution
contribution (RQ6). B5 is the Proposed framework with hard-only routing (every detected overlap
region is separated, no soft blending / uncertainty handling) -- the hard-vs-soft routing ablation
in section 17.4 is exactly B5 vs Proposed.
"""

from __future__ import annotations

from pipeline.config import Backend, PipelineConfig
from pipeline.router import RoutingMode


def build_variant(name: str, backend: Backend = Backend.DUMMY, **overrides) -> PipelineConfig:
    name = name.upper()
    base = _VARIANT_DEFAULTS.get(name)
    if base is None:
        raise ValueError(f"Unknown variant '{name}'. Known variants: {sorted(_VARIANT_DEFAULTS)}")

    cfg = PipelineConfig(name=name, backend=backend, **{**base, **overrides})
    return cfg


_VARIANT_DEFAULTS: dict[str, dict] = {
    "B1": {"use_osd": False, "use_separation": False, "use_attribution": False, "routing_mode": None},
    "B2": {"use_osd": True, "use_separation": False, "use_attribution": False, "routing_mode": None},
    "B3": {"use_osd": True, "use_separation": False, "use_attribution": True, "routing_mode": None},
    "B4": {"use_osd": True, "use_separation": True, "use_attribution": False, "routing_mode": RoutingMode.HARD},
    "B5": {"use_osd": True, "use_separation": True, "use_attribution": True, "routing_mode": RoutingMode.HARD},
    "PROPOSED": {"use_osd": True, "use_separation": True, "use_attribution": True, "routing_mode": RoutingMode.SOFT},
}

ALL_VARIANTS = tuple(_VARIANT_DEFAULTS.keys())
