"""Env-driven settings for the API server. No secrets/config values are hardcoded here --
everything has a sensible local-dev default and an environment-variable override."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 'tiny' by default: this session's testing found 'small' took ~14x real-time per
# recording on a CPU-only, memory-constrained machine -- 'tiny' is the practical default
# for a responsive local demo. Override via env var once running on stronger hardware.
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "tiny")

HF_TOKEN = os.environ.get("HF_TOKEN")

# Vite's default dev server port.
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "http://localhost:5173")

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", REPO_ROOT / "api" / "storage" / "uploads"))

# Matches pipeline/config.py::PipelineConfig.max_overlap_speakers default.
MAX_OVERLAP_SPEAKERS = int(os.environ.get("MAX_OVERLAP_SPEAKERS", "2"))
