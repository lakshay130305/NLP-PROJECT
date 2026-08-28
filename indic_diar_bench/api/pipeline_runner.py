"""Builds the pipeline once and runs jobs against it. Reuses scripts/demo.py's already-
validated transcript_to_dict/resolve_language rather than reimplementing that logic."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.jobs import Job, JobStatus, JobStore
from pipeline.audio_utils import load_audio_file
from pipeline.config import Backend
from pipeline.orchestrator import OverlapAwarePipeline
from pipeline.variants import build_variant
from scripts.demo import resolve_language, transcript_to_dict


def build_pipeline(asr_model_size: str, hf_token: str | None, max_overlap_speakers: int) -> OverlapAwarePipeline:
    cfg = build_variant(
        "PROPOSED",
        backend=Backend.PRETRAINED,
        asr_model_size=asr_model_size,
        hf_token=hf_token,
        max_overlap_speakers=max_overlap_speakers,
    )
    return OverlapAwarePipeline(cfg)


def _build_result_dict(transcript, stats, overlap_regions) -> dict:
    result = transcript_to_dict(transcript)
    speaker_count = len({u.speaker for u in transcript.utterances})
    result["stats"] = {
        "audio_duration": stats.audio_duration,
        "wall_clock_time": stats.wall_clock_time,
        "rtf": stats.rtf,
        "speaker_count": speaker_count,
    }
    result["overlap_regions"] = [
        {"start": r.start, "end": r.end, "speakers": list(r.speakers)} for r in overlap_regions
    ]
    return result


def run_job(pipeline: OverlapAwarePipeline, job_store: JobStore, job: Job) -> None:
    job_store.set_status(job.job_id, JobStatus.PROCESSING)
    try:
        audio, sample_rate = load_audio_file(job.audio_path)
        if len(audio) == 0:
            raise ValueError("Empty or unreadable audio file")

        language = resolve_language(job.language_hint)
        transcript, stats, overlap_regions = pipeline.run(
            audio, sample_rate, recording_id=job.audio_path.stem, language=language
        )

        result = _build_result_dict(transcript, stats, overlap_regions)
        job_store.set_done(job.job_id, result)
    except Exception as e:  # noqa: BLE001 -- must never let a job crash the worker thread
        job_store.set_failed(job.job_id, str(e))
