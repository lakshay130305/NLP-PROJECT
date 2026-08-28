"""In-memory job store. A dict guarded by a lock is correct here: single-user local demo,
no persistence requirement, and job history is expected to reset with the server process
(documented, not a bug) -- see the plan's "What NOT to build" section."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    filename: str
    audio_path: Path
    language_hint: str | None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "language_hint": self.language_hint,
        }

    def to_detail_dict(self) -> dict[str, Any]:
        return {
            **self.to_summary_dict(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "result": self.result,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, audio_path: Path, language_hint: str | None) -> Job:
        job = Job(job_id=str(uuid.uuid4()), filename=filename, audio_path=audio_path, language_hint=language_hint)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            if status == JobStatus.PROCESSING:
                job.started_at = datetime.now(UTC)

    def set_done(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.DONE
            job.result = result
            job.finished_at = datetime.now(UTC)

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.error = error
            job.finished_at = datetime.now(UTC)
