from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from api.transcript_export import to_srt, to_txt

router = APIRouter()

_CONTENT_TYPES = {"txt": "text/plain", "json": "application/json", "srt": "application/x-subrip"}


@router.get("/jobs/{job_id}/export/{fmt}")
def export_job(job_id: str, fmt: str, request: Request):
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported export format '{fmt}', expected txt/json/srt")

    job_store = request.app.state.job_store
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Job is not done yet (status: {job.status.value})")

    if fmt == "txt":
        content = to_txt(job.result)
    elif fmt == "srt":
        content = to_srt(job.result)
    else:
        content = json.dumps(job.result, indent=2, ensure_ascii=False)

    recording_id = job.result.get("recording_id", job_id)
    return Response(
        content=content,
        media_type=_CONTENT_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{recording_id}.{fmt}"'},
    )
