from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()

_MEDIA_TYPES = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".opus": "audio/opus", ".webm": "audio/webm", ".aiff": "audio/aiff", ".aif": "audio/aiff",
}


@router.get("/jobs/{job_id}/audio")
def get_job_audio(job_id: str, request: Request):
    job_store = request.app.state.job_store
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.audio_path or not job.audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on server")

    media_type = _MEDIA_TYPES.get(job.audio_path.suffix.lower(), "application/octet-stream")
    return FileResponse(job.audio_path, media_type=media_type, filename=job.filename)
