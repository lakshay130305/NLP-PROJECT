from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import config
from api.pipeline_runner import run_job
from scripts.demo import AUDIO_EXTENSIONS

router = APIRouter()


@router.post("/jobs", status_code=201)
async def create_jobs(request: Request, files: list[UploadFile], language: str | None = Form(default=None)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    job_store = request.app.state.job_store
    executor = request.app.state.executor
    pipeline = request.app.state.pipeline

    created = []
    for upload in files:
        filename = upload.filename or "audio"
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            created.append({"job_id": None, "filename": filename, "status": "rejected",
                             "error": f"Unsupported file type '{ext}'"})
            continue

        job = job_store.create(filename=filename, audio_path=Path(), language_hint=language)
        job_dir = config.UPLOAD_DIR / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dest_path = job_dir / filename
        with dest_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        job.audio_path = dest_path

        if pipeline is None:
            job_store.set_failed(job.job_id, "Pipeline is not ready yet (still loading models on server startup)")
        else:
            executor.submit(run_job, pipeline, job_store, job)

        created.append(job.to_summary_dict())

    return {"jobs": created}


@router.get("/jobs")
def list_jobs(request: Request):
    job_store = request.app.state.job_store
    return {"jobs": [j.to_summary_dict() for j in job_store.list_all()]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job_store = request.app.state.job_store
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_detail_dict()
