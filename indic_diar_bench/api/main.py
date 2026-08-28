"""FastAPI app entry point. Run from the repo root:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api.jobs import JobStore
from scripts._stdio import force_utf8_stdio

force_utf8_stdio()  # Windows' default console codepage crashes on non-Latin transcript text -- see README.


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    app.state.job_store = JobStore()
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    app.state.pipeline = None
    app.state.pipeline_load_error: str | None = None

    print(f"Loading PROPOSED pipeline (asr_model_size={config.ASR_MODEL_SIZE})... this can take a while.")
    try:
        from api.pipeline_runner import build_pipeline

        app.state.pipeline = build_pipeline(
            asr_model_size=config.ASR_MODEL_SIZE,
            hf_token=config.HF_TOKEN,
            max_overlap_speakers=config.MAX_OVERLAP_SPEAKERS,
        )
        print("Pipeline loaded.")
    except Exception as e:  # noqa: BLE001 -- a failed model load must not take the whole API down
        app.state.pipeline_load_error = f"{type(e).__name__}: {e}"
        print(f"WARNING: pipeline failed to load -- server will run but all jobs will fail: {app.state.pipeline_load_error}")
        traceback.print_exc()

    yield

    app.state.executor.shutdown(wait=False)


app = FastAPI(title="Indic DiarBench API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.ALLOWED_ORIGIN, "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import audio, export, jobs, languages

app.include_router(jobs.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(languages.router, prefix="/api")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": app.state.pipeline is not None,
        "pipeline_load_error": app.state.pipeline_load_error,
    }
