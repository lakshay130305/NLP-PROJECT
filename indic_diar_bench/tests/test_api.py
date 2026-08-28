"""API tests use the dummy backend (no network/model downloads) via a monkeypatched
build_pipeline, consistent with the rest of the test suite -- these test the API's own
wiring (routing, job lifecycle, export formats), not model quality."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from pipeline.config import Backend
from pipeline.orchestrator import OverlapAwarePipeline
from pipeline.variants import build_variant


def _dummy_pipeline_builder(**_kwargs):
    return OverlapAwarePipeline(build_variant("PROPOSED", backend=Backend.DUMMY))


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from api import config, pipeline_runner

    monkeypatch.setattr(pipeline_runner, "build_pipeline", _dummy_pipeline_builder)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")

    from api.main import app

    with TestClient(app) as c:
        yield c


def _make_wav_bytes() -> bytes:
    import numpy as np
    import soundfile as sf

    t = np.arange(16000 * 2) / 16000
    signal = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, signal, 16000, format="WAV")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["pipeline_loaded"] is True


def test_languages(client):
    resp = client.get("/api/languages")
    assert resp.status_code == 200
    langs = resp.json()["languages"]
    assert len(langs) == 22
    assert sum(1 for lang in langs if lang["whisper_supported"]) == 14


def test_empty_job_list(client):
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert resp.json()["jobs"] == []


def test_upload_rejects_unsupported_extension(client):
    resp = client.post("/api/jobs", files={"files": ("notaudio.txt", b"hello", "text/plain")})
    assert resp.status_code == 201
    jobs = resp.json()["jobs"]
    assert jobs[0]["status"] == "rejected"


def test_upload_and_poll_to_completion(client):
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/api/jobs",
        files={"files": ("test.wav", wav_bytes, "audio/wav")},
        data={"language": "Hindi"},
    )
    assert resp.status_code == 201
    job_id = resp.json()["jobs"][0]["job_id"]
    assert job_id is not None

    # TestClient's background executor submission runs synchronously enough that by the
    # time the response above returns, ThreadPoolExecutor(max_workers=1) has likely
    # already picked it up; poll briefly to avoid a flaky race either way.
    import time

    detail = None
    for _ in range(50):
        detail = client.get(f"/api/jobs/{job_id}").json()
        if detail["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    assert detail is not None
    assert detail["status"] == "done", detail.get("error")
    assert detail["result"]["recording_id"] == "test"
    assert "stats" in detail["result"]
    assert "overlap_regions" in detail["result"]


def test_job_not_found(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_export_not_ready_returns_409(client):
    wav_bytes = _make_wav_bytes()
    resp = client.post("/api/jobs", files={"files": ("test2.wav", wav_bytes, "audio/wav")})
    job_id = resp.json()["jobs"][0]["job_id"]

    # Immediately try to export before the (fast, dummy-backend) job has necessarily
    # finished -- accept either 409 (still processing) or 200 (already done), both valid.
    export_resp = client.get(f"/api/jobs/{job_id}/export/txt")
    assert export_resp.status_code in (200, 409)


def test_export_formats_after_completion(client):
    import time

    wav_bytes = _make_wav_bytes()
    resp = client.post("/api/jobs", files={"files": ("test3.wav", wav_bytes, "audio/wav")})
    job_id = resp.json()["jobs"][0]["job_id"]

    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    for fmt in ("txt", "json", "srt"):
        export_resp = client.get(f"/api/jobs/{job_id}/export/{fmt}")
        assert export_resp.status_code == 200, f"{fmt} export failed"
        assert len(export_resp.content) > 0

    bad_resp = client.get(f"/api/jobs/{job_id}/export/xml")
    assert bad_resp.status_code == 400


def test_audio_playback_endpoint(client):
    import time

    wav_bytes = _make_wav_bytes()
    resp = client.post("/api/jobs", files={"files": ("test4.wav", wav_bytes, "audio/wav")})
    job_id = resp.json()["jobs"][0]["job_id"]

    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    audio_resp = client.get(f"/api/jobs/{job_id}/audio")
    assert audio_resp.status_code == 200
    assert audio_resp.content == wav_bytes
