"""Local HTTP wrapper that exposes the worker through RunPod-like endpoints."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional, Union, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import torch

from app.runtime_common import post_json
import app.worker as worker

app = FastAPI(title="Visgate Local Worker", version="1.0.0")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.RLock()
_endpoint_meta: dict[str, Any] = {"endpoint_id": None, "profile": None, "name": None}


class LoadRequest(BaseModel):
    endpoint_id: str
    name: Optional[str] = None
    profile: str = "image"
    image: Optional[str] = None
    gpu_ids: Optional[Union[str, List[str]]] = None
    env: dict[str, str] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    endpoint_id: Optional[str] = None


def _clear_worker_state() -> None:
    with worker._state_lock:
        previous = worker._pipeline
        worker._pipeline = None
        worker._load_error = None
        worker._last_request_at = 0.0
        worker._failure_count = 0
        worker._task_kind = "text2img"
    if previous is not None:
        try:
            del previous
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _apply_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[str(key)] = str(value)
        setattr(worker, key, value)
    if "TASK" in env:
        os.environ["TASK"] = str(env["TASK"])
    if "HF_MODEL_ID" in env:
        worker.HF_MODEL_ID = str(env["HF_MODEL_ID"])
    if "HF_TOKEN" in env:
        worker.HF_TOKEN = str(env["HF_TOKEN"])
    if "VISGATE_WEBHOOK" in env:
        worker.VISGATE_WEBHOOK = str(env["VISGATE_WEBHOOK"])
    if "VISGATE_INTERNAL_SECRET" in env:
        worker.VISGATE_INTERNAL_SECRET = str(env["VISGATE_INTERNAL_SECRET"])
    if "VISGATE_DEPLOYMENT_ID" in env:
        worker.VISGATE_DEPLOYMENT_ID = str(env["VISGATE_DEPLOYMENT_ID"])
    if "MODEL_LOAD_WAIT_TIMEOUT_SECONDS" in env:
        worker.MODEL_LOAD_WAIT_TIMEOUT_SECONDS = int(str(env["MODEL_LOAD_WAIT_TIMEOUT_SECONDS"]))


def _job_record(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        record = _jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="job not found")
    return record


def _complete_job(job_id: str) -> None:
    record = _job_record(job_id)
    with _jobs_lock:
        if record.get("status") == "CANCELLED":
            return
        record["status"] = "IN_PROGRESS"
        record["started_at"] = time.time()

    payload = dict(record["payload"])
    result = worker.handler(payload)
    finished_at = time.time()
    webhook_url = record.get("webhook")

    with _jobs_lock:
        if isinstance(result, dict) and result.get("error"):
            record["status"] = "FAILED"
            record["error"] = result.get("error")
            record["output"] = result
        else:
            record["status"] = "COMPLETED"
            record["output"] = result
            record["error"] = None
        record["finished_at"] = finished_at
        record["execution_ms"] = max(int((finished_at - record["started_at"]) * 1000), 0)

    if webhook_url:
        try:
            post_json(
                webhook_url,
                {
                    "id": job_id,
                    "status": record["status"],
                    "output": record.get("output"),
                    "error": record.get("error"),
                    "executionTime": record.get("execution_ms"),
                },
            )
        except Exception:
            return


@app.get("/health")
async def health() -> dict[str, Any]:
    with worker._state_lock:
        is_ready = worker._pipeline is not None and worker._load_error is None
        load_error = worker._load_error
        task_kind = worker._task_kind
        model_id = worker.HF_MODEL_ID
    with _jobs_lock:
        running = sum(1 for item in _jobs.values() if item.get("status") == "IN_PROGRESS")
    status = "READY" if is_ready else ("FAILED" if load_error else "LOADING")
    return {
        "status": status,
        "model_id": model_id,
        "task": task_kind,
        "workers": {
            "ready": 1 if is_ready else 0,
            "idle": 1 if is_ready and running == 0 else 0,
            "running": running,
        },
        "error": load_error,
        "endpoint_id": _endpoint_meta.get("endpoint_id"),
        "profile": _endpoint_meta.get("profile"),
    }


@app.post("/load")
async def load_endpoint(body: LoadRequest) -> dict[str, Any]:
    _clear_worker_state()
    _apply_env(body.env)
    _endpoint_meta.update(
        {
            "endpoint_id": body.endpoint_id,
            "profile": body.profile,
            "name": body.name,
            "image": body.image,
            "gpu_ids": body.gpu_ids,
        }
    )
    threading.Thread(target=worker._load_model_background, daemon=True).start()
    return {"endpoint_id": body.endpoint_id, "status": "LOADING", "profile": body.profile}


@app.post("/reset")
async def reset_endpoint(body: ResetRequest) -> dict[str, Any]:
    _clear_worker_state()
    with _jobs_lock:
        _jobs.clear()
    _endpoint_meta.clear()
    return {"status": "RESET"}


@app.post("/run")
async def run_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid4().hex
    created_at = time.time()
    record = {
        "id": job_id,
        "status": "IN_QUEUE",
        "payload": payload,
        "created_at": created_at,
        "webhook": payload.get("webhook"),
        "started_at": None,
        "finished_at": None,
        "execution_ms": None,
        "output": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[job_id] = record
    threading.Thread(target=_complete_job, args=(job_id,), daemon=True).start()
    return {"id": job_id, "status": "IN_QUEUE"}


@app.get("/status/{job_id}")
async def status(job_id: str) -> dict[str, Any]:
    record = _job_record(job_id)
    delay_ms = None
    if record.get("started_at"):
        delay_ms = max(int((record["started_at"] - record["created_at"]) * 1000), 0)
    return {
        "id": job_id,
        "status": record.get("status", "UNKNOWN"),
        "output": record.get("output"),
        "error": record.get("error"),
        "delayTime": delay_ms,
        "executionTime": record.get("execution_ms"),
    }


@app.post("/cancel/{job_id}")
async def cancel(job_id: str) -> dict[str, Any]:
    record = _job_record(job_id)
    with _jobs_lock:
        if record.get("status") == "IN_QUEUE":
            record["status"] = "CANCELLED"
            record["finished_at"] = time.time()
    return {"id": job_id, "status": record.get("status", "CANCELLED")}


@app.post("/retry/{job_id}")
async def retry(job_id: str) -> dict[str, Any]:
    record = _job_record(job_id)
    retry_id = uuid4().hex
    retry_record = {
        **record,
        "id": retry_id,
        "status": "IN_QUEUE",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "execution_ms": None,
        "output": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[retry_id] = retry_record
    threading.Thread(target=_complete_job, args=(retry_id,), daemon=True).start()
    return {"id": retry_id, "status": "IN_QUEUE"}