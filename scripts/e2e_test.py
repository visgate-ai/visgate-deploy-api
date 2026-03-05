#!/usr/bin/env python3
"""Full end-to-end live test against the production API.

What is verified:
  ─ Deployment lifecycle (POST → poll → ready → DELETE)
  ─ GPU selection: was the allocated GPU optimal for the model's VRAM?
  ─ Cache mode: which path was taken (off / shared R2 / private S3)?
  ─ Inference: does the RunPod endpoint return a valid image?
  ─ Cleanup: is the deployment actually deleted?

Usage:
    # Keys are auto-loaded from .env in repo root
    python scripts/e2e_test.py

    # Or override:
    RUNPOD=rpa_xxx python scripts/e2e_test.py --model black-forest-labs/FLUX.1-schnell --cache shared

Options:
    --model       HF model ID        (default: stabilityai/sdxl-turbo)
    --prompt      Inference prompt   (default: "a red apple on a white table")
    --steps       Inference steps    (default: 4)
    --cache       off | shared | private  (default: shared)
    --no-infer    Skip inference step (save GPU cost; still verifies lifecycle)
    --no-delete   Keep deployment alive after test
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
from typing import Any

import httpx

# ── Load .env from repo root ──────────────────────────────────────────────────
_ENV_PATH = pathlib.Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = os.environ.get("API_BASE", "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app")
RUNPOD_KEY = os.environ.get("RUNPOD") or os.environ.get("VISGATE_DEPLOY_API_RUNPOD_ACCESS_TOKEN_USER", "")
HF_TOKEN   = os.environ.get("HF_TOKEN") or os.environ.get("VISGATE_DEPLOY_API_HF_ACCESS_TOKEN_USER", "")

POLL_INTERVAL = 8    # seconds between status polls
POLL_TIMEOUT  = 600  # max seconds to wait for ready (10 min)
INFER_TIMEOUT = 120  # RunPod runsync timeout seconds

# ── Colours ───────────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  D = "\033[2m"; X = "\033[0m"

TERMINAL_STATUSES = {"ready", "failed", "webhook_failed", "deleted"}

pass_count = fail_count = warn_count = 0


def ok(msg: str):
    global pass_count; pass_count += 1
    print(f"  {G}✅{X}  {msg}")

def fail(msg: str, fatal: bool = False):
    global fail_count; fail_count += 1
    print(f"  {R}❌{X}  {msg}")
    if fatal: sys.exit(1)

def info(msg: str): print(f"  {C}ℹ{X}   {msg}")

def warn(msg: str):
    global warn_count; warn_count += 1
    print(f"  {Y}⚠{X}   {msg}")

def check(label: str, cond: bool, got: Any = "", expected: Any = ""):
    if cond:
        ok(label)
    else:
        detail = f"  got={got!r}  expected={expected!r}" if got != "" else ""
        fail(f"{label}{detail}")

def stage(n: int, total: int, title: str):
    print(f"\n{B}{'─'*60}\n  {n}/{total}  {title}\n{'─'*60}{X}")


def api(method: str, path: str, *, auth: bool = True, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if auth:
        headers["Authorization"] = f"Bearer {RUNPOD_KEY}"
    return httpx.request(method, BASE + path, headers=headers, timeout=30, **kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="stabilityai/sdxl-turbo")
    p.add_argument("--prompt",    default="a red apple on a white table")
    p.add_argument("--steps",     type=int, default=4)
    p.add_argument("--cache",     default="shared", choices=["off", "shared", "private"])
    p.add_argument("--no-infer",  action="store_true")
    p.add_argument("--no-delete", action="store_true")
    args = p.parse_args()

    if not RUNPOD_KEY:
        print(f"{R}ERROR:{X} No RunPod key found.")
        print("  Set RUNPOD env var  OR  add VISGATE_DEPLOY_API_RUNPOD_ACCESS_TOKEN_USER to .env")
        sys.exit(1)

    TOTAL = 7 - (1 if args.no_infer else 0)

    print(f"\n{B}{'='*60}")
    print("  VISGATE DEPLOY API — FULL E2E TEST")
    print(f"  {BASE}")
    print(f"  Model  : {args.model}")
    print(f"  Cache  : {args.cache}")
    print(f"  Prompt : {args.prompt}")
    print(f"  Infer  : {'yes' if not args.no_infer else 'skipped'}")
    print(f"{'='*60}{X}")

    timings: dict[str, float] = {}
    dep_id:       str | None  = None
    endpoint_url: str | None  = None
    overall_start = time.perf_counter()

    # ── 1. Catalogue check ───────────────────────────────────────────────────
    stage(1, TOTAL, "Catalogue — GET /v1/models")
    r = api("GET", "/v1/models", auth=False)
    check("GET /v1/models returns 200", r.status_code == 200, r.status_code, 200)
    models_data   = r.json() if r.status_code == 200 else {}
    models        = {m["model_id"]: m for m in models_data.get("models", [])}
    cache_enabled = models_data.get("cache_enabled", False)
    model_entry   = models.get(args.model)
    registry_vram = model_entry["gpu_memory_gb"] if model_entry else None
    cached_in_r2  = model_entry["cached"]        if model_entry else False

    if model_entry:
        ok(f"Model in registry: vram={registry_vram} GB  r2_cached={cached_in_r2}")
    else:
        warn(f"Model not in registry — will fall back to safetensors byte-estimation")

    check("Platform cache enabled", cache_enabled)

    # ── 2. Health ────────────────────────────────────────────────────────────
    stage(2, TOTAL, "Health — GET /health")
    t = time.perf_counter()
    r = api("GET", "/health", auth=False)
    timings["health"] = time.perf_counter() - t
    check(f"API healthy ({timings['health']*1000:.0f}ms)", r.status_code == 200, r.status_code, 200)
    if r.status_code != 200:
        fail("Cannot continue without a healthy API", fatal=True)

    # ── 3. Create deployment ─────────────────────────────────────────────────
    stage(3, TOTAL, "POST /v1/deployments")
    payload: dict[str, Any] = {
        "hf_model_id": args.model,
        "user_webhook_url": "https://httpbin.org/post",
        "cache_scope": args.cache,
        "task": "text2img",
    }
    if HF_TOKEN:
        payload["hf_token"] = HF_TOKEN
        info("hf_token injected from .env")

    info(f"Payload: {json.dumps({k: v for k, v in payload.items() if k != 'hf_token'})}")

    t = time.perf_counter()
    r = api("POST", "/v1/deployments", json=payload)
    timings["post"] = time.perf_counter() - t

    if r.status_code not in (200, 201, 202):
        fail(f"POST failed: HTTP {r.status_code}  {r.text[:300]}", fatal=True)

    body   = r.json()
    dep_id = body.get("deployment_id")
    status = body.get("status")
    eta    = body.get("estimated_ready_seconds", "?")
    path   = body.get("path", "?")

    ok(f"Deployment accepted in {timings['post']*1000:.0f}ms")
    info(f"deployment_id = {dep_id}")
    info(f"status        = {status}  path={path}  eta=~{eta}s")
    check("deployment_id present",    bool(dep_id))
    check("status is accepted/ready", status in ("accepted_cold", "warm_ready", "ready"), status)

    if path == "warm":
        ok("WARM PATH — reusing existing endpoint (no cold start wait)")
    else:
        info("COLD PATH — new endpoint will be created")

    # ── 4. Poll until ready ──────────────────────────────────────────────────
    stage(4, TOTAL, "Poll GET /v1/deployments/{id} until ready")
    poll_start  = time.perf_counter()
    last_status = status
    poll_count  = 0
    data: dict  = {}

    while True:
        elapsed_poll = time.perf_counter() - poll_start
        if elapsed_poll > POLL_TIMEOUT:
            fail(f"Timed out after {POLL_TIMEOUT}s waiting for ready", fatal=True)

        time.sleep(POLL_INTERVAL)
        poll_count += 1
        r = api("GET", f"/v1/deployments/{dep_id}")
        if r.status_code != 200:
            warn(f"Poll {poll_count}: HTTP {r.status_code} — retrying"); continue

        data   = r.json()
        status = data.get("status", "unknown")

        if status != last_status:
            print(f"  {D}[{elapsed_poll:.0f}s]{X}  {Y}{last_status}{X} → {B}{status}{X}")
            last_status = status
        else:
            print(f"  {D}[{elapsed_poll:.0f}s]  still {status}...{X}")

        if status in TERMINAL_STATUSES:
            break

    timings["poll"] = time.perf_counter() - poll_start

    if status != "ready":
        error = data.get("error", "")
        fail(f"Deployment ended with status={status}  error={error}")
        if not args.no_delete and dep_id:
            api("DELETE", f"/v1/deployments/{dep_id}")
            info(f"Cleaned up {dep_id}")
        sys.exit(1)

    ok(f"Ready in {timings['poll']:.1f}s  (polls: {poll_count})")
    endpoint_url  = data.get("endpoint_url", "")
    gpu_allocated = data.get("gpu_allocated", "")
    model_vram_gb = data.get("model_vram_gb")
    logs          = data.get("logs", [])

    # ── GPU selection audit ──────────────────────────────────────────────
    print(f"\n  {B}GPU Selection Audit:{X}")
    info(f"model_vram_gb (API estimate) = {model_vram_gb} GB")
    info(f"gpu_allocated                = {gpu_allocated}")
    if registry_vram:
        info(f"registry_vram (expected)     = {registry_vram} GB")
        check("GPU was allocated",
              bool(gpu_allocated), gpu_allocated)
        if model_vram_gb and registry_vram:
            diff = abs(model_vram_gb - registry_vram)
            check(f"Estimated VRAM matches registry (diff={diff} GB, tolerance=4)",
                  diff <= 4, model_vram_gb, registry_vram)
    else:
        check("GPU allocated (byte-estimation path)", bool(gpu_allocated), gpu_allocated)
    check("endpoint_url present", bool(endpoint_url), endpoint_url)

    # ── Cache mode audit ─────────────────────────────────────────────────
    print(f"\n  {B}Cache Mode Audit:{X}")
    log_messages = " ".join(e.get("message", "") for e in logs).lower()
    info(f"Requested cache_scope = {args.cache}")
    if args.cache == "shared":
        check("Platform cache is enabled in API", cache_enabled)
        if cached_in_r2:
            ok("Model weights cached in R2 — cold start is fast (~2-5s)")
        else:
            warn("Model not yet cached in R2 — this run downloads from HF Hub at full speed")
            info("Subsequent cold starts will use R2 cache once the worker syncs")
    elif args.cache == "private":
        info("Private cache: user S3 keys should have been injected into worker env")
    else:
        info("cache_scope=off — no S3/R2 keys injected into worker")

    # ── Log trail ────────────────────────────────────────────────────────
    print(f"\n  {B}Deployment Log Trail:{X}")
    for e in logs:
        lvl = e.get("level", "INFO")
        msg = e.get("message", "")
        ts  = e.get("timestamp", "")[:19].replace("T", " ")
        clr = G if lvl == "INFO" else (Y if lvl == "WARNING" else R)
        print(f"  {D}{ts}{X}  {clr}{lvl:<8}{X}  {msg}")

    if not endpoint_url:
        fail("No endpoint_url — cannot run inference", fatal=not args.no_infer)

    # ── 5. Inference ─────────────────────────────────────────────────────────
    if not args.no_infer:
        stage(5, TOTAL, "RunPod inference")
        infer_payload = {
            "input": {
                "prompt": args.prompt,
                "num_inference_steps": args.steps,
                "guidance_scale": 0.0,
                "width": 512,
                "height": 512,
                "seed": 42,
            }
        }
        # endpoint_url ends with /run — use /runsync for sync response
        sync_url = endpoint_url.rstrip("/")
        if   sync_url.endswith("/run"):    sync_url = sync_url[:-4] + "/runsync"
        elif not sync_url.endswith("/runsync"): sync_url += "sync"
        info(f"POST {sync_url}")
        info(f"steps={args.steps}  prompt=\"{args.prompt}\"")

        t = time.perf_counter()
        try:
            r = httpx.post(
                sync_url,
                json=infer_payload,
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=INFER_TIMEOUT,
            )
            timings["inference"] = time.perf_counter() - t
        except httpx.ReadTimeout:
            fail(f"Inference timed out after {INFER_TIMEOUT}s")
            timings["inference"] = float(INFER_TIMEOUT)
        else:
            check(f"Inference HTTP 200 ({timings['inference']*1000:.0f}ms)",
                  r.status_code == 200, r.status_code, 200)
            if r.status_code == 200:
                infer_body   = r.json()
                output       = infer_body.get("output", {})
                image_b64    = output.get("image_base64", "")
                infer_status = infer_body.get("status", "?")
                info(f"RunPod job status: {infer_status}")
                check("image_base64 present in output", bool(image_b64))
                if image_b64:
                    img_path = "/tmp/e2e_output.png"
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(image_b64))
                    size_kb = os.path.getsize(img_path) // 1024
                    ok(f"Image saved: {size_kb} KB → {img_path}")
                else:
                    warn(f"Output keys: {list(output.keys())}")
                    warn(f"Full response: {json.dumps(infer_body)[:400]}")

    # ── 6. DELETE ────────────────────────────────────────────────────────────
    stage(6, TOTAL, "DELETE /v1/deployments/{id}")
    if args.no_delete:
        warn("--no-delete flag set, skipping teardown")
    else:
        t = time.perf_counter()
        r = api("DELETE", f"/v1/deployments/{dep_id}")
        timings["delete"] = time.perf_counter() - t
        check(f"DELETE returns 204 ({timings['delete']*1000:.0f}ms)", r.status_code == 204, r.status_code, 204)

    # ── 7. Verify deleted ────────────────────────────────────────────────────
    stage(7, TOTAL, "Verify deletion")
    if args.no_delete:
        warn("Skipped (--no-delete)")
    else:
        time.sleep(2)
        r = api("GET", f"/v1/deployments/{dep_id}")
        final_status = r.json().get("status", "") if r.status_code == 200 else ""
        check(
            "Deployment shows status=deleted",
            r.status_code == 404 or final_status == "deleted",
            f"HTTP {r.status_code}  status={final_status}",
            "404 or deleted",
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.perf_counter() - overall_start
    print(f"\n{B}{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}{X}")
    for label, t in [
        ("Health check",     timings.get("health")),
        ("POST deployment",  timings.get("post")),
        ("Poll until ready", timings.get("poll")),
        ("Inference",        timings.get("inference")),
        ("DELETE",           timings.get("delete")),
    ]:
        if t is not None:
            bar = int(t / total * 30) * "█"
            print(f"  {label:<22} {t:>6.1f}s  {D}{bar}{X}")
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':<22} {total:>6.1f}s")
    print()
    print(f"  {G}✅ Passed{X}  {pass_count}")
    if warn_count: print(f"  {Y}⚠  Warned{X}  {warn_count}")
    if fail_count: print(f"  {R}❌ Failed{X}  {fail_count}")
    else:          print(f"  {G}{B}ALL CHECKS PASSED{X}")
    print()
    print(f"  deployment_id = {dep_id}")
    print(f"  gpu_allocated = {gpu_allocated}")
    print(f"  model_vram_gb = {model_vram_gb} GB  (registry={registry_vram} GB)")
    print(f"  cache_scope   = {args.cache}  (platform_cache={cache_enabled}  in_r2={cached_in_r2})")
    print(f"{B}{'='*60}{X}\n")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
