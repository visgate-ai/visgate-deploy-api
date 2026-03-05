#!/usr/bin/env python3
"""Quick endpoint smoke test — no deployment wait, just API surface validation.

Usage:
    python scripts/endpoint_report.py

Set RUNPOD env var to test authenticated endpoints with a real key.
"""
from __future__ import annotations

import os
import sys
import time
import json
import textwrap

import httpx

BASE = "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app"
RUNPOD_KEY = os.environ.get("RUNPOD", "smoke_test_dummy_key_123")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

results: list[dict] = []


def req(name: str, method: str, path: str, expected: int | list[int], **kwargs) -> dict:
    url = BASE + path
    start = time.perf_counter()
    try:
        r = httpx.request(method, url, timeout=15, **kwargs)
        ms = round((time.perf_counter() - start) * 1000)
        status = r.status_code
        try:
            body = r.json()
        except Exception:
            body = r.text[:200]
    except Exception as exc:
        ms = round((time.perf_counter() - start) * 1000)
        status = 0
        body = str(exc)

    expected_list = expected if isinstance(expected, list) else [expected]
    ok = status in expected_list
    icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
    results.append({"name": name, "method": method, "path": path,
                    "status": status, "expected": expected_list, "ms": ms, "ok": ok, "body": body})
    summary = json.dumps(body)[:100] if isinstance(body, dict) else str(body)[:100]
    print(f"  {icon}  {ms:>5}ms  HTTP {status}  {name}")
    print(f"         └─ {summary}")
    return {"status": status, "body": body, "ms": ms}


def separator(title: str):
    print(f"\n{BOLD}── {title} {'─' * max(0, 50 - len(title))}{RESET}")


# ─────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  VISGATE DEPLOY API — ENDPOINT REPORT{RESET}")
print(f"  {BASE}")
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print(f"{BOLD}{'='*60}{RESET}")

separator("Health & Observability")
req("GET /                  root info",          "GET",  "/",           200)
req("GET /health            liveness",           "GET",  "/health",     200)
req("GET /readiness         firestore check",    "GET",  "/readiness",  [200, 503])
req("GET /metrics           json metrics",       "GET",  "/metrics",    200)

separator("Model Catalog (public, no auth)")
r_models = req("GET /v1/models         full catalog",  "GET", "/v1/models", 200)
if isinstance(r_models["body"], dict):
    total = r_models["body"].get("total", "?")
    cached = sum(1 for m in r_models["body"].get("models", []) if m.get("cached"))
    print(f"         ℹ  {total} models in registry, {cached} cached in R2")

separator("Auth Boundary")
req("GET /v1/deployments    no auth → 401",      "GET", "/v1/deployments", 401)
req("GET /v1/deployments    with Bearer → 200",  "GET", "/v1/deployments",  200,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"})

separator("List Deployments — Query Params")
req("GET /v1/deployments?limit=5",               "GET", "/v1/deployments?limit=5", 200,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
req("GET /v1/deployments?deployment_status=ready", "GET", "/v1/deployments?deployment_status=ready", 200,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
req("GET /v1/deployments?limit=9999 (capped)",   "GET", "/v1/deployments?limit=9999", 200,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"})

separator("POST /v1/deployments — Validation")
req("missing hf_model_id → 422",  "POST", "/v1/deployments", 422,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"},
    content=json.dumps({"user_webhook_url": "https://example.com/wh"}).encode())

req("invalid task for sdxl-turbo → 400",  "POST", "/v1/deployments", 400,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"},
    content=json.dumps({
        "hf_model_id": "stabilityai/sdxl-turbo",
        "user_webhook_url": "https://example.com/wh",
        "task": "text2video",
    }).encode())

req("cache_scope=private missing s3_url → 400",  "POST", "/v1/deployments", 400,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"},
    content=json.dumps({
        "hf_model_id": "stabilityai/sdxl-turbo",
        "user_webhook_url": "https://example.com/wh",
        "cache_scope": "private",
    }).encode())

req("private S3 fields without cache_scope=private → 400",  "POST", "/v1/deployments", 400,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"},
    content=json.dumps({
        "hf_model_id": "stabilityai/sdxl-turbo",
        "user_webhook_url": "https://example.com/wh",
        "cache_scope": "off",
        "user_s3_url": "s3://mybucket/models",
    }).encode())

req("no auth on POST → 401",  "POST", "/v1/deployments", 401,
    headers={"Content-Type": "application/json"},
    content=json.dumps({
        "hf_model_id": "stabilityai/sdxl-turbo",
        "user_webhook_url": "https://example.com/wh",
    }).encode())

separator("GET /v1/deployments/{id}")
req("non-existent id → 404",  "GET", "/v1/deployments/dep_0000_notexist", 404,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
req("no auth → 401",          "GET", "/v1/deployments/dep_0000_notexist", 401)

separator("DELETE /v1/deployments/{id}")
req("non-existent → 404",     "DELETE", "/v1/deployments/dep_0000_notexist", 404,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"})

# ─────────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r["ok"])
failed = sum(1 for r in results if not r["ok"])
total  = len(results)
avg_ms = round(sum(r["ms"] for r in results) / total) if total else 0
max_r  = max(results, key=lambda r: r["ms"])
min_r  = min(results, key=lambda r: r["ms"])

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  SUMMARY{RESET}")
print(f"  Total tests : {total}")
print(f"  {GREEN}✅ Passed{RESET}   : {passed}")
if failed:
    print(f"  {RED}❌ Failed{RESET}   : {failed}")
    for r in results:
        if not r["ok"]:
            print(f"     • {r['name'].strip()}  got HTTP {r['status']}, expected {r['expected']}")
else:
    print(f"  ❌ Failed   : 0")
print(f"  Avg latency : {avg_ms}ms")
print(f"  Fastest     : {min_r['ms']}ms  ({min_r['name'].strip()})")
print(f"  Slowest     : {max_r['ms']}ms  ({max_r['name'].strip()})")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if failed == 0 else 1)
