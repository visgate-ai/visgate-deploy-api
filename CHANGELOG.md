# Changelog

## [0.5.0] - 2026-02-20 — Local dev mode + OSS release

### Added
- **Local dev without GCP:** `GCP_PROJECT_ID` is now optional (default `""`). When empty, in-memory storage activates automatically — no Firestore, no GCP account needed.
- `USE_MEMORY_REPO=true` env var to explicitly force in-memory mode even when `GCP_PROJECT_ID` is set.
- `effective_use_memory_repo` property on `Settings` for combined flag logic.
- `memory_repo.py` now has full API parity with `firestore_repo.py`: added `get_gpu_registry()`, `get_tier_mapping()`, `find_reusable_deployment()`.
- `_get_repo()` selector in `deployment.py` — switches between Firestore and in-memory at runtime without code changes.
- `.env.example` updated with annotated explanation for every env var.
- `CONTRIBUTING.md` + GitHub issue templates.

### Changed
- Docker image default: `uzunenes/inference:latest` → `visgateai/inference:latest` (all references updated across 5 files).
- GitHub Actions (`inference.yaml`, `deploy.yaml`) updated to `visgateai` Docker Hub namespace.
- `get_api_key()` in `memory_repo.py`: no longer hardcodes `"test-key"` — accepts any non-empty key when `ORCHESTRATOR_API_KEY` is not set, or validates against it when set.

### Removed
- `vastai.py` — was an uninstantiable stub (missing `list_endpoints` ABC implementation); removed entirely.



### Fixed
- **GPU selection bug:** `min_gpu_memory` field in old registry was silently ignored; only `vram_gb` was used for GPU selection. FLUX.1-dev was selecting AMPERE_24 (24 GB) which OOM'd at runtime (model needs 28 GB minimum).
- Partial S3 sync no longer leaves a broken cache; `.visgate_sync_complete` marker is written only after full successful sync.

### Added
- `model_specs_registry.py`: 18+ models (FLUX, SDXL, SD 1/2/3, PixArt, Kandinsky, IF, Wan, CogVideoX) with hand-tuned minimum GPU memory values.
- Dtype-aware VRAM estimation for unknown models: parses `safetensors.parameters {BF16: N, F32: M}` → byte calculation → × 1.35 headroom → GPU tier snap.
- `_BYTES_PER_PARAM` table covering all HF safetensors dtypes (BF16, F16, F32, F8_E4M3, INT8, etc.).
- `RUNPOD_WORKERS_MIN`, `RUNPOD_WORKERS_MAX`, `RUNPOD_IDLE_TIMEOUT_SECONDS`, `RUNPOD_SCALER_TYPE`, `RUNPOD_SCALER_VALUE` config fields.
- 6 new unit tests for byte-path VRAM estimation.

### Changed
- Default `workers_min`: 1 → **0** (no idle GPU cost by default).
- Default `idle_timeout`: 300s → **120s** (shorter idle window after burst).
- Default `scaler_value`: 2 → **1** (scale out after 1s queue wait).
- S5CMD concurrency: 10 → **50** workers + `--part-size 50mb` (faster S3 model sync).
- FLUX pipeline: `float16` → `bfloat16` (native to Ampere/Ada, avoids precision loss).
- Both FLUX and SDXL pipelines now enable `vae_slicing`, `vae_tiling`, and xformers (best-effort).
- Readiness probe interval: 8s → **5s** (faster fallback detection).
- `TRANSFORMERS_OFFLINE=1` + `DIFFUSERS_OFFLINE=1` set after confirmed S3 cache hit (prevents HF network calls on warm restarts).
- `HF_HUB_ENABLE_HF_TRANSFER=1` set for HF Hub download path (~5× faster).

## [0.3.0] - 2026-02-19 — Shared cache hardening + beginner docs

### Added
- Shared cache allowlist: `SHARED_CACHE_ALLOWED_MODELS` env var + `SHARED_CACHE_REJECT_UNLISTED=true`.
- `private_fields` validation: `user_s3_url` / `user_aws_*` fields rejected with 400 unless `cache_scope=private`.
- Two new unit tests for cache scope validation.

### Changed
- Root `README.md` rewritten for complete beginners (health → deploy → poll → delete flow).
- Added cost model explanation and S3 security note.

## [0.2.0] - 2026-02-19 — Cloud Run + S3 secrets wired

### Added
- GCP Secret Manager integration for shared S3 credentials (`sm://` prefix resolution).
- Structured prod smoke test script (`scripts/prod_api_smoke.py`).

### Changed
- Stateless auth: `Authorization: Bearer <RUNPOD_KEY>` or `X-Runpod-Api-Key` header; `user_hash = sha256(key)`.
- Deployment flow: `ready` state preserved even if user webhook delivery fails.
- Cloud Run traffic always routed to latest revision after deploy.

## [0.1.0] - 2026-02-15 — Initial release

### Added
- FastAPI orchestration service deployable to Cloud Run.
- Runpod serverless inference worker (FLUX + SDXL pipelines via diffusers).
- HF model validation + GPU selection with cost-index sorted candidates and capacity fallback loop.
- Firestore deployment state machine (`validating → selecting_gpu → creating_endpoint → loading_model → ready`).
- S3/R2/MinIO model cache via s5cmd.
- Webhook notification on deployment ready.
- GitHub Actions CI/CD for Cloud Run (`deploy.yaml`) and Docker Hub (`inference.yaml`).
- MIT license.
