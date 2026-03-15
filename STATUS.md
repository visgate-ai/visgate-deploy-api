# Vast.ai On-Demand Instance Migration — Status (2026-03-15)

## Summary
Vast Serverless → On-Demand Instances migration. Serverless was fundamentally incompatible
(requires PyWorker, custom Docker images never become "ready"). On-demand gives direct
`public_ipaddr:port` access to the worker container.

## Commits (newest first)
- `d7846ab` — fix: probe /health regardless of actual_status, resolve worker URL on webhook ready
- `8f26e4b` — fix: make E2E poll resilient to socket timeouts
- `1b50661` — fix: convert gpu_ram GB→MB and use spaces in GPU names for bundles API
- `073d640` — fix: use cloud.vast.ai and JSON dict query for bundles API
- `b61e841` — feat: switch Vast from Serverless to On-Demand Instances (major rewrite)

## Files Changed
- `deploy-api/src/services/vast.py` — Full rewrite (~550 lines). On-demand flow: search_offers → create_instance → get_instance → extract_worker_url → probe /health → submit_job direct
- `deploy-api/src/services/deployment.py` — Updated for vast-inst:// URLs, worker_url resolution on ready
- `deploy-api/tests/unit/test_vast_provider.py` — Full rewrite for on-demand tests (33 tests pass)
- `scripts/vast_e2e.py` — Socket timeout resilience in _poll loop

## What Works (Verified)
- ✅ Vast offer search: bundles API with JSON dict query format (`cloud.vast.ai/api/v0/bundles/`)
- ✅ GPU RAM filter: converted GB→MB (API uses MB internally)
- ✅ GPU name filter: uses spaces not underscores ("RTX 3090" not "RTX_3090")
- ✅ Instance creation: `PUT /api/v0/asks/{offer_id}/` → gets instance_id
- ✅ Model download via R2: 12.1GB in 118s on RTX A4000
- ✅ Model loading: SDXLPipeline loads and pipeline is ready
- ✅ Webhook fires: worker notifies deploy API → deployment marked "ready"
- ✅ E2E poll loop: resilient to individual socket timeouts
- ✅ Instance deletion: `DELETE /api/v0/instances/{id}/`

## Current Bug — Job Submission Fails
**Last E2E result** (image modality, instance 32901962):
- Deployment created → health checks return "unknown" for ~4 min → webhook marks ready at 251s
- **FAIL**: `submit_job` gets `endpoint_url = vast-inst://32901962` (virtual URL)
- Tries to resolve instance → gets `status: null` → "Instance has no public IP:port yet"

**Root cause**: The readiness monitor exits when webhook marks deployment "ready", but
`endpoint_url` in Firestore is still set to `vast-inst://32901962` (the virtual URL from creation).
The final fix in `d7846ab` was supposed to resolve worker URL on webhook ready, but the health
check loop still returns `health_status=unknown` for ALL 40+ checks — meaning `actual_status`
from Vast API is not "running" or the instance dict doesn't have port info.

**Possible issues to investigate next**:
1. Vast API returns instance with `actual_status != "running"` even when container is running
   (some instances show status="loading" or empty in API but Docker is actually up)
2. `public_ipaddr` / `ports` fields might not be populated in the instance dict at query time
3. The fix in d7846ab probes /health when worker_url exists regardless of actual_status —
   need to verify this fix was deployed when the last E2E ran
4. The `endpoint_url` stored in Firestore stays as `vast-inst://` — need to update it to
   the resolved `http://ip:port` URL when worker becomes available

**Next step**: Re-run image E2E to test if d7846ab fix resolves it. If health checks still
return "unknown", need to debug what `get_instance()` actually returns (log the full instance dict).

## Vast API Notes
- Base domain: `cloud.vast.ai` (NOT console.vast.ai — redirects strip auth headers)
- Bundles search: `GET /api/v0/bundles/?q=<url-encoded-json>&order=dph_total&limit=N`
- Query format: `{"field": {"op": value}}` — e.g. `{"gpu_ram": {"gte": 8192}, "gpu_name": {"eq": "RTX 3090"}}`
- GPU RAM values in MB (not GB)
- GPU names use SPACES: "RTX 3090", "RTX 4090", "H100 SXM", "RTX PRO 6000 S"
- Instance creation: `PUT /api/v0/asks/{offer_id}/`
- Instance details: `GET /api/v0/instances/{id}/`
- Instance deletion: `DELETE /api/v0/instances/{id}/`
- Some endpoints redirect `cloud.vast.ai → console.vast.ai` stripping auth (e.g. /users/current)

## Remaining E2E Tests
- ❌ Image (Vast): deployment ready but job submission fails (virtual URL not resolved)
- ⬜ Audio (Vast): not tested yet
- ⬜ Video (Vast): not tested yet
- ⬜ Image (RunPod): blocked — user contacting RunPod support about GPU access
- ⬜ Audio (RunPod): blocked
- ⬜ Video (RunPod): blocked

## Credits & Resources
- Vast balance: ~$9 (approximate, last checked before recent tests)
- RunPod: user needs to contact support about GPU provisioning issues
- Deploy API: `https://visgate-deploy-api-93820292919.europe-west3.run.app/deployapi`
- Inference images: `visgateai/inference-{image,audio,video}:latest`
- All unit tests pass: 160 total (33 Vast-specific)
