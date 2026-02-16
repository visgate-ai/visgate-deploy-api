# Changelog

## [Unreleased]

### Added
- End-to-end reliability docs and hosted quick-start in root `README.md`.
- Build context optimization files for `deploy-api` (`.gcloudignore`, `.dockerignore`).
- Timed E2E and cleanup helper scripts for operational validation.

### Changed
- Stateless auth now uses request-time Runpod API key and user hash (no Firestore API key lookup).
- Deployment flow now keeps `ready` state even if user webhook delivery fails.
- CI/CD and deploy scripts now force traffic to latest Cloud Run revision after deploy.
- Runpod endpoint creation payload uses correct env schema (`[{key, value}]`).
- E2E scripts normalize Runpod endpoint URL handling for `/runsync` and `/status`.

## [Initial Release] - 2026-02-15

### Added
- **Core Orchestrator:** FastAPI-based gateway for Hugging Face and Runpod lifecycle management.
- **Inference Worker:** Dockerized Runpod worker with support for Flux and SDXL.
- **Security:** Firestore-backed API Key validation.
- **Reliability:** Google Cloud Tasks integration for long-running operations.
- **Namespacing:** Firestore collections prefixed with `visgate_deploy_api_` to avoid collisions.
- **CI/CD:** GitHub Actions workflow for automated deployment to `us-central1`.

### Changed
- Rebranded project as **Visgate Deploy API**.
- Full English translation of all documentation and system messages.
- Moved background tasks from `asyncio` to Cloud Tasks for better stability in Cloud Run.
