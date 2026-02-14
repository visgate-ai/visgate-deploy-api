# Changelog

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
