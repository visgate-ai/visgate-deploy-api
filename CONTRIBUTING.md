# Contributing to Visgate Deploy API

Thank you for your interest in contributing! This guide covers how to set up a development environment, the project structure, and contribution guidelines.

## Development Setup

```bash
git clone https://github.com/visgate-ai/visgate-deploy-api
cd visgate-deploy-api/deploy-api

pip install -r requirements.txt

# Run tests — no cloud credentials needed
python -m pytest tests/ -q

# Start the API locally (no GCP needed — uses in-memory storage automatically)
cp .env.example .env
# Edit .env: set RUNPOD_TEMPLATE_ID if you want to test actual RunPod deployments
uvicorn src.main:app --reload
```

## Project Structure

```
deploy-api/src/
  api/routes/          # HTTP endpoint handlers (deployments, health, tasks, internal)
  services/
    deployment.py      # Main orchestration logic
    runpod.py          # RunPod GraphQL provider
    huggingface.py     # HF model validation + VRAM estimation
    gpu_selection.py   # GPU candidate selection
    gpu_registry.py    # GPU specs + tier mappings
    firestore_repo.py  # Firestore state storage (production)
    memory_repo.py     # In-memory state storage (local dev / tests)
    provider_factory.py # Provider registration registry
  models/
    model_specs_registry.py  # Hand-tuned VRAM values for 18+ models
    schemas.py               # Request/response Pydantic models
    entities.py              # Internal domain models
  core/
    config.py          # All env var settings (pydantic-settings)
    errors.py          # Custom exception hierarchy
    logging.py         # Structured JSON logging
    telemetry.py       # OpenTelemetry tracing

inference/
  app/
    worker.py          # RunPod job handler
    loader.py          # S3 sync + pipeline loading
  pipelines/
    base.py            # Abstract pipeline interface
    flux.py            # FLUX.1 pipelines
    sdxl.py            # SDXL / SD-Turbo pipelines
    registry.py        # Pipeline registry
```

## How to Add a New Model to the Registry

Edit [`deploy-api/src/models/model_specs_registry.py`](deploy-api/src/models/model_specs_registry.py):

```python
ModelSpec(
    hf_model_id="org/model-name",
    pipeline_tag="text-to-image",
    gpu_memory_gb=24,   # minimum VRAM in GB (test on real hardware if possible)
    notes="Optional notes about the model",
),
```

Add a corresponding unit test in [`tests/unit/test_models.py`](deploy-api/tests/unit/test_models.py).

## How to Add a New Inference Pipeline

1. Create `inference/pipelines/yourpipeline.py` extending `BasePipeline` from `base.py`.
2. Implement `load(model_id, device, **kwargs)` and `run(job_input)`.
3. Register it in `inference/pipelines/registry.py`.
4. Add the pipeline tag to your model entries in `model_specs_registry.py`.

## How to Add a New GPU Provider

1. Create `deploy-api/src/services/yourprovider.py` implementing `BaseProvider` from `base_provider.py`.
2. Implement all abstract methods: `create_endpoint`, `delete_endpoint`, `list_endpoints`.
3. Register in `deploy-api/src/services/provider_factory.py`:
   ```python
   from src.services.yourprovider import YourProvider
   register_provider("yourprovider", YourProvider())
   ```

## Pull Request Guidelines

- **One concern per PR** — separate bug fixes from new features.
- **Tests required** — all new logic must have corresponding unit tests. Run `python -m pytest tests/ -q` before opening a PR.
- **Type hints** — all new functions must have type annotations.
- **No secrets** — never commit API keys, tokens, or credentials. Use `.env` locally.
- **CHANGELOG** — add an entry under the appropriate version in `CHANGELOG.md`.

## Running the Full Test Suite

```bash
cd deploy-api
python -m pytest tests/ -v
```

39 tests, 2 skipped (integration tests that require live RunPod). All others run without any cloud credentials.

## Reporting Issues

Use [GitHub Issues](https://github.com/visgate-ai/visgate-deploy-api/issues). Please include:
- Exact error message or unexpected behavior
- Steps to reproduce
- Environment (local / Cloud Run, Python version)
- Relevant log output

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
