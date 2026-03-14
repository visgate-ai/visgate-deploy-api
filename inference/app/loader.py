import os
import subprocess
import time
from typing import Any


def _log(level: str, message: str) -> None:
    """Forward to log_tunnel (real-time API stream) and stdout (RunPod dashboard)."""
    print(message, flush=True)
    try:
        from app.runtime_common import log_tunnel
        log_tunnel(level, message)
    except Exception:
        pass


# Completion marker written after a successful sync.
# Using a separate file (not model_index.json) means a partial sync is detected
# on the next cold start and re-synced cleanly.
_SYNC_MARKER = ".visgate_sync_complete"


def _count_local_files(path: str) -> int:
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def _round_s(value: float) -> float:
    return round(value, 3)


def sync_from_s3(s3_url: str, local_path: str) -> bool:
    """
    Sync model from S3 to local path using s5cmd.
    Returns True if the model is available locally after this call.
    """
    if not s3_url:
        return False

    # Check if s5cmd is installed
    try:
        subprocess.run(["s5cmd", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        _log("WARN", "s5cmd not found, skipping S3 sync")
        return False

    marker_path = os.path.join(local_path, _SYNC_MARKER)
    if os.path.exists(marker_path):
        _log("INFO", f"📦 Model cache hit (sync complete marker found): {local_path}")
        return True

    # Remove any partially synced files to avoid stale weights
    if os.path.exists(local_path):
        file_count = _count_local_files(local_path)
        if file_count > 0:
            _log("WARN", f"⚠️  Partial sync detected ({file_count} files, no marker). Re-syncing...")
            import shutil
            shutil.rmtree(local_path, ignore_errors=True)

    os.makedirs(local_path, exist_ok=True)
    endpoint_for_log = os.environ.get("VISGATE_R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL", "(none)")
    _log("INFO", f"🚀 Syncing model from S3: {s3_url} endpoint={endpoint_for_log}")

    concurrency = os.environ.get("S5CMD_CONCURRENCY", "50")
    part_size = os.environ.get("S5CMD_PART_SIZE_MB", "50")

    # Read R2 credentials from VISGATE_R2_* vars (avoids RunPod overriding standard AWS_* vars).
    # Fall back to AWS_* for local/custom deployments that set them directly.
    r2_key = os.environ.get("VISGATE_R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    r2_secret = os.environ.get("VISGATE_R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    endpoint_url = os.environ.get("VISGATE_R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL", "")

    cmd = ["s5cmd", "--numworkers", concurrency]
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])
    cmd.extend(["cp", "--part-size", part_size, f"{s3_url.rstrip('/')}/*", local_path])

    # Build a clean subprocess env: start from current env, then inject R2 credentials
    # explicitly so RunPod's own AWS_* vars cannot shadow them.
    subproc_env = {**os.environ}
    if r2_key:
        subproc_env["AWS_ACCESS_KEY_ID"] = r2_key
    if r2_secret:
        subproc_env["AWS_SECRET_ACCESS_KEY"] = r2_secret
    # Remove any RunPod-injected endpoint/region overrides — we pass --endpoint-url explicitly.
    subproc_env.pop("AWS_ENDPOINT_URL", None)
    subproc_env.pop("AWS_DEFAULT_REGION", None)
    subproc_env.pop("AWS_REGION", None)

    t_sync_start = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, env=subproc_env)
            t_sync_elapsed = time.time() - t_sync_start
            file_count = _count_local_files(local_path)
            if file_count == 0:
                _log("WARN", "S3 sync appeared to succeed but destination is empty. Falling back to HuggingFace.")
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
                return False
            with open(marker_path, "w") as f:
                f.write("ok")
            _log("INFO", f"S3 sync complete. files={file_count} elapsed={t_sync_elapsed:.1f}s")
            return True
        except subprocess.CalledProcessError as e:
            t_sync_elapsed = time.time() - t_sync_start
            if attempt < max_retries - 1:
                import random
                wait = (2 ** attempt) + random.uniform(0, 1)
                _log("WARN", f"S3 sync attempt {attempt + 1} failed (exit {e.returncode}) after {t_sync_elapsed:.1f}s, retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                _log("WARN", f"S3 sync failed after {max_retries} attempts (exit {e.returncode}), falling back to HuggingFace.")
                return False


def load_pipeline_optimized(model_id: str, token: str = None, device: str = "cuda") -> tuple[Any, bool, dict[str, float | bool | None]]:
    """
    Load pipeline with S3 caching and persistent volume support.

    Returns (pipeline, loaded_from_local, timings) tuple.
    loaded_from_local is True when model was synced from R2, False when downloaded from HF.

    Cold-start cost path:
      1. S3 cache hit  → sync skipped  → load from disk  (~2-5s)
      2. S3 cache miss → full S3 sync  → load from disk  (~35-60s)
      3. No S3 config  → HF Hub download (first time) or HF cache hit
    """
    model_id, use_local, t_r2_sync_s, loaded_from_cache = resolve_model_source(model_id)

    if not use_local:
        # Speed up HF Hub downloads with hf_transfer (C-based, ~5× faster)
        try:
            import hf_transfer  # noqa: F401
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        except ImportError:
            _log("WARN", "hf_transfer not installed, using slower Python download")
    else:
        # Only set offline mode after verifying model files actually exist
        model_index = os.path.join(model_id, "model_index.json")
        config_json = os.path.join(model_id, "config.json")
        if os.path.exists(model_index) or os.path.exists(config_json):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            _log("WARN", "Local model missing index/config, NOT setting offline mode")

    _log("INFO", f"Loading pipeline from: {model_id} (local={use_local})")
    from pipelines.registry import load_pipeline
    t_model_load_start = time.time()
    pipeline = load_pipeline(model_id=model_id, token=token, device=device)
    t_model_load_s = _round_s(time.time() - t_model_load_start)
    return (
        pipeline,
        use_local,
        {
            "t_r2_sync_s": t_r2_sync_s,
            "t_model_load_s": t_model_load_s,
            "loaded_from_cache": loaded_from_cache,
        },
    )


def resolve_model_source(model_id: str) -> tuple[str, bool, float | None, bool]:
    """Return the effective model source path, preferring a synced S3 path when available."""
    s3_url = os.environ.get("S3_MODEL_URL")
    volume_path = os.environ.get("MODEL_CACHE_DIR", "/tmp/models")  # FAST-PATH: NVMe ephemeral

    model_name_slug = model_id.replace("/", "--")
    local_path = os.path.join(volume_path, model_name_slug)

    use_local = False
    t_r2_sync_s: float | None = None
    loaded_from_cache = False
    if s3_url:
        _log("INFO", f"S3_MODEL_URL set — attempting R2 sync: {s3_url}")
        t0 = time.time()
        synced = sync_from_s3(s3_url, local_path)
        elapsed = _round_s(time.time() - t0)
        t_r2_sync_s = elapsed
        if synced:
            _log("INFO", f"✅ R2 sync done in {elapsed:.1f}s — loading from disk")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["DIFFUSERS_OFFLINE"] = "1"
            model_id = local_path
            use_local = True
            loaded_from_cache = True
        else:
            _log("INFO", f"R2 sync skipped/failed after {elapsed:.1f}s — downloading from HuggingFace")
    else:
        _log("INFO", f"No S3_MODEL_URL — downloading from HuggingFace: {model_id}")
    return model_id, use_local, t_r2_sync_s, loaded_from_cache
