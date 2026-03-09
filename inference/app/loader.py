import os
import subprocess
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
    _log("INFO", f"🚀 Syncing model from S3: {s3_url}")

    concurrency = os.environ.get("S5CMD_CONCURRENCY", "50")
    part_size = os.environ.get("S5CMD_PART_SIZE_MB", "50")
    cmd = ["s5cmd", "--numworkers", concurrency]
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])
    cmd.extend(["cp", "--part-size", part_size, f"{s3_url.rstrip('/')}/*", local_path])

    try:
        subprocess.run(cmd, check=True)
        # s5cmd exits 0 even when the source glob matched nothing ("no object found").
        # Detect this: if the destination has no files the sync silently did nothing.
        file_count = _count_local_files(local_path)
        if file_count == 0:
            _log("WARN", "❌ S3 sync appeared to succeed but destination is empty (s5cmd 'no object found'). Falling back to HuggingFace.")
            import shutil
            shutil.rmtree(local_path, ignore_errors=True)
            return False
        # Write completion marker only after confirming files were actually synced
        with open(marker_path, "w") as f:
            f.write("ok")
        _log("INFO", "✅ S3 sync complete.")
        return True
    except subprocess.CalledProcessError as e:
        _log("WARN", f"❌ S3 sync failed (exit {e.returncode}), falling back to HuggingFace.")
        return False


def load_pipeline_optimized(model_id: str, token: str = None, device: str = "cuda") -> tuple[Any, bool]:
    """
    Load pipeline with S3 caching and persistent volume support.

    Returns (pipeline, loaded_from_local) tuple.
    loaded_from_local is True when model was synced from R2, False when downloaded from HF.

    Cold-start cost path:
      1. S3 cache hit  → sync skipped  → load from disk  (~2-5s)
      2. S3 cache miss → full S3 sync  → load from disk  (~35-60s)
      3. No S3 config  → HF Hub download (first time) or HF cache hit
    """
    model_id, use_local = resolve_model_source(model_id)

    if not use_local:
        # Speed up HF Hub downloads with hf_transfer (C-based, ~5× faster)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    _log("INFO", f"Loading pipeline from: {model_id} (local={use_local})")
    from pipelines.registry import load_pipeline
    pipeline = load_pipeline(model_id=model_id, token=token, device=device)
    return (pipeline, use_local)


def resolve_model_source(model_id: str) -> tuple[str, bool]:
    """Return the effective model source path, preferring a synced S3 path when available."""
    s3_url = os.environ.get("S3_MODEL_URL")
    volume_path = "/runpod-volume"

    model_name_slug = model_id.replace("/", "--")
    local_path = os.path.join(volume_path, model_name_slug)

    use_local = False
    if s3_url:
        _log("INFO", f"S3_MODEL_URL set — attempting R2 sync: {s3_url}")
        synced = sync_from_s3(s3_url, local_path)
        if synced:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["DIFFUSERS_OFFLINE"] = "1"
            model_id = local_path
            use_local = True
        else:
            _log("INFO", "R2 sync skipped/failed — downloading from HuggingFace")
    else:
        _log("INFO", f"No S3_MODEL_URL — downloading from HuggingFace: {model_id}")
    return model_id, use_local
