import os
import subprocess
from typing import Any

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
        print("s5cmd not found, skipping S3 sync")
        return False

    marker_path = os.path.join(local_path, _SYNC_MARKER)
    if os.path.exists(marker_path):
        print(f"\U0001f4e6 Model cache hit (sync complete marker found): {local_path}")
        return True

    # Remove any partially synced files to avoid stale weights
    if os.path.exists(local_path):
        file_count = _count_local_files(local_path)
        if file_count > 0:
            print(f"\u26a0\ufe0f  Partial sync detected ({file_count} files, no marker). Re-syncing...")
            import shutil
            shutil.rmtree(local_path, ignore_errors=True)

    os.makedirs(local_path, exist_ok=True)
    print(f"\U0001f680 Syncing model from S3: {s3_url}")

    concurrency = os.environ.get("S5CMD_CONCURRENCY", "50")
    part_size = os.environ.get("S5CMD_PART_SIZE_MB", "50")
    cmd = ["s5cmd", "--numworkers", concurrency]
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])
    cmd.extend(["cp", "--part-size", part_size, f"{s3_url.rstrip('/')}/*", local_path])

    try:
        subprocess.run(cmd, check=True)
        # Write completion marker only after full success
        with open(marker_path, "w") as f:
            f.write("ok")
        print("\u2705 S3 sync complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\u274c S3 sync failed: {e}")
        return False


def load_pipeline_optimized(model_id: str, token: str = None, device: str = "cuda") -> Any:
    """
    Load pipeline with S3 caching and persistent volume support.

    Cold-start cost path:
      1. S3 cache hit  → sync skipped  → load from disk  (~2-5s)
      2. S3 cache miss → full S3 sync  → load from disk  (~35-60s)
      3. No S3 config  → HF Hub download (first time) or HF cache hit
    """
    s3_url = os.environ.get("S3_MODEL_URL")
    volume_path = "/runpod-volume"

    model_name_slug = model_id.replace("/", "--")
    local_path = os.path.join(volume_path, model_name_slug)

    use_local = False
    if s3_url:
        synced = sync_from_s3(s3_url, local_path)
        if synced:
            # Tell HF libraries not to try network calls — model is fully local
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["DIFFUSERS_OFFLINE"] = "1"
            model_id = local_path
            use_local = True

    if not use_local:
        # Speed up HF Hub downloads with hf_transfer (C-based, ~5× faster)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"Loading pipeline from: {model_id} (local={use_local})")
    from pipelines.registry import load_pipeline
    return load_pipeline(model_id=model_id, token=token, device=device)
