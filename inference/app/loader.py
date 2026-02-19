import os
import subprocess
from typing import Any

def sync_from_s3(s3_url: str, local_path: str):
    """
    Sync model from S3 to local path using s5cmd.
    """
    if not s3_url:
        return
    
    # Check if s5cmd is installed
    try:
        subprocess.run(["s5cmd", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print("s5cmd not found, skipping S3 sync")
        return

    marker_file = os.path.join(local_path, "model_index.json")
    if os.path.exists(marker_file):
        print(f"📦 Model already exists in local cache: {local_path}")
        return

    os.makedirs(local_path, exist_ok=True)
    print(f"🚀 Model not found locally. Syncing from S3: {s3_url}")

    # AWS credentials should be in environment variables
    # s5cmd cp s3://bucket/path/* local_path
    concurrency = os.environ.get("S5CMD_CONCURRENCY", "50")  # default 50 for fast model download
    part_size = os.environ.get("S5CMD_PART_SIZE_MB", "50")
    cmd = ["s5cmd", "--numworkers", concurrency]
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])

    cmd.extend(["cp", "--part-size", part_size, f"{s3_url.rstrip('/')}/*", local_path])
    try:
        subprocess.run(cmd, check=True)
        print("✅ S3 Sync complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ S3 Sync failed: {e}")

def load_pipeline_optimized(model_id: str, token: str = None, device: str = "cuda") -> Any:
    """
    Load pipeline with S3 caching and persistent volume support.
    """
    s3_url = os.environ.get("S3_MODEL_URL")
    volume_path = "/runpod-volume"
    
    # Target path on persistent volume
    model_name_slug = model_id.replace("/", "--")
    local_path = os.path.join(volume_path, model_name_slug)
    
    if s3_url:
        sync_from_s3(s3_url, local_path)
        # If successfully synced, load from local path
        if os.path.exists(os.path.join(local_path, "model_index.json")):
             model_id = local_path

    print(f"Loading pipeline from: {model_id}")
    
    # Use standard load_pipeline logic but with potential local path
    from pipelines.registry import load_pipeline
    return load_pipeline(model_id=model_id, token=token, device=device)
