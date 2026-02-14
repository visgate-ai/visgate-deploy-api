#!/usr/bin/env python3
"""
Deletes all endpoints in the Runpod account to free up quota.

Usage: python3 scripts/cleanup_runpod_endpoints.py
Reads RUNPOD from .env.local.
"""
import json
import os
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_LOCAL = os.path.join(REPO_ROOT, ".env.local")

RUNPOD_URL = "https://api.runpod.io/graphql"

QUERY_ENDPOINTS = """
query Endpoints {
  myself { endpoints { id name } }
}
"""

MUTATION_DELETE = """
mutation DeleteEndpoint($id: String!) { deleteEndpoint(id: $id) }
"""


def load_env():
    env = {}
    if os.path.isfile(ENV_LOCAL):
        with open(ENV_LOCAL) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def graphql(api_key: str, query: str, variables: dict | None = None) -> dict:
    url = f"{RUNPOD_URL}?api_key={api_key}"
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500] if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    if data.get("errors"):
        raise RuntimeError(data["errors"][0].get("message", str(data["errors"])))
    return data.get("data", {})


def main():
    env = load_env()
    api_key = env.get("RUNPOD", "").strip()
    if not api_key:
        print("ERROR: Set RUNPOD=... in .env.local.", file=sys.stderr)
        sys.exit(1)

    data = graphql(api_key, QUERY_ENDPOINTS)
    endpoints = (data.get("myself") or {}).get("endpoints") or []
    if not endpoints:
        print("No endpoints found, nothing to delete.")
        return

    print(f"Found {len(endpoints)} endpoints, deleting...")
    for ep in endpoints:
        eid = ep.get("id")
        name = ep.get("name", "")
        if not eid:
            continue
        try:
            graphql(api_key, MUTATION_DELETE, {"id": eid})
            print(f"  Deleted: {name or eid}")
        except Exception as e:
            print(f"  ERROR ({name or eid}): {e}", file=sys.stderr)
    print("Done.")


if __name__ == "__main__":
    main()
