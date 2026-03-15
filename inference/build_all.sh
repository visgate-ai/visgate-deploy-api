#!/usr/bin/env bash
# Trigger the remote GitHub Actions inference build instead of building locally.
# Usage: ./build_all.sh [targets]
# Example: ./build_all.sh image,audio
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="${GITHUB_REPOSITORY:-visgate-ai/visgate-deploy-api}"
WORKFLOW_FILE="${WORKFLOW_FILE:-inference.yaml}"
TARGETS="${1:-${TARGETS:-image,audio,video}}"
WAIT_FOR_RUN="${WAIT_FOR_RUN:-0}"

cd "${REPO_ROOT}"

REF="${REF:-}"
if [[ -z "${REF}" ]]; then
  REF="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
fi
if [[ -z "${REF}" || "${REF}" == "HEAD" ]]; then
  REF="main"
fi

if [[ "${ALLOW_DIRTY:-0}" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Refusing to trigger a remote build from a dirty worktree." >&2
  echo "GitHub Actions only builds committed state from ref=${REF}; your local edits will be ignored." >&2
  echo "Commit and push the changes first, or rerun with ALLOW_DIRTY=1 if you intentionally want the remote ref as-is." >&2
  exit 1
fi

print_run_summary() {
  local run_json="$1"
  python3 - <<'PY' "$run_json"
import json
import sys

run = json.loads(sys.argv[1]) if sys.argv[1] else {}
if not run:
    raise SystemExit(0)
print(f"Triggered run id: {run.get('databaseId')}")
print(f"Run URL: {run.get('url')}")
print(f"Status: {run.get('status')}")
PY
}

run_gh_dispatch() {
  if gh workflow run "${WORKFLOW_FILE}" --repo "${REPO}" --ref "${REF}" -f targets="${TARGETS}"; then
    return 0
  fi

  if [[ "${TARGETS}" != "image,audio,video" ]]; then
    echo "ERROR: Remote workflow does not accept the targets input yet. Push the workflow change first or run the full build." >&2
    return 1
  fi

  echo "Remote workflow does not accept targets yet; retrying full build without inputs..."
  gh workflow run "${WORKFLOW_FILE}" --repo "${REPO}" --ref "${REF}"
}

run_rest_dispatch() {
  local response_file
  response_file="$(mktemp)"

  if curl -fsSL -o "${response_file}" -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches" \
    -d "{\"ref\":\"${REF}\",\"inputs\":{\"targets\":\"${TARGETS}\"}}"; then
    rm -f "${response_file}"
    return 0
  fi

  if [[ "${TARGETS}" != "image,audio,video" ]]; then
    cat "${response_file}" >&2 || true
    rm -f "${response_file}"
    echo "ERROR: Remote workflow does not accept the targets input yet. Push the workflow change first or run the full build." >&2
    return 1
  fi

  echo "Remote workflow does not accept targets yet; retrying full build without inputs..."
  cat "${response_file}" >&2 || true
  rm -f "${response_file}"

  curl -fsSL -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches" \
    -d "{\"ref\":\"${REF}\"}"
}

if command -v gh >/dev/null 2>&1; then
  echo "Triggering ${WORKFLOW_FILE} on ${REPO} ref=${REF} targets=${TARGETS} via GitHub Actions..."
  run_gh_dispatch
  sleep 5
  RUN_JSON="$(gh run list --repo "${REPO}" --workflow "${WORKFLOW_FILE}" --branch "${REF}" --limit 1 --json databaseId,url,status 2>/dev/null | python3 -c 'import json,sys; runs=json.load(sys.stdin); print(json.dumps(runs[0] if runs else {}))')"
  print_run_summary "${RUN_JSON}"
  if [[ "${WAIT_FOR_RUN}" == "1" && -n "${RUN_JSON}" && "${RUN_JSON}" != "{}" ]]; then
    RUN_ID="$(python3 - <<'PY' "${RUN_JSON}"
import json
import sys
print(json.loads(sys.argv[1]).get('databaseId', ''))
PY
)"
    if [[ -n "${RUN_ID}" ]]; then
      gh run watch "${RUN_ID}" --repo "${REPO}" --exit-status
    fi
  fi
  exit 0
fi

TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [[ -z "${TOKEN}" ]]; then
  echo "ERROR: GitHub CLI is not installed and GITHUB_TOKEN/GH_TOKEN is not set." >&2
  echo "Use 'gh auth login' with GitHub CLI or export GITHUB_TOKEN to trigger the remote build." >&2
  exit 1
fi

echo "Triggering ${WORKFLOW_FILE} on ${REPO} ref=${REF} targets=${TARGETS} via GitHub REST API..."
run_rest_dispatch

echo "Triggered remote build successfully."
echo "Open: https://github.com/${REPO}/actions/workflows/${WORKFLOW_FILE}"
