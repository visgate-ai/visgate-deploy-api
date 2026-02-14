#!/usr/bin/env bash
# Uçtan uca: İstanbul prompt gönder, resmi kaydet ve göster.
# .env.local'dan RUNPOD okunur.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .env.local ]] && set -a && source .env.local && set +a
: "${RUNPOD:?RUNPOD gerekli - .env.local içinde RUNPOD=...}"

ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:-mxa0he79ljwapv}"
BASE="https://api.runpod.ai/v2/${ENDPOINT_ID}"
PROMPT="A beautiful panoramic view of Istanbul with the Bosphorus strait, Hagia Sophia and minarets, golden hour, photorealistic, 4k"

echo "Job başlatılıyor: $PROMPT"
RESP=$(curl -s -X POST "${BASE}/run" \
  -H "Authorization: Bearer ${RUNPOD}" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"prompt\": \"${PROMPT}\", \"num_inference_steps\": 28}}")
JOB_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
[[ -z "$JOB_ID" ]] && echo "Job id alınamadı: $RESP" && exit 1
echo "Job ID: $JOB_ID"

echo "Tamamlanması bekleniyor (poll 15s; ilk seferde worker cold start 2-5 dk sürebilir)..."
MAX_POLL=60
n=0
while true; do
  STATUS=$(curl -s -H "Authorization: Bearer ${RUNPOD}" "${BASE}/status/${JOB_ID}")
  S=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  echo "  status: $S"
  [[ "$S" == "COMPLETED" ]] && break
  [[ "$S" == "FAILED" ]] && echo "Job FAILED: $STATUS" && exit 1
  n=$((n+1))
  [[ $n -ge $MAX_POLL ]] && echo "Zaman aşımı (${MAX_POLL} poll)." && exit 1
  sleep 15
done

echo "$STATUS" | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
o = d.get('output') or {}
if o.get('error'):
    print('ERROR:', o['error'], file=sys.stderr)
    sys.exit(1)
b64 = o.get('image_base64')
if not b64:
    print('image_base64 yok', file=sys.stderr)
    sys.exit(2)
with open('istanbul.png', 'wb') as f:
    f.write(base64.b64decode(b64))
print('OK')
"
[[ $? -ne 0 ]] && exit 1
echo "Kaydedildi: $REPO_ROOT/istanbul.png"

if command -v xdg-open &>/dev/null; then
  xdg-open "$REPO_ROOT/istanbul.png"
elif command -v open &>/dev/null; then
  open "$REPO_ROOT/istanbul.png"
fi
echo "Resim açıldı."
