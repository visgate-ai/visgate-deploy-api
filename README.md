# Visgate Deploy API

Deploy Hugging Face diffusion models to Runpod with a single async API.

This service creates endpoints, tracks readiness, and lets you poll/stream status while models warm up.

## Hosted API

- Base URL: `https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app`
- Auth: `Authorization: Bearer <RUNPOD_API_KEY>` (or `X-Runpod-Api-Key`)

## Quick Start (Beginner)

1) Health check

```bash
curl -s "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app/health"
```

2) Create deployment

```bash
curl -X POST "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app/v1/deployments" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_id": "stabilityai/sd-turbo",
    "gpu_tier": "A10",
    "hf_token": "hf_xxx_optional",
    "user_webhook_url": "https://your-app.com/webhook",
    "cache_scope": "shared"
  }'
```

3) Poll status

```bash
curl -s \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app/v1/deployments/<deployment_id>"
```

4) Delete when done (cost control)

```bash
curl -X DELETE \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app/v1/deployments/<deployment_id>"
```

## Cache Modes (S3 destekli)

- `off`: cache yok (default)
- `shared`: platformun ortak S3 cache alanı
- `private`: kendi S3/R2/Minio alanın (`user_s3_url` + `user_aws_*` gerekir)

### Open-source güvenlik notu

- `shared` mod public isteklerde **salt-okunur ve allowlist** mantığıyla sınırlandırılır.
- Amaç: ortak S3 alanını spam yazma / maliyet şişirme / cache poisoning risklerinden korumak.
- Kendi özel cache yazımı için `private` kullan.

## Bu alana nasıl para ödenir?

Toplam maliyet genelde 2 yerde oluşur:

1) Runpod
- Endpoint açma süresi (cold start)
- GPU çalıştığı süre

2) S3 (veya uyumlu obje depolama)
- Storage (GB/ay)
- Request (PUT/GET)
- Egress (internet çıkışı varsa)

`shared` cache doğru model setinde cold start süresini düşürür; ama kontrolsüz bırakılırsa request/egress maliyetini artırır. Bu yüzden allowlist ve scope ayrımı önemlidir.

## Yardımcı Scriptler

```bash
API_BASE="https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app" \
python3 scripts/prod_api_smoke.py
```

```bash
API_BASE="https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app" \
RUNPOD="<runpod_key>" HF="<hf_token_optional>" \
python3 scripts/e2e_timed_istanbul.py
```

## Geliştirici Dokümanları

- `deploy-api/README.md`
- `deploy-api/USAGE.md`
- `inference/README.md`

## License

MIT. See `LICENSE`.
