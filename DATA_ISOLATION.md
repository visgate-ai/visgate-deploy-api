# Veri Karışıklığı Önlemi Alındı

Mevcut GCP projenizde (`visgate`) veri çakışmasını önlemek için Firestore koleksiyon isimlerine **"serverless_visgate_"** ön eki eklendi.

**Yeni Koleksiyon İsimleri:**
1.  `serverless_visgate_deployments`: Deployment takibi için.
2.  `serverless_visgate_logs`: Log kayıtları için.
3.  `serverless_visgate_api_keys`: API anahtarları için.

Bu değişiklikler hem `.env` dosyasında (lokal testler için) hem deGitHub Actions (`.github/workflows/deploy.yaml`) dosyasında (canlı ortam için) güncellendi.
