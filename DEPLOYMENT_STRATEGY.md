# Neden Firebase (Firestore) Kullanıyoruz?

1.  **Süreklilik (Persistence):** Orchestrator "serverless" çalışır. Yani kullanılmadığında kapanır ve her request belki farklı bir sunucuya düşer.
    - Deployment isteği gelir -> İşlem başlar -> Yanıt döner (202).
    - Arka planda 3-4 dakika süren işlem (Runpod endpoint açma, bekleme) devam eder.
    - Bu sürede orchestrator kapanıp açılabilir veya yeni bir instance görevi devralabilir (Cloud Tasks).
    - Eğer durumu **hafızada (RAM)** tutarsak, orchestrator yeniden başladığında "bu deployment hangi aşamadaydı?" bilgisini kaybederiz. Firestore bu bilgiyi kalıcı olarak tutar.

2.  **API Key Yönetimi:** API anahtarlarını kod veya environment variable içinde tutmak yerine bir veritabanında tutmak, anahtarları dinamik olarak ekleyip silmenizi (revoke) sağlar.

## En Ucuz Deployment Stratejisi

**GitHub Actions Workflow** oluşturuldu (`.github/workflows/deploy.yaml`).

Bu workflow:
- **Region:** `us-central1` (Iowa) olarak ayarlandı. Genellikle en ucuz ve "Tier 1" free tier kotasının geçerli olduğu bölgedir.
- **Maliyet:**
    - **Cloud Run:** İlk 2 milyon istek/ay ve belli bir işlemci süresi ücretsizdir. Trafik yoksa $0.
    - **Firestore:** Günde 50.000 okuma/yazma ücretsizdir (Spark Plan). Visgate için fazlasıyla yeterli.
    - **Cloud Build:** Günde 120 dakika build ücretsizdir.
    - **Cloud Tasks:** İlk 1 milyon işlem/ay ücretsizdir.

**Sonuç:** Bu altyapı, düşük/orta ölçekli kullanımda tamamen **ÜCRETSİZ** kalacaktır.

## Eğer Firebase İstemiyorsanız?

Deployment durumunu bir yerde tutmak zorundayız. Alternatifler:
1.  **Redis:** Cloud Run ile Redis instance bağlamak (Genelde aylık sabit ücreti vardır, Firestore'dan pahalı olabilir).
2.  **SQL (Cloud SQL):** En küçük instance bile aylık ~$20-30 maliyet çıkarır.
3.  **No-State (Önerilmez):** İsteği attıktan sonra kullanıcı bağlantıyı koparmadan 3-5 dakika beklemek zorunda kalır (Timeout riski çok yüksek).

Bu yüzden **Firestore**, serverless mimariler için hem en ucuz (bedava) hem de en pratik çözümdür.
