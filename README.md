# nvidiaAiChat

NVIDIA NIM (`build.nvidia.com`) üzerindeki modelleri tek bir API key ile
kullanan, kendi kontrolünde çalışan çok-modelli AI asistan. Claude'a benzer
şekilde: chat, dosya/görsel yükleme, otomatik model seçimi, ajan modunda
araç çağırma.

## Nasıl çalışıyor

- **Tek API, çok model** — NVIDIA NIM, OpenAI uyumlu tek bir endpoint
  (`https://integrate.api.nvidia.com/v1`) üzerinden 100'den fazla modele
  erişim veriyor. Bu proje `openai` Python SDK'sını sadece `base_url`
  değiştirerek buna yönlendiriyor (`nvidia_client.py`).
- **Katalog tahmin edilmiyor, canlı çekiliyor** — API key girildiğinde
  backend gerçek `/v1/models` listesini çeker (`catalog.py`), her modeli
  ailesine göre etiketler (`fast`, `code`, `reasoning`, `agent`, `vision`,
  `general`) ve `models_cache.json`'a yazar. Key girilmeden önce sadece
  örnek/placeholder bir liste gösterilir.
- **Model eşlemesi** (kullanıcı isteğine göre): Hız → DeepSeek (`fast`),
  Kodlama → MiniMax (`code`), Karmaşık akıl yürütme → Qwen (`reasoning`),
  Ajan/çoklu adım → Kimi (`agent`). Tanınmayan aileler `general` alır,
  ismi `vision`/`vl`/`vila` gibi ipucu içerenler ek olarak `vision`
  etiketi alır.
- **Otomatik yönlendirme** (`router.py`) — kullanıcı "Otomatik" seçiliyken
  mesajın içeriğine (kod anahtar kelimeleri, görsel eki var mı, ajan modu
  açık mı) bakıp uygun etiketteki ilk modeli seçer. Kullanıcı istediği an
  dropdown'dan manuel bir model de seçebilir.
- **Dosya / görsel analizi** — resimler base64 olarak OpenAI vision mesaj
  formatında gönderilir (`image_url` içerikli mesaj); metin/kod dosyaları
  ve PDF'ler (pypdf ile) okunup mesaj bağlamına eklenir.
- **Ajan modu** — açıldığında istek `tools` parametresiyle gönderilir,
  model bir araç çağırırsa (`tools.py`) backend aracı çalıştırıp sonucu
  modele geri verir ve nihai cevabı döner. Şu an tek örnek araç
  (`get_current_datetime`) var; yeni bir araç eklemek için `tools.py`
  içine fonksiyon tanımı + karşılığını yazman yeterli (dosya arama, kod
  çalıştırma, web isteği gibi araçlar aynı desene eklenebilir).
- **Model doğrulama** (`catalog.py`) — NVIDIA'nın kataloğu yüzlerce model
  içeriyor ama bazıları hesapta deploy edilmemiş (404) ya da aşırı
  talepten zaman aşımına uğruyor (DeepSeek gibi popüler modeller). Kataloğu
  yenilerken bilinen aileler (`CURATED_FAMILIES`) her biri gerçekten
  çağrılarak paralel test edilir; sadece o an çalışanlar gösterilir,
  elenenlerin gerçek hata mesajı `/api/models` yanıtındaki
  `diagnostics`'te görülebilir.
- **Uzun işler arka planda** (`app.py`) — hem sohbet cevabı hem katalog
  testi bir job olarak arka plan thread'inde başlatılır, istemci
  `/api/jobs/{id}`'yi birkaç saniyede bir sorar (`pollJob` — `app.js`).
  İlk sürüm bunun yerine uzun süre açık kalan tek bir bağlantı üzerinden
  "nabız" akıtıyordu ama Cloudflare/nginx zincirinde bu güvenilmez çıktı
  (sessiz aralıklar istemciye hiç ulaşmayabiliyordu). Kısa/bağımsız
  sorgulama isteği daha sağlam: telefon sekmeyi arka plana atsa bile iş
  sunucuda çalışmaya devam ediyor, sekmeye dönüldüğünde kaldığı yerden
  soruluyor. Bedeli: cevap artık token-token akmıyor, hazır olunca tek
  parça geliyor.

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

Tarayıcıda `http://localhost:8010` aç, sağ üstteki **⚙ Ayarlar**'dan
build.nvidia.com'dan aldığın `nvapi-...` key'ini gir. Kaydedince katalog
otomatik yenilenir.

API key'i almak için: `build.nvidia.com` → hesap oluştur/doğrula → bir
model sayfasında **Get API Key** → `nvapi-...` anahtarını kopyala. Kart
istemiyor, ücretsiz katman ~40 istek/dk sınırlı.

## Dosya yapısı

```
app.py           FastAPI router'ları — /api/status, /api/settings,
                 /api/models, /api/models/refresh, /api/upload, /api/chat,
                 /api/jobs/{id} (arka plan iş durumu sorgulama)
nvidia_client.py NVIDIA NIM'e OpenAI-uyumlu client + settings.json okuma/yazma
catalog.py       Canlı model listesi çekme + aile bazlı etiketleme
router.py        "Otomatik" moddaki model seçim mantığı
tools.py         Ajan modunda çağrılabilen araçlar
static/          Tek sayfalık chat arayüzü (index.html, app.js, style.css)
```

## Bilinen v1 sınırları

- Cevap token-token akmıyor (bkz. yukarıdaki "Uzun işler arka planda"),
  yazıyor animasyonu gösterilip cevap hazır olunca tek parça geliyor.
- Sohbet geçmişi (`history`) sadece düz metin olarak tutuluyor — önceki
  turdaki bir görsel, yeni turda modele tekrar gönderilmiyor.
- PDF'ler sadece metin katmanı varsa okunabiliyor (taranmış/görüntü PDF
  için OCR yok).
- Tek kullanıcılık: `settings.json` sunucu genelinde tek bir API key
  tutuyor, çoklu kullanıcı/oturum ayrımı yok.
- İş durumu (`_jobs`) sadece bellekte tutuluyor — `systemctl restart`
  sırasında yarım kalan bir iş kaybolur (istemci "iş bulunamadı" hatası
  görüp otomatik tekrar dener).

## Dağıtım (opsiyonel)

Sunucuda zaten 80 portunu nginx kullanıyor ve başka botlar başka portları
tutuyor (8000 whatsapp, 8001 supertonic-web, 8002 instube, 8003 sitebot,
5000/5001/5057 diğer projeler) — bu yüzden bu uygulama **8004** portunda,
sadece `127.0.0.1`'de dinleyip nginx arkasında bir subdomain üzerinden
yayınlanacak şekilde hazırlandı:

1. Sunucuda repoyu çek, `setup` gibi venv kur (yukarıdaki Kurulum adımları).
2. `deploy/nvidiaaichat.service.example` dosyasını
   `/etc/systemd/system/nvidiaaichat.service` olarak kopyala, yolları kendi
   sunucuna göre kontrol et, `systemctl enable --now nvidiaaichat`.
3. `deploy/nginx-nvidiaaichat.conf.example` dosyasındaki `ai.wizaicorp.com`
   adını istediğin subdomain ile değiştir, DNS'te bu subdomain için sunucunun
   IP'sine (`77.42.45.229`) A kaydı ekle, sonra dosyanın içindeki nginx +
   certbot adımlarını uygula.
4. `sitebot`'un müşteriye subdomain satmasını engellemek için seçtiğin
   subdomain'i `sitebot/config.py` içindeki `RESERVED_SUBDOMAINS` listesine
   ekletmeyi unutma (bkz. ana repo CLAUDE.md).
