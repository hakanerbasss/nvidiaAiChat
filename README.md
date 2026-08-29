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
                 /api/models, /api/upload, /api/chat
nvidia_client.py NVIDIA NIM'e OpenAI-uyumlu client + settings.json okuma/yazma
catalog.py       Canlı model listesi çekme + aile bazlı etiketleme
router.py        "Otomatik" moddaki model seçim mantığı
tools.py         Ajan modunda çağrılabilen araçlar
static/          Tek sayfalık chat arayüzü (index.html, app.js, style.css)
```

## Bilinen v1 sınırları

- Ajan modunda cevap token-token akmıyor, araç sonucu hazır olunca tek
  parça geliyor (araç çağrısı + streaming'i aynı anda doğru yapmak daha
  karmaşık; sıradaki adım olarak eklenebilir).
- Sohbet geçmişi (`history`) sadece düz metin olarak tutuluyor — önceki
  turdaki bir görsel, yeni turda modele tekrar gönderilmiyor.
- PDF'ler sadece metin katmanı varsa okunabiliyor (taranmış/görüntü PDF
  için OCR yok).
- Tek kullanıcılık: `settings.json` sunucu genelinde tek bir API key
  tutuyor, çoklu kullanıcı/oturum ayrımı yok.

## Dağıtım (opsiyonel)

Kendi sunucunda systemd servisi olarak çalıştırmak istersen
`deploy/nvidiaaichat.service.example` dosyasını örnek al — yolu, portu ve
kullanıcıyı kendi ortamına göre düzenleyip `/etc/systemd/system/` altına
kopyalaman yeterli.
