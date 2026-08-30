import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_client import CACHE_PATH, get_client

# Katalogtaki yüzlerce modelin çoğu eski/test modeli ya da hesapta gerçek bir
# çalışan "function" olarak deploy edilmemiş — çağrılınca 404 veriyor. Kullanıcıyı
# yüzlerce kriptik id ile boğmak yerine bilinen/popüler aileleri seçip her birini
# refresh_catalog() sırasında gerçekten çağırarak doğruluyoruz; sadece çalışanlar
# gösteriliyor. "match": canlı model id'sinde aranacak alt dize (küçük harf).
CURATED_FAMILIES = [
    {"match": "deepseek-ai/deepseek-v4-flash", "label": "DeepSeek V4 Flash (hızlı)", "tags": ["fast"]},
    {"match": "deepseek-ai/deepseek-v4-pro", "label": "DeepSeek V4 Pro (güçlü)", "tags": ["reasoning", "general"]},
    {"match": "minimaxai/minimax-m3", "label": "MiniMax M3 (kodlama)", "tags": ["code"]},
    {"match": "qwen/qwen3.5", "label": "Qwen 3.5 (akıl yürütme)", "tags": ["reasoning"]},
    {"match": "moonshotai/kimi-k3", "label": "Kimi K3 (ajan)", "tags": ["agent"]},
    {"match": "moonshotai/kimi-k2", "label": "Kimi K2 (ajan)", "tags": ["agent"]},
    {"match": "zhipuai/glm-5", "label": "GLM 5 (genel)", "tags": ["general"]},
    {"match": "meta/llama-3.2-90b-vision", "label": "Llama 3.2 Vision (görsel analiz)", "tags": ["vision"]},
    {"match": "meta/llama-3.3-70b-instruct", "label": "Llama 3.3 70B (genel)", "tags": ["general"]},
    {"match": "mistralai/mistral-large", "label": "Mistral Large (genel)", "tags": ["general"]},
]

# API key henüz girilmemişken gösterilecek örnek liste (test edilmemiş).
FALLBACK_CATALOG = [
    {"id": "deepseek-ai/deepseek-v4-flash", "label": "DeepSeek V4 Flash (hızlı)", "tags": ["fast"]},
    {"id": "minimaxai/minimax-m3", "label": "MiniMax M3 (kodlama)", "tags": ["code"]},
    {"id": "qwen/qwen3.5-397b-a17b", "label": "Qwen 3.5 (akıl yürütme)", "tags": ["reasoning"]},
    {"id": "moonshotai/kimi-k2.6", "label": "Kimi K2.6 (ajan)", "tags": ["agent"]},
    {"id": "zhipuai/glm-5.1", "label": "GLM 5.1 (genel)", "tags": ["general"]},
]


def _match_families(live_ids: list) -> list:
    """Canlı /v1/models listesinde CURATED_FAMILIES'teki her aile için en
    kısa/en sade eşleşen id'yi seçer (örn. 'deepseek-v4-flash-0731' değil de
    varsa düz 'deepseek-v4-flash')."""
    used = set()
    matches = []
    for fam in CURATED_FAMILIES:
        candidates = [mid for mid in live_ids if fam["match"] in mid.lower() and mid not in used]
        if not candidates:
            continue
        candidates.sort(key=len)
        chosen = candidates[0]
        used.add(chosen)
        matches.append({"id": chosen, "label": fam["label"], "tags": fam["tags"]})
    return matches


PROBE_ATTEMPTS = 3
PROBE_TIMEOUT = 45  # DeepSeek gibi çok talep gören modeller kuyrukta uzun bekleyebiliyor


def _probe_model(client, model_id: str) -> tuple:
    """Modelin hesapta gerçekten çağrılabilir olup olmadığını doğrular.
    429 (hız sınırı) veya zaman aşımı geçici olabilir — paralel çalıştığımız
    için başka bir model bu isteği tetiklemiş olabilir; ikisinde de artan
    bir bekleme ile birkaç kez dener. Sadece 404 (gerçekten deploy edilmemiş)
    hemen kesin sayılır. (ok: bool, error_str: str|None) döner."""
    last_err = "bilinmeyen hata"
    for attempt in range(PROBE_ATTEMPTS):
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "merhaba"}],
                max_tokens=1,
                timeout=PROBE_TIMEOUT,
            )
            return True, None
        except Exception as e:
            err = str(e)
            last_err = err[:300]
            is_404 = getattr(e, "status_code", None) == 404 or "404" in err
            if is_404:
                return False, last_err  # kalici hata, tekrar denemeye gerek yok
            if attempt < PROBE_ATTEMPTS - 1:
                time.sleep(2 + attempt * 2)  # 2sn, sonra 4sn
    return False, last_err


def refresh_catalog() -> dict:
    """NVIDIA'nın gerçek /v1/models listesini çeker, bilinen ailelerle eşleştirir
    ve her adayı PARALEL olarak gerçekten çağırarak doğrular. Bu fonksiyon
    birkaç dakika sürebilir (429/zaman aşımında yeniden deniyor) — çağıran taraf
    (app.py /api/models/refresh) bunu arka plan thread'inde çalıştırıp periyodik
    "nabız" göndererek Cloudflare'ın bağlantıyı erken kesmesini engelliyor, o
    yüzden burada süre konusunda cimri davranmaya gerek yok. Sadece çalışan
    modeller kaydedilir; elenenlerin hata mesajı diagnostics'e yazılır
    (/api/models yanıtında görülebilir — 404 mü, 429 hız sınırı mı, başka mı)."""
    client = get_client()
    resp = client.models.list()
    live_ids = [m.id for m in resp.data]
    matches = _match_families(live_ids)

    verified = []
    diagnostics = {}
    if matches:
        # Cok fazla ayni anda tetiklenirse NVIDIA'nin hiz sinirina takilma
        # ihtimalini azaltmak icin es zamanliligi sinirla.
        with ThreadPoolExecutor(max_workers=min(5, len(matches))) as pool:
            future_to_match = {pool.submit(_probe_model, client, m["id"]): m for m in matches}
            for future in as_completed(future_to_match):
                m = future_to_match[future]
                ok, err = future.result()
                if ok:
                    verified.append(m)
                else:
                    diagnostics[m["id"]] = err
    verified.sort(key=lambda m: m["id"])
    final = verified or matches  # hiçbiri doğrulanamazsa en azından eşleşenleri göster

    payload = {
        "fetched_at": time.time(),
        "models": final,
        "is_fallback": False,
        "verified": bool(verified),
        "raw_count": len(live_ids),
        "diagnostics": diagnostics,
    }
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def load_catalog() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"fetched_at": None, "models": FALLBACK_CATALOG, "is_fallback": True}
