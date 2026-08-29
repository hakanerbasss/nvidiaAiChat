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


def _probe_model(client, model_id: str) -> bool:
    """Modelin hesapta gerçekten çağrılabilir olup olmadığını tek, ucuz bir
    istekle doğrular (404 -> deploy edilmemiş, sessizce elenir). Kısa bir
    timeout ile: bu fonksiyon paralel çağrılıyor, tek bir yavaş model tüm
    yenilemeyi Cloudflare'ın ~100sn kenar zaman aşımının üstüne çıkarmasın."""
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "merhaba"}],
            max_tokens=1,
            timeout=12,
        )
        return True
    except Exception:
        return False


def refresh_catalog() -> dict:
    """NVIDIA'nın gerçek /v1/models listesini çeker, bilinen ailelerle eşleştirir
    ve her adayı PARALEL olarak gerçekten çağırarak doğrular (sıralı denemek
    10 model x 12-20sn ile Cloudflare'ın kenar zaman aşımını aşıyordu). Sadece
    çalışan modeller kaydedilir."""
    client = get_client()
    resp = client.models.list()
    live_ids = [m.id for m in resp.data]
    matches = _match_families(live_ids)

    verified = []
    if matches:
        with ThreadPoolExecutor(max_workers=len(matches)) as pool:
            future_to_match = {pool.submit(_probe_model, client, m["id"]): m for m in matches}
            for future in as_completed(future_to_match):
                if future.result():
                    verified.append(future_to_match[future])
    verified.sort(key=lambda m: m["id"])
    final = verified or matches  # hiçbiri doğrulanamazsa en azından eşleşenleri göster

    payload = {
        "fetched_at": time.time(),
        "models": final,
        "is_fallback": False,
        "verified": bool(verified),
        "raw_count": len(live_ids),
    }
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def load_catalog() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"fetched_at": None, "models": FALLBACK_CATALOG, "is_fallback": True}
