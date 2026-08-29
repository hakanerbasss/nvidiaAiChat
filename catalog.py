import json
import time

from nvidia_client import CACHE_PATH, get_client

# Model ailesi -> yetenek etiketi eşlemesi. Kaynak: kullanıcının build.nvidia.com
# panelinden aldığı kategori önerileri (hız/DeepSeek, kodlama/MiniMax,
# akıl yürütme/Qwen, ajan/Kimi). Bilinmeyen aileler "general" etiketiyle döner.
FAMILY_TAGS = {
    "deepseek": ["fast"],
    "minimax": ["code"],
    "minimaxai": ["code"],
    "qwen": ["reasoning"],
    "kimi": ["agent"],
    "moonshotai": ["agent"],
    "glm": ["general"],
    "zhipuai": ["general"],
    "llama": ["general"],
    "meta": ["general"],
    "nemotron": ["general"],
    "nvidia": ["general"],
    "mistral": ["general"],
    "mixtral": ["general"],
    "phi": ["general"],
    "microsoft": ["general"],
    "gemma": ["general"],
    "google": ["general"],
}

VISION_HINTS = [
    "vision", "-vl", "vl-", "vila", "neva", "kosmos", "florence", "paligemma", "llava",
]

# API key henüz girilmemişken veya /v1/models erişilemezken gösterilecek örnek
# liste. Kullanıcının kendi ekran görüntüsündeki gerçek model kimlikleri.
FALLBACK_CATALOG = [
    {"id": "deepseek-ai/deepseek-v4-flash", "tags": ["fast"]},
    {"id": "minimaxai/minimax-m3", "tags": ["code"]},
    {"id": "qwen/qwen3.5-397b-a17b", "tags": ["reasoning"]},
    {"id": "moonshotai/kimi-k2.6", "tags": ["agent"]},
    {"id": "zhipuai/glm-5.1", "tags": ["general"]},
]


def tag_model(model_id: str) -> list:
    lower = model_id.lower()
    tags = set()
    for key, fam_tags in FAMILY_TAGS.items():
        if key in lower:
            tags.update(fam_tags)
    if any(hint in lower for hint in VISION_HINTS):
        tags.add("vision")
    if not tags:
        tags.add("general")
    return sorted(tags)


def refresh_catalog() -> dict:
    """NVIDIA'nın gerçek /v1/models listesini çeker ve etiketleyip diske yazar.

    Bu, modelleri tahmin etmek yerine kullanıcının kendi API key'iyle o an
    gerçekten erişilebilir olan katalogdan besleniyor olmayı sağlar.
    """
    client = get_client()
    resp = client.models.list()
    models = [{"id": m.id, "tags": tag_model(m.id)} for m in resp.data]
    models.sort(key=lambda m: m["id"])
    payload = {"fetched_at": time.time(), "models": models, "is_fallback": False}
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def load_catalog() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"fetched_at": None, "models": FALLBACK_CATALOG, "is_fallback": True}
