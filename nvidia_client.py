import json
from pathlib import Path

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.json"
CACHE_PATH = BASE_DIR / "models_cache.json"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text())
    return {"nvidia_api_key": ""}


def save_settings(api_key: str) -> None:
    SETTINGS_PATH.write_text(
        json.dumps({"nvidia_api_key": api_key}, ensure_ascii=False, indent=2)
    )


def get_client() -> OpenAI:
    api_key = load_settings().get("nvidia_api_key", "")
    if not api_key:
        raise RuntimeError(
            "NVIDIA API anahtarı ayarlanmamış. Önce Ayarlar'dan build.nvidia.com API key'ini gir."
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
