from datetime import datetime
from zoneinfo import ZoneInfo

# Ajan modu açıkken modele tanıtılan araç listesi. Yeni bir araç eklemek için
# buraya bir "function" tanımı ekle ve execute_tool içine karşılığını yaz.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Şu anki tarih ve saati (Türkiye saati, Europe/Istanbul) döndürür.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    if name == "get_current_datetime":
        return datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"Bilinmeyen araç çağrıldı: {name}"
