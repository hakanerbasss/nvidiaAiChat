import re

CODE_HINTS = re.compile(
    r"(kod\b|fonksiyon|python|javascript|typescript|\bhtml\b|\bcss\b|\bsql\b|\bbug\b|"
    r"hata ayıkla|refactor|script yaz|def |class |```|api yaz|uygulama yaz)",
    re.IGNORECASE,
)
REASONING_HINTS = re.compile(
    r"(analiz et|karşılaştır|\bneden\b|adım adım|plan yap|strateji|değerlendir|artı.*eksi)",
    re.IGNORECASE,
)
AGENT_HINTS = re.compile(
    r"(araç kullan|\bajan\b|workflow|otomasyon|çoklu adım)",
    re.IGNORECASE,
)


def pick_tag(message: str, has_image: bool, agent_mode: bool) -> str:
    if has_image:
        return "vision"
    if agent_mode or AGENT_HINTS.search(message or ""):
        return "agent"
    if CODE_HINTS.search(message or ""):
        return "code"
    if REASONING_HINTS.search(message or ""):
        return "reasoning"
    return "fast"


def choose_model(catalog_models: list, message: str, has_image: bool, agent_mode: bool, manual_model: str = "auto"):
    """Returns (model_id, tag_used). manual_model overrides auto-routing."""
    if manual_model and manual_model != "auto":
        return manual_model, "manuel"

    tag = pick_tag(message, has_image, agent_mode)
    candidates = [m for m in catalog_models if tag in m.get("tags", [])]

    if not candidates:
        # İstenen yetenekte model yoksa (örn. katalogda hiç vision modeli yok)
        # genel amaçlı bir modele düş, sessizce hatasız devam et.
        candidates = [m for m in catalog_models if "general" in m.get("tags", [])] or catalog_models

    chosen = candidates[0]["id"] if candidates else None
    return chosen, tag
