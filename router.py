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


def choose_candidates(
    catalog_models: list,
    message: str,
    has_image: bool,
    agent_mode: bool,
    manual_model: str = "auto",
    max_candidates: int = 5,
):
    """Returns (candidate_model_ids, tag_used).

    manual_model verilmişse sadece o modeli döner (kullanıcının açık seçimini
    sessizce değiştirmiyoruz). "auto" modda katalog artık zaten doğrulanmış,
    küçük bir küratörlü liste olduğu için sadece aynı etiketteki modelleri
    sırayla döneriz — app.py yine de 404 ihtimaline karşı sıradakini dener.
    """
    if manual_model and manual_model != "auto":
        return [manual_model], "manuel"

    tag = pick_tag(message, has_image, agent_mode)
    tagged = [m["id"] for m in catalog_models if tag in m.get("tags", [])]

    if not tagged:
        # İstenen yetenekte model yoksa (örn. katalogda hiç vision modeli yok)
        # genel amaçlı modellere düş, sessizce hatasız devam et.
        tag = "general"
        tagged = [m["id"] for m in catalog_models if "general" in m.get("tags", [])]

    if not tagged:
        # Küratörlü katalog o an çok küçükse (bazı modeller doğrulanamamış
        # olabilir) elde ne varsa kullan — ama görsel amaçlı model varsa,
        # görsel yokken onu en sona at (düz bir "merhaba" boşuna vision
        # modeline gitmesin).
        non_vision = [m["id"] for m in catalog_models if "vision" not in m.get("tags", [])]
        vision_only = [m["id"] for m in catalog_models if "vision" in m.get("tags", [])]
        tagged = non_vision + vision_only if not has_image else vision_only + non_vision

    return tagged[:max_candidates], tag
