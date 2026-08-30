import base64
import io
import json
import threading
import uuid
from pathlib import Path

import httpx2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from catalog import load_catalog, refresh_catalog
from nvidia_client import NVIDIA_BASE_URL, get_client, load_settings, save_settings
from router import choose_candidates
from tools import TOOLS, execute_tool

BASE_DIR = Path(__file__).resolve().parent
MAX_TEXT_FILE_BYTES = 150_000
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".csv", ".log",
    ".html", ".css", ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".sh",
}

SYSTEM_PROMPT = (
    "Sen NVIDIA NIM modelleri üzerinde çalışan çok modelli bir yapay zeka asistanısın. "
    "Sorunun türüne göre (hız, kodlama, akıl yürütme, ajan/araç kullanımı, görsel analizi) "
    "farklı bir modele yönlendiriliyorsun. Kısa, net ve doğru cevap ver. Kod isteniyorsa "
    "çalışır, eksiksiz kod bloğu üret. Kullanıcı başka bir dilde yazarsa o dilde cevap ver."
)

app = FastAPI(title="NVIDIA AI Chat")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# --- Basit arka plan iş kuyruğu (tek process, bellek içi) ---
#
# İlk sürümde /api/chat ve /api/models/refresh uzun süre açık kalan bir
# bağlantı üzerinden token/nabız akıtıyordu (SSE). Cloudflare + nginx
# zincirinde bu, uzun sessiz aralıklarda güvenilmez çıktı: nabızlar bazen
# istemciye hiç ulaşmıyor, bağlantı saatlerce "sessiz" görünüp donuyordu.
# Çözüm: uzun işlemi arka planda bir thread'de başlat, istemciye hemen bir
# job_id dön; istemci /api/jobs/{id}'yi birkaç saniyede bir sorar. Her
# sorgu kısa, tamamlanmış, bağımsız bir istek olduğu için arada hiçbir şey
# buffer'lanamaz/kaybolamaz — token-token canlı akışı kaybediyoruz ama
# gerçekten çalışan bir sistem kazanıyoruz.
_jobs: dict = {}
_jobs_lock = threading.Lock()


def start_job(fn) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "result": None, "error": None}

    def runner():
        try:
            result = fn()
            with _jobs_lock:
                _jobs[job_id] = {"status": "done", "result": result, "error": None}
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "result": None, "error": str(e)}

    threading.Thread(target=runner, daemon=True).start()
    return job_id


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return job


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/status")
def status():
    configured = bool(load_settings().get("nvidia_api_key"))
    return {"configured": configured, "base_url": NVIDIA_BASE_URL}


class SettingsIn(BaseModel):
    api_key: str


@app.post("/api/settings")
def update_settings(body: SettingsIn):
    key = body.api_key.strip()
    if not key:
        return JSONResponse({"error": "API key boş olamaz."}, status_code=400)
    save_settings(key)
    return {"ok": True}


@app.get("/api/models")
def get_models():
    return load_catalog()


@app.post("/api/models/refresh")
def post_refresh_models():
    return {"job_id": start_job(refresh_catalog)}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    raw = await file.read()
    name = file.filename or "dosya"
    content_type = file.content_type or ""
    ext = Path(name).suffix.lower()

    if content_type.startswith("image/"):
        b64 = base64.b64encode(raw).decode()
        return {"kind": "image", "name": name, "data_url": f"data:{content_type};base64,{b64}"}

    if content_type == "application/pdf" or ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return {"kind": "text", "name": name, "content": text[:MAX_TEXT_FILE_BYTES]}
        except Exception as e:
            return JSONResponse({"error": f"PDF okunamadı: {e}"}, status_code=400)

    if ext in TEXT_EXTENSIONS or content_type.startswith("text/"):
        text = raw[:MAX_TEXT_FILE_BYTES].decode("utf-8", errors="replace")
        return {"kind": "text", "name": name, "content": text}

    return JSONResponse(
        {"error": f"Desteklenmeyen dosya türü: {content_type or ext}"}, status_code=400
    )


class ChatIn(BaseModel):
    message: str
    history: list = []
    model: str = "auto"
    agent_mode: bool = False
    attachments: list = []


def build_user_content(message: str, attachments: list):
    """(content, has_image) döndürür. Görsel varsa OpenAI vision formatında
    bir parça listesi, yoksa düz metin üretir; metin dosyaları mesaja eklenir."""
    has_image = False
    image_parts = []
    text_context = []
    for att in attachments:
        if att.get("kind") == "image":
            has_image = True
            image_parts.append({"type": "image_url", "image_url": {"url": att["data_url"]}})
        elif att.get("kind") == "text":
            text_context.append(f"\n\n[Dosya: {att.get('name', 'dosya')}]\n{att.get('content', '')}")

    full_text = (message or "") + "".join(text_context)
    if has_image:
        return [{"type": "text", "text": full_text}, *image_parts], True
    return full_text, False


SINGLE_CALL_TIMEOUT = httpx2.Timeout(10.0, read=50.0, write=10.0, pool=10.0)


def is_retryable_failure(e: Exception) -> bool:
    """NVIDIA bazı katalog modellerini hesapta gerçek bir function olarak
    deploy etmemiş oluyor (404) ya da model uzun süre hiç içerik üretmeden
    zaman aşımına uğratıyor — ikisi de o modele özgü, diğer adayları
    denemeye devam etmek mantıklı. 401/429 gibi hatalar ise hesap genelinde
    geçerli olduğundan hemen durup kullanıcıya gösteriyoruz."""
    if getattr(e, "status_code", None) == 404:
        return True
    if isinstance(e, (httpx2.TimeoutException, TimeoutError)):
        return True
    text = str(e).lower()
    return "error code: 404" in text or "'status': 404" in text or "timeout" in text or "timed out" in text


def run_chat_job(client, candidates, messages, agent_mode, chosen_tag, label_by_id) -> dict:
    """Aday modelleri sırayla dener (404/zaman aşımında sıradakine geçer),
    ilk başarılı cevabı {answer, model, label, tag, tools_used} olarak döner.
    Bu, arka plan thread'inde çalışıyor — burada istediğimiz kadar
    bekleyebiliriz, istemci ayrı isteklerle durumu soruyor."""
    last_err = None
    for candidate in candidates:
        try:
            if agent_mode:
                resp = client.chat.completions.create(
                    model=candidate, messages=messages, tools=TOOLS, tool_choice="auto",
                    timeout=SINGLE_CALL_TIMEOUT,
                )
                choice = resp.choices[0]
                tool_calls = choice.message.tool_calls or []
                tools_used = [c.function.name for c in tool_calls]
                answer = choice.message.content or ""

                if tool_calls:
                    messages.append(choice.message.model_dump())
                    for call in tool_calls:
                        args = json.loads(call.function.arguments or "{}")
                        tool_result = execute_tool(call.function.name, args)
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})
                    final = client.chat.completions.create(
                        model=candidate, messages=messages, timeout=SINGLE_CALL_TIMEOUT
                    )
                    answer = final.choices[0].message.content or ""

                return {
                    "answer": answer, "model": candidate, "label": label_by_id.get(candidate, candidate),
                    "tag": chosen_tag, "tools_used": tools_used,
                }

            resp = client.chat.completions.create(
                model=candidate, messages=messages, timeout=SINGLE_CALL_TIMEOUT
            )
            answer = resp.choices[0].message.content or ""
            return {
                "answer": answer, "model": candidate, "label": label_by_id.get(candidate, candidate),
                "tag": chosen_tag, "tools_used": [],
            }
        except Exception as e:
            last_err = e
            if not is_retryable_failure(e):
                break

    raise RuntimeError(f"Denenen modellerin hiçbiri yanıt üretmedi ({len(candidates)} model): {last_err}")


@app.post("/api/chat")
def chat(body: ChatIn):
    catalog = load_catalog().get("models", [])
    label_by_id = {m["id"]: m.get("label", m["id"]) for m in catalog}
    user_content, has_image = build_user_content(body.message, body.attachments)
    candidates, chosen_tag = choose_candidates(catalog, body.message, has_image, body.agent_mode, body.model)

    if not candidates:
        return JSONResponse(
            {"error": "Katalogda hiç model yok. Önce Ayarlar'dan API key gir."}, status_code=400
        )

    try:
        client = get_client()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *body.history,
                {"role": "user", "content": user_content}]

    job_id = start_job(lambda: run_chat_job(
        client, candidates, messages, body.agent_mode, chosen_tag, label_by_id
    ))
    return {"job_id": job_id}
