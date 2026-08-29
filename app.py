import asyncio
import base64
import io
import json
import queue
import threading
import time
from pathlib import Path

import httpx2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
    try:
        refresh_catalog()
        refreshed = True
    except Exception:
        refreshed = False
    return {"ok": True, "catalog_refreshed": refreshed}


@app.get("/api/models")
def get_models():
    return load_catalog()


@app.post("/api/models/refresh")
def post_refresh_models():
    try:
        return {"ok": True, **refresh_catalog()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


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


def sse(event: str, data: dict) -> str:
    payload = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    if event:
        return f"event: {event}\n{payload}"
    return payload


REQUEST_TIMEOUT = httpx2.Timeout(10.0, read=30.0, write=10.0, pool=10.0)
FIRST_TOKEN_GRACE = 25  # saniye — bu sürede ilk gerçek içerik gelmezse model "bozuk" sayılır


def is_retryable_failure(e: Exception) -> bool:
    """NVIDIA bazı katalog modellerini hesapta gerçek bir function olarak
    deploy etmemiş oluyor (404) ya da model sessizce hiç içerik üretmeden
    bağlantıyı zaman aşımına uğratıyor — ikisi de o modele özgü, diğer
    adayları denemeye devam etmek mantıklı. 401/429 gibi hatalar ise hesap
    genelinde geçerli olduğundan hemen durup kullanıcıya gösteriyoruz."""
    if getattr(e, "status_code", None) == 404:
        return True
    if isinstance(e, (httpx2.TimeoutException, TimeoutError)):
        return True
    text = str(e).lower()
    return "error code: 404" in text or "'status': 404" in text or "timeout" in text or "timed out" in text


KEEPALIVE_SECONDS = 15  # Cloudflare'in ~100sn kenar zaman aşımını asla riske atmayacak sıklık


def _blocking_worker(fn, q: "queue.Queue"):
    try:
        q.put(("end", fn()))
    except Exception as e:
        q.put(("error", e))


def _stream_worker(fn, q: "queue.Queue"):
    """fn() bir OpenAI stream objesi döner; her parçayı kuyruğa koyar."""
    try:
        for chunk in fn():
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                q.put(("delta", delta))
        q.put(("end", None))
    except Exception as e:
        q.put(("error", e))


async def _drain_with_keepalive(q: "queue.Queue"):
    """Kuyruktaki öğeleri async olarak sırayla verir; kuyruk KEEPALIVE_SECONDS
    boyunca boşsa ("ping", None) döner ki Cloudflare/nginx bağlantıyı sessiz
    sanıp erken kesmesin. ("end"|"error", ...) gelince durur."""
    while True:
        try:
            kind, payload = await asyncio.to_thread(q.get, True, KEEPALIVE_SECONDS)
        except queue.Empty:
            yield "ping", None
            continue
        yield kind, payload
        if kind in ("end", "error"):
            return


async def call_with_keepalive(fn):
    """fn'yi arka planda bir thread'de çalıştırır; bekleme sırasında
    ("ping", None) üretir, sonunda tam olarak bir kez ("end", result) veya
    ("error", exc) üretir."""
    q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_blocking_worker, args=(fn, q), daemon=True).start()
    async for kind, payload in _drain_with_keepalive(q):
        yield kind, payload


@app.post("/api/chat")
async def chat(body: ChatIn):
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

    if body.agent_mode:
        async def agent_stream():
            model_id, first, last_err = None, None, None
            for candidate in candidates:
                call = (lambda c=candidate: client.chat.completions.create(
                    model=c, messages=messages, tools=TOOLS, tool_choice="auto", timeout=REQUEST_TIMEOUT
                ))
                result, err = None, None
                async for kind, payload in call_with_keepalive(call):
                    if kind == "ping":
                        yield sse(None, {})
                        continue
                    if kind == "end":
                        result = payload
                    else:
                        err = payload
                if result is not None:
                    model_id, first = candidate, result
                    break
                last_err = err
                if not is_retryable_failure(err):
                    break

            if model_id is None:
                yield sse("meta", {"tag": chosen_tag, "tried": candidates})
                yield sse("error", {"error": str(last_err)})
                return

            choice = first.choices[0]
            tool_calls = choice.message.tool_calls or []
            tools_used = [c.function.name for c in tool_calls]

            answer = choice.message.content or ""
            if tool_calls:
                messages.append(choice.message.model_dump())
                for call in tool_calls:
                    args = json.loads(call.function.arguments or "{}")
                    tool_result = execute_tool(call.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})

                final, ferr = None, None
                async for kind, payload in call_with_keepalive(
                    lambda: client.chat.completions.create(model=model_id, messages=messages, timeout=REQUEST_TIMEOUT)
                ):
                    if kind == "ping":
                        yield sse(None, {})
                        continue
                    if kind == "end":
                        final = payload
                    else:
                        ferr = payload
                if final is None:
                    yield sse("meta", {"model": model_id, "label": label_by_id.get(model_id, model_id), "tag": chosen_tag})
                    yield sse("error", {"error": str(ferr)})
                    return
                answer = final.choices[0].message.content or ""

            yield sse("meta", {
                "model": model_id, "label": label_by_id.get(model_id, model_id),
                "tag": chosen_tag, "tools_used": tools_used,
            })
            yield sse(None, {"delta": answer})
            yield sse("done", {})

        return StreamingResponse(agent_stream(), media_type="text/event-stream")

    async def token_stream():
        last_err = None
        for candidate in candidates:
            call = (lambda c=candidate: client.chat.completions.create(
                model=c, messages=messages, stream=True, timeout=REQUEST_TIMEOUT
            ))
            q: "queue.Queue" = queue.Queue()
            threading.Thread(target=_stream_worker, args=(call, q), daemon=True).start()

            got_content = False
            start = time.monotonic()
            candidate_err = None
            async for kind, payload in _drain_with_keepalive(q):
                if kind == "ping":
                    if not got_content and time.monotonic() - start > FIRST_TOKEN_GRACE:
                        candidate_err = RuntimeError(f"{candidate}: {FIRST_TOKEN_GRACE}s içinde içerik gelmedi")
                        break
                    yield sse(None, {})
                    continue
                if kind == "delta":
                    if not got_content:
                        got_content = True
                        yield sse("meta", {
                            "model": candidate, "label": label_by_id.get(candidate, candidate),
                            "tag": chosen_tag,
                        })
                    yield sse(None, {"delta": payload})
                elif kind == "error":
                    candidate_err = payload
                    if got_content:
                        yield sse("error", {"error": str(payload)})
                        return
                    break
                elif kind == "end":
                    break

            if got_content:
                yield sse("done", {})
                return
            last_err = candidate_err or last_err
            # bu aday hic icerik uretmedi (404, zaman asimi, sessiz baglanti) -> sonrakini dene

        yield sse("meta", {"tag": chosen_tag, "tried": candidates})
        yield sse("error", {"error": f"Denenen modellerin hiçbiri yanıt üretmedi: {last_err}"})

    return StreamingResponse(token_stream(), media_type="text/event-stream")
