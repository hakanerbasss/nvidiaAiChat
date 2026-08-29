import base64
import io
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from catalog import load_catalog, refresh_catalog
from nvidia_client import NVIDIA_BASE_URL, get_client, load_settings, save_settings
from router import choose_model
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


@app.post("/api/chat")
def chat(body: ChatIn):
    catalog = load_catalog().get("models", [])
    user_content, has_image = build_user_content(body.message, body.attachments)
    model_id, chosen_tag = choose_model(catalog, body.message, has_image, body.agent_mode, body.model)

    if not model_id:
        return JSONResponse(
            {"error": "Katalogda hiç model yok. Önce Ayarlar'dan API key gir."}, status_code=400
        )

    try:
        client = get_client()
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *body.history,
                {"role": "user", "content": user_content}]

    def error_stream(msg: str):
        yield sse("meta", {"model": model_id, "tag": chosen_tag})
        yield sse("error", {"error": msg})

    if body.agent_mode:
        try:
            first = client.chat.completions.create(
                model=model_id, messages=messages, tools=TOOLS, tool_choice="auto"
            )
        except Exception as e:
            return StreamingResponse(error_stream(str(e)), media_type="text/event-stream")

        choice = first.choices[0]
        tool_calls = choice.message.tool_calls or []
        tools_used = [c.function.name for c in tool_calls]

        if tool_calls:
            messages.append(choice.message.model_dump())
            for call in tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = execute_tool(call.function.name, args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            try:
                final = client.chat.completions.create(model=model_id, messages=messages)
                answer = final.choices[0].message.content or ""
            except Exception as e:
                return StreamingResponse(error_stream(str(e)), media_type="text/event-stream")
        else:
            answer = choice.message.content or ""

        def agent_stream():
            yield sse("meta", {"model": model_id, "tag": chosen_tag, "tools_used": tools_used})
            yield sse(None, {"delta": answer})
            yield sse("done", {})

        return StreamingResponse(agent_stream(), media_type="text/event-stream")

    def token_stream():
        yield sse("meta", {"model": model_id, "tag": chosen_tag})
        try:
            stream = client.chat.completions.create(model=model_id, messages=messages, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield sse(None, {"delta": delta})
        except Exception as e:
            yield sse("error", {"error": str(e)})
            return
        yield sse("done", {})

    return StreamingResponse(token_stream(), media_type="text/event-stream")
