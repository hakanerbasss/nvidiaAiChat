const state = {
  models: [],
  attachments: [],
  conversations: [], // {id, title, updatedAt, messages: [{role, content, meta?}]}
  activeId: null,
};

const el = (id) => document.getElementById(id);
const messagesEl = el("messages");
const modelSelect = el("modelSelect");
const catalogList = el("catalogList");
const statusText = el("statusText");
const attachmentTray = el("attachmentTray");
const conversationListEl = el("conversationList");
const mobileTitleEl = el("mobileTitle");

const STORAGE_KEY = "nvidiaAiChat.conversations";
const THEME_KEY = "nvidiaAiChat.theme";

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Basit markdown-lite: kod bloğu, satır içi kod, kalın yazı.
function renderMarkdown(text) {
  const blocks = [];
  let withPlaceholders = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = blocks.length;
    blocks.push(`<pre><code class="lang-${escapeHtml(lang)}">${escapeHtml(code)}</code></pre>`);
    return ` BLOCK${idx} `;
  });

  withPlaceholders = escapeHtml(withPlaceholders)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");

  return withPlaceholders.replace(/ BLOCK(\d+) /g, (_, i) => blocks[Number(i)]);
}

function formatTime(ts) {
  return new Date(ts || Date.now()).toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
}

function addMessage(role, text, metaText, timestamp) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const timeStr = formatTime(timestamp);
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = renderMarkdown(text);
  wrap.appendChild(bubble);

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.textContent = metaText ? `${timeStr} · ${metaText}` : timeStr;
  wrap.appendChild(meta);

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return {
    wrap,
    bubble,
    setMeta: (t) => { meta.textContent = t ? `${timeStr} · ${t}` : timeStr; },
  };
}

function addTypingMessage() {
  const msg = addMessage("assistant", "", null, Date.now());
  msg.bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  return msg;
}

function addRetryButton(wrap, onRetry) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ghost-btn retry-btn";
  btn.textContent = "🔁 Tekrar dene";
  btn.addEventListener("click", () => { btn.remove(); onRetry(); });
  wrap.appendChild(btn);
}

// Uzun süren işler (katalog testi, model cevabı) arka planda bir thread'de
// çalışıp job_id dönüyor; burada kısa, bağımsız isteklerle durumunu
// soruyoruz. Önceki sürüm uzun süre açık kalan tek bir bağlantı üzerinden
// "nabız" akıtıyordu ama Cloudflare/nginx zincirinde bu güvenilmez çıktı
// (sessiz aralıklar bazen hiç istemciye ulaşmıyordu). Kısa, tamamlanmış
// istekler bu sorunu tamamen ortadan kaldırıyor — üstelik telefon sekmeyi
// arka plana atsa bile iş sunucuda çalışmaya devam ediyor, sekmeye
// dönüldüğünde kaldığı yerden soruluyor.
async function pollJob(jobId, { intervalMs = 1500, timeoutMs = 280000, onTick } = {}) {
  const start = Date.now();
  while (true) {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const job = await res.json();
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error || "bilinmeyen hata");
    if (job.status === "not_found") throw new Error("iş bulunamadı (sunucu yeniden başlamış olabilir), tekrar dene");
    if (Date.now() - start > timeoutMs) throw new Error("zaman aşımı — sunucu çok uzun süredir yanıt vermiyor");
    if (onTick) onTick(job);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

// --- Sohbetler (localStorage'da, tarayıcıya özel) ---
function activeConversation() {
  return state.conversations.find((c) => c.id === state.activeId) || null;
}

function saveConversations() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations));
  } catch (e) {
    // localStorage kapalı/dolu olabilir — sessizce devam et, sadece bu oturumda hatırlanmaz.
  }
}

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    state.conversations = raw ? JSON.parse(raw) : [];
  } catch (e) {
    state.conversations = [];
  }
  if (state.conversations.length === 0) {
    state.conversations.push(newConversationObject());
  }
  state.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  state.activeId = state.conversations[0].id;
}

function newConversationObject() {
  return {
    id: (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random())),
    title: "Yeni sohbet",
    updatedAt: Date.now(),
    messages: [],
  };
}

function createConversation() {
  const conv = newConversationObject();
  state.conversations.unshift(conv);
  state.activeId = conv.id;
  saveConversations();
  renderConversationList();
  renderActiveMessages();
  closeSidebar();
  messageInput.focus();
}

function switchConversation(id) {
  if (id === state.activeId) return;
  state.activeId = id;
  renderConversationList();
  renderActiveMessages();
  closeSidebar();
}

function renameConversation(id, newTitle) {
  const conv = state.conversations.find((c) => c.id === id);
  if (!conv) return;
  conv.title = newTitle.trim() || conv.title;
  saveConversations();
  renderConversationList();
}

function deleteConversation(id) {
  const idx = state.conversations.findIndex((c) => c.id === id);
  if (idx < 0) return;
  state.conversations.splice(idx, 1);
  if (state.conversations.length === 0) state.conversations.push(newConversationObject());
  if (state.activeId === id) state.activeId = state.conversations[0].id;
  saveConversations();
  renderConversationList();
  renderActiveMessages();
}

function renderConversationList() {
  conversationListEl.innerHTML = "";
  for (const conv of state.conversations) {
    const item = document.createElement("div");
    item.className = "conversation-item" + (conv.id === state.activeId ? " active" : "");

    const title = document.createElement("span");
    title.className = "title";
    title.textContent = conv.title;
    title.addEventListener("click", () => switchConversation(conv.id));

    const actions = document.createElement("div");
    actions.className = "conv-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.textContent = "✎";
    renameBtn.title = "Yeniden adlandır";
    renameBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const input = document.createElement("input");
      input.className = "title-edit";
      input.value = conv.title;
      item.replaceChild(input, title);
      input.focus();
      input.select();
      const commit = () => { renameConversation(conv.id, input.value); };
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); commit(); }
        if (e.key === "Escape") { renderConversationList(); }
      });
      input.addEventListener("blur", commit);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.textContent = "✕";
    deleteBtn.title = "Sohbeti sil";
    deleteBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (confirm(`"${conv.title}" silinsin mi?`)) deleteConversation(conv.id);
    });

    actions.appendChild(renameBtn);
    actions.appendChild(deleteBtn);
    item.appendChild(title);
    item.appendChild(actions);
    conversationListEl.appendChild(item);
  }
}

function renderActiveMessages() {
  const conv = activeConversation();
  messagesEl.innerHTML = "";
  if (conv) {
    for (const m of conv.messages) addMessage(m.role, m.content, m.meta, m.at);
    mobileTitleEl.textContent = conv.title;
  }
}

// --- Aydınlık / karanlık mod ---
function applyTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
    el("themeToggle").textContent = "☀️";
  } else {
    document.documentElement.removeAttribute("data-theme");
    el("themeToggle").textContent = "🌙";
  }
}

function initTheme() {
  let saved = "dark";
  try { saved = localStorage.getItem(THEME_KEY) || "dark"; } catch (e) { /* yoksay */ }
  applyTheme(saved);
}

el("themeToggle").addEventListener("click", () => {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  const next = isLight ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* yoksay */ }
});

async function loadStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  statusText.textContent = data.configured ? "bağlı" : "API key gerekli";
  if (!data.configured) openSettingsModal();
  return data;
}

async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  state.models = data.models || [];

  modelSelect.innerHTML = "";
  const autoOpt = document.createElement("option");
  autoOpt.value = "auto";
  autoOpt.textContent = "Otomatik (soruya göre seç)";
  modelSelect.appendChild(autoOpt);
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label || m.id;
    modelSelect.appendChild(opt);
  }

  catalogList.innerHTML = "";
  for (const m of state.models) {
    const item = document.createElement("div");
    item.className = "catalog-item";
    item.innerHTML = `<div class="label">${escapeHtml(m.label || m.id)}</div>` +
      `<div class="id">${escapeHtml(m.id)}</div>` +
      `<div class="tags">${m.tags.map((t) => `<span class="tag-pill">${escapeHtml(t)}</span>`).join("")}</div>`;
    catalogList.appendChild(item);
  }
  if (data.is_fallback) {
    const note = document.createElement("div");
    note.className = "muted";
    note.textContent = "Örnek liste — API key girip kataloğu yenile.";
    catalogList.appendChild(note);
  } else if (data.verified === false) {
    const note = document.createElement("div");
    note.className = "muted error-text";
    note.textContent = "Hiçbir model doğrulanamadı, liste test edilmemiş — API key'i kontrol et.";
    catalogList.appendChild(note);
  }
}

async function refreshCatalog(onTick) {
  const res = await fetch("/api/models/refresh", { method: "POST" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const { job_id } = await res.json();
  return await pollJob(job_id, { intervalMs: 2000, timeoutMs: 320000, onTick });
}

el("refreshModels").addEventListener("click", async () => {
  el("refreshModels").disabled = true;
  let dots = 0;
  const tick = () => {
    dots = (dots + 1) % 4;
    el("refreshModels").textContent = "Modeller test ediliyor" + ".".repeat(dots);
  };
  tick();
  try {
    await refreshCatalog(tick);
    await loadModels();
  } catch (e) {
    alert("Katalog yenilenemedi: " + e.message);
  } finally {
    el("refreshModels").disabled = false;
    el("refreshModels").textContent = "Kataloğu yenile";
  }
});

function openSettingsModal() { el("settingsModal").classList.remove("hidden"); }
function closeSettingsModal() { el("settingsModal").classList.add("hidden"); }
el("openSettings").addEventListener("click", openSettingsModal);
el("closeSettings").addEventListener("click", closeSettingsModal);

el("saveSettings").addEventListener("click", async () => {
  const key = el("apiKeyInput").value.trim();
  const msg = el("settingsMsg");
  if (!key) { msg.textContent = "API key gir."; msg.className = "error-text"; return; }
  msg.textContent = "Kaydediliyor…";
  msg.className = "muted";
  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Hata");
    el("apiKeyInput").value = "";
    await loadStatus();

    msg.textContent = "Kaydedildi, modeller test ediliyor (birkaç dakika sürebilir)…";
    try {
      await refreshCatalog();
      msg.textContent = "Kaydedildi, katalog yenilendi.";
    } catch (e) {
      msg.textContent = "Kaydedildi (katalog yenilenemedi: " + e.message + ")";
    }
    await loadModels();
    setTimeout(closeSettingsModal, 1500);
  } catch (e) {
    msg.textContent = "Kaydedilemedi: " + e.message;
    msg.className = "error-text";
  }
});

// --- Dosya / görsel ekleme ---
el("attachBtn").addEventListener("click", () => el("fileInput").click());
el("fileInput").addEventListener("change", async (ev) => {
  for (const file of ev.target.files) {
    const chip = renderAttachmentChip(file.name, "yükleniyor…");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Hata");
      state.attachments.push(data);
      chip.update(data.kind === "image" ? "görsel" : "dosya");
    } catch (e) {
      chip.update("hata: " + e.message, true);
    }
  }
  ev.target.value = "";
});

function renderAttachmentChip(name, statusLabel) {
  const chip = document.createElement("div");
  chip.className = "attachment-chip";
  const label = document.createElement("span");
  label.textContent = `${name} · ${statusLabel}`;
  const removeBtn = document.createElement("button");
  removeBtn.textContent = "✕";
  chip.appendChild(label);
  chip.appendChild(removeBtn);
  attachmentTray.appendChild(chip);

  const api = {
    update(newStatus, isError) {
      label.textContent = `${name} · ${newStatus}`;
      if (isError) chip.style.borderColor = "var(--danger)";
    },
  };
  removeBtn.addEventListener("click", () => {
    const idx = state.attachments.findIndex((a) => a.name === name);
    if (idx >= 0) state.attachments.splice(idx, 1);
    chip.remove();
  });
  return api;
}

// --- Sohbet gönderimi ---
const messageInput = el("messageInput");
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
});
messageInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    el("chatForm").requestSubmit();
  }
});

// conv.messages'a kullanıcı mesajını EKLEMEDEN sadece asistan cevabını ister —
// böylece "Tekrar dene" aynı fonksiyonu çağırdığında kullanıcı mesajı
// geçmişte ikilenmez, sadece asistan yanıtı yeniden denenir. Cevap artık
// token-token akmıyor (bkz. pollJob yorumu) — hazır olunca tek seferde geliyor.
async function requestAssistantReply(conv, text, attachmentsForRequest) {
  const sendBtn = document.querySelector(".send-btn");
  sendBtn.disabled = true;

  const priorHistory = conv.messages.map((m) => ({ role: m.role, content: m.content }));
  const assistant = addTypingMessage();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: priorHistory,
        model: modelSelect.value,
        agent_mode: el("agentMode").checked,
        attachments: attachmentsForRequest,
      }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    const { job_id } = await res.json();
    const result = await pollJob(job_id, { intervalMs: 1500, timeoutMs: 280000 });

    const toolNote = result.tools_used && result.tools_used.length
      ? ` · araç: ${result.tools_used.join(", ")}`
      : "";
    const metaText = `${result.label || result.model} (${result.tag})${toolNote}`;
    const answer = result.answer || "[boş cevap geldi — model muhtemelen bir şey döndürmedi]";

    assistant.bubble.innerHTML = renderMarkdown(answer);
    assistant.setMeta(metaText);
    sendBtn.disabled = false;

    conv.messages.push({ role: "user", content: text, at: Date.now() });
    conv.messages.push({ role: "assistant", content: answer, meta: metaText, at: Date.now() });
    conv.updatedAt = Date.now();
    if (conv.title === "Yeni sohbet" && text) {
      conv.title = text.length > 40 ? text.slice(0, 40) + "…" : text;
      mobileTitleEl.textContent = conv.title;
    }
    saveConversations();
    renderConversationList();
  } catch (e) {
    assistant.bubble.innerHTML = renderMarkdown(`[istek başarısız: ${e.message}]`);
    addRetryButton(assistant.wrap, () => {
      assistant.wrap.remove();
      requestAssistantReply(conv, text, attachmentsForRequest);
    });
    sendBtn.disabled = false;
    // hata durumunda geçmişe hiçbir şey eklenmedi, tekrar dene aynı turu yeniden dener
  }
}

el("chatForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = messageInput.value.trim();
  if (!text && state.attachments.length === 0) return;

  addMessage("user", text || "(dosya gönderildi)", null, Date.now());
  const attachmentsForRequest = state.attachments;
  state.attachments = [];
  attachmentTray.innerHTML = "";
  messageInput.value = "";
  messageInput.style.height = "auto";

  await requestAssistantReply(activeConversation(), text, attachmentsForRequest);
});

// --- Mobil kenar çubuğu (drawer) ---
const sidebar = el("sidebar");
const sidebarBackdrop = el("sidebarBackdrop");

function openSidebar() {
  sidebar.classList.add("open");
  sidebarBackdrop.classList.add("show");
}
function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarBackdrop.classList.remove("show");
}
el("menuToggle").addEventListener("click", openSidebar);
sidebarBackdrop.addEventListener("click", closeSidebar);

// Mobilde model seçince veya ayarları açınca menüyü otomatik kapat.
modelSelect.addEventListener("change", closeSidebar);
el("openSettings").addEventListener("click", closeSidebar);

el("newChatBtn").addEventListener("click", createConversation);

(async function init() {
  initTheme();
  loadConversations();
  renderConversationList();
  renderActiveMessages();
  await loadStatus();
  await loadModels();
})();
