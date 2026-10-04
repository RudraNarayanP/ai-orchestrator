/* OmniBrain front end.
 *
 * Two rules shape this file:
 *  1. The answer is the interface. Evidence, disagreements and raw captures sit
 *     behind expanders and are revealed when asked for.
 *  2. Provider text is untrusted input. Nothing from a model or a fetched page
 *     is ever assigned with innerHTML -- text goes in as text, and links are
 *     built from validated http(s) URLs only.
 */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};
const safeUrl = (raw) => {
  try {
    const url = new URL(String(raw));
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
};

const CONVERSATION_KEY = "omnibrain.conversationId";

const state = {
  conversationId: localStorage.getItem(CONVERSATION_KEY) || null,
  jobId: null,
  providers: new Map(),
  config: null,
  es: null,
  startedAt: 0,
};

const STATUS_GLYPH = {
  idle: "○", launching: "◐", connected: "◑", searching: "◑", responding: "◓",
  completed: "✓", logged_out: "🔑", rate_limited: "⏳", failed: "✕", timeout: "⌛", broken: "⚠",
};

async function boot() {
  state.config = await fetch("/api/config").then((r) => r.json());
  paintConfig();
  checkDoctor();
  $("#question").addEventListener("input", autosize);
  $("#askForm").addEventListener("submit", ask);
  $("#openSettings").addEventListener("click", () => toggleSettings(true));
  $("#closeSettings").addEventListener("click", () => toggleSettings(false));
  $("#saveSettings").addEventListener("click", saveSettings);
  $("#testVerifier").addEventListener("click", checkDoctor);
  $("#cancel").addEventListener("click", cancelJob);
  $("#newThread").addEventListener("click", newThread);
  $("#openHistory").addEventListener("click", () => openPanel("history"));
  $("#openMemory").addEventListener("click", () => openPanel("memory"));
  $("#openThreads").addEventListener("click", () => openPanel("threads"));
  $("#closeThreads").addEventListener("click", () => openPanel(null));
  $("#thCreate").addEventListener("click", createThread);
  $("#thBack").addEventListener("click", loadThreads);
  $("#thSend").addEventListener("click", sendThread);
  $("#thText").addEventListener("keydown", threadKeys);
  $("#thRecallBtn").addEventListener("click", recallThread);
  $("#thRecall").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); recallThread(); } });
  $("#thDelete").addEventListener("click", deleteThread);
  $("#closeMemory").addEventListener("click", () => openPanel(null));
  $("#memSearch").addEventListener("click", memSearch);
  $("#memQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); memSearch(); } });
  $("#memAdd").addEventListener("click", memAdd);
  $("#memStatus").addEventListener("change", loadMemory);
  $("#memForgetAll").addEventListener("click", memForgetAll);
  $("#memInject").addEventListener("change", () => memSetting({ inject: $("#memInject").checked }));
  $("#memCapture").addEventListener("change", () => memSetting({ capture: $("#memCapture").checked }));
  $("#memSensitive").addEventListener("change", () => memSetting({ sensitive: $("#memSensitive").checked }));
  $("#closeHistory").addEventListener("click", () => openPanel(null));
  $("#historyList").addEventListener("keydown", historyKeys);
  $("#question").addEventListener("keydown", questionKeys);
  document.addEventListener("keydown", globalKeys);
  await restoreThread();
  $("#question").focus();
}

function rememberThread(id) {
  state.conversationId = id || null;
  if (id) localStorage.setItem(CONVERSATION_KEY, id);
  else localStorage.removeItem(CONVERSATION_KEY);
}

/** Redraw the earlier turns of this thread after a reload (answers only; open a job for its evidence). */
async function restoreThread() {
  if (!state.conversationId) return;
  try {
    const res = await fetch(`/api/conversations/${encodeURIComponent(state.conversationId)}`);
    if (!res.ok) return;
    const thread = await res.json();
    for (const turn of thread.turns || []) {
      const q = el("div", "msg user");
      q.append(el("span", "q", turn.question));
      const a = el("div", "msg ai past");
      a.append(el("div", "answer", turn.answer));
      $("#stream").append(q, a);
    }
    $("#stream").scrollTop = $("#stream").scrollHeight;
  } catch {
    /* a thread that cannot be restored just starts fresh visually; the server still has it */
  }
}

function newThread() {
  state.es?.close();
  state.jobId = null;
  rememberThread(null);
  $("#stream").textContent = "";
  $("#live").hidden = true;
  $("#question").focus();
}

function paintConfig() {
  const list = $("#providerList");
  list.textContent = "";
  const providers = state.config.providers || {};
  for (const [key, cfg] of Object.entries(providers)) {
    const row = el("label", "provider-row");
    const box = el("input");
    box.type = "checkbox";
    box.checked = !!cfg.enabled;
    box.dataset.provider = key;
    const name = el("span", "provider-name", cfg.label || key);
    const login = el("button", "ghost tiny", "sign in");
    login.type = "button";
    login.title = "Open this provider's own window so you can log in once";
    login.addEventListener("click", async (e) => {
      e.preventDefault();
      login.textContent = "…";
      const res = await fetch(`/api/providers/${key}/login`, { method: "POST" }).then((r) => r.json());
      login.textContent = "opened";
      setTimeout(() => (login.textContent = "sign in"), 4000);
      note(res.instruction || "window opened");
    });
    row.append(box, name, login);
    list.append(row);
  }
  const v = state.config.verifier || {};
  $("#vProvider").value = v.provider || "ollama";
  $("#vModel").value = v.model || "";
  $("#vBaseUrl").value = v.base_url || "";
  $("#vTemp").value = v.temperature ?? 0.1;
  $("#vTokens").value = v.max_tokens ?? 4096;
  $("#vDeep").checked = !!v.deep_verification;
  $("#vInspect").checked = !!v.inspect_cited_sources;
  $("#maxRounds").value = state.config.research?.max_rounds ?? 3;
  $("#mode").value = state.config.research?.mode || "STANDARD";
}

function toggleSettings(open) {
  openPanel(open ? "settings" : null);
}

/** One side panel at a time ("settings", "history", or null). Focus moves in on open and back to the question on close. */
function openPanel(name) {
  for (const id of ["settings", "history", "memory", "threads"]) $("#" + id).hidden = id !== name;
  $("#openSettings").setAttribute("aria-expanded", String(name === "settings"));
  $("#openHistory").setAttribute("aria-expanded", String(name === "history"));
  $("#openMemory").setAttribute("aria-expanded", String(name === "memory"));
  $("#openThreads").setAttribute("aria-expanded", String(name === "threads"));
  if (name === "history") loadHistory();
  if (name === "memory") loadMemory();
  if (name === "threads") loadThreads();
  if (name) $("#" + name).querySelector("input, select, button")?.focus();
  else $("#question").focus();
}

/* ------------------------------------------------------------------ memory panel
 * Everything here is text-only (no innerHTML). The server is the source of truth; we just draw it.
 */
async function memCall(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "HTTP " + res.status);
  return body;
}

const jsonOpts = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function memCard(m, opts = {}) {
  const card = el("div", "mem");
  card.append(el("div", "mem-text", m.content));
  const bits = [m.memory_type === "interpretation" ? "your view, not a fact" : m.memory_type, m.sensitivity === "sensitive" ? "sensitive" : null, m.source === "user_explicit" ? "you said it" : "inferred", `confidence ${Math.round((m.confidence || 0) * 100)}%`, m.status.toLowerCase()];
  if (m.project) bits.push("project: " + m.project);
  bits.push("saved " + fmtTime(m.created_at));
  if (m.updated_at && Math.abs(m.updated_at - m.created_at) > 60) bits.push("updated " + fmtTime(m.updated_at));
  const chips = el("div", "chips");
  for (const b of bits.filter(Boolean)) chips.append(el("span", "chip", b));
  card.append(chips);
  if (m.why?.length) card.append(el("div", "mem-why", "why: " + m.why.join("; ")));
  if (!opts.readonly) {
    const row = el("div", "row");
    const edit = el("button", "ghost small", "edit");
    edit.type = "button";
    edit.addEventListener("click", async () => {
      const next = window.prompt("Edit this memory", m.content);
      if (next === null || !next.trim() || next.trim() === m.content) return;
      try { await memCall(`/api/memory/${encodeURIComponent(m.memory_id)}`, jsonOpts("PATCH", { content: next.trim() })); loadMemory(); } catch (e) { note("Could not edit: " + e.message); }
    });
    const del = el("button", "ghost small", "delete");
    del.type = "button";
    del.addEventListener("click", async () => {
      try { await memCall(`/api/memory/${encodeURIComponent(m.memory_id)}`, { method: "DELETE" }); loadMemory(); } catch (e) { note("Could not delete: " + e.message); }
    });
    row.append(edit, del);
    card.append(row);
  }
  return card;
}

async function loadMemory() {
  const box = $("#memList");
  box.textContent = "loading.";
  try {
    const s = await memCall("/api/memory/settings");
    $("#memInject").checked = !!s.inject;
    $("#memCapture").checked = !!s.capture;
    $("#memSensitive").checked = !!s.sensitive;
    const status = $("#memStatus").value;
    const data = await memCall("/api/memory?limit=200" + (status ? "&status=" + status : ""));
    const t = data.stats?.total ?? data.items.length;
    $("#memStats").textContent = `${t} stored - search: ${s.embedder}`;
    box.textContent = "";
    if (!data.items.length) box.append(el("p", "hint", "Nothing stored."));
    for (const m of data.items) box.append(memCard(m));
  } catch (err) {
    box.textContent = "";
    box.append(el("p", "err", "Memory is unavailable: " + err.message));
  }
}

async function memSearch() {
  const q = $("#memQuery").value.trim();
  const out = $("#memSearchOut");
  out.textContent = "";
  if (!q) return;
  try {
    const r = await memCall("/api/memory/search", jsonOpts("POST", { query: q, project: $("#project").value.trim() || undefined }));
    out.append(el("p", "hint", `${r.kind} question - up to ${r.budget} memories - ${r.hits.length} picked of ${r.candidates} candidates` + (r.reason ? " - " + r.reason : "")));
    for (const h of r.hits) out.append(memCard(h, { readonly: true }));
  } catch (err) {
    out.append(el("p", "err", err.message));
  }
}

async function memAdd() {
  const text = $("#memNew").value.trim();
  if (!text) return;
  try {
    const r = await memCall("/api/memory", jsonOpts("POST", { content: text, memory_type: $("#memNewType").value, project: $("#project").value.trim() || undefined }));
    $("#memNew").value = "";
    note(r.action === "merged" ? "Already remembered - merged." : r.action === "superseded" ? "Saved - it replaces an older memory." : "Saved.");
    loadMemory();
  } catch (err) {
    note("Could not save: " + err.message);
  }
}

async function memSetting(change) {
  try { await memCall("/api/memory/settings", jsonOpts("POST", change)); } catch (err) { note("Could not change that: " + err.message); }
}

async function memForgetAll() {
  if (!window.confirm("Delete EVERYTHING OmniBrain remembers about you? This cannot be undone.")) return;
  try { await memCall("/api/memory/forget-all", jsonOpts("POST", { confirm: true })); loadMemory(); } catch (err) { note("Could not delete: " + err.message); }
}

function fmtTime(seconds) {
  const d = new Date(Number(seconds) * 1000);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

/** The History view: every earlier question, newest first. Text only -- nothing here is trusted markup. */
async function loadHistory() {
  const box = $("#historyList");
  box.textContent = "loading.";
  try {
    const res = await fetch("/api/jobs?limit=100");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const jobs = await res.json();
    box.textContent = "";
    if (!jobs.length) {
      box.append(el("p", "hint", "Nothing asked yet."));
      return;
    }
    for (const job of jobs) {
      const item = el("button", "history-item");
      item.type = "button";
      item.dataset.jobId = job.id;
      item.append(el("span", "h-q", job.question || "(no question)"));
      const bits = [fmtTime(job.created_at), job.status, job.confidence && job.confidence !== "none" ? job.confidence : null].filter(Boolean);
      item.append(el("span", "meta", bits.join(" - ")));
      item.addEventListener("click", () => openPastJob(job.id));
      box.append(item);
    }
  } catch (err) {
    box.textContent = "";
    box.append(el("p", "err", "Could not load history: " + err.message));
  }
}

/** Show a finished job exactly as it looked when it ran (answer, evidence, raw research, export). */
async function openPastJob(jobId) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) return note("could not open that job");
  const job = await res.json();
  state.es?.close();
  $("#live").hidden = true;
  state.jobId = jobId;
  rememberThread(job.conversation_id || null);
  $("#stream").textContent = "";
  const bubble = el("div", "msg user");
  bubble.append(el("span", "q", job.question || job.live?.question || ""));
  const shell = buildAnswerShell();
  shell.jobId = jobId;
  $("#stream").append(bubble, shell.root);
  paintAnswer(shell, finalFrom(job));
  shell.sections.append(exportRow(jobId));
  await paintEvidence(shell, job);
  openPanel(null);
}

function exportRow(jobId) {
  const row = el("div", "export-row");
  row.append(el("span", "export-label", "export"));
  for (const [fmt, label] of [["md", "Markdown"], ["json", "JSON"]]) {
    const link = el("a", "export-link", label);
    link.href = `/api/jobs/${encodeURIComponent(jobId)}/export?format=${fmt}`;
    link.download = "";
    row.append(link);
  }
  return row;
}

function historyKeys(e) {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  const items = [...document.querySelectorAll("#historyList .history-item")];
  const at = items.indexOf(document.activeElement);
  if (!items.length) return;
  e.preventDefault();
  const next = e.key === "ArrowDown" ? Math.min(items.length - 1, at + 1) : Math.max(0, at - 1);
  items[next].focus();
}

/** Enter sends, Shift+Enter is a new line (and IME composition is left alone). */
function questionKeys(e) {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    ask();
  }
}

/** Escape closes the open side panel; "/" jumps to the question box. */
function globalKeys(e) {
  if (e.key === "Escape" && (!$("#settings").hidden || !$("#history").hidden || !$("#memory").hidden || !$("#threads").hidden)) {
    openPanel(null);
  } else if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "")) {
    e.preventDefault();
    $("#question").focus();
  }
}

/* ------------------------------------------------------------------ threads screen
 * One long conversation over replaceable provider chats. The server owns the thread; this only draws it.
 * Text-only (no innerHTML): provider replies are untrusted input.
 */
const thread = { id: null, busy: false };
const REASONS = { context_limit: "the previous chat was nearly full", provider_switch: "you changed provider", resume: "the thread was resumed" };

function enabledProviders() {
  const all = state.config?.providers || {};
  return Object.entries(all).filter(([name, p]) => p && p.enabled && name !== "search").map(([name, p]) => [name, p.label || name]);
}

function thStatus(text, bad = false) {
  const s = $("#thStatus");
  s.textContent = text || "";
  s.className = bad ? "err" : "hint";
}

async function loadThreads() {
  thread.id = null;
  $("#thDetail").hidden = true;
  $("#thListBox").hidden = false;
  const box = $("#thList");
  box.textContent = "loading.";
  try {
    const data = await memCall("/api/threads");
    box.textContent = "";
    if (!data.threads.length) box.append(el("p", "hint", "No threads yet. Create one above."));
    for (const t of data.threads) {
      const card = el("div", "mem th-card");
      card.append(el("div", "mem-text", t.title || "Untitled thread"));
      const chips = el("div", "chips");
      for (const b of [`${t.messages} message${t.messages === 1 ? "" : "s"}`, t.project ? "project: " + t.project : null, "last active " + fmtTime(t.updated_at)]) if (b) chips.append(el("span", "chip", b));
      card.append(chips);
      const open = el("button", "ghost small th-open", "open");
      open.type = "button";
      open.addEventListener("click", () => openThread(t.thread_id));
      card.append(open);
      box.append(card);
    }
  } catch (e) {
    box.textContent = "";
    box.append(el("p", "err", "Could not load threads: " + e.message));
  }
}

async function createThread() {
  const title = $("#thTitle").value.trim();
  const project = $("#thProject").value.trim();
  try {
    const made = await memCall("/api/threads", jsonOpts("POST", { title, project: project || null }));
    $("#thTitle").value = "";
    $("#thProject").value = "";
    await openThread(made.thread_id);
  } catch (e) {
    thStatus("Could not create the thread: " + e.message, true);
  }
}

async function openThread(id) {
  thread.id = id;
  $("#thListBox").hidden = true;
  $("#thDetail").hidden = false;
  thStatus("");
  const sel = $("#thProvider");
  if (!sel.options.length) for (const [name, label] of enabledProviders()) sel.append(new Option(label, name));
  await refreshThread();
  $("#thText").focus();
}

async function refreshThread() {
  const [view, segs] = await Promise.all([memCall(`/api/threads/${thread.id}?last=300`), memCall(`/api/threads/${thread.id}/segments`)]);
  paintThread(view, segs);
}

function paintThread(view, segs) {
  $("#thHeading").textContent = (view.title || "Untitled thread") + (view.project ? "  \u00b7  project: " + view.project : "");
  const strip = $("#thSegments");
  strip.textContent = "";
  for (const s of segs.segments) {
    const chip = el("span", "chip seg" + (s.open ? " seg-open" : ""), `${s.label} \u00b7 ${s.tokens}/${s.limit} tokens \u00b7 ${s.open ? "in use" : "closed"}`);
    chip.title = `${s.provider}; opened because: ${REASONS[s.reason] || s.reason || "start"}${s.method ? "; summary: " + s.method : ""}`;
    strip.append(chip);
  }
  if (!segs.segments.length) strip.append(el("span", "hint", "No AI chat yet \u2014 send a message."));
  const box = $("#thMessages");
  box.textContent = "";
  for (const m of view.messages) {
    const ctx = m.context;
    if (m.role === "user" && ctx && ctx.rotated) {
      box.append(el("div", "rotation", `\u21bb A new ${ctx.chat} chat was opened because ${REASONS[ctx.reason] || ctx.reason}. It was given the thread so far (${ctx.packet_tokens} tokens of context).`));
    }
    const row = el("div", "th-msg " + (m.role === "user" ? "th-user" : "th-ai"));
    const who = m.role === "user" ? "you" : (m.provider || "assistant");
    row.append(el("div", "th-who", who + " \u00b7 " + fmtTime(m.ts)));
    row.append(el("div", "th-text", m.content));
    if (m.role === "user" && ctx) {
      const bits = [`chat: ${ctx.chat}`];
      if (ctx.packet_tokens) bits.push(`thread context given: ${ctx.packet_tokens} tokens`);
      if (ctx.recalled) bits.push(`${ctx.recalled} earlier message(s) recalled`);
      const mem = ctx.memory_used || [];
      bits.push(mem.length ? `${mem.length} remembered thing(s) used` : "no personal memory used");
      const d = el("details", "th-ctx");
      d.append(el("summary", null, bits.join(" \u00b7 ")));
      if (mem.length) { const ul = el("ul", "th-mem"); for (const t of mem) ul.append(el("li", null, t)); d.append(ul); }
      row.append(d);
    }
    box.append(row);
  }
  box.scrollTop = box.scrollHeight;
  $("#thEmpty").hidden = view.messages.length > 0;
}

async function sendThread() {
  if (thread.busy || !thread.id) return;
  const text = $("#thText").value.trim();
  const provider = $("#thProvider").value;
  if (!text || !provider) { thStatus("Type a message and pick a provider.", true); return; }
  thread.busy = true;
  $("#thSend").disabled = true;
  thStatus(`Asking ${provider} \u2014 this uses its real chat window and can take a minute.`);
  try {
    const out = await memCall(`/api/threads/${thread.id}/chat`, jsonOpts("POST", { text, provider }));
    $("#thText").value = "";
    thStatus(out.rotated ? `Opened a new chat (${out.chat}).` : "");
  } catch (e) {
    thStatus("That didn't go through: " + e.message + " Your message is saved in the thread.", true);
  } finally {
    thread.busy = false;
    $("#thSend").disabled = false;
    await refreshThread().catch(() => {});
  }
}

async function recallThread() {
  const q = $("#thRecall").value.trim();
  const out = $("#thRecallOut");
  out.textContent = "";
  if (!q || !thread.id) return;
  try {
    const data = await memCall(`/api/threads/${thread.id}/recall`, jsonOpts("POST", { query: q }));
    if (!data.messages.length) out.append(el("p", "hint", "Nothing in this thread matches."));
    for (const s of data.summaries) out.append(el("div", "mem", "summary of an earlier part: " + s.text));
    for (const m of data.messages) {
      const card = el("div", "mem");
      card.append(el("div", "mem-text", m.content));
      card.append(el("div", "mem-why", `${m.role} \u00b7 ${fmtTime(m.ts)} \u00b7 found by ${m.why.join(", ")}`));
      out.append(card);
    }
  } catch (e) {
    out.append(el("p", "err", "Recall failed: " + e.message));
  }
}

async function deleteThread() {
  if (!thread.id || !window.confirm("Delete this whole thread (every message and summary)? This cannot be undone.")) return;
  try {
    await memCall(`/api/threads/${thread.id}`, { method: "DELETE" });
    await loadThreads();
  } catch (e) {
    thStatus("Could not delete: " + e.message, true);
  }
}

function threadKeys(e) {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendThread(); }
}

async function saveSettings() {  const providers = {};
  for (const box of document.querySelectorAll("#providerList input[type=checkbox]")) {
    providers[box.dataset.provider] = { enabled: box.checked };
  }
  const body = {
    providers,
    verifier: {
      provider: $("#vProvider").value,
      model: $("#vModel").value,
      base_url: $("#vBaseUrl").value,
      temperature: Number($("#vTemp").value),
      max_tokens: Number($("#vTokens").value),
      deep_verification: $("#vDeep").checked,
      inspect_cited_sources: $("#vInspect").checked,
    },
    research: { max_rounds: Number($("#maxRounds").value), mode: $("#mode").value },
  };
  const key = $("#vApiKey").value;
  if (key) body.verifier.api_key = key;
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) return note("could not save: " + (await res.text()));
  state.config = await res.json().then((r) => r.config);
  $("#vApiKey").value = "";
  note("saved");
  checkDoctor();
}

async function checkDoctor() {
  const dot = $("#doctorDot");
  dot.className = "dot checking";
  try {
    const d = await fetch("/api/doctor").then((r) => r.json());
    const ok = d.verifier?.ok;
    dot.className = "dot " + (ok ? "ok" : "warn");
    dot.title = ok
      ? `verifier ready: ${d.verifier.detail}`
      : `verifier: ${d.verifier?.state} — ${d.verifier?.detail}`;
    const models = d.verifier?.models || [];
    $("#modelList").textContent = models.length
      ? `${models.length} model(s) on that server: ` + models.slice(0, 6).join(", ")
      : d.verifier?.detail || "";
    state.lastDoctor = d;
  } catch {
    dot.className = "dot bad";
    dot.title = "backend unreachable";
  }
}

function autosize(e) {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
}

function note(text) {
  const line = el("div", "note", text);
  $("#stream").append(line);
  $("#stream").scrollTop = $("#stream").scrollHeight;
}

async function ask(event) {
  event?.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  $("#question").value = "";
  $("#question").style.height = "auto";

  const bubble = el("div", "msg user");
  bubble.append(el("span", "q", question));
  $("#stream").append(bubble);

  const answerShell = buildAnswerShell();
  $("#stream").append(answerShell.root);
  $("#stream").scrollTop = $("#stream").scrollHeight;

  $("#live").hidden = false;
  $("#providers").textContent = "";
  $("#log").textContent = "";
  $("#liveStage").textContent = "starting…";
  state.providers.clear();
  state.startedAt = Date.now();

  const reply = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      mode: $("#mode").value,
      max_rounds: Number($("#maxRounds").value),
      conversation_id: state.conversationId,
      project: $("#project").value.trim() || undefined,
    }),
  });
  const res = await reply.json().catch(() => ({}));
  if (!reply.ok || !res.job_id) {
    $("#live").hidden = true;
    answerShell.error.textContent = "Could not start: " + (typeof res.detail === "string" ? res.detail : `HTTP ${reply.status}`);
    return;
  }
  answerShell.jobId = res.job_id;
  if (res.conversation_id) rememberThread(res.conversation_id);
  state.jobId = res.job_id;
  listen(res.job_id, answerShell);
}

function listen(jobId, shell) {
  state.es?.close();
  const es = new EventSource(`/api/stream/${jobId}`);
  state.es = es;
  es.onmessage = async (e) => {
    const ev = JSON.parse(e.data);
    handleEvent(ev, shell);
    if (ev.kind === "done") {
      es.close();
      $("#live").hidden = true;
      await hydrate(jobId, shell);
    }
  };
  es.onerror = () => {
    /* the job keeps running server-side; reconnect below picks the buffer up */
    setTimeout(() => {
      if (state.jobId === jobId) listen(jobId, shell);
    }, 2500);
    es.close();
  };
}

function handleEvent(ev, shell) {
  const seconds = Math.round((Date.now() - state.startedAt) / 1000);
  if (ev.kind === "provider") {
    upsertProvider(ev.provider || "?", ev.message, shell);
  } else if (ev.kind === "status" || ev.kind === "assessment" || ev.kind === "escalation" || ev.kind === "verifier" || ev.kind === "disagreement" || ev.kind === "evidence") {
    $("#liveStage").textContent = `${ev.message}  ·  ${seconds}s`;
    pushLog(ev, seconds);
  } else if (ev.kind === "error") {
    pushLog(ev, seconds);
    shell.error.textContent = ev.message;
  } else if (ev.kind === "final") {
    paintAnswer(shell, { answer: ev.message });
  }
}

function pushLog(ev, seconds) {
  const li = el("li", "log-" + ev.kind, `[${seconds}s] ${ev.kind}: ${ev.message}`);
  $("#log").append(li);
  if ($("#log").children.length > 60) $("#log").firstChild.remove();
}

function upsertProvider(name, message, shell) {
  let row = state.providers.get(name);
  if (!row) {
    row = { li: el("li", "provider"), glyph: el("span", "glyph", "○"), label: el("span", "pname", labelFor(name)), detail: el("span", "pdetail", "") };
    row.li.append(row.glyph, row.label, row.detail);
    $("#providers").append(row.li);
    state.providers.set(name, row);
  }
  const status = /completed|✓/.test(message) ? "completed"
    : /logged_out|login/.test(message) ? "logged_out"
    : /rate_limited|rate limit/.test(message) ? "rate_limited"
    : /timeout|timed out/.test(message) ? "timeout"
    : /broken|selector/.test(message) ? "broken"
    : /failed|error/.test(message) ? "failed"
    : /answer|respond|generat/.test(message) ? "responding"
    : /search|open|compos/.test(message) ? "searching"
    : "connected";
  row.glyph.textContent = STATUS_GLYPH[status] || "○";
  row.li.className = "provider " + status;
  row.detail.textContent = message.replace(/^.*?:\s*/, "").slice(0, 90);
}

function labelFor(name) {
  return state.config?.providers?.[name]?.label || name;
}

function buildAnswerShell() {
  const root = el("div", "msg ai");
  const answer = el("div", "answer", "");
  const chips = el("div", "chips");
  const error = el("div", "err");
  const sections = el("div", "sections");
  root.append(answer, chips, error, sections);
  return { root, answer, chips, error, sections };
}

async function hydrate(jobId, shell) {
  const job = await fetch(`/api/jobs/${jobId}`).then((r) => r.json());
  paintAnswer(shell, finalFrom(job));
  shell.sections.append(exportRow(jobId));
  await paintEvidence(shell, job);
}

/**
 * The audit trail (spec section 19): every page we opened, whether it actually
 * said what the model claimed it said, and the raw capture behind each answer.
 */
function paintMemoryUsed(shell, job) {
  const used = job.memory_used || job.live?.memory_used || job.extra?.memory?.used || [];
  if (!used.length) return;
  shell.sections.append(
    expander(`Memory used as context (${used.length})`, () => {
      const wrap = el("div", "body");
      wrap.append(el("p", "meta", "Offered to the AIs as background about you. Not evidence - it is not in the sources or claims."));
      for (const m of used) wrap.append(memCard(m, { readonly: true }));
      return wrap;
    })
  );
}

async function paintEvidence(shell, job) {
  paintMemoryUsed(shell, job);
  const evidence = job.evidence || job.live?.evidence || [];
  const disagreements = job.disagreements || job.live?.disagreements || [];
  const claims = job.claims || job.live?.claims || [];
  const stop = job.stop_reason || job.live?.stop_reason;

  if (evidence.length) {
    shell.sections.append(
      expander(`Evidence we opened (${evidence.length})`, () => {
        const wrap = el("div", "sources");
        for (const e of evidence) {
          const url = safeUrl(e.url);
          if (!url) continue;
          const row = el("div", "source");
          const a = el("a", null, e.title || e.domain || url);
          a.href = url;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          const badge = el("span", "chip status-" + e.check_status, e.check_status);
          row.append(badge, a);
          const bits = [e.domain, e.tier !== "unknown" ? e.tier : null, e.published ? `published ${e.published}` : null]
            .filter(Boolean);
          if (bits.length) row.append(el("span", "meta", bits.join(" · ")));
          if (e.verbatim_excerpt) row.append(el("span", "meta", "found on page: " + e.verbatim_excerpt.slice(0, 180)));
          wrap.append(row);
        }
        return wrap;
      })
    );
  }

  if (claims.length) {
    shell.sections.append(
      expander(`Claims (${claims.length})`, () => {
        const wrap = el("div", "body");
        for (const c of claims) {
          const row = el("div", "claim");
          row.append(el("span", "claim-text", c.claim));
          row.append(el("span", "claim-status", `${c.status}${c.confidence ? " / " + c.confidence : ""}`));
          row.append(el("span", "meta", "asserted by: " + (c.providers_json || []).join(", ")));
          wrap.append(row);
        }
        return wrap;
      })
    );
  }

  if (disagreements.length) {
    shell.sections.append(
      expander(`Conflicts detected (${disagreements.length})`, () => {
        const wrap = el("div", "body");
        for (const d of disagreements) {
          wrap.append(el("p", null, `[${d.severity}] ${d.description}`));
        }
        if (stop) wrap.append(el("p", "meta", "why it stopped: " + stop));
        return wrap;
      })
    );
  } else if (stop) {
    shell.sections.append(el("p", "meta", "why it stopped: " + stop));
  }
}

function finalFrom(job) {
  const live = job.live || job;
  const final = live.final || null;
  if (!final) {
    return {
      answer: live.error ? `That did not work: ${live.error}` : "No answer produced.",
      confidence_label: "Insufficient evidence",
      caveats: ["research produced nothing checkable"],
    };
  }
  return final;
}

function paintAnswer(shell, final) {
  shell.answer.textContent = "";
  shell.answer.append(el("p", "answer-text", final.answer || "—"));

  shell.chips.textContent = "";
  if (final.confidence_label) shell.chips.append(el("span", "chip conf", final.confidence_label));
  if (final.rounds_run !== undefined) shell.chips.append(el("span", "chip", `${final.rounds_run} round${final.rounds_run === 1 ? "" : "s"}`));
  if (final.providers_used?.length) shell.chips.append(el("span", "chip", `${final.providers_used.length} researcher${final.providers_used.length === 1 ? "" : "s"}`));

  shell.sections.textContent = "";
  const why = final.why || "";
  const disagreement = final.important_disagreement || "";
  const sources = final.sources || [];
  const caveats = final.caveats || [];

  if (why) shell.sections.append(expander("Why", () => el("p", "body", why), true));
  if (disagreement) shell.sections.append(expander("Where they disagree", () => el("p", "body", disagreement), true));
  if (sources.length) shell.sections.append(expander(`Sources (${sources.length})`, () => sourceList(sources)));
  if (caveats.length) shell.sections.append(expander("Caveats", () => { const ul = el("ul", "body"); caveats.forEach((c) => ul.append(el("li", null, c))); return ul; }));
  shell.sections.append(expander("Raw research", () => rawPanel(shell.jobId || state.jobId)));
}

function expander(title, build, open = false) {
  const details = el("details", "fold");
  if (open) details.open = true;
  const summary = el("summary");
  summary.textContent = title;
  details.append(summary);
  let built = false;
  details.addEventListener("toggle", () => {
    if (details.open && !built) {
      built = true;
      const body = build();
      if (body) details.append(body);
    }
  });
  if (open) { const body = build(); if (body) details.append(body); built = true; }
  return details;
}

function sourceList(sources) {
  const wrap = el("div", "sources");
  for (const s of sources) {
    const url = safeUrl(s.url);
    if (!url) continue;
    const row = el("div", "source");
    const a = el("a", null, s.title || s.provider || url);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    row.append(a);
    const meta = [s.provider, s.published && `published ${s.published}`].filter(Boolean).join(" · ");
    if (meta) row.append(el("span", "meta", meta));
    wrap.append(row);
  }
  return wrap;
}

async function rawPanel(jobId) {
  const wrap = el("div", "raw");
  const job = await fetch(`/api/jobs/${jobId}`).then((r) => r.json());
  const responses = job.responses || job.live?.responses || [];
  if (!responses.length) wrap.append(el("p", "body", "no provider responses were captured"));
  for (const r of responses) {
    const card = el("details", "fold raw-card");
    const summary = el("summary");
    summary.textContent = `${labelFor(r.provider)} · round ${r.round} · ${r.status}` + (r.duration_s ? ` · ${Math.round(r.duration_s)}s` : "");
    card.append(summary);
    card.append(field("Prompt we sent", r.prompt));
    card.append(field("Raw answer", r.answer_text || r.raw_text || "(nothing captured)"));
    if (r.web_research) card.append(el("p", "meta", `web research: ${r.web_research}`));
    if (r.error) card.append(el("p", "meta err", `error: ${r.error}`));
    if (r.pages_json?.length) card.append(links(r.pages_json));
    card.append(field("Provider URL", r.ui_url, true));
    wrap.append(card);
  }
  const claims = job.claims || [];
  if (claims.length) {
    const table = el("details", "fold raw-card");
    table.append(el("summary", null, `Claims & evidence (${claims.length})`));
    for (const c of claims) {
      const row = el("div", "claim");
      row.append(el("span", "claim-text", c.claim));
      row.append(el("span", "claim-status", `${c.status}${c.confidence ? " / " + c.confidence : ""}`));
      const evs = (job.evidence || []).filter((e) => e.claim_id === c.id);
      if (evs.length) row.append(links(evs.map((e) => `[${e.check_status}] ${e.url}`)));
      table.append(row);
    }
    wrap.append(table);
  }
  const disagreements = job.disagreements || [];
  if (disagreements.length) {
    const d = el("details", "fold raw-card");
    d.append(el("summary", null, `Detected conflicts (${disagreements.length})`));
    disagreements.forEach((x) => d.append(el("p", "body", `[${x.severity}] ${x.description}`)));
    wrap.append(d);
  }
  if (job.stop_reason) wrap.append(el("p", "meta", `stopped: ${job.stop_reason}`));
  return wrap;
}

function field(label, value, asLink = false) {
  if (!value) return el("span");
  const wrap = el("div", "field-block");
  wrap.append(el("h4", "field-label", label));
  if (asLink) {
    const url = safeUrl(value);
    if (url) {
      const a = el("a", null, url);
      a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer";
      wrap.append(a);
      return wrap;
    }
  }
  const pre = el("pre", "pre");
  pre.textContent = value;
  wrap.append(pre);
  return wrap;
}

function links(items) {
  const ul = el("ul", "linklist");
  for (const raw of items) {
    const text = typeof raw === "string" ? raw : raw.url;
    const url = safeUrl(text);
    if (!url) continue;
    const li = el("li");
    const a = el("a", null, url);
    a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer";
    li.append(a);
    ul.append(li);
  }
  return ul;
}

async function cancelJob() {
  if (!state.jobId) return;
  await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
  $("#liveStage").textContent = "stopping…";
}

boot();
