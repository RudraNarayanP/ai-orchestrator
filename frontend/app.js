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
  $("#settings").hidden = !open;
}

async function saveSettings() {
  const providers = {};
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

  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      mode: $("#mode").value,
      max_rounds: Number($("#maxRounds").value),
      conversation_id: state.conversationId,
    }),
  }).then((r) => r.json());
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
  await paintEvidence(shell, job);
}

/**
 * The audit trail (spec section 19): every page we opened, whether it actually
 * said what the model claimed it said, and the raw capture behind each answer.
 */
async function paintEvidence(shell, job) {
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
  shell.sections.append(expander("Raw research", () => rawPanel(state.jobId)));
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
