"""Page-side DOM library installed into every OmniBrain page.

Written as one injected object rather than scattered one-off evaluate() calls,
so the mechanics live in a single auditable place. Everything here scores
candidates instead of trusting a single id, because a consumer AI site redesigns
about as often as we can test.
"""

from __future__ import annotations

# Bumped whenever the page-side API changes, so a page holding an older copy gets the new one.
DOM_LIBRARY_VERSION = 5

# Installed with page.evaluate(DOM_LIBRARY_JS) before any interaction.
DOM_LIBRARY_JS = r"""
(() => {
  if (window.__omnibrain && window.__omnibrain.version === 5) return 'already';
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const lower = (s) => (s || '').toLowerCase();

  // A sign-in form's email box looks exactly like a composer to a generic
  // scorer. Typing a research prompt into it is both useless and a bad habit to
  // teach a tool that has access to logged-in profiles, so credential fields are
  // recognised and refused at the selector layer.
  const CRED_RE = /(phone|e-?mail|password|passwd|username|user name|otp|one[- ]?time|verification code|captcha|sign[- ]?in|log[- ]?in|account number)/i;

  const isCredential = (el) => {
    if (!el || !el.getAttribute) return false;
    const hay = [el.getAttribute('placeholder'), el.getAttribute('aria-label'), el.getAttribute('name'),
                 el.getAttribute('id'), el.type, el.autocomplete, el.getAttribute('data-testid'), el.id]
      .filter(Boolean).join(' ');
    if (el.type === 'password' || el.type === 'tel') return true;
    return CRED_RE.test(hay);
  };

  const rectOf = (el) => { try { return el.getBoundingClientRect(); } catch (e) { return null; } };

  const visible = (el) => {
    if (!el || !el.isConnected) return false;
    const r = rectOf(el);
    if (!r) return false;
    let st;
    try { st = getComputedStyle(el); } catch (e) { return r.width > 0; }
    if (st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) === 0) return false;
    // Elements may sit far down a scroll container; size is the real test.
    return (r.width > 6 && r.height > 6) || (r.width > 6 && r.height > 0 && st.position === 'fixed');
  };

  const labelOf = (el) => norm(
    el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') ||
    el.getAttribute('data-testid') || el.id || ''
  );

  const textOf = (el) => norm((el.innerText || el.textContent || el.value || '').slice(0, 120));

  function describe(el, kind) {
    const r = rectOf(el) || {x: 0, y: 0, width: 0, height: 0};
    return {
      kind,
      tag: el.tagName.toLowerCase(),
      id: el.id || null,
      role: el.getAttribute('role'),
      testid: el.getAttribute('data-testid') || el.getAttribute('data-test-id'),
      aria: el.getAttribute('aria-label'),
      placeholder: el.getAttribute('placeholder'),
      editable: el.getAttribute('contenteditable'),
      classes: norm((el.className || '').toString()).slice(0, 160),
      label: labelOf(el),
      text: textOf(el),
      w: Math.round(r.width), h: Math.round(r.height), y: Math.round(r.y),
      disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
      el,
    };
  }

  // UI-chrome words that leak into captured answers. Ported from
  // paritoshv/multi-llm-web-orchestrator (content-helpers.js, MIT).
  const cleanAssistantText = (text) => (text || '')
    .replace(/\bCopied!?/gi, '')
    .replace(/\bRegenerate\b/gi, '')
    .replace(/\b\w+ said\b/gi, '')          // "Gemini said", "ChatGPT said" scaffolding
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\s*\n\s*\n\s*/g, '\n\n')
    .trim();

  // Removing a button wholesale once produced answers reading
  // "I can also , , or ." -- the suggestion chips held the sentence's words.
  // Unwrapping keeps the text and drops only the widget. (jumas45, MIT)
  const unwrap = (n) => {
    const parent = n.parentNode;
    if (!parent) return;
    while (n.firstChild) parent.insertBefore(n.firstChild, n);
    parent.removeChild(n);
  };

  // A generic "article" selector matches the *user's* turns too, and capturing
  // your own prompt as the model's answer is the classic way to fake a result.
  // Filter on the role attribute the sites publish, then on class, then on an
  // explicit "You:" prefix -- in that order, so a page that labels nothing still
  // gets the conservative reading.
  const isUserNode = (el) => {
    const roleAttrs = ['data-message-author-role', 'data-message-role', 'data-role', 'data-author', 'data-sender'];
    for (const attr of roleAttrs) {
      const v = lower(el.getAttribute(attr) || '');
      if (v && v !== 'assistant' && v !== 'model' && v !== 'ai' && v !== 'bot') return true;
    }
    const cls = lower((el.className || '').toString());
    if (/(^|[\s_\-])(user|human|query|prompt|my-message|user-message)([$\s_\-]|$)/.test(cls)) return true;
    if (el.closest('[data-message-author-role="user"], .user-message, [class*="user-turn"]')) return true;
    const first = norm(el.innerText).slice(0, 24);
    if (/^(you|your|user)\s*:/i.test(first)) return true;
    return false;
  };

  const strip = (el) => {
    el.querySelectorAll('svg, style, script, noscript, [aria-hidden="true"], .visually-hidden').forEach(n => n.remove());
    const noise = /^(copy|copied!?|regenerate|good response|bad response|share|more options|voice input|thumbs (up|down)|read aloud|show sources)$/i;
    el.querySelectorAll('button').forEach(b => {
      const t = norm(b.innerText);
      if (!t || noise.test(t)) b.remove();
      else unwrap(b);
    });
    el.querySelectorAll('[class*="cursor"], [class*="caret-blink"], .result-streaming > *').forEach(n => {
      if (!norm(n.textContent)) n.remove();
    });
    return el;
  };

  // ---- markdown-ish conversion: canonical plain text for the pipeline, and
  // nothing that leaks UI chrome like "Copied!" into an extracted answer.
  function toMarkdown(root) {
    const work = root.cloneNode(true);
    strip(work);
    work.querySelectorAll('a[href]').forEach(a => {
      const href = a.href;
      const label = norm(a.innerText) || norm(a.getAttribute('aria-label')) || '';
      if (label && href && !href.startsWith('javascript')) {
        const sup = a.closest('sup,cite,[class*="citation"],[class*="source"]');
        if (sup && (!label || /^\d+$/.test(label))) a.replaceWith(document.createTextNode(''));
        else a.replaceWith(document.createTextNode(label ? ` [${label}](${href})` : ` ${href} `));
      } else if (href && !href.startsWith('javascript')) {
        a.replaceWith(document.createTextNode(` ${href} `));
      } else {
        a.replaceWith(document.createTextNode(norm(a.innerText)));
      }
    });
    work.querySelectorAll('h1,h2,h3,h4').forEach(h => {
      const level = Number(h.tagName[1]);
      h.replaceWith(document.createTextNode('\n\n' + '#'.repeat(level) + ' ' + norm(h.innerText) + '\n'));
    });
    work.querySelectorAll('li').forEach(li => li.replaceWith(document.createTextNode('\n- ' + norm(li.innerText))));
    work.querySelectorAll('pre,code').forEach(c => {
      const t = c.textContent || '';
      c.replaceWith(document.createTextNode(c.tagName === 'PRE' ? '\n```\n' + t + '\n```\n' : '`' + t + '`'));
    });
    work.querySelectorAll('strong,b').forEach(b => b.replaceWith(document.createTextNode('**' + norm(b.innerText) + '**')));
    work.querySelectorAll('em,i').forEach(b => b.replaceWith(document.createTextNode('*' + norm(b.innerText) + '*')));
    work.querySelectorAll('p,div,section,tr').forEach(p => p.append(document.createTextNode('\n')));
    let text = work.innerText || work.textContent || '';
    return text.replace(/\u00a0/g, ' ').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  }

  // ---- candidate locators --------------------------------------------------
  function queryAll(css) {
    try { return Array.from(document.querySelectorAll(css)); } catch (e) { return []; }
  }

  function matchesConfig(el, cfg) {
    const hay = lower([el.id, el.getAttribute('role'), el.getAttribute('data-testid'),
                       el.getAttribute('aria-label'), el.getAttribute('placeholder'),
                       (el.className || '').toString(), textOf(el)].join(' '));
    let score = 0;
    (cfg.css || []).forEach((sel, i) => {
      try { if (el.matches(sel)) score += 40 - Math.min(i * 3, 20); } catch (e) {}
    });
    (cfg.testids || []).forEach(t => { if (lower(el.getAttribute('data-testid') || '').includes(lower(t))) score += 35; });
    (cfg.aria || []).forEach(a => { if (lower(labelOf(el)).includes(lower(a))) score += 30; });
    (cfg.placeholders || []).forEach(p => { if (lower(el.getAttribute('placeholder') || '').includes(lower(p))) score += 25; });
    if (cfg.text_regex) { try { if (new RegExp(cfg.text_regex, 'i').test(textOf(el))) score += 18; } catch (e) {} }
    return score;
  }

  function candidates(pool, cfg, opts) {
    const kind = (opts && opts.kind) || 'input';
    let els = [];
    (cfg.css || []).forEach(sel => { els = els.concat(queryAll(sel)); });
    const universal = kind === 'input'
      ? 'textarea, [contenteditable="true"], [contenteditable=""], input[type="text"], [role="textbox"], [role="combobox"]'
      : 'button, [role="button"], a[role="button"], [type="submit"]';
    els = els.concat(queryAll(universal));
    const seen = new Set();
    const out = [];
    for (const el of els) {
      if (!el || seen.has(el) || !visible(el)) continue;
      seen.add(el);
      let score = matchesConfig(el, cfg);
      if (kind === 'input') {
        if (isCredential(el)) continue;
        const isEdit = el.tagName === 'TEXTAREA' || el.getAttribute('contenteditable') === 'true' ||
                       el.getAttribute('contenteditable') === '' || el.getAttribute('role') === 'textbox';
        if (!isEdit) continue;
        if (el.tagName === 'INPUT' && (el.type === 'hidden' || el.type === 'search')) continue;
        const r = rectOf(el);
        if (r) {
          if (r.width > 180) score += 14;
          // A composer sits near the bottom of the viewport on every one of these UIs.
          if (r.top > innerHeight * 0.45 && r.top < innerHeight * 0.98) score += 22;
          else if (r.top > innerHeight * 0.3) score += 8;
        }
        if (el.disabled) score -= 45;
        if (el.readOnly) score -= 25;
        if (/search/i.test(labelOf(el)) && !/ask|message|prompt|chat/i.test(labelOf(el))) score -= 10;
      } else {
        if (el.disabled) score -= 50;
        const r = rectOf(el);
        if (r && r.top > innerHeight * 0.4) score += 6;
      }
      if (score <= 0) continue;
      out.push({score, el, desc: describe(el, kind)});
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }

  // ---- the send button -------------------------------------------------------
  // A send control is: the site's configured one, a button named send/submit, or the text-less icon button that
  // sits beside the composer (rightmost wins). Never help/feedback/settings/upgrade/attach/voice/share buttons and
  // never a mode toggle. Seen live on pi.ai (2026-10-08): "first button on the page" clicked "Help & feedback".
  const SEND_RE = /(^|[^a-z])(send|submit|envoyer|enviar|senden|invia)([^a-z]|$)|发送|提交|送信/i;
  const NOT_SEND_RE = /(help|feedback|setting|upgrade|premium|pricing|plan\b|subscri|share|attach|upload|file|image|photo|camera|voice|dictat|micro|speech|record|audio|think|reason|research|search|tool|mode\b|model|menu|more\b|option|new chat|sign|log ?in|regenerate|retry|copy|like|edit|stop|cancel|close|dismiss|sidebar|history|account|profile|expand|collapse|scroll|emoji|plus|add\b|canvas|image|create)/i;
  const HARD_NO_RE = /(help|feedback|setting|upgrade|premium|pricing|subscri|share|attach|upload|voice|dictat|micro|sign ?in|sign ?up|log ?in|account|profile|stop generating|^stop)/i;
  function isToggle(el) {
    const role = lower(el.getAttribute('role') || '');
    if (/^(switch|checkbox|menuitem|menuitemcheckbox|menuitemradio|tab|radio|option|combobox)$/.test(role)) return true;
    if (el.hasAttribute('aria-pressed') || el.hasAttribute('aria-checked') || el.hasAttribute('aria-expanded')) return true;
    const pop = lower(el.getAttribute('aria-haspopup') || '');
    return !!pop && pop !== 'false';
  }
  function sendCandidates(cfg) {
    const send = cfg.send || {};
    const comp = (candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'})[0] || {}).el || null;
    const cr = comp ? rectOf(comp) : null;
    const form = comp ? comp.closest('form') : null;
    let els = [];
    (send.css || []).forEach(sel => { els = els.concat(queryAll(sel)); });
    els = els.concat(queryAll('button, [role="button"], input[type="submit"]'));
    const seen = new Set();
    const out = [];
    for (const el of els) {
      if (!el || seen.has(el) || !visible(el)) continue;
      seen.add(el);
      if (el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
      if (comp && (el === comp || el.contains(comp))) continue;
      const label = norm([el.getAttribute('aria-label'), el.getAttribute('title'), el.getAttribute('data-testid'), textOf(el)].filter(Boolean).join(' '));
      const named = SEND_RE.test(label);
      // an exact configured selector (e.g. a search engine's "Search" button) still never wins as help/upgrade/etc.
      let pinned = false;
      (send.css || []).forEach(sel => { try { if (el.matches(sel)) pinned = true; } catch (e) {} });
      (send.testids || []).forEach(t => { if (lower(el.getAttribute('data-testid') || '').includes(lower(t))) pinned = true; });
      if (isToggle(el) || HARD_NO_RE.test(label)) continue;
      if (!named && !pinned && NOT_SEND_RE.test(label)) continue;
      const cfgScore = matchesConfig(el, send);
      const r = rectOf(el);
      let near = false;
      if (form && form.contains(el)) near = true;
      else if (cr && r) {
        near = r.top < cr.bottom + 90 && r.bottom > cr.top - 90 && r.left > cr.left + cr.width * 0.4 && r.left < cr.right + 140;
      }
      const textless = !textOf(el) || textOf(el).length <= 2;
      if (!(cfgScore >= 25 || named || (near && textless))) continue;
      let score = cfgScore + (named ? 60 : 0) + (near ? 30 : 0);
      if (near && textless && r) score += 10 + Math.round((r.right / Math.max(1, innerWidth)) * 10);
      out.push({el, score, why: [cfgScore >= 25 ? 'config' : '', named ? 'named' : '', near ? 'near-composer' : ''].filter(Boolean).join('+')});
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }

  function firstMessageRoots(cfg) {
    // Scoping matters: [role=log] is the transcript, while <aside> holds the
    // sidebar conversation list, which would be captured as "the answer".
    let roots = [];
    (cfg.response_root || []).forEach(sel => { roots = roots.concat(queryAll(sel)); });
    roots = roots.filter(el => visible(el) && el.innerText && el.innerText.length > 3);
    return roots;
  }

  function assistantBlocks(cfg) {
    const blocks = [];
    const seen = new Set();
    for (const root of firstMessageRoots(cfg)) {
      (cfg.assistant_message || []).forEach(sel => {
        let nodes;
        try { nodes = root.matches(sel) ? [root] : Array.from(root.querySelectorAll(sel)); } catch (e) { nodes = []; }
        for (const n of nodes) {
          if (!n || seen.has(n) || !visible(n)) continue;
          if (n.closest('aside') || n.closest('[role="complementary"]')) continue;
          if (isUserNode(n)) continue;
          if (n.querySelector('[data-message-author-role="user"]') && (n.innerText || '').length < 400) continue;
          // Selectors are tried in priority order and often match nested wrappers of the SAME message
          // (Gemini: message-content .markdown, message-content, model-response). Keep the first, skip
          // anything that contains it or sits inside it, or the answer is captured two or three times.
          if (blocks.some(b => b.contains(n) || n.contains(b))) continue;
          seen.add(n);
          blocks.push(n);
        }
      });
    }
    return blocks;
  }

  function busy(cfg) {
    const sig = [];
    (cfg.streaming || []).forEach(sel => {
      queryAll(sel).forEach(el => { if (visible(el)) sig.push('streaming:' + sel); });
    });
    queryAll('[aria-busy="true"]').forEach(el => { if (visible(el)) sig.push('aria-busy'); });
    const stopCfg = cfg.stop || {};
    candidates(stopCfg, {css: stopCfg.css || [], testids: stopCfg.testids || [], aria: stopCfg.aria || [], text_regex: stopCfg.text_regex}, {kind: 'button'})
      .slice(0, 3)
      .forEach(c => {
        const t = lower(c.desc.text + ' ' + (c.desc.aria || '') + ' ' + c.desc.classes);
        if (/stop|cancel|arr|êter|annuler|generating|pause/.test(t)) sig.push('stop-button:' + (c.desc.aria || c.desc.text));
      });
    (cfg.busy_markers || []).forEach(marker => {
      if (new RegExp(marker, 'i').test(norm(document.body.innerText.slice(0, 4000)))) sig.push('marker:' + marker);
    });
    return sig;
  }

  function linksFrom(node, cfg) {
    const out = [];
    const seen = new Set();
    const scope = node || document;
    const sel = (cfg.sources || []).concat(['a[href^="http"]']);
    for (const s of sel) {
      let nodes;
      try { nodes = Array.from(scope.querySelectorAll(s)); } catch (e) { continue; }
      for (const a of nodes) {
        let href = a.href || a.getAttribute('data-linkOnClickHref') || a.getAttribute('data-search-result-url') || '';
        if (!href || href.startsWith('javascript:') || seen.has(href)) continue;
        if (/^https?:\/\/([^./]*\.)?(openai|google|microsoft|meta|mistral|pi\.ai)/i.test(href) &&
            !/^(https?:\/\/www\.google\.com\/url\?q=)/i.test(href) &&
            !/(chatgpt\.com|gemini\.google|copilot\.microsoft|meta\.ai|chat\.mistral|pi\.ai)/i.test(href.split('/')[2] || '')) {
          // keep: cross-site links are legitimate sources too
        }
        if (/^(https?:)?\/\//.test(href) === false) continue;
        const host = (href.split('/')[2] || '').toLowerCase();
        if (!host) continue;
        seen.add(href);
        const label = norm(a.innerText || a.getAttribute('aria-label') || '');
        const near = a.closest('[class*="source"],[class*="citation"],cite,sup,li');
        out.push({
          href,
          host,
          title: label.slice(0, 220) || host,
          snippet: near ? norm(near.innerText).slice(0, 400) : label.slice(0, 400),
          inSourceList: !!((cfg.sources || []).some(s => { try { return a.closest(s); } catch (e) { return false; } })),
          marker: /^\d{1,2}$/.test(label) ? label : null,
        });
      }
    }
    return out;
  }

  // A modal that makes you accept Terms of Service / a privacy policy before the product works is a legal
  // agreement: only the person can make it. It is never clicked, and the page is reported blocked.
  const TERMS_NAMED_RE = /terms (of (service|use)|and conditions|& conditions)/i;
  const TERMS_REQUIRED_RE = /(must|need to|required to|have to)\s+(accept|agree)|(accept|agree)( and|&) continue/i;
  const TERMS_GATE_RE = {test: (t) => TERMS_NAMED_RE.test(t) && TERMS_REQUIRED_RE.test(t)};
  function termsGateText() {
    for (const dlg of document.querySelectorAll('[role="dialog"], [role="alertdialog"], [aria-modal="true"]')) {
      if (!visible(dlg)) continue;
      const t = norm(dlg.innerText || '');
      if (TERMS_GATE_RE.test(t)) return t.slice(0, 200);
    }
    return '';
  }
  function inTermsGate(el) {
    const dlg = el.closest && el.closest('[role="dialog"], [role="alertdialog"], [aria-modal="true"]');
    return !!dlg && TERMS_GATE_RE.test(norm(dlg.innerText || ''));
  }

  // ---- guest chat ------------------------------------------------------------------------------------------------
  // Several providers chat without an account but first put up a sign-up nudge (ChatGPT logged out: "Log in /
  // Sign up for free / Stay logged out"). Only the site's own VISIBLE "Stay logged out" / "Continue without account"
  // / close control inside such a nudge is pressed. Never: log in, sign up, "continue with Google", accept/agree,
  // upgrade -- and nothing at all inside a captcha / age / consent / terms dialog (those stay the person's).
  const GUEST_NUDGE_RE = /(log ?in|sign ?(in|up)|create (a |an |your )?(free )?account|stay logged out|without (an )?account)/i;
  const GUEST_GATE_RE = /(captcha|robot|are you (a )?human|verify (that )?you|birth|\bborn\b|your age|age verification|over 18|at least 1[3-8]|cookie|consent|terms (of|and|&)|privacy policy|\baccept|\bagree)/i;
  const GUEST_BTN_RE = /^(stay logged out|stay signed out|continue (without|w\/o) (an )?account|continue as (a )?guest|use (it )?without (an )?account|chat without (an )?account|try (it )?without (an )?account|close|dismiss|\u00d7|\u2715|\u2716)$/i;
  const GUEST_ARIA_RE = /^(close|dismiss|close (the )?(dialog|modal|popup|banner))$/i;
  const GUEST_PREFER_RE = /(logged out|signed out|without|guest)/i;
  function guestContainers() {
    const out = new Set();
    for (const d of document.querySelectorAll('[role="dialog"], [role="alertdialog"], [aria-modal="true"], dialog[open]')) {
      if (visible(d)) out.add(d);
    }
    // a nudge drawn as a plain fixed overlay over the middle of the page (no dialog role)
    try {
      let el = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
      while (el && el !== document.body && el !== document.documentElement) {
        const st = getComputedStyle(el);
        if (st.position === 'fixed' && visible(el)) { out.add(el); break; }
        el = el.parentElement;
      }
    } catch (e) {}
    return [...out];
  }
  function guestDismiss(cfg) {
    const clicked = [];
    const skipped = [];
    for (const box of guestContainers()) {
      const text = norm(box.innerText || '');
      if (!GUEST_NUDGE_RE.test(text)) continue;
      if (GUEST_GATE_RE.test(text) || TERMS_GATE_RE.test(text)) { skipped.push('gate:' + text.slice(0, 60)); continue; }
      // the composer itself lives in a fixed footer on some sites: that is the app, not a nudge
      if (box.querySelector('textarea, [contenteditable="true"], [role="textbox"]')) continue;
      const options = [];
      for (const b of box.querySelectorAll('button, [role="button"], a')) {
        if (!visible(b) || b.disabled || b.getAttribute('aria-disabled') === 'true') continue;
        if (b.tagName === 'A') {
          const href = (b.getAttribute('href') || '').trim();
          if (href && href !== '#' && !/^javascript:/i.test(href)) continue;  // a real link could lead to a login page
        }
        const label = norm(b.innerText || '');
        const aria = norm(b.getAttribute('aria-label') || b.getAttribute('title') || '');
        const ok = (label && GUEST_BTN_RE.test(label)) || (!label.replace(/[\u00d7\u2715\u2716]/g, '') && GUEST_ARIA_RE.test(aria));
        if (ok) options.push({b, label: label || aria});
      }
      if (!options.length) { skipped.push('no-guest-control:' + text.slice(0, 60)); continue; }
      options.sort((x, y) => (GUEST_PREFER_RE.test(y.label) ? 1 : 0) - (GUEST_PREFER_RE.test(x.label) ? 1 : 0));
      try { options[0].b.click(); clicked.push(options[0].label.slice(0, 40)); } catch (e) {}
      if (clicked.length >= 2) break;
    }
    return {clicked, skipped};
  }

  // ---- provider mode toggles (Thinking / Search / Deep Research) --------------------------------------------------
  // Read-only discovery of the toggles that sit beside the composer, with their on/off state when the page exposes one.
  // Only top-level controls near the composer count (a sidebar "Search chats" is not the web-search mode). A control
  // whose label talks about upgrading/plans, or that is disabled, is reported as plan-gated and is never clicked.
  const MODE_PATTERNS = [
    ['deep_research', /deep ?research/i],
    ['thinking', /(^|[^a-z])(think|thinking|deep ?think|reason|reasoning|think deeper|think longer)([^a-z]|$)/i],
    ['search', /(^|[^a-z])(search|web search|search the web|browse)([^a-z]|$)/i],
  ];
  const MODE_GATE_RE = /(upgrade|premium|subscribe|unlock|get plus|get pro|pro plan|limit reached|try .* free)/i;
  // live (DeepSeek 2026-10-08): DeepThink / Search are <div class="ds-toggle-button" aria-pressed> with no role
  const MODE_SEL = 'button, [role="button"], [role="switch"], [role="checkbox"], [role="menuitemcheckbox"], [aria-pressed], [class*="toggle-button"]';
  const MODE_MENU_RE = /(mode picker|model picker|switch model|choose model|select model|tools|\bmodes?\b|\bmodel\b)/i;
  function modeMenus(cfg) {
    // Modes that live inside a menu (Gemini "Open mode picker, currently Flash", "Upload and tools") are reported, not driven.
    const comp = (candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'})[0] || {}).el || null;
    const cr = comp ? rectOf(comp) : null;
    const out = [];
    for (const el of queryAll('button, [role="button"]')) {
      if (!visible(el)) continue;
      const pop = lower(el.getAttribute('aria-haspopup') || '');
      if (!pop || pop === 'false') continue;
      const label = norm([el.getAttribute('aria-label'), textOf(el)].filter(Boolean).join(' ')).slice(0, 90);
      if (!MODE_MENU_RE.test(label)) continue;
      const r = rectOf(el);
      if (cr && r && Math.abs((r.top + r.bottom) / 2 - (cr.top + cr.bottom) / 2) > 220) continue;
      out.push({label});
    }
    return out;
  }
  const MODE_SKIP_RE = /(search chats|search history|search conversations|new chat|history|sidebar)/i;
  function modeState(el) {
    const pressed = el.getAttribute('aria-pressed') || el.getAttribute('aria-checked') || el.getAttribute('aria-selected');
    if (pressed === 'true' || pressed === 'mixed') return 'on';
    if (pressed === 'false') return 'off';
    const ds = lower(el.getAttribute('data-state') || '');
    if (/^(on|checked|active|selected|open)$/.test(ds)) return 'on';
    if (/^(off|unchecked|inactive|closed)$/.test(ds)) return 'off';
    const cls = lower((el.className || '').toString());
    if (/(^|[\s_-])(active|selected|checked|is-on|toggled)([\s_-]|$)/.test(cls)) return 'on';
    return 'unknown';
  }
  function modeControls(cfg) {
    const comp = (candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'})[0] || {}).el || null;
    const cr = comp ? rectOf(comp) : null;
    const form = comp ? comp.closest('form') : null;
    const out = [];
    const seen = new Set();
    const els = queryAll(MODE_SEL);
    els.forEach((el, index) => {
      if (seen.has(el) || !visible(el)) return;
      seen.add(el);
      const label = norm([el.getAttribute('aria-label'), el.getAttribute('title'), textOf(el)].filter(Boolean).join(' ')).slice(0, 90);
      if (!label || label.length > 60 || MODE_SKIP_RE.test(label)) return;
      const hit = MODE_PATTERNS.find(([, re]) => re.test(label));
      if (!hit) return;
      const r = rectOf(el);
      let near = !!(form && form.contains(el));
      if (!near && cr && r) near = Math.abs((r.top + r.bottom) / 2 - (cr.top + cr.bottom) / 2) < 220 && r.left < cr.right + 260 && r.right > cr.left - 260;
      if (!near) return;
      const disabled = !!el.disabled || el.getAttribute('aria-disabled') === 'true';
      out.push({mode: hit[0], label, state: modeState(el), disabled, gated: MODE_GATE_RE.test(label) || HARD_NO_RE.test(label), index});
    });
    return out;
  }
  function modeElement(cfg, mode) {
    const found = modeControls(cfg).filter(c => c.mode === mode);
    if (!found.length) return [null, null];
    const els = queryAll(MODE_SEL);
    return [found[0], els[found[0].index] || null];
  }

  const api = {
    version: 5,
    guestDismiss(cfg) { return guestDismiss(cfg); },
    modeControls(cfg) { return modeControls(cfg); },
    modeMenus(cfg) { return modeMenus(cfg); },
    setMode(cfg, mode, want) {
      const [info, el] = modeElement(cfg, mode);
      if (!info || !el) return {ok: false, reason: 'unavailable'};
      if (info.gated) return {ok: false, reason: 'plan_gated', label: info.label};
      if (info.disabled) return {ok: false, reason: 'disabled', label: info.label};
      const target = want ? 'on' : 'off';
      if (info.state === target) return {ok: true, already: true, label: info.label, state: info.state};
      if (info.state === 'unknown') return {ok: false, reason: 'state_unreadable', label: info.label};
      el.click();
      return {ok: true, clicked: true, label: info.label, before: info.state};
    },
    util: {norm, visible, toMarkdown},

    ready(cfg) {
      const body = norm(document.body ? document.body.innerText : '');
      const url = location.href;
      const loginCfg = (cfg.login_wall || []).some(s => queryAll(s).some(visible));
      const loginWords = /sign in|log in|create (a |an )?account|continue with (google|facebook|apple|microsoft)|get started/i.test(body.slice(0, 2500));
      const rateWords = /(rate limit|too many requests|slow down|you'?ve reached .{0,30}(limit|cap)|try again in|upgrade to|daily (limit|cap)|out of (free )?credits|please wait)/i.test(body);
      const blockedWords = /(access denied|are you a robot|unusual traffic|verify you are a human|enable javascript|are you a human|confirm your age|what year were you born|date of birth|verify your age|what should i call you|preferred name)/i.test(body);
      const inputHere = candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'}).length > 0;
      const termsGate = termsGateText();
      const credentialForm = [...document.querySelectorAll('input, textarea, [contenteditable="true"]')]
        .some(el => isCredential(el) && visible(el));
      let state = 'unknown';
      if (blockedWords || termsGate) state = 'blocked';
      else if (rateWords && !inputHere) state = 'rate_limited';
      else if (credentialForm && !inputHere) state = 'login_wall';
      else if (loginCfg || (loginWords && !inputHere)) state = 'login_wall';
      else if (inputHere) state = 'ready';
      else if (cfg.chat_mode !== false && loginWords) state = 'login_wall';
      return {
        state, url, inputHere,
        credentialForm,
        hasComposer: inputHere,
        loginWall: loginCfg || loginWords,
        rateLimited: rateWords,
        blocked: blockedWords || !!termsGate,
        gate: termsGate ? 'terms-of-service acceptance required' : (blockedWords ? 'age/identity check or bot check' : ''),
        bodyLength: body.length,
        bodyHead: body.slice(0, 300),
        title: document.title,
        visibility: document.visibilityState,
      };
    },

    locateInput(cfg) {
      const list = candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'});
      return list.slice(0, 8).map(c => ({...c.desc, el: undefined, score: c.score}));
    },

    // Returns the live element (used with Playwright evaluate_handle) so the
    // Python side never re-implements candidate scoring in a second place.
    pickEl(cfg, kind) {
      const sub = kind === 'button' ? (cfg.send || {}) : (cfg.input || {});
      const list = candidates(sub, sub, {kind});
      return list.length ? list[0].el : null;
    },

    locateSend(cfg) {
      const list = candidates(cfg.send || {}, cfg.send || {}, {kind: 'button'});
      return list.slice(0, 8).map(c => ({...c.desc, el: undefined, score: c.score}));
    },

    dismiss(cfg) {
      const clicked = [];
      const cfgd = cfg.dismiss || {};
      // safe_dismiss (the live-Chrome driver, the person's own browser): never click anything that accepts / agrees / allows
      const unsafe = /(accept|agree|allow|consent|\byes\b)/i;
      const want = /(accept all|accept all cookies|i agree|got it|no thanks|not now|maybe later|close|dismiss|allow all|decline all|reject all|continue without|one ?time|later today|use week)/i;
      for (const c of candidates(cfgd, {css: cfgd.css || [], text_regex: cfgd.text_regex, aria: cfgd.aria || [], testids: cfgd.testids}, {kind: 'button'})) {
        const t = c.desc.text + ' ' + (c.desc.aria || '');
        if (!want.test(t)) continue;
        if (cfg.safe_dismiss && unsafe.test(t)) continue;
        if (inTermsGate(c.el)) continue;
        try { c.el.click({beacon: false}); clicked.push(norm(t).slice(0, 60)); } catch (e) {}
        if (clicked.length >= 4) break;
      }
      for (const sel of ['form[role="dialog"]', '#layers .zPTDog', '[data-testid="popup"]']) {
        queryAll(sel).forEach(dlg => {
          Array.from(dlg.querySelectorAll('button')).forEach(b => {
            if (want.test(norm(b.innerText)) && !(cfg.safe_dismiss && unsafe.test(norm(b.innerText))) && visible(b) && !inTermsGate(b)) { try { b.click(); clicked.push('modal:' + norm(b.innerText).slice(0, 40)); } catch (e) {} }
          });
        });
      }
      return clicked;
    },

    baseline(cfg) {
      const blocks = assistantBlocks(cfg);
      const texts = blocks.map(b => norm(b.innerText));
      return {
        count: blocks.length,
        lastText: texts.length ? texts[texts.length - 1] : '',
        lastLength: texts.length ? texts[texts.length - 1].length : 0,
        totalChars: texts.join('').length,
        roots: firstMessageRoots(cfg).length,
      };
    },

    capture(cfg, baseline) {
      const blocks = assistantBlocks(cfg);
      const recent = blocks.slice(Math.max(0, (baseline && baseline.count ? baseline.count : 0)), blocks.length);
      const scan = (recent.length ? recent : blocks.slice(-Math.max(1, cfg.scan_recent_blocks || 4)));
      const items = scan.map(node => {
        const mediaOnly = (() => {
          const media = node.querySelectorAll('img, canvas, video, figure');
          return media.length > 0 && norm(node.innerText).length < 24;
        })();
        return {
          text: cleanAssistantText(toMarkdown(node)),
          plain: cleanAssistantText(norm(node.innerText)),
          length: (node.innerText || '').length,
          links: linksFrom(node, cfg),
          mediaOnly,
          referenceOnly: (() => {
            const t = norm(node.innerText);
            return t.length < 420 && /^(sources|related|citations|explore|show more|learn more)/i.test(t);
          })(),
          roleGuess: lower((node.getAttribute('data-message-author-role') || node.className || '').toString()).includes('assistant') ? 'assistant' : 'unknown',
        };
      });
      const combined = items.map(i => i.text).filter(Boolean).join('\n\n');
      const allLinks = [];
      const seenL = new Set();
      items.forEach(i => i.links.forEach(l => { if (!seenL.has(l.href)) { seenL.add(l.href); allLinks.push(l); } }));
      return {
        busy: busy(cfg),
        blockCount: blocks.length,
        newBlock: baseline ? blocks.length > baseline.count : items.length > 0,
        items,
        text: combined,
        plainLength: norm(combined).length,
        links: allLinks.slice(0, 40),
        visibility: document.visibilityState,
        url: location.href,
        marker: (() => { try { return JSON.stringify({n: blocks.length, l: norm(combined).length}); } catch (e) { return null; } })(),
        error: null,
      };
    },

    // ---- the typing cascade. Method 0 is Playwright's real CDP text input,
    // which React/ProseMirror/Quill all accept because it is genuine input.
    // The rest are DOM-level fallbacks, each verified before we submit.
    typeInto(el, text) {
      const tried = [];
      if (!el) return {ok: false, tried, value: ''};
      const isEditable = el.getAttribute && (el.getAttribute('contenteditable') === 'true' || el.getAttribute('contenteditable') === '');
      const isTA = el.tagName === 'TEXTAREA' || el.tagName === 'INPUT';
      const read = () => (isTA ? (el.value || '') : (el.innerText || el.textContent || ''));
      try { el.scrollIntoView({block: 'center'}); } catch (e) {}
      try { el.focus({preventScroll: true}); } catch (e) {}
      if (isTA) {
        try {
          const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
          const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
          setter.call(el, text);
          el.dispatchEvent(new Event('input', {bubbles: true}));
          el.dispatchEvent(new Event('change', {bubbles: true}));
          tried.push('nativeSetter');
        } catch (e) { tried.push('nativeSetter:fail'); }
      } else {
        try {
          const sel = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(el);
          range.collapse(false);
          sel.removeAllRanges();
          sel.addRange(range);
        } catch (e) {}
        try {
          document.execCommand('selectAll', false, null);
          document.execCommand('delete', false, null);
          if (document.execCommand('insertText', false, text)) tried.push('execCommand');
        } catch (e) { tried.push('execCommand:fail'); }
        if (!read()) {
          try {
            const dt = new DataTransfer();
            dt.setData('text/plain', text);
            el.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
            tried.push('pasteEvent');
          } catch (e) { tried.push('pasteEvent:fail'); }
        }
        if (!read()) {
          try {
            el.textContent = text;
            el.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: text}));
            tried.push('textContent');
          } catch (e) { tried.push('textContent:fail'); }
        }
      }
      const value = read();
      return {ok: value.length >= Math.min(text.length, 8), tried, valueLength: value.length, valueHead: norm(value).slice(0, 80)};
    },

    // "Answer now" / "quick answer": several providers surface it while a
    // reasoning model is still thinking. Only offered in QUICK mode -- a
    // verification round must never take the shortcut past the real research.
    quickAnswer(cfg) {
      const cfgq = cfg.quick_answer || {};
      const list = candidates(cfgq, {css: cfgq.css || [], aria: cfgq.aria || [], text_regex: cfgq.text_regex, testids: cfgq.testids}, {kind: 'button'});
      for (const c of list) {
        const t = norm(c.desc.text + ' ' + (c.desc.aria || ''));
        if (!/answer now|quick answer|fast answer|get a quick answer/i.test(t)) continue;
        try { c.el.click(); return {ok: true, label: t.slice(0, 60)}; } catch (e) {}
      }
      return {ok: false};
    },

    // Best-effort model label for the audit trail. Sanitised hard: an earlier
    // implementation of this pattern captured whole page text when a switcher's
    // innerText was a menu, so length is capped and anything long is discarded.
    modelLabel() {
      const pattern = /(gpt-\s?[\w.\-]+|chatgpt\s?\w+|o\d[\w-]*|Gemini\s?[\d.]+(?:\s?(?:Pro|Flash|Ultra))?|Copilot\s?\w*|Claude\s?\w+|Llama\s?[\d.]+|DeepSeek[-\w]*|Qwen[-\w]*|Mistral[-\s]*\w*|Le\s?Chat|Pi\s?[\d.]+|Grok[-\w]*)/i;
      const nodes = [...document.querySelectorAll('button,[role="combobox"],[aria-label],[title],[data-testid*="model"]')];
      for (const el of nodes) {
        if (!visible(el)) continue;
        const hay = norm(el.getAttribute('aria-label') || el.title || el.innerText || el.getAttribute('data-testid') || '');
        if (!hay || hay.length > 80) continue;
        const m = pattern.exec(hay);
        if (m) return norm(m[0]).slice(0, 60);
      }
      return null;
    },

    pressEnter(el) {
      const target = el || document.activeElement;
      ['keydown', 'keypress', 'keyup'].forEach((type, i) => {
        try {
          target.dispatchEvent(new KeyboardEvent(type, {
            key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true,
          }));
        } catch (e) {}
      });
      return true;
    },

    clickSend(cfg) {
      const list = sendCandidates(cfg);
      for (const c of list) {
        try {
          c.el.scrollIntoView({block: 'nearest'});
          c.el.click();
          return {ok: true, via: {...describe(c.el, 'button'), el: undefined}, score: c.score, why: c.why};
        } catch (e) {}
      }
      return {ok: false, tried: list.slice(0, 4).map(c => ({...describe(c.el, 'button'), el: undefined}))};
    },

    locateSendStrict(cfg) {
      return sendCandidates(cfg).slice(0, 6).map(c => ({...describe(c.el, 'button'), el: undefined, score: c.score, why: c.why}));
    },

    pointerSend(cfg) {
      // A full pointer/mouse event sequence on the send button. In a BACKGROUND tab Chrome drops real input, so this
      // DOM-level route is what works for sites that act on pointerdown/mouseup (seen live: chat.deepseek.com).
      const c = sendCandidates(cfg)[0];
      if (!c) return {ok: false};
      const el = c.el, b = rectOf(el) || {x: 0, y: 0, width: 0, height: 0};
      const x = b.x + b.width / 2, y = b.y + b.height / 2;
      const base = {bubbles: true, cancelable: true, composed: true, clientX: x, clientY: y, button: 0, view: window};
      const ptr = {...base, pointerId: 1, pointerType: 'mouse', isPrimary: true};
      try {
        el.dispatchEvent(new PointerEvent('pointerdown', {...ptr, buttons: 1}));
        el.dispatchEvent(new MouseEvent('mousedown', {...base, buttons: 1}));
        el.dispatchEvent(new PointerEvent('pointerup', ptr));
        el.dispatchEvent(new MouseEvent('mouseup', base));
        el.dispatchEvent(new MouseEvent('click', base));
      } catch (e) { return {ok: false, error: String(e)}; }
      return {ok: true, via: {...describe(el, 'button'), el: undefined}, why: c.why};
    },

    enterOnComposer(cfg) {
      // DOM-dispatched Enter on the composer itself (caret at the end). Works in a background tab for sites whose
      // key handler does not check isTrusted (seen live: chat.deepseek.com).
      const c = candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'})[0];
      if (!c) return {ok: false};
      this.focusComposer(cfg);
      const o = {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true, composed: true};
      let taken = false;
      try {
        taken = !c.el.dispatchEvent(new KeyboardEvent('keydown', o));
        c.el.dispatchEvent(new KeyboardEvent('keypress', o));
        c.el.dispatchEvent(new KeyboardEvent('keyup', o));
      } catch (e) { return {ok: false, error: String(e)}; }
      return {ok: true, defaultPrevented: taken};
    },

    isBusy(cfg) {
      return busy(cfg);
    },

    focusComposer(cfg) {
      // Focus the composer and put the caret at the END, so a real Enter key submits instead of splitting the text.
      const c = candidates(cfg.input || {}, cfg.input || {}, {kind: 'input'})[0];
      if (!c) return false;
      const el = c.el;
      try { el.focus(); } catch (e) { return false; }
      try {
        if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
          const n = (el.value || '').length; el.setSelectionRange(n, n);
        } else {
          const r = document.createRange(); r.selectNodeContents(el); r.collapse(false);
          const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
        }
      } catch (e) {}
      return document.activeElement === el || el.contains(document.activeElement);
    },

    harvestResults(cfg, limit) {
      // Non-chat adapter for search engines: result blocks, in page order.
      // Page furniture (Help/Privacy/Maps/Sign in, image tiles, ads) looks like a
      // link but is not evidence, so it is filtered by host, by path, and by how
      // much readable text surrounds the anchor.
      const JUNK_HOST = /^(maps|accounts|support|policies|play|news\.google|www\.google)\.([a-z.]+)$/i;
      const JUNK_PATH = /(\/search\?|\/seturls?\?|\/url\?(?!q=)|\/imgres|\/maps\/|\/shots\/|\/logging\/|\/privacy|\/terms|\/services\/accounts|\/policies\/|save\.google|preferences\?)/i;
      const LABEL_ONLY = /^(help|sign in|privacy|terms|settings|feedback|tools|any time|past hour|past day|past week|past month|more settings|google apps|search|images|videos|news|maps|scholar|translate)/i;
      const out = [];
      const seen = new Set();
      const scope = firstMessageRoots(cfg)[0] || document;
      const anchors = Array.from(scope.querySelectorAll('a[href^="http"]'));
      for (const a of anchors) {
        let href = a.href;
        if (!href || seen.has(href)) continue;
        let real = href;
        if (/\/url\?q=/.test(href)) { try { real = new URLSearchParams(href.split('?')[1] || '').get('q') || href; } catch (e) {} }
        if (!/^https?:\/\//.test(real)) continue;
        let host = '';
        try { host = new URL(real).host.toLowerCase(); } catch (e) { continue; }
        if (JUNK_HOST.test(host) || JUNK_PATH.test(real)) continue;
        if (/(^|\.)google\.[a-z]{2,3}$/.test(host)) continue;
        const holder = a.closest('h2,h3,.g,div.s,[data-hveid],.b_algo,li,[class*="result"],[class*="result__body"]') || a.parentElement;
        const title = norm(a.innerText || a.getAttribute('aria-label') || '');
        if (!title || title.length < 12 || LABEL_ONLY.test(title)) continue;
        const block = holder ? norm(holder.innerText) : '';
        if (block.length < title.length + 25) continue;
        const snippet = block.slice(0, 420);
        seen.add(real);
        out.push({
          href: real, title: title.slice(0, 220), snippet, host,
          marker: null, inSourceList: false,
          sitelink: host,
        });
        if (out.length >= (limit || 12)) break;
      }
      return {items: out, pageText: norm(scope.innerText).slice(0, 3000), visibility: document.visibilityState, anchorsSeen: anchors.length};
    },
  };

  Object.defineProperty(api, '_hidden', {value: true, enumerable: false});
  window.__omnibrain = api;
  return 'installed';
})();
"""
