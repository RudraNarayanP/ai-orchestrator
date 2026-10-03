"""OpenAI-compatible text client for the verifier and the analysis stages.

Ollama, LM Studio, OpenRouter and any custom ``BASE_URL`` all speak roughly the
same dialect, so switching verifier substrate is a config edit, not an
architecture change (spec section 24).

Everything here degrades on purpose: a missing local model must produce a
clearly-labelled fallback, not a crashed research job and not a silent lie.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "lm_studio": "http://localhost:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass
class LLMReply:
    text: str
    ok: bool
    error: str | None = None
    latency_s: float = 0.0
    model: str | None = None
    usage: dict[str, Any] | None = None
    status: int = 0
    fatal: bool = False
    """True when no other model can help (bad key, forbidden, server unreachable)."""


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Endpoint:
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_s: int = 180
    fallback_models: tuple[str, ...] = ()
    headers: dict[str, str] | None = None
    """Extra request headers, e.g. OpenRouter's HTTP-Referer / X-Title."""

    @classmethod
    def from_config(cls, cfg: Any) -> "Endpoint":
        base = (cfg.base_url or "").strip() or DEFAULT_BASE_URLS.get(cfg.provider, "")
        key = cfg.api_key or ""
        if not key and cfg.provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY", "")
        return cls(
            provider=cfg.provider,
            model=cfg.model,
            base_url=base.rstrip("/"),
            api_key=key,
            fallback_models=tuple(getattr(cfg, "fallback_models", None) or ()),
            headers=dict(getattr(cfg, "headers", None) or {}) or None,
            temperature=float(cfg.temperature),
            max_tokens=int(cfg.max_tokens),
            timeout_s=int(getattr(cfg, "timeout_s", 180) or 180),
        )

    @property
    def enabled(self) -> bool:
        return self.provider != "disabled" and bool(self.base_url and self.model)


class LLMClient:
    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        elif self.endpoint.provider == "openrouter":
            headers["Authorization"] = "Bearer MISSING_OPENROUTER_KEY"
        if self.endpoint.provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "OmniBrain"
        headers.update(self.endpoint.headers or {})
        return headers

    async def complete(self, messages: list[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None, retries: int = 1) -> LLMReply:
        if not self.endpoint.enabled:
            return LLMReply(text="", ok=False, error="endpoint disabled or unconfigured", model=self.endpoint.model)
        # Free hosted models are rate limited and sometimes vanish; try the configured fallbacks in order, but only
        # for failures a different model can fix (limits, upstream errors, unknown model, empty output).
        models = [self.endpoint.model] + [m for m in self.endpoint.fallback_models if m and m != self.endpoint.model]
        reply = LLMReply(text="", ok=False, error="no model tried", model=self.endpoint.model)
        for model in models:
            reply = await self._complete_with(model, messages, temperature, max_tokens, retries)
            if reply.ok or reply.fatal:
                return reply
        return reply

    async def _complete_with(self, model: str, messages: list[dict[str, str]], temperature: float | None, max_tokens: int | None, retries: int) -> LLMReply:
        url = f"{self.endpoint.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.endpoint.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.endpoint.max_tokens,
            "stream": False,
        }
        # Reasoning models will happily spend the whole budget thinking and
        # return nothing. Ollama exposes a knob for it; ask for the answer.
        if self.endpoint.provider in {"ollama", "lm_studio"}:
            payload["think"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last_error = ""
        status = 0
        started = time.time()
        attempt = 0
        limit_waits = 0
        attempts = max(1, retries + 1)
        while attempt < attempts:
            attempt += 1
            started = time.time()
            delay = 1.5
            try:
                async with httpx.AsyncClient(timeout=self.endpoint.timeout_s) as client:
                    response = await client.post(url, json=payload, headers=self._headers())
                    status = response.status_code
                    if status >= 400:
                        last_error = f"HTTP {status}: {_scrub(response.text)[:300]}"
                        if status in {401, 403}:
                            return LLMReply(text="", ok=False, error=last_error, latency_s=round(time.time() - started, 2), model=model, status=status, fatal=True)
                        if status == 404:
                            break
                        if status == 429:
                            # A rate limit is not a failed attempt: wait as told (bounded) and ask again.
                            retry_after = _retry_after(response.headers.get("retry-after"))
                            delay = min(retry_after if retry_after is not None else 2.0 * (2 ** limit_waits), 20.0)
                            limit_waits += 1
                            if limit_waits <= 2:
                                attempts = max(attempts, attempt + 1)
                        await _sleep(delay)
                        continue
                    data = response.json()
                    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
                        # OpenRouter reports upstream failures with HTTP 200 and an error object.
                        err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
                        code = err.get("code")
                        last_error = f"upstream error {code or ''}: {_scrub(str(err.get('message') or ''))[:240]}".strip()
                        status = int(code) if str(code).isdigit() else 502
                        if status in {401, 403}:
                            return LLMReply(text="", ok=False, error=last_error, latency_s=round(time.time() - started, 2), model=model, status=status, fatal=True)
                        if status == 404:
                            break
                        await _sleep(2.0 if status == 429 else 1.5)
                        continue
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    text = message.get("content") or choice.get("text") or ""
                    if not text and message.get("reasoning") and choice.get("finish_reason") != "length":
                        # Some templates put the answer only in reasoning -- but a reasoning trace cut off by the
                        # token limit is scratch work, not an answer (live: "The user wants the exact word...").
                        text = message["reasoning"]
                    return LLMReply(
                        text=text.strip(),
                        ok=bool(text.strip()),
                        error=None if text.strip() else "empty completion",
                        latency_s=round(time.time() - started, 2),
                        model=model,
                        usage=data.get("usage"),
                        status=200,
                    )
            except httpx.TimeoutException:
                last_error = f"timeout after {self.endpoint.timeout_s}s"
            except httpx.ConnectError as exc:
                last_error = f"cannot reach {self.endpoint.base_url} ({exc}). Is {self.endpoint.provider} running?"
                return LLMReply(text="", ok=False, error=last_error, latency_s=round(time.time() - started, 2), model=model, fatal=True)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {_scrub(str(exc))}"
            if attempt < attempts:
                await _sleep(delay)
        return LLMReply(text="", ok=False, error=last_error, latency_s=round(time.time() - started, 2), model=model, status=status)

    async def complete_json(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> tuple[dict[str, Any] | None, LLMReply]:
        reply = await self.complete(messages, temperature=temperature)
        if not reply.ok:
            return None, reply
        return extract_json(reply.text), reply

    async def list_models(self) -> list[str]:
        if not self.endpoint.base_url:
            return []
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                response = await client.get(f"{self.endpoint.base_url}/models", headers=self._headers())
                data = response.json()
                items = data.get("data") or data.get("models") or []
                return [str(i.get("id") or i.get("name")) for i in items if (i.get("id") or i.get("name"))]
        except Exception:  # noqa: BLE001
            return []

    async def check_key(self) -> tuple[bool, str]:
        """OpenRouter: is the key accepted, and is the account on the free tier? Never returns the key."""
        if not self.endpoint.api_key:
            return False, "no OpenRouter API key configured (verifier/analysis api_key or OPENROUTER_API_KEY)"
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                response = await client.get(f"{self.endpoint.base_url}/auth/key", headers=self._headers())
        except Exception as exc:  # noqa: BLE001
            return False, f"could not check the OpenRouter key: {type(exc).__name__}"
        if response.status_code in {401, 403}:
            return False, "OpenRouter rejected the API key (HTTP %d)" % response.status_code
        if response.status_code >= 400:
            return True, f"key check unavailable (HTTP {response.status_code})"
        try:
            info = response.json().get("data") or {}
        except ValueError:
            return True, "key accepted"
        tier = "free tier" if info.get("is_free_tier") else "paid account"
        return True, f"key accepted, {tier}"

    async def health(self) -> dict[str, Any]:
        if not self.endpoint.enabled:
            return {"ok": False, "state": "disabled", "detail": "verifier provider set to disabled"}
        models = await self.list_models()
        if not models:
            return {
                "ok": False,
                "state": "unreachable",
                "detail": f"no model list from {self.endpoint.base_url} -- start {self.endpoint.provider} or fix BASE_URL",
            }
        if self.endpoint.provider == "openrouter":
            # Exact id only: "google/gemma:free" and its paid sibling are different products with different limits.
            present = self.endpoint.model in models
        else:
            present = self.endpoint.model in models or any(m.split(":")[0] == self.endpoint.model.split(":")[0] for m in models)
        key_note = ""
        if self.endpoint.provider == "openrouter":
            key_ok, key_note = await self.check_key()
            if not key_ok:
                return {"ok": False, "state": "auth_failed", "detail": key_note, "models": models[:40]}
        return {
            "ok": True,
            "state": "ready" if present else "model_missing",
            "detail": (
                f"{self.endpoint.model} available" + (f" ({key_note})" if key_note else "")
                if present
                else f"{self.endpoint.model} not pulled on this server; {len(models)} model(s) available"
            ),
            "models": models[:40],
        }


def _scrub(text: str) -> str:
    """Never let a key echoed back by a server reach an error string (and from there a log or the UI)."""
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", text or "")


def _retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def extract_json(text: str) -> dict[str, Any] | None:
    """Small models wrap JSON in prose, fences, or a trailing sentence.

    Try, in order: raw parse, fenced block, balanced-object scan, then a
    single-key rescue. Return None rather than guessing.
    """
    if not text:
        return None
    candidates: list[str] = []
    # Reasoning models put their scratch work in <think> blocks, which may contain braces.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    stripped = text.strip()
    candidates.append(stripped)
    for fence in re.findall(r"```(?:json)?\s*(.+?)```", text, re.S | re.I):
        candidates.append(fence.strip())
    obj = _balanced(stripped, "{", "}")
    if obj:
        candidates.append(obj)
    arr = _balanced(stripped, "[", "]")
    first_obj, first_arr = stripped.find("{"), stripped.find("[")
    if arr and first_obj != -1 and first_obj < first_arr:
        arr = None  # an array inside an (unparsable) object is a rescue case, not a bare list
    if arr:
        candidates.append(arr)
    for cand in candidates:
        for attempt in (cand, _tidy(cand)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
    # Single-key rescue, only after every whole-object candidate has failed:
    # {"claims": [ ... ]} with a broken tail. Doing it per candidate used to let a
    # fenced reply be "rescued" down to its first array, dropping answer/confidence.
    for cand in candidates:
        match = re.search(r'"([a-z_]+)"\s*:\s*(\[[\s\S]*\])', cand)
        if match:
            try:
                return {match.group(1): json.loads(_tidy(match.group(2)))}
            except json.JSONDecodeError:
                continue
    return None


def _balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _tidy(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\r", " ")
    )
