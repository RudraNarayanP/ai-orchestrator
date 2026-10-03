"""OpenAI-compatible text client for the verifier and the analysis stages.

Ollama, LM Studio, OpenRouter and any custom ``BASE_URL`` all speak roughly the
same dialect, so switching verifier substrate is a config edit, not an
architecture change (spec section 24).

Everything here degrades on purpose: a missing local model must produce a
clearly-labelled fallback, not a crashed research job and not a silent lie.
"""

from __future__ import annotations

import json
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

    @classmethod
    def from_config(cls, cfg: Any) -> "Endpoint":
        base = (cfg.base_url or "").strip() or DEFAULT_BASE_URLS.get(cfg.provider, "")
        return cls(
            provider=cfg.provider,
            model=cfg.model,
            base_url=base.rstrip("/"),
            api_key=cfg.api_key or "",
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
        return headers

    async def complete(self, messages: list[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None, retries: int = 1) -> LLMReply:
        if not self.endpoint.enabled:
            return LLMReply(text="", ok=False, error="endpoint disabled or unconfigured", model=self.endpoint.model)
        url = f"{self.endpoint.base_url}/chat/completions"
        payload = {
            "model": self.endpoint.model,
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
        for attempt in range(max(1, retries + 1)):
            started = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.endpoint.timeout_s) as client:
                    response = await client.post(url, json=payload, headers=self._headers())
                    if response.status_code >= 400:
                        last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                        if response.status_code in {401, 403, 404}:
                            break
                        continue
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    text = message.get("content") or choice.get("text") or ""
                    if not text and message.get("reasoning"):
                        # Some templates put the answer only in reasoning.
                        text = message["reasoning"]
                    return LLMReply(
                        text=text.strip(),
                        ok=bool(text.strip()),
                        error=None if text.strip() else "empty completion",
                        latency_s=round(time.time() - started, 2),
                        model=self.endpoint.model,
                        usage=data.get("usage"),
                    )
            except httpx.TimeoutException:
                last_error = f"timeout after {self.endpoint.timeout_s}s"
            except httpx.ConnectError as exc:
                last_error = f"cannot reach {self.endpoint.base_url} ({exc}). Is {self.endpoint.provider} running?"
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries + 1:
                await _sleep(1.5)
        return LLMReply(text="", ok=False, error=last_error, latency_s=round(time.time() - started, 2), model=self.endpoint.model)

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
        present = self.endpoint.model in models or any(m.split(":")[0] == self.endpoint.model.split(":")[0] for m in models)
        return {
            "ok": True,
            "state": "ready" if present else "model_missing",
            "detail": (
                f"{self.endpoint.model} available"
                if present
                else f"{self.endpoint.model} not pulled on this server; {len(models)} model(s) available"
            ),
            "models": models[:40],
        }


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
