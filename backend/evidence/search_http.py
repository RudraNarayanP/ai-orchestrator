"""HTTP discovery transport for the evidence layer.

The chat providers are driven through real browser windows -- that is the point of
OmniBrain. Finding *sources* is a different job: hitting an HTML search endpoint
over HTTP is faster, doesn't consume a browser session per query, and keeps working
when one engine's browser page decides to show a robot check.

Nothing here evades anything. If an endpoint refuses, we report it and move to the
next engine.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from backend.evidence.sources import UA

ENDPOINTS = {
    "duckduckgo": "https://html.duckduckgo.com/html/?q={query}",
    "duckduckgo_lite": "https://lite.duckduckgo.com/lite/?q={query}",
    "bing": "https://www.bing.com/search?q={query}&count={count}",
    "brave": "https://search.brave.com/search?q={query}",
}

_RESULT_ANCHOR = re.compile(
    r'<a[^>]+href="(?P<href>[^"]+)"[^>]*class="[^"]*(?:result__a|b_algo)[^"]*"[^>]*>(?P<title>.*?)</a>',
    re.S | re.I,
)
_DDG_LINK = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S | re.I)
_DDG_SNIPPET = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snip>.*?)</a>', re.S | re.I)
_BING_BLOCK = re.compile(r'<li class="b_algo".*?</li>', re.S | re.I)
_H2_LINK = re.compile(r'<h2[^>]*>\s*<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S | re.I)
_CAPTION = re.compile(r'<p[^>]*>(?P<snip>.*?)</p>', re.S | re.I)
_TAG = re.compile(r"<[^>]+>")

AD_HOSTS = re.compile(r"(googlesyndication|doubleclick|ads\.|/ads/|bing\.com/search\?q=|facebook\.com/ads)", re.I)


def _clean(html: str) -> str:
    return " ".join(unescape(_TAG.sub(" ", html or "")).split()).strip()


def _resolve(href: str) -> str | None:
    """DuckDuckGo wraps targets in a redirect; unwrap it, reject non-http."""
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href or "/url?" in href:
        try:
            query = href.split("?", 1)[1]
            target = (parse_qs(query).get("uddg") or parse_qs(query).get("q") or [""])[0]
            href = unquote(target) or href
        except Exception:  # noqa: BLE001
            pass
    if not href.startswith("http"):
        return None
    if AD_HOSTS.search(href):
        return None
    return href


def _parse_ddg(html: str, limit: int) -> list[dict[str, Any]]:
    links = [m.groupdict() for m in _DDG_LINK.finditer(html)]
    snippets = [m.group("snip") for m in _DDG_SNIPPET.finditer(html)]
    out: list[dict[str, Any]] = []
    for index, item in enumerate(links[:limit]):
        url = _resolve(item.get("href") or "")
        title = _clean(item.get("title") or "")
        if not url or len(title) < 10:
            continue
        out.append(
            {
                "href": url,
                "title": title[:220],
                "snippet": (_clean(snippets[index]) if index < len(snippets) else "")[:420],
                "host": (urlparse(url).netloc or "").lower(),
                "origin": "search",
            }
        )
    return out


def _parse_bing(html: str, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in _BING_BLOCK.finditer(html):
        chunk = block.group(0)
        link = _H2_LINK.search(chunk)
        if not link:
            continue
        url = _resolve(link.group("href") or "")
        title = _clean(link.group("title") or "")
        caption = _CAPTION.search(chunk)
        if not url or len(title) < 10:
            continue
        out.append(
            {
                "href": url,
                "title": title[:220],
                "snippet": (_clean(caption.group("snip")) if caption else "")[:420],
                "host": (urlparse(url).netloc or "").lower(),
                "origin": "search",
            }
        )
        if len(out) >= limit:
            break
    return out


PARSERS = {
    "duckduckgo": _parse_ddg,
    "duckduckgo_lite": _parse_ddg,
    "brave": _parse_ddg,
    "bing": _parse_bing,
}


async def search(query: str, *, engines: list[str] | None = None, limit: int = 8, timeout_s: int = 25) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return result links plus a trace of which engine answered."""
    engines = engines or ["duckduckgo", "bing"]
    trace: dict[str, Any] = {"tried": [], "used": None}
    for engine in engines:
        template = ENDPOINTS.get(engine)
        if not template:
            continue
        url = template.format(query=quote_plus(query), count=max(limit, 10))
        entry: dict[str, Any] = {"engine": engine, "url": url}
        try:
            async with httpx.AsyncClient(
                timeout=timeout_s,
                follow_redirects=True,
                headers={
                    "User-Agent": UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            ) as client:
                response = await client.get(url)
            html = response.text or ""
            entry["status"] = response.status_code
            entry["bytes"] = len(html)
            parsed = PARSERS[engine](html, limit)
            entry["results"] = len(parsed)
            trace["tried"].append(entry)
            if parsed:
                trace["used"] = engine
                return parsed, trace
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            trace["tried"].append(entry)
    return [], trace
