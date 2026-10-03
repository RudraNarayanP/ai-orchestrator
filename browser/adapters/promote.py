"""Selector promotion: turn a probe dump into ``verified="probe"`` selectors.

``scripts/probe.py`` writes ``data/probe/<provider>_<epoch>.json`` with what the real
page contained; nothing used to read it. ``python run.py promote-selectors <provider>``
does, conservatively:

* only the composer (``input``) and the send button (``send``) are promoted, because
  those are the only things a probe can actually observe (a stop button exists only
  mid-generation, so it stays ``prior``);
* the observed locators go to the FRONT of the existing ladder and the old entries
  stay behind them as fallbacks -- a probe on a bad day (consent wall, A/B variant)
  must not delete selectors that work;
* the result is written to ``browser/adapters/promoted_selectors.json`` (version it
  with git) and ``selectors_for`` overlays it, so no Python source is rewritten;
* a probe is refused if it is stale in meaning: wrong host, an error, a robot check,
  or no visible composer (i.e. you were not signed in);
* it never promotes a login / sign-in / consent / upgrade button as "send".

It reads a local JSON file and touches no browser, account, or credential.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from browser.adapters.selectors import PROMOTED_PATH, load_promotions

SEND_LABEL_RE = re.compile(r"^\s*(send|submit|ask|go)\b|\bsend (message|prompt)\b|\bsubmit\b", re.I)
DENY_RE = re.compile(
    r"log ?in|sign ?in|sign ?up|continue with|accept|allow|agree|retry|try again|subscribe|upgrade|"
    r"account|cookie|consent|captcha|verify|password|human",
    re.I,
)
BLOCK_BANNER_RE = re.compile(r"(are you a robot|unusual traffic|verify you are a human|captcha|access denied)", re.I)
GENERATED_ID_RE = re.compile(r"(\d{5,}|[0-9a-f]{10,}|^:r\w+:?$|^radix-|^headlessui-)", re.I)
LOGIN_PATH_RE = re.compile(r"/(sign[_-]?in|log[_-]?in|sign[_-]?up|register|auth|sso|oauth|accounts?/login)(/|$|\?)", re.I)
CREDENTIAL_FIELD_RE = re.compile(r"e-?mail|phone|mobile number|password|passcode|username|user name|verification code|one-time|otp\b", re.I)
COMPOSER_TAGS = {"textarea": 3, "div": 3, "p": 3, "span": 2, "input": 1}


class PromotionError(ValueError):
    """The probe cannot be promoted; the message says why and what to do."""


@dataclass
class PromotionResult:
    provider: str
    source: str
    observed_at: str
    promoted: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    written: bool = False
    path: str = ""


def newest_probe(provider: str, probe_dir: Path) -> Path | None:
    candidates = [p for p in probe_dir.glob(f"{provider}_*.json") if re.fullmatch(rf"{re.escape(provider)}_\d+\.json", p.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (int(p.stem.rsplit("_", 1)[1]), p.stat().st_mtime))


def _host(url: str) -> str:
    return (urlparse(url or "").netloc or "").lower().removeprefix("www.")


def check_probe(probe: dict[str, Any], provider_url: str) -> list[str]:
    """Reasons this probe must not be promoted (empty = fine)."""
    problems: list[str] = []
    if probe.get("error"):
        problems.append(f"the probe recorded an error ({probe['error']})")
    observed = probe.get("observed")
    if not isinstance(observed, dict):
        problems.append("the probe has no 'observed' section")
        return problems
    seen = observed.get("url_seen") or ""
    if provider_url and _host(seen) != _host(provider_url):
        problems.append(f"the probe landed on {_host(seen) or 'nothing'} but this provider is {_host(provider_url)}")
    if LOGIN_PATH_RE.search(urlparse(seen).path or ""):
        problems.append(f"the probe landed on a sign-in page ({urlparse(seen).path}) -- this provider does not answer anonymously right now; nothing to promote")
    banners = " ".join(str(b.get("text") or "") for b in observed.get("banners", []) or [])
    if BLOCK_BANNER_RE.search(banners + " " + str(observed.get("title") or "")):
        problems.append("the page showed a robot/access check -- complete it yourself in the window, then probe again")
    return problems


def _score(item: dict[str, Any]) -> tuple[int, int]:
    rect = item.get("rect") or {}
    kind = 3 if (item.get("contenteditable") not in (None, "false") or item.get("tag") == "textarea") else COMPOSER_TAGS.get(item.get("tag"), 1)
    if item.get("role") == "textbox":
        kind = max(kind, 2)
    return kind, int((rect.get("w") or 0) * (rect.get("h") or 0))


def _stable_id(value: str | None) -> bool:
    return bool(value) and not GENERATED_ID_RE.search(value or "")


def _locators(item: dict[str, Any], *, buttons: bool = False) -> dict[str, Any]:
    tag = item.get("tag") or "*"
    css: list[str] = []
    if _stable_id(item.get("id")):
        css.append(f"#{item['id']}")
    testid = item.get("testid")
    if testid and _stable_id(testid):
        css.append(f'{tag}[data-testid="{testid}"]')
    if not buttons and item.get("contenteditable") not in (None, "false") and item.get("role") == "textbox":
        css.append(f'{tag}[contenteditable="true"][role="textbox"]')
    out: dict[str, Any] = {"css": css}
    if item.get("aria"):
        out["aria"] = [item["aria"]]
    if testid and _stable_id(testid):
        out["testids"] = [testid]
    if item.get("placeholder") and not buttons:
        out["placeholders"] = [item["placeholder"]]
    return out


def fieldsets_from_probe(observed: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    promoted: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

    inputs = [i for i in observed.get("inputs", []) or [] if i.get("visible")]
    credential = [i for i in inputs if i.get("type") == "password" or CREDENTIAL_FIELD_RE.search(" ".join(str(i.get(k) or "") for k in ("placeholder", "aria", "id", "name", "label")))]
    if credential and len(credential) == len(inputs):
        raise PromotionError("the only visible inputs are login fields (email/phone/password) -- never promoted as a composer")
    inputs = [i for i in inputs if i not in credential]
    if not inputs:
        raise PromotionError(
            "the probe saw no visible composer -- you probably were not signed in. "
            "Run `python run.py login <provider>` (or `probe <provider> --login`) and probe again."
        )
    composer = max(inputs, key=_score)
    locators = _locators(composer)
    if not any(locators.get(k) for k in ("css", "aria", "testids", "placeholders")):
        raise PromotionError("the composer had no stable identifier (no id, test id, aria-label or placeholder) to promote")
    promoted["input"] = locators

    sends = []
    for button in observed.get("buttons", []) or []:
        label = (button.get("aria") or button.get("text") or "").strip()
        if not button.get("visible") or not label:
            continue
        if DENY_RE.search(label) or DENY_RE.search(str(button.get("testid") or "")):
            skipped.append(f"ignored button {label!r}: looks like login/consent/upgrade, never promoted as send")
            continue
        if SEND_LABEL_RE.search(label) or SEND_LABEL_RE.search(str(button.get("testid") or "").replace("-", " ")):
            sends.append(button)
    if sends:
        best = sends[0]
        entry = _locators(best, buttons=True)
        label = (best.get("aria") or best.get("text") or "").strip()
        entry["text_regex"] = "^(" + re.escape(label.lower()) + ")$" if not best.get("aria") else None
        if entry["text_regex"] is None:
            entry.pop("text_regex")
        if any(entry.get(k) for k in ("css", "aria", "testids", "text_regex")):
            promoted["send"] = entry
    else:
        skipped.append("no send button in the probe (it usually only appears once text is typed) -- send left unchanged")
    return promoted, skipped


def promote(
    provider: str,
    *,
    provider_url: str,
    probe_dir: Path,
    probe_path: Path | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
) -> PromotionResult:
    source = probe_path or newest_probe(provider, probe_dir)
    if source is None or not Path(source).exists():
        raise PromotionError(f"no probe for {provider!r} in {probe_dir} -- run `python run.py probe {provider} --login` first")
    try:
        probe = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot read {source}: {exc}") from None
    problems = check_probe(probe, provider_url)
    if problems:
        raise PromotionError("; ".join(problems))
    promoted, skipped = fieldsets_from_probe(probe["observed"])
    observed_at = str(probe.get("ts") or time.strftime("%Y-%m-%d %H:%M:%S"))
    stamp = f"{observed_at} (promoted from {Path(source).name})"
    for entry in promoted.values():
        entry["verified"] = "probe"
        entry["verified_at"] = stamp

    out = Path(out_path) if out_path else PROMOTED_PATH
    result = PromotionResult(provider=provider, source=str(source), observed_at=observed_at, promoted=promoted, skipped=skipped, path=str(out))
    if dry_run:
        return result
    existing = load_promotions(out)
    merged = dict(existing.get(provider) or {})
    merged.update(promoted)
    existing[provider] = merged
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    result.written = True
    return result