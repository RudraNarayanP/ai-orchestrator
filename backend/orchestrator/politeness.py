"""Rate-limit politeness: space out questions to one site, and back off after a limit.

Before this, a ``rate_limited`` result was retried by the adapter immediately and
the next round asked the same site again with no pause. Chat sites throttle that
and eventually block it, and hammering a limit is not something we should do.

* minimum spacing between one site's questions (``research.min_provider_spacing_s``);
* after a ``rate_limited`` result, exponential backoff for that site:
  ``rate_limit_backoff_s`` x 2 per consecutive limit, capped at
  ``rate_limit_backoff_max_s``, reset by a completed answer;
* a short remaining backoff is waited out; a long one is not -- the site is skipped
  and reported as rate-limited so escalation uses someone else, instead of the job
  sitting for minutes.

The gate is shared by every job in the process, so the backoff survives from one
question to the next. It never retries or bypasses anything; it only waits or declines.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from backend.settings import ResearchConfig


@dataclass
class _Site:
    last_end: float | None = None
    backoff_until: float = 0.0
    consecutive_limits: int = 0


class PolitenessGate:
    def __init__(
        self,
        cfg: ResearchConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.sleep = sleep
        self._sites: dict[str, _Site] = {}

    def _site(self, provider: str) -> _Site:
        return self._sites.setdefault(provider, _Site())

    def backoff_remaining(self, provider: str) -> float:
        return max(0.0, self._site(provider).backoff_until - self.clock())

    async def before(self, provider: str, sleep: Callable[[float], Awaitable[None]] | None = None) -> str | None:
        """Wait as needed. Returns a reason string if the site should be skipped instead."""
        site = self._site(provider)
        now = self.clock()
        remaining = max(0.0, site.backoff_until - now)
        if remaining > float(self.cfg.rate_limit_max_wait_s):
            return f"backing off {int(remaining)}s after a rate limit ({site.consecutive_limits} in a row); not asking again yet"
        spacing = 0.0
        if site.last_end is not None:
            spacing = max(0.0, site.last_end + float(self.cfg.min_provider_spacing_s) - now)
        wait = max(remaining, spacing)
        if wait > 0:
            await (sleep or self.sleep)(wait)
        return None

    def after(self, provider: str, status: str) -> None:
        site = self._site(provider)
        site.last_end = self.clock()
        if status == "rate_limited":
            site.consecutive_limits += 1
            delay = min(
                float(self.cfg.rate_limit_backoff_max_s),
                float(self.cfg.rate_limit_backoff_s) * (2 ** (site.consecutive_limits - 1)),
            )
            site.backoff_until = site.last_end + delay
        elif status == "completed":
            site.consecutive_limits = 0
            site.backoff_until = 0.0