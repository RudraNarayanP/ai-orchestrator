"""Cooperative cancellation.

``asyncio.Task.cancel()`` interrupts the next ``await`` of a job, but a browser step
that is deep in its own poll loop (or a ``gather`` that swallows a child's
``CancelledError``) can keep going. A ``CancelToken`` is shared by the job manager,
the runner and every adapter so that stopping a job is visible everywhere:

* adapters check it on every poll and sleep through it, so a mid-stream wait ends
  within a poll interval;
* the runner checks it between stages, so a cancelled child inside ``gather`` cannot
  let the job carry on to verification;
* on cancel the adapter closes its own provider tab, so a half-generated answer is
  not left streaming in the dedicated window.
"""

from __future__ import annotations

import asyncio


class JobCancelled(asyncio.CancelledError):
    """Raised at a checkpoint after the token was cancelled.

    A ``CancelledError`` subclass on purpose: every ``except asyncio.CancelledError:
    raise`` already in the code base lets it through, and ``except Exception`` cannot
    swallow it.
    """


class CancelToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancelled by user") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelled(self.reason or "cancelled")

    async def sleep(self, seconds: float) -> None:
        """``asyncio.sleep`` that ends early, by raising, when the token is cancelled."""
        self.raise_if_cancelled()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=max(0.0, seconds))
        except asyncio.TimeoutError:
            return
        self.raise_if_cancelled()