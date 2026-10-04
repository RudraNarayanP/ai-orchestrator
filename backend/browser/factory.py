"""Pick the browser driver from settings: the dedicated Playwright profile (default)
or the opt-in live-Chrome driver (``browser.driver: chrome_use``)."""

from __future__ import annotations

from typing import Any

from backend.settings import Settings


def create_engine(settings: Settings) -> Any:
    if settings.browser.driver == "chrome_use":
        from backend.browser.live_chrome import LiveChromeEngine

        return LiveChromeEngine(settings)
    from backend.browser.engine import BrowserEngine

    return BrowserEngine(settings)
