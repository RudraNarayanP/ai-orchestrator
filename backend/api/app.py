"""HTTP API: job control, live event stream, configuration, diagnostics.

The frontend is a chat box. Everything that makes OmniBrain more than a chat box
-- the escalation log, per-provider health, raw captures, the evidence ledger --
is reachable but tucked behind it, because the answer is what the user asked for.

API keys never cross this boundary: /api/config returns them masked.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.browser.engine import BrowserEngine
from backend.models import (
    EscalationLevel,
    Job,
    JobEvent,
    JobStatus,
    ResearchMode,
)
from backend.orchestrator.runner import ResearchRunner
from backend.providers.registry import ProviderCatalog, endpoint_for
from backend.settings import Settings, load_settings, reload_settings, save_settings
from backend.storage.db import Store
from backend.verification.llm import LLMClient

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"


class EventBroker:
    """Replayable per-job stream, so a page refresh mid-research loses nothing."""

    def __init__(self) -> None:
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._seq: dict[str, int] = {}

    def publish(self, job_id: str, event: JobEvent) -> dict[str, Any]:
        self._seq[job_id] = self._seq.get(job_id, 0) + 1
        record = {"seq": self._seq[job_id], **event.model_dump(mode="json")}
        buffer = self._buffers.setdefault(job_id, [])
        buffer.append(record)
        if len(buffer) > 2000:
            del buffer[: len(buffer) - 2000]
        for queue in self._subscribers.get(job_id, ()):
            queue.put_nowait(record)
        return record

    def since(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        return [e for e in self._buffers.get(job_id, []) if e["seq"] > after]

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        self._subscribers.get(job_id, set()).discard(queue)

    def forget(self, job_id: str) -> None:
        self._buffers.pop(job_id, None)
        self._subscribers.pop(job_id, None)


class JobManager:
    def __init__(self, settings: Settings, store: Store, broker: EventBroker) -> None:
        self.settings = settings
        self.store = store
        self.broker = broker
        self.engine: BrowserEngine | None = None
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.Semaphore = asyncio.Semaphore(1)
        self.health: dict[str, str] = {}

    async def engine_get(self) -> BrowserEngine:
        if self.engine is None:
            self.engine = BrowserEngine(self.settings)
            await self.engine.start()
        return self.engine

    def emit_factory(self, job_id: str) -> Callable[..., Awaitable[None]]:
        async def emit(
            kind: str,
            message: str,
            provider: str | None = None,
            round_no: int | None = None,
            **payload: Any,
        ) -> None:
            event = JobEvent(kind=kind, message=message, provider=provider, round=round_no, payload=payload)
            self.broker.publish(job_id, event)
            try:
                self.store.add_event(job_id, event)
            except Exception:  # noqa: BLE001
                pass

        return emit

    async def start(self, question: str, mode: str | None, max_rounds: int | None) -> Job:
        question = (question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        try:
            resolved_mode = ResearchMode((mode or self.settings.research.mode).upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}") from None
        job = Job(
            question=question,
            mode=resolved_mode,
            max_rounds=int(max_rounds or self.settings.research.max_rounds),
        )
        self.jobs[job.id] = job
        self.store.save_job(job)
        emit = self.emit_factory(job.id)
        await emit("status", f"queued: {question[:90]}")
        self.tasks[job.id] = asyncio.create_task(self._run(job, emit))
        return job

    async def _run(self, job: Job, emit: Callable[..., Awaitable[None]]) -> None:
        async with self._queue:
            try:
                engine = await self.engine_get()
                catalog = ProviderCatalog(self.settings, engine)
                adapters = catalog.all()
                runner = ResearchRunner(
                    self.settings,
                    adapters,
                    engine=engine,
                    bus=_Bus(emit),
                    health=self.store.health() or None,
                    analysis_endpoint=endpoint_for(self.settings, "analysis"),
                )
                await emit("status", f"starting with {len(adapters)} enabled providers")
                await runner.run(job)
                for provider, status in runner.health.items():
                    self.store.record_health(provider, status)
                self.health = self.store.health()
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.stop_reason = "cancelled"
                await emit("status", "cancelled")
            except Exception as exc:  # noqa: BLE001
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                await emit("error", job.error)
            finally:
                job.updated_at = time.time()
                job.rounds_run = max([r.number for r in job.rounds], default=job.rounds_run)
                self.store.save_job(job)
                await emit("done", job.status.value, payload={"final": bool(job.final)})

    async def cancel(self, job_id: str) -> bool:
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def open_login_window(self, provider: str) -> dict[str, Any]:
        """Open that provider's dedicated window so the user can sign in once."""
        cfg = self.settings.providers.get(provider)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
        engine = await self.engine_get()
        page = await engine.open_research_page(provider, cfg.url)
        await page.bring_to_front()
        return {
            "provider": provider,
            "url": page.url,
            "profile": provider,
            "instruction": (
                f"A dedicated OmniBrain window for {cfg.label} is now open -- sign in there once. "
                "It uses its own browser profile and never touches the Chrome you work in."
            ),
        }


class _Bus:
    """Adapts the broker's emit closure to the runner's EventBus protocol."""

    def __init__(self, emit: Callable[..., Awaitable[None]]) -> None:
        self._emit = emit

    async def emit(
        self,
        kind: str,
        message: str,
        provider: str | None = None,
        round_no: int | None = None,
        **payload: Any,
    ) -> None:
        await self._emit(kind, message, provider=provider, round_no=round_no, **payload)


def _make_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = Store(settings)
    broker = EventBroker()
    manager = JobManager(settings, store, broker)
    app = FastAPI(title="OmniBrain", version="0.1.0")
    app.state.settings = settings
    app.state.store = store
    app.state.broker = broker
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        page = FRONTEND / "index.html"
        if not page.exists():
            return "<h1>OmniBrain</h1><p>frontend/index.html missing</p>"
        return page.read_text(encoding="utf-8")

    @app.get("/app.js")
    async def app_js() -> HTMLResponse:
        return HTMLResponse((FRONTEND / "app.js").read_text(encoding="utf-8"), media_type="text/javascript")

    @app.get("/styles.css")
    async def styles_css() -> HTMLResponse:
        return HTMLResponse((FRONTEND / "styles.css").read_text(encoding="utf-8"), media_type="text/css")

    @app.post("/api/jobs")
    async def create_job(request: Request) -> JSONResponse:
        body = await request.json()
        job = await manager.start(
            body.get("question", ""),
            body.get("mode"),
            body.get("max_rounds"),
        )
        return JSONResponse({"job_id": job.id, "status": job.status.value})

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50) -> Any:
        return store.list_jobs(limit=limit)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> Any:
        snapshot = store.job_snapshot(job_id)
        live = manager.jobs.get(job_id)
        if snapshot is None and live is None:
            raise HTTPException(status_code=404, detail="no such job")
        if live is not None:
            snapshot = snapshot or {}
            snapshot["live"] = live.model_dump(mode="json")
        return snapshot

    @app.get("/api/jobs/{job_id}/raw")
    async def raw_audit(job_id: str) -> Any:
        live = manager.jobs.get(job_id)
        if live is not None:
            return json.loads(live.model_dump_json())
        snapshot = store.job_snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="no such job")
        return snapshot

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, after: int = 0) -> Any:
        return {"events": broker.since(job_id, after), "server_time": time.time()}

    @app.get("/api/stream/{job_id}")
    async def stream(job_id: str, request: Request) -> StreamingResponse:
        queue = broker.subscribe(job_id)

        async def pump():
            try:
                for record in broker.since(job_id, 0):
                    yield f"data: {json.dumps(record, ensure_ascii=False)}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        record = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(record, ensure_ascii=False)}\n\n"
                    if record.get("kind") == "done":
                        break
            finally:
                broker.unsubscribe(job_id, queue)

        return StreamingResponse(pump(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id: str) -> Any:
        return {"cancelled": await manager.cancel(job_id)}

    @app.get("/api/config")
    async def get_config() -> Any:
        data = app.state.settings.public_dict()
        data["levels"] = [e.name for e in EscalationLevel]
        data["modes"] = [m.value for m in ResearchMode]
        return data

    @app.post("/api/config")
    async def patch_config(request: Request) -> Any:
        body = await request.json()
        current = app.state.settings.model_dump(mode="json")
        _deep_merge(current, body or {})
        # an empty or masked api_key means "leave the stored one alone"
        for section in ("verifier", "analysis"):
            incoming = (body or {}).get(section, {}) or {}
            key = incoming.get("api_key", "")
            if key in ("", "***configured***"):
                current[section]["api_key"] = app.state.settings.__dict__[section].api_key
        try:
            updated = Settings.model_validate(current)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from None
        save_settings(updated)
        app.state.settings = updated
        manager.settings = updated
        reload_settings()
        return {"ok": True, "config": updated.public_dict()}

    @app.get("/api/doctor")
    async def doctor() -> Any:
        settings = app.state.settings
        checks: dict[str, Any] = {}
        verifier_endpoint = endpoint_for(settings, "verifier")
        analysis_endpoint = endpoint_for(settings, "analysis")
        checks["verifier"] = (
            await LLMClient(verifier_endpoint).health()
            if verifier_endpoint
            else {"ok": False, "state": "unconfigured", "detail": "no base_url/model set"}
        )
        checks["analysis"] = (
            await LLMClient(analysis_endpoint).health()
            if analysis_endpoint
            else {"ok": False, "state": "unconfigured", "detail": "claims fall back to the deterministic extractor"}
        )
        checks["providers"] = {
            name: {
                "enabled": cfg.enabled,
                "label": cfg.label,
                "url": cfg.url,
                "last_status": store.health().get(name, "unknown"),
            }
            for name, cfg in settings.providers.items()
        }
        checks["browser"] = {
            "channel": settings.browser.channel,
            "window_mode": settings.browser.window_mode,
            "reuse_tabs": settings.browser.reuse_tabs,
            "cdp_url": settings.browser.cdp_url,
            "profiles_dir": str((Path(__file__).resolve().parent.parent.parent / "browser" / "profiles")),
            "isolation": (
                "one dedicated OmniBrain window with a tab per provider; your everyday Chrome is never "
                "attached to, read, or copied"
            ),
            "automation_visible": True,
        }
        checks["storage"] = {"db": settings.storage.db_path}
        return checks

    @app.post("/api/providers/{provider}/login")
    async def provider_login(provider: str) -> Any:
        return await manager.open_login_window(provider)

    @app.get("/api/search-test")
    async def search_test(q: str = "Acme") -> Any:
        """Smoke-route for the web-research transport."""
        engine = await manager.engine_get()
        adapter = ProviderCatalog(app.state.settings, engine).adapter("search")
        if adapter is None:
            raise HTTPException(status_code=400, detail="search provider disabled")
        response = await adapter.ask("adhoc", q, 1)
        return response.model_dump(mode="json")

    return app


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


app = _make_app()
