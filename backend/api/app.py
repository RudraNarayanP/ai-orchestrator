"""HTTP API: job control, live event stream, configuration, diagnostics.

The frontend is a chat box. Everything that makes OmniBrain more than a chat box
-- the escalation log, per-provider health, raw captures, the evidence ledger --
is reachable but tucked behind it, because the answer is what the user asked for.

API keys never cross this boundary: /api/config returns them masked.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from backend.browser.engine import BrowserEngine
from backend.cancel import CancelToken
from backend.export import export_filename, job_markdown
from backend.logs import get_logger
from backend.models import (
    EscalationLevel,
    Job,
    JobEvent,
    JobStatus,
    ResearchMode,
)
from backend.orchestrator.politeness import PolitenessGate
from backend.orchestrator.runner import ResearchRunner
from backend.providers.registry import ProviderCatalog, endpoint_for
from backend.settings import Settings, load_settings, reload_settings, save_settings
from backend.storage.db import Store
from backend.verification.llm import Endpoint, LLMClient

log = get_logger("jobs")
FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"
CONVERSATION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_QUESTION_CHARS = 4000


async def _json_object(request: Request) -> dict[str, Any]:
    """A request body that must be a JSON object; anything else is the caller's mistake, not a 500."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="body must be valid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


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
        self.cancels: dict[str, CancelToken] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.Semaphore = asyncio.Semaphore(1)
        self.health: dict[str, str] = {}
        self.politeness = PolitenessGate(settings.research)  # shared by every job, so a backoff outlives one question

    def memory_service(self) -> Any:
        """One local memory store per server (None when switched off in settings or if it cannot open)."""
        if not self.settings.memory.enabled:
            return None
        if getattr(self, "_memory", None) is None:
            try:
                from backend.memory import MemoryService, MemoryStore
                from backend.memory.embed import get_embedder

                self._memory = MemoryService(MemoryStore(self.settings.memory.path, get_embedder(self.settings.memory.embedder, self.settings.memory.model)))
            except Exception as exc:  # noqa: BLE001
                log.warning("memory unavailable: %s", exc)
                self._memory = None
        return self._memory

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
            log.info("[%s] %s%s: %s", job_id, f"{provider} " if provider else "", kind, message[:600])
            self.broker.publish(job_id, event)
            try:
                self.store.add_event(job_id, event)
            except Exception:  # noqa: BLE001
                pass

        return emit

    async def start(
        self,
        question: str,
        mode: str | None,
        max_rounds: int | None,
        conversation_id: str | None = None,
        project: str | None = None,
    ) -> Job:
        question = (question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        try:
            resolved_mode = ResearchMode((mode or self.settings.research.mode).upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}") from None
        cid = (conversation_id or "").strip()
        if cid and not CONVERSATION_ID_RE.fullmatch(cid):
            raise HTTPException(status_code=400, detail="conversation_id must be 1-64 letters, digits, '_' or '-'")
        job = Job(
            question=question,
            mode=resolved_mode,
            max_rounds=int(max_rounds or self.settings.research.max_rounds),
            conversation_id=cid or f"conv_{uuid.uuid4().hex[:12]}",
            project=(project or "").strip()[:64] or None,
        )
        # earlier finished turns of this thread; the runner decides whether this question leans on them
        job.history = self.store.conversation_turns(job.conversation_id, exclude_job_id=job.id, limit=3)
        self.jobs[job.id] = job
        self.cancels[job.id] = CancelToken()
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
                    cancel=self.cancels.setdefault(job.id, CancelToken()),
                    politeness=self.politeness,
                    memory=self.memory_service(),
                )
                await emit("status", f"starting with {len(adapters)} enabled providers")
                await runner.run(job)
                for provider, status in runner.health.items():
                    self.store.record_health(provider, status)
                self.health = self.store.health()
            except asyncio.CancelledError:
                log.info("[%s] cancelled", job.id)
                job.status = JobStatus.CANCELLED
                job.stop_reason = "cancelled"
                await emit("status", "cancelled")
            except Exception as exc:  # noqa: BLE001
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                log.exception("[%s] job failed", job.id)
                await emit("error", job.error)
            finally:
                job.updated_at = time.time()
                job.rounds_run = max([r.number for r in job.rounds], default=job.rounds_run)
                self.store.save_job(job)
                await emit("done", job.status.value, payload={"final": bool(job.final)})

    async def cancel(self, job_id: str) -> bool:
        task = self.tasks.get(job_id)
        if task and not task.done():
            # flag first: adapters and the runner stop at their next checkpoint and the
            # adapter closes its tab; task.cancel() then interrupts whatever is awaiting
            token = self.cancels.get(job_id)
            if token is not None:
                token.cancel()
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
        body = await _json_object(request)
        question = body.get("question", "")
        if not isinstance(question, str):
            raise HTTPException(status_code=400, detail="question must be a string")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(status_code=400, detail=f"question is too long (max {MAX_QUESTION_CHARS} characters)")
        max_rounds = body.get("max_rounds")
        if max_rounds is not None:
            if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or not 1 <= max_rounds <= 8:
                raise HTTPException(status_code=400, detail="max_rounds must be an integer from 1 to 8")
        mode = body.get("mode")
        if mode is not None and not isinstance(mode, str):
            raise HTTPException(status_code=400, detail="mode must be a string")
        conversation_id = body.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise HTTPException(status_code=400, detail="conversation_id must be a string")
        project = body.get("project")
        if project is not None and not isinstance(project, str):
            raise HTTPException(status_code=400, detail="project must be a string")
        job = await manager.start(question, mode, max_rounds, conversation_id, project)
        return JSONResponse({"job_id": job.id, "status": job.status.value, "conversation_id": job.conversation_id})

    # ------------------------------------------------------------------ memory (local only; see backend/memory)
    def _mem() -> Any:
        svc = manager.memory_service()
        if svc is None:
            raise HTTPException(status_code=503, detail="memory is switched off in settings (memory.enabled)")
        return svc

    @app.get("/api/memory")
    async def memory_list(status: str | None = None, type: str | None = None, source: str | None = None, project: str | None = None,
                          q: str | None = None, limit: int = 100, offset: int = 0) -> Any:
        svc = _mem()
        items = svc.store.list(status=status, memory_type=type, source=source, project=project, q=q, limit=max(1, min(limit, 500)), offset=max(0, offset))
        return {"items": [m.public() for m in items], "stats": svc.store.stats()}

    @app.get("/api/memory/stats")
    async def memory_stats() -> Any:
        return _mem().store.stats()

    @app.get("/api/memory/export")
    async def memory_export() -> Any:
        return JSONResponse(_mem().store.export(), headers={"Content-Disposition": "attachment; filename=omnibrain-memory.json"})

    @app.post("/api/memory/search")
    async def memory_search(request: Request) -> Any:
        """What would be offered for this question, and WHY each memory was picked."""
        body = await _json_object(request)
        q = body.get("query")
        if not isinstance(q, str) or not q.strip():
            raise HTTPException(status_code=400, detail="query must be a non-empty string")
        project = body.get("project") if isinstance(body.get("project"), str) else None
        block, res = _mem().context_for(q, project=project)
        return {"kind": res.kind, "budget": res.budget, "candidates": res.candidates, "reason": res.reason, "elapsed_ms": round(res.elapsed_ms, 2),
                "context": block, "hits": [h.public() for h in res.hits]}

    @app.get("/api/memory/settings")
    async def memory_get_settings() -> Any:
        s = _mem().store
        return {"inject": s.inject_enabled, "capture": s.capture_enabled, "embedder": s.embedder.name}

    @app.post("/api/memory/settings")
    async def memory_set_settings(request: Request) -> Any:
        body = await _json_object(request)
        s = _mem().store
        for key in ("inject", "capture"):
            if key in body:
                if not isinstance(body[key], bool):
                    raise HTTPException(status_code=400, detail=f"{key} must be true or false")
                s.set_setting(key, "1" if body[key] else "0")
        return {"inject": s.inject_enabled, "capture": s.capture_enabled, "embedder": s.embedder.name}

    @app.post("/api/memory")
    async def memory_add(request: Request) -> Any:
        body = await _json_object(request)
        content = body.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 1000:
            raise HTTPException(status_code=400, detail="content must be a non-empty string (max 1000 characters)")
        mtype = body.get("memory_type", "fact")
        project = body.get("project") if isinstance(body.get("project"), str) and body.get("project") else None
        try:
            res = _mem().remember(content, memory_type=mtype, project=project)
        except ValueError:
            raise HTTPException(status_code=400, detail="unknown memory_type") from None
        return {"action": res.action, "memory": res.memory.public() if res.memory else None, "reason": res.reason}

    @app.patch("/api/memory/{memory_id}")
    async def memory_edit(memory_id: str, request: Request) -> Any:
        body = await _json_object(request)
        fields = {k: body[k] for k in ("content", "importance", "confidence", "status", "project", "goal_active") if k in body}
        try:
            m = _mem().store.update(memory_id, **fields)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid value") from None
        if m is None:
            raise HTTPException(status_code=404, detail="no such memory")
        return m.public()

    @app.delete("/api/memory/{memory_id}")
    async def memory_delete(memory_id: str) -> Any:
        if not _mem().store.delete(memory_id):
            raise HTTPException(status_code=404, detail="no such memory")
        return {"deleted": memory_id}

    @app.post("/api/memory/consolidate")
    async def memory_consolidate() -> Any:
        return {"merged": _mem().store.consolidate()}

    @app.post("/api/memory/forget-all")
    async def memory_forget_all(request: Request) -> Any:
        body = await _json_object(request)
        if body.get("confirm") is not True:
            raise HTTPException(status_code=400, detail="send {\"confirm\": true} to delete every memory")
        return {"deleted": _mem().store.delete_all()}

    @app.get("/api/memory/{memory_id}")
    async def memory_get(memory_id: str) -> Any:
        s = _mem().store
        m = s.get(memory_id)
        if m is None:
            raise HTTPException(status_code=404, detail="no such memory")
        return {**m.public(), "events": s.events(memory_id)}

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> Any:
        """The finished turns of a thread, oldest first, so the UI can redraw it after a reload."""
        if not CONVERSATION_ID_RE.fullmatch(conversation_id):
            raise HTTPException(status_code=400, detail="bad conversation_id")
        turns = store.conversation_turns(conversation_id, limit=50)
        return {"conversation_id": conversation_id, "turns": [t.model_dump(mode="json") for t in turns]}

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50) -> Any:
        """Recent jobs, newest first (the History view). `limit` is clamped to 1..200."""
        return store.list_jobs(limit=max(1, min(int(limit), 200)))

    @app.get("/api/jobs/{job_id}/export")
    async def export_job(job_id: str, format: str = "md") -> Response:
        """A finished job as a Markdown record (default) or its full JSON snapshot, as a download."""
        fmt = format.lower()
        if fmt not in {"md", "json"}:
            raise HTTPException(status_code=400, detail="format must be md or json")
        snapshot = store.job_snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="no such job")
        if not snapshot.get("final") and job_id in manager.jobs and manager.jobs[job_id].final is None:
            raise HTTPException(status_code=409, detail="the job has not finished yet")
        body = job_markdown(snapshot) if fmt == "md" else json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
        media = "text/markdown; charset=utf-8" if fmt == "md" else "application/json"
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{export_filename(job_id, fmt)}"'})

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
                replayed_done = False
                for record in broker.since(job_id, 0):
                    yield f"data: {json.dumps(record, ensure_ascii=False)}\n\n"
                    replayed_done = replayed_done or record.get("kind") == "done"
                if replayed_done:
                    return  # a finished job has nothing more to say; don't hold the connection open
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
        body = await _json_object(request)
        current = app.state.settings.model_dump(mode="json")
        _deep_merge(current, body or {})
        # an empty or masked api_key means "leave the stored one alone"
        for section in ("verifier", "analysis", "vision"):
            incoming = (body or {}).get(section, {}) or {}
            if not isinstance(incoming, dict):
                raise HTTPException(status_code=400, detail=f"{section} must be an object")
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
        vision = settings.vision
        checks["vision"] = (
            {"ok": False, "state": "disabled", "detail": "vision fallback is off (vision.provider = disabled)"}
            if vision.provider == "disabled"
            else {"ok": bool(vision.model), "state": "configured" if vision.model else "unconfigured", "detail": f"{vision.provider} / {vision.model or 'no model set'}"}
        )
        if vision.provider == "openrouter" and vision.model:
            # A hosted endpoint can be checked for real: reachable, key accepted, exact model id listed.
            checks["vision"] = await LLMClient(Endpoint.from_config(vision)).health()
        checks["providers"] = {
            name: {
                "enabled": cfg.enabled,
                "label": cfg.label,
                "url": cfg.url,
                "requires_login": cfg.requires_login,
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
