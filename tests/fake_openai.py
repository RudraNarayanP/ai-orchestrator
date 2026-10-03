"""A tiny fake OpenAI-compatible server for tests.

Serves ``/v1/chat/completions`` and ``/v1/models`` on 127.0.0.1 so the real
``LLMClient`` / vision client can be exercised over real HTTP without Ollama.
Replies are scripted: a string becomes the assistant message, a callable gets the
request body and returns one, and ``{"status": 500, "body": "..."}`` simulates a
server error. The last scripted reply repeats once the queue is exhausted.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class FakeOpenAI:
    def __init__(self, replies: list[Any] | None = None, models: list[str] | None = None) -> None:
        self.replies: list[Any] = list(replies or [])
        self.models = models if models is not None else ["fake-model"]
        self.requests: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------- lifecycle
    def start(self) -> "FakeOpenAI":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
                return

            def _send(self, status: int, payload: Any) -> None:
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"data": [{"id": m} for m in outer.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                body["_headers"] = {k.lower(): v for k, v in self.headers.items()}
                with outer._lock:
                    outer.requests.append(body)
                    if len(outer.replies) > 1:
                        reply = outer.replies.pop(0)
                    elif outer.replies:
                        reply = outer.replies[0]
                    else:
                        reply = ""
                if callable(reply):
                    reply = reply(body)
                if isinstance(reply, dict) and "status" in reply:
                    self._send(int(reply["status"]), reply.get("body", "error"))
                    return
                if not isinstance(reply, str):
                    reply = json.dumps(reply)
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": reply}}], "usage": {}})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.02), daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    # ---------------------------------------------------------------- helpers
    def script(self, *replies: Any) -> "FakeOpenAI":
        with self._lock:
            self.replies = list(replies)
        return self

    def last_messages(self) -> list[dict[str, Any]]:
        return self.requests[-1]["messages"] if self.requests else []


Reply = Callable[[dict[str, Any]], str]