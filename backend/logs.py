"""File logging: ``engine.log`` used to be an in-memory list that died with the process.

A rotating log at ``data/omnibrain.log`` (``--log-file`` to move it) records what
the app did: job events, browser engine notes, provider failures, tracebacks.

Two rules: it never records credentials (``redact`` runs on every line), and it is
one handler shared by every logger, because two rotating handlers on one file
cannot rotate on Windows.
"""

from __future__ import annotations

import copy
import logging
import logging.handlers
import re
from pathlib import Path
from typing import Any

LOGGER_NAME = "omnibrain"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
MAX_BYTES = 2_000_000
BACKUPS = 3

_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|token|password|secret)(['\"]?\s*[:=]\s*['\"]?)[^\s'\",;&]+"),
    re.compile(r"(?i)([?&](?:key|api_key|token|access_token)=)[^&\s]+"),
]


def redact(text: str) -> str:
    """Strip anything that looks like a credential from a log line."""
    out = text
    out = _SECRET_PATTERNS[0].sub("sk-***", out)
    out = _SECRET_PATTERNS[1].sub("Bearer ***", out)
    out = _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}***", out)
    out = _SECRET_PATTERNS[3].sub(lambda m: f"{m.group(1)}***", out)
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        clean = redact(message)
        if clean != message or record.args:
            record.msg, record.args = clean, ()
        if record.exc_info and record.exc_text is None:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


_HANDLER: logging.handlers.RotatingFileHandler | None = None


def shared_file_handler(path: str | Path | None = None) -> logging.Handler:
    """The one rotating handler (also usable as a dictConfig ``()`` factory)."""
    global _HANDLER
    if _HANDLER is None:
        if path is None:
            raise RuntimeError("setup_logging() has not been called")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(target, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8", delay=True)
        handler.setFormatter(logging.Formatter(FORMAT))
        handler.addFilter(RedactingFilter())
        _HANDLER = handler
    return _HANDLER


def setup_logging(path: str | Path, *, verbose: bool = False) -> Path:
    """Attach the rotating file handler to the ``omnibrain`` logger. Safe to call twice."""
    global _HANDLER
    target = Path(path).expanduser()
    if _HANDLER is not None and Path(_HANDLER.baseFilename) != target.resolve():
        close_logging()
    handler = shared_file_handler(target)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if handler not in logger.handlers:
        logger.addHandler(handler)
    return target


def close_logging() -> None:
    """Detach and close the file handler (tests, and a changed --log-file)."""
    global _HANDLER
    logger = logging.getLogger(LOGGER_NAME)
    if _HANDLER is not None:
        logger.removeHandler(_HANDLER)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).removeHandler(_HANDLER)
        _HANDLER.close()
        _HANDLER = None


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def uvicorn_log_config(path: str | Path, *, verbose: bool = False) -> dict[str, Any]:
    """uvicorn's default config plus the shared file handler on its own loggers.

    uvicorn calls ``dictConfig`` itself and would otherwise drop our handler.
    """
    from uvicorn.config import LOGGING_CONFIG

    shared_file_handler(path)  # make sure the singleton exists before dictConfig asks for it
    config = copy.deepcopy(LOGGING_CONFIG)
    config["disable_existing_loggers"] = False
    config["handlers"]["file"] = {"()": "backend.logs.shared_file_handler"}
    # uvicorn.error propagates to "uvicorn", so giving it the handler too would write every line twice
    for name in ("uvicorn", "uvicorn.access"):
        entry = config["loggers"].setdefault(name, {"handlers": [], "propagate": False})
        entry["handlers"] = [*entry.get("handlers", []), "file"]
        if verbose:
            entry["level"] = "DEBUG"
    return config