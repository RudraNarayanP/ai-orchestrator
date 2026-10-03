"""SQLite persistence: every job stays auditable after it finishes (section 19).

Raw provider text is stored verbatim alongside the claims, evidence, verdicts and
the escalation log, so a wrong answer can be traced back to the exact sentence
and the exact source that produced it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from backend.models import ConversationTurn, Job, JobEvent, ProviderStatus
from backend.research import memory
from backend.settings import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, question TEXT, mode TEXT, status TEXT,
  created_at REAL, updated_at REAL, finished_at REAL,
  level INTEGER, rounds_run INTEGER, max_rounds INTEGER, stop_reason TEXT,
  browser_sessions INTEGER, verifier_calls INTEGER,
  final_json TEXT, analysis_json TEXT, plan_json TEXT, error TEXT,
  answer_text TEXT, confidence TEXT
);
CREATE TABLE IF NOT EXISTS responses (
  id TEXT PRIMARY KEY, job_id TEXT, provider TEXT, round INTEGER, role TEXT,
  status TEXT, prompt TEXT, answer_text TEXT, raw_text TEXT, citations_json TEXT,
  started_at REAL, finished_at REAL, duration_s REAL, ui_url TEXT,
  web_research TEXT, failure_json TEXT, pages_json TEXT, error TEXT, fingerprint TEXT,
  escalation_reason TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS responses_job ON responses(job_id, round);
CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY, job_id TEXT, round INTEGER, claim TEXT, kind TEXT, topic TEXT,
  providers_json TEXT, web_json TEXT, supporting_json TEXT, contradicting_json TEXT,
  status TEXT, confidence TEXT, rationale TEXT
);
CREATE INDEX IF NOT EXISTS claims_job ON claims(job_id);
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, job_id TEXT, round INTEGER, claim_id TEXT, url TEXT, title TEXT,
  domain TEXT, snippet TEXT, published TEXT, tier TEXT, polarity TEXT,
  check_status TEXT, check_notes TEXT, origin TEXT, verbatim_excerpt TEXT, retrieved_at REAL
);
CREATE INDEX IF NOT EXISTS evidence_job ON evidence(job_id, claim_id);
CREATE TABLE IF NOT EXISTS disagreements (
  id TEXT PRIMARY KEY, job_id TEXT, round INTEGER, topic TEXT, description TEXT,
  positions_json TEXT, severity TEXT, claim_ids_json TEXT, resolution TEXT
);
CREATE TABLE IF NOT EXISTS reports (
  job_id TEXT, round INTEGER, verdicts_json TEXT, report_json TEXT, raw_output TEXT
);
CREATE INDEX IF NOT EXISTS reports_job ON reports(job_id, round);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts REAL, kind TEXT,
  message TEXT, provider TEXT, round INTEGER, payload_json TEXT
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, id);
CREATE TABLE IF NOT EXISTS provider_health (
  provider TEXT PRIMARY KEY, status TEXT, detail TEXT, updated_at REAL
);
"""


class Store:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = Path(settings.storage.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for databases created by an older build."""
        have = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, ddl in (("conversation_id", "TEXT"), ("confirmed_json", "TEXT")):
            if column not in have:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")
        self._conn.execute("CREATE INDEX IF NOT EXISTS jobs_conversation ON jobs(conversation_id, created_at)")

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------- jobs

    def save_job(self, job: Job) -> None:
        final = job.final.model_dump(mode="json") if job.final else None
        self._conn.execute(
            """INSERT INTO jobs (id, question, mode, status, created_at, updated_at, finished_at, level,
                                 rounds_run, max_rounds, stop_reason, browser_sessions, verifier_calls,
                                 final_json, analysis_json, plan_json, error, answer_text, confidence,
                                 conversation_id, confirmed_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at,
                 finished_at=excluded.finished_at, level=excluded.level, rounds_run=excluded.rounds_run,
                 stop_reason=excluded.stop_reason, browser_sessions=excluded.browser_sessions,
                 verifier_calls=excluded.verifier_calls, final_json=excluded.final_json,
                 analysis_json=excluded.analysis_json, plan_json=excluded.plan_json,
                 error=excluded.error, answer_text=excluded.answer_text, confidence=excluded.confidence,
                  conversation_id=excluded.conversation_id, confirmed_json=excluded.confirmed_json""",
            (
                job.id,
                job.question,
                job.mode.value,
                job.status.value,
                job.created_at,
                job.updated_at,
                job.finished_at,
                int(job.level.value),
                job.rounds_run,
                job.max_rounds,
                job.stop_reason,
                job.browser_sessions_used,
                job.verifier_calls,
                json.dumps(final, ensure_ascii=False) if final else None,
                json.dumps(job.analysis.model_dump(mode="json"), ensure_ascii=False) if job.analysis else None,
                json.dumps(job.plan.model_dump(mode="json"), ensure_ascii=False) if job.plan else None,
                job.error,
                (job.final.answer if job.final else None),
                (job.final.confidence.value if job.final else None),
                job.conversation_id,
                json.dumps(turn.confirmed_claims, ensure_ascii=False) if (turn := memory.turn_from_job(job)) else None,
            ),
        )
        self._conn.commit()
        self._save_artifacts(job)

    def _save_artifacts(self, job: Job) -> None:
        for response in job.responses:
            self._conn.execute(
                """INSERT INTO responses (id, job_id, provider, round, role, status, prompt, answer_text, raw_text,
                       citations_json, started_at, finished_at, duration_s, ui_url, web_research, failure_json,
                       pages_json, error, fingerprint, escalation_reason, detail)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, answer_text=excluded.answer_text,
                     raw_text=excluded.raw_text, citations_json=excluded.citations_json,
                     finished_at=excluded.finished_at, duration_s=excluded.duration_s, error=excluded.error""",
                (
                    response.id, job.id, response.provider, response.round, response.role, response.status.value,
                    response.prompt, response.answer_text,
                    response.raw_text if self.settings.storage.keep_raw_responses else "",
                    json.dumps([c.model_dump(mode="json") for c in response.citations], ensure_ascii=False),
                    response.started_at, response.finished_at, response.duration_s, response.ui_url,
                    response.web_research_status.value,
                    json.dumps(response.failure_signals, ensure_ascii=False),
                    json.dumps(response.pages_visited, ensure_ascii=False),
                    response.error, response.fingerprint, response.escalation_reason, response.detail,
                ),
            )
        for claim in job.claims:
            self._conn.execute(
                """INSERT INTO claims (id, job_id, round, claim, kind, topic, providers_json, web_json,
                       supporting_json, contradicting_json, status, confidence, rationale)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, confidence=excluded.confidence,
                     rationale=excluded.rationale, providers_json=excluded.providers_json""",
                (
                    claim.id, job.id, claim.round, claim.claim, claim.kind, claim.topic,
                    json.dumps(claim.provider_sources, ensure_ascii=False),
                    json.dumps(claim.web_sources, ensure_ascii=False),
                    json.dumps(claim.supporting, ensure_ascii=False),
                    json.dumps(claim.contradicting, ensure_ascii=False),
                    claim.status.value,
                    claim.confidence.value if claim.confidence else None,
                    claim.rationale,
                ),
            )
        for ev in job.evidence:
            self._conn.execute(
                """INSERT INTO evidence (id, job_id, round, claim_id, url, title, domain, snippet, published,
                       tier, polarity, check_status, check_notes, origin, verbatim_excerpt, retrieved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET check_status=excluded.check_status, polarity=excluded.polarity,
                     check_notes=excluded.check_notes, verbatim_excerpt=excluded.verbatim_excerpt""",
                (
                    ev.id, job.id, ev.round, ev.claim_id, ev.url, ev.title, ev.domain, ev.snippet, ev.published,
                    ev.tier.value, ev.polarity, ev.check_status.value, ev.check_notes, ev.origin,
                    ev.verbatim_excerpt, ev.retrieved_at,
                ),
            )
        for dis in job.disagreements:
            self._conn.execute(
                """INSERT OR REPLACE INTO disagreements
                   (id, job_id, round, topic, description, positions_json, severity, claim_ids_json, resolution)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    dis.id, job.id, dis.round, dis.topic, dis.description,
                    json.dumps(dis.positions, ensure_ascii=False), dis.severity,
                    json.dumps(dis.claim_ids, ensure_ascii=False), dis.resolution,
                ),
            )
        # reports have no natural key; rewrite them so repeated saves don't duplicate rows
        self._conn.execute("DELETE FROM reports WHERE job_id=?", (job.id,))
        for report in job.reports:
            self._conn.execute(
                "INSERT INTO reports (job_id, round, verdicts_json, report_json, raw_output) VALUES (?,?,?,?,?)",
                (
                    job.id,
                    report.round,
                    json.dumps([v.model_dump(mode="json") for v in report.verdicts], ensure_ascii=False),
                    json.dumps(report.model_dump(mode="json"), ensure_ascii=False),
                    report.raw_output[:8000],
                ),
            )
        self._conn.commit()

    def record_health(self, provider: str, status: str, detail: str = "") -> None:
        self._conn.execute(
            "INSERT INTO provider_health (provider, status, detail, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(provider) DO UPDATE SET status=excluded.status, detail=excluded.detail, updated_at=excluded.updated_at",
            (provider, status, detail, time.time()),
        )
        self._conn.commit()

    def health(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT provider, status FROM provider_health").fetchall()
        return {row["provider"]: row["status"] for row in rows}

    def add_event(self, job_id: str, event: JobEvent) -> None:
        self._conn.execute(
            "INSERT INTO events (job_id, ts, kind, message, provider, round, payload_json) VALUES (?,?,?,?,?,?,?)",
            (job_id, event.ts, event.kind, event.message, event.provider, event.round, json.dumps(event.payload, ensure_ascii=False)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ reads

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, question, mode, status, created_at, finished_at, level, rounds_run, "
            "browser_sessions, verifier_calls, answer_text, confidence, stop_reason, conversation_id "
            "FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def conversation_turns(self, conversation_id: str, *, exclude_job_id: str = "", limit: int = 3) -> list[ConversationTurn]:
        """The most recent finished turns of a thread, oldest first."""
        if not conversation_id:
            return []
        rows = self._conn.execute(
            "SELECT id, question, answer_text, confidence, confirmed_json, finished_at, updated_at FROM jobs "
            "WHERE conversation_id=? AND id<>? AND final_json IS NOT NULL ORDER BY created_at DESC LIMIT ?",
            (conversation_id, exclude_job_id, limit),
        ).fetchall()
        turns = [
            ConversationTurn(
                job_id=row["id"],
                question=row["question"] or "",
                answer=row["answer_text"] or "",
                confirmed_claims=json.loads(row["confirmed_json"] or "[]"),
                confidence=row["confidence"] or "",
                at=row["finished_at"] or row["updated_at"] or 0.0,
            )
            for row in rows
        ]
        return list(reversed(turns))

    def events_since(self, job_id: str, after: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, ts, kind, message, provider, round, payload_json FROM events "
            "WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
            (job_id, after, limit),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            out.append(item)
        return out

    def job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            return None
        out = dict(job)
        for key in ("final_json", "analysis_json", "plan_json"):
            out[key.replace("_json", "")] = json.loads(out.pop(key) or "null")
        out["responses"] = [
            _json_fields(dict(row), ("citations_json", "failure_json", "pages_json"))
            for row in self._conn.execute("SELECT * FROM responses WHERE job_id=? ORDER BY round, started_at", (job_id,)).fetchall()
        ]
        out["claims"] = [
            _json_fields(dict(row), ("providers_json", "web_json", "supporting_json", "contradicting_json"))
            for row in self._conn.execute("SELECT * FROM claims WHERE job_id=? ORDER BY round", (job_id,)).fetchall()
        ]
        out["evidence"] = [dict(row) for row in self._conn.execute("SELECT * FROM evidence WHERE job_id=? ORDER BY round, retrieved_at", (job_id,)).fetchall()]
        out["disagreements"] = [
            _json_fields(dict(row), ("positions_json", "claim_ids_json"))
            for row in self._conn.execute("SELECT * FROM disagreements WHERE job_id=?", (job_id,)).fetchall()
        ]
        out["reports"] = [
            _json_fields(dict(row), ("verdicts_json", "report_json"))
            for row in self._conn.execute("SELECT * FROM reports WHERE job_id=? ORDER BY round", (job_id,)).fetchall()
        ]
        return out


def _json_fields(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        if key in row:
            row[key.replace("_json", "")] = json.loads(row.pop(key) or "[]")
    return row
