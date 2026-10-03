"""The memory record and its enums."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    PREFERENCE = "preference"  # explicit likes/dislikes/style rules ("keep answers short")
    FACT = "fact"  # persistent facts about the user ("lives in Kyiv")
    GOAL = "goal"  # something the user is trying to do; active or not
    PROJECT = "project"  # knowledge scoped to a named project (namespace)
    EPISODIC = "episodic"  # something that happened, with a time
    CONVERSATION = "conversation"  # short-lived thread context; kept apart from long-term memory
    INTERPRETATION = "interpretation"  # the user's feeling/belief/reading of motives: their VIEW, never an objective fact


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"  # replaced by a correction; kept as history, never injected
    ARCHIVED = "ARCHIVED"  # no longer relevant (finished goal, merged duplicate)
    DELETED = "DELETED"  # tombstone state in the API; the row itself is physically removed


class Source(str, Enum):
    USER_EXPLICIT = "user_explicit"  # the user said it / asked to remember it / edited it
    MODEL_INFERRED = "model_inferred"  # we inferred it from how they talk; never overrides explicit


def new_memory_id() -> str:
    return "mem_" + uuid.uuid4().hex[:16]


class Memory(BaseModel):
    memory_id: str = Field(default_factory=new_memory_id)
    content: str
    memory_type: MemoryType = MemoryType.FACT
    scope: str = "global"  # global | project | conversation
    project: str | None = None
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    source_conversation: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    last_accessed_at: float | None = None
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 0.8
    status: Status = Status.ACTIVE
    supersedes: str | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Source = Source.USER_EXPLICIT
    slot: str | None = None  # what this memory is *about* ("residence", "pref:answer_length"); one ACTIVE per slot
    valid_from: float | None = None
    valid_to: float | None = None
    goal_active: bool | None = None
    sensitivity: str = "normal"  # normal | sensitive (family, money, health, relationships, legal, identity)
    embedding: bytes | None = Field(default=None, exclude=True, repr=False)

    def public(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", exclude={"embedding"})
        return d