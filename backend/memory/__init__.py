"""Long-term memory for OmniBrain: what the user told us about themselves, kept locally and offered to every AI as context.

Memory is **never evidence**. Nothing in this package is read by claim extraction, the source audit, sufficiency,
verification or citation. It only shapes the prompt a provider receives, and only when it is relevant to the question.
"""

from backend.memory.schema import Memory, MemoryType, Source, Status  # noqa: F401
from backend.memory.store import MemoryStore  # noqa: F401
from backend.memory.service import MemoryService  # noqa: F401