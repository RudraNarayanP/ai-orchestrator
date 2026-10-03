"""Unlimited OmniBrain thread.

The OmniBrain thread owns the canonical conversation. Provider chats are replaceable context windows underneath it:
the user sees one thread, made of ordered *segments*, each mapped to one provider conversation. When a segment nears
its provider's limit (or the user switches provider) a new chat opens in the same tab slot and receives a structured
continuation packet. Raw messages are never deleted; summaries are an index over them, not a replacement.

Thread facts are conversation context, NOT evidence: nothing here is read by claim extraction, source audit,
sufficiency or verification (tests/test_thread_isolation.py).
"""