# agents/assistant/ingest/sinks/mem0_sink.py
from __future__ import annotations
from typing import List, Dict, Any
from loguru import logger

from agents.assistant.ingest.sinks.base import BaseSink
from agents.assistant.ingest.models import MemoryRecord
from agents.assistant.ingest.config import settings  # ok to keep even if unused elsewhere

try:
    # Your mem0 package exposes a synchronous Memory class (no api_key kw).
    from mem0 import Memory  # type: ignore
except Exception:
    Memory = None  # type: ignore[assignment]


class Mem0Sink(BaseSink):
    """
    Ingest sink that writes records into Mem0 (local Memory API).

    Notes:
    - Mem0's current API expects `add(messages=..., user_id=..., infer=..., metadata=...)`.
    - There is no `external_id/team_id/visibility` first-class field; we preserve those
      by embedding them in `metadata` so you can filter or audit later.
    """

    def __init__(self) -> None:
        if Memory is None:
            raise RuntimeError("mem0 not installed/importable. Run: pip install mem0ai (your fork/build)")
        # Do NOT pass api_key/base_url here; your Memory() __init__ doesn't accept them.
        self.mem = Memory()

    def upsert(self, records: List[MemoryRecord]) -> None:
        if not records:
            logger.info("mem0 sink: 0 records")
            return

        ok, fail = 0, 0
        for r in records:
            try:
                text = (r.text or "").strip()
                if not text:
                    logger.debug(f"mem0 sink: skip empty text (id={getattr(r, 'id', None)})")
                    continue

                # Merge core metadata and carry auxiliary fields for traceability
                metadata: Dict[str, Any] = dict(r.metadata or {})
                if getattr(r, "team_id", None):
                    metadata["team_id"] = r.team_id
                if getattr(r, "visibility", None):
                    metadata["visibility"] = r.visibility
                metadata["external_id"] = getattr(r, "id", None)

                user_id = r.user_id or "ingest"

                # IMPORTANT: use messages=..., not text=...
                # Turn off LLM-based inference during bulk ingest to avoid slow/expensive calls.
                self.mem.add(
                    messages=text,
                    user_id=user_id,
                    metadata=metadata,
                    infer=False,
                )
                ok += 1
            except Exception as e:
                logger.warning(f"mem0 upsert failed for {getattr(r, 'id', None)}: {e}")
                fail += 1

        logger.info(f"mem0 sink: {ok} ok, {fail} failed")
