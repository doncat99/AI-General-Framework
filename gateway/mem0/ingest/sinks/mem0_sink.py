# gateway/mem0/sinks/mem0_sink.py
from __future__ import annotations

from typing import List, Dict, Any
from loguru import logger

from .base import BaseSink
from ..models import MemoryRecord
from ...memory import NeuraMemory


class Mem0Sink(BaseSink):
    """
    Sink that writes MemoryRecord items into Mem0 (AsyncMemory).
    """

    def __init__(self, memory: NeuraMemory) -> None:
        self.memory = memory

    async def upsert(self, records: List[MemoryRecord]) -> int:
        if not records:
            logger.info("mem0 sink: 0 records")
            return 0

        ok, fail = 0, 0
        for r in records:
            try:
                text = (r.text or "").strip()
                if not text:
                    continue

                metadata: Dict[str, Any] = dict(r.metadata or {})
                metadata["team_id"] = r.team_id
                metadata["visibility"] = r.visibility
                metadata["external_id"] = r.id

                await self.memory.add(
                    messages=text,
                    user_id=r.user_id or "ingest",
                    agent_id="mem0_sink",
                    metadata=metadata,
                )
                ok += 1
            except Exception as e:
                logger.warning(f"mem0 upsert failed for {r.id}: {e}")
                fail += 1

        logger.info(f"mem0 sink: {ok} ok, {fail} failed")
        return ok
